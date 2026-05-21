"""Route natural-language dataset refinement requests to one or many table edits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.llm import generate_json
from app.schema_ddl import ParsedSchema
from app.synthetic.generate import refine_table
from app.synthetic.validate import validate_tables_data
from app.tracing import LangfuseTraceContext

_MISSING_ASSISTANT_PLACEHOLDER = "(no reply)"


@dataclass(frozen=True)
class SyntheticRefinementPlan:
    assistant_reply: str
    apply_refinements: bool
    """Table stems passed to refine_table (already normalized to the schema)."""
    target_tables: tuple[str, ...]
    refinement_instruction: str


def schema_column_ref(schema: ParsedSchema) -> str:
    lines: list[str] = []
    for t in schema.ordered_table_names():
        cols = schema.tables[t].columns
        lines.append("- " + t + ": " + ", ".join(c.name for c in cols))
    return "\n".join(lines)


def _format_recent_chat(messages: Sequence[dict[str, str]], *, max_messages: int = 14) -> str:
    trimmed = list(messages[-max_messages:])
    chunks: list[str] = []
    for m in trimmed:
        role = str(m.get("role", "") or "").strip()
        body = str(m.get("content", "") or "").strip()
        if not body:
            continue
        chunks.append(f"{role.upper()}: {body}")
    return "\n".join(chunks) if chunks else "(no prior messages)"


def build_refinement_router_prompt(
    schema: ParsedSchema,
    *,
    ddl: str,
    original_instructions: str,
    conversation_transcript: str,
) -> str:
    ddl_blob = ddl.strip()
    if len(ddl_blob) > 9000:
        ddl_blob = ddl_blob[:8970] + "\n…[DDL truncated]"
    summary = schema.summary_for_prompt()
    colnames = schema_column_ref(schema)
    tbl_list = ", ".join(schema.ordered_table_names())

    return (
        "You coordinate edits to synthetic relational CSV data previously generated from DDL.\n"
        "\n"
        "The editor LLM receives refinement_instruction verbatim (typing + FK snapshots).\n"
        "\n"
        "Return ONLY JSON with keys:\n"
        '- "assistant_reply" (markdown-friendly string)\n'
        '- "apply_refinements" (boolean)\n'
        '- "target_tables": ["*"] for all stems, otherwise exact DDL table names\n'
        '- "refinement_instruction": concrete edit task for each refine pass\n'
        "\n"
        "Decision rules:\n"
        "- Ambiguous: apply_refinements false; clarify in assistant_reply.\n"
        '- Use "*" in target_tables only for edits that genuinely span every table.\n'
        "- Otherwise pick the smallest stem set.\n"
        "- Mention FK / NOT NULL expectations inside refinement_instruction when needed.\n"
        "\n"
        "Original generation instructions:\n"
        f"{original_instructions.strip() or '(none)'}\n"
        "\n"
        "Schema summary:\n"
        f"{summary}\n"
        "\n"
        "Column names per table:\n"
        f"{colnames}\n"
        "\n"
        f"Tables (ordered): {tbl_list}\n"
        "\n"
        "Full DDL:\n"
        f"{ddl_blob}\n"
        "\n"
        "Conversation:\n"
        f"{conversation_transcript}\n"
    )


def expand_target_tables(requested: Sequence[str], schema: ParsedSchema) -> list[str]:
    raw = [str(x).strip() for x in requested if str(x).strip()]
    if not raw:
        return []
    ordered = schema.ordered_table_names()
    if "*" in raw:
        return list(ordered)
    out: list[str] = []
    for t in raw:
        if t in schema.tables and t not in out:
            out.append(t)
    return out


def parse_router_payload(raw: Any, schema: ParsedSchema) -> SyntheticRefinementPlan:
    """Turn model JSON into a validated plan."""
    if not isinstance(raw, dict):
        raise ValueError("Router model returned non-object JSON")
    raw_reply = str(raw.get("assistant_reply", "") or "").strip()
    missing_reply = not raw_reply
    assistant = raw_reply if raw_reply else _MISSING_ASSISTANT_PLACEHOLDER
    apply_flag = raw.get("apply_refinements", False) is True
    targets_raw = raw.get("target_tables", [])
    if not isinstance(targets_raw, list):
        targets_raw = []
    instruction = str(raw.get("refinement_instruction", "") or "").strip()

    expanded = expand_target_tables([str(x) for x in targets_raw], schema)
    targets_tuple = tuple(expanded)

    if apply_flag and not instruction:
        clar = (
            "Please spell out **what** should change (columns, fractions, substitutions) "
            "and **where**."
        )
        merged = clar if missing_reply else f"{assistant}\n\n{clar}"
        return SyntheticRefinementPlan(
            assistant_reply=merged,
            apply_refinements=False,
            target_tables=(),
            refinement_instruction="",
        )

    if apply_flag and not targets_tuple:
        suffix = (
            "**Note:** I could not resolve which DDL table(s) to edit — name the table(s) from "
            "your schema or say “all tables”, then describe the edit."
        )
        merged = suffix if missing_reply else f"{assistant}\n\n{suffix}"
        return SyntheticRefinementPlan(
            assistant_reply=merged,
            apply_refinements=False,
            target_tables=(),
            refinement_instruction="",
        )

    return SyntheticRefinementPlan(
        assistant_reply=assistant,
        apply_refinements=apply_flag,
        target_tables=targets_tuple,
        refinement_instruction=instruction,
    )


def plan_dataset_refinement_turn(
    settings: Settings,
    *,
    schema: ParsedSchema,
    ddl: str,
    original_instructions: str,
    chat_messages: list[dict[str, str]],
    temperature: float = 0.2,
    trace_context: LangfuseTraceContext | None = None,
) -> SyntheticRefinementPlan:
    transcript = _format_recent_chat(chat_messages)
    prompt = build_refinement_router_prompt(
        schema,
        ddl=ddl,
        original_instructions=original_instructions,
        conversation_transcript=transcript,
    )
    raw = generate_json(
        settings,
        prompt,
        temperature=temperature,
        trace_context=trace_context,
    )
    return parse_router_payload(raw, schema)


def apply_refinement_plan(
    settings: Settings,
    *,
    schema: ParsedSchema,
    tables: dict[str, list[dict[str, Any]]],
    plan: SyntheticRefinementPlan,
    refinement_temperature: float,
    trace_context: LangfuseTraceContext | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Sequentially refine tables so parent rows/FK snapshots stay current."""
    if (
        not plan.apply_refinements
        or not plan.target_tables
        or not plan.refinement_instruction.strip()
    ):
        return tables, validate_tables_data(tables, schema)

    updated = {k: [dict(r) for r in v] for k, v in tables.items()}
    for tname in plan.target_tables:
        if tname not in schema.tables:
            continue
        updated[tname] = refine_table(
            settings,
            schema=schema,
            table_name=tname,
            all_tables=updated,
            user_feedback=plan.refinement_instruction,
            temperature=refinement_temperature,
            trace_context=trace_context,
        )

    errors = validate_tables_data(updated, schema)
    return updated, errors
