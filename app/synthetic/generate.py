"""LLM prompts and orchestration for Phase 1 synthetic data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.llm import generate_json
from app.schema_ddl import ParsedSchema, parse_ddl
from app.synthetic.storage import DEFAULT_DATA_ROOT, save_dataset, zip_dataset
from app.synthetic.validate import validate_tables_data
from app.tracing import LangfuseTraceContext


def build_full_generation_prompt(
    schema: ParsedSchema,
    ddl: str,
    instructions: str,
    rows_per_table: int,
) -> str:
    summary = schema.summary_for_prompt()
    order = ", ".join(schema.ordered_table_names())
    return f"""You generate realistic relational synthetic data as JSON only.

OUTPUT SHAPE (strict):
{{
  "tables": {{
    "<TableName>": [
      {{ "<column>": <value>, ... }},
      ...
    ],
    ...
  }}
}}

Rules:
- Include EVERY table listed in the schema summary below, \
exactly matching table names (case-sensitive).
- Produce exactly {rows_per_table} rows per table (or fewer only if logically impossible—then add \
a "note" string field at top level explaining why).
- Respect PRIMARY KEY (unique), NOT NULL, UNIQUE columns, ENUM literals, CHECK constraints, and \
FOREIGN KEYS: every FK value MUST exist as referenced PK in the parent table's rows you generate.
- Use plausible realistic values; match date formats as ISO strings (YYYY-MM-DD), datetimes \
as ISO 8601 strings when needed.
- INTEGER / INT primary keys: integers starting at 1 per table unless seed data requires \
otherwise; DECIMAL as JSON numbers; BOOLEAN as true/false; TEXT as strings.
- ENUM columns: only allowed enum literal strings from the DDL.

Suggested generation / validation order for referential integrity: {order}

Schema summary:
{summary}

Full DDL:
{ddl}

User instructions (style, domain, constraints):
{instructions}
"""


def build_table_refinement_prompt(
    schema: ParsedSchema,
    table_name: str,
    current_rows: list[dict[str, Any]],
    all_tables: dict[str, list[dict[str, Any]]],
    user_feedback: str,
) -> str:
    tdef = schema.tables[table_name]
    pk_context: dict[str, list[Any]] = {}
    for fk in tdef.foreign_keys:
        parent = fk.ref_table
        if parent in all_tables:
            pk_context[parent] = [
                r.get(fk.ref_column)
                for r in all_tables[parent]
                if fk.ref_column in r and r.get(fk.ref_column) is not None
            ]
    return f"""You adjust rows for ONE table in a synthetic dataset.

Return ONLY a JSON array of row objects for table {table_name!r} (no wrapper object).

Schema columns: {", ".join(c.name + ":" + c.sql_type for c in tdef.columns)}
Foreign keys must reference valid parent keys. Allowed parent keys snapshot: {pk_context}

Current rows (JSON):
{current_rows}

User request:
{user_feedback}
"""


def generate_full_dataset(
    settings: Settings,
    *,
    ddl: str,
    instructions: str,
    rows_per_table: int,
    temperature: float,
    trace_context: LangfuseTraceContext | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[str], ParsedSchema]:
    schema = parse_ddl(ddl)
    if not schema.tables:
        raise ValueError("No tables found in DDL. Check CREATE TABLE syntax.")
    if len(schema.tables) > 12:
        raise ValueError("This demo supports at most ~12 tables; simplify the schema.")
    prompt = build_full_generation_prompt(schema, ddl, instructions, rows_per_table)
    raw: dict[str, Any] = generate_json(
        settings,
        prompt,
        temperature=temperature,
        trace_context=trace_context,
    )
    if not isinstance(raw, dict):
        raise ValueError("Model returned non-object JSON root")
    tables_raw = raw.get("tables")
    if not isinstance(tables_raw, dict):
        raise ValueError("JSON must contain a 'tables' object")
    tables: dict[str, list[dict[str, Any]]] = {}
    for k, v in tables_raw.items():
        if k not in schema.tables:
            continue
        if not isinstance(v, list):
            continue
        tables[k] = [dict(r) for r in v if isinstance(r, dict)]
    errors = validate_tables_data(tables, schema)
    return tables, errors, schema


def refine_table(
    settings: Settings,
    *,
    schema: ParsedSchema,
    table_name: str,
    all_tables: dict[str, list[dict[str, Any]]],
    user_feedback: str,
    temperature: float,
    trace_context: LangfuseTraceContext | None,
) -> list[dict[str, Any]]:
    if table_name not in schema.tables:
        raise ValueError(f"Unknown table {table_name!r}")
    current = [dict(r) for r in all_tables.get(table_name, [])]
    prompt = build_table_refinement_prompt(schema, table_name, current, all_tables, user_feedback)
    out = generate_json(settings, prompt, temperature=temperature, trace_context=trace_context)
    if not isinstance(out, list):
        raise ValueError("Refinement must return a JSON array of rows")
    return [dict(r) for r in out if isinstance(r, dict)]


def persist_and_zip(
    *,
    dataset_id: str,
    tables: dict[str, list[dict[str, Any]]],
    ddl: str,
    instructions: str,
    rows_per_table: int,
    temperature: float,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> tuple[Path, bytes]:
    folder = save_dataset(
        data_root=data_root,
        dataset_id=dataset_id,
        tables=tables,
        ddl=ddl,
        instructions=instructions,
        rows_per_table=rows_per_table,
        temperature=temperature,
    )
    return folder, zip_dataset(folder)
