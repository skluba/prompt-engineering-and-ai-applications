"""Tests for conversational synthetic-data refinement routing."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.schema_ddl import parse_ddl
import app.synthetic.refine_router as refine_router
from app.synthetic.refine_router import (
    SyntheticRefinementPlan,
    apply_refinement_plan,
    build_refinement_router_prompt,
    expand_target_tables,
    parse_router_payload,
    plan_dataset_refinement_turn,
    schema_column_ref,
)


def _ddl_pair() -> str:
    return """
    CREATE TABLE A (id INT PRIMARY KEY NOT NULL, name VARCHAR(20));
    CREATE TABLE B (id INT PRIMARY KEY NOT NULL, a_id INT NOT NULL,
      FOREIGN KEY (a_id) REFERENCES A(id));
    """


def test_schema_column_ref_lists_tables_and_columns() -> None:
    schema = parse_ddl(_ddl_pair())
    blob = schema_column_ref(schema)
    assert "A:" in blob and "B:" in blob
    assert "id" in blob and "name" in blob


def test_expand_target_tables_star() -> None:
    schema = parse_ddl(_ddl_pair())
    names = schema.ordered_table_names()
    assert expand_target_tables(["*"], schema) == names
    assert expand_target_tables([" A ", " B "], schema) == names


def test_expand_target_tables_filters_unknown() -> None:
    schema = parse_ddl(_ddl_pair())
    assert expand_target_tables(["A", "Nope"], schema) == ["A"]


def test_parse_router_payload_handles_missing_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = parse_ddl(_ddl_pair())
    out = parse_router_payload(
        {
            "assistant_reply": "",
            "apply_refinements": True,
            "target_tables": ["A"],
            "refinement_instruction": "",
        },
        schema,
    )
    assert out.apply_refinements is False
    assert not out.refinement_instruction


def test_plan_dataset_refinement_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    ddl = _ddl_pair()
    schema = parse_ddl(ddl)

    def fake_json(_s, _p: str, **_k):
        return {
            "assistant_reply": "Will update null counts.",
            "apply_refinements": True,
            "target_tables": ["A"],
            "refinement_instruction": "About 40% NULL in column name.",
        }

    monkeypatch.setattr("app.synthetic.refine_router.generate_json", fake_json)
    plan = plan_dataset_refinement_turn(
        get_settings(),
        schema=schema,
        ddl=ddl,
        original_instructions="seed",
        chat_messages=[
            {"role": "user", "content": "Make name often null"},
        ],
        trace_context=None,
    )
    assert plan.apply_refinements is True
    assert plan.target_tables == ("A",)
    assert "NULL" in plan.refinement_instruction


def test_apply_refinement_plan_calls_refine_once_per_table_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    schema = parse_ddl(_ddl_pair())

    captured: list[str] = []

    def fake_refine(
        *_a,
        table_name: str,
        **_k,
    ):
        captured.append(table_name)
        return [{"id": 9, "a_id": 1}] if table_name == "B" else [{"id": 1, "name": None}]

    monkeypatch.setattr("app.synthetic.refine_router.refine_table", fake_refine)

    tables = {"A": [{"id": 1, "name": "x"}], "B": [{"id": 9, "a_id": 1}]}
    plan = SyntheticRefinementPlan(
        assistant_reply="-",
        apply_refinements=True,
        target_tables=("A", "B"),
        refinement_instruction="apply change",
    )
    out, errs = apply_refinement_plan(
        get_settings(),
        schema=schema,
        tables=tables,
        plan=plan,
        refinement_temperature=0.0,
        trace_context=None,
    )
    assert captured == ["A", "B"]
    assert errs == []
    assert out["A"][0]["name"] is None


def test_format_recent_chat_skips_blank_lines() -> None:
    assert refine_router._format_recent_chat([]) == "(no prior messages)"
    out = refine_router._format_recent_chat(
        [
            {"role": "user", "content": "   "},
            {"role": "user", "content": "keep me"},
        ]
    )
    assert "USER: keep me" in out
    assert out.count("USER:") == 1


def test_build_refinement_router_prompt_truncates_huge_ddl() -> None:
    schema = parse_ddl("CREATE TABLE A (id INT PRIMARY KEY);")
    huge = "x\n" * 4500
    prompt = build_refinement_router_prompt(
        schema,
        ddl="CREATE TABLE A (id INT);" + huge,
        original_instructions="y",
        conversation_transcript="z",
    )
    assert "…[DDL truncated]" in prompt


def test_expand_target_tables_empty_requested() -> None:
    schema = parse_ddl(_ddl_pair())
    assert expand_target_tables([], schema) == []
    assert expand_target_tables(["", "  ", "\t"], schema) == []


def test_parse_router_payload_rejects_non_object() -> None:
    schema = parse_ddl(_ddl_pair())
    with pytest.raises(ValueError, match="non-object"):
        parse_router_payload(["not", "a", "dict"], schema)


def test_parse_router_payload_non_list_targets_with_instruction_merges_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = parse_ddl(_ddl_pair())
    out = parse_router_payload(
        {
            "assistant_reply": "picked none",
            "apply_refinements": True,
            "target_tables": "garbage-not-a-list",
            "refinement_instruction": "still do something",
        },
        schema,
    )
    assert out.apply_refinements is False
    assert "picked none" in out.assistant_reply


def test_parse_router_payload_missing_instruction_merges_when_assistant_present() -> None:
    schema = parse_ddl(_ddl_pair())
    out = parse_router_payload(
        {
            "assistant_reply": "Try again with columns",
            "apply_refinements": True,
            "target_tables": ["A"],
            "refinement_instruction": "",
        },
        schema,
    )
    assert out.apply_refinements is False
    assert "Try again with columns" in out.assistant_reply
    assert "spell out" in out.assistant_reply.lower()


def test_apply_refinement_plan_early_exit_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    schema = parse_ddl(_ddl_pair())
    tables = {"A": [{"id": 1}], "B": [{"id": 1, "a_id": 1}]}
    calls: list[str] = []

    def track_refine(*_a, **k):
        calls.append(k.get("table_name", ""))
        return []

    monkeypatch.setattr("app.synthetic.refine_router.refine_table", track_refine)

    noop, _e = apply_refinement_plan(
        get_settings(),
        schema=schema,
        tables=tables,
        plan=SyntheticRefinementPlan("x", False, ("A",), "x"),
        refinement_temperature=0.0,
        trace_context=None,
    )
    assert noop is tables
    assert calls == []

    noop2, _ = apply_refinement_plan(
        get_settings(),
        schema=schema,
        tables=tables,
        plan=SyntheticRefinementPlan("x", True, (), "x"),
        refinement_temperature=0.0,
        trace_context=None,
    )
    assert noop2 is tables
    assert calls == []

    noop3, _ = apply_refinement_plan(
        get_settings(),
        schema=schema,
        tables=tables,
        plan=SyntheticRefinementPlan("x", True, ("A",), "   "),
        refinement_temperature=0.0,
        trace_context=None,
    )
    assert noop3 is tables
    assert calls == []


def test_apply_refinement_plan_skips_unknown_table_stem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    schema = parse_ddl(_ddl_pair())

    def boom(*_a, **k):
        raise AssertionError("refine_table should not run for unknown stems")

    monkeypatch.setattr("app.synthetic.refine_router.refine_table", boom)

    tables = {"A": [{"id": 1, "name": "y"}], "B": [{"id": 9, "a_id": 1}]}
    out, errs = apply_refinement_plan(
        get_settings(),
        schema=schema,
        tables=tables,
        plan=SyntheticRefinementPlan("x", True, ("NoSuchTable",), "nope"),
        refinement_temperature=0.0,
        trace_context=None,
    )
    assert out == tables
    assert errs == []


def test_plan_dataset_refinement_turn_builds_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    schema = parse_ddl(_ddl_pair())
    captured: dict[str, str] = {}

    def capture_json(settings, prompt: str, **_k):
        captured["prompt"] = prompt
        return {
            "assistant_reply": "ok",
            "apply_refinements": False,
            "target_tables": [],
            "refinement_instruction": "",
        }

    monkeypatch.setattr("app.synthetic.refine_router.generate_json", capture_json)
    plan_dataset_refinement_turn(
        get_settings(),
        schema=schema,
        ddl=_ddl_pair(),
        original_instructions="seed",
        chat_messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ],
        trace_context=None,
    )
    assert "USER: a" in captured["prompt"] and "ASSISTANT: b" in captured["prompt"]
