"""Tests for conversational synthetic-data refinement routing."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.schema_ddl import parse_ddl
from app.synthetic.refine_router import (
    SyntheticRefinementPlan,
    apply_refinement_plan,
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
