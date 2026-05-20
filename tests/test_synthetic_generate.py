"""Tests for synthetic generation orchestration (LLM calls mocked)."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.schema_ddl import parse_ddl
from app.synthetic.generate import (
    build_full_generation_prompt,
    build_table_refinement_prompt,
    generate_full_dataset,
    persist_and_zip,
    refine_table,
)


def _tiny_ddl() -> str:
    return """
    CREATE TABLE P (id INT PRIMARY KEY NOT NULL, name VARCHAR(5));
    CREATE TABLE C (id INT PRIMARY KEY NOT NULL, p_id INT NOT NULL,
      FOREIGN KEY (p_id) REFERENCES P(id));
    """


def test_build_full_generation_prompt_contains_order_and_summary() -> None:
    ddl = _tiny_ddl()
    schema = parse_ddl(ddl)
    p = build_full_generation_prompt(schema, ddl, "be brief", 3)
    assert "OUTPUT SHAPE" in p
    assert "P" in p and "C" in p
    assert "be brief" in p
    assert "3" in p


def test_build_table_refinement_prompt_includes_fk_context() -> None:
    ddl = _tiny_ddl()
    schema = parse_ddl(ddl)
    all_tables = {
        "P": [{"id": 1, "name": "a"}],
        "C": [{"id": 10, "p_id": 1}],
    }
    out = build_table_refinement_prompt(schema, "C", all_tables["C"], all_tables, "more rows")
    assert "C" in out
    assert "P" in out
    assert "more rows" in out


def test_generate_full_dataset_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    ddl = _tiny_ddl()
    schema = parse_ddl(ddl)

    def fake_json(_settings, _prompt: str, **_kwargs):
        return {
            "tables": {
                "P": [{"id": 1, "name": "x"}],
                "C": [{"id": 2, "p_id": 1}],
            },
        }

    monkeypatch.setattr("app.synthetic.generate.generate_json", fake_json)
    tables, errors, out_schema = generate_full_dataset(
        get_settings(),
        ddl=ddl,
        instructions="",
        rows_per_table=1,
        temperature=0.1,
        trace_context=None,
    )
    assert out_schema.tables.keys() == schema.tables.keys()
    assert set(tables.keys()) == {"P", "C"}
    assert len(errors) == 0


def test_generate_full_dataset_no_tables_raises() -> None:
    with pytest.raises(ValueError, match="No tables"):
        generate_full_dataset(
            get_settings(),
            ddl="SELECT 1;",
            instructions="",
            rows_per_table=1,
            temperature=0.0,
            trace_context=None,
        )


def test_generate_full_dataset_too_many_tables_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    ddl = "\n".join(f"CREATE TABLE t{i} (id INT PRIMARY KEY);" for i in range(13))
    with pytest.raises(ValueError, match="12"):
        generate_full_dataset(
            get_settings(),
            ddl=ddl,
            instructions="",
            rows_per_table=1,
            temperature=0.0,
            trace_context=None,
        )


def test_generate_full_dataset_rejects_non_object_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    monkeypatch.setattr("app.synthetic.generate.generate_json", lambda *a, **k: [])
    with pytest.raises(ValueError, match="non-object"):
        generate_full_dataset(
            get_settings(),
            ddl=_tiny_ddl(),
            instructions="",
            rows_per_table=1,
            temperature=0.0,
            trace_context=None,
        )


def test_generate_full_dataset_requires_tables_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    monkeypatch.setattr("app.synthetic.generate.generate_json", lambda *a, **k: {"x": 1})
    with pytest.raises(ValueError, match="tables"):
        generate_full_dataset(
            get_settings(),
            ddl=_tiny_ddl(),
            instructions="",
            rows_per_table=1,
            temperature=0.0,
            trace_context=None,
        )


def test_generate_full_dataset_ignores_unknown_table_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()

    def fake(_s, _p, **_k):
        return {"tables": {"P": [{"id": 1, "name": "z"}], "Nope": [{"a": 1}]}}

    monkeypatch.setattr("app.synthetic.generate.generate_json", fake)
    tables, _, _ = generate_full_dataset(
        get_settings(),
        ddl=_tiny_ddl(),
        instructions="",
        rows_per_table=1,
        temperature=0.0,
        trace_context=None,
    )
    assert "P" in tables and "Nope" not in tables


def test_generate_full_dataset_skips_non_list_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()

    def fake(_s, _p, **_k):
        return {"tables": {"P": "broken", "C": [{"id": 2, "p_id": 1}]}}

    monkeypatch.setattr("app.synthetic.generate.generate_json", fake)
    tables, errors, _ = generate_full_dataset(
        get_settings(),
        ddl=_tiny_ddl(),
        instructions="",
        rows_per_table=1,
        temperature=0.0,
        trace_context=None,
    )
    assert "P" not in tables or tables.get("P") == []
    assert any("Missing table" in e for e in errors) or any("zero rows" in e for e in errors)


def test_refine_table_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    ddl = "CREATE TABLE T (id INT PRIMARY KEY);"
    schema = parse_ddl(ddl)
    monkeypatch.setattr(
        "app.synthetic.generate.generate_json",
        lambda *a, **k: [{"id": 42}],
    )
    out = refine_table(
        get_settings(),
        schema=schema,
        table_name="T",
        all_tables={"T": [{"id": 1}]},
        user_feedback="change id",
        temperature=0.0,
        trace_context=None,
    )
    assert out == [{"id": 42}]


def test_refine_table_unknown_name() -> None:
    schema = parse_ddl("CREATE TABLE T (id INT PRIMARY KEY);")
    with pytest.raises(ValueError, match="Unknown table"):
        refine_table(
            get_settings(),
            schema=schema,
            table_name="X",
            all_tables={},
            user_feedback="",
            temperature=0.0,
            trace_context=None,
        )


def test_refine_table_requires_array(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    schema = parse_ddl("CREATE TABLE T (id INT PRIMARY KEY);")
    monkeypatch.setattr("app.synthetic.generate.generate_json", lambda *a, **k: {})
    with pytest.raises(ValueError, match="array"):
        refine_table(
            get_settings(),
            schema=schema,
            table_name="T",
            all_tables={"T": []},
            user_feedback="x",
            temperature=0.0,
            trace_context=None,
        )


def test_persist_and_zip_round_trip(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    tables = {"P": [{"id": 1, "name": "a"}]}
    folder, zbuf = persist_and_zip(
        dataset_id="rid",
        tables=tables,
        ddl="x",
        instructions="y",
        rows_per_table=5,
        temperature=0.3,
        data_root=tmp_path,
    )
    assert folder.name == "rid"
    assert (folder / "P.csv").is_file()
    assert len(zbuf) > 50
