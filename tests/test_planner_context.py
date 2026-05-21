"""Tests for Phase 3 NL→SQL prompt augmentation helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.chat_with_data import planner_context
from app.chat_with_data.planner_context import (
    narrow_schema_context,
    reset_nl_sql_prompt_caches,
    retrieve_few_shot_block,
)
from app.config import get_settings


@pytest.fixture(autouse=True)
def planner_hygiene(request: pytest.FixtureRequest) -> None:
    reset_nl_sql_prompt_caches()
    request.addfinalizer(reset_nl_sql_prompt_caches)


@pytest.fixture(autouse=True)
def ephemeral_fewshot_json(
    planner_hygiene,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(planner_context, "_FEWSHOT_PATH", tmp_path / "few.json")
    (tmp_path / "few.json").write_text(
        '[{"question": "AAA unique seed phrase one", '
        '"sql": "SELECT 1"}, {"question": "BBB other phrase", '
        '"sql": "SELECT 2"}]',
        encoding="utf-8",
    )
    reset_nl_sql_prompt_caches()


def test_retrieve_few_shot_fallback_to_lexical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    settings = get_settings()

    with patch.object(planner_context, "embed_texts", side_effect=RuntimeError("no network")):
        block = retrieve_few_shot_block(settings, retrieval_blob="AAA unique seed phrase one")

    assert "AAA unique seed phrase one" in block
    assert "SELECT 1" in block


def test_retrieve_skips_when_count_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("NL_SQL_FEW_SHOT_COUNT", "0")
    get_settings.cache_clear()
    assert retrieve_few_shot_block(get_settings(), retrieval_blob="hello") == ""


def test_narrow_schema_truncates_with_stub_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NL_SQL_FULL_SCHEMA_TABLE_THRESHOLD", "2")
    monkeypatch.setenv("NL_SQL_MAX_DETAILED_SCHEMA_TABLES", "2")
    get_settings.cache_clear()
    settings = get_settings()
    schemas = {
        "alpha": "- alpha: id (INTEGER)",
        "beta": "- beta: amt (DOUBLE)",
        "gamma": "- gamma: zoo (VARCHAR)",
    }
    _text, supplemental = narrow_schema_context(
        settings,
        user_message="only alpha matters here",
        history_lines=["User: unrelated"],
        tables=["alpha", "beta", "gamma"],
        schema_by_table=schemas,
    )
    assert supplemental.startswith("Other DuckDB CSV views omitted")


def test_narrow_schema_keeps_everything_small_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NL_SQL_FULL_SCHEMA_TABLE_THRESHOLD", "5")
    get_settings.cache_clear()
    settings = get_settings()
    schemas = {"a": "- a: x INT", "b": "- b: y INT"}
    text, supplemental = narrow_schema_context(
        settings,
        user_message="q",
        history_lines=[],
        tables=["a", "b"],
        schema_by_table=schemas,
    )
    assert "- a:" in text and "- b:" in text
    assert supplemental == ""


def test_l2_normalize_rows_empty_matrix() -> None:
    mat = np.zeros((0, 3), dtype=float)
    out = planner_context._l2_normalize_rows(mat)
    assert out.size == 0


def test_clip_blob_truncates() -> None:
    raw = planner_context._clip_blob("abcdefghij" * 200, max_chars=25)
    assert raw.endswith("…[truncated]")


def test_load_fewshots_json_not_list_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(planner_context, "_FEWSHOT_PATH", p)
    reset_nl_sql_prompt_caches()
    assert planner_context._load_fewshots() == []


def test_load_fewshots_skips_rows_missing_q_or_sql(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    p = tmp_path / "partial.json"
    p.write_text(
        '[{"question":"","sql":"SELECT 1"}, {"question":"ok", "sql":""}, '
        '{"question":"good", "sql":"SELECT 2"}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(planner_context, "_FEWSHOT_PATH", p)
    reset_nl_sql_prompt_caches()
    rows = planner_context._load_fewshots()
    assert len(rows) == 1 and rows[0]["question"] == "good"


def test_retrieve_few_shot_with_dense_embedding_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    settings = get_settings()

    def fake_embed(_s, texts: list[str]):
        if len(texts) == 2:
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        if len(texts) == 1:
            return [[1.0, 0.0, 0.0]]
        return []

    with patch.object(planner_context, "embed_texts", side_effect=fake_embed):
        block = retrieve_few_shot_block(settings, retrieval_blob="AAA unique seed phrase one")

    assert "AAA unique seed phrase one" in block


def test_matrix_cache_skipped_when_example_vectors_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    settings = get_settings()

    def fake_embed(_s, texts: list[str]):
        if len(texts) == 2:
            return [[1.0]]  # wrong length — cannot build matrix
        return [[1.0, 0.0]]

    with patch.object(planner_context, "embed_texts", side_effect=fake_embed):
        block = retrieve_few_shot_block(settings, retrieval_blob="AAA unique seed phrase one")

    assert "AAA" in block


def test_dense_scoring_falls_back_when_query_embedding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    settings = get_settings()

    calls = 0

    def fake_embed(_s, texts: list[str]):
        nonlocal calls
        if len(texts) == 2:
            return [[1.0, 0.0], [0.0, 1.0]]
        calls += 1
        if calls == 1:
            raise RuntimeError("query embed boom")
        return [[0.0, 1.0]]

    with patch.object(planner_context, "embed_texts", side_effect=fake_embed):
        block = retrieve_few_shot_block(settings, retrieval_blob="AAA unique seed phrase one")

    assert "AAA" in block


def test_ensure_matrix_returns_none_when_vertex_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    get_settings.cache_clear()
    settings = get_settings()
    mat = planner_context._ensure_example_matrix(settings, ["a", "b"], "k")
    assert mat is None


def test_narrow_schema_prioritizes_table_from_history_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NL_SQL_FULL_SCHEMA_TABLE_THRESHOLD", "2")
    monkeypatch.setenv("NL_SQL_MAX_DETAILED_SCHEMA_TABLES", "1")
    get_settings.cache_clear()
    settings = get_settings()
    schemas = {
        "alpha": "- alpha: id (INTEGER)",
        "beta": "- beta: amt (DOUBLE)",
        "gamma": "- gamma: zoo (VARCHAR)",
    }
    text, supplemental = narrow_schema_context(
        settings,
        user_message="totally vague",
        history_lines=["User: SELECT gamma.zoo FROM gamma"],
        tables=["alpha", "beta", "gamma"],
        schema_by_table=schemas,
    )
    assert "- gamma:" in text
    assert supplemental.startswith("Other DuckDB CSV views omitted")


def test_build_nl_sql_augmentations_combines_few_shots_and_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("NL_SQL_FULL_SCHEMA_TABLE_THRESHOLD", "5")
    get_settings.cache_clear()
    settings = get_settings()
    schemas = {"t1": "- t1: id INT", "t2": "- t2: v TEXT"}
    few, combo = planner_context.build_nl_sql_augmentations(
        settings,
        user_message="show me rows",
        history_lines=[],
        tables=["t1", "t2"],
        schema_by_table=schemas,
    )
    assert "Retrieved few-shot" in few
    assert "- t1:" in combo and "- t2:" in combo


def test_retrieve_empty_when_no_fewshot_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    miss = tmp_path / "missing-fewshots.json"
    monkeypatch.setattr(planner_context, "_FEWSHOT_PATH", miss)
    reset_nl_sql_prompt_caches()
    assert retrieve_few_shot_block(get_settings(), retrieval_blob="anything") == ""
