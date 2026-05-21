"""Tests for Phase 3 NL→SQL prompt augmentation helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
