"""prepare_turn / prepare_sql_only with LLM and DuckDB mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.chat_with_data.turn import prepare_sql_only, prepare_turn
from app.config import get_settings
from app.tracing import LangfuseTraceContext


def test_prepare_turn_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    (tmp_path / "t.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    settings = get_settings()
    plan = {
        "assistant_message": "sum b",
        "sql": "SELECT SUM(b) AS s FROM t",
        "chart": {"kind": "none"},
    }
    ctx = LangfuseTraceContext()
    with patch("app.chat_with_data.turn.generate_json", return_value=plan):
        out = prepare_turn(
            settings,
            tmp_path,
            [],
            "total of b",
            trace_json=ctx,
        )
    assert out.error is None
    assert "SUM" in out.sql.upper()
    assert not out.df.empty
    assert int(out.df.iloc[0]["s"]) == 6


def test_prepare_turn_invalid_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    (tmp_path / "t.csv").write_text("a\n1\n", encoding="utf-8")
    settings = get_settings()
    with patch("app.chat_with_data.turn.generate_json", return_value=[]):
        out = prepare_turn(settings, tmp_path, [], "q", trace_json=LangfuseTraceContext())
    assert out.error is not None


def test_prepare_turn_passes_conversation_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    (tmp_path / "t.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    settings = get_settings()
    captured: dict = {}

    def fake_json(_settings, prompt: str, **_kwargs):
        captured["prompt"] = prompt
        return {
            "assistant_message": "ok",
            "sql": "SELECT SUM(b) AS s FROM t",
            "chart": {"kind": "none"},
        }

    with patch("app.chat_with_data.turn.generate_json", side_effect=fake_json):
        prepare_turn(
            settings,
            tmp_path,
            [
                {"role": "user", "text": "first"},
                {"role": "assistant", "text": "hi", "sql": "SELECT 1"},
            ],
            "second question",
            trace_json=LangfuseTraceContext(),
        )
    assert "first" in captured.get("prompt", "")
    assert "SELECT 1" in captured.get("prompt", "")
    assert "second question" in captured.get("prompt", "")


def test_prepare_sql_only_runs_guarded_query(tmp_path: Path) -> None:
    (tmp_path / "t.csv").write_text("x\n1\n2\n", encoding="utf-8")
    out = prepare_sql_only(tmp_path, "SELECT COUNT(*) AS c FROM t", chart_spec=None)
    assert out.error is None
    assert int(out.df.iloc[0]["c"]) == 2
