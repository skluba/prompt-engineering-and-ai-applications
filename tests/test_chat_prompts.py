"""Tests for nl_sql_chart_prompt."""

from __future__ import annotations

from app.chat_with_data.prompts import nl_sql_chart_prompt


def test_nl_sql_chart_prompt_includes_user_and_schema() -> None:
    p = nl_sql_chart_prompt(
        schema_text="Table a: x INT",
        table_names=["a", "b"],
        history_lines=["User: hi"],
        user_message="count rows",
    )
    assert "count rows" in p
    assert "Table a" in p
    assert '"a"' in p
    assert "JSON" in p
