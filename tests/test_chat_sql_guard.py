"""Tests for read-only SQL guard."""

from __future__ import annotations

import pytest

from app.chat_with_data.sql_guard import validate_read_only_select


def test_validate_accepts_select() -> None:
    assert validate_read_only_select("  SELECT 1 AS x  ") == "SELECT 1 AS x"


def test_validate_accepts_with() -> None:
    sql = "WITH t AS (SELECT 1) SELECT * FROM t"
    assert validate_read_only_select(sql) == sql


def test_validate_strips_trailing_semicolon() -> None:
    assert validate_read_only_select("SELECT 1;") == "SELECT 1"


def test_validate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_read_only_select("   ")


def test_validate_rejects_multiple_statements() -> None:
    with pytest.raises(ValueError, match="single"):
        validate_read_only_select("SELECT 1; SELECT 2")


def test_validate_rejects_non_select() -> None:
    with pytest.raises(ValueError, match="SELECT"):
        validate_read_only_select("DELETE FROM t")


def test_validate_rejects_forbidden_update_in_cte() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_read_only_select(
            "WITH u AS (UPDATE orders SET id = id RETURNING *) SELECT * FROM u",
        )
