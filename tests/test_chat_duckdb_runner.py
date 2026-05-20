"""DuckDB dataset session over CSV files."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.chat_with_data.duckdb_runner import open_dataset_session


def test_open_dataset_session_join(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text("order_id,customer_id\n1,10\n2,10\n", encoding="utf-8")
    (tmp_path / "customers.csv").write_text("customer_id,name\n10,Ann\n", encoding="utf-8")
    con, tables, schema = open_dataset_session(tmp_path)
    try:
        assert sorted(tables) == ["customers", "orders"]
        assert "orders" in schema and "customers" in schema
        df = con.execute(
            """
            SELECT o.order_id, c.name
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            """
        ).df()
        assert len(df) == 2
        assert set(df["name"]) == {"Ann"}
    finally:
        con.close()


def test_open_dataset_session_rejects_bad_csv_stem(tmp_path: Path) -> None:
    (tmp_path / "bad-name.csv").write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid table name"):
        open_dataset_session(tmp_path)
