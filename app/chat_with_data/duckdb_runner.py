"""In-memory DuckDB over CSV files in a dataset folder."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb


def _safe_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        raise ValueError(f"Invalid table name for CSV stem: {name!r}")
    return name


def _csv_path_literal(csv_path: Path) -> str:
    """Single-quoted SQL string literal for read_csv_auto (path is local, not user SQL)."""
    return str(csv_path.resolve()).replace("'", "''")


def open_dataset_session(dataset_dir: Path) -> tuple[duckdb.DuckDBPyConnection, list[str], str]:
    """In-memory DuckDB: one view per ``*.csv``; returns connection, table names, schema text."""
    con = duckdb.connect(database=":memory:")
    tables: list[str] = []
    schema_lines: list[str] = []
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        stem = _safe_ident(csv_path.stem)
        quoted = f'"{stem}"'
        path_sql = _csv_path_literal(csv_path)
        con.execute(
            f"CREATE OR REPLACE VIEW {quoted} AS SELECT * FROM read_csv_auto('{path_sql}');",
        )
        tables.append(stem)
        desc = con.execute(f"DESCRIBE SELECT * FROM {quoted}").fetchall()
        cols = ", ".join(f"{row[0]} ({row[1]})" for row in desc)
        schema_lines.append(f"- {stem}: {cols}")
    schema_text = "Available tables (use these exact names, double-quote if needed):\n" + "\n".join(
        schema_lines
    )
    return con, tables, schema_text


def run_query(con: duckdb.DuckDBPyConnection, sql: str):
    """Execute SQL and return a DuckDB relation (caller converts to DataFrame)."""
    return con.execute(sql)
