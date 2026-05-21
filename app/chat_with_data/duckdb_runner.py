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


def open_dataset_session(
    dataset_dir: Path,
) -> tuple[duckdb.DuckDBPyConnection, list[str], dict[str, str], str]:
    """In-memory DuckDB: one view per ``*.csv``.

    Returns ``(connection, table_stems_sorted, stem_to_schema_line, full_schema_blob)``.
    Close the connection in a ``finally`` block.
    """
    con = duckdb.connect(database=":memory:")
    tables: list[str] = []
    schema_by_table: dict[str, str] = {}
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
        line = f"- {stem}: {cols}"
        schema_lines.append(line)
        schema_by_table[stem] = line
    schema_text = "Available tables (use these exact names, double-quote if needed):\n" + "\n".join(
        schema_lines
    )
    return con, tables, schema_by_table, schema_text


def run_query(con: duckdb.DuckDBPyConnection, sql: str):
    """Execute SQL and return a DuckDB relation (caller converts to DataFrame)."""
    return con.execute(sql)
