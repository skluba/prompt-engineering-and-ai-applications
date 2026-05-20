"""Read-only SQL guard for DuckDB (single SELECT / WITH statement)."""

from __future__ import annotations

import re

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|CREATE|ALTER|TRUNCATE|COPY|ATTACH|DETACH|"
    r"PRAGMA|CALL|EXPORT|IMPORT|INSTALL|LOAD|CHECKPOINT|VACUUM|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)


def validate_read_only_select(sql: str) -> str:
    """Return normalized single-statement SQL or raise ValueError."""
    raw = (sql or "").strip()
    if not raw:
        raise ValueError("SQL is empty.")
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    if len(parts) != 1:
        raise ValueError("Only a single SQL statement is allowed (no multiple statements).")
    stmt = parts[0]
    head = stmt.lstrip()
    upper = head[:20].upper() if head else ""
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("Only SELECT or WITH (CTE) queries are allowed.")
    if _FORBIDDEN.search(stmt):
        raise ValueError("Query contains forbidden keywords for read-only mode.")
    return stmt
