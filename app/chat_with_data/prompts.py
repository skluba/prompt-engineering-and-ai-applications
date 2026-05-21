"""Prompts for NL → structured SQL + chart plan."""

from __future__ import annotations


def nl_sql_chart_prompt(
    *,
    schema_text: str,
    table_names: list[str],
    history_lines: list[str],
    user_message: str,
    phase3_few_shots: str = "",
) -> str:
    """Ask the model for JSON: assistant_message, sql, chart."""
    hist = "\n".join(history_lines[-20:]) if history_lines else "(no prior messages)"
    tables_csv = ", ".join(f'"{t}"' for t in table_names)
    extras = ""
    fs = phase3_few_shots.strip()
    if fs:
        extras = f"""

{fs}

Reminder: Below examples may use markdown fences solely for readability; YOUR answer must remain a
single bare JSON object (no prose, no fences) as specified."""

    return f"""You are an analytics assistant. The user has CSV-backed tables in DuckDB.
{schema_text}{extras}

Table names for SQL: {tables_csv}

Conversation so far:
{hist}

User message:
{user_message}

Return a single JSON object with keys:
- "assistant_message": short (1-2 sentences) plan in plain language for the user.
- "sql": one DuckDB-compatible SELECT or WITH query only. Use joins and aggregates when helpful.
  Quote identifiers with double quotes if names are mixed case. No semicolons at the end.
- "chart": object describing an optional chart, or {{"kind":"none"}} if a chart is not useful.
  Allowed kind values: "bar", "line", "scatter", "hist", "none".
  For bar/line/scatter set "x" and "y" to column names **as returned by the query** (aliases count).
  Optional "hue" for grouping. Optional "title".

Rules:
- sql must be read-only (SELECT/WITH only).
- Prefer limiting heavy scans; you may use LIMIT 500 in SQL if the user does not need all rows.
- If the question cannot be answered from these tables, set sql to
  SELECT 1 AS ok WHERE false and explain in assistant_message.

Output JSON only, no markdown fences.
"""
