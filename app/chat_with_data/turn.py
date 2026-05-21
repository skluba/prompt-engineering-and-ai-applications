"""Orchestrate NL → SQL, execution, chart; explanation streams in the UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.chat_with_data.charts import build_chart_png
from app.chat_with_data.duckdb_runner import open_dataset_session, run_query
from app.chat_with_data.prompts import nl_sql_chart_prompt
from app.chat_with_data.sql_guard import validate_read_only_select
from app.config import Settings
from app.llm import generate_json
from app.tracing import LangfuseTraceContext


@dataclass
class PreparedTurn:
    """SQL executed; ready for streamed narration in the caller."""

    sql: str
    df: pd.DataFrame
    chart_png: bytes | None
    plan_message: str
    stream_prompt: str
    chart_spec: dict[str, Any] | None = None
    error: str | None = None


def _history_lines(messages: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "")
        if role == "user":
            # Prefer model-facing text (PII-masked); fall back for legacy messages.
            lines.append(f"User: {m.get('text', m.get('display_text', ''))}")
        elif role == "assistant":
            t = m.get("text", "")
            sql = m.get("sql")
            if sql:
                lines.append(f"Assistant: {t}\n(SQL used: {sql})")
            else:
                lines.append(f"Assistant: {t}")
    return lines


def _df_summary(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df.empty:
        return "0 rows."
    head = df.head(max_rows).to_csv(index=False)
    return f"shape={df.shape}\nColumns: {list(df.columns)}\nSample:\n{head}"


def prepare_turn(
    settings: Settings,
    dataset_dir: Path,
    messages: list[dict[str, Any]],
    user_message: str,
    *,
    trace_json: LangfuseTraceContext,
) -> PreparedTurn:
    """Call Gemini for SQL+chart plan, execute query, build chart; narration streams separately."""
    con, table_names, schema_text = open_dataset_session(dataset_dir)
    try:
        prompt = nl_sql_chart_prompt(
            schema_text=schema_text,
            table_names=table_names,
            history_lines=_history_lines(messages),
            user_message=user_message,
        )
        raw_plan = generate_json(settings, prompt, temperature=0.2, trace_context=trace_json)
        if not isinstance(raw_plan, dict):
            raise ValueError("Model returned non-object JSON.")
        sql_raw = str(raw_plan.get("sql") or "").strip()
        plan_msg = str(raw_plan.get("assistant_message") or "").strip()
        chart_spec = raw_plan.get("chart")
        if isinstance(chart_spec, dict):
            chart_dict: dict[str, Any] | None = chart_spec
        else:
            chart_dict = None

        sql = validate_read_only_select(sql_raw)
        df = run_query(con, sql).df()
        chart_png = build_chart_png(df, chart_dict)

        summary = _df_summary(df)
        stream_prompt = (
            f"The user asked:\n{user_message}\n\n"
            f"Planner note:\n{plan_msg}\n\n"
            f"Executed SQL:\n{sql}\n\n"
            f"Result summary:\n{summary}\n\n"
            "Write a concise answer for the user: interpret the results, mention key numbers, "
            "and note caveats (e.g. row limits). Do not repeat the full SQL unless essential."
        )

        return PreparedTurn(
            sql=sql,
            df=df,
            chart_png=chart_png,
            plan_message=plan_msg,
            stream_prompt=stream_prompt,
            chart_spec=chart_dict,
        )
    except Exception as exc:  # noqa: BLE001
        return PreparedTurn(
            sql="",
            df=pd.DataFrame(),
            chart_png=None,
            plan_message="",
            stream_prompt="",
            chart_spec=None,
            error=str(exc),
        )
    finally:
        con.close()


def prepare_sql_only(
    dataset_dir: Path,
    sql: str,
    *,
    chart_spec: dict[str, Any] | None,
) -> PreparedTurn:
    """Execute user-edited SQL with guard; caller streams explanation."""
    con, _names, _schema = open_dataset_session(dataset_dir)
    try:
        safe = validate_read_only_select(sql)
        df = run_query(con, safe).df()
        chart_png = build_chart_png(df, chart_spec)
        summary = _df_summary(df)
        stream_prompt = (
            f"User ran custom SQL:\n{safe}\n\n"
            f"Result summary:\n{summary}\n\n"
            "Briefly describe what the result shows (2-4 sentences)."
        )
        return PreparedTurn(
            sql=safe,
            df=df,
            chart_png=chart_png,
            plan_message="",
            stream_prompt=stream_prompt,
            chart_spec=chart_spec,
        )
    except Exception as exc:  # noqa: BLE001
        return PreparedTurn(
            sql=sql.strip(),
            df=pd.DataFrame(),
            chart_png=None,
            plan_message="",
            stream_prompt="",
            chart_spec=chart_spec,
            error=str(exc),
        )
    finally:
        con.close()
