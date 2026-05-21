"""Phase 3: few-shot retrieval and schema routing for NL → SQL DuckDB prompts."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import Settings
from app.llm import embed_texts

_FEWSHOT_PATH = Path(__file__).resolve().parent / "nl_sql_fewshots.json"

_TOKEN_RE = re.compile(r"[A-Za-z_]\w{2,}")

_example_matrix_cache: dict[str, np.ndarray] = {}


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-9)


def _clip_blob(*parts: str, max_chars: int = 3200) -> str:
    blob = "\n".join(p for p in parts if p).strip()
    if len(blob) <= max_chars:
        return blob
    return blob[: max_chars - 12] + "…[truncated]"


def _load_fewshots() -> list[dict[str, str]]:
    if not _FEWSHOT_PATH.is_file():
        return []
    data = json.loads(_FEWSHOT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    out: list[dict[str, str]] = []
    for row in data:
        if isinstance(row, dict):
            q = str(row.get("question", "") or "").strip()
            s = str(row.get("sql", "") or "").strip()
            if q and s:
                out.append({"question": q, "sql": s})
    return out


@lru_cache(maxsize=1)
def load_nl_sql_examples() -> tuple[dict[str, str], ...]:
    """Frozen few-shot QA pairs bundled with the app (clear cache after editing JSON tests)."""
    return tuple(_load_fewshots())


def _lexical_overlap_score(query_blob: str, doc: str) -> float:
    qt = _tokens(query_blob)
    dt = _tokens(doc)
    if not qt or not dt:
        return 0.0
    inter = len(qt & dt)
    union = len(qt | dt)
    return float(inter / union)


def _ensure_example_matrix(settings: Settings, qs: list[str], model_key: str) -> np.ndarray | None:
    if not settings.vertex_configured():
        return None
    try:
        if model_key not in _example_matrix_cache:
            vectors = embed_texts(settings, qs)
            if len(vectors) == len(qs):
                arr = np.array(vectors, dtype=float)
                _example_matrix_cache[model_key] = _l2_normalize_rows(arr)
        return _example_matrix_cache.get(model_key)
    except Exception:
        return None


def _few_shot_combined_scores(
    settings: Settings,
    retrieval_blob: str,
    qs: list[str],
    scores_lex: np.ndarray,
    mat: np.ndarray | None,
) -> np.ndarray:
    if mat is None or mat.shape[0] != len(qs):
        return scores_lex
    try:
        qv_raw = embed_texts(settings, [retrieval_blob])
        qv = _l2_normalize_rows(np.asarray(qv_raw, dtype=float))
        dense = (mat @ qv[0]) if qv.shape[0] else np.zeros(len(qs))
        return 0.62 * dense + 0.38 * scores_lex
    except Exception:
        return scores_lex


def _format_few_shot_block(indices: Sequence[int], qlines: list[dict[str, str]]) -> str:
    lines: list[str] = [
        "Retrieved few-shot DuckDB-SQL patterns (table/column names are illustrative "
        "— adapt them strictly to THIS dataset's schemas):",
    ]
    for idx in indices:
        pair = qlines[int(idx)]
        lines.extend(["", f"Q: {pair['question']}", f"SQL:\n```sql\n{pair['sql']}\n```"])
    return "\n".join(lines)


def retrieve_few_shot_block(
    settings: Settings,
    *,
    retrieval_blob: str,
) -> str:
    """Markdown few-shot snippets most similar to the retrieval blob."""
    k = settings.nl_sql_few_shot_count
    if k <= 0:
        return ""

    qlines = list(load_nl_sql_examples())
    if not qlines:
        return ""

    qs = [r["question"] for r in qlines]
    model_key = settings.text_embedding_model.strip() or "__default__"
    mat = _ensure_example_matrix(settings, qs, model_key)

    scores_lex = np.array([_lexical_overlap_score(retrieval_blob, q) for q in qs], dtype=float)
    scores = _few_shot_combined_scores(settings, retrieval_blob, qs, scores_lex, mat)

    order = scores.argsort()[::-1][:k]
    return _format_few_shot_block(order, qlines)


def _stem_mentions(blob: str, tables: list[str]) -> set[str]:
    ulow = blob.lower()
    out: set[str] = set()
    for stem in tables:
        if re.search(rf"(?<!\w){re.escape(stem.lower())}(?!\w)", ulow):
            out.add(stem)
    return out


def _lexical_ranked_tables(
    retrieval: str,
    tables: list[str],
    schema_by_table: dict[str, str],
) -> list[str]:
    score_pairs = [
        (
            _lexical_overlap_score(
                retrieval,
                schema_by_table.get(stem, f"- {stem}: <unknown>") + " " + stem.lower(),
            ),
            stem,
        )
        for stem in tables
    ]
    score_pairs.sort(key=lambda pair: pair[0], reverse=True)
    return [pair[1] for pair in score_pairs]


def _ordered_detail_tables(
    *,
    limit: int,
    forced_by_mention: set[str],
    hist_refs: set[str],
    ranked_names: list[str],
    tables: list[str],
    schema_by_table: dict[str, str],
) -> list[str]:
    ordered_detail: list[str] = []

    def push(name: str) -> None:
        if name in schema_by_table and name not in ordered_detail:
            ordered_detail.append(name)

    for group in (sorted(forced_by_mention), sorted(hist_refs), ranked_names):
        for name in group:
            push(name)
            if len(ordered_detail) >= limit:
                return ordered_detail
    return ordered_detail or list(tables)


def _tables_referenced_in_sql(history_blob: str, stems: Sequence[str]) -> set[str]:
    """Return dataset stems appearing as identifiers in pasted SQL snippets."""
    out: set[str] = set()
    for stem in stems:
        pat = rf'(?is)\b(?:from|join)\s+(?:"|`)?{re.escape(stem)}(?:"|`)?\b'
        if re.search(pat, history_blob):
            out.add(stem)
    return out


def narrow_schema_context(
    settings: Settings,
    *,
    user_message: str,
    history_lines: list[str],
    tables: list[str],
    schema_by_table: dict[str, str],
) -> tuple[str, str]:
    """Return ``(detailed_schema_block, supplementary_note_about_unlisted_tables)``."""
    header = "Available tables (CSV-backed DuckDB views; identifiers must match filenames/stems):\n"

    merged_history = "\n".join(history_lines[-10:])
    blob = _clip_blob(user_message, "\n".join(history_lines[-8:]))

    if len(tables) <= settings.nl_sql_full_schema_table_threshold:
        joined = header + "\n".join(schema_by_table[t] for t in tables if t in schema_by_table)
        return joined, ""

    forced_by_mention = _stem_mentions(blob, tables)
    hist_refs = _tables_referenced_in_sql(merged_history, tables)
    retrieval = blob + "\n" + merged_history.lower()
    ranked_names = _lexical_ranked_tables(retrieval, tables, schema_by_table)
    ordered_detail = _ordered_detail_tables(
        limit=settings.nl_sql_max_detailed_schema_tables,
        forced_by_mention=forced_by_mention,
        hist_refs=hist_refs,
        ranked_names=ranked_names,
        tables=tables,
        schema_by_table=schema_by_table,
    )

    remainder = [t for t in tables if t not in ordered_detail]
    detailed_block = header + "\n".join(schema_by_table[t] for t in ordered_detail)

    supplemental = ""
    if remainder:
        stub = ", ".join(f"`{x}`" for x in remainder)
        supplemental = (
            f"Other DuckDB CSV views omitted from the detailed breakdown: {stub}.\n"
            "Do NOT invent columns for tables you cannot see listed above plus this line; explain "
            "uncertainty instead of hallucinating schemas."
        )
    return detailed_block, supplemental


def build_nl_sql_augmentations(
    settings: Settings,
    *,
    user_message: str,
    history_lines: list[str],
    tables: list[str],
    schema_by_table: dict[str, str],
) -> tuple[str, str]:
    """Few-shot snippets + routed schema prose for Gemini JSON planner."""
    few = retrieve_few_shot_block(
        settings,
        retrieval_blob=_clip_blob(user_message, *history_lines[-8:]),
    )
    schema_block, supplemental = narrow_schema_context(
        settings,
        user_message=user_message,
        history_lines=history_lines,
        tables=tables,
        schema_by_table=schema_by_table,
    )
    combo = schema_block + ("\n\n" + supplemental if supplemental else "")
    return few, combo


def reset_nl_sql_prompt_caches() -> None:
    """Clear embeddings + few-shot file cache (primarily for tests)."""
    load_nl_sql_examples.cache_clear()
    _example_matrix_cache.clear()
