"""Seaborn/Matplotlib charts for tabular results."""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def _hue(df: pd.DataFrame, hue: Any) -> str | None:
    return hue if isinstance(hue, str) and hue in df.columns else None


def _xy_valid(df: pd.DataFrame, x: Any, y: Any) -> bool:
    return bool(x) and y is not None and x in df.columns and y in df.columns


def _plot_xy(
    ax: Axes,
    df: pd.DataFrame,
    kind: str,
    x: Any,
    y: Any,
    hue: Any,
) -> bool:
    if not _xy_valid(df, x, y):
        return False
    h = _hue(df, hue)
    if kind == "bar":
        sns.barplot(data=df, x=x, y=y, hue=h, ax=ax)
    elif kind == "line":
        sns.lineplot(data=df, x=x, y=y, hue=h, ax=ax)
    elif kind == "scatter":
        sns.scatterplot(data=df, x=x, y=y, hue=h, ax=ax)
    else:
        return False
    return True


def _plot_hist(ax: Axes, df: pd.DataFrame, x: Any, y: Any) -> bool:
    col = x or y
    if not col or col not in df.columns:
        return False
    sns.histplot(data=df, x=col, ax=ax, kde=False)
    return True


def _finalize_figure(fig: plt.Figure, ax: Axes, title: str) -> bytes:
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_chart_png(df: pd.DataFrame, spec: dict[str, Any] | None) -> bytes | None:
    """Build a PNG from a chart spec; return None if chart skipped or unusable."""
    if not spec or not isinstance(spec, dict):
        return None
    kind = (spec.get("kind") or "none").strip().lower()
    if kind in ("", "none", "skip") or df.empty:
        return None

    x, y = spec.get("x"), spec.get("y")
    title = str(spec.get("title") or "")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        ok = False
        if kind in ("bar", "line", "scatter"):
            ok = _plot_xy(ax, df, kind, x, y, spec.get("hue"))
        elif kind == "hist":
            ok = _plot_hist(ax, df, x, y)
        if not ok:
            plt.close(fig)
            return None
        return _finalize_figure(fig, ax, title)
    except Exception:
        plt.close(fig)
        return None
