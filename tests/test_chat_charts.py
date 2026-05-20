"""Chart PNG helpers."""

from __future__ import annotations

import pandas as pd

from app.chat_with_data.charts import build_chart_png


def test_build_chart_bar_png() -> None:
    df = pd.DataFrame({"region": ["A", "B"], "total": [3, 7]})
    png = build_chart_png(df, {"kind": "bar", "x": "region", "y": "total", "title": "T"})
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_chart_none_for_skip() -> None:
    df = pd.DataFrame({"a": [1]})
    assert build_chart_png(df, {"kind": "none"}) is None


def test_build_chart_empty_df() -> None:
    df = pd.DataFrame()
    assert build_chart_png(df, {"kind": "bar", "x": "a", "y": "b"}) is None


def test_build_chart_line_png() -> None:
    df = pd.DataFrame({"t": [1, 2, 3], "v": [1.0, 4.0, 2.0]})
    png = build_chart_png(df, {"kind": "line", "x": "t", "y": "v"})
    assert png is not None
    assert png[:4] == b"\x89PNG"


def test_build_chart_scatter_png() -> None:
    df = pd.DataFrame({"t": [1, 2], "v": [3, 4]})
    png = build_chart_png(df, {"kind": "scatter", "x": "t", "y": "v"})
    assert png is not None


def test_build_chart_hist_png() -> None:
    df = pd.DataFrame({"v": [1, 2, 2, 3]})
    png = build_chart_png(df, {"kind": "hist", "x": "v"})
    assert png is not None


def test_build_chart_unknown_kind() -> None:
    df = pd.DataFrame({"a": [1]})
    assert build_chart_png(df, {"kind": "boxplot", "x": "a"}) is None


def test_build_chart_bad_columns_returns_none() -> None:
    df = pd.DataFrame({"a": [1]})
    assert build_chart_png(df, {"kind": "bar", "x": "nope", "y": "a"}) is None
