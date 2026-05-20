"""Tests for synthetic dataset persistence."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pytest

from app.synthetic.storage import (
    DEFAULT_DATA_ROOT,
    dataset_dir,
    list_datasets,
    save_dataset,
    zip_dataset,
)


def test_dataset_dir_joins_under_root(tmp_path) -> None:
    d = dataset_dir(tmp_path, "my-id")
    assert d == tmp_path / "my-id"


def test_save_dataset_writes_csv_and_manifest(tmp_path) -> None:
    tables = {"T": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]}
    long_ddl = "Z" * 2500
    out = save_dataset(
        data_root=tmp_path,
        dataset_id="ds1",
        tables=tables,
        ddl=long_ddl,
        instructions="go",
        rows_per_table=10,
        temperature=0.5,
    )
    assert out.is_dir()
    csv = (out / "T.csv").read_text(encoding="utf-8")
    assert "a,b" in csv or "b,a" in csv
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "ds1"
    assert manifest["tables"] == ["T"]
    assert manifest["row_counts"] == {"T": 2}
    assert manifest["temperature"] == pytest.approx(0.5)
    assert manifest["instructions"] == "go"
    assert manifest["ddl_excerpt"].endswith("…")
    assert len(manifest["ddl_excerpt"]) <= 2002


def test_zip_dataset_includes_csv_and_json(tmp_path) -> None:
    save_dataset(
        data_root=tmp_path,
        dataset_id="z1",
        tables={"A": [{"id": 1}]},
        ddl="CREATE TABLE A (id INT);",
        instructions="",
        rows_per_table=1,
        temperature=0.0,
    )
    folder = tmp_path / "z1"
    blob = zip_dataset(folder)
    zf = zipfile.ZipFile(BytesIO(blob))
    names = zf.namelist()
    assert "A.csv" in names
    assert "manifest.json" in names


def test_list_datasets_sorted_newest_first(tmp_path) -> None:
    save_dataset(
        data_root=tmp_path,
        dataset_id="old",
        tables={"T": [{"x": 1}]},
        ddl="",
        instructions="",
        rows_per_table=1,
        temperature=0.0,
    )
    save_dataset(
        data_root=tmp_path,
        dataset_id="new",
        tables={"T": [{"x": 2}]},
        ddl="",
        instructions="",
        rows_per_table=1,
        temperature=0.0,
    )
    lst = list_datasets(tmp_path)
    assert len(lst) == 2
    assert lst[0]["dataset_id"] in ("new", "old")
    assert "_path" in lst[0]


def test_list_datasets_empty_root(tmp_path) -> None:
    assert list_datasets(tmp_path / "nope") == []


def test_list_datasets_skips_bad_manifest(tmp_path) -> None:
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "manifest.json").write_text("not json{", encoding="utf-8")
    out = list_datasets(tmp_path)
    assert len(out) == 1
    assert out[0]["dataset_id"] == "broken"


def test_default_data_root_is_under_repo() -> None:
    assert DEFAULT_DATA_ROOT.parts[-2:] == ("data", "generated")
