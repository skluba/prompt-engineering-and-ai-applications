"""Tests for optional PostgreSQL synthetic dataset storage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.synthetic.pg_storage import (
    list_all_datasets,
    list_synthetic_datasets_postgres,
    materialize_postgres_dataset_to_folder,
    save_synthetic_dataset_to_postgres,
)
from app.synthetic.storage import save_dataset


def test_list_all_datasets_filesystem_only(tmp_path: Path) -> None:
    save_dataset(
        data_root=tmp_path,
        dataset_id="d1",
        tables={"T": [{"a": 1}]},
        ddl="",
        instructions="",
        rows_per_table=1,
        temperature=0.0,
    )
    out = list_all_datasets(tmp_path, engine=None, include_postgres=False)
    assert len(out) == 1
    assert out[0]["_storage"] == "filesystem"
    assert out[0]["_dataset_key"] == "d1"


def test_list_all_datasets_includes_postgres_when_engine_provided() -> None:
    engine = MagicMock()

    def fake_list_pg(_e):
        return [{"dataset_id": "pg1", "created_at": "2021-01-01", "_dataset_key": "pg1"}]

    with patch(
        "app.synthetic.pg_storage.list_synthetic_datasets_postgres",
        side_effect=fake_list_pg,
    ):
        out = list_all_datasets(Path("/nonexistent"), engine, include_postgres=True)
    keys = {m.get("dataset_id") for m in out}
    assert "pg1" in keys


def test_save_synthetic_dataset_to_postgres_executes_delete_and_inserts() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    save_synthetic_dataset_to_postgres(
        engine,
        dataset_key="kid",
        tables={"P": [{"id": 1}]},
        ddl="CREATE TABLE p",
        instructions="hi",
        rows_per_table=3,
        temperature=0.5,
    )
    assert conn.execute.call_count >= 3


def test_materialize_postgres_dataset_to_folder_writes_csv(tmp_path: Path) -> None:
    engine = MagicMock()
    manifest = {"dataset_id": "kid", "tables": ["P"]}
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    class ManifestResult:
        def one_or_none(self):
            row = MagicMock()
            row.manifest = manifest
            return row

    class TableRows:
        def __iter__(self):
            return iter([("P", [{"id": 1, "n": "x"}])])

    def exec_side_effect(stmt, params=None):
        sql = str(stmt)
        if "FROM app_synthetic_dataset WHERE" in sql:
            return ManifestResult()
        if "FROM app_synthetic_dataset_table" in sql:
            return TableRows()
        return MagicMock()

    conn.execute.side_effect = exec_side_effect

    dest = tmp_path / "out"
    with patch("app.synthetic.pg_storage.ensure_synthetic_dataset_tables"):
        materialize_postgres_dataset_to_folder(engine, "kid", dest)
    assert (dest / "P.csv").is_file()
    assert (dest / "manifest.json").is_file()
    body = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert body["dataset_id"] == "kid"


def test_list_synthetic_datasets_postgres_parses_manifest() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    row = MagicMock()
    row.dataset_key = "k"
    row.manifest = {"dataset_id": "k", "created_at": "2020-01-01T00:00:00"}

    class Rows:
        def __iter__(self):
            return iter([row])

    conn.execute.return_value = Rows()
    with patch("app.synthetic.pg_storage.ensure_synthetic_dataset_tables"):
        out = list_synthetic_datasets_postgres(engine)
    assert len(out) == 1
    assert out[0]["_storage"] == "postgres"
    assert out[0]["_dataset_key"] == "k"
