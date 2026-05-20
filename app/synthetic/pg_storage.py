"""Optional PostgreSQL storage for synthetic datasets (JSONB tables + manifest)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.synthetic.storage import list_datasets

_DATASET_DDLS = (
    """
CREATE TABLE IF NOT EXISTS app_synthetic_dataset (
    dataset_key TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    manifest JSONB NOT NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS app_synthetic_dataset_table (
    dataset_key TEXT NOT NULL REFERENCES app_synthetic_dataset(dataset_key) ON DELETE CASCADE,
    table_name TEXT NOT NULL,
    rows JSONB NOT NULL,
    PRIMARY KEY (dataset_key, table_name)
)
""",
    """
CREATE INDEX IF NOT EXISTS ix_app_synthetic_dataset_created_at
ON app_synthetic_dataset (created_at DESC)
""",
)


def ensure_synthetic_dataset_tables(engine: Engine) -> None:
    """Create app_synthetic_* tables if they do not exist."""
    with engine.begin() as conn:
        for ddl in _DATASET_DDLS:
            conn.execute(text(ddl))


def save_synthetic_dataset_to_postgres(
    engine: Engine,
    *,
    dataset_key: str,
    tables: dict[str, list[dict[str, Any]]],
    ddl: str,
    instructions: str,
    rows_per_table: int,
    temperature: float,
) -> None:
    """Replace any existing row with the same ``dataset_key`` and insert table JSONB payloads."""
    ensure_synthetic_dataset_tables(engine)
    manifest: dict[str, Any] = {
        "dataset_id": dataset_key,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "tables": list(tables.keys()),
        "row_counts": {k: len(v) for k, v in tables.items()},
        "rows_per_table_requested": rows_per_table,
        "temperature": temperature,
        "instructions": instructions,
        "ddl_excerpt": ddl[:2000] + ("…" if len(ddl) > 2000 else ""),
    }
    manifest_json = json.dumps(manifest)
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM app_synthetic_dataset WHERE dataset_key = :k"),
            {"k": dataset_key},
        )
        conn.execute(
            text(
                "INSERT INTO app_synthetic_dataset (dataset_key, manifest) "
                "VALUES (:k, CAST(:m AS jsonb))"
            ),
            {"k": dataset_key, "m": manifest_json},
        )
        for tname, rows in tables.items():
            conn.execute(
                text(
                    "INSERT INTO app_synthetic_dataset_table (dataset_key, table_name, rows) "
                    "VALUES (:k, :t, CAST(:r AS jsonb))"
                ),
                {"k": dataset_key, "t": tname, "r": json.dumps(rows)},
            )


def list_synthetic_datasets_postgres(engine: Engine) -> list[dict[str, Any]]:
    """Return manifest dicts for each stored dataset, newest first."""
    ensure_synthetic_dataset_tables(engine)
    out: list[dict[str, Any]] = []
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT dataset_key, manifest FROM app_synthetic_dataset "
                "ORDER BY created_at DESC"
            )
        )
        for row in result:
            raw_m = row.manifest
            if isinstance(raw_m, dict):
                meta = dict(raw_m)
            else:
                meta = json.loads(str(raw_m))
            meta["_dataset_key"] = row.dataset_key
            meta["_storage"] = "postgres"
            out.append(meta)
    return out


def materialize_postgres_dataset_to_folder(
    engine: Engine, dataset_key: str, dest: Path
) -> Path:
    """Write CSVs + manifest.json under ``dest`` (same layout as filesystem datasets)."""
    ensure_synthetic_dataset_tables(engine)
    dest.mkdir(parents=True, exist_ok=True)
    with engine.connect() as conn:
        mr = conn.execute(
            text("SELECT manifest FROM app_synthetic_dataset WHERE dataset_key = :k"),
            {"k": dataset_key},
        ).one_or_none()
        if mr is None:
            raise ValueError(f"No Postgres dataset {dataset_key!r}")
        manifest = mr.manifest
        manifest_dict = manifest if isinstance(manifest, dict) else json.loads(str(manifest))
        (dest / "manifest.json").write_text(
            json.dumps(manifest_dict, indent=2), encoding="utf-8"
        )
        tr = conn.execute(
            text(
                "SELECT table_name, rows FROM app_synthetic_dataset_table "
                "WHERE dataset_key = :k ORDER BY table_name"
            ),
            {"k": dataset_key},
        )
        for tname, rows in tr:
            row_list = rows if isinstance(rows, list) else json.loads(str(rows))
            pd.DataFrame(row_list).to_csv(dest / f"{tname}.csv", index=False)
    return dest


def zip_synthetic_dataset_from_postgres(engine: Engine, dataset_key: str) -> bytes:
    """Build ZIP bytes (CSV + manifest) without touching ``data/generated``."""
    buf = io.BytesIO()
    with engine.connect() as conn:
        mr = conn.execute(
            text("SELECT manifest FROM app_synthetic_dataset WHERE dataset_key = :k"),
            {"k": dataset_key},
        ).one_or_none()
        if mr is None:
            raise ValueError(f"No Postgres dataset {dataset_key!r}")
        manifest = mr.manifest
        manifest_dict = manifest if isinstance(manifest, dict) else json.loads(str(manifest))
        tr = conn.execute(
            text(
                "SELECT table_name, rows FROM app_synthetic_dataset_table "
                "WHERE dataset_key = :k ORDER BY table_name"
            ),
            {"k": dataset_key},
        )
        rows_by_table = [(tname, rows) for tname, rows in tr]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(manifest_dict, indent=2).encode("utf-8"),
        )
        for tname, rows in rows_by_table:
            row_list = rows if isinstance(rows, list) else json.loads(str(rows))
            csv_buf = io.StringIO()
            pd.DataFrame(row_list).to_csv(csv_buf, index=False)
            zf.writestr(f"{tname}.csv", csv_buf.getvalue().encode("utf-8"))
    return buf.getvalue()


def list_all_datasets(
    data_root: Path,
    engine: Engine | None,
    *,
    include_postgres: bool,
) -> list[dict[str, Any]]:
    """Filesystem datasets plus optional Postgres-backed entries (for Streamlit picker)."""
    out: list[dict[str, Any]] = []
    for m in list_datasets(data_root):
        m = dict(m)
        m["_storage"] = "filesystem"
        m["_dataset_key"] = m.get("dataset_id") or Path(m["_path"]).name
        out.append(m)
    if include_postgres and engine is not None:
        try:
            for m in list_synthetic_datasets_postgres(engine):
                out.append(dict(m))
        except Exception:
            # DB down or permissions — filesystem list still works
            pass
    out.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return out
