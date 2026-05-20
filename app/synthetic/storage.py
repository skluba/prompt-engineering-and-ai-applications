"""Persist generated datasets as CSV + manifest (and optional ZIP)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DATA_ROOT = Path("data") / "generated"


def dataset_dir(data_root: Path, dataset_id: str) -> Path:
    return data_root / dataset_id


def save_dataset(
    *,
    data_root: Path,
    dataset_id: str,
    tables: dict[str, list[dict[str, Any]]],
    ddl: str,
    instructions: str,
    rows_per_table: int,
    temperature: float,
) -> Path:
    d = dataset_dir(data_root, dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(d / f"{name}.csv", index=False)
    manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "tables": list(tables.keys()),
        "row_counts": {k: len(v) for k, v in tables.items()},
        "rows_per_table_requested": rows_per_table,
        "temperature": temperature,
        "instructions": instructions,
        "ddl_excerpt": ddl[:2000] + ("…" if len(ddl) > 2000 else ""),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return d


def zip_dataset(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.iterdir()):
            if p.suffix in (".csv", ".json"):
                zf.write(p, arcname=p.name)
    return buf.getvalue()


def list_datasets(data_root: Path) -> list[dict[str, Any]]:
    if not data_root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for sub in sorted(data_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not sub.is_dir():
            continue
        mf = sub / "manifest.json"
        if mf.is_file():
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {"dataset_id": sub.name}
        else:
            meta = {"dataset_id": sub.name}
        meta["_path"] = str(sub.resolve())
        out.append(meta)
    return out
