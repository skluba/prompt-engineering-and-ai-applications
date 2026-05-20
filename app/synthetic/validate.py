"""Validation for synthetic rows against a parsed schema."""

from __future__ import annotations

import re
from typing import Any

from app.schema_ddl import ColumnDef, ForeignKeyDef, ParsedSchema, TableDef


def _enum_allowed(sql_type: str) -> list[str] | None:
    if "ENUM" not in sql_type.upper():
        return None
    u = sql_type.upper()
    idx = u.index("ENUM")
    body = sql_type[idx + 4 :].strip()
    if not body.startswith("("):
        return None
    depth = 0
    inner_start = 0
    inner = ""
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
            if depth == 1:
                inner_start = i + 1
        elif ch == ")":
            if depth == 1:
                inner = body[inner_start:i]
                break
            depth -= 1
    parts = re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
    vals = [a or b for a, b in parts]
    return vals or None


def _append_unknown_columns(
    errors: list[str],
    tname: str,
    row_index: int,
    row: dict[str, Any],
    col_names: set[str],
) -> None:
    unk = set(row.keys()) - col_names
    if unk:
        errors.append(f"{tname} row {row_index}: unknown columns {sorted(unk)}")


def _append_not_null_and_enum_errors(
    errors: list[str],
    tname: str,
    row_index: int,
    row: dict[str, Any],
    columns: list[ColumnDef],
) -> None:
    for c in columns:
        val = row.get(c.name)
        if not c.nullable and (c.name not in row or val is None):
            errors.append(f"{tname}[{row_index}].{c.name}: violates NOT NULL")
        if val is None:
            continue
        allowed = _enum_allowed(c.sql_type)
        if allowed is not None and str(val) not in allowed:
            errors.append(f"{tname}[{row_index}].{c.name}: {val!r} not in ENUM {allowed!r}")


def _append_pk_errors(
    errors: list[str],
    tname: str,
    row_index: int,
    row: dict[str, Any],
    pk_cols: list[str],
    pk_seen: set[tuple[Any, ...]],
) -> None:
    if not pk_cols:
        return
    key = tuple(row.get(p) for p in pk_cols)
    if any(v is None for v in key):
        errors.append(f"{tname}[{row_index}]: incomplete primary key {pk_cols}")
        return
    if key in pk_seen:
        errors.append(f"{tname}: duplicate primary key {key!r}")
        return
    pk_seen.add(key)


def _validate_unique_columns(
    errors: list[str],
    tname: str,
    rows: list[dict[str, Any]],
    tdef: TableDef,
) -> None:
    for c in tdef.columns:
        if not c.is_unique:
            continue
        vals = [str(r.get(c.name)) for r in rows if c.name in r and r.get(c.name) is not None]
        if len(vals) != len(set(vals)):
            errors.append(f"{tname}: column {c.name!r} must be unique")


def _validate_single_table(
    errors: list[str],
    tname: str,
    tdef: TableDef,
    data: dict[str, list[dict[str, Any]]],
) -> None:
    rows = data.get(tname)
    if rows is None:
        errors.append(f"Missing table {tname!r} in model output.")
        return
    if len(rows) == 0:
        errors.append(f"Table {tname!r} has zero rows.")
        return

    col_names = {c.name for c in tdef.columns}
    pk_cols = tdef.pk_columns()
    pk_seen: set[tuple[Any, ...]] = set()

    for i, row in enumerate(rows):
        _append_unknown_columns(errors, tname, i, row, col_names)
        _append_not_null_and_enum_errors(errors, tname, i, row, tdef.columns)
        _append_pk_errors(errors, tname, i, row, pk_cols, pk_seen)

    _validate_unique_columns(errors, tname, rows, tdef)


def _parent_key_set(
    data: dict[str, list[dict[str, Any]]],
    parent: str,
    pref: str,
) -> set[Any]:
    return {r.get(pref) for r in data[parent] if pref in r and r.get(pref) is not None}


def _validate_fk_on_table(
    errors: list[str],
    tname: str,
    rows: list[dict[str, Any]],
    fk: ForeignKeyDef,
    parent_keys: set[Any],
) -> None:
    for i, row in enumerate(rows):
        v = row.get(fk.column)
        if v is None:
            continue
        if v not in parent_keys:
            errors.append(
                f"{tname}[{i}].{fk.column}={v!r} has no match in {fk.ref_table}.{fk.ref_column!r}"
            )


def _validate_all_foreign_keys(
    errors: list[str],
    schema: ParsedSchema,
    data: dict[str, list[dict[str, Any]]],
) -> None:
    for tname, tdef in schema.tables.items():
        rows = data.get(tname, [])
        for fk in tdef.foreign_keys:
            parent = fk.ref_table
            pref = fk.ref_column
            if parent not in schema.tables or parent not in data:
                continue
            parent_keys = _parent_key_set(data, parent, pref)
            _validate_fk_on_table(errors, tname, rows, fk, parent_keys)


def validate_tables_data(data: dict[str, list[dict[str, Any]]], schema: ParsedSchema) -> list[str]:
    """Return human-readable validation errors (empty list if acceptable)."""
    errors: list[str] = []
    for tname, tdef in schema.tables.items():
        _validate_single_table(errors, tname, tdef, data)
    _validate_all_foreign_keys(errors, schema, data)
    return errors
