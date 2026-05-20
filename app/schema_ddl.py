"""Lightweight DDL parsing for CREATE TABLE / ALTER TABLE ADD FK (MySQL-ish samples)."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


def _strip_block_comments(sql: str) -> str:
    """Remove /* */ blocks with a linear scan (avoids catastrophic backtracking)."""
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        if i + 1 < n and sql[i] == "/" and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            if end == -1:
                break
            i = end + 2
            continue
        out.append(sql[i])
        i += 1
    return "".join(out)


def _strip_sql_comments(sql: str) -> str:
    sql = _strip_block_comments(sql)
    lines = []
    for line in sql.splitlines():
        if "--" in line:
            line = line.split("--", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _split_statements_handle_in_string(
    sql: str,
    i: int,
    buf: list[str],
    in_str: str,
) -> tuple[int, str | None]:
    c = sql[i]
    buf.append(c)
    if c == in_str and (i == 0 or sql[i - 1] != "\\"):
        return i + 1, None
    return i + 1, in_str


def _split_statements_handle_normal(
    sql: str,
    i: int,
    buf: list[str],
    depth: int,
) -> tuple[int, int, list[str] | None]:
    c = sql[i]
    if c == "(":
        buf.append(c)
        return i + 1, depth + 1, None
    if c == ")":
        buf.append(c)
        return i + 1, max(0, depth - 1), None
    if c == ";" and depth == 0:
        stmt = "".join(buf).strip()
        parts: list[str] = [] if not stmt else [stmt]
        return i + 1, depth, parts
    buf.append(c)
    return i + 1, depth, None


def _split_statements(sql: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_str: str | None = None
    i = 0
    while i < len(sql):
        if in_str:
            i, in_str = _split_statements_handle_in_string(sql, i, buf, in_str)
            continue
        c = sql[i]
        if c in ("'", '"'):
            buf.append(c)
            in_str = c
            i += 1
            continue
        i, depth, new_parts = _split_statements_handle_normal(sql, i, buf, depth)
        if new_parts is not None:
            parts.extend(new_parts)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _split_top_level_commas(blob: str) -> list[str]:
    blob = blob.strip()
    parts: list[str] = []
    start = 0
    depth = 0
    in_str: str | None = None
    for i, c in enumerate(blob):
        if in_str:
            if c == in_str and (i == 0 or blob[i - 1] != "\\"):
                in_str = None
            continue
        if c in ("'", '"'):
            in_str = c
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == "," and depth == 0:
            parts.append(blob[start:i].strip())
            start = i + 1
    parts.append(blob[start:].strip())
    return [p for p in parts if p]


@dataclass
class ColumnDef:
    name: str
    sql_type: str
    nullable: bool = True
    is_pk: bool = False
    is_unique: bool = False


@dataclass
class ForeignKeyDef:
    table: str
    column: str
    ref_table: str
    ref_column: str


@dataclass
class TableDef:
    name: str
    columns: list[ColumnDef] = field(default_factory=list)
    foreign_keys: list[ForeignKeyDef] = field(default_factory=list)

    def pk_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_pk]


def _fk_dependency_sets(tables: dict[str, TableDef], names: list[str]) -> dict[str, set[str]]:
    deps: dict[str, set[str]] = {n: set() for n in names}
    for t in names:
        for fk in tables[t].foreign_keys:
            if fk.ref_table in tables and fk.ref_table != t:
                deps[t].add(fk.ref_table)
    return deps


def _kahn_topological_order(names: list[str], deps: dict[str, set[str]]) -> list[str]:
    in_deg = {n: len(deps[n]) for n in names}
    q = deque([n for n in names if in_deg[n] == 0])
    out: list[str] = []
    while q:
        n = q.popleft()
        out.append(n)
        for m in names:
            if n in deps[m]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    q.append(m)
    if len(out) != len(names):
        rest = [x for x in names if x not in out]
        out.extend(rest)
    return out


@dataclass
class ParsedSchema:
    tables: dict[str, TableDef] = field(default_factory=dict)
    ddl_source: str = ""

    def ordered_table_names(self) -> list[str]:
        """Dependency order: parents before children.

        Unknown cycles fall back to insertion order.
        """
        names = list(self.tables.keys())
        deps = _fk_dependency_sets(self.tables, names)
        return _kahn_topological_order(names, deps)

    def summary_for_prompt(self) -> str:
        lines: list[str] = []
        for name in self.ordered_table_names():
            t = self.tables[name]
            cols = ", ".join(
                f"{c.name} {c.sql_type}{' PK' if c.is_pk else ''}"
                f"{' NOT NULL' if not c.nullable else ''}{' UNIQUE' if c.is_unique else ''}"
                for c in t.columns
            )
            fks = "; ".join(
                f"FK {fk.column} -> {fk.ref_table}.{fk.ref_column}" for fk in t.foreign_keys
            )
            lines.append(f"TABLE {name}: {cols}" + (f" | {fks}" if fks else ""))
        return "\n".join(lines)


_re_create_prefix = re.compile(
    r"^CREATE\s+TABLE\s+`?(?P<name>\w+)`?\s*\(",
    re.IGNORECASE,
)


def _ctb_step_inside_string(stmt: str, i: int, in_str: str) -> tuple[int, str | None]:
    c = stmt[i]
    if c == "\\" and i + 1 < len(stmt):
        return i + 2, in_str
    if c == in_str and (i == 0 or stmt[i - 1] != "\\"):
        return i + 1, None
    return i + 1, in_str


def _ctb_paren_only_step(
    stmt: str,
    i: int,
    depth: int,
    start_inner: int,
) -> tuple[int, int, int, str | None, bool]:
    """Paren/brace step: returns (i, depth, start_inner, body_or_none, invalid)."""
    c = stmt[i]
    if c == "(":
        new_depth = depth + 1
        new_start = i + 1 if new_depth == 1 else start_inner
        return i + 1, new_depth, new_start, None, False
    if c == ")":
        new_depth = depth - 1
        if new_depth == 0:
            body = stmt[start_inner:i].strip()
            return i + 1, new_depth, start_inner, body, False
        if new_depth < 0:
            return i + 1, new_depth, start_inner, None, True
        return i + 1, new_depth, start_inner, None, False
    return i + 1, depth, start_inner, None, False


def _extract_create_table_body(stmt: str, open_paren_idx: int) -> str | None:
    """Return text inside the outer CREATE TABLE ( … ) using balanced-paren scan."""
    if open_paren_idx >= len(stmt) or stmt[open_paren_idx] != "(":
        return None
    depth = 0
    i = open_paren_idx
    in_str: str | None = None
    start_inner = open_paren_idx + 1
    while i < len(stmt):
        if in_str:
            i, in_str = _ctb_step_inside_string(stmt, i, in_str)
            continue
        c = stmt[i]
        if c in ("'", '"'):
            in_str = c
            i += 1
            continue
        i, depth, start_inner, body, invalid = _ctb_paren_only_step(stmt, i, depth, start_inner)
        if invalid:
            return None
        if body is not None:
            return body
    return None


def _parse_enum_sql_type(r: str) -> tuple[str, str] | None:
    if not r.upper().startswith("ENUM"):
        return None
    depth = 0
    for i, ch in enumerate(r):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return r[: i + 1].strip(), r[i + 1 :].strip()
    return r, ""


def _parse_non_enum_sql_type(r: str) -> tuple[str, str]:
    depth = 0
    end = 0
    while end < len(r):
        ch = r[end]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch.isspace() and depth == 0:
            break
        end += 1
    typ = r[:end].strip()
    return typ, r[end:].strip()


def _parse_sql_type(rest: str) -> tuple[str, str]:
    """Return (sql_type_string, remainder after type for flags / REFERENCES)."""
    r = rest.strip()
    enum = _parse_enum_sql_type(r)
    if enum is not None:
        return enum
    return _parse_non_enum_sql_type(r)


def _column_name_and_rest(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s:
        return None
    if s[0] == "`":
        end = s.find("`", 1)
        if end == -1:
            return None
        name = s[1:end]
        rest = s[end + 1 :].lstrip()
        return name, rest
    parts = s.split(None, 1)
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _try_parse_standalone_foreign_key_line(
    line: str,
    add_fk: Callable[[ForeignKeyDef], None],
) -> bool:
    fk_m = re.search(
        r"^FOREIGN\s+KEY\s*\((?P<col>\w+)\)\s*REFERENCES\s+`?(?P<ref_t>\w+)`?\s*\((?P<ref_c>\w+)\)",
        line,
        re.IGNORECASE,
    )
    if not fk_m:
        return False
    add_fk(
        ForeignKeyDef(
            table="",
            column=fk_m.group("col"),
            ref_table=fk_m.group("ref_t"),
            ref_column=fk_m.group("ref_c"),
        )
    )
    return True


_re_inline_ref = re.compile(
    r"REFERENCES\s+`?(?P<ref_t>\w+)`?\s*\((?P<ref_c>\w+)\)",
    re.IGNORECASE,
)


def _apply_inline_references(name: str, tail: str, add_fk: Callable[[ForeignKeyDef], None]) -> None:
    fk_inline = _re_inline_ref.search(tail)
    if fk_inline:
        add_fk(
            ForeignKeyDef(
                table="",
                column=name,
                ref_table=fk_inline.group("ref_t"),
                ref_column=fk_inline.group("ref_c"),
            )
        )


def _build_column_def_from_rest(
    name: str,
    rest: str,
    add_fk: Callable[[ForeignKeyDef], None],
) -> ColumnDef:
    if "--" in rest:
        rest = rest.split("--", 1)[0].strip()
    sql_type, tail = _parse_sql_type(rest)
    tail_u = tail.upper()
    is_pk = "PRIMARY KEY" in tail_u
    nullable = "NOT NULL" not in tail_u
    is_unique = "UNIQUE" in tail_u
    _apply_inline_references(name, tail, add_fk)
    return ColumnDef(
        name=name,
        sql_type=sql_type,
        nullable=nullable,
        is_pk=is_pk,
        is_unique=is_unique,
    )


def _parse_column_line(line: str, add_fk: Callable[[ForeignKeyDef], None]) -> ColumnDef | None:
    line = line.strip()
    if not line or line.upper().startswith("PRIMARY KEY") or line.upper().startswith("KEY "):
        return None
    if line.upper().startswith("UNIQUE"):
        return None
    if _try_parse_standalone_foreign_key_line(line, add_fk):
        return None
    parsed = _column_name_and_rest(line)
    if parsed is None:
        return None
    name, rest = parsed
    return _build_column_def_from_rest(name, rest, add_fk)


def _parse_create_table(stmt: str) -> TableDef | None:
    stmt = stmt.strip()
    m = _re_create_prefix.match(stmt)
    if not m:
        return None
    name = m.group("name")
    open_idx = m.end() - 1
    body = _extract_create_table_body(stmt, open_idx)
    if body is None:
        return None
    cols_lines = _split_top_level_commas(body)
    columns: list[ColumnDef] = []
    fks: list[ForeignKeyDef] = []

    def add_fk(fk: ForeignKeyDef) -> None:
        fk = ForeignKeyDef(
            table=name,
            column=fk.column,
            ref_table=fk.ref_table,
            ref_column=fk.ref_column,
        )
        fks.append(fk)

    for line in cols_lines:
        cd = _parse_column_line(line, add_fk)
        if cd:
            columns.append(cd)
    return TableDef(name=name, columns=columns, foreign_keys=fks)


_re_alter_fk = re.compile(
    r"^ALTER\s+TABLE\s+`?(?P<t>\w+)`?\s+ADD\s+CONSTRAINT\s+\w+\s+FOREIGN\s+KEY\s*\((?P<col>\w+)\)\s*REFERENCES\s+`?(?P<ref_t>\w+)`?\s*\((?P<ref_c>\w+)\)",
    re.IGNORECASE,
)


def parse_ddl(ddl: str) -> ParsedSchema:
    clean = _strip_sql_comments(ddl)
    schema = ParsedSchema(ddl_source=ddl.strip())
    for stmt in _split_statements(clean):
        stmt = stmt.strip()
        if not stmt:
            continue
        ct = _parse_create_table(stmt)
        if ct:
            schema.tables[ct.name] = ct
            continue
        am = _re_alter_fk.match(stmt)
        if am:
            tname = am.group("t")
            if tname not in schema.tables:
                continue
            schema.tables[tname].foreign_keys.append(
                ForeignKeyDef(
                    table=tname,
                    column=am.group("col"),
                    ref_table=am.group("ref_t"),
                    ref_column=am.group("ref_c"),
                )
            )
    return schema
