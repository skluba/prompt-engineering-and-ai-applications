"""Tests for synthetic row validation against parsed schema."""

from __future__ import annotations

from app.schema_ddl import (
    ColumnDef,
    ForeignKeyDef,
    ParsedSchema,
    TableDef,
    parse_ddl,
)
from app.synthetic.validate import validate_tables_data


def _simple_schema() -> ParsedSchema:
    ddl = """
    CREATE TABLE parent (
      id INT NOT NULL PRIMARY KEY,
      code ENUM('A', 'B') NOT NULL
    );
    CREATE TABLE child (
      id INT NOT NULL PRIMARY KEY,
      parent_id INT NOT NULL,
      tag VARCHAR(10) NOT NULL UNIQUE,
      FOREIGN KEY (parent_id) REFERENCES parent(id)
    );
    """
    return parse_ddl(ddl)


def test_validate_ok_minimal() -> None:
    schema = _simple_schema()
    data = {
        "parent": [{"id": 1, "code": "A"}],
        "child": [{"id": 10, "parent_id": 1, "tag": "x"}],
    }
    assert validate_tables_data(data, schema) == []


def test_validate_missing_table() -> None:
    schema = _simple_schema()
    data = {"parent": [{"id": 1, "code": "A"}]}
    err = validate_tables_data(data, schema)
    assert any("Missing table" in e for e in err)


def test_validate_zero_rows() -> None:
    schema = _simple_schema()
    data = {"parent": [], "child": []}
    err = validate_tables_data(data, schema)
    assert any("zero rows" in e for e in err)


def test_validate_not_null() -> None:
    schema = _simple_schema()
    data = {
        "parent": [{"id": 1, "code": "A"}],
        "child": [{"id": 10, "parent_id": 1}],
    }
    err = validate_tables_data(data, schema)
    assert any("NOT NULL" in e for e in err)


def test_validate_unknown_column() -> None:
    schema = _simple_schema()
    data = {
        "parent": [{"id": 1, "code": "A"}],
        "child": [{"id": 10, "parent_id": 1, "tag": "z", "oops": 1}],
    }
    err = validate_tables_data(data, schema)
    assert any("unknown columns" in e for e in err)


def test_validate_enum() -> None:
    schema = _simple_schema()
    data = {
        "parent": [{"id": 1, "code": "Z"}],
        "child": [{"id": 10, "parent_id": 1, "tag": "a"}],
    }
    err = validate_tables_data(data, schema)
    assert any("ENUM" in e for e in err)


def test_validate_duplicate_pk() -> None:
    schema = _simple_schema()
    data = {
        "parent": [{"id": 1, "code": "A"}, {"id": 1, "code": "B"}],
        "child": [{"id": 10, "parent_id": 1, "tag": "a"}],
    }
    err = validate_tables_data(data, schema)
    assert any("duplicate primary key" in e for e in err)


def test_validate_incomplete_pk() -> None:
    p = ParsedSchema()
    p.tables["t"] = TableDef(
        name="t",
        columns=[
            ColumnDef(name="a", sql_type="INT", nullable=False, is_pk=True),
            ColumnDef(name="b", sql_type="INT", nullable=False, is_pk=True),
        ],
    )
    data = {"t": [{"a": 1, "b": None}]}
    err = validate_tables_data(data, p)
    assert any("incomplete primary key" in e for e in err)


def test_validate_unique_column() -> None:
    schema = _simple_schema()
    data = {
        "parent": [{"id": 1, "code": "A"}],
        "child": [
            {"id": 10, "parent_id": 1, "tag": "same"},
            {"id": 11, "parent_id": 1, "tag": "same"},
        ],
    }
    err = validate_tables_data(data, schema)
    assert any("must be unique" in e for e in err)


def test_validate_fk_missing_parent() -> None:
    schema = _simple_schema()
    data = {
        "parent": [{"id": 1, "code": "A"}],
        "child": [{"id": 10, "parent_id": 99, "tag": "a"}],
    }
    err = validate_tables_data(data, schema)
    assert any("has no match" in e for e in err)


def test_validate_fk_skips_when_parent_table_missing_in_data() -> None:
    p = ParsedSchema()
    p.tables["c"] = TableDef(
        name="c",
        columns=[ColumnDef(name="id", sql_type="INT", nullable=False, is_pk=True)],
        foreign_keys=[
            ForeignKeyDef(table="c", column="p", ref_table="ghost", ref_column="id"),
        ],
    )
    data = {"c": [{"id": 1}]}
    assert validate_tables_data(data, p) == []


def test_validate_null_fk_skipped() -> None:
    ddl = """
    CREATE TABLE parent (id INT PRIMARY KEY);
    CREATE TABLE child (
      id INT PRIMARY KEY,
      parent_id INT,
      FOREIGN KEY (parent_id) REFERENCES parent(id)
    );
    """
    schema = parse_ddl(ddl)
    data = {
        "parent": [{"id": 1}],
        "child": [{"id": 10, "parent_id": None}],
    }
    err = validate_tables_data(data, schema)
    fk_msgs = [e for e in err if "has no match" in e]
    assert fk_msgs == []
