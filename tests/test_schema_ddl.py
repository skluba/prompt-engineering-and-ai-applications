"""Tests for DDL parsing used in synthetic data generation."""

from pathlib import Path

import pytest

from app.schema_ddl import parse_ddl

ROOT = Path(__file__).resolve().parents[1]


def test_parse_ddl_sample_company_employee() -> None:
    text = (ROOT / "company_employee_schema.ddl").read_text(encoding="utf-8")
    schema = parse_ddl(text)
    assert "Employees" in schema.tables
    emp = schema.tables["Employees"]
    pk = [c.name for c in emp.columns if c.is_pk]
    assert pk == ["employee_id"]
    email = next(c for c in emp.columns if c.name == "email")
    assert email.is_unique is True


def test_ordered_table_names_respects_foreign_keys() -> None:
    ddl = """
    CREATE TABLE parent (
      id INT NOT NULL PRIMARY KEY,
      name VARCHAR(10)
    );
    CREATE TABLE child (
      id INT NOT NULL PRIMARY KEY,
      parent_id INT NOT NULL,
      FOREIGN KEY (parent_id) REFERENCES parent(id)
    );
    """
    schema = parse_ddl(ddl)
    order = schema.ordered_table_names()
    assert order.index("parent") < order.index("child")


@pytest.mark.parametrize(
    "filename",
    [
        "library_mgm_schema.ddl",
        "restrurants_schema.ddl",
    ],
)
def test_parse_sample_schemas_roundtrip(filename: str) -> None:
    path = ROOT / filename
    if not path.exists():
        pytest.skip(f"missing sample {filename}")
    schema = parse_ddl(path.read_text(encoding="utf-8"))
    assert len(schema.tables) >= 1
    for name, table in schema.tables.items():
        assert name
        assert table.columns
