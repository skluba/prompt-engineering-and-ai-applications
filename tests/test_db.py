"""Database helper tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from sqlalchemy.engine import Engine

from app.db import check_connection


def test_check_connection_success() -> None:
    engine = MagicMock(spec=Engine)
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    assert check_connection(engine) is True
    conn.execute.assert_called_once()


def test_check_connection_failure() -> None:
    engine = MagicMock(spec=Engine)
    engine.connect.side_effect = OSError("no db")
    assert check_connection(engine) is False
