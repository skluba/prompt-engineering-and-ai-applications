"""Pytest fixtures."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Isolate tests that call cached get_settings()."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
