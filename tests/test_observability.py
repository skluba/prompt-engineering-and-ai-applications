"""Tests for optional Langfuse client factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import get_settings
from app.observability import get_langfuse


def test_get_langfuse_returns_none_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    assert get_langfuse(get_settings()) is None


def test_get_langfuse_instantiates_when_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://example.com/")
    get_settings.cache_clear()
    fake = MagicMock()
    with patch("langfuse.Langfuse", return_value=fake) as ctor:
        client = get_langfuse(get_settings())
    assert client is fake
    ctor.assert_called_once()
    call_kw = ctor.call_args.kwargs
    assert call_kw["public_key"] == "pk-test"
    assert call_kw["host"] == "https://example.com"


def test_get_langfuse_passes_release_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_RELEASE", "1.0.0")
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "ci")
    get_settings.cache_clear()
    with patch("langfuse.Langfuse", MagicMock()) as ctor:
        get_langfuse(get_settings())
    kw = ctor.call_args.kwargs
    assert kw["release"] == "1.0.0"
    assert kw["environment"] == "ci"
