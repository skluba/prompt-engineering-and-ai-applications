"""Tests for application settings."""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings


def test_get_settings_reads_gcp_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ci-project")
    s = get_settings()
    assert s.google_cloud_project == "ci-project"
    assert s.vertex_configured() is True


def test_vertex_configured_false_when_project_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    s = get_settings()
    assert s.vertex_configured() is False


def test_settings_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    s = Settings()
    assert s.gemini_model == "gemini-2.0-flash"
