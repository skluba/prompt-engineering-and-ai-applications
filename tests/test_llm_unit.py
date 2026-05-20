"""Tests for LLM helpers and JSON generation (Vertex client mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import get_settings
from app.llm import _extract_json_payload, generate_json, generate_text


class _FakeModels:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_config = None

    def generate_content(self, model: str, contents: str, config=None):
        self.last_model = model
        self.last_contents = contents
        self.last_config = config
        out = MagicMock()
        out.text = self._text
        return out


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.models = _FakeModels(text)


def test_generate_text_requires_vertex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        generate_text(get_settings(), "x")


def test_extract_json_plain_and_fenced() -> None:
    assert _extract_json_payload('{"x": 1}') == {"x": 1}
    raw = '```json\n{"a": true}\n```'
    assert _extract_json_payload(raw) == {"a": True}


def test_generate_text_returns_model_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    settings = get_settings()
    fake = _FakeClient("  hello  ")
    with patch("app.llm.genai.Client", return_value=fake):
        assert generate_text(settings, "ping") == "hello"


def test_generate_json_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    settings = get_settings()
    fake = _FakeClient('```\n{"tables": {}}\n```')
    with patch("app.llm.genai.Client", return_value=fake):
        data = generate_json(settings, "prompt", temperature=0.2)
    assert data == {"tables": {}}
    assert fake.models.last_config.temperature == pytest.approx(0.2)
    assert fake.models.last_config.response_mime_type == "application/json"


def test_generate_json_truncates_long_prompt_in_trace_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    settings = get_settings()
    long_prompt = "x" * 9000
    fake = _FakeClient("{}")
    with patch("app.llm.genai.Client", return_value=fake):
        generate_json(settings, long_prompt)
    assert len(fake.models.last_contents) == 9000


def test_generate_json_propagates_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    settings = get_settings()

    class _BoomModels:
        def generate_content(self, *_, **__):
            raise RuntimeError("vertex down")

    boom = MagicMock()
    boom.models = _BoomModels()
    with patch("app.llm.genai.Client", return_value=boom):
        with pytest.raises(RuntimeError, match="vertex"):
            generate_json(settings, "{}")


def test_generate_text_langfuse_error_on_failed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    get_settings.cache_clear()
    settings = get_settings()

    gen = MagicMock()
    trace = MagicMock()
    trace.generation.return_value = gen
    lf = MagicMock()
    lf.trace.return_value = trace

    class _Fail:
        def generate_content(self, *_, **__):
            raise ValueError("fail")

    client = MagicMock()
    client.models = _Fail()

    with (
        patch("app.llm.genai.Client", return_value=client),
        patch("app.llm.get_langfuse", return_value=lf),
    ):
        with pytest.raises(ValueError, match="fail"):
            generate_text(settings, "hi")
    gen.end.assert_called_with(level="ERROR", status_message="fail")
    lf.flush.assert_called()
