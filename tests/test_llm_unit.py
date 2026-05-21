"""Tests for LLM helpers and JSON generation (Vertex client mocked)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.config import get_settings
from app.llm import (
    _extract_json_payload,
    embed_texts,
    generate_json,
    generate_text,
    generate_text_stream,
)
from app.tracing import LangfuseTraceContext


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


def test_generate_text_stream_yields_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    settings = get_settings()

    class _Chunk:
        def __init__(self, t: str) -> None:
            self.text = t

    class _StreamModels:
        def generate_content_stream(self, **_kwargs):
            yield _Chunk("hel")
            yield _Chunk("lo")

    fake = MagicMock()
    fake.models = _StreamModels()
    with patch("app.llm.genai.Client", return_value=fake):
        parts = list(generate_text_stream(settings, "hi"))
    assert parts == ["hel", "lo"]


def test_generate_text_stream_langfuse_success(monkeypatch: pytest.MonkeyPatch) -> None:
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

    class _Chunk:
        text = "x"

    class _StreamModels:
        def generate_content_stream(self, **_kwargs):
            yield _Chunk()

    client = MagicMock()
    client.models = _StreamModels()
    with (
        patch("app.llm.genai.Client", return_value=client),
        patch("app.llm.get_langfuse", return_value=lf),
        patch("app.llm.usage_details_from_genai_response", return_value={"input": 1}),
    ):
        parts = list(generate_text_stream(settings, "stream this"))
    assert parts == ["x"]
    gen.end.assert_called_once()
    lf.flush.assert_called()


def test_generate_text_stream_langfuse_error(monkeypatch: pytest.MonkeyPatch) -> None:
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

    class _Boom:
        def generate_content_stream(self, **_kwargs):
            raise RuntimeError("stream fail")

    client = MagicMock()
    client.models = _Boom()
    with (
        patch("app.llm.genai.Client", return_value=client),
        patch("app.llm.get_langfuse", return_value=lf),
    ):
        with pytest.raises(RuntimeError, match="stream fail"):
            list(generate_text_stream(settings, "x"))
    gen.end.assert_called_with(level="ERROR", status_message="stream fail")
    lf.flush.assert_called()


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


def test_extract_json_fenced_no_extra_blank_line() -> None:
    """Opening fence must be followed by newline (same as typical Markdown)."""
    assert _extract_json_payload('```\n{"k": 2}\n```') == {"k": 2}


def test_generate_text_langfuse_success_with_usage(
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

    fake = _FakeClient("  out  ")
    with (
        patch("app.llm.genai.Client", return_value=fake),
        patch("app.llm.get_langfuse", return_value=lf),
        patch("app.llm.usage_details_from_genai_response", return_value={"input_tokens": 3}),
    ):
        assert generate_text(settings, "hi") == "out"
    gen.end.assert_called_once()
    assert gen.end.call_args.kwargs["usage_details"] == {"input_tokens": 3}
    trace.update.assert_called_once_with(output="out")
    lf.flush.assert_called()


def test_generate_text_langfuse_success_no_usage_details(
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

    fake = _FakeClient("x")
    with (
        patch("app.llm.genai.Client", return_value=fake),
        patch("app.llm.get_langfuse", return_value=lf),
        patch("app.llm.usage_details_from_genai_response", return_value=None),
    ):
        generate_text(settings, "hi")
    gen.end.assert_called_once_with(output="x")


def test_generate_json_langfuse_success_long_raw_truncates_in_trace(
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

    blob = json.dumps({"a": "x" * 13000})
    fake = _FakeClient(blob)
    with (
        patch("app.llm.genai.Client", return_value=fake),
        patch("app.llm.get_langfuse", return_value=lf),
        patch("app.llm.usage_details_from_genai_response", return_value={"total": 1}),
    ):
        generate_json(settings, "{}", temperature=0.5)
    out_kw = gen.end.call_args.kwargs
    assert len(out_kw["output"]) == 12000
    trace.update.assert_called_with(output="json-ok")
    lf.flush.assert_called()


def test_generate_json_langfuse_open_trace_metadata_and_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_TRACE_VERSION", "  v99  ")
    get_settings.cache_clear()
    settings = get_settings()

    gen = MagicMock()
    trace = MagicMock()
    trace.generation.return_value = gen
    lf = MagicMock()
    lf.trace.return_value = trace

    fake = _FakeClient("{}")
    ctx = LangfuseTraceContext(
        metadata={"k": "v"},
    )
    with (
        patch("app.llm.genai.Client", return_value=fake),
        patch("app.llm.get_langfuse", return_value=lf),
        patch("app.llm.usage_details_from_genai_response", return_value={}),
    ):
        generate_json(settings, "prompt", trace_context=ctx)
    lf.trace.assert_called_once()
    call_kw = lf.trace.call_args.kwargs
    assert call_kw["metadata"] == {"k": "v"}
    assert call_kw["version"] == "v99"


def test_generate_json_langfuse_error_calls_finish(
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

    boom = MagicMock()

    class _Boom:
        def generate_content(self, *_, **__):
            raise OSError("api")

    boom.models = _Boom()
    with (
        patch("app.llm.genai.Client", return_value=boom),
        patch("app.llm.get_langfuse", return_value=lf),
    ):
        with pytest.raises(OSError, match="api"):
            generate_json(settings, "p")
    gen.end.assert_called_with(level="ERROR", status_message="api")
    lf.flush.assert_called()


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


def test_embed_texts_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    assert embed_texts(get_settings(), []) == []


def test_embed_texts_maps_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    settings = get_settings()

    class _Vec:
        def __init__(self, values):
            self.values = values

    class _EmbResp:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    models = MagicMock()
    models.embed_content.return_value = _EmbResp(
        [_Vec([0.5, 0.5]), _Vec(None), _Vec([])]  # None branch -> [], empty vals
    )
    client = MagicMock()
    client.models = models
    with patch("app.llm.genai.Client", return_value=client):
        out = embed_texts(settings, ["a", "b", "c"])
    assert out == [[0.5, 0.5], [], []]


def test_embed_texts_none_embeddings_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    settings = get_settings()

    class _EmbResp:
        embeddings = None

    models = MagicMock()
    models.embed_content.return_value = _EmbResp()
    client = MagicMock()
    client.models = models
    with patch("app.llm.genai.Client", return_value=client):
        assert embed_texts(settings, ["x"]) == []
