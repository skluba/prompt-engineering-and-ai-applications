"""Tests for GenAI → Langfuse usage mapping."""

from __future__ import annotations

from types import SimpleNamespace

from app.tracing import usage_details_from_genai_response


def test_usage_details_maps_tokens() -> None:
    resp = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
    )
    assert usage_details_from_genai_response(resp) == {"input": 10, "output": 5}


def test_usage_details_none_when_missing() -> None:
    assert usage_details_from_genai_response(SimpleNamespace(usage_metadata=None)) is None
