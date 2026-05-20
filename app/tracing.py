"""Langfuse trace context and GenAI usage mapping.

Follows instrumentation guidance in `.cursor/skills/langfuse/references/instrumentation.md`:
session grouping, optional user id, descriptive names, token usage, minimal trace input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LangfuseTraceContext:
    """Per-request attributes for Langfuse (sessions, users, tags, metadata)."""

    session_id: str | None = None
    user_id: str | None = None
    tags: tuple[str, ...] = ("streamlit-chat", "vertex-gemini")
    trace_name: str = "vertex-gemini-chat"
    generation_name: str = "gemini-generate-content"
    metadata: Mapping[str, str] | None = None


def usage_details_from_genai_response(response: Any) -> dict[str, int] | None:
    """Map ``google.genai`` response ``usage_metadata`` to Langfuse ``usage_details``."""
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return None
    out: dict[str, int] = {}
    pt = getattr(um, "prompt_token_count", None)
    ct = getattr(um, "candidates_token_count", None)
    if pt is not None:
        out["input"] = int(pt)
    if ct is not None:
        out["output"] = int(ct)
    return out or None
