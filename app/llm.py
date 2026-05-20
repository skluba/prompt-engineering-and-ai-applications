"""Gemini via Google Gen AI SDK with Vertex AI authentication."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types

from app.observability import get_langfuse
from app.tracing import LangfuseTraceContext, usage_details_from_genai_response

if TYPE_CHECKING:
    from app.config import Settings


def _get_client(settings: Settings) -> genai.Client:
    if not settings.vertex_configured():
        raise ValueError(
            "Set GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION) "
            "and use Application Default Credentials or a service account."
        )
    return genai.Client(
        vertexai=settings.google_genai_use_vertexai,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )


def generate_text(
    settings: Settings,
    user_prompt: str,
    *,
    trace_context: LangfuseTraceContext | None = None,
) -> str:
    """Return Gemini (Vertex) text; emit a Langfuse trace when keys are configured."""
    client = _get_client(settings)
    lf = get_langfuse(settings)
    trace = None
    generation = None
    ctx = trace_context or LangfuseTraceContext()
    version = settings.langfuse_trace_version.strip() or None

    if lf is not None:
        meta = dict(ctx.metadata) if ctx.metadata else None
        trace = lf.trace(
            name=ctx.trace_name,
            input=user_prompt,
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            tags=list(ctx.tags),
            metadata=meta,
            version=version,
        )
        generation = trace.generation(
            name=ctx.generation_name,
            model=settings.gemini_model,
            input=user_prompt,
        )

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=user_prompt,
        )
        text = (getattr(response, "text", None) or "").strip()
        usage = usage_details_from_genai_response(response)

        if generation is not None:
            kwargs: dict = {"output": text}
            if usage:
                kwargs["usage_details"] = usage
            generation.end(**kwargs)
        if trace is not None:
            trace.update(output=text)
        return text
    except Exception as exc:
        if generation is not None:
            generation.end(level="ERROR", status_message=str(exc))
        raise
    finally:
        if lf is not None:
            lf.flush()


def _extract_json_payload(text: str) -> Any:
    """Parse JSON, stripping optional Markdown code fences without backtracking regex."""
    t = text.strip()
    if not t.startswith("```"):
        return json.loads(t)
    first_nl = t.find("\n")
    if first_nl != -1:
        t = t[first_nl + 1 :]
    t = t.rstrip()
    if t.endswith("```"):
        t = t[:-3].rstrip()
    return json.loads(t)


def _langfuse_open_json_trace(
    settings: Settings,
    ctx: LangfuseTraceContext,
    prompt_for_trace: str,
):
    """Return (lf_client, trace, generation) or (lf_client, None, None) if tracing off."""
    lf = get_langfuse(settings)
    if lf is None:
        return lf, None, None
    version = settings.langfuse_trace_version.strip() or None
    meta = dict(ctx.metadata) if ctx.metadata else None
    trace = lf.trace(
        name=ctx.trace_name,
        input=prompt_for_trace,
        session_id=ctx.session_id,
        user_id=ctx.user_id,
        tags=list(ctx.tags),
        metadata=meta,
        version=version,
    )
    generation = trace.generation(
        name=ctx.generation_name,
        model=settings.gemini_model,
        input=prompt_for_trace,
    )
    return lf, trace, generation


def _langfuse_finish_json_ok(
    lf,
    trace,
    generation,
    raw: str,
    usage: dict | None,
) -> None:
    if generation is not None:
        out = raw[:12000] if len(raw) > 12000 else raw
        kwargs: dict = {"output": out}
        if usage:
            kwargs["usage_details"] = usage
        generation.end(**kwargs)
    if trace is not None:
        trace.update(output="json-ok")
    if lf is not None:
        lf.flush()


def _langfuse_finish_json_err(lf, generation, exc: BaseException) -> None:
    if generation is not None:
        generation.end(level="ERROR", status_message=str(exc))
    if lf is not None:
        lf.flush()


def generate_json(
    settings: Settings,
    prompt: str,
    *,
    temperature: float = 0.7,
    trace_context: LangfuseTraceContext | None = None,
) -> Any:
    """Call Gemini with JSON response MIME type; return decoded dict or list."""
    client = _get_client(settings)
    ctx = trace_context or LangfuseTraceContext(
        trace_name="synthetic-data-json",
        generation_name="gemini-generate-json",
        tags=("streamlit", "synthetic-data", "vertex-gemini"),
    )
    prompt_for_trace = prompt if len(prompt) <= 8000 else f"{prompt[:8000]}\n…"
    lf, trace, generation = _langfuse_open_json_trace(settings, ctx, prompt_for_trace)

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        raw = (getattr(response, "text", None) or "").strip()
        data = _extract_json_payload(raw)
        usage = usage_details_from_genai_response(response)
        _langfuse_finish_json_ok(lf, trace, generation, raw, usage)
        return data
    except Exception as exc:
        _langfuse_finish_json_err(lf, generation, exc)
        raise
