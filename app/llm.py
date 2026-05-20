"""Gemini via Google Gen AI SDK with Vertex AI authentication."""

from __future__ import annotations

from typing import TYPE_CHECKING

from google import genai

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
