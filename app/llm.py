"""Gemini via Google Gen AI SDK with Vertex AI authentication."""

from __future__ import annotations

from typing import TYPE_CHECKING

from google import genai

from app.observability import get_langfuse

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


def generate_text(settings: Settings, user_prompt: str) -> str:
    """Generate a short text response with Gemini (Vertex AI). Optionally trace in Langfuse."""
    client = _get_client(settings)
    lf = get_langfuse(settings)
    trace = None
    generation = None
    if lf is not None:
        trace = lf.trace(name="streamlit-chat", input={"prompt": user_prompt})
        generation = trace.generation(
            name="vertex-gemini",
            model=settings.gemini_model,
            input=user_prompt,
        )

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=user_prompt,
        )
        text = (getattr(response, "text", None) or "").strip()
        if generation is not None:
            generation.end(output=text)
        if trace is not None:
            trace.update(output=text)
        if lf is not None:
            lf.flush()
        return text
    except Exception as exc:
        if generation is not None:
            generation.end(level="ERROR", status_message=str(exc))
        if lf is not None:
            lf.flush()
        raise
