"""Streamlit UI entrypoint: conversational AI shell (Gemini + Postgres + Langfuse)."""

from __future__ import annotations

import streamlit as st
from sqlalchemy.engine import Engine

from app.config import Settings, get_settings
from app.db import check_connection, get_engine
from app.llm import generate_text


@st.cache_resource
def db_engine() -> Engine:
    return get_engine(get_settings())


def _render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.subheader("Environment")
        st.write("**Project:**", settings.google_cloud_project or "—")
        st.write("**Location:**", settings.google_cloud_location)
        st.write("**Model:**", settings.gemini_model)
        db_ok = check_connection(db_engine())
        st.write("**PostgreSQL:**", "reachable" if db_ok else "not reachable")

        if not settings.vertex_configured():
            st.warning(
                "Set `GOOGLE_CLOUD_PROJECT` and authenticate with Application Default "
                "Credentials (e.g. `gcloud auth application-default login`)."
            )
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            st.info("Langfuse keys unset — tracing disabled until `LANGFUSE_*` is configured.")


def _append_assistant_reply(settings: Settings, prompt: str) -> None:
    with st.chat_message("assistant"):
        if not settings.vertex_configured():
            st.error("Vertex / GCP project is not configured.")
            return
        with st.spinner("Generating…"):
            try:
                reply = generate_text(settings, prompt)
            except Exception as err:  # noqa: BLE001 — surface LLM/network errors in UI
                st.exception(err)
                return
        if reply:
            st.markdown(reply)
            st.session_state.messages.append(("assistant", reply))


def main() -> None:
    st.set_page_config(page_title="Conversational AI", layout="wide")
    settings = get_settings()

    st.title("Conversational AI")
    st.caption(
        "Gemini 2.0 Flash (Vertex AI) · PostgreSQL · Langfuse (optional) · "
        "Extend this app for synthetic data and talk-to-your-data."
    )

    _render_sidebar(settings)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    prompt = st.chat_input("Message Gemini…")
    if not prompt:
        return

    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    _append_assistant_reply(settings, prompt)


if __name__ == "__main__":
    main()
