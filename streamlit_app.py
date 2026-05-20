"""Streamlit: Phase 1 synthetic data generation and dataset browser."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy.engine import Engine

from app.config import Settings, get_settings
from app.db import check_connection, get_engine
from app.llm import generate_text
from app.schema_ddl import parse_ddl
from app.synthetic.generate import generate_full_dataset, persist_and_zip, refine_table
from app.synthetic.storage import DEFAULT_DATA_ROOT, list_datasets
from app.synthetic.validate import validate_tables_data
from app.tracing import LangfuseTraceContext


@st.cache_resource
def db_engine() -> Engine:
    return get_engine(get_settings())


def _env_sidebar(settings: Settings) -> None:
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


def _streamlit_session_id() -> str:
    if "streamlit_session_id" not in st.session_state:
        st.session_state.streamlit_session_id = str(uuid.uuid4())
    return st.session_state.streamlit_session_id


def _langfuse_chat_context(settings: Settings) -> LangfuseTraceContext:
    su = getattr(st, "user", None)
    user_id: str | None = None
    if su is not None:
        for attr in ("email", "id", "sub"):
            raw = getattr(su, attr, None)
            if raw is not None and str(raw).strip():
                user_id = str(raw).strip()
                break
    meta = {"surface": "streamlit", "model": settings.gemini_model}
    return LangfuseTraceContext(
        session_id=_streamlit_session_id(),
        user_id=user_id,
        metadata=meta,
    )


def _langfuse_synthetic_context(settings: Settings) -> LangfuseTraceContext:
    meta = {"surface": "streamlit-data-generation", "model": settings.gemini_model}
    return LangfuseTraceContext(
        session_id=_streamlit_session_id(),
        trace_name="synthetic-data-json",
        generation_name="gemini-synthetic-json",
        tags=("streamlit", "synthetic-data", "vertex-gemini"),
        metadata=meta,
    )


def _init_synthetic_session_state() -> None:
    for k, v in (
        ("syn_ddl", ""),
        ("syn_instructions", ""),
        ("syn_data", None),
        ("syn_errors", []),
        ("syn_dataset_id", None),
        ("syn_zip", None),
    ):
        if k not in st.session_state:
            st.session_state[k] = v


def _syn_handle_file_upload() -> None:
    up = st.file_uploader(
        "DDL schema file",
        type=["sql", "txt", "ddl"],
        help="Sample DDL files are in the repo root (e.g. restrurants_schema.ddl).",
    )
    if up is not None:
        st.session_state.syn_ddl = up.getvalue().decode("utf-8", errors="replace")


def _syn_run_generate_action(
    settings: Settings,
    rows_n: int,
    temperature: float,
) -> None:
    if not st.button("Generate", type="primary"):
        return
    if not settings.vertex_configured():
        st.error("Configure Vertex (GCP project) first.")
        return
    if not st.session_state.syn_ddl.strip():
        st.error("Provide DDL text or upload a file.")
        return
    with st.spinner("Generating with Gemini (JSON)…"):
        try:
            tables, errors, _schema = generate_full_dataset(
                settings,
                ddl=st.session_state.syn_ddl,
                instructions=st.session_state.syn_instructions,
                rows_per_table=int(rows_n),
                temperature=float(temperature),
                trace_context=_langfuse_synthetic_context(settings),
            )
            st.session_state.syn_data = tables
            st.session_state.syn_errors = errors
            st.session_state.syn_dataset_id = None
            st.session_state.syn_zip = None
        except Exception as err:  # noqa: BLE001
            st.exception(err)


def _syn_show_post_generation_messages() -> None:
    if st.session_state.syn_errors:
        st.warning(
            "Validation issues (you can still inspect / refine):\n- "
            + "\n- ".join(st.session_state.syn_errors)
        )
    elif st.session_state.syn_data:
        st.success("Generation finished — see previews below.")


def _syn_try_refine_table(
    settings: Settings,
    schema,
    tname: str,
    feedback: str,
    temperature: float,
) -> None:
    if not st.button("Submit", key=f"sub_{tname}"):
        return
    if not settings.vertex_configured():
        st.error("Vertex not configured.")
        return
    if not feedback.strip():
        st.warning("Enter feedback first.")
        return
    with st.spinner(f"Refining `{tname}`…"):
        try:
            new_rows = refine_table(
                settings,
                schema=schema,
                table_name=tname,
                all_tables=st.session_state.syn_data,
                user_feedback=feedback,
                temperature=float(temperature),
                trace_context=_langfuse_synthetic_context(settings),
            )
            st.session_state.syn_data[tname] = new_rows
            st.session_state.syn_errors = validate_tables_data(st.session_state.syn_data, schema)
            st.session_state.syn_zip = None
            st.rerun()
        except Exception as err:  # noqa: BLE001
            st.exception(err)


def _syn_render_table_expanders(settings: Settings, schema, temperature: float) -> None:
    for tname in schema.ordered_table_names():
        rows = st.session_state.syn_data.get(tname, [])
        with st.expander(f"Table `{tname}` ({len(rows)} rows)", expanded=False):
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No rows for this table.")
            fb_key = f"syn_refine_{tname}"
            if fb_key not in st.session_state:
                st.session_state[fb_key] = ""
            feedback = st.text_area(
                f"Changes for `{tname}`",
                key=fb_key,
                placeholder="e.g. Set all emails to use @example.com",
            )
            _syn_try_refine_table(settings, schema, tname, feedback, temperature)


def _syn_save_and_download(rows_n: int, temperature: float) -> None:
    did = st.session_state.syn_dataset_id or str(uuid.uuid4())
    if st.button("Save dataset to disk & prepare download"):
        try:
            folder, zbytes = persist_and_zip(
                dataset_id=did,
                tables=st.session_state.syn_data,
                ddl=st.session_state.syn_ddl,
                instructions=st.session_state.syn_instructions,
                rows_per_table=int(rows_n),
                temperature=float(temperature),
                data_root=DEFAULT_DATA_ROOT,
            )
            st.session_state.syn_dataset_id = did
            st.session_state.syn_zip = zbytes
            st.success(f"Saved under `{folder.resolve()}`.")
        except Exception as err:  # noqa: BLE001
            st.exception(err)

    if st.session_state.syn_zip and st.session_state.syn_dataset_id:
        st.download_button(
            label="Download ZIP (CSV + manifest)",
            data=st.session_state.syn_zip,
            file_name=f"synthetic_{st.session_state.syn_dataset_id}.zip",
            mime="application/zip",
        )


def render_data_generation(settings: Settings) -> None:
    _init_synthetic_session_state()
    st.header("Synthetic data generation")
    st.caption(
        "Upload a DDL (MySQL-style samples work well), add instructions, then generate. "
        "Refine per table with feedback; save CSVs + manifest for the Talk tab."
    )

    _syn_handle_file_upload()

    st.text_area(
        "DDL (edit or paste)",
        height=220,
        placeholder="CREATE TABLE …",
        key="syn_ddl",
    )

    st.text_area(
        "Instructions for the data",
        height=100,
        placeholder="e.g. Use realistic Polish addresses; skew toward small businesses",
        key="syn_instructions",
    )

    c1, c2 = st.columns(2)
    with c1:
        rows_n = st.number_input("Rows per table", min_value=1, max_value=50, value=10, step=1)
    with c2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    _syn_run_generate_action(settings, rows_n, temperature)
    _syn_show_post_generation_messages()

    if not st.session_state.syn_data:
        return

    schema = parse_ddl(st.session_state.syn_ddl)
    st.subheader("Table previews & refinement")
    _syn_render_table_expanders(settings, schema, temperature)
    _syn_save_and_download(rows_n, temperature)


def render_talk_to_data(settings: Settings) -> None:
    st.header("Talk to your data")
    st.info(
        "SQL + RAG over uploaded schemas is coming next. Browse saved datasets below; "
        "optional generic Gemini chat does not query CSVs yet."
    )
    root = DEFAULT_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    datasets = list_datasets(root)
    if not datasets:
        st.write(f"No datasets yet under `{root.resolve()}`. Generate and save data first.")
        return
    labels = [f"{d.get('dataset_id', '?')} — {str(d.get('created_at', ''))[:19]}" for d in datasets]
    idx = st.selectbox("Dataset", range(len(labels)), format_func=lambda i: labels[i])
    meta = datasets[idx]
    path = Path(meta["_path"])
    st.json({k: v for k, v in meta.items() if k != "_path"})
    for f in sorted(path.glob("*.csv")):
        with st.expander(f.name):
            st.dataframe(pd.read_csv(f), use_container_width=True)

    st.subheader("Quick chat (Gemini)")
    if "ttd_messages" not in st.session_state:
        st.session_state.ttd_messages = []
    for role, content in st.session_state.ttd_messages:
        with st.chat_message(role):
            st.markdown(content)
    q = st.chat_input("Ask something…")
    if q and settings.vertex_configured():
        with st.spinner("Generating answer…"):
            try:
                ans = generate_text(settings, q, trace_context=_langfuse_chat_context(settings))
                st.session_state.ttd_messages.append(("user", q))
                st.session_state.ttd_messages.append(("assistant", ans))
            except Exception as err:  # noqa: BLE001
                st.exception(err)


def main() -> None:
    st.set_page_config(page_title="Conversational AI — Data", layout="wide")
    settings = get_settings()
    st.title("Conversational AI")

    with st.sidebar:
        page = st.radio(
            "Main",
            ("Data Generation", "Talk to your data"),
            label_visibility="collapsed",
        )
        st.divider()
        _env_sidebar(settings)

    if page == "Data Generation":
        render_data_generation(settings)
    else:
        render_talk_to_data(settings)


if __name__ == "__main__":
    main()
