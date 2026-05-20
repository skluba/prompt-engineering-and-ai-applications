"""Unit tests for streamlit_app helpers (Streamlit UI mocked)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import streamlit_app
from app.config import get_settings


class _FakeSessionState:
    """Minimal st.session_state stand-in (attribute + key access)."""

    def __init__(self) -> None:
        object.__setattr__(self, "_data", {})

    @property
    def _d(self) -> dict:
        return object.__getattribute__(self, "_data")

    def __getattr__(self, name: str):
        if name == "_data":
            return object.__getattribute__(self, "_data")
        try:
            return self._d[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value) -> None:
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            self._d[name] = value

    def __getitem__(self, key: str):
        return self._d[key]

    def __setitem__(self, key: str, value) -> None:
        self._d[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def get(self, key: str, default=None):
        return self._d.get(key, default)

    def setdefault(self, key: str, default=None):
        if key not in self._d:
            self._d[key] = default
        return self._d[key]

    def update(self, other: dict) -> None:
        self._d.update(other)


@pytest.fixture
def mock_st(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    m = MagicMock()
    m.session_state = _FakeSessionState()
    monkeypatch.setattr(streamlit_app, "st", m)
    return m


def test_streamlit_session_id_stable(mock_st: MagicMock) -> None:
    s1 = streamlit_app._streamlit_session_id()
    s2 = streamlit_app._streamlit_session_id()
    assert s1 == s2
    assert "streamlit_session_id" in mock_st.session_state


def test_langfuse_nl_sql_context(mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    mock_st.user = SimpleNamespace(email="  a@b.co  ", id=None, sub=None)
    ctx = streamlit_app._langfuse_nl_sql_context(get_settings())
    assert ctx.user_id == "a@b.co"
    assert ctx.trace_name == "chat-with-data-nl-sql"


def test_langfuse_stream_context(mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    mock_st.user = SimpleNamespace(email=None, id="usr-1", sub=None)
    ctx = streamlit_app._langfuse_stream_context(get_settings())
    assert ctx.user_id == "usr-1"
    assert "stream" in ctx.trace_name


def test_langfuse_nl_sql_context_no_user(
    mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    mock_st.user = None
    ctx = streamlit_app._langfuse_nl_sql_context(get_settings())
    assert ctx.user_id is None


def test_langfuse_synthetic_context(mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    ctx = streamlit_app._langfuse_synthetic_context(get_settings())
    assert ctx.trace_name == "synthetic-data-json"
    assert "streamlit-data-generation" in (ctx.metadata or {}).get("surface", "")


def test_init_synthetic_session_state(mock_st: MagicMock) -> None:
    streamlit_app._init_synthetic_session_state()
    assert mock_st.session_state["syn_ddl"] == ""
    assert mock_st.session_state["syn_data"] is None


def test_syn_handle_file_upload(mock_st: MagicMock) -> None:
    up = MagicMock()
    up.getvalue.return_value = b"CREATE TABLE t (id INT);"
    mock_st.file_uploader.return_value = up
    streamlit_app._syn_handle_file_upload()
    assert "CREATE TABLE" in mock_st.session_state["syn_ddl"]


def test_syn_run_generate_action_no_button(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_st.button.return_value = False
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    streamlit_app._syn_run_generate_action(get_settings(), 3, 0.1)
    mock_st.error.assert_not_called()


def test_syn_run_generate_action_vertex_missing(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_st.button.return_value = True
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    get_settings.cache_clear()
    streamlit_app._syn_run_generate_action(get_settings(), 3, 0.1)
    mock_st.error.assert_called()


def test_syn_run_generate_action_empty_ddl(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_st.button.return_value = True
    mock_st.session_state["syn_ddl"] = "   "
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    streamlit_app._syn_run_generate_action(get_settings(), 3, 0.1)
    mock_st.error.assert_called()


def test_syn_show_post_generation_messages_errors(mock_st: MagicMock) -> None:
    mock_st.session_state["syn_errors"] = ["a", "b"]
    mock_st.session_state["syn_data"] = {"T": []}
    streamlit_app._syn_show_post_generation_messages()
    mock_st.warning.assert_called()


def test_syn_show_post_generation_messages_success(mock_st: MagicMock) -> None:
    mock_st.session_state["syn_errors"] = []
    mock_st.session_state["syn_data"] = {"T": [{}]}
    streamlit_app._syn_show_post_generation_messages()
    mock_st.success.assert_called()


def test_env_sidebar_reachable_db(mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gp")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    with (
        patch.object(streamlit_app, "db_engine", return_value=MagicMock()),
        patch.object(streamlit_app, "check_connection", return_value=True),
    ):
        streamlit_app._env_sidebar(get_settings())
    mock_st.subheader.assert_called()


def test_env_sidebar_vertex_warning(mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    with (
        patch.object(streamlit_app, "db_engine", return_value=MagicMock()),
        patch.object(streamlit_app, "check_connection", return_value=False),
    ):
        streamlit_app._env_sidebar(get_settings())
    mock_st.warning.assert_called()


def test_render_talk_to_data_empty(
    mock_st: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    with patch.object(streamlit_app, "DEFAULT_DATA_ROOT", tmp_path):
        with patch.object(streamlit_app, "list_datasets", return_value=[]):
            streamlit_app.render_talk_to_data(get_settings())
    mock_st.write.assert_called()


def test_render_data_generation_no_data_early_exit(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    mock_st.button.return_value = False
    mock_st.columns.return_value = (MagicMock(), MagicMock())
    mock_st.session_state.update(
        {
            "syn_ddl": "",
            "syn_instructions": "",
            "syn_data": None,
            "syn_errors": [],
            "syn_dataset_id": None,
            "syn_zip": None,
        }
    )
    streamlit_app.render_data_generation(get_settings())
    mock_st.header.assert_called()


def test_main_routes_to_talk(mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    mock_st.radio.return_value = "Talk to your data"
    with (
        patch.object(streamlit_app, "render_talk_to_data") as rt,
        patch.object(streamlit_app, "render_data_generation") as rd,
    ):
        streamlit_app.main()
    rt.assert_called_once()
    rd.assert_not_called()


def test_normalize_ttd_messages_tuple_and_dict(mock_st: MagicMock) -> None:
    raw = [("user", "u"), {"role": "assistant", "text": "a", "sql": "SELECT 1"}]
    out = streamlit_app._normalize_ttd_messages(raw)
    assert out[0]["role"] == "user"
    assert out[1]["sql"] == "SELECT 1"


def test_table_preview_records_truncates(mock_st: MagicMock) -> None:
    prev = streamlit_app._table_preview_records(pd.DataFrame({"a": [1, 2, 3]}), max_rows=2)
    assert len(prev["rows"]) == 2


TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmWQQ"
    "AAAABJRU5ErkJggg=="
)


def test_render_ttd_assistant_message_with_chart(mock_st: MagicMock) -> None:
    streamlit_app._render_ttd_assistant_message(
        {
            "text": "hi",
            "sql": "SELECT 1",
            "table_preview": {"columns": ["a"], "rows": [{"a": 1}]},
            "chart_png_b64": TINY_PNG_B64,
        }
    )
    mock_st.markdown.assert_called()
    mock_st.code.assert_called()
    mock_st.dataframe.assert_called()
    mock_st.image.assert_called()


def test_render_talk_to_data_chat_submits(
    mock_st: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.chat_with_data.turn import PreparedTurn

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    ds = tmp_path / "dset"
    ds.mkdir()
    (ds / "manifest.json").write_text('{"dataset_id":"id1","created_at":"2020-01-01T00:00:00"}')
    (ds / "t.csv").write_text("a\n1\n")
    mock_st.selectbox.return_value = 0
    mock_st.button.return_value = False
    mock_st.columns.return_value = (MagicMock(), MagicMock())
    mock_st.chat_input.return_value = "hello"
    mock_st.write_stream.return_value = "answer text"
    prep = PreparedTurn(
        sql="SELECT 1 AS c",
        df=pd.DataFrame({"c": [1]}),
        chart_png=None,
        plan_message="plan",
        stream_prompt="sp",
        chart_spec={"kind": "none"},
    )
    with (
        patch.object(streamlit_app, "DEFAULT_DATA_ROOT", tmp_path),
        patch.object(
            streamlit_app,
            "list_datasets",
            return_value=[
                {
                    "dataset_id": "id1",
                    "_path": str(ds.resolve()),
                    "created_at": "2020-01-01T00:00:00",
                }
            ],
        ),
        patch.object(streamlit_app, "prepare_turn", return_value=prep),
        patch.object(streamlit_app, "generate_text_stream", return_value=iter(["x"])),
    ):
        streamlit_app.render_talk_to_data(get_settings())
    roles = [m.get("role") for m in mock_st.session_state.ttd_messages]
    assert "user" in roles
    assert "assistant" in roles
