"""Unit tests for streamlit_app helpers (Streamlit UI mocked)."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import streamlit_app
from app.chat_with_data.turn import PreparedTurn
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

    def pop(self, key: str, default=None):
        return self._d.pop(key, default)

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


def test_dataset_csv_stems(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("y\n2\n", encoding="utf-8")
    (tmp_path / "note.txt").write_text("n", encoding="utf-8")
    assert streamlit_app._dataset_csv_stems(tmp_path) == ["a", "b"]


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


def test_langfuse_nl_sql_context_uses_sub_when_email_blank(
    mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    mock_st.user = SimpleNamespace(email="   ", id=None, sub="sub-u")
    assert streamlit_app._langfuse_nl_sql_context(get_settings()).user_id == "sub-u"


def test_langfuse_stream_context_uses_id_when_present(
    mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    mock_st.user = SimpleNamespace(email=None, id="  usr  ", sub="sub")
    assert streamlit_app._langfuse_stream_context(get_settings()).user_id == "usr"


def test_env_sidebar_no_warnings_when_vertex_and_langfuse_configured(
    mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gp")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    get_settings.cache_clear()
    with (
        patch.object(streamlit_app, "db_engine", return_value=MagicMock()),
        patch.object(streamlit_app, "check_connection", return_value=True),
    ):
        streamlit_app._env_sidebar(get_settings())
    mock_st.warning.assert_not_called()
    mock_st.info.assert_not_called()


def test_syn_handle_file_upload_no_selection(mock_st: MagicMock) -> None:
    mock_st.file_uploader.return_value = None
    mock_st.session_state["syn_ddl"] = "unchanged sentinel"
    streamlit_app._syn_handle_file_upload()
    assert mock_st.session_state["syn_ddl"] == "unchanged sentinel"


def test_syn_run_generate_success_stores_tables(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    streamlit_app._init_synthetic_session_state()
    mock_st.button.return_value = True
    mock_st.session_state["syn_ddl"] = "CREATE TABLE t (id INT);"
    fake_out = ({"t": [{}]}, [], MagicMock())
    with patch.object(streamlit_app, "generate_full_dataset", return_value=fake_out):
        streamlit_app._syn_run_generate_action(get_settings(), 5, 0.2)
    assert mock_st.session_state.syn_data == {"t": [{}]}
    mock_st.exception.assert_not_called()


def test_normalize_ttd_messages_empty() -> None:
    assert streamlit_app._normalize_ttd_messages([]) == []


def test_table_preview_records_empty_but_with_columns() -> None:
    df = pd.DataFrame({"col_a": pd.Series(dtype="int64")})
    assert df.empty
    prev = streamlit_app._table_preview_records(df)
    assert prev == {"columns": ["col_a"], "rows": []}


def test_ttd_chart_png_b64_is_none_when_no_png() -> None:
    prepared = PreparedTurn(
        sql="SELECT 1",
        df=pd.DataFrame({"c": [1]}),
        chart_png=None,
        plan_message="p",
        stream_prompt="sp",
        chart_spec={"kind": "none"},
    )
    assert streamlit_app._ttd_chart_png_b64(prepared) is None


def test_render_ttd_assistant_message_plain_text(mock_st: MagicMock) -> None:
    streamlit_app._render_ttd_assistant_message({"text": "hello"})
    mock_st.markdown.assert_called_once()


def test_ttd_display_streaming_optional_ui_branches(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    png = base64.b64decode(TINY_PNG_B64)
    prepared_empty_df = PreparedTurn(
        sql="SELECT 1",
        df=pd.DataFrame(),
        chart_png=None,
        plan_message="",
        stream_prompt="go",
        chart_spec={"kind": "none"},
    )
    mock_st.write_stream.return_value = ""
    with patch.object(streamlit_app, "generate_text_stream", return_value=iter(("a",))):
        out = streamlit_app._ttd_display_streaming_assistant(
            get_settings(), prepared_empty_df, show_plan_caption=False
        )
    assert out == ""
    mock_st.chat_message.assert_called()
    prepared_rich = PreparedTurn(
        sql="SELECT x",
        df=pd.DataFrame({"x": [9]}),
        chart_png=png,
        plan_message="Plan here",
        stream_prompt="rp",
        chart_spec={"kind": "none"},
    )
    with patch.object(streamlit_app, "generate_text_stream", return_value=iter(("b",))):
        streamlit_app._ttd_display_streaming_assistant(
            get_settings(), prepared_rich, show_plan_caption=True
        )
    mock_st.caption.assert_called()


def test_ttd_try_run_edited_sql_branches(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    p = tmp_path / "d"
    p.mkdir()
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    get_settings.cache_clear()
    streamlit_app._ttd_try_run_edited_sql(get_settings(), p, run_edited=True)
    mock_st.warning.assert_called()
    mock_st.reset_mock()
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gp")
    get_settings.cache_clear()
    mock_st.session_state["ttd_sql_editor"] = "   "
    streamlit_app._ttd_try_run_edited_sql(get_settings(), p, run_edited=True)
    mock_st.warning.assert_called()
    mock_st.reset_mock()
    err_turn = PreparedTurn(
        sql="",
        df=pd.DataFrame(),
        chart_png=None,
        plan_message="",
        stream_prompt="",
        chart_spec=None,
        error="bad sql",
    )
    mock_st.session_state["ttd_sql_editor"] = "SELECT bad"
    with patch.object(streamlit_app, "prepare_sql_only", return_value=err_turn):
        streamlit_app._ttd_try_run_edited_sql(get_settings(), p, run_edited=True)
    mock_st.error.assert_called_once()


def test_ttd_refuse_guardrail_injection_keeps_history_user_free(mock_st: MagicMock) -> None:
    mock_st.session_state["ttd_messages"] = []
    gv = streamlit_app.GuardrailViolation("injection", "blocked")
    streamlit_app._ttd_refuse_guardrail(gv, q="evil")
    assert mock_st.session_state.ttd_messages == [{"role": "assistant", "text": "blocked"}]


def test_ttd_refuse_guardrail_topic_keeps_raw_user_turn(mock_st: MagicMock) -> None:
    mock_st.session_state["ttd_messages"] = []
    gv = streamlit_app.GuardrailViolation("topic", "stay focused")
    streamlit_app._ttd_refuse_guardrail(gv, q="recipe")
    assert mock_st.session_state.ttd_messages[0]["role"] == "user"
    assert mock_st.session_state.ttd_messages[1]["text"] == "stay focused"


def test_ttd_run_nl_turn_guardrail(
    monkeypatch: pytest.MonkeyPatch,
    mock_st: MagicMock,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "t.csv").write_text("x\n1\n", encoding="utf-8")
    mock_st.session_state["ttd_messages"] = []

    def boom(*_a, **_kw):
        raise streamlit_app.GuardrailViolation("injection", "blocked")

    with patch.object(streamlit_app, "apply_chat_guardrails", side_effect=boom):
        streamlit_app._ttd_run_nl_turn(get_settings(), ds, q="IGNORE PREVIOUS")
    assert mock_st.session_state.ttd_messages[-1]["text"] == "blocked"


def test_ttd_run_nl_turn_prepare_failure(
    monkeypatch: pytest.MonkeyPatch, mock_st: MagicMock, tmp_path: Path
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    ds = tmp_path / "ds2"
    ds.mkdir()
    (ds / "t.csv").write_text("x\n1\n", encoding="utf-8")
    mock_st.session_state.ttd_messages = []
    failed = PreparedTurn(
        sql="",
        df=pd.DataFrame(),
        chart_png=None,
        plan_message="",
        stream_prompt="",
        chart_spec=None,
        error="planner exploded",
    )
    gr_ok = MagicMock(pii_labels=())
    with (
        patch.object(
            streamlit_app,
            "apply_chat_guardrails",
            return_value=("safe", gr_ok),
        ),
        patch.object(streamlit_app, "prepare_turn", return_value=failed),
    ):
        streamlit_app._ttd_run_nl_turn(get_settings(), ds, q="counts please")
    assert "planner exploded" in mock_st.session_state.ttd_messages[-1]["text"]
    mock_st.error.assert_called_once()


def test_ttd_run_nl_turn_success_updates_state(
    monkeypatch: pytest.MonkeyPatch, mock_st: MagicMock, tmp_path: Path
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    ds = tmp_path / "ds3"
    ds.mkdir()
    (ds / "orders.csv").write_text("id\n1\n", encoding="utf-8")
    mock_st.session_state.ttd_messages = []
    mock_st.session_state["ttd_last_sql"] = ""
    ok_turn = PreparedTurn(
        sql="SELECT 1",
        df=pd.DataFrame({"a": [1]}),
        chart_png=None,
        plan_message="p",
        stream_prompt="stream",
        chart_spec={"kind": "none"},
    )
    gr_mask = MagicMock(pii_labels=("email",))
    with (
        patch.object(
            streamlit_app,
            "apply_chat_guardrails",
            return_value=("masked q", gr_mask),
        ),
        patch.object(streamlit_app, "prepare_turn", return_value=ok_turn),
        patch.object(
            streamlit_app,
            "_ttd_display_streaming_assistant",
            return_value="narr",
        ),
    ):
        streamlit_app._ttd_run_nl_turn(get_settings(), ds, q="q with x@y.co")
    assert mock_st.session_state["ttd_pending_editor_sql"] == ok_turn.sql
    assert mock_st.session_state["ttd_last_sql"] == ok_turn.sql
    mock_st.caption.assert_called()
    mock_st.rerun.assert_called_once()


def test_render_talk_chat_input_vertex_missing(
    mock_st: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    get_settings.cache_clear()
    ds = tmp_path / "dx"
    ds.mkdir()
    (ds / "m.csv").write_text("z\n", encoding="utf-8")
    mock_st.chat_input.return_value = "something"
    mock_st.selectbox.return_value = 0
    mock_st.button.return_value = False
    mock_st.columns.return_value = (MagicMock(), MagicMock())
    with (
        patch.object(streamlit_app, "DEFAULT_DATA_ROOT", tmp_path),
        patch.object(
            streamlit_app,
            "list_datasets",
            return_value=[
                {
                    "dataset_id": "d1",
                    "_path": str(ds.resolve()),
                    "created_at": "2024-05-06T01:02:03",
                }
            ],
        ),
    ):
        streamlit_app.render_talk_to_data(get_settings())
    mock_st.warning.assert_called()


def test_syn_run_generate_action_exception_logged(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    streamlit_app._init_synthetic_session_state()
    mock_st.button.return_value = True
    mock_st.session_state["syn_ddl"] = "CREATE TABLE t (id INT);"
    with patch.object(streamlit_app, "generate_full_dataset", side_effect=RuntimeError("boom")):
        streamlit_app._syn_run_generate_action(get_settings(), 3, 0.7)
    mock_st.exception.assert_called_once()


def test_syn_try_refine_vertex_missing(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    get_settings.cache_clear()
    mock_st.button.return_value = True
    schema = MagicMock()
    streamlit_app._syn_try_refine_table(get_settings(), schema, "tname", "fix it", 0.2)
    mock_st.error.assert_called_once()


def test_syn_try_refine_empty_feedback_warnings(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    mock_st.button.return_value = True
    streamlit_app._syn_try_refine_table(get_settings(), MagicMock(), "tname", "\t\n  ", 0.5)
    mock_st.warning.assert_called_once()


def test_syn_try_refine_updates_rows(monkeypatch: pytest.MonkeyPatch, mock_st: MagicMock) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    streamlit_app._init_synthetic_session_state()
    mock_st.session_state.syn_data = {"widgets": [{"a": 1}]}
    mock_st.button.return_value = True

    class _Sch:
        def ordered_table_names(self) -> list[str]:
            return ["widgets"]

    with (
        patch.object(streamlit_app, "refine_table", return_value=[{"a": 99}]),
        patch.object(streamlit_app, "validate_tables_data", return_value=[]),
    ):
        streamlit_app._syn_try_refine_table(get_settings(), _Sch(), "widgets", "bigger counts", 0.1)
    assert mock_st.session_state.syn_data["widgets"] == [{"a": 99}]
    mock_st.rerun.assert_called_once()


def test_syn_render_table_expanders_shows_dataframe(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    mock_st.button.return_value = False
    streamlit_app._init_synthetic_session_state()
    mock_st.session_state.syn_data = {"acct": [{"x": "y"}]}
    schema = MagicMock()
    schema.ordered_table_names.return_value = ["acct"]
    streamlit_app._syn_render_table_expanders(get_settings(), schema, temperature=0.3)
    mock_st.expander.assert_called()
    mock_st.dataframe.assert_called()


def test_syn_save_and_download_persists_dataset(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    get_settings.cache_clear()
    streamlit_app._init_synthetic_session_state()
    mock_st.session_state.update(
        {
            "syn_dataset_id": "did-9",
            "syn_data": {"t": [{}]},
            "syn_ddl": "CREATE TABLE t (id INT);",
            "syn_instructions": "",
        }
    )
    mock_st.button.side_effect = [True, False]
    out_folder = tmp_path / "saved"
    with (
        patch.object(streamlit_app, "DEFAULT_DATA_ROOT", tmp_path),
        patch.object(
            streamlit_app,
            "persist_and_zip",
            return_value=(out_folder, b"ZIPBYTES"),
        ),
    ):
        streamlit_app._syn_save_and_download(rows_n=8, temperature=0.4)
    assert mock_st.session_state.syn_zip == b"ZIPBYTES"
    mock_st.download_button.assert_called_once()


def test_normalize_ttd_messages_skips_unknown_items() -> None:
    mixed: list[Any] = [None, {}, ["oops"], ["user", "only valid"]]
    out = streamlit_app._normalize_ttd_messages(mixed)
    assert out == [{"role": "user", "text": "only valid"}]


def test_render_data_generation_parses_ddl_when_data_present(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gp")
    get_settings.cache_clear()
    streamlit_app._init_synthetic_session_state()
    mock_st.session_state.syn_errors = []
    mock_st.session_state.syn_dataset_id = None
    mock_st.session_state.syn_zip = None
    mock_st.session_state.syn_data = {"t": [{"id": 1}]}
    mock_st.session_state.syn_ddl = "CREATE TABLE t (id INT);"
    mock_st.columns.return_value = (MagicMock(), MagicMock())
    mock_st.slider.return_value = 0.4
    mock_st.number_input.return_value = 3
    mock_st.button.return_value = False

    schema = MagicMock()
    schema.ordered_table_names.return_value = ["t"]

    with (
        patch.object(streamlit_app, "parse_ddl", return_value=schema),
        patch.object(streamlit_app, "_syn_render_table_expanders") as rex,
        patch.object(streamlit_app, "_syn_save_and_download"),
    ):
        streamlit_app.render_data_generation(get_settings())

    rex.assert_called_once()


def test_ttd_try_run_edited_sql_success_appends_turn(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    d = tmp_path / "ddb"
    d.mkdir()
    (d / "t.csv").write_text("c\n9\n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gp")
    get_settings.cache_clear()
    mock_st.session_state.update(
        {"ttd_messages": [], "ttd_sql_editor": "SELECT c FROM t", "ttd_last_chart_spec": None}
    )
    ok = PreparedTurn(
        sql="SELECT c FROM t",
        df=pd.DataFrame({"c": [9]}),
        chart_png=None,
        plan_message="",
        stream_prompt="s",
        chart_spec={"kind": "none"},
    )
    with (
        patch.object(streamlit_app, "prepare_sql_only", return_value=ok),
        patch.object(streamlit_app, "_ttd_display_streaming_assistant", return_value="said"),
        patch.object(streamlit_app, "generate_text_stream", return_value=iter(["x"])),
    ):
        streamlit_app._ttd_try_run_edited_sql(get_settings(), d, run_edited=True)
    assert mock_st.session_state.ttd_messages[-1]["role"] == "assistant"


def test_ttd_run_nl_turn_prepare_raises_shows_exception(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gp")
    get_settings.cache_clear()
    ds = tmp_path / "dse"
    ds.mkdir()
    (ds / "t.csv").write_text("k\n", encoding="utf-8")
    mock_st.session_state.ttd_messages = []
    fake_gr = MagicMock(pii_labels=())
    with (
        patch.object(
            streamlit_app,
            "apply_chat_guardrails",
            return_value=("ok", fake_gr),
        ),
        patch.object(streamlit_app, "prepare_turn", side_effect=OSError("disk")),
    ):
        streamlit_app._ttd_run_nl_turn(get_settings(), ds, q="hi")
    mock_st.exception.assert_called_once()


def test_render_talk_resolves_pending_sql_editor(
    mock_st: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gp")
    get_settings.cache_clear()
    dx = tmp_path / "rdx"
    dx.mkdir()
    (dx / "q.csv").write_text("z\n", encoding="utf-8")
    mock_st.chat_input.return_value = None
    mock_st.selectbox.return_value = 0
    mock_st.button.return_value = False
    mock_st.columns.return_value = (MagicMock(), MagicMock())
    mock_st.session_state = mock_st.session_state.__class__()
    mock_st.session_state["ttd_pending_editor_sql"] = "SELECT 1 pending"
    with (
        patch.object(streamlit_app, "DEFAULT_DATA_ROOT", tmp_path),
        patch.object(
            streamlit_app,
            "list_datasets",
            return_value=[
                {
                    "dataset_id": "d1",
                    "_path": str(dx.resolve()),
                    "created_at": "2025-06-06T09:09:09",
                }
            ],
        ),
    ):
        streamlit_app.render_talk_to_data(get_settings())
    assert mock_st.session_state["ttd_sql_editor"] == "SELECT 1 pending"


def test_langfuse_nl_when_all_identity_fields_blank(
    mock_st: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    mock_st.user = SimpleNamespace(email=" ", id=" ", sub="")
    ctx = streamlit_app._langfuse_nl_sql_context(get_settings())
    assert ctx.user_id is None


def test_langfuse_stream_context_skips_blank_user(
    mock_st: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    get_settings.cache_clear()
    mock_st.user = None
    ctx = streamlit_app._langfuse_stream_context(get_settings())
    assert ctx.user_id is None
