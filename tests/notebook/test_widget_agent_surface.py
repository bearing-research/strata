"""What an agent can see and do with a widget cell.

Round 6 had to combine a REST call, an MCP edit and a raw WebSocket frame to
drive one slider: ``add_cell`` refused ``language="widget"``, ``get_cell``
reported a widget's source and status but not its controls or their values, and
no tool set one. A widget's selection is runtime state rather than source, so an
agent reading the cell could not tell 0.9 from the declared default and editing
the cell could not change what the notebook computed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strata.notebook.mcp_server import _add_cell, _get_cell, _set_widget_value
from strata.notebook.ops import NotebookOpsError
from strata.notebook.parser import parse_notebook
from strata.notebook.runtime_state import load_runtime_state
from strata.notebook.scopes import (
    CLASSIFIED_TOOLS,
    NOTEBOOK_SCOPE_EXECUTE,
    required_scope_for_tool,
)
from strata.notebook.session import NotebookSession, SessionManager
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

WIDGET_SRC = "alpha = slider(0, 1, default=0.5)\nmode = dropdown(['a', 'b'], default='b')\n"


def _registered(nb: Path) -> tuple[SessionManager, str, NotebookSession]:
    sm = SessionManager()
    session = NotebookSession(parse_notebook(nb), nb)
    # A widget cell runs in-process with no subprocess, but the shared execute
    # gate still checks the environment before letting anything run.
    session.environment_sync_state = "ready"
    sm._sessions[session.id] = session
    return sm, session.id, session


@pytest.fixture
def widget_nb(tmp_path):
    nb = create_notebook(tmp_path, "Widget Agent", initialize_environment=False)
    (nb / ".venv").mkdir(exist_ok=True)
    add_cell_to_notebook(nb, "controls", None, language="widget")
    write_cell(nb, "controls", WIDGET_SRC)
    add_cell_to_notebook(nb, "consume", "controls")
    write_cell(nb, "consume", "beta = alpha * 2\nbeta\n")
    return nb


@pytest.mark.asyncio
async def test_an_agent_can_add_a_widget_cell(tmp_path):
    """The HTTP route took ``widget``; the shared operation contract did not,
    so an agent had to reach past MCP to create one."""
    nb = create_notebook(tmp_path, "Add Widget", initialize_environment=False)
    (nb / ".venv").mkdir(exist_ok=True)
    sm, session_id, _ = _registered(nb)

    view = await _add_cell(sm, session_id, WIDGET_SRC, language="widget")

    assert view["language"] == "widget"
    assert [control["name"] for control in view["controls"]] == ["alpha", "mode"]


@pytest.mark.asyncio
async def test_an_unknown_language_is_still_refused(tmp_path):
    nb = create_notebook(tmp_path, "Bad Language", initialize_environment=False)
    (nb / ".venv").mkdir(exist_ok=True)
    sm, session_id, _ = _registered(nb)

    with pytest.raises(NotebookOpsError, match="unsupported language"):
        await _add_cell(sm, session_id, "x = 1", language="cobol")


def test_get_cell_reports_the_controls_and_their_defaults(widget_nb):
    sm, session_id, _ = _registered(widget_nb)

    view = _get_cell(sm, session_id, "controls")

    alpha = next(c for c in view["controls"] if c["name"] == "alpha")
    assert alpha["kind"] == "slider"
    assert alpha["default"] == 0.5
    # Nothing selected yet, so the declared default is what the cell runs at,
    # and that is what ``value`` reports. It used to be ``None`` here, which
    # described the storage rather than the notebook: an agent looking for the
    # input behind a result read it as "unset".
    assert alpha["value"] == 0.5


def test_a_non_widget_cell_has_no_controls(widget_nb):
    sm, session_id, _ = _registered(widget_nb)

    assert _get_cell(sm, session_id, "consume")["controls"] == []


@pytest.mark.asyncio
async def test_setting_a_control_moves_the_value_the_notebook_computes_from(widget_nb):
    import json

    sm, session_id, session = _registered(widget_nb)

    result = await _set_widget_value(sm, session_id, "controls", {"alpha": 0.9})

    assert result["values"]["alpha"] == 0.9
    assert result["run"]["status"] == "ok"
    # Persisted, so it survives a reopen.
    assert load_runtime_state(widget_nb).cells["controls"].widget_values == {"alpha": 0.9}
    # And re-materialized: the artifact a downstream cell reads holds 0.9.
    mgr = session.get_artifact_manager()
    art = mgr.artifact_store.get_latest_version(
        f"nb_{session.notebook_state.id}_cell_controls_var_alpha"
    )
    assert json.loads(mgr.load_artifact_data(art.id, art.version)) == 0.9
    # The cell an agent reads back reports it too.
    alpha = next(
        c for c in _get_cell(sm, session_id, "controls")["controls"] if c["name"] == "alpha"
    )
    assert alpha["value"] == 0.9


@pytest.mark.asyncio
async def test_a_control_the_cell_does_not_declare_is_refused(widget_nb):
    """Silently dropping it would report success for a value never applied."""
    sm, session_id, _ = _registered(widget_nb)

    with pytest.raises(NotebookOpsError, match="alpha, mode"):
        await _set_widget_value(sm, session_id, "controls", {"alhpa": 0.9})

    entry = load_runtime_state(widget_nb).cells.get("controls")
    assert entry is None or entry.widget_values == {}


@pytest.mark.asyncio
async def test_a_busy_notebook_leaves_the_stored_values_alone(widget_nb):
    """The write happens under the execution reservation, not before it.

    Persisting first would leave a refused value on disk: the agent is told the
    notebook is busy, nothing re-materializes, and the next run quietly uses the
    value the server rejected.
    """
    import asyncio

    from strata.notebook.ws import _ensure_execution_state

    sm, session_id, _ = _registered(widget_nb)
    await _set_widget_value(sm, session_id, "controls", {"alpha": 0.25})

    async def _never_finishes() -> None:
        await asyncio.Event().wait()

    execution_state = _ensure_execution_state(session_id)
    blocker = asyncio.create_task(_never_finishes())
    async with execution_state.control_lock:
        execution_state.execution_task = blocker
        execution_state.running_cell = "consume"
    try:
        with pytest.raises(NotebookOpsError, match="already executing"):
            await _set_widget_value(sm, session_id, "controls", {"alpha": 0.75})
    finally:
        blocker.cancel()
        async with execution_state.control_lock:
            execution_state.reset_execution()

    assert load_runtime_state(widget_nb).cells["controls"].widget_values == {"alpha": 0.25}


@pytest.mark.asyncio
async def test_a_python_cell_is_refused(widget_nb):
    sm, session_id, _ = _registered(widget_nb)

    with pytest.raises(NotebookOpsError, match="not a widget"):
        await _set_widget_value(sm, session_id, "consume", {"alpha": 0.9})


@pytest.mark.asyncio
async def test_setting_a_control_tells_an_attached_viewer(widget_nb, monkeypatch):
    """Moving the slider in the browser broadcasts; an agent doing the same
    thing must too, or a watching human keeps the pre-change staleness badges."""
    import strata.notebook.mcp_server as mcp_server
    import strata.notebook.ws as ws

    sm, session_id, _ = _registered(widget_nb)
    frames: list[tuple[str, str]] = []
    notes: list[str] = []

    async def _fake_broadcast_message(nb_id, message):
        frames.append((message.get("type", ""), str(message.get("payload", ""))))

    async def _fake_note(sid, source, text):
        notes.append(text)

    monkeypatch.setattr(ws, "_broadcast_message", _fake_broadcast_message)
    monkeypatch.setattr(mcp_server, "_agent_note", _fake_note)

    await _set_widget_value(sm, session_id, "controls", {"alpha": 0.9})

    # The run broadcasts its own frames, the way ``run_cell`` does; the tool
    # adds no reload on top, because it changes no committed config.
    assert any(kind == "cell_status" and "controls" in payload for kind, payload in frames), frames
    assert notes and "alpha=0.9" in notes[0]


def test_the_new_tool_is_in_the_scope_table():
    assert "set_widget_value" in CLASSIFIED_TOOLS
    assert required_scope_for_tool("set_widget_value") == NOTEBOOK_SCOPE_EXECUTE
