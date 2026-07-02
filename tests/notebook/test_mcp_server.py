"""Unit tests for the MCP server tool logic + mount gating (Phase 1, read tools).

The tools' logic lives in module-level ``_*`` functions that take a
``SessionManager``, so these tests exercise them directly — no MCP client, no
live socket (avoids the TestClient-WS portal hang, and keeps them fast). A
notebook session is built in-process and registered without a venv sync, since
the read tools never execute a cell.
"""

from __future__ import annotations

import pytest

from strata.notebook.mcp_server import (
    _add_cell,
    _dag,
    _edit_cell,
    _get_cell,
    _get_notebook,
    _list_notebooks,
    _move_cell,
    _remove_cell,
    _run_cell,
    _run_tests,
    _status,
    build_mcp_app,
)
from strata.notebook.ops import LocalNotebookOps, NotebookOpsError
from tests.notebook.test_cli import _build_notebook


@pytest.fixture
def sm_with_session(tmp_path):
    """A SessionManager holding one live session for a two-cell chain a→b."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession, SessionManager

    nb_dir = _build_notebook(tmp_path, cells=[("a", "x = 1", None), ("b", "y = x + 1", "a")])
    sm = SessionManager()
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    # Register directly: read tools don't run cells, so skip the venv sync that
    # open_notebook would do.
    sm._sessions[session.id] = session
    return sm, session.id, nb_dir


def test_list_notebooks_reports_open_sessions(sm_with_session):
    sm, session_id, nb_dir = sm_with_session
    notebooks = _list_notebooks(sm)
    assert len(notebooks) == 1
    entry = notebooks[0]
    assert entry["session_id"] == session_id
    assert entry["path"] == str(nb_dir)
    assert entry["name"]


def test_get_notebook_returns_cells_in_order(sm_with_session):
    sm, session_id, _ = sm_with_session
    result = _get_notebook(sm, session_id)
    assert [c["id"] for c in result["cells"]] == ["a", "b"]
    assert result["cells"][0]["source"] == "x = 1"
    # Curated view — internal bookkeeping doesn't leak to the agent.
    assert "last_provenance_hash" not in result["cells"][0]


def test_get_cell_and_unknown_cell(sm_with_session):
    sm, session_id, _ = sm_with_session
    cell = _get_cell(sm, session_id, "a")
    assert cell["id"] == "a"
    assert cell["source"] == "x = 1"
    with pytest.raises(NotebookOpsError):
        _get_cell(sm, session_id, "ghost")


def test_dag_exposes_the_edge(sm_with_session):
    sm, session_id, _ = sm_with_session
    dag = _dag(sm, session_id)
    assert any(
        e["from_cell_id"] == "a" and e["to_cell_id"] == "b" and e["variable"] == "x"
        for e in dag["edges"]
    )
    assert dag["topological_order"].index("a") < dag["topological_order"].index("b")


def test_status_summary(sm_with_session):
    sm, session_id, _ = sm_with_session
    status = _status(sm, session_id)
    assert status["name"]
    assert {row["id"] for row in status["cells"]} == {"a", "b"}


def test_unknown_session_raises_valueerror(sm_with_session):
    sm, _, _ = sm_with_session
    # A missing session is a client error, surfaced to the agent as a tool error.
    with pytest.raises(ValueError, match="no open notebook session"):
        _get_notebook(sm, "nope")


def test_from_session_reuses_the_live_session(sm_with_session):
    sm, session_id, _ = sm_with_session
    live = sm.get_session(session_id)
    ops = LocalNotebookOps.from_session(live)
    # Same underlying session object — the warm state, not an offline reopen.
    assert ops._session is live
    assert ops.notebook_dir == live.path


@pytest.mark.asyncio
async def test_run_cell_broadcasts_and_maps(sm_with_session, monkeypatch):
    sm, session_id, _ = sm_with_session
    seen = {}

    async def fake_broadcast(session, cell_id, execution_state, notebook_id, mode="normal"):
        seen["args"] = (cell_id, notebook_id, mode)

        class _Result:
            def to_dict(self):
                return {
                    "cell_id": cell_id,
                    "status": "ready",
                    "cache_hit": False,
                    "execution_method": "subprocess",
                    "duration_ms": 12.0,
                    "stdout": "hi\n",
                    "stderr": "",
                    "error": None,
                }

        return _Result()

    # Patch the shared broadcast path — _run_cell imports it at call time, so
    # patching the source module is enough. No subprocess, no real WS.
    monkeypatch.setattr("strata.notebook.ws.execute_cell_and_broadcast", fake_broadcast)
    # A directly-built test session has no synced venv, so the env-ready guard
    # would refuse; a UI/CLI-opened session in production is ready. Simulate that.
    monkeypatch.setattr(
        sm.get_session(session_id), "environment_execution_block_message", lambda: None
    )

    result = await _run_cell(sm, session_id, "a", mode="rerun")
    assert result["cell_id"] == "a"
    assert result["status"] == "ok"  # "ready" → "ok"
    assert result["stdout"] == "hi\n"
    # The live session id is threaded through as the broadcast notebook_id.
    assert seen["args"] == ("a", session_id, "rerun")


@pytest.mark.asyncio
async def test_run_cell_rejects_bad_mode_missing_cell_and_session(sm_with_session):
    sm, session_id, _ = sm_with_session
    with pytest.raises(ValueError, match="unknown run mode"):
        await _run_cell(sm, session_id, "a", mode="bogus")
    with pytest.raises(NotebookOpsError):
        await _run_cell(sm, session_id, "ghost")
    with pytest.raises(ValueError, match="no open notebook session"):
        await _run_cell(sm, "nope", "a")


@pytest.mark.asyncio
async def test_run_cell_refuses_when_env_not_ready(sm_with_session):
    sm, session_id, _ = sm_with_session
    # A freshly-built session has no synced venv → run_cell refuses with a clear
    # message rather than running into a broken environment.
    with pytest.raises(ValueError, match="environment"):
        await _run_cell(sm, session_id, "a")


@pytest.mark.asyncio
async def test_run_tests_maps_outcomes(sm_with_session, monkeypatch):
    sm, session_id, _ = sm_with_session
    from strata.notebook.models import CellTestCase, CellTestResult

    async def fake_run_cell_tests(self, cell_id, test_source):
        return CellTestResult(
            passed=1,
            failed=1,
            tests=[
                CellTestCase(name="t_ok", outcome="passed"),
                CellTestCase(name="t_bad", outcome="failed", message="assert 1 == 2"),
            ],
        )

    monkeypatch.setattr("strata.notebook.executor.CellExecutor.run_cell_tests", fake_run_cell_tests)
    # run_tests refuses a cell with no test file.
    with pytest.raises(NotebookOpsError):
        await _run_tests(sm, session_id, "a")
    # Give cell 'a' a test source, then it maps the executor's result.
    sm.get_session(session_id).notebook_state.get_cell("a").test_source = "def test_x(cell): pass"
    result = await _run_tests(sm, session_id, "a")
    assert result["passed"] == 1 and result["failed"] == 1
    assert [c["name"] for c in result["cases"]] == ["t_ok", "t_bad"]


@pytest.mark.asyncio
async def test_authoring_add_edit_move_remove_and_broadcast(sm_with_session, monkeypatch):
    sm, session_id, _ = sm_with_session
    broadcasts = []

    async def fake_sync(notebook_id, session):
        broadcasts.append(notebook_id)

    # Each mutation should push a live state sync to the session's spectators.
    monkeypatch.setattr("strata.notebook.ws.broadcast_notebook_sync", fake_sync)

    added = await _add_cell(sm, session_id, "z = 9", after="a", language="python")
    assert added["source"] == "z = 9"
    new_id = added["id"]
    assert _get_cell(sm, session_id, new_id)["source"] == "z = 9"

    edited = await _edit_cell(sm, session_id, new_id, "z = 10")
    assert edited["source"] == "z = 10"
    assert _get_cell(sm, session_id, new_id)["source"] == "z = 10"

    order = await _move_cell(sm, session_id, new_id, 0)
    assert order["cells"][0]["id"] == new_id

    removed = await _remove_cell(sm, session_id, new_id)
    assert removed == {"removed": new_id}
    with pytest.raises(NotebookOpsError):
        _get_cell(sm, session_id, new_id)

    # add, edit, move, remove → four live syncs, all for this session.
    assert broadcasts == [session_id] * 4


@pytest.mark.asyncio
async def test_authoring_errors(sm_with_session):
    sm, session_id, _ = sm_with_session
    with pytest.raises(ValueError, match="no open notebook session"):
        await _add_cell(sm, "nope", "x = 1")
    with pytest.raises(NotebookOpsError):
        await _edit_cell(sm, session_id, "ghost", "x = 1")
    with pytest.raises(NotebookOpsError):
        await _remove_cell(sm, session_id, "ghost")


def test_build_mcp_app_returns_mountable_app(sm_with_session):
    sm, _, _ = sm_with_session
    mcp_app = build_mcp_app(sm)
    # [mcp] extra is installed in the dev/CI env (--all-extras), so we get an app.
    assert mcp_app is not None
    # Mountable ASGI app with a lifespan the host can enter.
    assert hasattr(mcp_app, "router")
    assert hasattr(mcp_app.router, "lifespan_context")
