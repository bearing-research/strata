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
    _dag,
    _get_cell,
    _get_notebook,
    _list_notebooks,
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


def test_build_mcp_app_returns_mountable_app(sm_with_session):
    sm, _, _ = sm_with_session
    mcp_app = build_mcp_app(sm)
    # [mcp] extra is installed in the dev/CI env (--all-extras), so we get an app.
    assert mcp_app is not None
    # Mountable ASGI app with a lifespan the host can enter.
    assert hasattr(mcp_app, "router")
    assert hasattr(mcp_app.router, "lifespan_context")
