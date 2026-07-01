"""End-to-end: drive RemoteNotebookOps against the *real* FastAPI app.

``test_ops_remote.py`` exercises the remote backend with an ``httpx.MockTransport``
and hand-built payloads — fast, but blind to drift between the server's real
``serialize()`` output and the wire mappers. This module closes that gap: it
points ``RemoteNotebookOps`` at the live app via Starlette's ``TestClient`` (an
in-process ASGI ``httpx.Client``), opens a real session, and round-trips the
read + authoring verbs. A field rename in ``serialize_cell`` / ``_format_dag``
or a changed response envelope fails here even though the mock tests would pass.

REST only (no WebSocket), so it sidesteps the py3.12-macOS TestClient-WS hang.
Remote *execution* mapping (``run``/``tests``) is covered by the route-dispatch
test plus the mock unit tests; it is not re-driven here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from strata.notebook.ops import NotebookOpsError, RemoteNotebookOps
from strata.notebook.routes import router
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.fixture(autouse=True)
def _no_uv_sync(monkeypatch):
    """Skip real venv/pool creation — we only exercise HTTP + serialization."""
    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)

    async def _fake_stream(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(success=True, error=None, operation_log=None)

    monkeypatch.setattr("strata.notebook.dependencies.run_uv_command_streaming", _fake_stream)

    async def _noop_start(self):
        pass

    monkeypatch.setattr("strata.notebook.pool.WarmProcessPool.start", _noop_start)
    # The session-state endpoint RemoteNotebookOps reads is personal-mode only.
    monkeypatch.setattr(
        "strata.server._state",
        SimpleNamespace(config=SimpleNamespace(transforms_config={}, deployment_mode="personal")),
    )


@pytest.fixture(scope="module")
def app():
    fastapi_app = FastAPI()
    fastapi_app.include_router(router)
    return fastapi_app


@pytest.fixture
def remote(app, tmp_path):
    """A RemoteNotebookOps wired to a live session on the real app via TestClient.

    Returns ``(ops, client)``; the notebook has cells a (`x = 1`) → b (`y = x+1`).
    """
    nb = create_notebook(tmp_path, "E2E Notebook", initialize_environment=False)
    add_cell_to_notebook(nb, "a", None, language="python")
    write_cell(nb, "a", "x = 1\n")
    add_cell_to_notebook(nb, "b", "a", language="python")
    write_cell(nb, "b", "y = x + 1\n")

    client = TestClient(app)
    resp = client.post("/v1/notebooks/open", json={"path": str(nb)})
    assert resp.status_code == 200, resp.text
    session_id = resp.json()["session_id"]
    # TestClient *is* a sync httpx.Client; RemoteNotebookOps drives it like any other.
    return RemoteNotebookOps("http://testserver", session_id, client=client), client


def test_e2e_list_get_dag_status(remote):
    ops, _ = remote

    cells = ops.list_cells()
    assert [c.id for c in cells] == ["a", "b"]
    assert cells[0].source.strip() == "x = 1"

    assert ops.get_cell("b").source.strip() == "y = x + 1"
    with pytest.raises(NotebookOpsError, match="no cell"):
        ops.get_cell("ghost")

    dag = ops.dag()
    assert any(
        e.from_cell_id == "a" and e.to_cell_id == "b" and e.variable == "x" for e in dag.edges
    )
    assert dag.topological_order.index("a") < dag.topological_order.index("b")

    status = ops.status()
    assert status.name == "E2E Notebook"
    assert [c.id for c in status.cells] == ["a", "b"]


def test_e2e_authoring_roundtrip(remote):
    ops, _ = remote

    created = ops.add_cell("z = 9\n", after="a")
    new_id = created.id
    assert created.source.strip() == "z = 9"
    ids = [c.id for c in ops.list_cells()]
    assert ids[ids.index("a") + 1] == new_id  # inserted right after a

    edited = ops.edit_cell(new_id, "z = 99\n")
    assert edited.source.strip() == "z = 99"
    # The edit is visible on a fresh read through the real server.
    assert ops.get_cell(new_id).source.strip() == "z = 99"

    order = [c.id for c in ops.move_cell(new_id, 0)]
    assert order[0] == new_id

    ops.remove_cell(new_id)
    assert new_id not in [c.id for c in ops.list_cells()]
    with pytest.raises(NotebookOpsError, match="no cell"):
        ops.remove_cell(new_id)


def test_e2e_annotation_roundtrip(remote):
    ops, _ = remote
    # Edit a cell to carry an annotation; the server must parse it and surface
    # the name through serialize → wire → CellView (the mapper drift this guards).
    ops.edit_cell("a", "# @name loader\n# @worker gpu-box\nx = 1\n")
    cell = ops.get_cell("a")
    assert cell.name == "loader"


def test_e2e_add_cell_bad_after_is_ops_error(remote):
    ops, _ = remote
    with pytest.raises(NotebookOpsError):
        ops.add_cell("q = 1\n", after="ghost")


def test_e2e_set_cell_tests(remote):
    ops, _ = remote
    # Author a cell's test source over the wire (the gap the live demo exposed):
    # the round-trip writes cells/a.test.py and returns the cell.
    src = "def test_x(cell):\n    assert cell.x == 1\n"
    cell = ops.set_cell_tests("a", src)
    assert cell.id == "a"
    with pytest.raises(NotebookOpsError):
        ops.set_cell_tests("ghost", src)
