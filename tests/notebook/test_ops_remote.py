"""Tests for RemoteNotebookOps — the read-only remote inspect backend (P3b).

Drives a canned session-state endpoint through an httpx ``MockTransport`` (no
server, no WebSocket) and asserts the remote backend projects the server's JSON
through the *same* wire mapper the local backend uses — so a remote view is
byte-for-byte the local view for that notebook.
"""

from __future__ import annotations

import json

import httpx
import pytest

from strata.cli import main
from strata.notebook.ops import (
    CellView,
    LocalNotebookOps,
    NotebookOpsError,
    OutputView,
    RemoteNotebookOps,
    RunResult,
)
from tests.notebook.test_cli import _build_notebook

_SESSION_ID = "sess-123"


def _dag_json(dag) -> dict:
    """The DAG dict the server's ``_format_dag`` would emit for this session."""
    if not dag:
        return {
            "edges": [],
            "topological_order": [],
            "leaves": [],
            "roots": [],
            "variable_producer": {},
        }
    return {
        "edges": [
            {"from_cell_id": e.from_cell_id, "to_cell_id": e.to_cell_id, "variable": e.variable}
            for e in dag.edges
        ],
        "topological_order": list(dag.topological_order),
        "leaves": list(dag.leaves),
        "roots": list(dag.roots),
        "variable_producer": dict(dag.variable_producer),
    }


def _session_payload(local: LocalNotebookOps) -> dict:
    """Build the ``GET /sessions/{id}`` JSON the server returns for this notebook."""
    session = local._session
    payload = {
        "id": session.notebook_state.id,
        "name": session.notebook_state.name,
        "cells": session.serialize_cells(),
        "dag": _dag_json(session.dag),
    }
    # Round-trip through JSON so the fixture is exactly what crosses the wire
    # (StrEnums collapse to plain strings, etc.).
    return json.loads(json.dumps(payload, default=str))


@pytest.fixture
def remote_ops(tmp_path):
    """A RemoteNotebookOps backed by a MockTransport serving a real notebook.

    Returns ``(remote, local)`` so tests can assert remote ≡ local.
    """
    nb = _build_notebook(tmp_path, cells=[("a", "x = 1", None), ("b", "y = x + 1", "a")])
    local = LocalNotebookOps(nb)
    payload = _session_payload(local)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/notebooks/sessions/{_SESSION_ID}":
            return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"detail": "Session not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return RemoteNotebookOps("http://test", _SESSION_ID, client=client), local


def _json(views) -> list[dict]:
    return [v.model_dump(mode="json") for v in views]


def test_remote_list_cells_matches_local(remote_ops):
    remote, local = remote_ops
    assert _json(remote.list_cells()) == _json(local.list_cells())
    assert [c.id for c in remote.list_cells()] == ["a", "b"]


def test_remote_get_cell_and_unknown(remote_ops):
    remote, local = remote_ops
    assert remote.get_cell("a").model_dump(mode="json") == local.get_cell("a").model_dump(
        mode="json"
    )
    with pytest.raises(NotebookOpsError):
        remote.get_cell("ghost")


def test_remote_dag_matches_local(remote_ops):
    remote, local = remote_ops
    assert remote.dag().model_dump(mode="json") == local.dag().model_dump(mode="json")
    assert any(e.from_cell_id == "a" and e.to_cell_id == "b" for e in remote.dag().edges)


def test_remote_status_matches_local(remote_ops):
    remote, local = remote_ops
    assert remote.status().model_dump(mode="json") == local.status().model_dump(mode="json")
    # The notebook name survives the wire (the cells endpoint alone drops it).
    assert remote.status().name and remote.status().name == local.status().name


def test_remote_unknown_session_is_ops_error():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(404, json={"detail": "nope"}))
    )
    remote = RemoteNotebookOps("http://test", "missing", client=client)
    with pytest.raises(NotebookOpsError, match="no session"):
        remote.list_cells()


def test_remote_server_error_is_ops_error():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(500, text="boom"))
    )
    remote = RemoteNotebookOps("http://test", "s", client=client)
    with pytest.raises(NotebookOpsError, match="500"):
        remote.status()


def test_remote_unreachable_server_is_ops_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(boom))
    remote = RemoteNotebookOps("http://test", "s", client=client)
    with pytest.raises(NotebookOpsError, match="cannot reach"):
        remote.dag()


# -- CLI wiring: the --server/--session selector -----------------------------


def test_cli_server_without_session_is_exit_2(capsys):
    rc = main(["cell", "list", "--server", "http://localhost:8765"])
    assert rc == 2
    assert "requires --session" in capsys.readouterr().err


def test_cli_no_target_is_exit_2(capsys):
    rc = main(["dag"])
    assert rc == 2
    assert "notebook directory or --server" in capsys.readouterr().err


def test_cli_list_routes_to_remote(monkeypatch, capsys):
    class _FakeRemote:
        def __init__(self, base_url, session_id, **_):
            assert base_url == "http://srv" and session_id == "s1"

        def list_cells(self):
            return [
                CellView(
                    id="a",
                    name="",
                    language="python",
                    status="ready",
                    source="x = 1",
                    staleness_reasons=[],
                    upstream_ids=[],
                    downstream_ids=[],
                    outputs=[OutputView()],
                    console_stdout="",
                    console_stderr="",
                )
            ]

    monkeypatch.setattr("strata.notebook.ops.RemoteNotebookOps", _FakeRemote)
    rc = main(["cell", "list", "--server", "http://srv", "--session", "s1", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert [c["id"] for c in data] == ["a"]


# -- remote execution (run / test) -------------------------------------------


def _exec_remote(handler) -> RemoteNotebookOps:
    return RemoteNotebookOps(
        "http://test", "s1", client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_remote_add_cell_malformed_response_raises():
    # POST mints with no "id" → a clean ops error, not a KeyError traceback.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    with pytest.raises(NotebookOpsError, match="missing 'id'"):
        _exec_remote(handler).add_cell("x = 1")


def test_remote_edit_cell_malformed_response_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"dag": {}})  # no "cell"

    with pytest.raises(NotebookOpsError, match="missing 'cell'"):
        _exec_remote(handler).edit_cell("c1", "x = 1")


def test_cli_read_closes_remote_client(monkeypatch, capsys):
    closed = {"n": 0}

    class _FakeRemote:
        def __init__(self, base_url, session_id, **_):
            pass

        def list_cells(self):
            return []

        def close(self):
            closed["n"] += 1

    monkeypatch.setattr("strata.notebook.ops.RemoteNotebookOps", _FakeRemote)
    rc = main(["cell", "list", "--server", "http://srv", "--session", "s1", "--format", "json"])
    assert rc == 0
    assert closed["n"] == 1  # the remote client was closed on the sync read path


@pytest.mark.asyncio
async def test_remote_run_cell_maps_and_passes_mode():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["mode"] = request.url.params.get("mode")
        return httpx.Response(
            200,
            json={
                "cell_id": "c1",
                "status": "ready",  # the server renames success → status
                "cache_hit": True,
                "execution_method": "warm",
                "duration_ms": 12.5,
                "error": None,
                "stdout": "hi\n",
                "stderr": "",
            },
        )

    result = await _exec_remote(handler).run_cell("c1", mode="rerun")
    assert result.status == "ok" and result.cache_hit and result.stdout == "hi\n"
    assert result.execution_method == "warm"
    assert seen["path"] == "/v1/notebooks/s1/cells/c1/execute"
    assert seen["mode"] == "rerun"


@pytest.mark.asyncio
async def test_remote_run_cell_error_status_maps_to_error():
    handler = lambda req: httpx.Response(  # noqa: E731
        200, json={"cell_id": "c1", "status": "error", "error": "boom"}
    )
    result = await _exec_remote(handler).run_cell("c1")
    assert result.status == "error" and result.error == "boom"


@pytest.mark.asyncio
async def test_remote_run_cell_http_errors():
    def make(code: int, body: dict) -> RemoteNotebookOps:
        return _exec_remote(lambda req: httpx.Response(code, json=body))

    with pytest.raises(NotebookOpsError, match="no cell"):
        await make(404, {"detail": "Cell not found"}).run_cell("c1")
    with pytest.raises(NotebookOpsError, match="environment busy"):
        await make(409, {"detail": {"message": "syncing"}}).run_cell("c1")
    with pytest.raises(NotebookOpsError, match="unknown run mode"):
        await make(400, {"detail": "unknown run mode 'x'"}).run_cell("c1")


@pytest.mark.asyncio
async def test_remote_run_tests_maps():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/notebooks/s1/cells/c1/tests"
        return httpx.Response(
            200,
            json={
                "cell_id": "c1",
                "passed": 1,
                "failed": 1,
                "errored": 0,
                "skipped": 0,
                "pytest_unavailable": False,
                "tests": [
                    {"name": "t_ok", "nodeid": "n1", "outcome": "passed", "message": ""},
                    {"name": "t_bad", "nodeid": "n2", "outcome": "failed", "message": "assert"},
                ],
            },
        )

    result = await _exec_remote(handler).run_tests("c1")
    assert result.passed == 1 and result.failed == 1 and not result.pytest_unavailable
    assert [c.name for c in result.cases] == ["t_ok", "t_bad"]
    assert result.cases[1].outcome == "failed" and result.cases[1].message == "assert"


@pytest.mark.asyncio
async def test_remote_run_tests_no_tests_is_ops_error():
    remote = _exec_remote(lambda req: httpx.Response(400, json={"detail": "Cell c1 has no tests"}))
    with pytest.raises(NotebookOpsError, match="no tests"):
        await remote.run_tests("c1")


def test_cli_cell_run_routes_to_remote(monkeypatch, capsys):
    class _FakeRemote:
        def __init__(self, base_url, session_id, **_):
            assert base_url == "http://srv" and session_id == "s1"

        async def run_cell(self, cell_id, *, mode="normal"):
            assert cell_id == "c1" and mode == "rerun"  # --rerun → mode=rerun
            return RunResult(
                cell_id="c1",
                status="ok",
                cache_hit=False,
                execution_method="cold",
                duration_ms=3.0,
            )

        def close(self):
            pass

    monkeypatch.setattr("strata.notebook.ops.RemoteNotebookOps", _FakeRemote)
    rc = main(
        [
            "cell",
            "run",
            "--server",
            "http://srv",
            "--session",
            "s1",
            "c1",
            "--rerun",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


# -- remote authoring (add / edit / rm / mv / dep) ---------------------------


def _wire_cell(cell_id: str, source: str, *, name: str = "") -> dict:
    """A minimal serialized-cell wire dict (what the server returns for a cell)."""
    return {
        "id": cell_id,
        "language": "python",
        "status": "idle",
        "source": source,
        "annotations": {"name": name},
    }


def test_remote_add_cell_mints_then_sets_source():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/notebooks/s1/cells":
            assert json.loads(request.content) == {"after_cell_id": "a", "language": "python"}
            return httpx.Response(200, json={"id": "new1"})  # add mints an empty cell
        if request.method == "PUT" and request.url.path == "/v1/notebooks/s1/cells/new1":
            assert json.loads(request.content) == {"source": "z = 9"}
            return httpx.Response(
                200, json={"cell": _wire_cell("new1", "z = 9", name="n"), "dag": {}}
            )
        return httpx.Response(404)

    cell = _exec_remote(handler).add_cell("z = 9", after="a")
    assert cell.id == "new1" and cell.source == "z = 9" and cell.name == "n"


def test_remote_edit_cell_and_unknown():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.url.path == "/v1/notebooks/s1/cells/c1":
            return httpx.Response(200, json={"cell": _wire_cell("c1", "x = 2"), "dag": {}})
        return httpx.Response(404, json={"detail": "Cell not found"})

    assert _exec_remote(handler).edit_cell("c1", "x = 2").source == "x = 2"
    with pytest.raises(NotebookOpsError, match="no cell"):
        _exec_remote(handler).edit_cell("ghost", "y")


def test_remote_remove_cell_and_unknown():
    ok = _exec_remote(
        lambda req: httpx.Response(200, json={"message": "Cell deleted", "cell_id": "c1"})
    )
    assert ok.remove_cell("c1") is None
    with pytest.raises(NotebookOpsError, match="no cell"):
        _exec_remote(lambda req: httpx.Response(404)).remove_cell("ghost")


def test_remote_move_cell_reorders():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "/sessions/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "id": "nb",
                    "name": "N",
                    "cells": [_wire_cell("a", ""), _wire_cell("b", "")],
                    "dag": {},
                },
            )
        if request.method == "PUT" and request.url.path == "/v1/notebooks/s1/cells/reorder":
            assert json.loads(request.content)["cell_ids"] == ["b", "a"]
            return httpx.Response(
                200, json={"notebook_id": "nb", "cells": [_wire_cell("b", ""), _wire_cell("a", "")]}
            )
        return httpx.Response(404)

    cells = _exec_remote(handler).move_cell("a", 1)
    assert [c.id for c in cells] == ["b", "a"]


def test_remote_move_cell_unknown_is_ops_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"id": "nb", "name": "N", "cells": [_wire_cell("a", "")], "dag": {}}
        )

    with pytest.raises(NotebookOpsError, match="no cell"):
        _exec_remote(handler).move_cell("ghost", 0)


@pytest.mark.asyncio
async def test_remote_add_dependency_maps():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and request.url.path == "/v1/notebooks/s1/dependencies"
        assert json.loads(request.content) == {"package": "pandas"}
        return httpx.Response(
            200, json={"success": True, "package": "pandas", "lockfile_changed": True}
        )

    res = await _exec_remote(handler).add_dependency("pandas")
    assert res.success and res.package == "pandas" and res.action == "add" and res.lockfile_changed


@pytest.mark.asyncio
async def test_remote_remove_dependency_uses_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/v1/notebooks/s1/dependencies/pandas"
        return httpx.Response(
            200, json={"success": True, "package": "pandas", "lockfile_changed": True}
        )

    res = await _exec_remote(handler).remove_dependency("pandas")
    assert res.success and res.action == "remove"


@pytest.mark.asyncio
async def test_remote_dependency_failure_is_structured_not_raised():
    # A failed `uv` resolve (400) is a result with success=False, not an exception.
    remote = _exec_remote(
        lambda req: httpx.Response(400, json={"detail": {"message": "no such package"}})
    )
    res = await remote.add_dependency("nope")
    assert res.success is False and "no such package" in (res.error or "")


@pytest.mark.asyncio
async def test_remote_dependency_busy_raises():
    remote = _exec_remote(lambda req: httpx.Response(409, json={"detail": {"message": "syncing"}}))
    with pytest.raises(NotebookOpsError, match="environment busy"):
        await remote.remove_dependency("pandas")


def test_cli_cell_add_routes_to_remote(monkeypatch, tmp_path, capsys):
    class _FakeRemote:
        def __init__(self, base_url, session_id, **_):
            assert base_url == "http://srv" and session_id == "s1"

        def add_cell(self, source, *, after=None, language="python"):
            assert source == "w = 5"
            return CellView(
                id="new1",
                name="",
                language="python",
                status="idle",
                source="w = 5",
                staleness_reasons=[],
                upstream_ids=[],
                downstream_ids=[],
                outputs=[],
                console_stdout="",
                console_stderr="",
            )

    monkeypatch.setattr("strata.notebook.ops.RemoteNotebookOps", _FakeRemote)
    src = tmp_path / "s.py"
    src.write_text("w = 5")
    rc = main(
        [
            "cell",
            "add",
            "--server",
            "http://srv",
            "--session",
            "s1",
            "--file",
            str(src),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["id"] == "new1"
