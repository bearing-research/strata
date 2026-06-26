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
