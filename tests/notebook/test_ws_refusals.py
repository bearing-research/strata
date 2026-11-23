"""What the WebSocket handlers refuse, and that a refusal leaves the notebook as it was."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strata.notebook import ws
from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from tests.shim_helpers import write_interpreter_shim


class _Socket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    def frames(self, kind: str) -> list[dict]:
        return [frame["payload"] for frame in self.sent if frame["type"] == kind]

    def errors(self) -> list[str]:
        return [payload["error"] for payload in self.frames("error")]


@pytest.fixture
def connected(tmp_path: Path):
    """``c1`` and ``c2`` (which reads it), with a socket that receives the broadcasts."""
    notebook_dir = create_notebook(tmp_path, "Refusals", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", "x = 1\n")
    add_cell_to_notebook(notebook_dir, "c2", after_cell_id="c1")
    write_cell(notebook_dir, "c2", "y = x + 1\n")
    # An interpreter in place: with no .venv the refresh runs a real ``uv sync``.
    python = notebook_dir / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    write_interpreter_shim(python)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    socket = _Socket()
    ws._notebook_connections[session.id] = [socket]
    try:
        yield session, socket, ws._ensure_execution_state(session.id)
    finally:
        ws._notebook_connections.pop(session.id, None)
        ws._notebook_inspect_managers.pop(session.id, None)
        ws.forget_notebook_execution_state(session.id)


RUNS = {
    "force": (ws._handle_cell_execute_force, {"cell_id": "c2"}),
    "rerun": (ws._handle_cell_execute_rerun, {"cell_id": "c2"}),
    "cascade": (ws._handle_cell_execute_cascade, {"cell_id": "c2", "plan_id": "p"}),
    "tests": (ws._handle_cell_run_tests, {"cell_id": "c2", "test_source": "def test_y(): pass"}),
}


@pytest.mark.parametrize("kind", sorted(RUNS))
async def test_a_run_while_another_runs_is_refused_and_names_the_cell(connected, kind):
    session, socket, state = connected
    handler, payload = RUNS[kind]
    state.running_cell = "c1"

    await handler(socket, session, payload, state, session.id)

    (refusal,) = socket.frames("error")
    assert (refusal["code"], refusal["cell_id"]) == ("notebook_busy", "c2")
    # The run that holds the notebook keeps it, and nothing was reserved for c2.
    assert (state.running_cell, state.requested_cell, state.execution_task) == ("c1", None, None)


@pytest.mark.parametrize("kind", sorted(RUNS))
async def test_a_missing_cell_id_is_refused(connected, kind):
    session, socket, state = connected
    handler, _payload = RUNS[kind]

    await handler(socket, session, {}, state, session.id)

    (error,) = socket.errors()
    assert error.startswith("Missing cell_id")
    # Nothing was reserved: the next run is admitted.
    assert await ws._reserve_execution_request(state, "c1") is None


async def test_an_expired_cascade_plan_is_refused_and_releases_the_notebook(connected):
    session, socket, state = connected

    await ws._handle_cell_execute_cascade(
        socket, session, {"cell_id": "c2", "plan_id": "gone"}, state, session.id
    )

    assert socket.errors() == ["Cascade plan not found or expired"]
    # A reservation left behind would refuse every later run as busy.
    assert state.requested_cell is None
    assert await ws._reserve_execution_request(state, "c1") is None


class TestSourceUpdates:
    async def test_a_source_over_a_megabyte_is_refused(self, connected):
        session, socket, state = connected

        await ws._handle_cell_source_update(
            socket, session, {"cell_id": "c1", "source": "#" * 1_000_001}, state, session.id
        )

        assert socket.errors() == ["Cell source exceeds 1MB limit"]
        cell = session.notebook_state.get_cell("c1")
        assert cell is not None and cell.source == "x = 1\n"
        assert (session.path / "cells" / "c1.py").read_text() == "x = 1\n"

    @pytest.mark.parametrize("held", ["running_cell", "requested_cell"])
    async def test_the_running_cell_cannot_be_edited(self, connected, held):
        session, socket, state = connected
        setattr(state, held, "c1")

        await ws._handle_cell_source_update(
            socket, session, {"cell_id": "c1", "source": "x = 2\n"}, state, session.id
        )

        (refusal,) = socket.frames("error")
        assert (refusal["code"], refusal["cell_id"]) == ("cell_busy", "c1")
        cell = session.notebook_state.get_cell("c1")
        assert cell is not None and cell.source == "x = 1\n"
        assert (session.path / "cells" / "c1.py").read_text() == "x = 1\n"


async def test_a_run_that_raises_is_an_error_and_its_readers_go_stale(connected, monkeypatch):
    session, socket, state = connected
    reader = session.notebook_state.get_cell("c2")
    assert reader is not None
    reader.status = CellStatus.READY

    async def _boom(self, cell_id, source, init=None):
        raise RuntimeError("the executor broke")

    monkeypatch.setattr(CellExecutor, "execute_cell", _boom)

    assert await ws.execute_cell_and_broadcast(session, "c1", state, session.id) is None

    assert [e["error"] for e in socket.frames("cell_error")] == ["the executor broke"]
    statuses = [(p["cell_id"], p["status"]) for p in socket.frames("cell_status")]
    assert ("c1", "error") in statuses
    assert ("c2", "stale") in statuses
    assert session.notebook_state.get_cell("c1").status == CellStatus.ERROR
    assert reader.status == CellStatus.STALE
    assert state.running_cell is None


class TestDependencies:
    @pytest.mark.parametrize("handler", [ws._handle_dependency_add, ws._handle_dependency_remove])
    async def test_a_missing_package_is_refused(self, connected, handler):
        session, socket, state = connected

        await handler(socket, session, {}, state, session.id)

        assert socket.errors() == ["Missing 'package' in payload"]

    @pytest.mark.parametrize("handler", [ws._handle_dependency_add, ws._handle_dependency_remove])
    async def test_a_name_that_is_not_a_package_never_reaches_uv(
        self, connected, monkeypatch, handler
    ):
        session, socket, state = connected
        submitted: list[dict] = []

        async def _submit(**kwargs):
            submitted.append(kwargs)

        monkeypatch.setattr(session.environment_jobs, "submit", _submit)

        await handler(socket, session, {"package": "--index-url=http://x"}, state, session.id)

        (error,) = socket.errors()
        assert error.startswith("Not a single package requirement")
        assert submitted == []

    @pytest.mark.parametrize("handler", [ws._handle_dependency_add, ws._handle_dependency_remove])
    async def test_a_job_already_running_is_reported(self, connected, monkeypatch, handler):
        session, socket, state = connected

        async def _busy(**kwargs):
            raise RuntimeError("an environment job is already running")

        monkeypatch.setattr(session.environment_jobs, "submit", _busy)

        await handler(socket, session, {"package": "requests"}, state, session.id)

        (busy,) = socket.frames("error")
        assert (busy["error"], busy["code"]) == (
            "an environment job is already running",
            "ENVIRONMENT_BUSY",
        )


async def test_evaluating_without_an_inspect_session_is_refused(connected):
    session, socket, state = connected

    await ws._handle_inspect_eval(
        socket, session, {"cell_id": "c1", "expr": "x"}, state, session.id
    )

    (result,) = socket.frames("inspect_result")
    assert (result["ok"], result["error"]) == (False, "No inspect session open for this cell")
