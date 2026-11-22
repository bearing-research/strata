"""A lock that moved outside Strata is installed before the next run.

A cell's key reads ``uv.lock`` on disk, so a run in the environment as it was last synced
is stored as the new lock's. Every run entry point asks the session to sync first.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from typing import cast

import pytest
from fastapi import HTTPException, WebSocket

from strata.notebook import routes, ws
from strata.notebook.env import compute_lockfile_hash
from strata.notebook.environment_jobs import EnvironmentJobSnapshot
from strata.notebook.mcp_server import _run_cell, _scratch_run
from strata.notebook.parser import parse_notebook
from strata.notebook.project_env import ProjectEnvBackend
from strata.notebook.runtime_state import persist_environment_synced_lockfile_hash
from strata.notebook.session import NotebookSession, SessionManager
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from tests.notebook.e2e_fixtures import FakeNotebookWebSocket


@pytest.fixture
def session(tmp_path):
    """A notebook synced at one lock, with one Python cell."""
    notebook = create_notebook(tmp_path, "moved", initialize_environment=False)
    add_cell_to_notebook(notebook, "a")
    write_cell(notebook, "a", "x = 1")
    (notebook / "uv.lock").write_text("as synced")
    persist_environment_synced_lockfile_hash(notebook, compute_lockfile_hash(notebook))
    return NotebookSession(parse_notebook(notebook), notebook)


class _Jobs:
    """The session's environment jobs, recorded; a sync that *installs* records the lock."""

    def __init__(self, session, monkeypatch, *, installs=True, busy=False):
        self.session, self.installs, self.busy = session, installs, busy
        self.submitted: list[str] = []
        self.waited = 0
        monkeypatch.setattr(session.environment_jobs, "submit", self.submit)
        monkeypatch.setattr(session.environment_jobs, "wait", self.wait)

    async def submit(self, *, action):
        if self.busy:
            raise RuntimeError("Notebook execution is currently running.")
        self.submitted.append(action)

    async def wait(self):
        self.waited += 1
        if self.installs:
            path = self.session.path
            persist_environment_synced_lockfile_hash(path, compute_lockfile_hash(path))


def _move_lock(session, text="moved outside") -> None:
    (session.path / "uv.lock").write_text(text)


async def test_a_moved_lock_is_synced_and_the_run_waits_for_it(session, monkeypatch):
    jobs = _Jobs(session, monkeypatch)
    _move_lock(session)

    await session.sync_a_moved_lock()

    assert jobs.submitted == ["sync"]
    assert jobs.waited == 1


async def test_a_lock_where_it_was_asks_for_no_sync(session, monkeypatch):
    jobs = _Jobs(session, monkeypatch)

    await session.sync_a_moved_lock()

    assert jobs.submitted == []


async def test_a_notebook_never_synced_asks_for_no_sync(session, monkeypatch):
    """An open's own sync sees to that; a session built on a prepared venv has none."""
    jobs = _Jobs(session, monkeypatch)
    persist_environment_synced_lockfile_hash(session.path, "")
    _move_lock(session)

    await session.sync_a_moved_lock()

    assert jobs.submitted == []


async def test_a_sync_that_failed_is_not_asked_again_until_the_lock_moves_again(
    session, monkeypatch
):
    jobs = _Jobs(session, monkeypatch, installs=False)
    _move_lock(session)

    await session.sync_a_moved_lock()
    await session.sync_a_moved_lock()
    assert jobs.submitted == ["sync"]

    _move_lock(session, "moved again")
    await session.sync_a_moved_lock()
    assert jobs.submitted == ["sync", "sync"]


async def test_a_busy_notebook_runs_on_and_is_asked_again_by_the_next_run(session, monkeypatch):
    jobs = _Jobs(session, monkeypatch, busy=True)
    _move_lock(session)

    await session.sync_a_moved_lock()
    assert jobs.submitted == []

    jobs.busy = False
    await session.sync_a_moved_lock()
    assert jobs.submitted == ["sync"]


async def test_a_caller_that_goes_away_leaves_the_sync_to_finish(session, monkeypatch):
    jobs = _Jobs(session, monkeypatch)
    release, finished = asyncio.Event(), asyncio.Event()

    async def wait():
        await release.wait()
        finished.set()

    monkeypatch.setattr(session.environment_jobs, "wait", wait)
    _move_lock(session)
    run = asyncio.create_task(session.sync_a_moved_lock())
    while not jobs.submitted:
        await asyncio.sleep(0)
    await asyncio.sleep(0)

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    release.set()

    await asyncio.wait_for(finished.wait(), 5)


async def test_a_scratchpad_in_its_projects_environment_is_left_to_its_own_refresh(
    session, monkeypatch
):
    jobs = _Jobs(session, monkeypatch)
    session.backend = object.__new__(ProjectEnvBackend)
    _move_lock(session)

    await session.sync_a_moved_lock()

    assert jobs.submitted == []


@pytest.mark.integration  # a real sync
async def test_a_package_locked_outside_is_installed_before_the_cell_that_imports_it(tmp_path):
    """The whole path, with uv: the run waits for the install, then runs in it."""
    package = tmp_path / "tinydep"
    (package / "tinydep").mkdir(parents=True)
    (package / "tinydep" / "__init__.py").write_text("VALUE = 7\n")
    (package / "pyproject.toml").write_text('[project]\nname = "tinydep"\nversion = "0.1.0"\n')
    notebook = create_notebook(tmp_path, "outside")
    add_cell_to_notebook(notebook, "c")
    write_cell(notebook, "c", "import tinydep\nseven = tinydep.VALUE\n")
    manager = SessionManager()
    session = manager.open_notebook(notebook)
    try:
        assert session.environment_attestation_error() is None
        uv = shutil.which("uv")
        assert uv is not None
        # As a pull or a terminal leaves it: the lock moved, the environment did not.
        subprocess.run(
            [uv, "add", "--no-sync", str(package)], cwd=notebook, check=True, capture_output=True
        )
        assert session.environment_attestation_error() is not None

        result = await _run_cell(manager, session.id, "c")

        assert result["status"] == "ok", result
        assert session.environment_attestation_error() is None
    finally:
        manager.close_session(session.id)


# --- every run entry point asks, before it reserves the run or checks the environment ---


@pytest.fixture
def asked(session, monkeypatch):
    """*session* with its environment busy, so each entry point stops at its check, and the
    reservation state each ask for a sync saw."""
    session.environment_sync_state = "ready"
    session.environment_job = EnvironmentJobSnapshot(
        id="job-1",
        action="add",
        command="uv add x",
        status="running",
        phase="uv_running",
        started_at=1,
    )
    seen: list[bool] = []

    async def ask():
        seen.append(ws.notebook_has_active_execution(session.id))

    monkeypatch.setattr(session, "sync_a_moved_lock", ask)
    return seen


_BY_PAYLOAD = {
    "_handle_cell_execute": {"cell_id": "a"},
    "_handle_cell_execute_cascade": {"cell_id": "a", "plan_id": "p"},
    "_handle_cell_execute_force": {"cell_id": "a"},
    "_handle_cell_execute_rerun": {"cell_id": "a"},
    "_handle_cell_run_tests": {"cell_id": "a", "test_source": "def test_x():\n    pass\n"},
}


def _socket(session):
    fake = FakeNotebookWebSocket()
    ws._notebook_connections.setdefault(session.id, []).append(cast(WebSocket, fake))
    return fake, ws._ensure_execution_state(session.id)


@pytest.mark.parametrize("handler", sorted(_BY_PAYLOAD))
async def test_a_websocket_run_asks_first(session, asked, handler):
    fake, state = _socket(session)

    await getattr(ws, handler)(cast(WebSocket, fake), session, _BY_PAYLOAD[handler], state, "nb")

    assert asked == [False], "asked once, with no run reserved yet"
    assert not ws.notebook_has_active_execution(session.id)


@pytest.mark.parametrize("handler", ["_handle_notebook_run_all", "_handle_notebook_rerun_all"])
async def test_a_websocket_run_of_every_cell_asks_first(session, asked, handler):
    fake, state = _socket(session)

    await getattr(ws, handler)(cast(WebSocket, fake), session, state, "nb", {})

    assert asked == [False], "asked once, with no run reserved yet"
    assert [f["payload"]["code"] for f in fake.frames_of("error")] == ["ENVIRONMENT_BUSY"]


async def test_an_agents_run_and_its_scratchpad_run_ask_first(session, asked):
    manager = SessionManager()
    manager._sessions[session.id] = session

    with pytest.raises(ValueError, match="add"):
        await _run_cell(manager, session.id, "a")
    refused = await _scratch_run(manager, "y = 2")

    assert refused["status"] == "error", refused
    assert asked == [False, False]


async def test_a_rest_run_and_a_rest_test_run_ask_first(session, asked):
    session.notebook_state.get_cell("a").test_source = "def test_x():\n    pass\n"

    for route in (routes.execute_cell, routes.run_cell_tests_endpoint):
        with pytest.raises(HTTPException) as refused:
            await route(session.id, session, "a")
        assert refused.value.detail["code"] == "ENVIRONMENT_BUSY"

    assert asked == [False, False]
