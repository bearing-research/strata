"""Session limits: warm pool size, idle timeout by user activity, count, memory, close."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from strata.notebook import mcp_server, quiesce
from strata.notebook import routes as notebook_routes
from strata.notebook.dependencies import EnvironmentOperationLog
from strata.notebook.session import EnvironmentJobSnapshot, SessionManager
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from tests.notebook.e2e_fixtures import FakeNotebookWebSocket, create_test_app
from tests.notebook.e2e_fixtures import _reset_ws_globals as _reset_ws


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class _Memory:
    def __init__(self, mb: int | None) -> None:
        self.mb = mb

    def __call__(self) -> int | None:
        return self.mb


class _LiveSocket(FakeNotebookWebSocket):
    """Stays open until the client or the server closes it."""

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def receive_text(self) -> str:
        item = await self._queue.get()
        if item is None:
            raise WebSocketDisconnect(code=1000)
        return item

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await super().close(code, reason)
        self._queue.put_nowait(None)

    def send(self, msg_type: str, payload: dict | None = None) -> None:
        self._queue.put_nowait(json.dumps({"type": msg_type, "seq": 1, "payload": payload or {}}))

    def closed_frames(self) -> list[dict]:
        return [f["payload"] for f in self.frames_of("session_closed")]


async def _until(predicate) -> None:
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


@pytest.fixture(autouse=True)
def _clean_ws_state():
    _reset_ws()
    quiesce.reset()
    yield
    _reset_ws()
    quiesce.reset()


@pytest.fixture
def config(monkeypatch, tmp_path):
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.server import ServerState

    config = StrataConfig(artifact_dir=str(tmp_path / "artifacts"))
    config.notebook_storage_dir = tmp_path
    config.notebook_warm_pool_size = 1
    config.notebook_session_ttl_seconds = 900
    monkeypatch.setattr(server_module, "_state", ServerState(config), raising=False)
    return config


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def memory() -> _Memory:
    return _Memory(None)


@pytest.fixture
def manager(monkeypatch, clock, memory) -> SessionManager:
    manager = SessionManager(clock=clock, available_memory_mb=memory)
    # Routes and the socket handler share this one.
    monkeypatch.setattr(notebook_routes, "_session_manager", manager)
    return manager


async def _uv_succeeds(*args, **kwargs):
    return SimpleNamespace(
        success=True, error=None, operation_log=EnvironmentOperationLog(command="uv")
    )


def _fake_rscript(tmp_path, monkeypatch) -> Path:
    """An ``Rscript`` on PATH; ``fast_notebook_env`` keeps WarmProcessPool.start a no-op, so no
    pool runs it."""
    rscript = tmp_path / "bin" / "Rscript"
    rscript.parent.mkdir()
    rscript.write_text("#!/bin/sh\nexit 1\n")
    rscript.chmod(0o755)
    monkeypatch.setenv("PATH", f"{rscript.parent}{os.pathsep}{os.environ['PATH']}")
    return rscript


def _notebook(tmp_path, name: str, *, environment: bool = False):
    notebook_dir = create_notebook(tmp_path, name, initialize_environment=environment)
    add_cell_to_notebook(notebook_dir, "root")
    write_cell(notebook_dir, "root", "x = 1")
    return notebook_dir


async def _connect(session) -> tuple[_LiveSocket, asyncio.Task]:
    from strata.notebook.ws import notebook_websocket

    socket = _LiveSocket()
    task = asyncio.create_task(notebook_websocket(cast(WebSocket, socket), session.id))
    await _until(lambda: socket.frames_of("presence"))
    return socket, task


async def _hold_a_run(session) -> asyncio.Task:
    """Make *session* look like it is running a cell until the task is cancelled."""
    from strata.notebook import ws as notebook_ws

    task = asyncio.create_task(asyncio.Event().wait())
    notebook_ws._ensure_execution_state(session.id).execution_task = task
    return task


class TestWarmPoolAndTimeout:
    @pytest.mark.warm_pool
    async def test_one_warm_process_and_closed_after_the_timeout_alone(
        self, config, manager, clock, tmp_path
    ):
        session = manager.open_notebook(
            _notebook(tmp_path, "pooled", environment=True), skip_initial_venv_sync=True
        )
        pool = session.warm_pool
        assert pool is not None
        await asyncio.gather(*pool._background_tasks)
        assert pool._available.qsize() == 1
        warm = pool._available.get_nowait()
        pool._available.put_nowait(warm)

        clock.now += 899
        await manager.sweep()
        assert manager.list_sessions() == [session.id]

        # No other notebook is opened: the sweep alone closes it.
        clock.now += 2
        await manager.sweep()
        assert manager.list_sessions() == []
        # The process, not the queue: the drain dequeues before the kill finishes.
        await _until(lambda: warm.process.returncode is not None)

    @pytest.mark.warm_pool
    @pytest.mark.parametrize("sync_after_open", [False, True])
    async def test_a_notebook_opened_through_the_server_runs_warm(
        self, config, manager, monkeypatch, tmp_path, sync_after_open
    ):
        # The open's environment job, or a later sync route, must not block its own pool.
        monkeypatch.setattr("strata.notebook.dependencies.run_uv_command_streaming", _uv_succeeds)
        notebook_dir = _notebook(tmp_path, "served", environment=True)
        transport = httpx.ASGITransport(app=create_test_app())
        session = None
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                opened = await client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})
                assert opened.status_code == 200, opened.text
                session = manager.get_session(opened.json()["session_id"])
                assert session is not None
                if sync_after_open:
                    if session.warm_pool is not None:
                        await asyncio.gather(*session.warm_pool._background_tasks)
                        await session.warm_pool.drain()
                        session.warm_pool = None
                    synced = await client.post(f"/v1/notebooks/{session.id}/environment/sync")
                    assert synced.status_code == 200, synced.text
                assert session.warm_pool is not None
                await asyncio.gather(*session.warm_pool._background_tasks)

                ran = await client.post(f"/v1/notebooks/{session.id}/cells/root/execute")

            assert ran.status_code == 200, ran.text
            assert ran.json()["status"] == "ready"
            assert ran.json()["execution_method"] == "warm"
        finally:
            if session is not None:
                await asyncio.gather(*manager.close_session(session.id))

    async def test_a_notebook_with_r_cells_opened_through_the_server_gets_an_r_pool(
        self, config, manager, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("strata.notebook.dependencies.run_uv_command_streaming", _uv_succeeds)
        rscript = _fake_rscript(tmp_path, monkeypatch)
        notebook_dir = _notebook(tmp_path, "served-r", environment=True)
        add_cell_to_notebook(notebook_dir, "rcell", language="r")
        write_cell(notebook_dir, "rcell", "y <- 1")
        transport = httpx.ASGITransport(app=create_test_app())
        session = None
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                opened = await client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})

            assert opened.status_code == 200, opened.text
            session = manager.get_session(opened.json()["session_id"])
            assert session is not None
            assert session.r_warm_pool is not None
            assert session.r_warm_pool.worker_command is not None
            assert session.r_warm_pool.worker_command[0] == str(rscript)
        finally:
            if session is not None:
                await asyncio.gather(*manager.close_session(session.id))

    @pytest.mark.parametrize("surface", ["rest", "mcp"])
    async def test_the_first_r_cell_added_through_the_server_starts_an_r_pool(
        self, config, manager, monkeypatch, tmp_path, surface
    ):
        monkeypatch.setattr("strata.notebook.dependencies.run_uv_command_streaming", _uv_succeeds)
        rscript = _fake_rscript(tmp_path, monkeypatch)
        notebook_dir = _notebook(tmp_path, "python-then-r", environment=True)
        transport = httpx.ASGITransport(app=create_test_app())
        session = None
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                opened = await client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})
                assert opened.status_code == 200, opened.text
                session = manager.get_session(opened.json()["session_id"])
                assert session is not None
                await session.wait_for_environment_job()
                assert session.warm_pool is not None
                assert session.r_warm_pool is None

                if surface == "rest":
                    added = await client.post(
                        f"/v1/notebooks/{session.id}/cells",
                        json={"after_cell_id": "root", "language": "r"},
                    )
                    assert added.status_code == 200, added.text
                else:
                    await mcp_server._add_cell(manager, session.id, "x <- 1", "root", "r")

            assert session.r_warm_pool is not None
            assert session.r_warm_pool.worker_command is not None
            assert session.r_warm_pool.worker_command[0] == str(rscript)
        finally:
            if session is not None:
                await asyncio.gather(*manager.close_session(session.id))

    async def test_a_sync_starts_the_r_pool_beside_a_running_python_pool(
        self, config, manager, monkeypatch, tmp_path
    ):
        # R installed after the first R cell was added: the next sync must still start its pool.
        monkeypatch.setattr("strata.notebook.dependencies.run_uv_command_streaming", _uv_succeeds)
        which = shutil.which
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name, *a, **k: None if name == "Rscript" else which(name, *a, **k),
        )
        notebook_dir = _notebook(tmp_path, "r-installed-later", environment=True)
        transport = httpx.ASGITransport(app=create_test_app())
        session = None
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                opened = await client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})
                assert opened.status_code == 200, opened.text
                session = manager.get_session(opened.json()["session_id"])
                assert session is not None
                await session.wait_for_environment_job()
                added = await client.post(
                    f"/v1/notebooks/{session.id}/cells",
                    json={"after_cell_id": "root", "language": "r"},
                )
                assert added.status_code == 200, added.text
                assert session.warm_pool is not None
                assert session.r_warm_pool is None

                rscript = _fake_rscript(tmp_path, monkeypatch)
                monkeypatch.setattr(shutil, "which", which)
                synced = await client.post(f"/v1/notebooks/{session.id}/environment/sync")

            assert synced.status_code == 200, synced.text
            assert session.r_warm_pool is not None
            assert session.r_warm_pool.worker_command is not None
            assert session.r_warm_pool.worker_command[0] == str(rscript)
        finally:
            if session is not None:
                await asyncio.gather(*manager.close_session(session.id))

    @pytest.mark.warm_pool
    @pytest.mark.parametrize("change", ["add", "remove", "requirements", "mcp add", "mcp remove"])
    async def test_a_dependency_change_through_the_server_leaves_cells_running_warm(
        self, config, manager, monkeypatch, tmp_path, change
    ):
        # The open's sync fails, so no pool exists; the change that repairs the env starts one.
        from strata.notebook.dependencies import DependencyChangeResult, RequirementsImportResult

        async def _uv_fails(*args, **kwargs):
            return SimpleNamespace(
                success=False, error="boom", operation_log=EnvironmentOperationLog(command="uv")
            )

        def _touch_lock(notebook_dir) -> EnvironmentOperationLog:
            with (notebook_dir / "uv.lock").open("a") as lock:
                lock.write(f"\n# {change}\n")
            return EnvironmentOperationLog(command=f"uv {change}")

        def _fake_mutation(action):
            def _mutate(notebook_dir, package):
                return DependencyChangeResult(
                    success=True,
                    package=package,
                    action=action,
                    lockfile_changed=True,
                    operation_log=_touch_lock(notebook_dir),
                )

            return _mutate

        def _fake_import(notebook_dir, text):
            return RequirementsImportResult(
                success=True, lockfile_changed=True, operation_log=_touch_lock(notebook_dir)
            )

        monkeypatch.setattr("strata.notebook.dependencies.run_uv_command_streaming", _uv_fails)
        monkeypatch.setattr("strata.notebook.dependencies.add_dependency", _fake_mutation("add"))
        monkeypatch.setattr(
            "strata.notebook.dependencies.remove_dependency", _fake_mutation("remove")
        )
        monkeypatch.setattr("strata.notebook.session.import_requirements_text", _fake_import)
        notebook_dir = _notebook(tmp_path, "dep-change", environment=True)
        transport = httpx.ASGITransport(app=create_test_app())
        session = None
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                opened = await client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})
                assert opened.status_code == 200, opened.text
                session = manager.get_session(opened.json()["session_id"])
                assert session is not None
                await session.wait_for_environment_job()
                assert session.warm_pool is None

                base = f"/v1/notebooks/{session.id}"
                if change == "add":
                    changed = await client.post(f"{base}/dependencies", json={"package": "six"})
                elif change == "remove":
                    changed = await client.delete(f"{base}/dependencies/six")
                elif change == "requirements":
                    changed = await client.post(
                        f"{base}/environment/requirements.txt", json={"requirements": "six\n"}
                    )
                elif change == "mcp add":
                    await mcp_server._add_dependency(manager, session.id, "six")
                else:
                    await mcp_server._remove_dependency(manager, session.id, "six")
                if not change.startswith("mcp"):
                    assert changed.status_code == 200, changed.text
                assert session.warm_pool is not None
                await asyncio.gather(*session.warm_pool._background_tasks)

                ran = await client.post(f"{base}/cells/root/execute")

            assert ran.status_code == 200, ran.text
            assert ran.json()["status"] == "ready"
            assert ran.json()["execution_method"] == "warm"
        finally:
            if session is not None:
                await asyncio.gather(*manager.close_session(session.id))

    def test_only_the_mutation_starting_the_pool_is_exempt(self, config, manager, tmp_path):
        session = manager.open_notebook(
            _notebook(tmp_path, "mutating", environment=True), skip_initial_venv_sync=True
        )

        def _job() -> EnvironmentJobSnapshot:
            return EnvironmentJobSnapshot(
                id="job", action="sync", command="uv sync", status="running", started_at=0
            )

        running = _job()
        session.environment_job = running
        assert session._should_start_warm_pool(running)
        assert not session._should_start_warm_pool(_job())
        assert not session._should_start_warm_pool()

        session.environment_job = None
        session._synchronous_environment_mutation = "environment sync"
        assert session._should_start_warm_pool("environment sync")
        assert not session._should_start_warm_pool("add numpy")
        assert not session._should_start_warm_pool(running)

    def test_a_pool_size_of_zero_starts_no_pool(self, config, manager, tmp_path):
        config.notebook_warm_pool_size = 0
        session = manager.open_notebook(
            _notebook(tmp_path, "unpooled", environment=True), skip_initial_venv_sync=True
        )
        assert session.warm_pool is None

    async def test_the_sweep_loop_calls_the_manager(self, monkeypatch):
        import strata.server as server_module

        swept = asyncio.Event()

        class _Manager:
            async def sweep(self) -> None:
                swept.set()

        monkeypatch.setattr(server_module, "_NOTEBOOK_SESSION_SWEEP_SECONDS", 0.0)
        monkeypatch.setattr(notebook_routes, "_session_manager", _Manager())
        task = asyncio.create_task(server_module._notebook_session_sweep_loop())
        try:
            await asyncio.wait_for(swept.wait(), timeout=30)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


class TestIdleByActivity:
    async def test_a_tab_sending_only_keep_alives_is_closed_and_told(
        self, config, manager, clock, tmp_path
    ):
        session = manager.open_notebook(_notebook(tmp_path, "kept_alive"))
        socket, task = await _connect(session)

        for _ in range(4):
            clock.now += 300
            pings = len(socket.frames_of("error"))
            socket.send("notebook_sync")
            socket.send("ping")
            # The unknown frame answers with an error, after the sync before it.
            await _until(lambda pings=pings: len(socket.frames_of("error")) > pings)
        await manager.sweep()

        assert manager.list_sessions() == []
        await _until(task.done)
        assert [f["reason"] for f in socket.closed_frames()] == ["idle"]
        assert socket.closed == (1000, "Session closed")

    async def test_a_focus_keeps_a_session_open(self, config, manager, clock, tmp_path):
        session = manager.open_notebook(_notebook(tmp_path, "focused"))
        socket, task = await _connect(session)

        clock.now += 600
        socket.send("cell_focus", {"cell_id": "root"})
        await _until(lambda: session.last_accessed == clock.now)
        clock.now += 600
        await manager.sweep()
        assert manager.list_sessions() == [session.id]

        clock.now += 301
        await manager.sweep()
        assert manager.list_sessions() == []
        await _until(task.done)

    def test_a_rest_read_is_not_activity_and_an_edit_is(self, config, manager, clock, tmp_path):
        session = manager.open_notebook(_notebook(tmp_path, "rest"))
        opened_at = session.last_accessed
        client = TestClient(create_test_app())

        clock.now += 100
        assert client.get(f"/v1/notebooks/{session.id}/cells").status_code == 200
        assert session.last_accessed == opened_at

        response = client.put(f"/v1/notebooks/{session.id}/cells/root", json={"source": "x = 2"})
        assert response.status_code == 200
        assert session.last_accessed == clock.now


class TestCountAndMemory:
    async def test_over_the_limit_the_least_recently_used_goes_even_with_a_tab(
        self, config, manager, clock, tmp_path
    ):
        config.notebook_max_sessions = 2
        first = manager.open_notebook(_notebook(tmp_path, "first"))
        socket, task = await _connect(first)
        clock.now += 10
        second = manager.open_notebook(_notebook(tmp_path, "second"))
        clock.now += 10
        third = manager.open_notebook(_notebook(tmp_path, "third"))

        assert sorted(manager.list_sessions()) == sorted([second.id, third.id])
        await _until(task.done)
        assert [f["reason"] for f in socket.closed_frames()] == ["session_limit"]

    async def test_low_memory_closes_the_least_recently_used_idle_one_and_tells_it(
        self, config, manager, clock, memory, tmp_path
    ):
        config.notebook_session_min_available_mb = 500
        oldest = manager.open_notebook(_notebook(tmp_path, "oldest"))
        socket, task = await _connect(oldest)
        clock.now += 10
        newer = manager.open_notebook(_notebook(tmp_path, "newer"))

        memory.mb = 100
        closed: list[str] = []
        real_close = manager.close_session

        def close_and_recover(session_id, **kwargs):
            closed.append(session_id)
            memory.mb = 1000
            return real_close(session_id, **kwargs)

        manager.close_session = close_and_recover  # type: ignore[method-assign]
        await manager.relieve_memory_pressure()

        assert closed == [oldest.id]
        assert manager.list_sessions() == [newer.id]
        await _until(task.done)
        assert [f["reason"] for f in socket.closed_frames()] == ["memory"]

    async def test_low_memory_repeats_until_nothing_is_idle(
        self, config, manager, clock, memory, tmp_path
    ):
        config.notebook_session_min_available_mb = 500
        memory.mb = 100
        sessions = []
        for name in ("a", "b", "c"):
            sessions.append(manager.open_notebook(_notebook(tmp_path, name)))
            clock.now += 1
        running = await _hold_a_run(sessions[1])
        try:
            await manager.relieve_memory_pressure()
            assert manager.list_sessions() == [sessions[1].id]
        finally:
            running.cancel()

    def test_opening_past_the_threshold_closes_an_idle_one(self, config, manager, memory, tmp_path):
        config.notebook_session_min_available_mb = 500
        idle = manager.open_notebook(_notebook(tmp_path, "idle"))
        memory.mb = 100
        client = TestClient(create_test_app())

        response = client.post(
            "/v1/notebooks/open", json={"path": str(_notebook(tmp_path, "opened"))}
        )

        assert response.status_code == 200, response.text
        assert manager.list_sessions() == [response.json()["session_id"]]
        assert idle.id not in manager.list_sessions()

    async def test_a_soft_lock_or_a_quiesce_hold_keeps_a_session(
        self, config, manager, memory, tmp_path
    ):
        config.notebook_session_min_available_mb = 500
        config.notebook_cell_lock_seconds = 3600
        locked = manager.open_notebook(_notebook(tmp_path, "locked"))
        held = manager.open_notebook(_notebook(tmp_path, "held"))
        manager.open_notebook(_notebook(tmp_path, "idle"))
        locked.presence.record_edit("root", "alice")
        quiesce.begin(held.path.resolve(), 3600)
        memory.mb = 100

        await manager.relieve_memory_pressure()

        # The idle one goes; the lock and the hold keep the other two.
        assert sorted(manager.list_sessions()) == sorted([locked.id, held.id])


class TestRunningCell:
    async def test_a_session_running_a_cell_is_never_closed(
        self, config, manager, clock, memory, tmp_path
    ):
        session = manager.open_notebook(_notebook(tmp_path, "running"))
        running = await _hold_a_run(session)
        try:
            # Timeout, count and memory each close an idle session and pass it over.
            manager.open_notebook(_notebook(tmp_path, "idle"))
            clock.now += 10_000
            await manager.sweep()
            assert manager.list_sessions() == [session.id]

            config.notebook_max_sessions = 1
            manager.open_notebook(_notebook(tmp_path, "over_the_limit"))
            manager._evict_stale(making_room=True)
            assert manager.list_sessions() == [session.id]

            config.notebook_session_min_available_mb = 500
            memory.mb = 100
            manager.open_notebook(_notebook(tmp_path, "hungry"))
            await manager.relieve_memory_pressure()
            assert manager.list_sessions() == [session.id]

            response = TestClient(create_test_app()).post(f"/v1/notebooks/{session.id}/close")
            assert response.status_code == 409
            assert manager.list_sessions() == [session.id]
        finally:
            running.cancel()

    async def test_idleness_starts_when_the_run_ends(self, config, manager, clock, tmp_path):
        session = manager.open_notebook(_notebook(tmp_path, "long_run"))
        running = await _hold_a_run(session)
        clock.now += 10_000
        await manager.sweep()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)

        clock.now += 60
        await manager.sweep()
        assert manager.list_sessions() == [session.id]

        # The idle clock restarted at the end of the run, so it does expire.
        clock.now += 900
        await manager.sweep()
        assert manager.list_sessions() == []


class TestCloseRoute:
    def test_close_keeps_the_notebook_and_ends_the_session(self, config, manager, tmp_path):
        notebook_dir = _notebook(tmp_path, "closable")
        session = manager.open_notebook(notebook_dir)

        response = TestClient(create_test_app()).post(f"/v1/notebooks/{session.id}/close")

        assert response.status_code == 200
        assert response.json()["closed"] is True
        assert manager.list_sessions() == []
        assert (notebook_dir / "notebook.toml").is_file()

    def test_close_needs_the_scope_open_needs(self):
        from strata.notebook.scopes import NOTEBOOK_SCOPE_WRITE, required_scope_for_route

        close = required_scope_for_route("POST", "/v1/notebooks/{notebook_id}/close")

        assert close == required_scope_for_route("POST", "/v1/notebooks/open")
        assert close == NOTEBOOK_SCOPE_WRITE


class TestServerShutdown:
    async def test_every_connected_client_is_told_before_uvicorn_closes_it(
        self, config, manager, monkeypatch, tmp_path
    ):
        import uvicorn

        from strata.server import _ShutdownAnnouncingServer

        first, second = (manager.open_notebook(_notebook(tmp_path, n)) for n in ("one", "two"))
        connected = [await _connect(first), await _connect(second)]
        sockets = [socket for socket, _ in connected]
        reasons_when_uvicorn_shut_down: list[list[str]] = []

        async def uvicorn_shutdown(self, sockets_=None):
            reasons_when_uvicorn_shut_down.extend(
                [f["reason"] for f in s.closed_frames()] for s in sockets
            )

        monkeypatch.setattr(uvicorn.Server, "shutdown", uvicorn_shutdown)
        await _ShutdownAnnouncingServer(uvicorn.Config("strata.server:app")).shutdown()

        assert reasons_when_uvicorn_shut_down == [["shutdown"], ["shutdown"]]
        for socket, task in connected:
            await _until(task.done)
            assert socket.closed == (1000, "Session closed")


class TestServiceModeReuse:
    def test_open_reuses_only_the_callers_own_session(self, manager, tmp_path):
        notebook_dir = _notebook(tmp_path, "shared")
        alice = ("alice", "team")

        first = manager.open_notebook(notebook_dir, reuse_existing=True, opened_by=alice)
        again = manager.open_notebook(notebook_dir, reuse_existing=True, opened_by=alice)
        bob = manager.open_notebook(notebook_dir, reuse_existing=True, opened_by=("bob", "team"))

        assert again is first
        assert bob is not first

    def test_service_mode_reuses_by_principal_and_never_anonymously(self, config):
        from strata.auth import set_principal
        from strata.types import Principal

        config.deployment_mode = "service"
        set_principal(None)
        assert notebook_routes._reuse_open_session_by_path() == (False, None)
        set_principal(Principal(id="alice", tenant="team"))
        try:
            assert notebook_routes._reuse_open_session_by_path() == (True, ("alice", "team"))
        finally:
            set_principal(None)

        config.deployment_mode = "personal"
        assert notebook_routes._reuse_open_session_by_path() == (True, None)
