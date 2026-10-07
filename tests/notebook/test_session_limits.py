"""Session limits: warm pool size, idle timeout by user activity, count, memory, close."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import cast

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from strata.notebook import quiesce
from strata.notebook import routes as notebook_routes
from strata.notebook.session import SessionManager
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
