"""A cell run on a shared server stores its outputs under the member who ran it.

Driven through the real WebSocket endpoint and the real REST app with trusted-proxy
auth, so the principal comes from the same place it does in service mode.
"""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from fastapi import WebSocket, WebSocketDisconnect

from strata.auth import principal_context
from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from strata.types import Principal
from tests.notebook.e2e_fixtures import FakeNotebookWebSocket
from tests.notebook.e2e_fixtures import _reset_ws_globals as _reset

SCOPES = "notebook:read notebook:write notebook:execute"


class _LiveSocket(FakeNotebookWebSocket):
    """A fake socket that stays open until the test closes it."""

    def __init__(self, headers=None):
        super().__init__(headers=headers)
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def receive_text(self) -> str:
        item = await self._queue.get()
        if item is None:
            raise WebSocketDisconnect(code=1000)
        return item

    def send(self, msg_type: str, payload: dict) -> None:
        self._queue.put_nowait(json.dumps({"type": msg_type, "seq": 1, "payload": payload}))

    def close_client(self) -> None:
        self._queue.put_nowait(None)

    def finished(self, cell_id: str) -> bool:
        return any(
            f["payload"].get("cell_id") == cell_id
            and f["payload"].get("status") in ("ready", "error")
            for f in self.frames_of("cell_status")
        )


async def _until(predicate) -> None:
    for _ in range(6000):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


@pytest.fixture(autouse=True)
def _reset_ws_state():
    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def _close_opened_sessions():
    """The manager is a module global; a session left open counts against later tests."""
    from strata.notebook.routes import get_session_manager

    manager = get_session_manager()
    before = set(manager.list_sessions())
    yield
    for session_id in manager.list_sessions():
        if session_id not in before:
            manager.close_session(session_id)


@pytest.fixture
def session(tmp_path):
    from strata.notebook.routes import get_session_manager

    notebook_dir = create_notebook(tmp_path, "Shared")
    add_cell_to_notebook(notebook_dir, "root")
    write_cell(notebook_dir, "root", 'x = 1\ndisplay(Markdown("# hi"))')
    add_cell_to_notebook(notebook_dir, "leaf", after_cell_id="root")
    write_cell(notebook_dir, "leaf", "y = x + 1\ny")
    return get_session_manager().open_notebook(notebook_dir)


@pytest.fixture
def server_config(monkeypatch, tmp_path):
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.server import ServerState

    config = StrataConfig(artifact_dir=str(tmp_path / "artifacts"))
    monkeypatch.setattr(server_module, "_state", ServerState(config), raising=False)
    return config


@pytest.fixture
def trusted_proxy(server_config):
    server_config.auth_mode = "trusted_proxy"
    server_config.proxy_token = "sekrit"
    return server_config


def _as(principal: str) -> dict[str, str]:
    return {
        "x-strata-proxy-token": "sekrit",
        "x-strata-principal": principal,
        "x-strata-scopes": SCOPES,
    }


def _stored(session) -> dict[str, str | None]:
    """Every artifact the notebook stored, by id, with the principal it records."""
    store = session.get_artifact_manager().artifact_store
    rows = store.list_latest_by_id_prefix(f"nb_{session.notebook_state.id}_")
    return {row.id: row.principal for row in rows}


def _has_variable_and_display(stored: dict[str, str | None]) -> bool:
    return any(i.endswith("_var_x") for i in stored) and any("__display__" in i for i in stored)


async def _connect(session, headers=None):
    from strata.notebook.ws import notebook_websocket

    socket = _LiveSocket(headers=headers)
    task = asyncio.create_task(notebook_websocket(cast(WebSocket, socket), session.id))
    await _until(lambda: socket.frames_of("presence"))
    return socket, task


async def _disconnect(socket, task) -> None:
    socket.close_client()
    await task


class TestOverTheWebSocket:
    async def test_the_member_who_runs_a_cell_is_recorded_not_the_one_watching(
        self, session, trusted_proxy
    ):
        alice, alice_task = await _connect(session, _as("alice"))
        bob, bob_task = await _connect(session, _as("bob"))

        alice.send("cell_execute", {"cell_id": "root"})
        await _until(lambda: bob.finished("root"))

        stored = _stored(session)
        assert _has_variable_and_display(stored), stored
        assert set(stored.values()) == {"alice"}, stored
        await _disconnect(bob, bob_task)
        await _disconnect(alice, alice_task)

    async def test_run_all_records_the_member_on_every_cell(self, session, trusted_proxy):
        bob, bob_task = await _connect(session, _as("bob"))

        bob.send("notebook_run_all", {})
        await _until(lambda: bob.finished("leaf"))

        stored = _stored(session)
        assert any("_cell_leaf_" in i for i in stored), stored
        assert _has_variable_and_display(stored), stored
        assert set(stored.values()) == {"bob"}, stored
        await _disconnect(bob, bob_task)

    async def test_personal_mode_records_no_one(self, session, server_config):
        socket, task = await _connect(session)

        socket.send("cell_execute", {"cell_id": "root"})
        await _until(lambda: socket.finished("root"))

        stored = _stored(session)
        assert _has_variable_and_display(stored), stored
        assert set(stored.values()) == {None}, stored
        await _disconnect(socket, task)


class TestOverRest:
    def test_the_caller_of_the_execute_route_is_recorded(self, session, trusted_proxy):
        from fastapi.testclient import TestClient

        from strata.server import app

        response = TestClient(app).post(
            f"/v1/notebooks/{session.id}/cells/root/execute", headers=_as("carol")
        )

        assert response.status_code == 200, response.text
        stored = _stored(session)
        assert _has_variable_and_display(stored), stored
        assert set(stored.values()) == {"carol"}, stored


class TestTheDefault:
    def _store(self, manager, **kwargs):
        return manager.store_cell_output(
            cell_id="c1",
            variable_name="v",
            blob_data=b"[1]",
            content_type="json/object",
            provenance_hash="a1" * 32,
            **kwargs,
        )

    def test_a_run_task_keeps_the_caller_after_the_request_has_returned(self, tmp_path):
        manager = NotebookArtifactManager("nb", artifact_dir=tmp_path)
        release = asyncio.Event()

        async def run():
            await release.wait()
            return self._store(manager)

        async def scenario():
            with principal_context(Principal(id="alice")):
                task = asyncio.create_task(run())
            release.set()
            return await task

        assert asyncio.run(scenario()).principal == "alice"

    def test_a_pulled_result_keeps_its_publisher_even_when_unrecorded(self, tmp_path):
        manager = NotebookArtifactManager("nb", artifact_dir=tmp_path)

        with principal_context(Principal(id="alice")):
            stored = self._store(manager, principal=None)

        assert stored.principal is None
