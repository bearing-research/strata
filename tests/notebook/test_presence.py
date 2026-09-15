"""Presence, cell focus and soft locks on a notebook session. Item 45."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from strata.notebook.presence import API_PRESENCE_SECONDS, SessionPresence
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from tests.notebook.e2e_fixtures import FakeNotebookWebSocket
from tests.notebook.e2e_fixtures import _reset_ws_globals as _reset


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class TestTheTables:
    def test_a_different_editor_is_held_off_until_the_window_passes(self):
        clock = _Clock()
        presence = SessionPresence(clock=clock)
        presence.record_edit("c1", "alice")

        assert presence.holder("c1", "bob", 5.0) == "alice"
        assert presence.holder("c1", "alice", 5.0) is None
        assert presence.holder("c2", "bob", 5.0) is None
        clock.now += 5.0
        assert presence.holder("c1", "bob", 5.0) is None

    def test_an_identity_is_one_entry_and_its_latest_focus_wins(self):
        clock = _Clock()
        presence = SessionPresence(clock=clock)
        presence.join("tab1", "alice")
        presence.join("tab2", "alice")
        presence.join("other", "bob")
        clock.now += 1
        presence.focus("tab2", "alice", "c2")
        clock.now += 1
        presence.focus("tab1", "alice", "c1")

        assert [(e["principal"], e["focused_cell_id"]) for e in presence.snapshot()] == [
            ("alice", "c1"),
            ("bob", None),
        ]

    def test_an_api_editor_is_present_until_it_goes_quiet(self):
        clock = _Clock()
        presence = SessionPresence(clock=clock)
        presence.api_edit("agent:claude", "c1")

        assert presence.snapshot()[0]["focused_cell_id"] == "c1"
        clock.now += API_PRESENCE_SECONDS
        assert presence.snapshot() == []


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

    def last_presence(self) -> dict | None:
        frames = self.frames_of("presence")
        return frames[-1]["payload"] if frames else None

    def errors(self) -> list[dict]:
        return [f["payload"] for f in self.frames_of("error")]


async def _until(predicate) -> None:
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


@pytest.fixture(autouse=True)
def _reset_ws_state():
    _reset()
    yield
    _reset()


@pytest.fixture
def session(tmp_path):
    from strata.notebook.routes import get_session_manager

    notebook_dir = create_notebook(tmp_path, "Shared")
    add_cell_to_notebook(notebook_dir, "root")
    write_cell(notebook_dir, "root", "x = 1")
    return get_session_manager().open_notebook(notebook_dir)


@pytest.fixture
def server_config(monkeypatch, tmp_path):
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.server import ServerState

    config = StrataConfig(artifact_dir=str(tmp_path / "artifacts"))
    # A window no test run can outlast, so "within the window" is not a race.
    config.notebook_cell_lock_seconds = 3600.0
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
        "x-strata-scopes": "notebook:read notebook:write",
    }


async def _connect(session, headers=None):
    from strata.notebook.ws import notebook_websocket

    socket = _LiveSocket(headers=headers)
    task = asyncio.create_task(notebook_websocket(cast(WebSocket, socket), session.id))
    await _until(lambda: socket.last_presence() is not None)
    return socket, task


async def _disconnect(socket, task) -> None:
    socket.close_client()
    await task


def _on_disk(session) -> str:
    return (session.path / "cells" / "root.py").read_text()


class TestTwoPrincipals:
    async def test_each_sees_the_other_and_which_cell_they_are_on(self, session, trusted_proxy):
        alice, alice_task = await _connect(session, _as("alice"))
        bob, bob_task = await _connect(session, _as("bob"))

        await _until(lambda: len(alice.last_presence()["principals"]) == 2)
        assert alice.last_presence()["you"] == "alice"
        assert bob.last_presence()["you"] == "bob"

        alice.send("cell_focus", {"cell_id": "root"})
        await _until(
            lambda: {"principal": "alice", "focused_cell_id": "root"}.items()
            <= next(
                e for e in bob.last_presence()["principals"] if e["principal"] == "alice"
            ).items()
        )

        await _disconnect(bob, bob_task)
        await _until(
            lambda: [e["principal"] for e in alice.last_presence()["principals"]] == ["alice"]
        )
        await _disconnect(alice, alice_task)

    async def test_a_competing_edit_is_refused_with_the_holder_and_accepted_with_force(
        self, session, trusted_proxy
    ):
        alice, alice_task = await _connect(session, _as("alice"))
        bob, bob_task = await _connect(session, _as("bob"))

        alice.send("cell_source_update", {"cell_id": "root", "source": "x = 'alice'"})
        await _until(lambda: _on_disk(session).strip() == "x = 'alice'")

        bob.send("cell_source_update", {"cell_id": "root", "source": "x = 'bob'"})
        await _until(lambda: bob.errors())
        (refused,) = bob.errors()
        assert (refused["code"], refused["cell_id"], refused["held_by"]) == (
            "cell_locked",
            "root",
            "alice",
        )
        assert _on_disk(session).strip() == "x = 'alice'"

        # Alice keeps editing her own cell without contention.
        alice.send("cell_source_update", {"cell_id": "root", "source": "x = 'alice 2'"})
        await _until(lambda: _on_disk(session).strip() == "x = 'alice 2'")
        assert alice.errors() == []

        bob.send("cell_source_update", {"cell_id": "root", "source": "x = 'bob'", "force": True})
        await _until(lambda: _on_disk(session).strip() == "x = 'bob'")
        assert len(bob.errors()) == 1

        await _disconnect(alice, alice_task)
        await _disconnect(bob, bob_task)


class TestPersonalMode:
    async def test_one_user_in_two_tabs_never_contends(self, session, server_config):
        first, first_task = await _connect(session)
        second, second_task = await _connect(session)

        first.send("cell_source_update", {"cell_id": "root", "source": "x = 1"})
        await _until(lambda: _on_disk(session).strip() == "x = 1")
        second.send("cell_source_update", {"cell_id": "root", "source": "x = 2"})
        await _until(lambda: _on_disk(session).strip() == "x = 2")

        assert first.errors() == second.errors() == []
        assert [e["principal"] for e in second.last_presence()["principals"]] == ["local"]
        await _disconnect(first, first_task)
        await _disconnect(second, second_task)

    async def test_an_agent_editing_over_rest_is_present_and_holds_the_cell(
        self, session, server_config
    ):
        from strata.notebook.routes import UpdateCellSourceRequest, update_cell_source

        browser, browser_task = await _connect(session)

        await update_cell_source(
            session.id,
            session,
            "root",
            UpdateCellSourceRequest(source="x = 'agent'", author="agent:claude"),
        )
        await _until(
            lambda: {"principal": "agent:claude", "focused_cell_id": "root"}.items()
            <= next(
                (
                    e
                    for e in browser.last_presence()["principals"]
                    if e["principal"] == "agent:claude"
                ),
                {},
            ).items()
        )

        browser.send("cell_source_update", {"cell_id": "root", "source": "x = 'me'"})
        await _until(lambda: browser.errors())
        assert browser.errors()[0]["held_by"] == "agent:claude"

        # And the agent, over REST, is told the same about the browser's edit.
        browser.send("cell_source_update", {"cell_id": "root", "source": "x = 'me'", "force": True})
        await _until(lambda: _on_disk(session).strip() == "x = 'me'")
        with pytest.raises(HTTPException) as refused:
            await update_cell_source(
                session.id,
                session,
                "root",
                UpdateCellSourceRequest(source="x = 'agent again'", author="agent:claude"),
            )
        assert refused.value.status_code == 409
        assert refused.value.detail["held_by"] == "local"
        await update_cell_source(
            session.id,
            session,
            "root",
            UpdateCellSourceRequest(source="x = 'agent again'", author="agent:claude", force=True),
        )
        assert _on_disk(session).strip() == "x = 'agent again'"

        await _disconnect(browser, browser_task)
