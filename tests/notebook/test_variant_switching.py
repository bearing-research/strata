"""Switching and adding variants over the WebSocket and REST: what each answers and changes."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from strata.notebook.ws import (
    _ensure_execution_state,
    _handle_variant_add,
    _handle_variant_set_active,
    _notebook_connections,
    forget_notebook_execution_state,
)

SWITCH = [("a", "# @variant g a\nx = 1\n"), ("b", "# @variant g b\nx = 2\n"), ("r", "y = x\n")]


def _notebook(tmp_path: Path) -> Path:
    notebook_dir = create_notebook(tmp_path, "Variants", initialize_environment=False)
    previous: str | None = None
    for cell_id, source in SWITCH:
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=previous)
        write_cell(notebook_dir, cell_id, source)
        previous = cell_id
    return notebook_dir


def _active_in_toml(notebook_dir: Path) -> dict[str, str | None]:
    data = tomllib.loads((notebook_dir / "notebook.toml").read_text())
    return {g["group"]: g.get("active") for g in data.get("variant_group", [])}


class _Socket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    def frames(self, kind: str) -> list[dict]:
        return [frame["payload"] for frame in self.sent if frame["type"] == kind]


@pytest.fixture
def connected(tmp_path):
    """A session on the switch notebook, with a socket that receives its broadcasts."""
    notebook_dir = _notebook(tmp_path)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    socket = _Socket()
    _notebook_connections[session.id] = [socket]
    try:
        yield session, socket, _ensure_execution_state(session.id)
    finally:
        _notebook_connections.pop(session.id, None)
        forget_notebook_execution_state(session.id)


def _producer_of_x(session: NotebookSession) -> str:
    reader = session.notebook_state.get_cell("r")
    assert reader is not None
    (producer,) = reader.upstream_ids
    return producer


class TestSetActiveOverTheWebSocket:
    async def test_switching_moves_the_producer_and_says_so(self, connected):
        session, socket, state = connected
        assert _producer_of_x(session) == "a"

        await _handle_variant_set_active(
            socket, session, {"group": "g", "name": "b"}, state, session.id
        )

        assert socket.frames("error") == []
        assert _producer_of_x(session) == "b"
        assert _active_in_toml(session.path) == {"g": "b"}
        (update,) = socket.frames("dag_update")
        flags = {c["id"]: c["variant_active"] for c in update["cells"] if c["variant_group"]}
        assert flags == {"a": False, "b": True}
        edges = [(e["from_cell_id"], e["to_cell_id"], e["variable"]) for e in update["edges"]]
        assert edges == [("b", "r", "x")]

    async def test_a_missing_name_is_an_error_and_changes_nothing(self, connected):
        session, socket, state = connected

        await _handle_variant_set_active(socket, session, {"group": "g"}, state, session.id)

        assert [e["error"] for e in socket.frames("error")] == ["Missing group or name"]
        assert _producer_of_x(session) == "a"
        assert socket.frames("dag_update") == []

    async def test_in_sweep_mode_the_pointer_is_left_alone(self, connected):
        session, socket, state = connected
        session.set_variant_mode("g", "sweep")
        before = (session.path / "notebook.toml").read_text()

        await _handle_variant_set_active(
            socket, session, {"group": "g", "name": "b"}, state, session.id
        )

        assert socket.sent == []
        assert (session.path / "notebook.toml").read_text() == before
        (group,) = session.notebook_state.variant_groups
        assert group.active_name == "a"


class TestAddOverTheWebSocket:
    async def test_the_new_variant_is_active_and_sent_whole(self, connected):
        session, socket, state = connected

        await _handle_variant_add(socket, session, {"group": "g"}, state, session.id)

        assert socket.frames("error") == []
        (group,) = session.notebook_state.variant_groups
        assert [m.name for m in group.members] == ["a", "b", "a_copy"]
        assert group.active_name == "a_copy"
        new_id = group.active_cell_id
        assert _producer_of_x(session) == new_id
        # A new cell: only a whole notebook_state carries it to the client.
        (sent,) = socket.frames("notebook_state")
        assert new_id in [cell["id"] for cell in sent["cells"]]
        assert sent["dag"]["variant_groups"][0]["active_name"] == "a_copy"

    async def test_an_unknown_group_is_an_error(self, connected):
        session, socket, state = connected

        await _handle_variant_add(socket, session, {"group": "nope"}, state, session.id)

        assert [e["error"] for e in socket.frames("error")] == [
            "Variant group 'nope' does not exist"
        ]
        assert len(session.notebook_state.cells) == 3

    async def test_a_missing_group_is_an_error(self, connected):
        session, socket, state = connected

        await _handle_variant_add(socket, session, {}, state, session.id)

        assert [e["error"] for e in socket.frames("error")] == ["Missing group"]
        assert len(session.notebook_state.cells) == 3


class TestOverRest:
    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient

        from tests.notebook.e2e_fixtures import create_test_app, open_notebook_session

        notebook_dir = _notebook(tmp_path)
        client = TestClient(create_test_app())
        with open_notebook_session(client, notebook_dir) as (session_id, session):
            yield client, f"/v1/notebooks/{session_id}/variant-groups/g", session

    def test_put_active_switches_the_group(self, client):
        http, url, session = client

        response = http.put(url, json={"active": "b"})

        assert response.status_code == 200, response.text
        assert response.json()["variant_groups"][0]["active_name"] == "b"
        assert _active_in_toml(session.path) == {"g": "b"}
        assert _producer_of_x(session) == "b"

    def test_post_adds_an_active_copy(self, client):
        http, url, session = client

        response = http.post(f"{url}/variants")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["new_variant_name"] == "a_copy"
        assert body["new_cell_id"] in [cell["id"] for cell in body["cells"]]
        assert body["variant_groups"][0]["active_name"] == "a_copy"

    def test_adding_to_an_unknown_group_is_not_found(self, client):
        http, url, _session = client

        response = http.post(url.replace("/g", "/nope") + "/variants")

        assert response.status_code == 404
        assert response.json()["detail"] == "Variant group 'nope' does not exist"
