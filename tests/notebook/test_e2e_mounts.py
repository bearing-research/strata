"""E2E tests: notebook mount configuration and live execution.

Exercises mount persistence through the notebook APIs and verifies that
mounted paths behave correctly through the real WebSocket execution flow.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    create_test_app,
    execute_cell_and_wait,
    open_notebook_session,
    ws_connect,
)


@pytest.fixture
def setup():
    """Create app, client, and temp directory."""
    app = create_test_app()
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:
        yield client, Path(tmpdir)


def _mount_payload(name: str, path: Path, *, mode: str = "ro") -> dict[str, str]:
    return {
        "name": name,
        "uri": path.resolve().as_uri(),
        "mode": mode,
    }


def _put_notebook_mounts(
    client: TestClient,
    session_id: str,
    mounts: list[dict[str, str]],
) -> dict:
    response = client.put(
        f"/v1/notebooks/{session_id}/mounts",
        json={"mounts": mounts},
    )
    assert response.status_code == 200
    return response.json()


def _list_cells(client: TestClient, session_id: str) -> list[dict]:
    response = client.get(f"/v1/notebooks/{session_id}/cells")
    assert response.status_code == 200
    return response.json()["cells"]


def _cell(cells: list[dict], cell_id: str) -> dict:
    return next(cell for cell in cells if cell["id"] == cell_id)


class TestNotebookMountDefaults:
    """Notebook-level mounts should persist and inject into live execution."""

    def test_notebook_mount_reads_file_over_websocket(self, setup):
        client, tmp = setup
        data_dir = tmp / "mounted-data"
        data_dir.mkdir()
        (data_dir / "data.txt").write_text("hello mount", encoding="utf-8")

        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", 'value = (raw_data / "data.txt").read_text()')
            .add_cell("c2", "result = value.upper()", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            mounts = _put_notebook_mounts(
                client,
                sid,
                [_mount_payload("raw_data", data_dir)],
            )
            assert mounts["mounts"][0]["name"] == "raw_data"
            assert mounts["cells"][0]["mounts"][0]["name"] == "raw_data"

            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"
                assert result["payload"]["outputs"]["value"]["preview"] == "hello mount"

                cells = _list_cells(client, sid)
                c1 = _cell(cells, "c1")
                assert c1["status"] == "ready"
                assert c1["mounts"][0]["name"] == "raw_data"
                assert c1["mounts"][0]["uri"] == data_dir.resolve().as_uri()

    def test_force_rerun_keeps_mount_injected(self, setup):
        client, tmp = setup
        data_dir = tmp / "mounted-data"
        data_dir.mkdir()
        (data_dir / "data.txt").write_text("warm mount", encoding="utf-8")

        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", 'value = (raw_data / "data.txt").read_text()')
            .add_cell("c2", "result = value.upper()", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            _put_notebook_mounts(client, sid, [_mount_payload("raw_data", data_dir)])

            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                execute_cell_and_wait(ws, "c2")
                ws.clear()

                ws.execute_force("c1")

                result = None
                while True:
                    msg = ws.receive()
                    if (
                        msg["type"] in ("cell_output", "cell_error")
                        and msg["payload"].get("cell_id") == "c1"
                    ):
                        result = msg
                    if (
                        msg["type"] == "cell_status"
                        and msg["payload"].get("cell_id") == "c1"
                        and msg["payload"].get("status") in ("ready", "error")
                    ):
                        break

                assert result is not None
                assert result["type"] == "cell_output"
                assert result["payload"]["outputs"]["value"]["preview"] == "warm mount"
                assert result["payload"].get("cache_hit") is not True


class TestMountInvalidation:
    """Changing mount config should invalidate dependent cells."""

    def test_updating_notebook_mount_invalidates_and_reruns_with_new_data(self, setup):
        client, tmp = setup
        first_dir = tmp / "first-data"
        first_dir.mkdir()
        (first_dir / "data.txt").write_text("hello", encoding="utf-8")

        second_dir = tmp / "second-data"
        second_dir.mkdir()
        (second_dir / "data.txt").write_text("goodbye", encoding="utf-8")

        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", 'value = (raw_data / "data.txt").read_text()')
            .add_cell("c2", "result = value.upper()", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            _put_notebook_mounts(client, sid, [_mount_payload("raw_data", first_dir)])

            with ws_connect(client, sid) as ws:
                first_result = execute_cell_and_wait(ws, "c2")
                assert first_result["type"] == "cell_output"
                assert first_result["payload"]["outputs"]["result"]["preview"] == "HELLO"

                updated = _put_notebook_mounts(
                    client,
                    sid,
                    [_mount_payload("raw_data", second_dir)],
                )
                assert updated["mounts"][0]["uri"] == second_dir.resolve().as_uri()

                state = ws.sync()
                cells = state["payload"]["cells"]
                assert _cell(cells, "c1")["status"] == "idle"
                # c2 ran; repointing c1's mount invalidated c2's cached result
                # via its stale upstream → STALE, not idle (#361).
                assert _cell(cells, "c2")["status"] == "stale"

                ws.clear()
                rerun = execute_cell_and_wait(ws, "c2")
                assert rerun["type"] == "cell_output"
                assert rerun["payload"]["outputs"]["result"]["preview"] == "GOODBYE"

                upstream = next(
                    message
                    for message in ws.messages
                    if message["type"] == "cell_output"
                    and message["payload"].get("cell_id") == "c1"
                )
                assert upstream["payload"]["outputs"]["value"]["preview"] == "goodbye"
