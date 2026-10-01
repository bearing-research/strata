"""E2E tests: source annotations beat persisted notebook config over REST and WebSocket."""

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


def _put_notebook_env(client: TestClient, session_id: str, env: dict[str, str]) -> dict:
    response = client.put(
        f"/v1/notebooks/{session_id}/env",
        json={"env": env},
    )
    assert response.status_code == 200
    return response.json()


def _put_notebook_timeout(client: TestClient, session_id: str, timeout: float) -> dict:
    response = client.put(
        f"/v1/notebooks/{session_id}/timeout",
        json={"timeout": timeout},
    )
    assert response.status_code == 200
    return response.json()


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


def _put_notebook_workers(
    client: TestClient,
    session_id: str,
    workers: list[dict],
) -> dict:
    response = client.put(
        f"/v1/notebooks/{session_id}/workers",
        json={"workers": workers},
    )
    assert response.status_code == 200
    return response.json()


def _mount_payload(name: str, path: Path, *, mode: str = "ro") -> dict[str, str]:
    return {
        "name": name,
        "uri": path.resolve().as_uri(),
        "mode": mode,
    }


def _update_source_and_wait(ws, cell_id: str, source: str) -> None:
    ws.update_source(cell_id, source)
    ws.receive_until("dag_update")
    ws.clear()


def _sync_cell(ws, cell_id: str) -> dict:
    state = ws.sync()
    return next(cell for cell in state["payload"]["cells"] if cell["id"] == cell_id)


class TestAnnotationOverrides:
    """Source annotations should beat saved runtime config."""

    def test_env_annotation_beats_notebook_env(self, setup):
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell(
            "c1",
            "import os\nvalue = os.getenv('APP_MODE')",
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            notebook_env = _put_notebook_env(client, sid, {"APP_MODE": "notebook"})
            assert notebook_env["env"] == {"APP_MODE": "notebook"}

            with ws_connect(client, sid) as ws:
                _update_source_and_wait(
                    ws,
                    "c1",
                    "# @env APP_MODE=annotated\nimport os\nvalue = os.getenv('APP_MODE')",
                )

                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"
                assert result["payload"]["outputs"]["value"]["preview"] == "annotated"

                cell = _sync_cell(ws, "c1")
                assert cell["env"] == {"APP_MODE": "notebook"}
                assert cell["annotations"]["env"] == {"APP_MODE": "annotated"}

    def test_timeout_annotation_beats_notebook_timeout(self, setup):
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell(
            "c1",
            "import time\ntime.sleep(0.05)\nvalue = 'done'",
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            notebook_timeout = _put_notebook_timeout(client, sid, 0.01)
            assert notebook_timeout["timeout"] == 0.01

            with ws_connect(client, sid) as ws:
                _update_source_and_wait(
                    ws,
                    "c1",
                    "# @timeout 5\nimport time\ntime.sleep(0.05)\nvalue = 'done'",
                )

                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"
                assert result["payload"]["outputs"]["value"]["preview"] == "done"

                cell = _sync_cell(ws, "c1")
                assert cell["timeout"] == 0.01
                assert cell["annotations"]["timeout"] == 5.0

    def test_mount_annotation_beats_notebook_mount(self, setup):
        client, tmp = setup
        notebook_dir = tmp / "notebook-data"
        notebook_dir.mkdir()
        (notebook_dir / "data.txt").write_text("notebook", encoding="utf-8")

        annotated_dir = tmp / "annotated-data"
        annotated_dir.mkdir()
        (annotated_dir / "data.txt").write_text("annotated", encoding="utf-8")

        nb = NotebookBuilder(tmp).add_cell(
            "c1",
            'value = (raw_data / "data.txt").read_text()',
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            _put_notebook_mounts(client, sid, [_mount_payload("raw_data", notebook_dir)])

            with ws_connect(client, sid) as ws:
                _update_source_and_wait(
                    ws,
                    "c1",
                    (
                        f"# @mount raw_data {annotated_dir.resolve().as_uri()} ro\n"
                        'value = (raw_data / "data.txt").read_text()'
                    ),
                )

                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"
                assert result["payload"]["outputs"]["value"]["preview"] == "annotated"

                cell = _sync_cell(ws, "c1")
                assert cell["mounts"][0]["uri"] == notebook_dir.resolve().as_uri()
                assert cell["annotations"]["mounts"][0]["uri"] == annotated_dir.resolve().as_uri()

    def test_worker_annotation_beats_notebook_worker(self, setup, notebook_executor_server):
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "value = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            worker_catalog = _put_notebook_workers(
                client,
                sid,
                [
                    {
                        "name": "gpu-http",
                        "backend": "executor",
                        "runtime_id": "gpu-http-a100",
                        "config": {"url": notebook_executor_server["execute_url"]},
                    }
                ],
            )
            assert any(worker["name"] == "gpu-http" for worker in worker_catalog["workers"])

            worker_response = client.put(
                f"/v1/notebooks/{sid}/worker",
                json={"worker": "gpu-http"},
            )
            assert worker_response.status_code == 200
            assert worker_response.json()["worker"] == "gpu-http"

            with ws_connect(client, sid) as ws:
                _update_source_and_wait(
                    ws,
                    "c1",
                    "# @worker local\nvalue = 1",
                )

                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"
                assert result["payload"]["outputs"]["value"]["preview"] == 1
                assert result["payload"]["execution_method"] != "executor"
                assert result["payload"].get("remote_worker") in (None, "")

                cell = _sync_cell(ws, "c1")
                assert cell["worker"] == "gpu-http"
                assert cell["annotations"]["worker"] == "local"
                assert cell["remote_worker"] is None
