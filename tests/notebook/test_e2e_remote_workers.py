"""E2E tests: service-managed remote notebook workers.

Exercises the full notebook flow through the core service app:
admin worker registry -> notebook assignment -> WebSocket execution.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.config import StrataConfig
from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    execute_cell_and_wait,
    ws_connect,
)


@pytest.fixture
def service_mode_notebook_client(tmp_path):
    """Create a core service-mode app client with notebook routes enabled."""
    import strata.server as server_module
    from strata.artifact_store import get_artifact_store, reset_artifact_store
    from strata.notebook.routes import get_session_manager
    from strata.notebook.ws import _notebook_connections, _notebook_execution_state
    from strata.server import ServerState, app
    from strata.transforms.build_store import get_build_store, reset_build_store
    from strata.transforms.registry import (
        TransformRegistry,
        reset_transform_registry,
        set_transform_registry,
    )

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    config = StrataConfig(
        host="127.0.0.1",
        port=8765,
        deployment_mode="service",
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
        notebook_storage_dir=tmp_path,
        transforms_config={"enabled": True},
    )

    reset_artifact_store()
    reset_transform_registry()
    reset_build_store()
    set_transform_registry(TransformRegistry.from_config(config.transforms_config))
    assert config.artifact_dir is not None
    get_artifact_store(config.artifact_dir)
    get_build_store(config.artifact_dir / "artifacts.sqlite")

    session_manager = get_session_manager()
    for session_id in session_manager.list_sessions():
        session_manager.close_session(session_id)
    _notebook_connections.clear()
    _notebook_execution_state.clear()

    original_state = server_module._state
    server_module._state = ServerState(config)

    client = TestClient(app)
    try:
        yield client, tmp_path
    finally:
        client.close()
        for session_id in session_manager.list_sessions():
            session_manager.close_session(session_id)
        _notebook_connections.clear()
        _notebook_execution_state.clear()
        server_module._state = original_state
        reset_artifact_store()
        reset_transform_registry()
        reset_build_store()


def _open_notebook_via_api(client: TestClient, notebook_dir) -> str:
    opened = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})
    assert opened.status_code == 200
    return opened.json()["session_id"]


def _create_admin_worker(
    client: TestClient,
    *,
    name: str,
    runtime_id: str,
    config: dict[str, str],
    enabled: bool = True,
) -> None:
    created = client.post(
        "/v1/admin/notebook-workers",
        json={
            "name": name,
            "backend": "executor",
            "runtime_id": runtime_id,
            "config": config,
            "enabled": enabled,
        },
    )
    assert created.status_code == 200


def test_admin_managed_http_worker_executes_over_websocket(
    service_mode_notebook_client,
    notebook_executor_server,
):
    client, tmp = service_mode_notebook_client
    notebook = NotebookBuilder(tmp, "service_remote_http").add_cell("c1", "x = 1")

    _create_admin_worker(
        client,
        name="gpu-http",
        runtime_id="gpu-http-a100",
        config={"url": notebook_executor_server["execute_url"]},
    )

    session_id = _open_notebook_via_api(client, notebook.path)

    workers = client.get(f"/v1/notebooks/{session_id}/workers")
    assert workers.status_code == 200
    worker = next(item for item in workers.json()["workers"] if item["name"] == "gpu-http")
    assert worker["source"] == "server"
    assert worker["allowed"] is True

    assigned = client.put(
        f"/v1/notebooks/{session_id}/worker",
        json={"worker": "gpu-http"},
    )
    assert assigned.status_code == 200
    assert assigned.json()["worker"] == "gpu-http"

    with ws_connect(client, session_id) as ws:
        result = execute_cell_and_wait(ws, "c1")

        assert result["type"] == "cell_output"
        assert result["payload"]["execution_method"] == "executor"
        assert result["payload"]["remote_worker"] == "gpu-http"
        assert result["payload"]["remote_transport"] == "direct"
        assert result["payload"]["outputs"]["x"]["preview"] == 1

        state = ws.sync()
        cell = next(cell for cell in state["payload"]["cells"] if cell["id"] == "c1")
        assert cell["status"] == "ready"
        assert cell["worker"] == "gpu-http"
        assert cell["remote_worker"] == "gpu-http"
        assert cell["remote_transport"] == "direct"


def test_admin_disabled_worker_blocks_existing_notebook_execution_over_websocket(
    service_mode_notebook_client,
    notebook_executor_server,
):
    client, tmp = service_mode_notebook_client
    notebook = NotebookBuilder(tmp, "service_remote_blocked").add_cell("c1", "x = 1")

    _create_admin_worker(
        client,
        name="gpu-http",
        runtime_id="gpu-http-a100",
        config={"url": notebook_executor_server["execute_url"]},
    )

    session_id = _open_notebook_via_api(client, notebook.path)

    assigned = client.put(
        f"/v1/notebooks/{session_id}/worker",
        json={"worker": "gpu-http"},
    )
    assert assigned.status_code == 200

    disabled = client.patch(
        "/v1/admin/notebook-workers/gpu-http",
        json={"enabled": False},
    )
    assert disabled.status_code == 200

    workers = client.get(f"/v1/notebooks/{session_id}/workers")
    assert workers.status_code == 200
    worker = next(item for item in workers.json()["workers"] if item["name"] == "gpu-http")
    assert worker["enabled"] is False
    assert worker["allowed"] is False

    with ws_connect(client, session_id) as ws:
        result = execute_cell_and_wait(ws, "c1")

        assert result["type"] == "cell_error"
        assert "disabled by server policy" in result["payload"]["error"]

        statuses = [
            message["payload"]["status"]
            for message in ws.messages_of_type("cell_status")
            if message["payload"]["cell_id"] == "c1"
        ]
        assert statuses[-1] == "error"

        state = ws.sync()
        cell = next(cell for cell in state["payload"]["cells"] if cell["id"] == "c1")
        assert cell["status"] == "error"
        assert cell["worker"] == "gpu-http"


def test_admin_managed_signed_worker_executes_over_websocket(
    service_mode_notebook_client,
    notebook_executor_server,
    notebook_build_server,
):
    client, tmp = service_mode_notebook_client
    notebook = NotebookBuilder(tmp, "service_remote_signed").add_cell("c1", "x = 1")

    _create_admin_worker(
        client,
        name="gpu-http-signed",
        runtime_id="gpu-http-signed-a100",
        config={
            "url": notebook_executor_server["execute_url"],
            "transport": "signed",
            "strata_url": notebook_build_server["base_url"],
        },
    )

    session_id = _open_notebook_via_api(client, notebook.path)
    assigned = client.put(
        f"/v1/notebooks/{session_id}/worker",
        json={"worker": "gpu-http-signed"},
    )
    assert assigned.status_code == 200

    with ws_connect(client, session_id) as ws:
        first = execute_cell_and_wait(ws, "c1")
        assert first["type"] == "cell_output"
        assert first["payload"]["execution_method"] == "executor"
        assert first["payload"]["remote_worker"] == "gpu-http-signed"
        assert first["payload"]["remote_transport"] == "signed"
        assert isinstance(first["payload"]["remote_build_id"], str)
        assert first["payload"]["remote_build_state"] == "ready"
        assert first["payload"]["outputs"]["x"]["preview"] == 1

        state = ws.sync()
        cell = next(cell for cell in state["payload"]["cells"] if cell["id"] == "c1")
        assert cell["status"] == "ready"
        assert cell["worker"] == "gpu-http-signed"
        assert cell["remote_worker"] == "gpu-http-signed"
        assert cell["remote_transport"] == "signed"
        assert isinstance(cell["remote_build_id"], str)
        assert cell["remote_build_state"] == "ready"


def test_admin_managed_signed_worker_preserves_exported_class_instances(
    service_mode_notebook_client,
    notebook_executor_server,
    notebook_build_server,
):
    client, tmp = service_mode_notebook_client
    notebook = (
        NotebookBuilder(tmp, "service_remote_signed_class_instances")
        .add_cell(
            "c1",
            """
class Person:
    name = "John"
    age = 20

    def __str__(self):
        return f"{self.name}:{self.age}"
""".strip(),
        )
        .add_cell("c2", "p = Person()", after="c1")
        .add_cell("c3", "rendered = str(p)", after="c2")
    )

    _create_admin_worker(
        client,
        name="gpu-http-signed",
        runtime_id="gpu-http-signed-a100",
        config={
            "url": notebook_executor_server["execute_url"],
            "transport": "signed",
            "strata_url": notebook_build_server["base_url"],
        },
    )

    session_id = _open_notebook_via_api(client, notebook.path)
    assigned = client.put(
        f"/v1/notebooks/{session_id}/worker",
        json={"worker": "gpu-http-signed"},
    )
    assert assigned.status_code == 200

    with ws_connect(client, session_id) as ws:
        first = execute_cell_and_wait(ws, "c1")
        second = execute_cell_and_wait(ws, "c2")
        third = execute_cell_and_wait(ws, "c3")

        assert first["type"] == "cell_output"
        assert first["payload"]["remote_transport"] == "signed"
        assert first["payload"]["remote_build_state"] == "ready"

        assert second["type"] == "cell_output"
        assert second["payload"]["remote_transport"] == "signed"
        assert second["payload"]["remote_build_state"] == "ready"
        assert second["payload"]["outputs"]["p"]["content_type"] == "module/cell-instance"

        assert third["type"] == "cell_output"
        assert third["payload"]["remote_transport"] == "signed"
        assert third["payload"]["remote_build_state"] == "ready"
        assert third["payload"]["outputs"]["rendered"]["preview"] == "John:20"

        state = ws.sync()
        cell2 = next(cell for cell in state["payload"]["cells"] if cell["id"] == "c2")
        cell3 = next(cell for cell in state["payload"]["cells"] if cell["id"] == "c3")
        assert "p" in cell2["artifact_uris"]
        assert cell2["remote_transport"] == "signed"
        assert cell2["remote_build_state"] == "ready"
        assert cell2["status"] == "ready"
        assert cell3["remote_transport"] == "signed"
        assert cell3["remote_build_state"] == "ready"
        assert cell3["status"] == "ready"


def test_admin_worker_rename_and_delete_drift_propagates_into_existing_notebook(
    service_mode_notebook_client,
    notebook_executor_server,
):
    client, tmp = service_mode_notebook_client
    notebook = NotebookBuilder(tmp, "service_remote_drift").add_cell("c1", "x = 1")

    _create_admin_worker(
        client,
        name="gpu-http",
        runtime_id="gpu-http-a100",
        config={"url": notebook_executor_server["execute_url"]},
    )

    session_id = _open_notebook_via_api(client, notebook.path)
    assigned = client.put(
        f"/v1/notebooks/{session_id}/worker",
        json={"worker": "gpu-http"},
    )
    assert assigned.status_code == 200

    renamed = client.put(
        "/v1/admin/notebook-workers/gpu-http",
        json={
            "name": "gpu-http-renamed",
            "backend": "executor",
            "runtime_id": "gpu-http-a100",
            "config": {"url": notebook_executor_server["execute_url"]},
            "enabled": True,
        },
    )
    assert renamed.status_code == 200

    workers = client.get(f"/v1/notebooks/{session_id}/workers")
    assert workers.status_code == 200
    payload = workers.json()["workers"]
    old_entry = next(worker for worker in payload if worker["name"] == "gpu-http")
    renamed_entry = next(worker for worker in payload if worker["name"] == "gpu-http-renamed")
    assert old_entry["source"] == "referenced"
    assert old_entry["allowed"] is False
    assert renamed_entry["source"] == "server"
    assert renamed_entry["allowed"] is True

    with ws_connect(client, session_id) as ws:
        blocked_old = execute_cell_and_wait(ws, "c1")
        assert blocked_old["type"] == "cell_error"
        assert "not allowed in service mode" in blocked_old["payload"]["error"]

    reassigned = client.put(
        f"/v1/notebooks/{session_id}/worker",
        json={"worker": "gpu-http-renamed"},
    )
    assert reassigned.status_code == 200

    with ws_connect(client, session_id) as ws:
        result = execute_cell_and_wait(ws, "c1")
        assert result["type"] == "cell_output"
        assert result["payload"]["remote_worker"] == "gpu-http-renamed"
        assert result["payload"]["remote_transport"] == "direct"

    deleted = client.delete("/v1/admin/notebook-workers/gpu-http-renamed")
    assert deleted.status_code == 200

    workers = client.get(f"/v1/notebooks/{session_id}/workers")
    assert workers.status_code == 200
    deleted_entry = next(
        worker for worker in workers.json()["workers"] if worker["name"] == "gpu-http-renamed"
    )
    assert deleted_entry["source"] == "referenced"
    assert deleted_entry["allowed"] is False

    with ws_connect(client, session_id) as ws:
        blocked_deleted = execute_cell_and_wait(ws, "c1")
        assert blocked_deleted["type"] == "cell_error"
        assert "not allowed in service mode" in blocked_deleted["payload"]["error"]
