"""Tests for notebook REST routes."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from strata.notebook.routes import get_session_manager, router
from strata.notebook.session import EnvironmentJobSnapshot
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_uv_sync(monkeypatch):
    """Skip real venv/pool creation — route tests only test HTTP routing."""
    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)

    async def _fake_run_uv_command_streaming(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(success=True, error=None, operation_log=None)

    monkeypatch.setattr(
        "strata.notebook.dependencies.run_uv_command_streaming",
        _fake_run_uv_command_streaming,
    )

    async def _noop_start(self):
        pass

    monkeypatch.setattr("strata.notebook.pool.WarmProcessPool.start", _noop_start)


@pytest.fixture(scope="module")
def app():
    """FastAPI app with notebook router. Module-scoped — the router is stateless."""
    fastapi_app = FastAPI()
    fastapi_app.include_router(router)
    return fastapi_app


@pytest.fixture
def client(app):
    """TestClient bound to the module-scoped app."""
    return TestClient(app)


def set_server_state(monkeypatch, **config):
    """Set ``strata.server._state`` to a SimpleNamespace with the given config keys.

    ``transforms_config`` defaults to an empty dict so most callers can drop
    that boilerplate. Pass it explicitly when you need workers or other
    transforms.
    """
    monkeypatch.setattr(
        "strata.server._state",
        SimpleNamespace(config=SimpleNamespace(**{"transforms_config": {}, **config})),
    )


def open_session_id(client, notebook_dir) -> str:
    """POST /v1/notebooks/open against ``notebook_dir`` and return its session_id."""
    response = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


@pytest.fixture
def service_mode_worker_state(monkeypatch):
    """Configure a fake server state with a service-mode worker registry."""

    def _configure(workers: list[dict] | None = None) -> None:
        set_server_state(
            monkeypatch,
            deployment_mode="service",
            transforms_config={
                "notebook_workers": workers
                or [
                    {
                        "name": "gpu-a100",
                        "backend": "executor",
                        "runtime_id": "cuda-12.4",
                        "config": {"url": "embedded://local"},
                    }
                ]
            },
        )

    return _configure


@pytest.fixture
def deployment_mode_state(monkeypatch):
    """Configure a fake server state with only deployment-mode settings."""

    def _configure(mode: str) -> None:
        set_server_state(monkeypatch, deployment_mode=mode)

    return _configure


# ---------------------------------------------------------------------------
# Open / create / delete
# ---------------------------------------------------------------------------


def test_open_notebook(client, tmp_path):
    """POST /v1/notebooks/open returns the canonical open-notebook payload."""
    notebook_dir = create_notebook(tmp_path, "Test Notebook")

    response = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Notebook"
    assert "session_id" in data
    assert "id" in data
    assert data["default_parent_path"] == str(Path.home() / ".strata" / "notebooks")
    assert "environment" in data
    env_fields = {
        "python_version",
        "requested_python_version",
        "runtime_python_version",
        "sync_state",
        "declared_package_count",
        "interpreter_source",
        "last_sync_duration_ms",
    }
    assert env_fields <= data["environment"].keys()
    assert "environment_job_history" in data
    assert "Server-Timing" in response.headers
    assert "session_open" in response.headers["Server-Timing"]


def test_open_notebook_reuses_existing_session_in_personal_mode(client, monkeypatch, tmp_path):
    """Opening the same path twice should reuse the live session in personal mode."""
    notebook_dir = create_notebook(tmp_path, "Reusable Notebook")
    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=tmp_path,
        notebook_python_versions=["3.13"],
    )

    first = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})
    second = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["session_id"] == second.json()["session_id"]


def test_open_notebook_rehydrates_environment_job_history(client, tmp_path):
    """Opening a notebook should expose persisted recent environment jobs."""
    notebook_dir = create_notebook(tmp_path, "Job History Notebook")
    history_path = notebook_dir / ".strata" / "environment_jobs.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            [
                {
                    "id": "job-789",
                    "action": "import",
                    "command": "uv sync",
                    "status": "completed",
                    "phase": "completed",
                    "started_at": 1234567890,
                    "finished_at": 1234567990,
                    "duration_ms": 100,
                    "stdout": "Resolved 4 packages\n",
                    "stderr": "",
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                    "lockfile_changed": True,
                    "stale_cell_count": 1,
                    "stale_cell_ids": ["cell-1"],
                    "error": None,
                }
            ]
        )
    )

    response = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})

    assert response.status_code == 200
    data = response.json()
    assert data["environment_job"]["action"] == "import"
    assert data["environment_job"]["status"] == "completed"
    assert len(data["environment_job_history"]) == 1
    assert data["environment_job_history"][0]["stale_cell_count"] == 1


def test_open_notebook_rehydrates_cached_status(client, tmp_path):
    """Opening an existing notebook should restore cached cell statuses."""
    notebook_dir = create_notebook(tmp_path, "Rehydrate Test")
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", "x = 1")
    add_cell_to_notebook(notebook_dir, "c2", after_cell_id="c1")
    write_cell(notebook_dir, "c2", "y = x + 1")

    from strata.notebook.executor import CellExecutor

    session = get_session_manager().open_notebook(notebook_dir)

    async def _prime() -> None:
        executor = CellExecutor(session)
        assert (await executor.execute_cell("c1", "x = 1")).success

    asyncio.run(_prime())

    response = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})

    assert response.status_code == 200
    cells = {cell["id"]: cell for cell in response.json()["cells"]}
    assert cells["c1"]["status"] == "ready"
    assert cells["c2"]["status"] == "idle"


def test_list_cells_includes_remote_execution_metadata(
    client,
    tmp_path,
    notebook_executor_server,
    notebook_build_server,
):
    """List-cells should retain remote execution metadata from the current session."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec

    notebook_build_server["config"].notebook_storage_dir = tmp_path
    notebook_dir = create_notebook(tmp_path, "Remote Metadata Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    write_cell(notebook_dir, "cell-1", "x = 1")
    session_id = open_session_id(client, notebook_dir)

    worker_config = {
        "url": notebook_executor_server["execute_url"],
        "transport": "signed",
        "strata_url": notebook_build_server["base_url"],
    }
    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {
            "name": "gpu-http-signed",
            "backend": "executor",
            "runtime_id": "gpu-http-signed-a100",
            "config": worker_config,
        }
    ]

    session = get_session_manager().get_session(session_id)
    assert session is not None
    session.notebook_state.workers = [
        WorkerSpec(
            name="gpu-http-signed",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="gpu-http-signed-a100",
            config=worker_config,
        )
    ]
    session.notebook_state.worker = "gpu-http-signed"
    cell = next(c for c in session.notebook_state.cells if c.id == "cell-1")
    cell.worker = "gpu-http-signed"

    async def _prime() -> None:
        executor = CellExecutor(session)
        assert (await executor.execute_cell("cell-1", "x = 1")).success

    asyncio.run(_prime())

    response = client.get(f"/v1/notebooks/{session_id}/cells")
    assert response.status_code == 200
    cell_payload = response.json()["cells"][0]
    assert cell_payload["execution_method"] == "executor"
    assert cell_payload["remote_worker"] == "gpu-http-signed"
    assert cell_payload["remote_transport"] == "signed"
    assert isinstance(cell_payload["remote_build_id"], str)
    assert cell_payload["remote_build_state"] == "ready"
    assert cell_payload["remote_error_code"] is None


def test_open_notebook_not_found(client):
    """Opening a non-existent notebook returns 404."""
    response = client.post("/v1/notebooks/open", json={"path": "/nonexistent/notebook"})
    assert response.status_code == 404


def test_open_notebook_rejects_path_outside_configured_storage_root(client, monkeypatch, tmp_path):
    """Opening a notebook outside the configured storage root should be rejected."""
    storage_root = tmp_path / "allowed"
    storage_root.mkdir()
    outside_root = tmp_path / "outside"
    notebook_dir = create_notebook(outside_root, "Outside Notebook")

    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=storage_root,
        notebook_python_versions=["3.13"],
    )

    response = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid notebook path: must be inside configured notebook storage"
    )


def test_create_notebook_endpoint(client, tmp_path):
    """POST /v1/notebooks/create returns the canonical create-notebook payload."""
    response = client.post(
        "/v1/notebooks/create", json={"parent_path": str(tmp_path), "name": "New Notebook"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "New Notebook"
    assert "session_id" in data
    assert data["default_parent_path"] == str(Path.home() / ".strata" / "notebooks")
    assert data["available_python_versions"]
    assert data["default_python_version"] == data["available_python_versions"][0]
    assert "python_selection_fixed" in data
    env = data["environment"]
    assert {
        "lockfile_hash",
        "requested_python_version",
        "runtime_python_version",
        "resolved_package_count",
    } <= env.keys()
    assert "Server-Timing" in response.headers
    assert "create_notebook" in response.headers["Server-Timing"]


def test_create_notebook_endpoint_defers_initial_environment_sync(client, monkeypatch):
    """Fresh notebook creation should bootstrap the initial env as a background job."""
    captured: dict[str, object] = {}

    def fake_create_notebook(
        parent_path,
        name,
        python_version=None,
        *,
        initialize_environment=True,
        owner=None,
    ):
        captured["initialize_environment"] = initialize_environment
        captured["python_version"] = python_version
        captured["owner"] = owner
        return Path("/tmp/fake-notebook")

    class FakeSession:
        id = "session-123"
        path = Path("/tmp/fake-notebook")
        environment_job = None
        environment_sync_state = "pending"
        environment_sync_error = None
        environment_sync_notice = "Notebook environment is initializing."

        def serialize_notebook_state(self):
            return {
                "id": "notebook-123",
                "name": "Fast Notebook",
                "cells": [],
                "environment": {"sync_state": self.environment_sync_state},
                "environment_job": self.environment_job,
            }

        async def submit_environment_job(self, *, action: str, **_kwargs):
            captured["environment_job_action"] = action
            self.environment_job = {
                "id": "job-123",
                "action": action,
                "status": "running",
                "command": "uv sync",
            }
            return self.environment_job

    def fake_open_notebook(
        directory,
        *,
        skip_initial_venv_sync=False,
        defer_initial_venv_sync=False,
        timing=None,
    ):
        captured["directory"] = directory
        captured["skip_initial_venv_sync"] = skip_initial_venv_sync
        captured["defer_initial_venv_sync"] = defer_initial_venv_sync
        captured["timing"] = timing
        return FakeSession()

    monkeypatch.setattr("strata.notebook.routes.create_notebook", fake_create_notebook)
    monkeypatch.setattr("strata.notebook.routes._session_manager.open_notebook", fake_open_notebook)

    response = client.post(
        "/v1/notebooks/create",
        json={"parent_path": "/tmp/notebooks", "name": "Fast Notebook"},
    )

    assert response.status_code == 200
    data = response.json()
    assert captured["initialize_environment"] is False
    assert captured["skip_initial_venv_sync"] is False
    assert captured["defer_initial_venv_sync"] is True
    assert captured["environment_job_action"] == "sync"
    assert captured["timing"] is not None
    assert data["environment"]["sync_state"] == "pending"
    assert data["environment_job"]["action"] == "sync"
    assert data["environment_job"]["status"] == "running"


def test_create_notebook_endpoint_with_starter_cell(client, tmp_path):
    """Scratch-style create requests can return a starter empty cell."""
    response = client.post(
        "/v1/notebooks/create",
        json={"parent_path": str(tmp_path), "name": "Scratch Notebook", "starter_cell": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["cells"]) == 1
    assert data["cells"][0]["source"] == ""
    assert data["cells"][0]["language"] == "python"


def test_create_notebook_endpoint_rejects_unsupported_python_version(client, monkeypatch, tmp_path):
    """Notebook creation should validate requested Python versions against server config."""
    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=tmp_path,
        notebook_python_versions=["3.13"],
    )

    response = client.post(
        "/v1/notebooks/create",
        json={"parent_path": str(tmp_path), "name": "New Notebook", "python_version": "3.12"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Python 3.12 is not available for notebook creation"


def test_create_notebook_endpoint_rejects_parent_path_outside_configured_storage_root(
    client, monkeypatch, tmp_path
):
    """Notebook creation parent paths must stay inside the configured storage root."""
    storage_root = tmp_path / "allowed"
    storage_root.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()

    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=storage_root,
        notebook_python_versions=["3.13"],
    )

    response = client.post(
        "/v1/notebooks/create",
        json={"parent_path": str(outside_root), "name": "New Notebook"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid parent path: must be inside configured notebook storage"
    )


def test_delete_notebook_endpoint_removes_directory_and_closes_session(client, tmp_path):
    """Deleting a notebook should remove its files and close the live session."""
    notebook_dir = create_notebook(tmp_path, "Delete Me")
    artifact_file = notebook_dir / ".strata" / "artifacts" / "result.bin"
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_bytes(b"artifact")
    venv_marker = notebook_dir / ".venv" / "bin" / "python"
    venv_marker.parent.mkdir(parents=True, exist_ok=True)
    venv_marker.write_text("", encoding="utf-8")

    session_id = open_session_id(client, notebook_dir)
    delete_response = client.delete(f"/v1/notebooks/{session_id}")

    assert delete_response.status_code == 200
    data = delete_response.json()
    assert data["deleted"] is True
    assert data["path"] == str(notebook_dir.resolve())
    assert not notebook_dir.exists()
    assert get_session_manager().get_session(session_id) is None


def test_delete_notebook_endpoint_rejects_service_mode(client, deployment_mode_state, tmp_path):
    """Notebook deletion should remain disabled in service mode."""
    notebook_dir = create_notebook(tmp_path, "Service Delete")
    session_id = open_session_id(client, notebook_dir)

    deployment_mode_state("service")
    response = client.delete(f"/v1/notebooks/{session_id}")

    assert response.status_code == 403
    assert response.json()["detail"] == "Notebook deletion is only available in personal mode"
    assert notebook_dir.exists()


def test_delete_notebook_endpoint_rejects_active_environment_job(client, tmp_path):
    """Notebook deletion should be blocked while env mutation is running."""
    notebook_dir = create_notebook(tmp_path, "Busy Notebook")
    session_id = open_session_id(client, notebook_dir)

    session = get_session_manager().get_session(session_id)
    assert session is not None
    session.environment_job = EnvironmentJobSnapshot(
        id="job-123",
        action="sync",
        command="uv sync",
        status="running",
        phase="uv_running",
        started_at=1,
    )

    response = client.delete(f"/v1/notebooks/{session_id}")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "ENVIRONMENT_BUSY"
    assert "environment update is in progress" in detail["message"]
    assert notebook_dir.exists()


def test_delete_notebook_endpoint_rejects_running_execution(client, monkeypatch, tmp_path):
    """Notebook deletion should be blocked while notebook execution is active."""
    notebook_dir = create_notebook(tmp_path, "Running Notebook")
    session_id = open_session_id(client, notebook_dir)

    session = get_session_manager().get_session(session_id)
    assert session is not None
    monkeypatch.setattr(session, "_has_active_execution", lambda: True)

    response = client.delete(f"/v1/notebooks/{session_id}")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Notebook deletion is blocked while notebook execution is running."
    )
    assert notebook_dir.exists()


def test_delete_by_path_removes_directory_without_session(client, tmp_path):
    """Path-based delete should work for a notebook that was never opened."""
    notebook_dir = create_notebook(tmp_path, "Forgotten Notebook")
    assert notebook_dir.exists()

    response = client.post("/v1/notebooks/delete-by-path", json={"path": str(notebook_dir)})

    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is True
    assert not notebook_dir.exists()


def test_delete_by_path_closes_open_session_before_removal(client, tmp_path):
    """If a session happens to be open for the path, it should be closed first."""
    notebook_dir = create_notebook(tmp_path, "Open And Delete")
    session_id = open_session_id(client, notebook_dir)

    response = client.post("/v1/notebooks/delete-by-path", json={"path": str(notebook_dir)})

    assert response.status_code == 200
    assert not notebook_dir.exists()
    assert get_session_manager().get_session(session_id) is None


def test_delete_by_path_rejects_missing_notebook(client, tmp_path):
    """Deleting a path that is not a notebook directory should 404."""
    bogus = tmp_path / "not-a-notebook"
    bogus.mkdir()

    response = client.post("/v1/notebooks/delete-by-path", json={"path": str(bogus)})

    assert response.status_code == 404
    assert bogus.exists()


def test_validate_recent_notebooks_filters_to_real_notebook_dirs(client, tmp_path):
    """Only paths whose ``notebook.toml`` is present on disk survive validation."""
    real = create_notebook(tmp_path, "Real Notebook")
    missing = tmp_path / "deleted-notebook"  # never created
    bare_dir = tmp_path / "no-toml-here"
    bare_dir.mkdir()

    response = client.post(
        "/v1/notebooks/recents/validate",
        json={"paths": [str(real), str(missing), str(bare_dir), "", "   "]},
    )

    assert response.status_code == 200
    assert response.json() == {"valid": [str(real)]}


def test_validate_recent_notebooks_handles_empty_list(client):
    response = client.post("/v1/notebooks/recents/validate", json={"paths": []})
    assert response.status_code == 200
    assert response.json() == {"valid": []}


def test_delete_by_path_rejects_service_mode(client, deployment_mode_state, tmp_path):
    """Path-based delete should also be disabled in service mode."""
    notebook_dir = create_notebook(tmp_path, "Service Path Delete")
    deployment_mode_state("service")

    response = client.post("/v1/notebooks/delete-by-path", json={"path": str(notebook_dir)})

    assert response.status_code == 403
    assert notebook_dir.exists()


def test_get_notebook_runtime_config_endpoint(client, monkeypatch):
    """The runtime config endpoint should expose the server default notebook path."""
    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=Path("/srv/strata-notebooks"),
        notebook_python_versions=["3.12", "3.13"],
    )

    response = client.get("/v1/notebooks/config")

    assert response.status_code == 200
    assert response.json() == {
        "deployment_mode": "personal",
        "default_parent_path": "/srv/strata-notebooks",
        "available_python_versions": ["3.12", "3.13"],
        "default_python_version": "3.12",
        "python_selection_fixed": False,
    }


# ---------------------------------------------------------------------------
# Environment endpoints
# ---------------------------------------------------------------------------


def test_get_environment_status_endpoint(client, tmp_path):
    """GET /v1/notebooks/{id}/environment exposes the environment payload."""
    notebook_dir = create_notebook(tmp_path, "Environment Status Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.get(f"/v1/notebooks/{session_id}/environment")

    assert response.status_code == 200
    env = response.json()["environment"]
    assert {
        "python_version",
        "requested_python_version",
        "runtime_python_version",
        "lockfile_hash",
        "declared_package_count",
        "resolved_package_count",
        "sync_state",
        "last_synced_at",
        "interpreter_source",
        "last_sync_duration_ms",
    } <= env.keys()


def test_sync_environment_endpoint(client, monkeypatch, tmp_path):
    """POST /v1/notebooks/{id}/environment/sync delegates to ``session.sync_environment``."""
    from strata.notebook.models import CellStaleness, CellStatus

    notebook_dir = create_notebook(tmp_path, "Environment Sync Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    write_cell(notebook_dir, "cell-1", "x = 1")
    session_id = open_session_id(client, notebook_dir)

    session = get_session_manager().get_session(session_id)
    assert session is not None

    async def _fake_sync_environment():
        session.environment_sync_state = "ready"
        session.environment_sync_error = None
        session.environment_sync_notice = "Using existing notebook venv."
        session.environment_last_synced_at = 1234567890
        session.environment_last_sync_duration_ms = 42
        session.environment_python_version = "3.13.2"
        session.environment_interpreter_source = "venv"
        return {"cell-1": CellStaleness(status=CellStatus.IDLE)}

    monkeypatch.setattr(session, "sync_environment", _fake_sync_environment)

    response = client.post(f"/v1/notebooks/{session_id}/environment/sync")

    assert response.status_code == 200
    data = response.json()
    env = data["environment"]
    assert env["sync_state"] == "ready"
    assert "requested_python_version" in env
    assert env["runtime_python_version"] == "3.13.2"
    assert env["python_version"] == "3.13.2"
    assert env["sync_notice"] == "Using existing notebook venv."
    assert env["last_sync_duration_ms"] == 42
    assert env["interpreter_source"] == "venv"
    assert "dependencies" in data
    assert data["stale_cell_count"] == 1
    assert data["stale_cell_ids"] == ["cell-1"]
    assert "cells" in data


def test_submit_environment_job_endpoint(client, monkeypatch, tmp_path):
    """POST /environment/jobs should accept a background job and expose its snapshot."""
    notebook_dir = create_notebook(tmp_path, "Environment Job Test")
    session_id = open_session_id(client, notebook_dir)

    session = get_session_manager().get_session(session_id)
    assert session is not None

    async def _fake_submit_environment_job(
        *,
        action: str,
        package: str | None = None,
        requirements_text: str | None = None,
        environment_yaml_text: str | None = None,
    ):
        del requirements_text, environment_yaml_text
        job = EnvironmentJobSnapshot(
            id="job-123",
            action=action,
            package=package,
            command=f"uv {action} {package}".strip(),
            status="running",
            phase="uv_running",
            started_at=1234567890,
        )
        session.environment_job = job
        return job

    monkeypatch.setattr(session, "submit_environment_job", _fake_submit_environment_job)

    response = client.post(
        f"/v1/notebooks/{session_id}/environment/jobs",
        json={"action": "add", "package": "six"},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["accepted"] is True
    assert data["environment_job"]["action"] == "add"
    assert data["environment_job"]["package"] == "six"
    assert data["environment_job"]["status"] == "running"


def test_submit_environment_import_job_endpoint(client, monkeypatch, tmp_path):
    """POST /environment/jobs should accept async requirements/environment imports."""
    notebook_dir = create_notebook(tmp_path, "Environment Import Job Test")
    session_id = open_session_id(client, notebook_dir)

    session = get_session_manager().get_session(session_id)
    assert session is not None

    captured: dict[str, str | None] = {}

    async def _fake_submit_environment_job(
        *,
        action: str,
        package: str | None = None,
        requirements_text: str | None = None,
        environment_yaml_text: str | None = None,
    ):
        captured["action"] = action
        captured["package"] = package
        captured["requirements_text"] = requirements_text
        captured["environment_yaml_text"] = environment_yaml_text
        job = EnvironmentJobSnapshot(
            id="job-456",
            action=action,
            package=package,
            command="uv sync",
            status="running",
            phase="preparing_import",
            started_at=1234567890,
        )
        session.environment_job = job
        return job

    monkeypatch.setattr(session, "submit_environment_job", _fake_submit_environment_job)

    response = client.post(
        f"/v1/notebooks/{session_id}/environment/jobs",
        json={"action": "import", "requirements": "pyarrow>=18.0.0\nsix==1.17.0\n"},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["accepted"] is True
    assert data["environment_job"]["action"] == "import"
    assert data["environment_job"]["status"] == "running"
    assert captured == {
        "action": "import",
        "package": None,
        "requirements_text": "pyarrow>=18.0.0\nsix==1.17.0\n",
        "environment_yaml_text": None,
    }


def test_submit_environment_import_job_endpoint_rejects_invalid_payload(client, tmp_path):
    """Import jobs must provide exactly one import source and no package."""
    notebook_dir = create_notebook(tmp_path, "Environment Import Validation Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.post(
        f"/v1/notebooks/{session_id}/environment/jobs",
        json={
            "action": "import",
            "requirements": "six==1.17.0\n",
            "environment_yaml": "dependencies: [six=1.17.0]\n",
        },
    )

    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"]


def test_submit_environment_job_endpoint_conflict_when_execution_running(client, tmp_path):
    """Background environment jobs should be rejected while cells are running."""
    from strata.notebook.models import CellStatus

    notebook_dir = create_notebook(tmp_path, "Environment Busy Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    write_cell(notebook_dir, "cell-1", "x = 1")
    session_id = open_session_id(client, notebook_dir)

    session = get_session_manager().get_session(session_id)
    assert session is not None
    session.notebook_state.cells[0].status = CellStatus.RUNNING

    response = client.post(
        f"/v1/notebooks/{session_id}/environment/jobs",
        json={"action": "sync"},
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "ENVIRONMENT_BUSY"


# ---------------------------------------------------------------------------
# Sessions discovery / reconnect
# ---------------------------------------------------------------------------


def test_list_sessions_personal_mode(client, deployment_mode_state, tmp_path):
    """Session listing should work in personal mode for reconnect UX."""
    deployment_mode_state("personal")
    notebook_dir = create_notebook(tmp_path, "Session Listing Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.get("/v1/notebooks/sessions")

    assert response.status_code == 200
    sessions = response.json()["sessions"]
    matching = [s for s in sessions if s["session_id"] == session_id]
    assert len(matching) == 1
    assert matching[0]["name"] == "Session Listing Test"
    assert Path(matching[0]["path"]).resolve() == notebook_dir.resolve()


def test_get_session_personal_mode_includes_execution_metadata(
    client, deployment_mode_state, tmp_path
):
    """Session reconnect should preserve the same serialized runtime metadata as open."""
    deployment_mode_state("personal")
    notebook_dir = create_notebook(tmp_path, "Session Metadata Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    write_cell(notebook_dir, "cell-1", "x = 1")
    session_id = open_session_id(client, notebook_dir)

    session = get_session_manager().get_session(session_id)
    assert session is not None
    cell = next(c for c in session.notebook_state.cells if c.id == "cell-1")
    cell.execution_method = "executor"
    cell.remote_worker = "gpu-http-signed"
    cell.remote_transport = "signed"
    cell.remote_build_id = "build-123"
    cell.remote_build_state = "ready"
    cell.remote_error_code = None

    response = client.get(f"/v1/notebooks/sessions/{session_id}")

    assert response.status_code == 200
    assert "Server-Timing" in response.headers
    assert "lookup" in response.headers["Server-Timing"]
    cell_payload = response.json()["cells"][0]
    assert cell_payload["execution_method"] == "executor"
    assert cell_payload["remote_worker"] == "gpu-http-signed"
    assert cell_payload["remote_transport"] == "signed"
    assert cell_payload["remote_build_id"] == "build-123"
    assert cell_payload["remote_build_state"] == "ready"
    assert cell_payload["remote_error_code"] is None


def test_session_endpoints_blocked_in_service_mode(client, deployment_mode_state):
    """Session discovery/reconnect should not be exposed in service mode."""
    deployment_mode_state("service")

    list_response = client.get("/v1/notebooks/sessions")
    assert list_response.status_code == 403
    assert "personal mode" in list_response.json()["detail"]

    get_response = client.get("/v1/notebooks/sessions/fake-session")
    assert get_response.status_code == 403
    assert "personal mode" in get_response.json()["detail"]


# ---------------------------------------------------------------------------
# Cell CRUD
# ---------------------------------------------------------------------------


def test_list_cells(client, tmp_path):
    """GET /v1/notebooks/{id}/cells returns the cells with source."""
    notebook_dir = create_notebook(tmp_path, "Cells Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    write_cell(notebook_dir, "cell-1", "x = 1")
    session_id = open_session_id(client, notebook_dir)

    response = client.get(f"/v1/notebooks/{session_id}/cells")

    assert response.status_code == 200
    data = response.json()
    assert len(data["cells"]) == 1
    assert data["cells"][0]["id"] == "cell-1"
    assert data["cells"][0]["source"] == "x = 1"


def test_update_notebook_mounts(client, tmp_path):
    """PUT /v1/notebooks/{id}/mounts replaces the mount list."""
    notebook_dir = create_notebook(tmp_path, "Mount Update Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(
        f"/v1/notebooks/{session_id}/mounts",
        json={"mounts": [{"name": "raw_data", "uri": "s3://bucket/raw", "mode": "ro"}]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mounts"][0]["name"] == "raw_data"
    assert data["cells"][0]["mounts"][0]["name"] == "raw_data"


def test_update_notebook_worker(client, tmp_path):
    """PUT /v1/notebooks/{id}/worker assigns the notebook-level worker."""
    notebook_dir = create_notebook(tmp_path, "Worker Update Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(f"/v1/notebooks/{session_id}/worker", json={"worker": "gpu-default"})

    assert response.status_code == 200
    data = response.json()
    assert data["worker"] == "gpu-default"
    assert any(worker["name"] == "gpu-default" for worker in data["workers"])
    assert data["cells"][0]["worker"] == "gpu-default"


def test_list_notebook_workers(client, tmp_path):
    """GET /v1/notebooks/{id}/workers returns the worker catalog."""
    notebook_dir = create_notebook(tmp_path, "Worker Catalog Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    response = client.get(f"/v1/notebooks/{session_id}/workers")

    assert response.status_code == 200
    data = response.json()
    assert any(worker["name"] == "local" for worker in data["workers"])
    assert data["definitions_editable"] is True
    assert isinstance(data["health_checked_at"], int)


def test_list_notebook_workers_refresh_bypasses_health_cache(client, monkeypatch, tmp_path):
    """Refreshing the worker list should bypass the short health cache."""
    import strata.notebook.routes as notebook_routes

    calls: list[bool] = []

    async def _fake_build_worker_catalog_with_health(notebook_state, *, force_refresh=False):
        calls.append(force_refresh)
        return [
            {
                "name": "local",
                "backend": "local",
                "runtime_id": None,
                "config": {},
                "source": "builtin",
                "health": "healthy",
                "allowed": True,
            }
        ]

    monkeypatch.setattr(
        notebook_routes,
        "build_worker_catalog_with_health",
        _fake_build_worker_catalog_with_health,
    )

    notebook_dir = create_notebook(tmp_path, "Worker Refresh Test")
    session_id = open_session_id(client, notebook_dir)

    first = client.get(f"/v1/notebooks/{session_id}/workers")
    assert first.status_code == 200
    assert first.json()["health_checked_at"] > 0

    second = client.get(f"/v1/notebooks/{session_id}/workers?refresh=true")
    assert second.status_code == 200
    assert second.json()["health_checked_at"] > 0

    assert calls == [False, True]


def test_list_notebook_workers_includes_health_history(client, monkeypatch, tmp_path):
    """Notebook worker catalog responses should include recent health probes."""
    import strata.notebook.routes as notebook_routes

    history_entry = {
        "checked_at": 123,
        "health": "unavailable",
        "error": "Health endpoint returned 503",
        "duration_ms": 87,
    }

    async def _fake_build_worker_catalog_with_health(notebook_state, *, force_refresh=False):
        del notebook_state, force_refresh
        return [
            {
                "name": "gpu-http",
                "backend": "executor",
                "runtime_id": None,
                "config": {"url": "https://executor.internal/v1/execute"},
                "source": "server",
                "health": "unavailable",
                "allowed": True,
                "enabled": True,
                "transport": "direct",
                "health_url": "https://executor.internal/health",
                "health_checked_at": 123,
                "last_error": "Health endpoint returned 503",
                "probe_count": 4,
                "healthy_probe_count": 1,
                "unavailable_probe_count": 2,
                "unknown_probe_count": 1,
                "consecutive_failures": 2,
                "last_healthy_at": 120,
                "last_unavailable_at": 123,
                "last_unknown_at": 118,
                "last_status_change_at": 123,
                "last_probe_duration_ms": 87,
                "health_history": [history_entry],
            }
        ]

    monkeypatch.setattr(
        notebook_routes,
        "build_worker_catalog_with_health",
        _fake_build_worker_catalog_with_health,
    )

    notebook_dir = create_notebook(tmp_path, "Worker History Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.get(f"/v1/notebooks/{session_id}/workers")

    assert response.status_code == 200
    worker = response.json()["workers"][0]
    assert worker["name"] == "gpu-http"
    assert worker["health_history"] == [history_entry]
    assert worker["probe_count"] == 4
    assert worker["consecutive_failures"] == 2
    assert worker["last_healthy_at"] == 120
    assert worker["last_unavailable_at"] == 123
    assert worker["last_probe_duration_ms"] == 87


def test_list_notebook_workers_in_service_mode(client, service_mode_worker_state, tmp_path):
    """Service mode should expose a server-managed worker registry."""
    service_mode_worker_state()
    notebook_dir = create_notebook(tmp_path, "Service Worker Catalog Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.get(f"/v1/notebooks/{session_id}/workers")

    assert response.status_code == 200
    data = response.json()
    assert data["definitions_editable"] is False
    assert any(
        worker["name"] == "gpu-a100" and worker["source"] == "server" and worker["allowed"] is True
        for worker in data["workers"]
    )


def test_update_notebook_workers(client, tmp_path):
    """PUT /v1/notebooks/{id}/workers updates the notebook-level worker catalog."""
    notebook_dir = create_notebook(tmp_path, "Worker Catalog Update Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(
        f"/v1/notebooks/{session_id}/workers",
        json={
            "workers": [
                {
                    "name": "gpu-a100",
                    "backend": "executor",
                    "runtime_id": "cuda-12.4",
                    "config": {"url": "https://executor.internal/gpu-a100"},
                }
            ]
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["configured_workers"][0]["name"] == "gpu-a100"
    assert data["configured_workers"][0]["backend"] == "executor"
    assert any(worker["name"] == "local" for worker in data["workers"])
    assert any(
        worker["name"] == "gpu-a100" and worker["health"] == "unavailable"
        for worker in data["workers"]
    )
    assert data["definitions_editable"] is True


def test_update_notebook_workers_forbidden_in_service_mode(
    client, service_mode_worker_state, tmp_path
):
    """Notebook-scoped worker definitions should be disabled in service mode."""
    service_mode_worker_state()
    notebook_dir = create_notebook(tmp_path, "Service Worker Update Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(
        f"/v1/notebooks/{session_id}/workers",
        json={
            "workers": [
                {
                    "name": "gpu-local",
                    "backend": "executor",
                    "config": {"url": "https://executor.internal/gpu-local"},
                }
            ]
        },
    )

    assert response.status_code == 403
    assert "managed by the server" in response.json()["detail"]


def test_update_notebook_worker_requires_allowlisted_service_worker(
    client, service_mode_worker_state, tmp_path
):
    """Service mode should reject worker names outside the server registry."""
    service_mode_worker_state()
    notebook_dir = create_notebook(tmp_path, "Service Worker Assignment Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    blocked = client.put(f"/v1/notebooks/{session_id}/worker", json={"worker": "gpu-shadow"})
    assert blocked.status_code == 403
    assert "not allowed in service mode" in blocked.json()["detail"]

    allowed = client.put(f"/v1/notebooks/{session_id}/worker", json={"worker": "gpu-a100"})
    assert allowed.status_code == 200
    payload = allowed.json()
    assert payload["worker"] == "gpu-a100"
    assert payload["definitions_editable"] is False


def test_update_notebook_worker_rejects_disabled_service_worker(
    client, service_mode_worker_state, tmp_path
):
    """Service mode should reject server-managed workers that are disabled."""
    service_mode_worker_state(
        [
            {
                "name": "gpu-a100",
                "backend": "executor",
                "runtime_id": "cuda-12.4",
                "config": {"url": "embedded://local"},
                "enabled": False,
            }
        ]
    )
    notebook_dir = create_notebook(tmp_path, "Disabled Service Worker Assignment Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    blocked = client.put(f"/v1/notebooks/{session_id}/worker", json={"worker": "gpu-a100"})

    assert blocked.status_code == 403
    assert "disabled by server policy" in blocked.json()["detail"]


def test_update_notebook_workers_probes_executor_health(client, notebook_executor_server, tmp_path):
    """Configured notebook workers should surface healthy executor probes."""
    notebook_dir = create_notebook(tmp_path, "Worker Health Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(
        f"/v1/notebooks/{session_id}/workers",
        json={
            "workers": [
                {
                    "name": "gpu-a100",
                    "backend": "executor",
                    "runtime_id": "cuda-12.4",
                    "config": {"url": notebook_executor_server["execute_url"]},
                }
            ]
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert any(
        worker["name"] == "gpu-a100" and worker["health"] == "healthy" for worker in data["workers"]
    )


def test_update_notebook_timeout_and_env(client, tmp_path):
    """Notebook-level timeout/env endpoints update the persisted defaults."""
    notebook_dir = create_notebook(tmp_path, "Runtime Update Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    timeout_response = client.put(f"/v1/notebooks/{session_id}/timeout", json={"timeout": 7.5})
    assert timeout_response.status_code == 200
    assert timeout_response.json()["timeout"] == 7.5

    env_response = client.put(
        f"/v1/notebooks/{session_id}/env", json={"env": {"APP_MODE": "secret"}}
    )
    assert env_response.status_code == 200
    data = env_response.json()
    assert data["env"] == {"APP_MODE": "secret"}
    assert data["cells"][0]["env"] == {"APP_MODE": "secret"}


def test_update_notebook_env_restores_sensitive_values_on_cells(client, tmp_path):
    """Sensitive keys (API keys, tokens) get blanked on disk by the writer,
    but the in-memory session must hold the real values — otherwise cells
    launched immediately after the update can't see them. Regression for
    the bug where ALPACA_API_KEY appeared set in the Runtime panel but
    the executor saw an empty string."""
    notebook_dir = create_notebook(tmp_path, "Sensitive Env Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(
        f"/v1/notebooks/{session_id}/env",
        json={"env": {"ALPACA_API_KEY": "AKXYZ123", "DEBUG": "true"}},
    )

    assert response.status_code == 200
    data = response.json()
    # Both the notebook-level view AND each cell's resolved env must
    # carry the real sensitive value, not the blanked placeholder.
    assert data["env"]["ALPACA_API_KEY"] == "AKXYZ123"
    assert data["env"]["DEBUG"] == "true"
    assert data["cells"][0]["env"]["ALPACA_API_KEY"] == "AKXYZ123"
    assert data["cells"][0]["env"]["DEBUG"] == "true"

    # Server-side state the executor actually reads — cell.env — must match
    # too, otherwise the executor sees a blanked value.
    session = get_session_manager().get_session(session_id)
    assert session is not None
    cell = session.notebook_state.cells[0]
    assert cell.env["ALPACA_API_KEY"] == "AKXYZ123"


def test_update_cell_source(client, tmp_path):
    """PUT /v1/notebooks/{id}/cells/{cell_id} updates the source in-memory and on disk."""
    notebook_dir = create_notebook(tmp_path, "Update Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    new_source = "x = 2 + 2"
    response = client.put(f"/v1/notebooks/{session_id}/cells/cell-1", json={"source": new_source})

    assert response.status_code == 200
    assert response.json()["cell"]["source"] == new_source

    cell_file = notebook_dir / "cells" / "cell-1.py"
    assert cell_file.read_text() == new_source


def test_add_cell(client, tmp_path):
    """POST /v1/notebooks/{id}/cells creates a new empty cell."""
    notebook_dir = create_notebook(tmp_path, "Add Cell Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.post(f"/v1/notebooks/{session_id}/cells", json={})

    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["source"] == ""


def test_delete_cell(client, tmp_path):
    """DELETE /v1/notebooks/{id}/cells/{cell_id} removes the cell."""
    notebook_dir = create_notebook(tmp_path, "Delete Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    response = client.delete(f"/v1/notebooks/{session_id}/cells/cell-1")
    assert response.status_code == 200

    response = client.get(f"/v1/notebooks/{session_id}/cells")
    assert len(response.json()["cells"]) == 0


def test_reorder_cells(client, tmp_path):
    """PUT /v1/notebooks/{id}/cells/reorder reorders cells in-place."""
    notebook_dir = create_notebook(tmp_path, "Reorder Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    add_cell_to_notebook(notebook_dir, "cell-2")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(
        f"/v1/notebooks/{session_id}/cells/reorder",
        json={"cell_ids": ["cell-2", "cell-1"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["cells"][0]["id"] == "cell-2"
    assert data["cells"][1]["id"] == "cell-1"


def test_rename_notebook(client, tmp_path):
    """PUT /v1/notebooks/{id}/name updates the notebook display name."""
    notebook_dir = create_notebook(tmp_path, "Original Name")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(f"/v1/notebooks/{session_id}/name", json={"name": "New Name"})

    assert response.status_code == 200
    assert response.json()["name"] == "New Name"


def test_rename_notebook_rejects_blank_name(client, tmp_path):
    """Renaming should reject empty notebook names."""
    notebook_dir = create_notebook(tmp_path, "Original Name")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(f"/v1/notebooks/{session_id}/name", json={"name": "   "})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Cell execution (REST)
# ---------------------------------------------------------------------------


def test_execute_cell(client, tmp_path):
    """POST /v1/notebooks/{id}/cells/{cell_id}/execute runs a cell."""
    notebook_dir = create_notebook(tmp_path, "Execute Test")
    add_cell_to_notebook(notebook_dir, "test-cell")
    write_cell(notebook_dir, "test-cell", "x = 1 + 1\ny = 'hello'")
    session_id = open_session_id(client, notebook_dir)

    response = client.post(f"/v1/notebooks/{session_id}/cells/test-cell/execute")

    assert response.status_code == 200
    data = response.json()
    assert data["cell_id"] == "test-cell"
    assert {"outputs", "stdout", "stderr", "duration_ms"} <= data.keys()
    assert data["status"] == "ready", (
        f"Expected 'ready' but got '{data['status']}': {data.get('error')}"
    )
    assert "x" in data["outputs"], f"Missing x in outputs: {data}"
    assert "y" in data["outputs"], f"Missing y in outputs: {data}"


def test_execute_cell_updates_session_state_and_history(client, tmp_path):
    """REST execution should update backend cell state and profiling history."""
    notebook_dir = create_notebook(tmp_path, "Execute Session State")
    add_cell_to_notebook(notebook_dir, "test-cell")
    write_cell(notebook_dir, "test-cell", "x = 41 + 1")
    add_cell_to_notebook(notebook_dir, "consumer", after_cell_id="test-cell")
    write_cell(notebook_dir, "consumer", "y = x + 1")
    session_id = open_session_id(client, notebook_dir)

    response = client.post(f"/v1/notebooks/{session_id}/cells/test-cell/execute")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

    session = get_session_manager().get_session(session_id)
    assert session is not None
    cell = next(c for c in session.notebook_state.cells if c.id == "test-cell")
    consumer = next(c for c in session.notebook_state.cells if c.id == "consumer")
    assert cell.status == "ready"
    assert consumer.status == "idle"
    assert cell.cache_hit is False
    assert cell.artifact_uri is not None
    assert len(session.execution_history["test-cell"]) == 1


def test_execute_cell_not_found(client, tmp_path):
    """Executing a non-existent cell returns 404."""
    notebook_dir = create_notebook(tmp_path, "Execute Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.post(f"/v1/notebooks/{session_id}/cells/nonexistent/execute")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Cell iterations
# ---------------------------------------------------------------------------


class TestCellIterationsEndpoint:
    """Tests for GET /v1/notebooks/{id}/cells/{cid}/iterations.

    The endpoint backs the inspect panel's iteration picker. It must be a safe
    poll target — empty list for non-loop cells, empty list while a loop cell
    has yet to run — and it must pick up the carry variable from the cell's
    ``@loop`` annotation without requiring the caller to know the variable name.
    """

    def _open(self, client, tmp_path: Path, cells: dict[str, str]) -> str:
        notebook_dir = create_notebook(tmp_path, "IterationsTest")
        for cell_id, source in cells.items():
            add_cell_to_notebook(notebook_dir, cell_id)
            write_cell(notebook_dir, cell_id, source)
        return open_session_id(client, notebook_dir)

    def test_non_loop_cell_returns_empty_list(self, client, tmp_path):
        session_id = self._open(client, tmp_path, {"c1": "x = 1"})

        response = client.get(f"/v1/notebooks/{session_id}/cells/c1/iterations")

        assert response.status_code == 200
        payload = response.json()
        assert payload["cell_id"] == "c1"
        assert payload["variable"] is None
        assert payload["iterations"] == []

    def test_missing_cell_returns_404(self, client, tmp_path):
        session_id = self._open(client, tmp_path, {"c1": "x = 1"})

        response = client.get(f"/v1/notebooks/{session_id}/cells/ghost/iterations")

        assert response.status_code == 404

    def test_loop_cell_without_executions_returns_empty(self, client, tmp_path):
        loop_source = "# @loop max_iter=3 carry=state\nstate = {'n': state['n'] + 1}\n"
        session_id = self._open(client, tmp_path, {"loop": loop_source})

        response = client.get(f"/v1/notebooks/{session_id}/cells/loop/iterations")

        assert response.status_code == 200
        payload = response.json()
        assert payload["variable"] == "state"
        assert payload["iterations"] == []

    def test_endpoint_surfaces_recorded_iteration_artifacts(self, client, tmp_path):
        """After directly storing iteration artifacts, the endpoint returns them
        in ascending order with their content type and size."""
        loop_source = "# @loop max_iter=3 carry=state\nstate = {'n': state['n'] + 1}\n"
        session_id = self._open(client, tmp_path, {"loop": loop_source})

        session = get_session_manager().get_session(session_id)
        assert session is not None
        artifact_mgr = session.get_artifact_manager()
        for k, n in enumerate([1, 2, 3]):
            artifact_mgr.store_cell_output(
                cell_id="loop",
                variable_name="state",
                blob_data=json.dumps({"n": n}).encode(),
                content_type="json/object",
                provenance_hash=f"prov-{k}",
                iteration=k,
            )

        response = client.get(f"/v1/notebooks/{session_id}/cells/loop/iterations")

        assert response.status_code == 200
        payload = response.json()
        assert payload["variable"] == "state"
        assert [item["iteration"] for item in payload["iterations"]] == [0, 1, 2]
        first = payload["iterations"][0]
        assert first["content_type"] == "json/object"
        assert first["byte_size"] > 0
        assert first["artifact_uri"].endswith("@iter=0@v=1")

    def test_variable_query_param_overrides_inferred_carry(self, client, tmp_path):
        """Passing ``?variable=`` lets the caller inspect iterations of any
        variable a cell might persist per iteration (e.g. multi-carry in a
        future phase)."""
        loop_source = "# @loop max_iter=3 carry=state\nstate = {'n': 1}\n"
        session_id = self._open(client, tmp_path, {"loop": loop_source})

        session = get_session_manager().get_session(session_id)
        assert session is not None
        artifact_mgr = session.get_artifact_manager()
        artifact_mgr.store_cell_output(
            cell_id="loop",
            variable_name="other",
            blob_data=b'{"k": 1}',
            content_type="json/object",
            provenance_hash="prov-other",
            iteration=0,
        )

        response = client.get(f"/v1/notebooks/{session_id}/cells/loop/iterations?variable=other")

        assert response.status_code == 200
        payload = response.json()
        assert payload["variable"] == "other"
        assert len(payload["iterations"]) == 1
        assert payload["iterations"][0]["iteration"] == 0


# ---------------------------------------------------------------------------
# Personal mode per-user scoping
# ---------------------------------------------------------------------------


class TestPersonalModeUserScoping:
    """Per-user subdir scoping when STRATA_PERSONAL_MODE_USER_HEADER is set.

    Mirrors the proxy-fronted personal deployment shape: a header injected by
    Cloudflare Access (or similar) identifies the calling user. Each user gets
    a private storage subdirectory and physically cannot see — let alone
    create or delete — notebooks belonging to anyone else.
    """

    HEADER = "X-Strata-Test-User"

    @pytest.fixture
    def configured_state(self, monkeypatch):
        """Inject server config with ``personal_mode_user_header`` set."""

        def _configure(storage_root: Path) -> None:
            set_server_state(
                monkeypatch,
                deployment_mode="personal",
                notebook_storage_dir=storage_root,
                notebook_python_versions=["3.13"],
                personal_mode_user_header=self.HEADER,
            )

        return _configure

    def _read_owner(self, notebook_dir: Path) -> str | None:
        import tomllib

        with open(notebook_dir / "notebook.toml", "rb") as f:
            return tomllib.load(f).get("owner")

    def _user_subdir(self, storage_root: Path, identity: str) -> Path:
        """The per-user subdir as the runtime config endpoint would advertise it."""
        from strata.notebook.routes import _sanitize_user_dir_name

        sanitized = _sanitize_user_dir_name(identity)
        assert sanitized is not None
        return storage_root / sanitized

    def test_runtime_config_returns_user_subdir(self, client, configured_state, tmp_path):
        """``/config`` returns ``<base>/<user>`` so the frontend creates there."""
        configured_state(tmp_path)

        response = client.get("/v1/notebooks/config", headers={self.HEADER: "alice@example.com"})

        assert response.status_code == 200
        expected = self._user_subdir(tmp_path, "alice@example.com").resolve()
        assert Path(response.json()["default_parent_path"]) == expected

    def test_create_lands_in_user_subdir(self, client, configured_state, tmp_path):
        """Creating with the user's parent_path stamps owner and lands in their subdir."""
        configured_state(tmp_path)
        alice_root = self._user_subdir(tmp_path, "alice@example.com")

        response = client.post(
            "/v1/notebooks/create",
            json={"parent_path": str(alice_root), "name": "Alice NB"},
            headers={self.HEADER: "alice@example.com"},
        )

        assert response.status_code == 200
        notebook_dir = alice_root / "alice_nb"
        assert notebook_dir.exists()
        assert self._read_owner(notebook_dir) == "alice@example.com"

    def test_create_outside_own_subdir_is_rejected(self, client, configured_state, tmp_path):
        """A user passing another user's subdir as parent_path → 400."""
        configured_state(tmp_path)
        bob_root = self._user_subdir(tmp_path, "bob@example.com")
        bob_root.mkdir(parents=True)  # rule out "rejected because dir missing"

        response = client.post(
            "/v1/notebooks/create",
            json={"parent_path": str(bob_root), "name": "Sneak"},
            headers={self.HEADER: "alice@example.com"},
        )

        assert response.status_code == 400

    def test_discover_returns_only_callers_subdir(self, client, configured_state, tmp_path):
        """Alice sees her notebooks; Bob's are physically isolated."""
        configured_state(tmp_path)

        client.post(
            "/v1/notebooks/create",
            json={
                "parent_path": str(self._user_subdir(tmp_path, "alice@example.com")),
                "name": "Alice NB",
            },
            headers={self.HEADER: "alice@example.com"},
        )
        client.post(
            "/v1/notebooks/create",
            json={
                "parent_path": str(self._user_subdir(tmp_path, "bob@example.com")),
                "name": "Bob NB",
            },
            headers={self.HEADER: "bob@example.com"},
        )

        response = client.get("/v1/notebooks/discover", headers={self.HEADER: "alice@example.com"})

        assert response.status_code == 200
        names = {nb["name"] for nb in response.json()["notebooks"]}
        assert names == {"Alice NB"}

    def test_delete_by_path_rejects_cross_user_path(self, client, configured_state, tmp_path):
        """Bob cannot delete a notebook by passing alice's path — boundary rejects."""
        configured_state(tmp_path)
        alice_root = self._user_subdir(tmp_path, "alice@example.com")

        client.post(
            "/v1/notebooks/create",
            json={"parent_path": str(alice_root), "name": "Alice NB"},
            headers={self.HEADER: "alice@example.com"},
        )
        alice_nb = alice_root / "alice_nb"

        response = client.post(
            "/v1/notebooks/delete-by-path",
            json={"path": str(alice_nb)},
            headers={self.HEADER: "bob@example.com"},
        )

        # Path validator rejects the cross-subdir reference at the boundary.
        assert response.status_code == 400
        assert alice_nb.exists()

    def test_delete_by_path_allows_owner(self, client, configured_state, tmp_path):
        configured_state(tmp_path)
        alice_root = self._user_subdir(tmp_path, "alice@example.com")

        client.post(
            "/v1/notebooks/create",
            json={"parent_path": str(alice_root), "name": "Alice NB"},
            headers={self.HEADER: "alice@example.com"},
        )
        alice_nb = alice_root / "alice_nb"

        response = client.post(
            "/v1/notebooks/delete-by-path",
            json={"path": str(alice_nb)},
            headers={self.HEADER: "alice@example.com"},
        )

        assert response.status_code == 200
        assert not alice_nb.exists()

    def test_sanitize_user_dir_name_collapses_unsafe_chars(self):
        """Sanity check: hostile header values can't escape the storage root."""
        from strata.notebook.routes import _sanitize_user_dir_name

        assert _sanitize_user_dir_name("alice@example.com") == "alice@example.com"
        assert _sanitize_user_dir_name("../etc/passwd") == "etc_passwd"
        assert _sanitize_user_dir_name("alice/bob") == "alice_bob"
        assert _sanitize_user_dir_name("") is None
        assert _sanitize_user_dir_name("...") is None  # all-trim chars

    def test_session_keyed_routes_owner_gated(self, client, configured_state, tmp_path):
        """A leaked session_id must not be a bearer capability across users.

        Before #41 the WS upgrade refused cross-owner reconnects but the
        ``/{session_id}/...`` REST routes did not. Pin that any
        SessionDep-backed route owned by Alice returns 404 (the same
        generic body the WS upgrade closes with) when Bob holds the id.

        ``cells`` is a representative read; the test covers the dependency
        itself, so it implies the same gate for every other SessionDep
        route by construction.
        """
        configured_state(tmp_path)
        alice_root = self._user_subdir(tmp_path, "alice@example.com")

        create_resp = client.post(
            "/v1/notebooks/create",
            json={"parent_path": str(alice_root), "name": "Alice NB"},
            headers={self.HEADER: "alice@example.com"},
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["session_id"]

        # Alice (owner): allowed.
        ok = client.get(
            f"/v1/notebooks/{session_id}/cells",
            headers={self.HEADER: "alice@example.com"},
        )
        assert ok.status_code == 200

        # Bob (non-owner) with a valid session_id: 404 — same shape the
        # dep returns for a genuinely missing notebook so probes can't
        # enumerate owners.
        denied = client.get(
            f"/v1/notebooks/{session_id}/cells",
            headers={self.HEADER: "bob@example.com"},
        )
        assert denied.status_code == 404
        assert denied.json() == {"detail": "Notebook not found"}

        # Missing identity header: must NOT silently bypass the gate.
        # ``_caller_identity`` returns None for both "header unset" and
        # "header omitted from this request"; the second case must close
        # on owned notebooks — otherwise the bypass is "just don't send
        # the header."
        bypass = client.get(f"/v1/notebooks/{session_id}/cells")
        assert bypass.status_code == 404
        assert bypass.json() == {"detail": "Notebook not found"}

    def test_legacy_unowned_notebook_still_accessible_when_scoping_on(
        self, client, configured_state, tmp_path
    ):
        """An ``owner = None`` notebook stays accessible to any caller.

        Per-user scoping was added to a backend that already had
        notebooks in the wild without an owner field. Closing the
        missing-header bypass for owned notebooks must not also close
        legacy unowned notebooks, or every pre-scoping notebook becomes
        suddenly inaccessible.
        """
        configured_state(tmp_path)
        alice_root = self._user_subdir(tmp_path, "alice@example.com")

        # Pre-scoping notebooks were created without ``owner``; mimic
        # that by stamping it then stripping the field on disk before
        # opening the session.
        create_resp = client.post(
            "/v1/notebooks/create",
            json={"parent_path": str(alice_root), "name": "Legacy NB"},
            headers={self.HEADER: "alice@example.com"},
        )
        assert create_resp.status_code == 200
        legacy_nb = alice_root / "legacy_nb"
        toml_path = legacy_nb / "notebook.toml"
        text = toml_path.read_text(encoding="utf-8")
        toml_path.write_text(
            "\n".join(line for line in text.splitlines() if not line.startswith("owner =")),
            encoding="utf-8",
        )

        opened = client.post(
            "/v1/notebooks/open",
            json={"path": str(legacy_nb)},
            headers={self.HEADER: "alice@example.com"},
        )
        assert opened.status_code == 200
        session_id = opened.json()["session_id"]

        no_header = client.get(f"/v1/notebooks/{session_id}/cells")
        assert no_header.status_code == 200

        wrong_header = client.get(
            f"/v1/notebooks/{session_id}/cells",
            headers={self.HEADER: "bob@example.com"},
        )
        assert wrong_header.status_code == 200


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


def test_list_and_update_notebook_connections(client, tmp_path):
    """Round-trip the [connections.<name>] surface through the API.

    Mirrors the mount/worker patterns: a PUT replaces the whole
    list, a follow-up GET returns the current state. Auth literals
    are blanked at write time, so the response reflects the
    on-disk shape — UI components rely on that to highlight which
    keys still need ``${VAR}`` indirection.
    """
    notebook_dir = create_notebook(tmp_path, "Conn Routes Test")
    nb_id = open_session_id(client, notebook_dir)

    # Initially empty.
    resp = client.get(f"/v1/notebooks/{nb_id}/connections")
    assert resp.status_code == 200
    assert resp.json()["connections"] == []

    # PUT a SQLite + Postgres pair.
    payload = {
        "connections": [
            {"name": "warehouse", "driver": "sqlite", "path": "analytics.db"},
            {
                "name": "prod",
                "driver": "postgresql",
                "uri": "postgresql://localhost:5432/prod",
                "auth": {"user": "${PGUSER}", "password": "hunter2"},
            },
        ]
    }
    resp = client.put(f"/v1/notebooks/{nb_id}/connections", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    names = sorted(c["name"] for c in body["connections"])
    assert names == ["prod", "warehouse"]

    # The relative path survives the round-trip exactly. (The
    # cell executor resolves against the notebook dir at
    # adapter-open time; notebook.toml stays portable.)
    warehouse = next(c for c in body["connections"] if c["name"] == "warehouse")
    assert warehouse["path"] == "analytics.db"

    # Literal "hunter2" is blanked — UI sees the slot is set
    # but value isn't viable until ${PGPASS} is provided.
    prod = next(c for c in body["connections"] if c["name"] == "prod")
    assert prod["auth"]["user"] == "${PGUSER}"
    assert prod["auth"]["password"] == ""

    # GET returns the same shape as PUT's response.
    resp2 = client.get(f"/v1/notebooks/{nb_id}/connections")
    assert resp2.status_code == 200
    assert sorted(c["name"] for c in resp2.json()["connections"]) == ["prod", "warehouse"]

    # Sending an empty list deletes every connection.
    resp3 = client.put(f"/v1/notebooks/{nb_id}/connections", json={"connections": []})
    assert resp3.status_code == 200
    assert resp3.json()["connections"] == []


def test_update_notebook_connections_rejects_duplicate_names(client, tmp_path):
    """Two entries with the same name → 400. The annotation
    validator would surface this even if it landed on disk, but
    the API layer is the better place to catch it (no on-disk
    side effects)."""
    notebook_dir = create_notebook(tmp_path, "Conn Dup Test")
    nb_id = open_session_id(client, notebook_dir)

    resp = client.put(
        f"/v1/notebooks/{nb_id}/connections",
        json={
            "connections": [
                {"name": "db", "driver": "sqlite", "path": "a.db"},
                {"name": "db", "driver": "sqlite", "path": "b.db"},
            ]
        },
    )

    assert resp.status_code == 400
    assert "duplicate" in resp.json()["detail"].lower()


def test_update_notebook_connections_preserves_malformed_blocks(client, tmp_path):
    """Codex review fix: a PUT that touches one connection must NOT
    erase ``[connections.<name>]`` blocks that previously failed to
    parse. The parser flags those entries as ``MalformedConnection``
    and the writer round-trips them. The route used to drop the
    malformed list before passing it to the writer, silently
    erasing the on-disk record on every save.
    """
    from strata.notebook.parser import parse_notebook

    notebook_dir = create_notebook(tmp_path, "Conn Malformed Test")
    # Inject a malformed [connections.<name>] block (driver missing).
    toml_path = notebook_dir / "notebook.toml"
    toml_path.write_text(
        toml_path.read_text() + '\n[connections.broken]\nhost = "localhost"\nport = 5432\n'
    )

    nb_id = open_session_id(client, notebook_dir)

    # The list endpoint exposes only valid connections; the malformed sibling
    # is preserved on disk and surfaced via parse_notebook's
    # malformed_connections field. Pin the latter directly — that's the
    # contract the route should honor when wiring the writer.
    before = parse_notebook(notebook_dir)
    assert "broken" in {m.name for m in before.malformed_connections}

    # Add a valid connection through the API. The malformed block must
    # survive on disk.
    resp = client.put(
        f"/v1/notebooks/{nb_id}/connections",
        json={
            "connections": [
                {"name": "warehouse", "driver": "sqlite", "path": "analytics.db"},
            ]
        },
    )
    assert resp.status_code == 200, resp.text

    after = parse_notebook(notebook_dir)
    assert {c.name for c in after.connections} == {"warehouse"}
    assert {m.name for m in after.malformed_connections} == {"broken"}
    broken = next(m for m in after.malformed_connections if m.name == "broken")
    assert broken.body == {"host": "localhost", "port": 5432}


def test_update_notebook_connections_preserves_unknown_driver_extras(client, tmp_path):
    """Codex review fix: ``ConnectionSpec`` is open-ended (Pydantic
    extra=allow) so a driver-specific ``options`` table or any
    forward-compat key set by a future driver must round-trip
    unchanged. The route persists exactly what the UI sends; the
    UI in turn preserves anything outside its known field set."""
    notebook_dir = create_notebook(tmp_path, "Conn Extras Test")
    nb_id = open_session_id(client, notebook_dir)

    body = {
        "name": "snowflake_dev",
        "driver": "snowflake",
        "uri": "snowflake://acct.region/db",
        "options": {"warehouse": "ANALYTICS", "schema": "public"},
        "future_extra": "preserve-me",
        "auth": {"user": "${SF_USER}", "api_token": "${SF_TOKEN}"},
    }
    resp = client.put(f"/v1/notebooks/{nb_id}/connections", json={"connections": [body]})

    assert resp.status_code == 200, resp.text
    out = resp.json()["connections"][0]
    # Every field round-trips, including the unknown driver, the options
    # table, the future-extra key, and the non-standard auth.api_token.
    assert out["driver"] == "snowflake"
    assert out["uri"] == "snowflake://acct.region/db"
    assert out["options"] == {"warehouse": "ANALYTICS", "schema": "public"}
    assert out["future_extra"] == "preserve-me"
    assert out["auth"]["user"] == "${SF_USER}"
    assert out["auth"]["api_token"] == "${SF_TOKEN}"


def test_update_notebook_connections_keeps_relative_paths_relative(client, tmp_path):
    """Codex review fix: a relative SQLite path round-trips
    byte-for-byte through a no-op edit. The parser stopped
    resolving paths so the writer persists exactly what the UI
    sends; the cell executor handles resolution at adapter-open
    time."""
    import tomllib

    notebook_dir = create_notebook(tmp_path, "Conn Rel Path Test")
    nb_id = open_session_id(client, notebook_dir)

    client.put(
        f"/v1/notebooks/{nb_id}/connections",
        json={
            "connections": [
                {"name": "warehouse", "driver": "sqlite", "path": "analytics.db"},
            ]
        },
    )

    # Round-trip a no-op edit by re-sending what the GET returned.
    listing = client.get(f"/v1/notebooks/{nb_id}/connections").json()
    client.put(
        f"/v1/notebooks/{nb_id}/connections",
        json={"connections": listing["connections"]},
    )

    # On disk the path is still relative.
    with open(notebook_dir / "notebook.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["connections"]["warehouse"]["path"] == "analytics.db"


def test_get_connection_schema_endpoint_lists_tables_and_columns(client, tmp_path):
    """Schema endpoint opens the connection read-only, runs the
    adapter's list_schema, and returns a JSON tree the UI can
    render. Pins the SQLite happy path end-to-end."""
    import sqlite3

    pytest.importorskip("adbc_driver_sqlite")

    db_path = tmp_path / "warehouse.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE events (id INTEGER PRIMARY KEY, label TEXT NOT NULL);
            CREATE TABLE attrs (id INTEGER PRIMARY KEY, value REAL);
            """
        )

    notebook_dir = create_notebook(tmp_path / "nb_dir", "Schema Endpoint")
    toml = notebook_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + f'\n[connections.warehouse]\ndriver = "sqlite"\npath = "{db_path}"\n'
    )
    nb_id = open_session_id(client, notebook_dir)

    resp = client.get(f"/v1/notebooks/{nb_id}/connections/warehouse/schema")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["connection"] == "warehouse"
    assert body["driver"] == "sqlite"
    table_names = {t["name"] for t in body["tables"]}
    assert table_names == {"events", "attrs"}

    events = next(t for t in body["tables"] if t["name"] == "events")
    col_names = [c["name"] for c in events["columns"]]
    assert col_names == ["id", "label"]
    label = next(c for c in events["columns"] if c["name"] == "label")
    assert label["nullable"] is False


def test_get_connection_schema_endpoint_unknown_connection_404(client, tmp_path):
    """Asking for a connection that isn't declared returns 404
    with the connection name in the error so the UI can surface
    a useful message."""
    notebook_dir = create_notebook(tmp_path, "Schema 404")
    nb_id = open_session_id(client, notebook_dir)

    resp = client.get(f"/v1/notebooks/{nb_id}/connections/nope/schema")

    assert resp.status_code == 404
    assert "nope" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_endpoint_defaults_to_zip(client, tmp_path):
    """No fmt param -> ZIP bundle (backward-compatible default)."""
    notebook_dir = create_notebook(tmp_path, "ExportZipDefault", initialize_environment=False)
    nb_id = open_session_id(client, notebook_dir)

    resp = client.get(f"/v1/notebooks/{nb_id}/export")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert ".zip" in resp.headers["content-disposition"]


def test_export_endpoint_returns_markdown_when_requested(client, tmp_path):
    """fmt=markdown -> rendered markdown with the right headers."""
    notebook_dir = create_notebook(tmp_path, "ExportRoute", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", "x = 1\n")
    nb_id = open_session_id(client, notebook_dir)

    resp = client.get(f"/v1/notebooks/{nb_id}/export?fmt=markdown")

    assert resp.status_code == 200
    assert "text/markdown" in resp.headers["content-type"]
    assert "attachment" in resp.headers["content-disposition"]
    # Filename is the notebook directory name, not the session UUID
    assert notebook_dir.name in resp.headers["content-disposition"]
    assert ".md" in resp.headers["content-disposition"]
    assert "x = 1" in resp.text


def test_export_endpoint_html_format(client, tmp_path):
    notebook_dir = create_notebook(tmp_path, "ExportHTML", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", "x = 1\n")
    nb_id = open_session_id(client, notebook_dir)

    resp = client.get(f"/v1/notebooks/{nb_id}/export?fmt=html")

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert resp.text.startswith("<!doctype html>")
    assert ".html" in resp.headers["content-disposition"]


def test_export_endpoint_rejects_unknown_format(client, tmp_path):
    notebook_dir = create_notebook(tmp_path, "ExportBadFmt", initialize_environment=False)
    nb_id = open_session_id(client, notebook_dir)

    resp = client.get(f"/v1/notebooks/{nb_id}/export?fmt=pdf")

    assert resp.status_code == 400


def test_export_endpoint_missing_notebook_404(client):
    resp = client.get("/v1/notebooks/not-a-real-session/export")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/notebooks/import — Jupyter notebook upload + convert
# ---------------------------------------------------------------------------


def _ipynb_bytes(cells: list[dict]) -> bytes:
    """Serialize a minimal nbformat-4 notebook with the given cells."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb).encode("utf-8")


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "source": source,
        "outputs": [],
        "execution_count": None,
        "metadata": {},
    }


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "source": source, "metadata": {}}


def _import_storage(monkeypatch, tmp_path: Path) -> Path:
    """Point the routes module at ``tmp_path`` as the storage root."""
    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=tmp_path,
    )
    return tmp_path


def test_import_endpoint_happy_path(client, monkeypatch, tmp_path):
    """A clean .ipynb with one markdown + one code cell comes back as
    an opened session with a session_id, a path inside the storage
    root, and a populated import_report."""
    storage = _import_storage(monkeypatch, tmp_path)

    payload = _ipynb_bytes([_md("# Hi\n"), _code("x = 1\n")])
    resp = client.post(
        "/v1/notebooks/import",
        files={"file": ("demo.ipynb", payload, "application/x-ipynb+json")},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "session_id" in data
    assert data["name"] == "demo"
    # Notebook materialized inside the configured storage root.
    assert str(storage) in data["path"]

    report = data["import_report"]
    assert report["markdown_cells"] == 1
    assert report["code_cells"] == 1
    assert report["captured_deps"] == []
    assert report["report_path"].endswith("import_report.md")
    assert "Imported from demo.ipynb" in report["report_text"]


def test_import_endpoint_reports_magic_translation(client, monkeypatch, tmp_path):
    """Magics, !shell, and pip-install lines are surfaced through the
    import_report fields without re-querying the converter."""
    _import_storage(monkeypatch, tmp_path)

    payload = _ipynb_bytes(
        [
            _code("%matplotlib inline\n%pip install httpx\nimport httpx\n"),
            _code("%%javascript\nalert('x')\n"),
            _code("!ls /data\nx = 1\n"),
        ]
    )
    resp = client.post(
        "/v1/notebooks/import",
        files={"file": ("magics.ipynb", payload, "application/x-ipynb+json")},
    )

    assert resp.status_code == 200, resp.text
    report = resp.json()["import_report"]
    assert "httpx" in report["captured_deps"]
    # Each section's count survives the JSON round-trip.
    assert len(report["translated_magics"]) >= 2
    assert any("javascript" in m for m in report["dropped_magics"])
    assert any("ls" in s for s in report["dropped_shells"])


def test_import_endpoint_rejects_empty_upload(client, monkeypatch, tmp_path):
    _import_storage(monkeypatch, tmp_path)

    resp = client.post(
        "/v1/notebooks/import",
        files={"file": ("empty.ipynb", b"", "application/x-ipynb+json")},
    )

    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_import_endpoint_rejects_invalid_json(client, monkeypatch, tmp_path):
    _import_storage(monkeypatch, tmp_path)

    resp = client.post(
        "/v1/notebooks/import",
        files={"file": ("bad.ipynb", b"this is not json", "application/x-ipynb+json")},
    )

    assert resp.status_code == 400
    assert "Invalid .ipynb JSON" in resp.json()["detail"]


def test_import_endpoint_rejects_collision(client, monkeypatch, tmp_path):
    """A second import with the same name into the same storage root
    should not silently overwrite the existing notebook."""
    _import_storage(monkeypatch, tmp_path)

    payload = _ipynb_bytes([_code("x = 1\n")])
    first = client.post(
        "/v1/notebooks/import",
        files={"file": ("dup.ipynb", payload, "application/x-ipynb+json")},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        "/v1/notebooks/import",
        files={"file": ("dup.ipynb", payload, "application/x-ipynb+json")},
    )

    assert second.status_code == 409
    assert "already exists" in second.json()["detail"]


def test_import_endpoint_enforces_upload_size_cap(client, monkeypatch, tmp_path):
    """Tiny cap monkeypatched on the route — the upload should be
    rejected before the converter touches disk."""
    _import_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "strata.notebook.routes._MAX_IPYNB_UPLOAD_BYTES",
        50,  # 50 bytes — too small for any real .ipynb
    )

    payload = _ipynb_bytes([_code("x = 1\n")])
    assert len(payload) > 50  # sanity

    resp = client.post(
        "/v1/notebooks/import",
        files={"file": ("huge.ipynb", payload, "application/x-ipynb+json")},
    )

    assert resp.status_code == 413
    assert "MB cap" in resp.json()["detail"]


def test_import_endpoint_rejects_path_traversal_in_name(client, monkeypatch, tmp_path):
    """Regression: ``name=../escaped`` must not let the imported notebook
    land outside the configured storage root."""
    storage = _import_storage(monkeypatch, tmp_path / "storage")
    storage.mkdir(parents=True, exist_ok=True)

    escape_target = (storage / ".." / "escaped").resolve()
    assert storage.resolve() not in escape_target.parents
    assert not escape_target.exists()

    payload = _ipynb_bytes([_code("x = 1\n")])
    for bad_name in ("../escaped", "..", "foo/bar", "foo\\bar", "foo/../escaped"):
        resp = client.post(
            "/v1/notebooks/import",
            files={"file": ("safe.ipynb", payload, "application/x-ipynb+json")},
            data={"name": bad_name},
        )
        assert resp.status_code == 400, (bad_name, resp.text)
        assert "Invalid notebook name" in resp.json()["detail"]

    assert not escape_target.exists()


def test_import_endpoint_rejects_structurally_invalid_notebook(client, monkeypatch, tmp_path):
    """Regression: a JSON file whose top-level value isn't an object,
    or whose ``cells`` field isn't a list of dicts, used to crash the
    converter mid-loop with AttributeError → 500. All shapes should
    surface as a clean 400 now."""
    _import_storage(monkeypatch, tmp_path)

    bad_payloads = (
        b"[]",
        b'"a string"',
        b"null",
        b"42",
        # ``cells`` entry isn't a dict — the reported regression case.
        b'{"cells":[1],"metadata":{},"nbformat":4,"nbformat_minor":5}',
        # ``cells`` itself isn't a list.
        b'{"cells":"oops","metadata":{},"nbformat":4,"nbformat_minor":5}',
    )
    for bad_payload in bad_payloads:
        resp = client.post(
            "/v1/notebooks/import",
            files={"file": ("malformed.ipynb", bad_payload, "application/x-ipynb+json")},
        )
        assert resp.status_code == 400, (bad_payload, resp.text)


def test_import_endpoint_stamps_caller_owner(client, monkeypatch, tmp_path):
    """When personal-mode per-user scoping is on, imported notebooks
    should inherit the caller identity the same way ``create`` does."""
    import tomllib

    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=tmp_path,
        personal_mode_user_header="X-Notebook-User",
    )

    payload = _ipynb_bytes([_code("x = 1\n")])
    resp = client.post(
        "/v1/notebooks/import",
        files={"file": ("owned.ipynb", payload, "application/x-ipynb+json")},
        headers={"X-Notebook-User": "alice@example.com"},
    )

    assert resp.status_code == 200, resp.text

    notebook_dir = Path(resp.json()["path"])
    with (notebook_dir / "notebook.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data.get("owner") == "alice@example.com"


def test_import_endpoint_uses_custom_name_form_field(client, monkeypatch, tmp_path):
    """The ``name`` form field overrides the upload's filename stem so
    the user can re-import into a different directory layout."""
    storage = _import_storage(monkeypatch, tmp_path)

    payload = _ipynb_bytes([_code("x = 1\n")])
    resp = client.post(
        "/v1/notebooks/import",
        files={"file": ("uploaded_name.ipynb", payload, "application/x-ipynb+json")},
        data={"name": "Renamed Notebook"},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["name"] == "Renamed Notebook"
    # Slugified by create_notebook (spaces → underscores, lowercased).
    assert (storage / "renamed_notebook").is_dir()


# ---------------------------------------------------------------------------
# PUT /v1/notebooks/{id}/python-version
# ---------------------------------------------------------------------------


def _open_for_python_version_tests(client, parent_dir: Path) -> tuple[str, Path]:
    """Helper: create a notebook at 3.13, return (session_id, notebook_dir)."""
    notebook_dir = create_notebook(
        parent_dir, "PyVer Test", python_version="3.13", initialize_environment=False
    )
    session_id = open_session_id(client, notebook_dir)
    return session_id, notebook_dir


def test_python_version_update_rejects_unknown_version(client, monkeypatch, tmp_path):
    """Asking for a Python the deployment doesn't allow returns 400."""
    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=tmp_path,
        notebook_python_versions=["3.13"],
    )
    session_id, _ = _open_for_python_version_tests(client, tmp_path)

    resp = client.put(
        f"/v1/notebooks/{session_id}/python-version",
        json={"python_version": "3.14"},
    )

    assert resp.status_code == 400, resp.text
    assert "not available" in resp.json()["detail"]


def test_python_version_update_no_op_for_current_version(client, monkeypatch, tmp_path):
    """Picking the version that's already declared returns 200 without
    accepting a new job."""
    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=tmp_path,
        notebook_python_versions=["3.12", "3.13"],
    )
    session_id, _ = _open_for_python_version_tests(client, tmp_path)

    resp = client.put(
        f"/v1/notebooks/{session_id}/python-version",
        json={"python_version": "3.13"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["accepted"] is False
    assert body["reason"] == "already_at_requested_version"


def test_python_version_update_unknown_notebook_returns_404(client):
    """Posting to a stale session id returns 404."""
    resp = client.put(
        "/v1/notebooks/nonexistent-session/python-version",
        json={"python_version": "3.13"},
    )
    assert resp.status_code == 404


def test_python_version_update_rejects_malformed_version(client, monkeypatch, tmp_path):
    """The Pydantic field validator rejects anything that isn't major.minor."""
    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=tmp_path,
        notebook_python_versions=["3.13"],
    )
    session_id, _ = _open_for_python_version_tests(client, tmp_path)

    # ``3.13.5`` has a patch component — must be rejected before reaching
    # the runtime-config allowlist check.
    resp = client.put(
        f"/v1/notebooks/{session_id}/python-version",
        json={"python_version": "3.13.5"},
    )
    assert resp.status_code == 422  # Pydantic validation error


class TestRuntimeConfigRegistryFlag:
    """``registry_enabled`` gates the registry UI: true in personal mode
    (where the registry routes are reachable), false in service mode."""

    def test_registry_enabled_in_personal_mode(self, monkeypatch):
        from strata.notebook.routes import _serialize_notebook_runtime_config

        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(config=SimpleNamespace(deployment_mode="personal")),
        )
        cfg = _serialize_notebook_runtime_config()
        assert cfg["deployment_mode"] == "personal"
        assert cfg["registry_enabled"] is True

    def test_registry_disabled_in_service_mode(self, monkeypatch):
        from strata.notebook.routes import _serialize_notebook_runtime_config

        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(config=SimpleNamespace(deployment_mode="service")),
        )
        cfg = _serialize_notebook_runtime_config()
        assert cfg["deployment_mode"] == "service"
        assert cfg["registry_enabled"] is False
