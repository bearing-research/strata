"""Tests for notebook REST routes."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from strata.notebook.dependencies import EnvironmentOperationLog
from strata.notebook.routes import get_session_manager, router
from strata.notebook.session import EnvironmentJobSnapshot
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)

# Fixtures + helpers


@pytest.fixture(autouse=True)
def no_uv_sync(monkeypatch):
    """Skip real venv/pool creation; route tests only exercise HTTP routing."""
    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)

    async def _fake_run_uv_command_streaming(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            success=True, error=None, operation_log=EnvironmentOperationLog(command="uv")
        )

    monkeypatch.setattr(
        "strata.notebook.dependencies.run_uv_command_streaming",
        _fake_run_uv_command_streaming,
    )

    async def _noop_start(self):
        pass

    monkeypatch.setattr("strata.notebook.pool.WarmProcessPool.start", _noop_start)


@pytest.fixture(scope="module")
def app():
    """FastAPI app with the notebook router (module-scoped; the router is stateless)."""
    fastapi_app = FastAPI()
    fastapi_app.include_router(router)
    return fastapi_app


@pytest.fixture
def client(app):
    """TestClient bound to the module-scoped app."""
    return TestClient(app)


def set_server_state(monkeypatch, **config):
    """Set ``strata.server._state`` to a SimpleNamespace with the given config keys.

    ``transforms_config`` defaults to an empty dict.
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
    """Fake server state with a service-mode worker registry."""

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
    """Fake server state with only deployment-mode settings."""

    def _configure(mode: str) -> None:
        set_server_state(monkeypatch, deployment_mode=mode)

    return _configure


# Open / create / delete


def test_open_notebook(client, tmp_path):
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


@pytest.mark.parametrize("case", ["port_typo", "corrupt_index"])
def test_open_notebook_with_a_bad_fetch_still_opens(client, tmp_path, case):
    """A typo'd ``@fetch`` URL or a damaged fetch index is the cell's problem, not the open's."""
    notebook_dir = create_notebook(tmp_path, "Fetch open")
    add_cell_to_notebook(notebook_dir, "c1")
    url = "http://localhost:80a/x.csv" if case == "port_typo" else "https://example.invalid/x"
    write_cell(notebook_dir, "c1", f"# @fetch zones {url}\nrows = 1")
    if case == "corrupt_index":
        fetch_dir = notebook_dir / ".strata" / "fetch"
        fetch_dir.mkdir(parents=True, exist_ok=True)
        (fetch_dir / "index.json").write_text('{"a": 1}\n}')

    response = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})

    assert response.status_code == 200, response.text
    (cell,) = response.json()["cells"]
    assert cell["status"] != "ready"


def test_open_syncs_the_environment_in_a_job_off_the_event_loop(client, monkeypatch, tmp_path):
    """``uv sync`` and the renv restore run as an environment job, not inline in the route."""
    from strata.notebook.session import NotebookSession

    notebook_dir = create_notebook(tmp_path, "Deferred Sync")
    (notebook_dir / "renv.lock").write_text('{"R": {"Version": "4.4.0"}, "Packages": {}}')
    inline_syncs: list[Path] = []
    monkeypatch.setattr(
        "strata.notebook.session._uv_sync",
        lambda path, **kw: inline_syncs.append(path) or True,
    )
    restores: list[bool] = []

    def recording_renv_sync(path):
        try:
            asyncio.get_running_loop()
            restores.append(True)
        except RuntimeError:
            restores.append(False)
        return True

    monkeypatch.setattr("strata.notebook.session._renv_sync", recording_renv_sync)
    submitted: list[str] = []
    real_submit = NotebookSession.submit_environment_job

    async def recording_submit(self, **kwargs):
        submitted.append(kwargs["action"])
        return await real_submit(self, **kwargs)

    monkeypatch.setattr(NotebookSession, "submit_environment_job", recording_submit)

    response = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})

    assert response.status_code == 200, response.text
    assert inline_syncs == []
    assert submitted == ["sync"]
    # Awaited: the open still answers with an environment cells can run in.
    assert response.json()["environment"]["sync_state"] == "ready"
    # False: no event loop in the thread the restore ran on.
    assert restores == [False]


def test_open_notebook_reuses_existing_session_in_personal_mode(client, monkeypatch, tmp_path):
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
    # Newest first: the open's own environment sync, then the persisted import.
    assert [job["action"] for job in data["environment_job_history"]] == ["sync", "import"]
    assert data["environment_job_history"][1]["status"] == "completed"
    assert data["environment_job_history"][1]["stale_cell_count"] == 1


def test_open_notebook_rehydrates_cached_status(client, tmp_path):
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
    response = client.post("/v1/notebooks/open", json={"path": "/nonexistent/notebook"})
    assert response.status_code == 404


def test_open_notebook_rejects_path_outside_configured_storage_root(client, monkeypatch, tmp_path):
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


@pytest.mark.parametrize("route", ["create", "import"])
def test_a_new_notebooks_session_records_its_creators_tenant(app, monkeypatch, tmp_path, route):
    """MCP hides a session from other tenants by the tenant recorded here."""
    from strata.auth import principal_context
    from strata.types import Principal

    set_server_state(monkeypatch, deployment_mode="service", notebook_storage_dir=tmp_path)
    opened: dict[str, object] = {}
    real_open = get_session_manager().open_notebook

    def recording_open(directory, **kwargs):
        opened["opened_by"] = kwargs.get("opened_by")
        return real_open(directory, **kwargs)

    monkeypatch.setattr(get_session_manager(), "open_notebook", recording_open)

    async def as_ana(scope, receive, send):
        with principal_context(Principal(id="ana", tenant="acme")):
            await app(scope, receive, send)

    client = TestClient(as_ana)
    if route == "create":
        response = client.post(
            "/v1/notebooks/create", json={"parent_path": str(tmp_path), "name": "nb"}
        )
    else:
        response = client.post(
            "/v1/notebooks/import", files={"file": ("nb.ipynb", _ipynb_bytes([_code("x = 1\n")]))}
        )

    assert response.status_code == 200, response.text
    assert opened["opened_by"] == ("ana", "acme")


def test_create_notebook_endpoint_defers_initial_environment_sync(client, monkeypatch):
    """Fresh notebook creation bootstraps the initial env as a background job."""
    captured: dict[str, object] = {}

    def fake_create_notebook(
        parent_path,
        name,
        python_version=None,
        *,
        initialize_environment=True,
    ):
        captured["initialize_environment"] = initialize_environment
        captured["python_version"] = python_version
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
        opened_by=None,
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
    notebook_dir = create_notebook(tmp_path, "Delete Me")
    artifact_file = notebook_dir / ".strata" / "artifacts" / "result.bin"
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_bytes(b"artifact")
    # Never write to .venv/bin/python: under a shared environment it links to the real
    # interpreter, and writing through it truncates the developer's Python.
    venv_marker = notebook_dir / ".venv" / "deleted-with-the-notebook"
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
    notebook_dir = create_notebook(tmp_path, "Service Delete")
    session_id = open_session_id(client, notebook_dir)

    deployment_mode_state("service")
    response = client.delete(f"/v1/notebooks/{session_id}")

    assert response.status_code == 403
    assert response.json()["detail"] == "Notebook deletion is only available in personal mode"
    assert notebook_dir.exists()


def test_delete_notebook_endpoint_rejects_active_environment_job(client, tmp_path):
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
    notebook_dir = create_notebook(tmp_path, "Forgotten Notebook")
    assert notebook_dir.exists()

    response = client.post("/v1/notebooks/delete-by-path", json={"path": str(notebook_dir)})

    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is True
    assert not notebook_dir.exists()


def test_delete_by_path_closes_open_session_before_removal(client, tmp_path):
    notebook_dir = create_notebook(tmp_path, "Open And Delete")
    session_id = open_session_id(client, notebook_dir)

    response = client.post("/v1/notebooks/delete-by-path", json={"path": str(notebook_dir)})

    assert response.status_code == 200
    assert not notebook_dir.exists()
    assert get_session_manager().get_session(session_id) is None


def test_delete_by_path_rejects_missing_notebook(client, tmp_path):
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
    notebook_dir = create_notebook(tmp_path, "Service Path Delete")
    deployment_mode_state("service")

    response = client.post("/v1/notebooks/delete-by-path", json={"path": str(notebook_dir)})

    assert response.status_code == 403
    assert notebook_dir.exists()


def test_get_notebook_runtime_config_endpoint(client, monkeypatch):
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
        "registry_enabled": True,
        "team_store_configured": False,
    }


# Environment endpoints


def test_get_environment_status_endpoint(client, tmp_path):
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
    """POST /environment/jobs accepts async requirements/environment imports."""
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


# Sessions discovery / reconnect


def test_list_sessions_personal_mode(client, deployment_mode_state, tmp_path):
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
    """Reconnect returns the same serialized runtime metadata as open."""
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
    deployment_mode_state("service")

    list_response = client.get("/v1/notebooks/sessions")
    assert list_response.status_code == 403
    assert "personal mode" in list_response.json()["detail"]

    get_response = client.get("/v1/notebooks/sessions/fake-session")
    assert get_response.status_code == 403
    assert "personal mode" in get_response.json()["detail"]


# Cell CRUD


def test_list_cells(client, tmp_path):
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


def test_update_notebook_workers_refuses_an_unknown_transport(client, tmp_path):
    from strata.notebook.parser import parse_notebook

    notebook_dir = create_notebook(tmp_path, "Worker Transport Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(
        f"/v1/notebooks/{session_id}/workers",
        json={
            "workers": [
                {
                    "name": "gpu",
                    "backend": "executor",
                    "config": {"url": "https://executor.internal/gpu", "transport": "http"},
                }
            ]
        },
    )

    assert response.status_code == 400
    assert "unknown worker transport 'http'" in response.json()["detail"]
    assert parse_notebook(notebook_dir).workers == []


def test_update_notebook_workers_forbidden_in_service_mode(
    client, service_mode_worker_state, tmp_path
):
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
    """Sensitive keys are blanked on disk but the session must hold the real values.

    Otherwise a cell launched right after the update sees an empty string.
    """
    notebook_dir = create_notebook(tmp_path, "Sensitive Env Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(
        f"/v1/notebooks/{session_id}/env",
        json={"env": {"ALPACA_API_KEY": "AKXYZ123", "DEBUG": "true"}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["env"]["DEBUG"] == "true"
    assert data["cells"][0]["env"]["DEBUG"] == "true"

    # The notebook-level env and each cell's resolved env must carry the real
    # value, not the blanked placeholder: the executor reads cell.env.
    session = get_session_manager().get_session(session_id)
    assert session is not None
    assert session.notebook_state.env["ALPACA_API_KEY"] == "AKXYZ123"
    cell = session.notebook_state.cells[0]
    assert cell.env["ALPACA_API_KEY"] == "AKXYZ123"


def test_secret_env_values_never_reach_clients(client, tmp_path):
    """Sensitive values are masked in every serialized view; a masked save keeps them."""
    from strata.notebook.secret_manager.session_integration import MASKED_ENV_VALUE

    notebook_dir = create_notebook(tmp_path, "Masked Env Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)
    response = client.put(
        f"/v1/notebooks/{session_id}/env",
        json={"env": {"OPENAI_API_KEY": "sk-secret", "LOG_LEVEL": "info", "DB_PASSWORD": ""}},
    )
    assert response.status_code == 200, response.text
    session = get_session_manager().get_session(session_id)
    assert session is not None

    views = {
        "put": response.json(),
        "get_cells": client.get(f"/v1/notebooks/{session_id}/cells").json(),
        "notebook_sync": session.serialize_notebook_state(),
    }
    for name, view in views.items():
        assert "sk-secret" not in json.dumps(view, default=str), name
    expected = {"OPENAI_API_KEY": MASKED_ENV_VALUE, "LOG_LEVEL": "info", "DB_PASSWORD": ""}
    assert views["put"]["env"] == expected
    assert views["notebook_sync"]["env"] == expected
    assert views["notebook_sync"]["cells"][0]["env"] == expected

    # Saving the masked marker back is "unchanged"; an edit replaces the value.
    response = client.put(
        f"/v1/notebooks/{session_id}/env",
        json={"env": {"OPENAI_API_KEY": MASKED_ENV_VALUE, "LOG_LEVEL": "debug"}},
    )
    assert response.status_code == 200, response.text
    assert session.notebook_state.env == {"OPENAI_API_KEY": "sk-secret", "LOG_LEVEL": "debug"}
    assert session.notebook_state.cells[0].env["OPENAI_API_KEY"] == "sk-secret"

    response = client.put(
        f"/v1/notebooks/{session_id}/env", json={"env": {"OPENAI_API_KEY": "sk-new"}}
    )
    assert response.status_code == 200, response.text
    assert session.notebook_state.env == {"OPENAI_API_KEY": "sk-new"}


@pytest.mark.parametrize(
    "others",
    [{}, {"LOG_LEVEL": "info"}],
    ids=["only-secrets-no-env-block", "secret-blank-in-env-block"],
)
def test_a_typed_secret_survives_a_reload(client, tmp_path, others):
    """A secret is never written to disk, so a reload must not blank it for the session."""
    import tomllib

    from strata.notebook.secret_manager.session_integration import MASKED_ENV_VALUE

    notebook_dir = create_notebook(tmp_path, "Secret Reload Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)
    session = get_session_manager().get_session(session_id)
    assert session is not None
    response = client.put(
        f"/v1/notebooks/{session_id}/env", json={"env": {"OPENAI_API_KEY": "sk-typed", **others}}
    )
    assert response.status_code == 200, response.text

    def on_disk() -> dict:
        with open(notebook_dir / "notebook.toml", "rb") as f:
            return tomllib.load(f).get("env", {})

    # PUT /timeout writes notebook.toml and reloads the session from it.
    assert client.put(f"/v1/notebooks/{session_id}/timeout", json={"timeout": 9}).status_code == 200
    assert session.notebook_state.env == {"OPENAI_API_KEY": "sk-typed", **others}
    assert session.notebook_state.cells[0].env["OPENAI_API_KEY"] == "sk-typed"
    assert on_disk().get("OPENAI_API_KEY", "") == ""
    assert "sk-typed" not in (notebook_dir / "notebook.toml").read_text()

    # The masked marker a client got back still means "unchanged" after the reload.
    response = client.put(
        f"/v1/notebooks/{session_id}/env",
        json={"env": {"OPENAI_API_KEY": MASKED_ENV_VALUE, **others}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["env"]["OPENAI_API_KEY"] == MASKED_ENV_VALUE
    assert session.notebook_state.env["OPENAI_API_KEY"] == "sk-typed"

    # A secret the env editor removes stays removed across the next reload.
    response = client.put(f"/v1/notebooks/{session_id}/env", json={"env": dict(others)})
    assert response.status_code == 200, response.text
    assert client.put(f"/v1/notebooks/{session_id}/timeout", json={"timeout": 8}).status_code == 200
    assert "OPENAI_API_KEY" not in session.notebook_state.env
    assert "OPENAI_API_KEY" not in session.notebook_state.cells[0].env


def test_a_secret_removed_from_notebook_toml_goes_away_on_reload(client, tmp_path):
    notebook_dir = create_notebook(tmp_path, "Secret Removed Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)
    session = get_session_manager().get_session(session_id)
    assert session is not None
    response = client.put(
        f"/v1/notebooks/{session_id}/env",
        json={"env": {"OPENAI_API_KEY": "sk-typed", "LOG_LEVEL": "info"}},
    )
    assert response.status_code == 200, response.text

    toml_path = notebook_dir / "notebook.toml"
    text = toml_path.read_text()
    assert 'OPENAI_API_KEY = ""\n' in text
    toml_path.write_text(text.replace('OPENAI_API_KEY = ""\n', ""))

    assert client.put(f"/v1/notebooks/{session_id}/timeout", json={"timeout": 9}).status_code == 200
    assert session.notebook_state.env == {"LOG_LEVEL": "info"}


def test_update_cell_source(client, tmp_path):
    """PUT /cells/{cell_id} updates the source in memory and on disk."""
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
    notebook_dir = create_notebook(tmp_path, "Add Cell Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.post(f"/v1/notebooks/{session_id}/cells", json={})

    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["source"] == ""


def test_delete_cell(client, tmp_path):
    notebook_dir = create_notebook(tmp_path, "Delete Test")
    add_cell_to_notebook(notebook_dir, "cell-1")
    session_id = open_session_id(client, notebook_dir)

    response = client.delete(f"/v1/notebooks/{session_id}/cells/cell-1")
    assert response.status_code == 200

    response = client.get(f"/v1/notebooks/{session_id}/cells")
    assert len(response.json()["cells"]) == 0


def test_delete_unknown_cell_is_404_not_500(client, tmp_path):
    """The route's catch-all must not mask the 404 as a 500."""
    notebook_dir = create_notebook(tmp_path, "Delete 404 Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.delete(f"/v1/notebooks/{session_id}/cells/ghost")
    assert response.status_code == 404, response.text


def test_add_cell_bad_after_is_400(client, tmp_path):
    """An after_cell_id that names no cell is rejected, as in the local backend."""
    notebook_dir = create_notebook(tmp_path, "Add After Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.post(f"/v1/notebooks/{session_id}/cells", json={"after_cell_id": "ghost"})
    assert response.status_code == 400, response.text


def test_rest_cell_crud_broadcasts_to_ws_spectators(client, tmp_path):
    """REST add / edit / reorder / delete each push a ``notebook_state`` frame to WS watchers.

    So an agent driving via REST/CLI/MCP shows up live in the TUI.
    """
    import strata.notebook.ws as ws_module

    notebook_dir = create_notebook(tmp_path, "CrudMirrorNb", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1", None, language="python")
    write_cell(notebook_dir, "c1", "x = 1\n")
    session_id = open_session_id(client, notebook_dir)

    sent: list[str] = []

    class _FakeSpectator:
        async def send_text(self, text):
            sent.append(text)

    ws_module._notebook_connections.setdefault(session_id, []).append(_FakeSpectator())
    try:
        created = client.post(f"/v1/notebooks/{session_id}/cells", json={})
        assert created.status_code == 200, created.text
        new_id = created.json()["id"]
        assert (
            client.put(
                f"/v1/notebooks/{session_id}/cells/{new_id}", json={"source": "y = 2\n"}
            ).status_code
            == 200
        )
        assert (
            client.put(
                f"/v1/notebooks/{session_id}/cells/reorder", json={"cell_ids": [new_id, "c1"]}
            ).status_code
            == 200
        )
        assert client.delete(f"/v1/notebooks/{session_id}/cells/{new_id}").status_code == 200
    finally:
        ws_module._notebook_connections.pop(session_id, None)

    types = [json.loads(t)["type"] for t in sent]
    # one full-state frame per structural edit (add, edit, reorder, delete).
    assert types.count("notebook_state") >= 4


def test_reorder_cells(client, tmp_path):
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
    notebook_dir = create_notebook(tmp_path, "Original Name")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(f"/v1/notebooks/{session_id}/name", json={"name": "New Name"})

    assert response.status_code == 200
    assert response.json()["name"] == "New Name"


def test_rename_notebook_rejects_blank_name(client, tmp_path):
    notebook_dir = create_notebook(tmp_path, "Original Name")
    session_id = open_session_id(client, notebook_dir)

    response = client.put(f"/v1/notebooks/{session_id}/name", json={"name": "   "})

    assert response.status_code == 422


# Cell execution (REST)


def test_execute_cell(client, tmp_path):
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
    notebook_dir = create_notebook(tmp_path, "Execute Test")
    session_id = open_session_id(client, notebook_dir)

    response = client.post(f"/v1/notebooks/{session_id}/cells/nonexistent/execute")
    assert response.status_code == 404


# Cell iterations


class TestCellIterationsEndpoint:
    """GET /cells/{cid}/iterations, the inspect panel's iteration picker.

    It must be safe to poll (empty list for non-loop or not-yet-run cells) and must infer
    the carry variable from the cell's ``@loop`` annotation.
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
        """Stored iteration artifacts come back in ascending order with content type and size."""
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
        """``?variable=`` inspects iterations of any variable, not only the inferred carry."""
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


# A notebook.toml stamped with an owner by an older server


def test_legacy_owner_key_neither_gates_nor_surfaces(client, monkeypatch, tmp_path):
    """Personal mode has one user: a leftover ``owner`` key is ignored on open and discover."""
    set_server_state(monkeypatch, deployment_mode="personal", notebook_storage_dir=tmp_path)
    notebook_dir = create_notebook(tmp_path, "Legacy Owned", initialize_environment=False)
    toml_path = notebook_dir / "notebook.toml"
    toml_path.write_text(
        'owner = "alice@example.com"\n' + toml_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    discovered = client.get("/v1/notebooks/discover").json()["notebooks"]
    assert [entry["path"] for entry in discovered] == [str(notebook_dir.resolve())]
    assert "owner" not in discovered[0]

    opened = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)})
    assert opened.status_code == 200, opened.text
    assert "owner" not in opened.json()
    session_id = opened.json()["session_id"]
    assert client.get(f"/v1/notebooks/{session_id}/cells").status_code == 200


# Connections


def test_list_and_update_notebook_connections(client, tmp_path):
    """PUT replaces the whole connection list; GET returns it.

    Auth literals are blanked at write time, so the response is the on-disk shape the UI
    uses to flag keys that still need ``${VAR}`` indirection.
    """
    notebook_dir = create_notebook(tmp_path, "Conn Routes Test")
    nb_id = open_session_id(client, notebook_dir)

    resp = client.get(f"/v1/notebooks/{nb_id}/connections")
    assert resp.status_code == 200
    assert resp.json()["connections"] == []

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

    # The relative path round-trips exactly; the executor resolves it against the
    # notebook dir at adapter-open time, so notebook.toml stays portable.
    warehouse = next(c for c in body["connections"] if c["name"] == "warehouse")
    assert warehouse["path"] == "analytics.db"

    # Literal "hunter2" is blanked: the UI sees the slot is set, but the value
    # isn't viable until ${PGPASS} is provided.
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
    """Duplicate names are rejected at the API, before any on-disk side effect."""
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
    """A PUT must not erase ``[connections.<name>]`` blocks that failed to parse.

    The parser flags them as ``MalformedConnection`` and the writer round-trips them.
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
    # stays on disk and surfaces via parse_notebook's malformed_connections,
    # pinned directly here.
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
    """Driver-specific ``options`` and unknown keys round-trip unchanged.

    ``ConnectionSpec`` allows extras; the route persists exactly what the UI sends.
    """
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
    """A relative SQLite path round-trips byte-for-byte; the executor resolves it at open time."""
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
    """Opens the connection read-only and returns the adapter's schema as a JSON tree (SQLite)."""
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
    """The 404 names the connection so the UI can show a useful message."""
    notebook_dir = create_notebook(tmp_path, "Schema 404")
    nb_id = open_session_id(client, notebook_dir)

    resp = client.get(f"/v1/notebooks/{nb_id}/connections/nope/schema")

    assert resp.status_code == 404
    assert "nope" in resp.json()["detail"]


# Export


def test_export_endpoint_defaults_to_zip(client, tmp_path):
    notebook_dir = create_notebook(tmp_path, "ExportZipDefault", initialize_environment=False)
    nb_id = open_session_id(client, notebook_dir)

    resp = client.get(f"/v1/notebooks/{nb_id}/export")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert ".zip" in resp.headers["content-disposition"]


def test_export_endpoint_returns_markdown_when_requested(client, tmp_path):
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


# POST /v1/notebooks/import: Jupyter notebook upload + convert


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
    """One markdown + one code cell come back as an opened session inside the storage root."""
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
    """Magics, !shell and pip-install lines surface in the import_report."""
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
    """A second import with the same name must not overwrite the first."""
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
    """With a tiny cap, the upload is rejected before the converter touches disk."""
    _import_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "strata.notebook.routes._MAX_IPYNB_UPLOAD_BYTES",
        50,  # 50 bytes: too small for any real .ipynb
    )

    payload = _ipynb_bytes([_code("x = 1\n")])
    assert len(payload) > 50  # sanity

    resp = client.post(
        "/v1/notebooks/import",
        files={"file": ("huge.ipynb", payload, "application/x-ipynb+json")},
    )

    assert resp.status_code == 413
    assert "MB cap" in resp.json()["detail"]


def test_import_endpoint_reads_at_most_one_byte_past_the_cap(client, monkeypatch, tmp_path):
    """An oversized upload is never pulled whole into memory to learn it is too big."""
    from starlette.datastructures import UploadFile

    _import_storage(monkeypatch, tmp_path)
    monkeypatch.setattr("strata.notebook.routes._MAX_IPYNB_UPLOAD_BYTES", 50)
    read_sizes: list[int] = []
    original_read = UploadFile.read

    async def recording_read(self, size: int = -1) -> bytes:
        data = await original_read(self, size)
        read_sizes.append(len(data))
        return data

    monkeypatch.setattr(UploadFile, "read", recording_read)
    payload = _ipynb_bytes([_code("x = 1\n" * 1000)])

    resp = client.post(
        "/v1/notebooks/import",
        files={"file": ("huge.ipynb", payload, "application/x-ipynb+json")},
    )

    assert resp.status_code == 413
    assert read_sizes == [51]


def test_import_endpoint_rejects_path_traversal_in_name(client, monkeypatch, tmp_path):
    """``name=../escaped`` must not land the notebook outside the storage root."""
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
    """A non-object top level, or ``cells`` not a list of dicts, is a clean 400, not a 500."""
    _import_storage(monkeypatch, tmp_path)

    bad_payloads = (
        b"[]",
        b'"a string"',
        b"null",
        b"42",
        # A ``cells`` entry isn't a dict.
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


def test_import_endpoint_uses_custom_name_form_field(client, monkeypatch, tmp_path):
    """The ``name`` form field overrides the upload's filename stem."""
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


# POST /v1/notebooks/import-snapshot: snapshot bundle upload


def _snapshot_bytes(tmp_path: Path) -> tuple[bytes, str]:
    """A snapshot of a one-cell notebook with one stored artifact, built without running
    anything."""
    import io
    import zipfile

    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.snapshot import write_committed_files, write_snapshot

    nb = create_notebook(tmp_path / "snapshot-src", "Snap Source", initialize_environment=False)
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", "x = 1\n")

    session = NotebookSession(parse_notebook(nb), nb)
    session.get_artifact_manager().store_cell_output(
        cell_id="c1",
        variable_name="x",
        blob_data=b"1",
        content_type="json/object",
        provenance_hash="c1" * 32,
        input_versions={},
        source="x = 1",
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        write_committed_files(session, archive)
        write_snapshot(session, archive, include="all")
    return buffer.getvalue(), session.notebook_state.id


def test_import_snapshot_opens_a_session_with_the_artifacts(client, monkeypatch, tmp_path):
    storage = _import_storage(monkeypatch, tmp_path / "storage")
    payload, source_id = _snapshot_bytes(tmp_path)

    resp = client.post(
        "/v1/notebooks/import-snapshot",
        files={"file": ("demo.snapshot.zip", payload, "application/zip")},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "session_id" in data
    assert Path(data["path"]).parent == storage.resolve()
    # The `.snapshot` infix is the export's naming, not part of the notebook's.
    assert Path(data["path"]).name == "demo"
    assert data["import_report"]["imported_artifacts"] == 1
    assert data["import_report"]["replaced_notebook_id"] is None
    assert data["id"] == source_id


def test_import_snapshot_replaces_an_id_already_in_the_storage_root(client, monkeypatch, tmp_path):
    """The second copy gets its own id; two copies sharing one collide in a shared store."""
    _import_storage(monkeypatch, tmp_path / "storage")
    payload, source_id = _snapshot_bytes(tmp_path)

    first = client.post(
        "/v1/notebooks/import-snapshot",
        files={"file": ("one.zip", payload, "application/zip")},
    )
    second = client.post(
        "/v1/notebooks/import-snapshot",
        files={"file": ("two.zip", payload, "application/zip")},
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["id"] == source_id
    assert second.json()["import_report"]["replaced_notebook_id"] == source_id
    assert second.json()["id"] != source_id


def test_import_snapshot_refuses_what_is_not_a_snapshot(client, monkeypatch, tmp_path):
    import io
    import zipfile

    _import_storage(monkeypatch, tmp_path / "storage")
    not_zip = client.post(
        "/v1/notebooks/import-snapshot",
        files={"file": ("a.zip", b"this is not a zip", "application/zip")},
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notebook.toml", 'notebook_id = "x"\n')
    no_manifest = client.post(
        "/v1/notebooks/import-snapshot",
        files={"file": ("b.zip", buffer.getvalue(), "application/zip")},
    )

    assert not_zip.status_code == 400
    assert no_manifest.status_code == 400
    assert "not a snapshot" in no_manifest.json()["detail"]


def test_import_snapshot_enforces_its_upload_cap(client, monkeypatch, tmp_path):
    _import_storage(monkeypatch, tmp_path / "storage")
    monkeypatch.setattr("strata.notebook.routes._MAX_SNAPSHOT_UPLOAD_BYTES", 10)
    payload, _ = _snapshot_bytes(tmp_path)

    resp = client.post(
        "/v1/notebooks/import-snapshot",
        files={"file": ("big.zip", payload, "application/zip")},
    )

    assert resp.status_code == 413
    assert not any((tmp_path / "storage").iterdir())


def test_import_snapshot_rejects_path_traversal_in_name(client, monkeypatch, tmp_path):
    """The name goes through the same checks as a Jupyter import's."""
    _import_storage(monkeypatch, tmp_path / "storage")
    payload, _ = _snapshot_bytes(tmp_path)

    resp = client.post(
        "/v1/notebooks/import-snapshot",
        files={"file": ("x.zip", payload, "application/zip")},
        data={"name": "../escaped"},
    )

    assert resp.status_code == 400
    assert not (tmp_path / "escaped").exists()


# PUT /v1/notebooks/{id}/python-version


def _open_for_python_version_tests(client, parent_dir: Path) -> tuple[str, Path]:
    """Create a notebook at 3.13; return (session_id, notebook_dir)."""
    notebook_dir = create_notebook(
        parent_dir, "PyVer Test", python_version="3.13", initialize_environment=False
    )
    session_id = open_session_id(client, notebook_dir)
    return session_id, notebook_dir


def test_python_version_update_rejects_unknown_version(client, monkeypatch, tmp_path):
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
    """The already-declared version returns 200 without accepting a job."""
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
    resp = client.put(
        "/v1/notebooks/nonexistent-session/python-version",
        json={"python_version": "3.13"},
    )
    assert resp.status_code == 404


def test_python_version_update_rejects_malformed_version(client, monkeypatch, tmp_path):
    """Anything that isn't major.minor fails validation."""
    set_server_state(
        monkeypatch,
        deployment_mode="personal",
        notebook_storage_dir=tmp_path,
        notebook_python_versions=["3.13"],
    )
    session_id, _ = _open_for_python_version_tests(client, tmp_path)

    # ``3.13.5`` has a patch component; it must be rejected before the
    # runtime-config allowlist check.
    resp = client.put(
        f"/v1/notebooks/{session_id}/python-version",
        json={"python_version": "3.13.5"},
    )
    assert resp.status_code == 422  # Pydantic validation error


class TestRuntimeConfigRegistryFlag:
    """``registry_enabled`` says whether there is a registry to show.

    Service mode has one too (the registry routes read through the tenant-scoped gate), so
    the flag is not "personal mode".
    """

    @pytest.mark.parametrize("mode", ["personal", "service"])
    def test_the_registry_is_offered_in_both_modes(self, mode, monkeypatch):
        from strata.notebook.routes import _serialize_notebook_runtime_config

        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(config=SimpleNamespace(deployment_mode=mode)),
        )
        cfg = _serialize_notebook_runtime_config()
        assert cfg["deployment_mode"] == mode
        assert cfg["registry_enabled"] is True

    @pytest.mark.parametrize(("url", "offered"), [(None, False), ("https://store.example", True)])
    def test_promotion_is_offered_only_with_a_team_store(self, url, offered, monkeypatch):
        """Without a team store the promote route answers 409, so the button would offer an
        error."""
        from strata.notebook.routes import _serialize_notebook_runtime_config

        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(deployment_mode="personal", notebook_remote_store_url=url)
            ),
        )
        assert _serialize_notebook_runtime_config()["team_store_configured"] is offered


# Cell tests endpoint (REST twin of WS cell_run_tests)


def test_set_cell_tests_endpoint(client, tmp_path):
    """PUT /cells/{id}/tests writes the test source and updates the session."""
    notebook_dir = create_notebook(tmp_path, "SetTestsNb", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "feat", None, language="python")
    write_cell(notebook_dir, "feat", "def double(x):\n    return x * 2\n")
    session_id = open_session_id(client, notebook_dir)

    src = "def test_double(cell):\n    assert cell.double(2) == 4\n"
    resp = client.put(f"/v1/notebooks/{session_id}/cells/feat/tests", json={"source": src})
    assert resp.status_code == 200, resp.text
    # written to its committed sibling file …
    assert (notebook_dir / "cells" / "feat.test.py").read_text() == src
    # … and the in-memory session reflects it.
    assert resp.json().get("test_source") == src

    # Unknown cell → 404; non-Python cell → 400.
    assert (
        client.put(
            f"/v1/notebooks/{session_id}/cells/ghost/tests", json={"source": "x"}
        ).status_code
        == 404
    )
    add_cell_to_notebook(notebook_dir, "md", None, language="markdown")
    # reopen so the session sees the markdown cell
    session_id = open_session_id(client, notebook_dir)
    assert (
        client.put(f"/v1/notebooks/{session_id}/cells/md/tests", json={"source": "x"}).status_code
        == 400
    )


def test_run_cell_tests_endpoint(client, tmp_path, monkeypatch):
    from strata.notebook.models import CellTestCase, CellTestResult
    from strata.notebook.writer import write_cell_tests

    notebook_dir = create_notebook(tmp_path, "TestsEndpointNb", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "feat", None, language="python")
    write_cell(notebook_dir, "feat", "def double(x):\n    return x * 2\n")
    write_cell_tests(
        notebook_dir, "feat", "def test_double(cell):\n    assert cell.double(2) == 4\n"
    )
    session_id = open_session_id(client, notebook_dir)

    async def _fake_run(self, cell_id, test_source):
        assert cell_id == "feat" and "test_double" in test_source
        return CellTestResult(
            passed=1,
            failed=1,
            tests=[
                CellTestCase(name="test_double", outcome="passed"),
                CellTestCase(name="test_bad", outcome="failed", message="boom"),
            ],
        )

    monkeypatch.setattr("strata.notebook.routes.CellExecutor.run_cell_tests", _fake_run)

    response = client.post(f"/v1/notebooks/{session_id}/cells/feat/tests")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["passed"] == 1 and body["failed"] == 1
    assert [t["outcome"] for t in body["tests"]] == ["passed", "failed"]
    assert body["tests"][1]["message"] == "boom"

    assert client.post(f"/v1/notebooks/{session_id}/cells/ghost/tests").status_code == 404


def test_run_cell_tests_endpoint_no_test_source_is_400(client, tmp_path):
    notebook_dir = create_notebook(tmp_path, "NoTestsNb", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "plain", None, language="python")
    write_cell(notebook_dir, "plain", "x = 1\n")
    session_id = open_session_id(client, notebook_dir)

    response = client.post(f"/v1/notebooks/{session_id}/cells/plain/tests")
    assert response.status_code == 400, response.text


@pytest.mark.parametrize(
    "mode,method",
    [("rerun", "execute_cell_rerun"), ("force", "execute_cell_force"), ("normal", "execute_cell")],
)
def test_execute_cell_mode_dispatches(client, tmp_path, monkeypatch, mode, method):
    from strata.notebook.executor import CellExecutionResult

    notebook_dir = create_notebook(tmp_path, "ModeNb", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1", None, language="python")
    write_cell(notebook_dir, "c1", "x = 1\n")
    session_id = open_session_id(client, notebook_dir)

    called: list[str] = []

    def _fake(name):
        async def _run(self, cell_id, source):
            called.append(name)
            return CellExecutionResult(
                cell_id=cell_id, success=True, duration_ms=1.0, execution_method="cold"
            )

        return _run

    for candidate in ("execute_cell", "execute_cell_rerun", "execute_cell_force"):
        monkeypatch.setattr(f"strata.notebook.routes.CellExecutor.{candidate}", _fake(candidate))

    response = client.post(f"/v1/notebooks/{session_id}/cells/c1/execute", params={"mode": mode})
    assert response.status_code == 200, response.text
    assert called == [method]
    assert response.json()["status"] == "ready"


def test_execute_cell_unknown_mode_is_400(client, tmp_path):
    """An unrecognized run mode is rejected before any execution."""
    notebook_dir = create_notebook(tmp_path, "BadModeNb", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1", None, language="python")
    write_cell(notebook_dir, "c1", "x = 1\n")
    session_id = open_session_id(client, notebook_dir)

    response = client.post(f"/v1/notebooks/{session_id}/cells/c1/execute", params={"mode": "bogus"})
    assert response.status_code == 400, response.text


def test_rest_execute_broadcasts_to_ws_spectators(client, tmp_path, monkeypatch):
    """A REST-driven run broadcasts the same frames to WS spectators as a WS-driven run.

    A watcher sees the cell go running and the result land, not just a poll catch-up.
    """
    import strata.notebook.ws as ws_module
    from strata.notebook.executor import CellExecutionResult

    notebook_dir = create_notebook(tmp_path, "MirrorNb", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1", None, language="python")
    write_cell(notebook_dir, "c1", "x = 1\n")
    session_id = open_session_id(client, notebook_dir)

    async def _fake_exec(self, cell_id, source):
        return CellExecutionResult(
            cell_id=cell_id, success=True, duration_ms=1.0, execution_method="cold"
        )

    monkeypatch.setattr("strata.notebook.executor.CellExecutor.execute_cell", _fake_exec)

    sent: list[str] = []

    class _FakeSpectator:
        async def send_text(self, text):
            sent.append(text)

    ws_module._notebook_connections.setdefault(session_id, []).append(_FakeSpectator())
    try:
        resp = client.post(f"/v1/notebooks/{session_id}/cells/c1/execute")
        assert resp.status_code == 200, resp.text
    finally:
        ws_module._notebook_connections.pop(session_id, None)

    types = [json.loads(t)["type"] for t in sent]
    # The spectator saw the cell go running plus at least one post-run frame.
    assert "cell_status" in types
    assert len(sent) >= 2


# GET /{id}/cells/{cell_id}/data: interactive data-viewer paging


def _store_table_artifact(session, cell_id, dataframe):
    """Store *dataframe* as a cell-output artifact; return its ``strata://artifact/...`` URI."""
    import tempfile

    from strata.notebook.serializer import serialize_value

    with tempfile.TemporaryDirectory() as tmpdir:
        meta = serialize_value(dataframe, Path(tmpdir), "df")
        blob = (Path(tmpdir) / meta["file"]).read_bytes()
    artifact = session.get_artifact_manager().store_cell_output(
        cell_id=cell_id,
        variable_name="df",
        blob_data=blob,
        content_type=meta["content_type"],
        row_count=meta["rows"],
        provenance_hash=f"prov-{cell_id}",
    )
    return f"strata://artifact/{artifact.id}@v={artifact.version}"


def _open_with_table(client, tmp_path, dataframe):
    """Create and open a one-cell notebook with *dataframe* as its output."""
    notebook_dir = create_notebook(tmp_path, "Data Viewer")
    add_cell_to_notebook(notebook_dir, "cell-1")
    write_cell(notebook_dir, "cell-1", "df = ...")
    session_id = open_session_id(client, notebook_dir)
    session = get_session_manager().get_session(session_id)
    assert session is not None
    uri = _store_table_artifact(session, "cell-1", dataframe)
    return session_id, uri


def test_cell_data_page_slices_and_reports_total(client, tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": list(range(50)), "b": [x * 2 for x in range(50)]})
    session_id, uri = _open_with_table(client, tmp_path, df)

    response = client.get(
        f"/v1/notebooks/{session_id}/cells/cell-1/data",
        params={"artifact_uri": uri, "offset": 5, "limit": 3},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["pageable"] is True
    assert data["total"] == 50
    assert data["columns"] == ["a", "b"]
    assert data["rows"] == [[5, 10], [6, 12], [7, 14]]


def test_cell_data_page_sorts_globally_before_slicing(client, tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": [3, 1, 2, 5, 4]})
    session_id, uri = _open_with_table(client, tmp_path, df)

    response = client.get(
        f"/v1/notebooks/{session_id}/cells/cell-1/data",
        params={"artifact_uri": uri, "limit": 2, "sort_by": "a", "sort_dir": "desc"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["rows"] == [[5], [4]]


def test_cell_data_page_rejects_bad_sort_dir(client, tmp_path):
    import pandas as pd

    session_id, uri = _open_with_table(client, tmp_path, pd.DataFrame({"a": [1]}))

    response = client.get(
        f"/v1/notebooks/{session_id}/cells/cell-1/data",
        params={"artifact_uri": uri, "sort_dir": "sideways"},
    )

    assert response.status_code == 400


def test_cell_data_page_missing_cell_is_404(client, tmp_path):
    import pandas as pd

    session_id, uri = _open_with_table(client, tmp_path, pd.DataFrame({"a": [1]}))

    response = client.get(
        f"/v1/notebooks/{session_id}/cells/nope/data",
        params={"artifact_uri": uri},
    )

    assert response.status_code == 404


def test_cell_data_page_malformed_uri_is_400(client, tmp_path):
    import pandas as pd

    session_id, _ = _open_with_table(client, tmp_path, pd.DataFrame({"a": [1]}))

    response = client.get(
        f"/v1/notebooks/{session_id}/cells/cell-1/data",
        params={"artifact_uri": "not-a-strata-uri"},
    )

    assert response.status_code == 400


def test_cell_data_page_search_and_filter(client, tmp_path):
    import pandas as pd

    df = pd.DataFrame({"id": [1, 2, 3], "region": ["North", "South", "north"], "v": [10, 200, 30]})
    session_id, uri = _open_with_table(client, tmp_path, df)

    search = client.get(
        f"/v1/notebooks/{session_id}/cells/cell-1/data",
        params={"artifact_uri": uri, "search": "north"},
    )
    assert search.status_code == 200, search.text
    assert search.json()["total"] == 2

    filtered = client.get(
        f"/v1/notebooks/{session_id}/cells/cell-1/data",
        params={"artifact_uri": uri, "filters": '[{"col": "v", "op": "gt", "value": 50}]'},
    )
    assert filtered.status_code == 200, filtered.text
    assert filtered.json()["total"] == 1
    assert filtered.json()["rows"][0][0] == 2


def test_cell_data_page_bad_filters_json_is_400(client, tmp_path):
    import pandas as pd

    session_id, uri = _open_with_table(client, tmp_path, pd.DataFrame({"a": [1]}))

    response = client.get(
        f"/v1/notebooks/{session_id}/cells/cell-1/data",
        params={"artifact_uri": uri, "filters": "{not json"},
    )
    assert response.status_code == 400


def test_cell_data_summary(client, tmp_path):
    import pandas as pd

    df = pd.DataFrame({"n": [1, 2, 2], "label": ["a", "b", "a"]})
    session_id, uri = _open_with_table(client, tmp_path, df)

    response = client.get(
        f"/v1/notebooks/{session_id}/cells/cell-1/data/summary",
        params={"artifact_uri": uri},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["pageable"] is True
    by_name = {c["name"]: c for c in data["columns"]}
    assert by_name["n"]["distinct"] == 2
    assert by_name["n"]["min"] == 1
    assert by_name["n"]["max"] == 2


def test_cell_data_export_csv(client, tmp_path):
    import io

    import pandas as pd

    df = pd.DataFrame({"id": [1, 2, 3], "v": [10, 200, 30]})
    session_id, uri = _open_with_table(client, tmp_path, df)

    response = client.get(
        f"/v1/notebooks/{session_id}/cells/cell-1/data/export",
        params={
            "artifact_uri": uri,
            "fmt": "csv",
            "filters": '[{"col": "v", "op": "gt", "value": 50}]',
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    out = pd.read_csv(io.BytesIO(response.content))
    assert out["id"].tolist() == [2]


def test_cell_data_export_rejects_bad_format(client, tmp_path):
    import pandas as pd

    session_id, uri = _open_with_table(client, tmp_path, pd.DataFrame({"a": [1]}))

    response = client.get(
        f"/v1/notebooks/{session_id}/cells/cell-1/data/export",
        params={"artifact_uri": uri, "fmt": "xlsx"},
    )
    assert response.status_code == 400
