"""Tests for server-mode transforms (async materialize + build polling)."""

import json

import pytest
from fastapi.testclient import TestClient

from strata.artifact_store import get_artifact_store, reset_artifact_store
from strata.config import StrataConfig
from strata.notebook.writer import create_notebook
from strata.transforms.build_qos import TenantQuotaExceededError
from strata.transforms.build_store import get_build_store, reset_build_store
from strata.transforms.registry import (
    TransformRegistry,
    reset_transform_registry,
    set_transform_registry,
)


@pytest.fixture
def server_mode_config(tmp_path):
    """Create config for server mode with transforms enabled."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    return StrataConfig(
        host="127.0.0.1",
        port=8765,
        deployment_mode="service",  # Server mode
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
        notebook_storage_dir=tmp_path,
        transforms_config={
            "enabled": True,
            "registry": [
                {
                    "ref": "duckdb_sql@v1",
                    "executor_url": "http://executor:8080/execute",
                    "timeout_seconds": 300,
                },
                {
                    "ref": "allowed_transform@*",
                    "executor_url": "http://allowed:8080/execute",
                },
            ],
        },
    )


@pytest.fixture
def personal_mode_config(tmp_path):
    """Create config for personal mode (client-side execution)."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    return StrataConfig(
        host="127.0.0.1",
        port=8765,
        deployment_mode="personal",
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )


@pytest.fixture
def server_mode_auth_config(tmp_path):
    """Create config for server mode with trusted-proxy auth enabled."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    return StrataConfig(
        host="127.0.0.1",
        port=8765,
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
        notebook_storage_dir=tmp_path,
        transforms_config={
            "enabled": True,
            "registry": [
                {
                    "ref": "duckdb_sql@v1",
                    "executor_url": "http://executor:8080/execute",
                    "timeout_seconds": 300,
                },
                {
                    "ref": "restricted_transform@v1",
                    "executor_url": "http://restricted:8080/execute",
                    "requires_scope": "transform:restricted",
                },
            ],
        },
    )


@pytest.fixture
def server_mode_app(server_mode_config):
    """Create test app with server mode config."""
    import strata.server as server_module
    from strata.server import app

    # Reset singletons
    reset_artifact_store()
    reset_transform_registry()
    reset_build_store()

    # Initialize transform registry
    transform_registry = TransformRegistry.from_config(server_mode_config.transforms_config)
    set_transform_registry(transform_registry)

    # Initialize artifact store
    get_artifact_store(server_mode_config.artifact_dir)

    # Initialize build store
    db_path = server_mode_config.artifact_dir / "artifacts.sqlite"
    get_build_store(db_path)

    # Set up server state with mock planner
    from unittest.mock import MagicMock

    mock_state = MagicMock()
    mock_state.config = server_mode_config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()

    # Patch get_state
    original_state = server_module._state
    server_module._state = mock_state

    yield TestClient(app)

    # Restore
    server_module._state = original_state
    reset_artifact_store()
    reset_transform_registry()
    reset_build_store()


@pytest.fixture
def personal_mode_app(personal_mode_config):
    """Create test app with personal mode config."""
    import strata.server as server_module
    from strata.server import app

    # Reset singletons
    reset_artifact_store()
    reset_transform_registry()
    reset_build_store()

    # Initialize artifact store
    get_artifact_store(personal_mode_config.artifact_dir)

    # Personal mode ships the embedded registry (real lifespan does this)
    from strata.transforms.registry import TransformRegistry, set_transform_registry

    set_transform_registry(TransformRegistry.create_embedded_registry())

    # Set up server state with mock planner
    from unittest.mock import MagicMock

    mock_state = MagicMock()
    mock_state.config = personal_mode_config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()

    # Patch get_state
    original_state = server_module._state
    server_module._state = mock_state

    yield TestClient(app)

    # Restore
    server_module._state = original_state
    reset_artifact_store()
    reset_transform_registry()
    reset_build_store()


@pytest.fixture
def server_mode_auth_app(server_mode_auth_config):
    """Create test app with server mode config and trusted-proxy auth."""
    import strata.server as server_module
    from strata.server import app

    reset_artifact_store()
    reset_transform_registry()
    reset_build_store()

    transform_registry = TransformRegistry.from_config(server_mode_auth_config.transforms_config)
    set_transform_registry(transform_registry)
    get_artifact_store(server_mode_auth_config.artifact_dir)
    get_build_store(server_mode_auth_config.artifact_dir / "artifacts.sqlite")

    from unittest.mock import MagicMock

    mock_state = MagicMock()
    mock_state.config = server_mode_auth_config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()

    original_state = server_module._state
    server_module._state = mock_state

    yield TestClient(app)

    server_module._state = original_state
    reset_artifact_store()
    reset_transform_registry()
    reset_build_store()


def _auth_headers(
    tenant: str = "team-a",
    principal: str = "user-1",
    scopes: str | None = None,
) -> dict[str, str]:
    headers = {
        "X-Strata-Proxy-Token": "test-token",
        "X-Strata-Principal": principal,
        "X-Tenant-ID": tenant,
    }
    if scopes:
        headers["X-Strata-Scopes"] = scopes
    return headers


class TestNotebookWorkerAdminApi:
    """Tests for the server-managed notebook worker admin API."""

    def test_list_notebook_workers_service_mode(self, server_mode_app):
        """Service mode should expose the server-managed notebook worker registry."""
        response = server_mode_app.get("/v1/admin/notebook-workers")

        assert response.status_code == 200
        data = response.json()
        assert data["configured_workers"] == []
        assert data["definitions_editable"] is False
        assert isinstance(data["health_checked_at"], int)
        assert any(
            worker["name"] == "local"
            and worker["source"] == "builtin"
            and worker["health"] == "healthy"
            for worker in data["workers"]
        )

    def test_update_notebook_workers_service_mode(self, server_mode_app):
        """Replacing the registry should update both stored specs and catalog."""
        response = server_mode_app.put(
            "/v1/admin/notebook-workers",
            json={
                "workers": [
                    {
                        "name": "gpu-a100",
                        "backend": "executor",
                        "runtime_id": "cuda-12.4",
                        "config": {"url": "embedded://local"},
                    }
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["definitions_editable"] is False
        assert data["configured_workers"][0]["name"] == "gpu-a100"
        assert data["configured_workers"][0]["backend"] == "executor"
        assert data["configured_workers"][0]["enabled"] is True
        assert any(
            worker["name"] == "gpu-a100"
            and worker["source"] == "server"
            and worker["health"] == "healthy"
            and worker["transport"] == "embedded"
            for worker in data["workers"]
        )

        listed = server_mode_app.get("/v1/admin/notebook-workers")
        assert listed.status_code == 200
        assert listed.json()["configured_workers"][0]["name"] == "gpu-a100"

    def test_create_update_delete_notebook_worker_service_mode(self, server_mode_app):
        """Service mode should support targeted worker CRUD by name."""
        created = server_mode_app.post(
            "/v1/admin/notebook-workers",
            json={
                "name": "gpu-http",
                "backend": "executor",
                "runtime_id": "cuda-12.4",
                "config": {"url": "https://executor.internal/v1/execute"},
                "enabled": True,
            },
        )
        assert created.status_code == 200
        created_payload = created.json()
        assert created_payload["configured_workers"][0]["name"] == "gpu-http"

        updated = server_mode_app.put(
            "/v1/admin/notebook-workers/gpu-http",
            json={
                "name": "gpu-signed",
                "backend": "executor",
                "runtime_id": "cuda-12.5",
                "config": {
                    "url": "https://executor.internal/v1/execute",
                    "transport": "signed",
                },
                "enabled": False,
            },
        )
        assert updated.status_code == 200
        updated_payload = updated.json()
        assert updated_payload["configured_workers"][0]["name"] == "gpu-signed"
        assert updated_payload["configured_workers"][0]["enabled"] is False

        deleted = server_mode_app.delete("/v1/admin/notebook-workers/gpu-signed")
        assert deleted.status_code == 200
        assert deleted.json()["configured_workers"] == []

    def test_create_notebook_worker_rejects_duplicate_name(self, server_mode_app):
        """Targeted create should reject an existing worker name."""
        seeded = server_mode_app.post(
            "/v1/admin/notebook-workers",
            json={
                "name": "gpu-http",
                "backend": "executor",
                "config": {"url": "https://executor.internal/v1/execute"},
                "enabled": True,
            },
        )
        assert seeded.status_code == 200

        duplicate = server_mode_app.post(
            "/v1/admin/notebook-workers",
            json={
                "name": "gpu-http",
                "backend": "executor",
                "config": {"url": "https://executor.internal/v1/execute"},
                "enabled": True,
            },
        )
        assert duplicate.status_code == 409
        assert "already exists" in duplicate.json()["detail"]

    def test_patch_notebook_worker_enabled_state(self, server_mode_app):
        """Service-mode worker admin can disable and re-enable one worker."""
        seeded = server_mode_app.put(
            "/v1/admin/notebook-workers",
            json={
                "workers": [
                    {
                        "name": "gpu-a100",
                        "backend": "executor",
                        "config": {"url": "embedded://local"},
                    }
                ]
            },
        )
        assert seeded.status_code == 200

        disabled = server_mode_app.patch(
            "/v1/admin/notebook-workers/gpu-a100",
            json={"enabled": False},
        )
        assert disabled.status_code == 200
        disabled_payload = disabled.json()
        assert disabled_payload["configured_workers"][0]["enabled"] is False
        assert any(
            worker["name"] == "gpu-a100"
            and worker["allowed"] is False
            and worker["enabled"] is False
            for worker in disabled_payload["workers"]
        )

        enabled = server_mode_app.patch(
            "/v1/admin/notebook-workers/gpu-a100",
            json={"enabled": True},
        )
        assert enabled.status_code == 200
        assert enabled.json()["configured_workers"][0]["enabled"] is True

    def test_refresh_notebook_worker_health(self, server_mode_app):
        """Service-mode worker admin can force-refresh one worker by name."""
        seeded = server_mode_app.put(
            "/v1/admin/notebook-workers",
            json={
                "workers": [
                    {
                        "name": "gpu-a100",
                        "backend": "executor",
                        "config": {"url": "embedded://local"},
                    }
                ]
            },
        )
        assert seeded.status_code == 200

        refreshed = server_mode_app.post("/v1/admin/notebook-workers/gpu-a100/refresh")
        assert refreshed.status_code == 200
        payload = refreshed.json()
        assert isinstance(payload["health_checked_at"], int)
        assert any(
            worker["name"] == "gpu-a100" and worker["health_checked_at"] is not None
            for worker in payload["workers"]
        )

    def test_refresh_notebook_worker_health_records_recent_history(
        self,
        server_mode_app,
        monkeypatch,
    ):
        """Forced refreshes should accumulate a short recent probe trail."""
        import strata.notebook.workers as notebook_workers

        class _FakeResponse:
            def __init__(self, status_code: int, payload: dict):
                self.status_code = status_code
                self._payload = payload

            def json(self) -> dict:
                return self._payload

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return None

            async def get(self, url: str):
                del url
                status_code, payload = responses.pop(0)
                return _FakeResponse(status_code, payload)

        responses = [
            (503, {"status": "unavailable"}),
            (
                200,
                {
                    "status": "healthy",
                    "capabilities": {"transform_refs": ["notebook_cell@v1"]},
                },
            ),
        ]

        monkeypatch.setattr(notebook_workers, "_worker_health_cache", {})
        monkeypatch.setattr(notebook_workers.httpx, "AsyncClient", _FakeAsyncClient)

        seeded = server_mode_app.put(
            "/v1/admin/notebook-workers",
            json={
                "workers": [
                    {
                        "name": "gpu-http",
                        "backend": "executor",
                        "config": {"url": "https://executor.internal/v1/execute"},
                    }
                ]
            },
        )
        assert seeded.status_code == 200
        seeded_worker = next(
            worker for worker in seeded.json()["workers"] if worker["name"] == "gpu-http"
        )
        assert seeded_worker["health"] == "unavailable"
        assert seeded_worker["health_history"][0]["health"] == "unavailable"
        assert seeded_worker["probe_count"] == 1
        assert seeded_worker["unavailable_probe_count"] == 1
        assert seeded_worker["consecutive_failures"] == 1
        assert seeded_worker["last_unavailable_at"] is not None

        refreshed = server_mode_app.post("/v1/admin/notebook-workers/gpu-http/refresh")
        assert refreshed.status_code == 200
        refreshed_worker = next(
            worker for worker in refreshed.json()["workers"] if worker["name"] == "gpu-http"
        )
        assert refreshed_worker["health"] == "healthy"
        assert refreshed_worker["last_error"] is None
        assert refreshed_worker["probe_count"] == 2
        assert refreshed_worker["healthy_probe_count"] == 1
        assert refreshed_worker["unavailable_probe_count"] == 1
        assert refreshed_worker["consecutive_failures"] == 0
        assert refreshed_worker["last_healthy_at"] is not None
        assert refreshed_worker["last_status_change_at"] is not None
        assert [entry["health"] for entry in refreshed_worker["health_history"][:2]] == [
            "healthy",
            "unavailable",
        ]

    def test_update_notebook_workers_rejects_duplicate_names(self, server_mode_app):
        """Service-mode worker admin should reject duplicate worker names."""
        response = server_mode_app.put(
            "/v1/admin/notebook-workers",
            json={
                "workers": [
                    {
                        "name": "gpu-a100",
                        "backend": "executor",
                        "config": {"url": "embedded://local"},
                    },
                    {
                        "name": "gpu-a100",
                        "backend": "executor",
                        "config": {"url": "embedded://local"},
                    },
                ]
            },
        )

        assert response.status_code == 400
        assert "Duplicate notebook worker names" in response.json()["detail"]

    def test_admin_disable_propagates_into_notebook_catalog_and_assignment(
        self,
        server_mode_app,
        tmp_path,
        monkeypatch,
    ):
        """Admin enable/disable should flow through notebook worker APIs."""

        monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)

        async def _noop_start(self):
            del self

        monkeypatch.setattr("strata.notebook.pool.WarmProcessPool.start", _noop_start)

        configured = server_mode_app.put(
            "/v1/admin/notebook-workers",
            json={
                "workers": [
                    {
                        "name": "gpu-a100",
                        "backend": "executor",
                        "runtime_id": "cuda-12.4",
                        "config": {"url": "embedded://local"},
                    }
                ]
            },
        )
        assert configured.status_code == 200

        notebook_dir = create_notebook(tmp_path, "Service Notebook Worker Policy")
        opened = server_mode_app.post(
            "/v1/notebooks/open",
            json={"path": str(notebook_dir)},
        )
        assert opened.status_code == 200
        session_id = opened.json()["session_id"]

        workers = server_mode_app.get(f"/v1/notebooks/{session_id}/workers")
        assert workers.status_code == 200
        before_disable = next(
            worker for worker in workers.json()["workers"] if worker["name"] == "gpu-a100"
        )
        assert before_disable["enabled"] is True
        assert before_disable["allowed"] is True

        disabled = server_mode_app.patch(
            "/v1/admin/notebook-workers/gpu-a100",
            json={"enabled": False},
        )
        assert disabled.status_code == 200

        workers = server_mode_app.get(f"/v1/notebooks/{session_id}/workers")
        assert workers.status_code == 200
        after_disable = next(
            worker for worker in workers.json()["workers"] if worker["name"] == "gpu-a100"
        )
        assert after_disable["enabled"] is False
        assert after_disable["allowed"] is False
        assert "not selectable" in after_disable["last_error"]

        blocked = server_mode_app.put(
            f"/v1/notebooks/{session_id}/worker",
            json={"worker": "gpu-a100"},
        )
        assert blocked.status_code == 403
        assert "disabled by server policy" in blocked.json()["detail"]

        enabled = server_mode_app.patch(
            "/v1/admin/notebook-workers/gpu-a100",
            json={"enabled": True},
        )
        assert enabled.status_code == 200

        workers = server_mode_app.get(f"/v1/notebooks/{session_id}/workers")
        assert workers.status_code == 200
        after_enable = next(
            worker for worker in workers.json()["workers"] if worker["name"] == "gpu-a100"
        )
        assert after_enable["enabled"] is True
        assert after_enable["allowed"] is True

        allowed = server_mode_app.put(
            f"/v1/notebooks/{session_id}/worker",
            json={"worker": "gpu-a100"},
        )
        assert allowed.status_code == 200
        assert allowed.json()["worker"] == "gpu-a100"

    def test_admin_worker_crud_propagates_into_notebook_catalog_and_assignment(
        self,
        server_mode_app,
        tmp_path,
        monkeypatch,
    ):
        """Create/update/delete should propagate into notebook-visible worker policy."""

        monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)

        async def _noop_start(self):
            del self

        monkeypatch.setattr("strata.notebook.pool.WarmProcessPool.start", _noop_start)

        notebook_dir = create_notebook(tmp_path, "Service Notebook Worker CRUD")
        opened = server_mode_app.post(
            "/v1/notebooks/open",
            json={"path": str(notebook_dir)},
        )
        assert opened.status_code == 200
        session_id = opened.json()["session_id"]

        created = server_mode_app.post(
            "/v1/admin/notebook-workers",
            json={
                "name": "gpu-http",
                "backend": "executor",
                "runtime_id": "cuda-12.4",
                "config": {"url": "embedded://local"},
                "enabled": True,
            },
        )
        assert created.status_code == 200

        workers = server_mode_app.get(f"/v1/notebooks/{session_id}/workers")
        assert workers.status_code == 200
        created_entry = next(
            worker for worker in workers.json()["workers"] if worker["name"] == "gpu-http"
        )
        assert created_entry["source"] == "server"
        assert created_entry["allowed"] is True

        assigned_created = server_mode_app.put(
            f"/v1/notebooks/{session_id}/worker",
            json={"worker": "gpu-http"},
        )
        assert assigned_created.status_code == 200
        assert assigned_created.json()["worker"] == "gpu-http"

        updated = server_mode_app.put(
            "/v1/admin/notebook-workers/gpu-http",
            json={
                "name": "gpu-signed",
                "backend": "executor",
                "runtime_id": "cuda-12.5",
                "config": {
                    "url": "embedded://local",
                    "transport": "signed",
                },
                "enabled": True,
            },
        )
        assert updated.status_code == 200

        workers = server_mode_app.get(f"/v1/notebooks/{session_id}/workers")
        assert workers.status_code == 200
        payload = workers.json()["workers"]
        renamed_entry = next(worker for worker in payload if worker["name"] == "gpu-signed")
        assert renamed_entry["source"] == "server"
        assert renamed_entry["allowed"] is True
        old_entry = next(worker for worker in payload if worker["name"] == "gpu-http")
        assert old_entry["source"] == "referenced"
        assert old_entry["allowed"] is False

        blocked_old = server_mode_app.put(
            f"/v1/notebooks/{session_id}/worker",
            json={"worker": "gpu-http"},
        )
        assert blocked_old.status_code == 403
        assert "not allowed in service mode" in blocked_old.json()["detail"]

        assigned_renamed = server_mode_app.put(
            f"/v1/notebooks/{session_id}/worker",
            json={"worker": "gpu-signed"},
        )
        assert assigned_renamed.status_code == 200
        assert assigned_renamed.json()["worker"] == "gpu-signed"

        deleted = server_mode_app.delete("/v1/admin/notebook-workers/gpu-signed")
        assert deleted.status_code == 200

        workers = server_mode_app.get(f"/v1/notebooks/{session_id}/workers")
        assert workers.status_code == 200
        deleted_entry = next(
            worker for worker in workers.json()["workers"] if worker["name"] == "gpu-signed"
        )
        assert deleted_entry["source"] == "referenced"
        assert deleted_entry["allowed"] is False

        blocked_deleted = server_mode_app.put(
            f"/v1/notebooks/{session_id}/worker",
            json={"worker": "gpu-signed"},
        )
        assert blocked_deleted.status_code == 403
        assert "not allowed in service mode" in blocked_deleted.json()["detail"]

    def test_notebook_workers_admin_requires_service_mode(self, personal_mode_app):
        """The admin registry should not exist in personal mode."""
        response = personal_mode_app.get("/v1/admin/notebook-workers")

        assert response.status_code == 409
        assert "service mode" in response.json()["detail"]

    def test_notebook_workers_admin_requires_scope(self, server_mode_auth_app):
        """Trusted-proxy mode should require the notebook worker admin scope."""
        blocked = server_mode_auth_app.get(
            "/v1/admin/notebook-workers",
            headers=_auth_headers(),
        )
        assert blocked.status_code == 403
        assert blocked.json()["detail"] == "Insufficient scope"

        allowed = server_mode_auth_app.put(
            "/v1/admin/notebook-workers",
            headers=_auth_headers(scopes="admin:notebook-workers"),
            json={
                "workers": [
                    {
                        "name": "gpu-signed",
                        "backend": "executor",
                        "config": {
                            "url": "https://executor.internal/v1/execute",
                            "transport": "signed",
                        },
                    }
                ]
            },
        )
        assert allowed.status_code == 200
        assert allowed.json()["configured_workers"][0]["name"] == "gpu-signed"

        patched = server_mode_auth_app.patch(
            "/v1/admin/notebook-workers/gpu-signed",
            headers=_auth_headers(scopes="admin:notebook-workers"),
            json={"enabled": False},
        )
        assert patched.status_code == 200
        assert patched.json()["configured_workers"][0]["enabled"] is False

        created = server_mode_auth_app.post(
            "/v1/admin/notebook-workers",
            headers=_auth_headers(scopes="admin:notebook-workers"),
            json={
                "name": "gpu-http",
                "backend": "executor",
                "config": {"url": "https://executor.internal/v1/execute"},
                "enabled": True,
            },
        )
        assert created.status_code == 200
        assert any(worker["name"] == "gpu-http" for worker in created.json()["configured_workers"])

        replaced = server_mode_auth_app.put(
            "/v1/admin/notebook-workers/gpu-http",
            headers=_auth_headers(scopes="admin:notebook-workers"),
            json={
                "name": "gpu-http-renamed",
                "backend": "executor",
                "config": {
                    "url": "https://executor.internal/v1/execute",
                    "transport": "signed",
                },
                "enabled": True,
            },
        )
        assert replaced.status_code == 200
        assert any(
            worker["name"] == "gpu-http-renamed" for worker in replaced.json()["configured_workers"]
        )

        deleted = server_mode_auth_app.delete(
            "/v1/admin/notebook-workers/gpu-http-renamed",
            headers=_auth_headers(scopes="admin:notebook-workers"),
        )
        assert deleted.status_code == 200


class TestTransformValidation:
    """Tests for transform allowlist validation in server mode."""

    def test_allowed_transform_succeeds(self, server_mode_app):
        """Materialize with registered transform succeeds."""
        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/table"],
                "transform": {
                    "executor": "local://duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM input"},
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["hit"] is False
        assert data["build_id"] is not None
        assert data["state"] == "pending"
        # In server mode, no build_spec (server executes)
        assert data["build_spec"] is None

    def test_unregistered_transform_rejected(self, server_mode_app):
        """Materialize with unregistered transform returns 403."""
        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/table"],
                "transform": {
                    "executor": "local://unknown_executor@v1",
                    "params": {},
                },
            },
        )

        assert response.status_code == 403
        data = response.json()
        assert data["detail"]["error"] == "transform_not_allowed"
        assert "unknown_executor" in data["detail"]["message"]

    def test_wildcard_version_matches(self, server_mode_app):
        """Wildcard version in registry matches any version."""
        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/table"],
                "transform": {
                    "executor": "local://allowed_transform@v99",
                    "params": {},
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["build_id"] is not None

    def test_personal_mode_runs_embedded_and_rejects_unknown(self, personal_mode_app):
        """Personal mode executes embedded transforms; unknown ones fail fast.

        An executor the registry can't resolve used to be accepted and
        parked in 'building' forever (no executor existed in personal
        mode). The embedded runner now executes registered transforms,
        and unknown refs are rejected with a 400.
        """
        response = personal_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/table"],
                "transform": {
                    "executor": "local://any_executor@v1",
                    "params": {"sql": "SELECT 1"},
                },
            },
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "transform_unknown"

        response = personal_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": [],
                "transform": {
                    "executor": "local://duckdb_sql@v1",
                    "params": {"sql": "SELECT 1 as x"},
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["build_id"] is not None
        assert data["state"] == "pending"

    def test_transform_requires_scope_without_scope_is_rejected(self, server_mode_auth_app):
        """Registered transforms can still require an explicit principal scope."""
        response = server_mode_auth_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/table"],
                "transform": {
                    "executor": "local://restricted_transform@v1",
                    "params": {},
                },
            },
            headers=_auth_headers(),
        )

        assert response.status_code == 403
        data = response.json()
        assert data["detail"]["error"] == "insufficient_scope"
        assert data["detail"]["required_scope"] == "transform:restricted"

    def test_transform_requires_scope_with_scope_succeeds(self, server_mode_auth_app):
        """Transforms guarded by requires_scope should run for authorized callers."""
        response = server_mode_auth_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/table"],
                "transform": {
                    "executor": "local://restricted_transform@v1",
                    "params": {},
                },
            },
            headers=_auth_headers(scopes="transform:restricted"),
        )

        assert response.status_code == 200
        assert response.json()["build_id"] is not None

    @pytest.mark.asyncio
    async def test_server_mode_materialize_respects_quota_estimate(
        self, server_mode_config, monkeypatch
    ):
        """Quota checks should use a real output estimate, not zero."""
        from unittest.mock import MagicMock

        import strata.server as server_module
        from strata.server import materialize_artifact
        from strata.types import MaterializeRequest

        reset_artifact_store()
        reset_transform_registry()
        reset_build_store()

        transform_registry = TransformRegistry.from_config(server_mode_config.transforms_config)
        set_transform_registry(transform_registry)
        get_artifact_store(server_mode_config.artifact_dir)
        get_build_store(server_mode_config.artifact_dir / "artifacts.sqlite")

        mock_state = MagicMock()
        mock_state.config = server_mode_config
        mock_state.planner = MagicMock()
        mock_state.fetcher = MagicMock()
        mock_state.scans = {}
        mock_state.metrics = MagicMock()

        original_state = server_module._state
        server_module._state = mock_state
        captured: dict[str, object] = {}

        class FakeQoS:
            def classify_build(
                self,
                estimated_output_bytes=None,
                input_count=0,
                explicit_priority=None,
            ):
                captured["classified_estimated_bytes"] = estimated_output_bytes
                captured["classified_input_count"] = input_count
                return "interactive"

            async def check_quota(self, tenant_id, estimated_bytes):
                captured["quota_tenant_id"] = tenant_id
                captured["quota_estimated_bytes"] = estimated_bytes
                raise TenantQuotaExceededError(
                    tenant_id=tenant_id,
                    used_bytes=0,
                    limit_bytes=1,
                    reset_in_seconds=60.0,
                )

            async def acquire(self, tenant_id, priority):
                captured["acquired"] = (tenant_id, priority)
                raise AssertionError("quota rejection should happen before acquire")

        qos = FakeQoS()
        monkeypatch.setattr("strata.transforms.build_qos.get_build_qos", lambda: qos)

        try:
            response = await materialize_artifact(
                MaterializeRequest.model_validate(
                    {
                        "inputs": ["file:///fake/table"],
                        "transform": {
                            "executor": "local://duckdb_sql@v1",
                            "params": {"sql": "SELECT * FROM input"},
                        },
                    }
                )
            )

            assert response.status_code == 429
            assert response.body
            assert b"quota_exceeded" in response.body
            assert captured["quota_tenant_id"] == "__default__"
            assert (
                captured["quota_estimated_bytes"]
                == server_mode_config.build_runner_default_max_output
            )
        finally:
            server_module._state = original_state
            reset_artifact_store()
            reset_transform_registry()
            reset_build_store()


class TestAsyncBuildFlow:
    """Tests for async build flow in server mode."""

    def test_materialize_returns_build_id(self, server_mode_app):
        """Materialize in server mode returns build_id for polling."""
        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/table"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM t"},
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["hit"] is False
        assert data["build_id"] is not None
        assert data["state"] == "pending"
        assert data["artifact_uri"].startswith("strata://artifact/")

    def test_poll_build_status(self, server_mode_app):
        """Can poll build status using build_id."""
        # Create a build
        create_resp = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/table"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {},
                },
            },
        )

        build_id = create_resp.json()["build_id"]

        # Poll status
        status_resp = server_mode_app.get(f"/v1/artifacts/builds/{build_id}")

        assert status_resp.status_code == 200
        data = status_resp.json()

        assert data["build_id"] == build_id
        assert data["state"] == "pending"
        assert data["executor_ref"] == "duckdb_sql@v1"
        assert data["created_at"] > 0

    def test_poll_nonexistent_build(self, server_mode_app):
        """Polling nonexistent build returns 404."""
        response = server_mode_app.get("/v1/artifacts/builds/nonexistent-id")

        assert response.status_code == 404

    def test_build_polling_nonexistent_build_in_personal_mode(self, personal_mode_app):
        """Personal mode exposes build polling, but missing builds still return 404."""
        response = personal_mode_app.get("/v1/artifacts/builds/some-id")

        assert response.status_code == 404
        assert response.json()["detail"] == "Build not found"


class TestProvenanceDeduplication:
    """Tests for provenance-based deduplication in server mode."""

    def test_same_inputs_same_provenance(self, server_mode_app):
        """Same inputs + transform should have same provenance."""
        # First materialize
        resp1 = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/table"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM t"},
                },
            },
        )

        artifact_uri = resp1.json()["artifact_uri"]

        # Simulate build completion by directly updating the artifact store
        from strata.artifact_store import get_artifact_store

        store = get_artifact_store()
        assert store is not None

        # Parse artifact_id and version from URI
        import re

        match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", artifact_uri)
        assert match is not None
        artifact_id = match.group(1)
        version = int(match.group(2))

        # Write dummy blob and finalize
        store.write_blob(artifact_id, version, b"dummy data")
        store.finalize_artifact(artifact_id, version, "{}", 10, 10)

        # Second materialize with same inputs
        resp2 = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/table"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM t"},
                },
            },
        )

        # Should be a cache hit
        data2 = resp2.json()
        assert data2["hit"] is True
        assert data2["artifact_uri"] == artifact_uri
        assert data2["state"] == "ready"

    def test_named_inputs_resolve_with_tenant_context(self, server_mode_auth_app):
        """Tenant-scoped name inputs should drive provenance and rebuilds correctly."""
        store = get_artifact_store()
        assert store is not None

        version_a1 = store.create_artifact(
            artifact_id="team-a-input-v1",
            provenance_hash="team-a-input-v1",
            tenant="team-a",
        )
        store.write_blob("team-a-input-v1", version_a1, b"a1")
        store.finalize_artifact("team-a-input-v1", version_a1, "{}", 1, 2)
        store.set_name("shared-input", "team-a-input-v1", version_a1, tenant="team-a")

        response1 = server_mode_auth_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["strata://name/shared-input"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM input0"},
                },
            },
            headers=_auth_headers("team-a"),
        )
        assert response1.status_code == 200
        first_uri = response1.json()["artifact_uri"]
        first_artifact_id, first_version = first_uri.removeprefix("strata://artifact/").split("@v=")
        first_artifact = store.get_artifact(first_artifact_id, int(first_version))
        assert first_artifact is not None
        assert first_artifact.input_versions is not None
        assert json.loads(first_artifact.input_versions)["strata://name/shared-input"] == (
            f"team-a-input-v1@v={version_a1}"
        )
        materialized_bytes = b"materialized-a1"
        store.write_blob(first_artifact_id, int(first_version), materialized_bytes)
        store.finalize_artifact(
            first_artifact_id,
            int(first_version),
            "{}",
            1,
            len(materialized_bytes),
        )

        version_a2 = store.create_artifact(
            artifact_id="team-a-input-v2",
            provenance_hash="team-a-input-v2",
            tenant="team-a",
        )
        store.write_blob("team-a-input-v2", version_a2, b"a2")
        store.finalize_artifact("team-a-input-v2", version_a2, "{}", 1, 2)
        store.set_name("shared-input", "team-a-input-v2", version_a2, tenant="team-a")

        response2 = server_mode_auth_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["strata://name/shared-input"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM input0"},
                },
            },
            headers=_auth_headers("team-a"),
        )

        assert response2.status_code == 200
        assert response2.json()["hit"] is False
        assert response2.json()["artifact_uri"] != first_uri


class TestServerModeConfig:
    """Tests for server-mode configuration."""

    def test_server_transforms_enabled_property(self, tmp_path):
        """server_transforms_enabled returns True with proper config."""
        config = StrataConfig(
            deployment_mode="service",
            artifact_dir=tmp_path / "artifacts",  # transforms persist; store required
            transforms_config={"enabled": True},
        )
        assert config.server_transforms_enabled is True

    def test_server_transforms_disabled_by_default(self):
        """server_transforms_enabled is False by default."""
        config = StrataConfig(deployment_mode="service")
        assert config.server_transforms_enabled is False

    def test_server_transforms_disabled_in_personal_mode(self):
        """server_transforms_enabled is False in personal mode."""
        config = StrataConfig(
            deployment_mode="personal",
            transforms_config={"enabled": True},
        )
        assert config.server_transforms_enabled is False

    def test_transform_registry_from_config(self):
        """TransformRegistry.from_config parses properly."""
        config = {
            "enabled": True,
            "registry": [
                {
                    "ref": "duckdb_sql@v1",
                    "executor_url": "http://exec:8080",
                    "timeout_seconds": 600,
                    "max_output_bytes": 1024000,
                },
            ],
        }

        registry = TransformRegistry.from_config(config)

        assert registry.enabled is True
        assert len(registry.definitions) == 1

        defn = registry.definitions[0]
        assert defn.ref == "duckdb_sql@v1"
        assert defn.executor_url == "http://exec:8080"
        assert defn.timeout_seconds == 600
        assert defn.max_output_bytes == 1024000


class TestMixedModeScenarios:
    """Tests for mixed scenarios (e.g., server mode with auth)."""

    def test_materialize_without_transforms_enabled(self, tmp_path):
        """Materialize in service mode without transforms returns 403."""
        from unittest.mock import MagicMock

        import strata.server as server_module
        from strata.server import app

        # Reset singletons
        reset_artifact_store()
        reset_transform_registry()
        reset_build_store()

        # Config: service mode, no transforms
        config = StrataConfig(
            deployment_mode="service",
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            # transforms_config not enabled
        )
        (tmp_path / "artifacts").mkdir()

        mock_state = MagicMock()
        mock_state.config = config

        original_state = server_module._state
        server_module._state = mock_state

        client = TestClient(app)

        try:
            response = client.post(
                "/v1/artifacts/materialize",
                json={
                    "inputs": ["file:///fake/table"],
                    "transform": {"executor": "duckdb_sql@v1", "params": {}},
                },
            )

            assert response.status_code == 403
            assert response.json()["detail"]["error"] == "writes_disabled"
        finally:
            server_module._state = original_state
            reset_artifact_store()
            reset_transform_registry()
            reset_build_store()


class TestServiceModeReviewFindings:
    """Regressions for the service-mode review findings (authz + manifest)."""

    def test_materialize_propagates_denied_input_authz(self, server_mode_app, monkeypatch):
        """A 403 from input resolution (e.g. a table-ACL deny) must propagate.

        Regression: ``materialize_artifact`` caught every ``HTTPException`` from
        ``_resolve_input_version`` and fell back to the raw URI, so a denied
        table input still created a build.
        """
        from fastapi import HTTPException

        def deny(*_args, **_kwargs):
            raise HTTPException(status_code=403, detail="Access denied")

        monkeypatch.setattr("strata.server._resolve_input_version", deny)

        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///denied/table"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 1"},
                },
            },
        )
        assert response.status_code == 403

    def test_materialize_falls_back_on_unresolvable_input(self, server_mode_app, monkeypatch):
        """A 400 (unresolvable / legacy fake URI) still falls back to the raw
        URI so the build proceeds — only authz / not-found short-circuit."""
        from fastapi import HTTPException

        def unresolvable(*_args, **_kwargs):
            raise HTTPException(status_code=400, detail="Unknown input URI type")

        monkeypatch.setattr("strata.server._resolve_input_version", unresolvable)

        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["weird://thing"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 1"},
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["build_id"] is not None

    def test_materialize_build_carries_inputs_and_params_into_manifest(self, server_mode_app):
        """The build created by the materialize route must carry input_uris and
        params, so the pull-model manifest isn't empty.

        Regression: ``create_build`` was called without ``input_uris`` / ``params``,
        producing an empty-input, empty-param manifest.
        """
        from strata.transforms.signed_urls import reset_signing_secret, set_signing_secret

        set_signing_secret(b"test-secret-key-12345678901234")
        try:
            create = server_mode_app.post(
                "/v1/artifacts/materialize",
                json={
                    "inputs": ["strata://artifact/seed@v=1"],
                    "transform": {
                        "executor": "duckdb_sql@v1",
                        "params": {"sql": "SELECT * FROM input"},
                    },
                },
            )
            assert create.status_code == 200, create.text
            build_id = create.json()["build_id"]
            assert build_id is not None

            manifest = server_mode_app.get(f"/v1/builds/{build_id}/manifest")
            assert manifest.status_code == 200, manifest.text
            data = manifest.json()

            # Inputs flow through (was empty before the fix).
            assert len(data["inputs"]) == 1
            assert data["inputs"][0]["artifact_id"] == "seed"
            assert data["inputs"][0]["version"] == 1
            # Params flow through.
            assert data["metadata"]["params"] == {"sql": "SELECT * FROM input"}
        finally:
            reset_signing_secret()

    def test_registry_reads_work_in_service_mode(self, server_mode_app):
        """Registry read routes must serve in service mode (allow_read=True),
        not 403 like the neighboring write routes."""
        for path in ("/v1/names", "/v1/registry/summary", "/v1/registry/audit"):
            response = server_mode_app.get(path)
            assert response.status_code == 200, (path, response.status_code, response.text)

    def test_admin_tenants_requires_admin_scope(self, server_mode_auth_app):
        """The cross-tenant admin observability routes require an admin scope
        under trusted-proxy auth (regression: they were ungated)."""
        # No admin scope -> 403.
        denied = server_mode_auth_app.get("/v1/admin/tenants", headers=_auth_headers())
        assert denied.status_code == 403
        denied_one = server_mode_auth_app.get("/v1/admin/tenants/team-a", headers=_auth_headers())
        assert denied_one.status_code == 403

        # admin:tenants grants access.
        ok = server_mode_auth_app.get(
            "/v1/admin/tenants", headers=_auth_headers(scopes="admin:tenants")
        )
        assert ok.status_code == 200
        assert "tenants" in ok.json()

        # admin:* also grants access (wildcard).
        ok_wild = server_mode_auth_app.get(
            "/v1/admin/tenants", headers=_auth_headers(scopes="admin:*")
        )
        assert ok_wild.status_code == 200
