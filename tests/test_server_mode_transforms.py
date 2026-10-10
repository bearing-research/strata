"""Server-mode transforms: async materialize and build polling."""

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
    """Server-mode config with transforms enabled."""
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
                    "ref": "allowed_transform@*",
                    "executor_url": "http://allowed:8080/execute",
                },
            ],
        },
    )


@pytest.fixture
def personal_mode_config(tmp_path):
    """Personal-mode config."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    return StrataConfig(
        host="127.0.0.1",
        port=8765,
        deployment_mode="personal",
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
        notebook_storage_dir=tmp_path,
    )


@pytest.fixture
def server_mode_auth_config(tmp_path):
    """Server-mode config with trusted-proxy auth."""
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
    """A test app with the server-mode config."""
    import strata.server as server_module
    from strata.server import app

    reset_artifact_store()
    reset_transform_registry()
    reset_build_store()

    transform_registry = TransformRegistry.from_config(server_mode_config.transforms_config)
    set_transform_registry(transform_registry)

    get_artifact_store(server_mode_config.artifact_dir)

    db_path = server_mode_config.artifact_dir / "artifacts.sqlite"
    get_build_store(db_path)

    from unittest.mock import MagicMock

    from strata.transforms.signed_urls import URLSigner

    mock_state = MagicMock()
    mock_state._planning_executor = None  # the loop's default pool
    mock_state.config = server_mode_config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()
    # A real signer so signed-URL routes produce a verifiable manifest (a MagicMock
    # would serialize to a non-manifest blob).
    mock_state.url_signer = URLSigner(b"test-secret-key-12345678901234")

    original_state = server_module._state
    server_module._state = mock_state

    yield TestClient(app, headers=_auth_headers(scopes="admin:*"))

    server_module._state = original_state
    reset_artifact_store()
    reset_transform_registry()
    reset_build_store()


@pytest.fixture
def personal_mode_app(personal_mode_config):
    """A test app with the personal-mode config."""
    import strata.server as server_module
    from strata.server import app

    reset_artifact_store()
    reset_transform_registry()
    reset_build_store()

    get_artifact_store(personal_mode_config.artifact_dir)

    # Personal mode ships the embedded registry (the real lifespan does this).
    from strata.transforms.registry import TransformRegistry, set_transform_registry

    set_transform_registry(TransformRegistry.create_embedded_registry())

    from unittest.mock import MagicMock

    mock_state = MagicMock()
    mock_state._planning_executor = None  # the loop's default pool
    mock_state.config = personal_mode_config
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


@pytest.fixture
def server_mode_auth_app(server_mode_auth_config):
    """A test app with server mode and trusted-proxy auth."""
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

    from strata.transforms.signed_urls import URLSigner

    mock_state = MagicMock()
    mock_state._planning_executor = None  # the loop's default pool
    mock_state.config = server_mode_auth_config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()
    # A real signer so signed-URL routes produce a verifiable manifest (a MagicMock
    # would serialize to a non-manifest blob).
    mock_state.url_signer = URLSigner(b"test-secret-key-12345678901234")

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
    """The server-managed notebook worker admin API."""

    def test_list_notebook_workers_service_mode(self, server_mode_app):
        """Service mode exposes the server-managed notebook worker registry."""
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
        """Replacing the registry updates both the stored specs and the catalog."""
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
        """Targeted worker CRUD by name."""
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

    def test_the_built_in_worker_name_is_refused(self, server_mode_app):
        # Resolution returns the built-in first, so a registered "local" would never run.
        entry = {"name": "local", "backend": "executor", "config": {"url": "https://x.internal"}}
        refused = [
            server_mode_app.post("/v1/admin/notebook-workers", json=entry),
            server_mode_app.put("/v1/admin/notebook-workers", json={"workers": [entry]}),
        ]
        for response in refused:
            assert response.status_code == 422
            assert "reserved for the built-in worker" in response.text
        listed = server_mode_app.get("/v1/admin/notebook-workers").json()
        assert listed["configured_workers"] == []

    def test_patch_notebook_worker_enabled_state(self, server_mode_app):
        """One worker can be disabled and re-enabled."""
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
        """One worker's health can be force-refreshed by name."""
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
        """Forced refreshes accumulate a short recent probe trail."""
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
        """Admin enable/disable flows through the notebook worker APIs."""

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
        """Create, update and delete reach the notebook-visible worker policy."""

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

    def test_a_running_personal_server_takes_a_worker_through_the_admin_routes(
        self,
        personal_mode_app,
        tmp_path,
        monkeypatch,
    ):
        """A machine type enabled after the first start reaches an open notebook, no restart."""
        monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)

        async def _noop_start(self):
            del self

        monkeypatch.setattr("strata.notebook.pool.WarmProcessPool.start", _noop_start)

        notebook_dir = create_notebook(tmp_path, "Personal Registry")
        opened = personal_mode_app.post("/v1/notebooks/open", json={"path": str(notebook_dir)})
        assert opened.status_code == 200, opened.text
        session_id = opened.json()["session_id"]

        def _offered() -> dict[str, dict]:
            response = personal_mode_app.get(f"/v1/notebooks/{session_id}/workers")
            assert response.status_code == 200
            return {worker["name"]: worker for worker in response.json()["workers"]}

        assert "gpu-a100" not in _offered()

        entry = {
            "name": "gpu-a100",
            "backend": "executor",
            "runtime_id": "gpu-a100-v1",
            "config": {"url": "https://gpu.internal/v1/execute"},
        }
        # The origin guard still refuses a page the owner happens to visit.
        cross_origin = personal_mode_app.post(
            "/v1/admin/notebook-workers",
            json=entry,
            headers={"Origin": "https://evil.example"},
        )
        assert cross_origin.status_code == 403
        assert "gpu-a100" not in _offered()

        created = personal_mode_app.post("/v1/admin/notebook-workers", json=entry)
        assert created.status_code == 200
        assert [w["name"] for w in created.json()["configured_workers"]] == ["gpu-a100"]
        assert [e["name"] for e in get_artifact_store().notebook_worker_entries()] == ["gpu-a100"]

        offered = _offered()["gpu-a100"]
        assert offered["source"] == "server"
        assert offered["allowed"] is True
        assigned = personal_mode_app.put(
            f"/v1/notebooks/{session_id}/worker", json={"worker": "gpu-a100"}
        )
        assert assigned.status_code == 200

        disabled = personal_mode_app.patch(
            "/v1/admin/notebook-workers/gpu-a100", json={"enabled": False}
        )
        assert disabled.status_code == 200
        assert _offered()["gpu-a100"]["allowed"] is False

    def test_notebook_workers_admin_requires_scope(self, server_mode_auth_app):
        """Trusted-proxy mode requires the notebook worker admin scope."""
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
    """Transform allowlist validation in server mode."""

    def test_allowed_transform_succeeds(self, server_mode_app):
        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/wh#db.events"],
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
        # In server mode the server executes, so there is no build_spec.
        assert data["build_spec"] is None

    def test_unregistered_transform_rejected(self, server_mode_app):
        """An unregistered transform returns 403."""
        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/wh#db.events"],
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
        """A wildcard version in the registry matches any version."""
        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/wh#db.events"],
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
        """Personal mode runs registered transforms embedded; unknown refs are a 400.

        Accepting an unresolvable executor would park the build in 'building' forever.
        """
        response = personal_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/wh#db.events"],
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
        """A registered transform can require an explicit principal scope."""
        response = server_mode_auth_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/wh#db.events"],
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
        """A requires_scope transform runs for an authorized caller."""
        response = server_mode_auth_app.post(
            "/v1/artifacts/materialize",
            json={
                # A URI that names a table, so the table ACL (default allow) can admit it; one that
                # names none is denied under trusted-proxy auth.
                "inputs": ["file:///fake/warehouse#fake.table"],
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
        """Quota checks use a real output estimate, not zero."""
        from unittest.mock import MagicMock

        import strata.server as server_module
        from strata.api.routers.materialize import materialize_artifact
        from strata.auth import principal_context
        from strata.types import MaterializeRequest, Principal

        reset_artifact_store()
        reset_transform_registry()
        reset_build_store()

        transform_registry = TransformRegistry.from_config(server_mode_config.transforms_config)
        set_transform_registry(transform_registry)
        get_artifact_store(server_mode_config.artifact_dir)
        get_build_store(server_mode_config.artifact_dir / "artifacts.sqlite")

        mock_state = MagicMock()
        mock_state._planning_executor = None  # the loop's default pool
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
            with principal_context(Principal(id="user-1", scopes=frozenset({"admin:*"}))):
                response = await materialize_artifact(
                    MaterializeRequest.model_validate(
                        {
                            "inputs": ["file:///fake/wh#db.events"],
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
    """The async build flow in server mode."""

    def test_materialize_returns_build_id(self, server_mode_app):
        """Materialize returns a build_id for polling."""
        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/wh#db.events"],
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
        create_resp = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/wh#db.events"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {},
                },
            },
        )

        build_id = create_resp.json()["build_id"]

        status_resp = server_mode_app.get(f"/v1/artifacts/builds/{build_id}")

        assert status_resp.status_code == 200
        data = status_resp.json()

        assert data["build_id"] == build_id
        assert data["state"] == "pending"
        assert data["executor_ref"] == "duckdb_sql@v1"
        assert data["created_at"] > 0

    def test_poll_nonexistent_build(self, server_mode_app):
        response = server_mode_app.get("/v1/artifacts/builds/nonexistent-id")

        assert response.status_code == 404

    def test_build_polling_nonexistent_build_in_personal_mode(self, personal_mode_app):
        """Personal mode exposes build polling; a missing build is still 404."""
        response = personal_mode_app.get("/v1/artifacts/builds/some-id")

        assert response.status_code == 404
        assert response.json()["detail"] == "Build not found"


class TestProvenanceDeduplication:
    """Provenance-based deduplication in server mode."""

    def test_same_inputs_same_provenance(self, server_mode_app):
        """Same inputs and transform give the same provenance."""
        resp1 = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/wh#db.events"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM t"},
                },
            },
        )

        artifact_uri = resp1.json()["artifact_uri"]

        # Simulate build completion directly in the artifact store.
        from strata.artifact_store import get_artifact_store

        store = get_artifact_store()
        assert store is not None

        import re

        match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", artifact_uri)
        assert match is not None
        artifact_id = match.group(1)
        version = int(match.group(2))

        store.write_blob(artifact_id, version, b"dummy data")
        store.finalize_artifact(artifact_id, version, "{}", 10, 10)

        resp2 = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["file:///fake/wh#db.events"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM t"},
                },
            },
        )

        data2 = resp2.json()
        assert data2["hit"] is True
        assert data2["artifact_uri"] == artifact_uri
        assert data2["state"] == "ready"

    def test_named_inputs_resolve_with_tenant_context(self, server_mode_auth_app):
        """Tenant-scoped name inputs drive provenance and rebuilds."""
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
    def test_server_transforms_enabled_property(self, tmp_path):
        """server_transforms_enabled is True with the right config."""
        config = StrataConfig(
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            artifact_dir=tmp_path / "artifacts",  # transforms persist; store required
            transforms_config={"enabled": True},
        )
        assert config.server_transforms_enabled is True

    def test_server_transforms_disabled_by_default(self):
        config = StrataConfig(
            deployment_mode="service", auth_mode="trusted_proxy", proxy_token="test-token"
        )
        assert config.server_transforms_enabled is False

    def test_server_transforms_disabled_in_personal_mode(self):
        config = StrataConfig(
            deployment_mode="personal",
            transforms_config={"enabled": True},
        )
        assert config.server_transforms_enabled is False

    def test_transform_registry_from_config(self):
        """TransformRegistry.from_config parses the config."""
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
    """Mixed scenarios, such as server mode with auth."""

    def test_materialize_without_transforms_enabled(self, tmp_path):
        """Service mode without transforms returns 403."""
        from unittest.mock import MagicMock

        import strata.server as server_module
        from strata.server import app

        reset_artifact_store()
        reset_transform_registry()
        reset_build_store()

        config = StrataConfig(
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
        )
        (tmp_path / "artifacts").mkdir()

        mock_state = MagicMock()
        mock_state._planning_executor = None  # the loop's default pool
        mock_state.config = config

        original_state = server_module._state
        server_module._state = mock_state

        client = TestClient(app, headers=_auth_headers(scopes="admin:*"))

        try:
            response = client.post(
                "/v1/artifacts/materialize",
                json={
                    "inputs": ["file:///fake/wh#db.events"],
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
    """Service-mode authz and manifest regressions."""

    def test_materialize_propagates_denied_input_authz(self, server_mode_app, monkeypatch):
        """A 403 from input resolution (such as a table-ACL deny) must propagate.

        Falling back to the raw URI would still create a build for a denied input.
        """
        from fastapi import HTTPException

        async def deny(*_args, **_kwargs):
            raise HTTPException(status_code=403, detail="Access denied")

        monkeypatch.setattr("strata.api.routers.materialize.resolve_input_version", deny)

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

    def test_materialize_refuses_an_unresolvable_input(self, server_mode_app, monkeypatch):
        """A 400 from input resolution is the answer; the raw URI is never built past."""
        from fastapi import HTTPException

        async def unresolvable(*_args, **_kwargs):
            raise HTTPException(status_code=400, detail="Unknown input URI type")

        monkeypatch.setattr("strata.api.routers.materialize.resolve_input_version", unresolvable)

        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["strata://other/thing"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 1"},
                },
            },
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Unknown input URI type"
        assert get_artifact_store().stats()["total_versions"] == 0

    def test_materialize_surfaces_a_failed_table_plan(self, server_mode_app):
        """A plan that fails for a table input answers with the planner's 400.

        Building past it would record a snapshot-less version, and a later request whose plan
        also fails would dedup onto that result after the table advanced.
        """
        import strata.server as server_module

        server_module._state.planner.plan.side_effect = RuntimeError(
            "Failed to read Parquet metadata"
        )

        response = server_mode_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["s3://lake/wh#taxi.trips"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 1"},
                },
            },
        )
        assert response.status_code == 400, response.text
        assert "Failed to read Parquet metadata" in response.json()["detail"]
        assert get_artifact_store().stats()["total_versions"] == 0

    @staticmethod
    def _blocking_planner(entered, release, plan_done):
        def plan(**_kwargs):
            entered.set()
            # A guard, so a plan stuck on the loop fails the test rather than hanging it.
            release.wait(timeout=30)
            plan_done.set()
            raise RuntimeError("catalog unreachable")

        return plan

    async def test_a_table_input_plans_off_the_event_loop(self, server_mode_app):
        """While a transform's table input plans, the server keeps answering other requests."""
        import asyncio
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module

        entered, release, plan_done = threading.Event(), threading.Event(), threading.Event()
        state = server_module._state
        state.planner.plan.side_effect = self._blocking_planner(entered, release, plan_done)
        state._planning_executor = ThreadPoolExecutor(max_workers=1)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=server_module.app),
                base_url="http://test",
                headers=_auth_headers(scopes="admin:*"),
            ) as client:
                materialize = asyncio.create_task(
                    client.post(
                        "/v1/artifacts/materialize",
                        json={
                            "inputs": ["s3://lake/wh#taxi.trips"],
                            "transform": {
                                "executor": "duckdb_sql@v1",
                                "params": {"sql": "SELECT 1"},
                            },
                        },
                    )
                )
                assert await asyncio.to_thread(entered.wait, 30)
                other = await client.get("/v1/builds/no-such-build")
                served_while_planning = not plan_done.is_set()
                release.set()
                response = await materialize
        finally:
            release.set()
            state._planning_executor.shutdown(wait=True)

        assert other.status_code == 404
        assert served_while_planning
        assert response.status_code == 400
        assert "catalog unreachable" in response.json()["detail"]

    async def test_a_table_input_plan_past_the_timeout_is_a_504(
        self, server_mode_app, server_mode_config
    ):
        """The transform-input plan gets the scan path's ``plan_timeout_seconds`` and 504."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module

        entered, release, plan_done = threading.Event(), threading.Event(), threading.Event()
        state = server_module._state
        state.config = server_mode_config.model_copy(update={"plan_timeout_seconds": 0.05})
        state.planner.plan.side_effect = self._blocking_planner(entered, release, plan_done)
        state._planning_executor = ThreadPoolExecutor(max_workers=1)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=server_module.app),
                base_url="http://test",
                headers=_auth_headers(scopes="admin:*"),
            ) as client:
                response = await client.post(
                    "/v1/artifacts/materialize",
                    json={
                        "inputs": ["s3://lake/wh#taxi.trips"],
                        "transform": {"executor": "duckdb_sql@v1", "params": {"sql": "SELECT 1"}},
                    },
                )
                answered_before_the_plan_ended = not plan_done.is_set()
        finally:
            release.set()
            state._planning_executor.shutdown(wait=True)

        assert response.status_code == 504, response.text
        assert "Planning timed out" in response.json()["detail"]
        assert answered_before_the_plan_ended
        assert get_artifact_store().stats()["total_versions"] == 0

    def test_materialize_build_carries_inputs_and_params_into_manifest(self, server_mode_auth_app):
        """The build carries input_uris and params, so the pull-model manifest is not empty."""
        # Artifact inputs resolve through the store (tenant-gated), so a fictional id is a
        # 404. It must carry the caller's tenant: a tenantless artifact persists with
        # tenant='', which the gate compares against the request's tenant.
        store = get_artifact_store()
        store.create_artifact(artifact_id="seed", provenance_hash="seed-prov", tenant="team-a")

        create = server_mode_auth_app.post(
            "/v1/artifacts/materialize",
            json={
                "inputs": ["strata://artifact/seed@v=1"],
                "transform": {
                    "executor": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM input"},
                },
            },
            headers=_auth_headers(),
        )
        assert create.status_code == 200, create.text
        build_id = create.json()["build_id"]
        assert build_id is not None

        manifest = server_mode_auth_app.get(
            f"/v1/builds/{build_id}/manifest", headers=_auth_headers()
        )
        assert manifest.status_code == 200, manifest.text
        data = manifest.json()

        assert len(data["inputs"]) == 1
        assert data["inputs"][0]["artifact_id"] == "seed"
        assert data["inputs"][0]["version"] == 1
        assert data["metadata"]["params"] == {"sql": "SELECT * FROM input"}

    def test_materialize_name_takes_the_write_gate(
        self, server_mode_auth_app, server_mode_auth_config
    ):
        """``name`` writes a registry name when the build finalizes, so it needs write access."""
        body = {
            "inputs": ["file:///fake/warehouse#fake.table"],
            "transform": {"executor": "duckdb_sql@v1", "params": {"sql": "SELECT 1"}},
            "name": "prod",
        }
        writer = _auth_headers(scopes="artifacts:write")

        off = server_mode_auth_app.post("/v1/artifacts/materialize", json=body, headers=writer)
        assert off.status_code == 403
        assert off.json()["detail"]["error"] == "writes_disabled"

        server_mode_auth_config.service_writes_enabled = True
        unscoped = server_mode_auth_app.post(
            "/v1/artifacts/materialize", json=body, headers=_auth_headers()
        )
        assert unscoped.status_code == 403
        assert unscoped.json()["detail"]["error"] == "missing_scope"
        allowed = server_mode_auth_app.post("/v1/artifacts/materialize", json=body, headers=writer)
        assert allowed.status_code == 200, allowed.text

    def test_registry_reads_work_in_service_mode(self, server_mode_app):
        """Registry read routes serve in service mode (allow_read=True), unlike the write routes."""
        for path in ("/v1/names", "/v1/registry/summary", "/v1/registry/audit"):
            response = server_mode_app.get(path)
            assert response.status_code == 200, (path, response.status_code, response.text)

    def test_admin_tenants_requires_admin_scope(self, server_mode_auth_app):
        """The cross-tenant admin routes require an admin scope under trusted-proxy auth."""
        # No admin scope: 403.
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
