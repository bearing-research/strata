"""API-key auth, the admin worker routes and the health, metrics and metadata routes make their
store calls off the event loop.

A store call that waits (a SQLite write lock, a Postgres round trip) made inline would stall every
request the server is serving. Each test blocks one request's store call on an event and checks
that another request completes meanwhile.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import anyio.to_thread
import pytest
from httpx import ASGITransport, AsyncClient

from strata.artifact_store import get_artifact_store, reset_artifact_store
from strata.config import StrataConfig


def _gate(monkeypatch, obj, method: str) -> SimpleNamespace:
    """Make ``obj.method`` block until ``release`` is set; it then runs as before."""
    gate = SimpleNamespace(
        entered=threading.Event(), release=threading.Event(), done=threading.Event()
    )
    original = getattr(obj, method)

    def gated(*args, **kwargs):
        gate.entered.set()
        # A guard, so a call stuck on the loop fails the test rather than hanging it.
        gate.release.wait(timeout=30)
        try:
            return original(*args, **kwargs)
        finally:
            gate.done.set()

    monkeypatch.setattr(obj, method, gated)
    return gate


async def _blocked_while_health_answers(app, gate, method, path, **kwargs):
    """Send one request, wait until its store call blocks, then GET ``/health``."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        blocked = asyncio.ensure_future(c.request(method, path, **kwargs))
        try:
            assert await anyio.to_thread.run_sync(gate.entered.wait, 30), "store call not made"
            other = await c.get("/health")
            other_finished_first = not gate.done.is_set()
        finally:
            gate.release.set()
        response = await blocked

    assert other.status_code == 200
    assert other_finished_first, "the second request waited for the blocked store call"
    return response


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """Install a server state built from ``StrataConfig(**overrides)``; yield the app."""
    import strata.server as server_module
    from strata.api_keys import reset_api_key_store
    from strata.metadata_cache import reset_caches
    from strata.server import ServerState
    from strata.tenant_registry import reset_tenant_registry

    states = []

    def start(**overrides):
        config = StrataConfig(
            host="127.0.0.1",
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            metadata_db=tmp_path / "meta.sqlite",
            rate_limit_enabled=False,
            **overrides,
        )
        state = ServerState(config)
        states.append(state)
        monkeypatch.setattr(server_module, "_state", state)
        return server_module.app

    reset_artifact_store()
    reset_api_key_store()
    reset_caches()
    reset_tenant_registry()
    try:
        yield start
    finally:
        for state in states:
            state.streams.shutdown_cleanups()
            state._planning_executor.shutdown(wait=False)
            state._fetch_executor.shutdown(wait=False)
        reset_artifact_store()
        reset_api_key_store()
        reset_caches()
        reset_tenant_registry()


async def test_api_key_verify_does_not_block_the_loop(serve, tmp_path, monkeypatch):
    from strata.api_keys import get_api_key_store

    app = serve(deployment_mode="service", auth_mode="api_key")
    keys = get_api_key_store(tmp_path / "keys.sqlite")
    key, _ = keys.create_key(principal_id="svc")
    gate = _gate(monkeypatch, keys, "verify")

    response = await _blocked_while_health_answers(
        app, gate, "GET", "/v1/config/timeouts", headers={"Authorization": f"Bearer {key}"}
    )

    assert response.status_code == 200, response.text
    assert "planning" in response.json()


async def test_a_revoked_key_is_refused_after_the_offload(serve, tmp_path):
    from strata.api_keys import get_api_key_store

    app = serve(deployment_mode="service", auth_mode="api_key")
    keys = get_api_key_store(tmp_path / "keys.sqlite")
    key, record = keys.create_key(principal_id="svc")
    headers = {"Authorization": f"Bearer {key}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/v1/config/timeouts", headers=headers)).status_code == 200
        keys.revoke(record.key_id)
        assert (await c.get("/v1/config/timeouts", headers=headers)).status_code == 401


@pytest.fixture
def workers_app(serve, tmp_path):
    """A personal server whose worker registry holds one local worker, ``gpu``."""
    from strata.notebook.models import WorkerSpec
    from strata.notebook.workers import ManagedWorkerRecord, replace_server_managed_worker_records

    app = serve(deployment_mode="personal")
    store = get_artifact_store(tmp_path / "artifacts")
    replace_server_managed_worker_records([ManagedWorkerRecord(worker=WorkerSpec(name="gpu"))])
    return SimpleNamespace(app=app, store=store)


@pytest.mark.parametrize(
    ("gated", "method", "path", "kwargs", "check"),
    [
        pytest.param(
            "notebook_worker_entries",
            "GET",
            "/v1/admin/notebook-workers",
            {},
            lambda body: [w["name"] for w in body["configured_workers"]] == ["gpu"],
            id="list",
        ),
        pytest.param(
            "update_notebook_workers",
            "PATCH",
            "/v1/admin/notebook-workers/gpu",
            {"json": {"enabled": False}},
            lambda body: body["configured_workers"][0]["enabled"] is False,
            id="patch",
        ),
        pytest.param(
            "update_notebook_workers",
            "DELETE",
            "/v1/admin/notebook-workers/gpu",
            {},
            lambda body: body["configured_workers"] == [],
            id="delete",
        ),
    ],
)
async def test_admin_worker_route_store_call_does_not_block_the_loop(
    workers_app, monkeypatch, gated, method, path, kwargs, check
):
    gate = _gate(monkeypatch, workers_app.store, gated)

    response = await _blocked_while_health_answers(workers_app.app, gate, method, path, **kwargs)

    assert response.status_code == 200, response.text
    assert check(response.json())


async def test_admin_worker_route_still_maps_a_missing_worker_to_404(workers_app):
    async with AsyncClient(
        transport=ASGITransport(app=workers_app.app), base_url="http://test"
    ) as c:
        response = await c.patch("/v1/admin/notebook-workers/nope", json={"enabled": False})

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("gated", "method", "path", "check"),
    [
        pytest.param(
            "stats",
            "GET",
            "/health/ready",
            lambda r: r.json()["checks"]["metadata_store"] is True,
            id="ready",
        ),
        pytest.param(
            "stats",
            "GET",
            "/health/dependencies",
            lambda r: (
                {c["name"]: c["status"] for c in r.json()["checks"]}["metadata_store"] == "healthy"
            ),
            id="dependencies",
        ),
        pytest.param(
            "stats",
            "GET",
            "/metrics/prometheus",
            lambda r: "strata_metadata_manifest_hits_total" in r.text,
            id="prometheus",
        ),
        pytest.param(
            "stats",
            "GET",
            "/v1/metadata/stats",
            lambda r: r.json()["metadata_store"]["manifest_entries"] == 0,
            id="metadata-stats",
        ),
        pytest.param(
            "cleanup_stale_parquet_meta",
            "POST",
            "/v1/metadata/cleanup",
            lambda r: r.json()["stale_entries_removed"] == 0,
            id="metadata-cleanup",
        ),
    ],
)
async def test_metadata_store_call_does_not_block_the_loop(
    serve, monkeypatch, gated, method, path, check
):
    from strata.metadata_cache import get_metadata_store
    from strata.server import get_state

    app = serve(deployment_mode="personal")
    get_state().config.cache_dir.mkdir(parents=True, exist_ok=True)
    gate = _gate(monkeypatch, get_metadata_store(get_state().config.cache_dir), gated)

    response = await _blocked_while_health_answers(app, gate, method, path)

    assert response.status_code == 200, response.text
    assert check(response)


async def test_a_metadata_store_that_never_answers_is_not_ready(serve, monkeypatch):
    from strata.api.routers import metrics_health
    from strata.metadata_cache import get_metadata_store
    from strata.server import get_state

    serve(deployment_mode="personal")
    store = get_metadata_store(get_state().config.cache_dir)
    release = threading.Event()
    monkeypatch.setattr(store, "stats", lambda: release.wait())
    monkeypatch.setattr(metrics_health, "ARTIFACT_STORE_PROBE_TIMEOUT_SECONDS", 0.01)

    try:
        response = await metrics_health.health_ready()
    finally:
        release.set()

    body = json.loads(response.body)
    assert response.status_code == 503
    assert body["checks"]["metadata_store"] is False
    assert body["checks"]["metadata_store_error"] == "TimeoutError"
