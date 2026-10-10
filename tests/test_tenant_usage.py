"""What a tenant holds in a shared store, readable by that tenant.

``GET /v1/artifacts/usage`` and ``/stats`` answer in service mode, scoped to the caller's tenant and
never across tenants.
"""

from __future__ import annotations

import json

import httpx
import pyarrow as pa
import pytest

from strata.artifact_store import ArtifactStore
from tests.conftest import (
    LIVE_SERVER_TIMEOUT,
    run_server_with_context,
    service_auth,
    table_to_ipc_bytes,
)

PROXY_TOKEN = "usage-token"


def _headers(tenant: str | None, principal: str, scopes: str | None = None) -> dict:
    headers = {"X-Strata-Proxy-Token": PROXY_TOKEN, "X-Strata-Principal": principal}
    if tenant is not None:
        headers["X-Tenant-ID"] = tenant
    if scopes:
        headers["X-Strata-Scopes"] = scopes
    return headers


def _publish(base_url: str, rows: int, headers: dict) -> None:
    metadata = {
        "inputs": [],
        "transform": {"executor": "researcher_local@v1", "params": {"rows": rows}},
    }
    table = pa.table({"id": list(range(rows))})
    files = {
        "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        "data": ("data.arrow", table_to_ipc_bytes(table), "application/vnd.apache.arrow.stream"),
    }
    response = httpx.put(f"{base_url}/v1/artifacts", files=files, headers=headers, timeout=30)
    assert response.status_code == 200, response.text


@pytest.fixture
def team_server(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    with run_server_with_context(
        tmp_path / "cache",
        artifact_dir,
        "service",
        auth_mode="trusted_proxy",
        proxy_token=PROXY_TOKEN,
        multi_tenant_enabled=True,
        service_writes_enabled=True,
    ) as ctx:
        base_url = ctx.base_url
        write = "artifacts:write"
        _publish(base_url, 3, _headers("team-a", "alice", write))
        _publish(base_url, 5, _headers("team-a", "alice", write))
        _publish(base_url, 7, _headers("team-b", "carol", write))
        # A legacy tenantless row, which belongs to no one.
        store = ArtifactStore(artifact_dir)
        version = store.create_artifact("legacy", "ab" * 32)
        store.finalize_artifact("legacy", version, "{}", 1, 10_000)
        yield {"base_url": base_url, "artifact_dir": artifact_dir}


def _bytes_of(artifact_dir, tenant: str) -> int:
    """Summed from the store, not recomputed with a copy of the query."""
    store = ArtifactStore(artifact_dir)
    return sum(
        a.byte_size or 0
        for a in store.list_artifacts(tenant=tenant, limit=1000)
        if a.state == "ready" and a.tenant == tenant
    )


@pytest.mark.parametrize("route", ["usage", "stats"])
def test_a_tenant_sees_what_it_holds(team_server, route):
    response = httpx.get(
        f"{team_server['base_url']}/v1/artifacts/{route}",
        headers=_headers("team-a", "alice"),
        timeout=LIVE_SERVER_TIMEOUT,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_versions"] == 2
    assert body["total_rows"] == 8
    assert body["total_bytes"] > 0
    assert body["total_bytes"] == _bytes_of(team_server["artifact_dir"], "team-a")


def test_tenantless_rows_are_charged_to_nobody(team_server):
    """Charging them to each tenant would bill everyone for the same bytes."""
    body = httpx.get(
        f"{team_server['base_url']}/v1/artifacts/usage",
        headers=_headers("team-b", "carol"),
        timeout=LIVE_SERVER_TIMEOUT,
    ).json()

    assert body["total_versions"] == 1
    assert body["total_bytes"] < 10_000


def test_no_tenant_is_refused(team_server):
    response = httpx.get(
        f"{team_server['base_url']}/v1/artifacts/usage",
        headers=_headers(None, "alice"),
        timeout=LIVE_SERVER_TIMEOUT,
    )

    assert response.status_code == 400


def test_naming_another_tenant_is_refused(team_server):
    response = httpx.get(
        f"{team_server['base_url']}/v1/artifacts/usage",
        params={"tenant": "team-b"},
        headers=_headers("team-a", "alice"),
        timeout=LIVE_SERVER_TIMEOUT,
    )

    assert response.status_code == 403


def test_an_admin_can_name_a_tenant(team_server):
    response = httpx.get(
        f"{team_server['base_url']}/v1/artifacts/usage",
        params={"tenant": "team-b"},
        headers=_headers("ops", "root", scopes="admin:*"),
        timeout=LIVE_SERVER_TIMEOUT,
    )

    assert response.status_code == 200, response.text
    assert response.json()["total_rows"] == 7


def test_a_service_store_without_auth_does_not_answer_for_everyone(tmp_path, monkeypatch):
    """With no authenticated caller there is no tenant, and the unscoped answer is everyone's."""
    from fastapi.testclient import TestClient

    import strata.server as server_module
    from strata.artifact_store import reset_artifact_store
    from strata.config import StrataConfig
    from strata.server import ServerState, app

    config = StrataConfig(
        deployment_mode="service",
        **service_auth(),
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )
    # Startup refuses this config; the route gate is the second line of defence.
    config = config.model_copy(update={"auth_mode": "none"})
    monkeypatch.setattr(server_module, "_state", ServerState(config))
    reset_artifact_store()
    try:
        response = TestClient(app).get("/v1/artifacts/usage")
    finally:
        reset_artifact_store()

    assert response.status_code == 403


def test_personal_mode_reports_the_whole_store(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    with run_server_with_context(tmp_path / "cache", artifact_dir, "personal") as ctx:
        store = ArtifactStore(artifact_dir)
        version = store.create_artifact("mine", "cd" * 32)
        store.finalize_artifact("mine", version, "{}", 4, 64)

        usage = httpx.get(f"{ctx.base_url}/v1/artifacts/usage", timeout=LIVE_SERVER_TIMEOUT).json()
        stats = httpx.get(f"{ctx.base_url}/v1/artifacts/stats", timeout=LIVE_SERVER_TIMEOUT).json()

    assert (usage["total_versions"], usage["total_bytes"]) == (1, 64)
    assert stats["total_rows"] == 4
