"""What a tenant holds in a shared store, readable by that tenant. Item 54.

``GET /v1/artifacts/usage`` and ``/stats`` were personal-mode only — 403 in
service mode, where "the store" is every tenant's at once. A platform metering
storage per tenant had to read the store's database itself. They now answer in
service mode, scoped to the caller's tenant and never across tenants.
"""

from __future__ import annotations

import json

import httpx
import pyarrow as pa
import pytest

from strata.artifact_store import ArtifactStore
from tests.conftest import run_server_with_context, table_to_ipc_bytes

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
        f"{team_server['base_url']}/v1/artifacts/{route}", headers=_headers("team-a", "alice")
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_versions"] == 2
    assert body["total_rows"] == 8
    assert body["total_bytes"] > 0
    assert body["total_bytes"] == _bytes_of(team_server["artifact_dir"], "team-a")


def test_tenantless_rows_are_charged_to_nobody(team_server):
    """Counting them as each tenant's would bill every tenant for the same
    bytes — 10 KB each here, on top of what they actually hold."""
    body = httpx.get(
        f"{team_server['base_url']}/v1/artifacts/usage", headers=_headers("team-b", "carol")
    ).json()

    assert body["total_versions"] == 1
    assert body["total_bytes"] < 10_000


def test_no_tenant_is_refused(team_server):
    response = httpx.get(
        f"{team_server['base_url']}/v1/artifacts/usage", headers=_headers(None, "alice")
    )

    assert response.status_code == 400


def test_naming_another_tenant_is_refused(team_server):
    response = httpx.get(
        f"{team_server['base_url']}/v1/artifacts/usage",
        params={"tenant": "team-b"},
        headers=_headers("team-a", "alice"),
    )

    assert response.status_code == 403


def test_an_admin_can_name_a_tenant(team_server):
    response = httpx.get(
        f"{team_server['base_url']}/v1/artifacts/usage",
        params={"tenant": "team-b"},
        headers=_headers("ops", "root", scopes="admin:*"),
    )

    assert response.status_code == 200, response.text
    assert response.json()["total_rows"] == 7


def test_a_service_store_without_auth_does_not_answer_for_everyone(tmp_path):
    """No authenticated caller means no tenant to scope to, and the unscoped
    answer would be every tenant's usage at once."""
    with run_server_with_context(tmp_path / "cache", tmp_path / "artifacts", "service") as ctx:
        response = httpx.get(f"{ctx.base_url}/v1/artifacts/usage")

    assert response.status_code == 403


def test_personal_mode_reports_the_whole_store(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    with run_server_with_context(tmp_path / "cache", artifact_dir, "personal") as ctx:
        store = ArtifactStore(artifact_dir)
        version = store.create_artifact("mine", "cd" * 32)
        store.finalize_artifact("mine", version, "{}", 4, 64)

        usage = httpx.get(f"{ctx.base_url}/v1/artifacts/usage").json()
        stats = httpx.get(f"{ctx.base_url}/v1/artifacts/stats").json()

    assert (usage["total_versions"], usage["total_bytes"]) == (1, 64)
    assert stats["total_rows"] == 4
