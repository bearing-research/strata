"""End-to-end: the shared research store (W4).

Mirrors the deployment scenario exactly, over a real HTTP server in genuine
service mode with authenticated write-back + multi-tenancy:

  - Researcher A (team-a, `artifacts:write`) PUBLISHES a dataset under a name.
  - Teammate B (team-a) RESOLVES the name and READS the data.
  - Other-team C (team-b) is DENIED — tenant isolation.
  - A team-a member WITHOUT the write scope cannot publish.
"""

import json

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc

from tests.conftest import run_server_with_context, table_to_ipc_bytes

PROXY_TOKEN = "shared-store-token"


def _headers(tenant: str, principal: str, scopes: str | None = None) -> dict:
    h = {
        "X-Strata-Proxy-Token": PROXY_TOKEN,
        "X-Strata-Principal": principal,
        "X-Tenant-ID": tenant,
    }
    if scopes:
        h["X-Strata-Scopes"] = scopes
    return h


def _publish(base_url: str, table: pa.Table, name: str, headers: dict) -> httpx.Response:
    metadata = {
        "inputs": [],
        "transform": {"executor": "researcher_local@v1", "params": {}},
        "name": name,
    }
    files = {
        "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        "data": ("data.arrow", table_to_ipc_bytes(table), "application/vnd.apache.arrow.stream"),
    }
    return httpx.put(f"{base_url}/v1/artifacts", files=files, headers=headers, timeout=30.0)


def test_shared_research_store_publish_resolve_read_isolation(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    with run_server_with_context(
        cache_dir,
        artifact_dir,
        "service",
        auth_mode="trusted_proxy",
        proxy_token=PROXY_TOKEN,
        multi_tenant_enabled=True,
        service_writes_enabled=True,
        hide_forbidden_as_not_found=True,
    ) as ctx:
        base = ctx.base_url
        dataset = pa.table({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})

        # --- Researcher A publishes (has the write scope) ---
        pub = _publish(
            base,
            dataset,
            "team/cleaned-events",
            _headers("team-a", "alice", scopes="artifacts:write"),
        )
        assert pub.status_code == 200, pub.text

        # --- Teammate B (same team, read-only) resolves the name ---
        resolved = httpx.get(
            f"{base}/v1/names/team/cleaned-events",
            headers=_headers("team-a", "bob"),
        )
        assert resolved.status_code == 200
        artifact_uri = resolved.json()["artifact_uri"]
        # strata://artifact/{id}@v={n}
        ref = artifact_uri.removeprefix("strata://artifact/")
        art_id, version = ref.split("@v=")

        # --- ... and reads the data back ---
        data_resp = httpx.get(
            f"{base}/v1/artifacts/{art_id}/v/{version}/data",
            headers=_headers("team-a", "bob"),
        )
        assert data_resp.status_code == 200
        round_trip = ipc.open_stream(data_resp.content).read_all()
        assert round_trip.equals(dataset)

        # --- Other-team C cannot resolve team-a's name (tenant isolation) ---
        cross = httpx.get(
            f"{base}/v1/names/team/cleaned-events",
            headers=_headers("team-b", "carol"),
        )
        assert cross.status_code == 404

        # --- A team-a member WITHOUT the write scope cannot publish ---
        denied = _publish(
            base,
            dataset,
            "team/other",
            _headers("team-a", "dave"),  # no artifacts:write
        )
        assert denied.status_code == 403
