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


def _name_ref(base: str, name: str, headers: dict) -> tuple[str, int]:
    """Resolve a name to its (artifact_id, version)."""
    resp = httpx.get(f"{base}/v1/names/{name}", headers=headers, timeout=30.0)
    assert resp.status_code == 200, resp.text
    ref = resp.json()["artifact_uri"].removeprefix("strata://artifact/")
    art_id, version = ref.split("@v=")
    return art_id, int(version)


def _request_champion(base: str, art_id: str, version: int, headers: dict) -> httpx.Response:
    return httpx.put(
        f"{base}/v1/names/team/model/aliases/champion",
        json={"artifact_id": art_id, "version": version},
        headers=headers,
        timeout=30.0,
    )


def test_protected_alias_approval_requires_scope_and_distinct_approver(tmp_path):
    """The registry governance path for the shared store: a protected alias
    (``champion``) queues for approval, and deciding it requires the
    ``admin:registry`` scope *and* a distinct approver — the requester can't
    self-approve. This is the multi-tenant security claim, untested until now."""
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
        registry_protected_aliases=["champion"],
    ) as ctx:
        base = ctx.base_url
        model = pa.table({"weight": [0.1, 0.2, 0.3]})

        # Researcher alice publishes the model and requests the protected alias.
        assert (
            _publish(
                base, model, "team/model", _headers("team-a", "alice", "artifacts:write")
            ).status_code
            == 200
        )
        art_id, version = _name_ref(base, "team/model", _headers("team-a", "alice"))

        req = _request_champion(
            base, art_id, version, _headers("team-a", "alice", "artifacts:write")
        )
        assert req.status_code == 202  # protected → queued, not applied
        assert req.json()["status"] == "pending"

        body = {"name": "team/model", "alias": "champion"}

        # (1) Approve without admin:registry → 403.
        no_scope = httpx.post(
            f"{base}/v1/registry/pending/approve", json=body, headers=_headers("team-a", "frank")
        )
        assert no_scope.status_code == 403

        # (2) The requester cannot self-approve, even with admin:registry → 403.
        self_app = httpx.post(
            f"{base}/v1/registry/pending/approve",
            json=body,
            headers=_headers("team-a", "alice", "admin:registry"),
        )
        assert self_app.status_code == 403
        assert "Separation of duty" in self_app.json()["detail"]

        # The alias is still not applied.
        with httpx.Client() as c:
            still_pending = c.get(
                f"{base}/v1/names/team/model/aliases/champion", headers=_headers("team-a", "bob")
            )
            assert still_pending.status_code == 404

        # (3) A distinct approver with admin:registry applies it.
        ok = httpx.post(
            f"{base}/v1/registry/pending/approve",
            json=body,
            headers=_headers("team-a", "frank", "admin:registry"),
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["status"] == "approved"

        # Now champion resolves for the team.
        resolved = httpx.get(
            f"{base}/v1/names/team/model/aliases/champion", headers=_headers("team-a", "bob")
        )
        assert resolved.status_code == 200


def test_protected_alias_admin_star_is_break_glass_self_approve(tmp_path):
    """``admin:*`` is the break-glass scope: it satisfies admin:registry *and*
    waives separation of duty, so a superadmin can self-approve their own
    protected-alias request (the one-operator escape hatch)."""
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
        registry_protected_aliases=["champion"],
    ) as ctx:
        base = ctx.base_url
        model = pa.table({"weight": [1.0]})
        admin = _headers("team-a", "root", "admin:*")

        assert _publish(base, model, "team/model", admin).status_code == 200
        art_id, version = _name_ref(base, "team/model", admin)
        assert _request_champion(base, art_id, version, admin).status_code == 202

        # Same principal self-approves — allowed because admin:* is break-glass.
        ok = httpx.post(
            f"{base}/v1/registry/pending/approve",
            json={"name": "team/model", "alias": "champion"},
            headers=admin,
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["status"] == "approved"

        resolved = httpx.get(f"{base}/v1/names/team/model/aliases/champion", headers=admin)
        assert resolved.status_code == 200


def test_reject_requires_registry_scope(tmp_path):
    """Rejecting a pending change is also a governance action — a tenant member
    without ``admin:registry`` can't quietly drop a colleague's promotion."""
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
        registry_protected_aliases=["champion"],
    ) as ctx:
        base = ctx.base_url
        assert (
            _publish(
                base,
                pa.table({"w": [1.0]}),
                "team/model",
                _headers("team-a", "alice", "artifacts:write"),
            ).status_code
            == 200
        )
        art_id, version = _name_ref(base, "team/model", _headers("team-a", "alice"))
        assert (
            _request_champion(
                base, art_id, version, _headers("team-a", "alice", "artifacts:write")
            ).status_code
            == 202
        )

        body = {"name": "team/model", "alias": "champion"}
        denied = httpx.post(
            f"{base}/v1/registry/pending/reject", json=body, headers=_headers("team-a", "mallory")
        )
        assert denied.status_code == 403

        # An approver can reject it.
        ok = httpx.post(
            f"{base}/v1/registry/pending/reject",
            json=body,
            headers=_headers("team-a", "frank", "admin:registry"),
        )
        assert ok.status_code == 200
        assert ok.json()["status"] == "rejected"
        # And the alias never resolves — the change was discarded.
        gone = httpx.get(
            f"{base}/v1/names/team/model/aliases/champion",
            headers=_headers("team-a", "alice"),
        )
        assert gone.status_code == 404
