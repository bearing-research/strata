"""End-to-end shared research store, over a real service-mode server.

Researcher A (team-a, `artifacts:write`) publishes a dataset under a name; teammate B resolves and
reads it; C on team-b is denied; a team-a member without the write scope cannot publish.
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

        # Researcher A publishes (has the write scope).
        pub = _publish(
            base,
            dataset,
            "team/cleaned-events",
            _headers("team-a", "alice", scopes="artifacts:write"),
        )
        assert pub.status_code == 200, pub.text

        # Teammate B (same team, read-only) resolves the name...
        resolved = httpx.get(
            f"{base}/v1/names/team/cleaned-events",
            headers=_headers("team-a", "bob"),
        )
        assert resolved.status_code == 200
        artifact_uri = resolved.json()["artifact_uri"]
        # strata://artifact/{id}@v={n}
        ref = artifact_uri.removeprefix("strata://artifact/")
        art_id, version = ref.split("@v=")

        # ...and reads the data back.
        data_resp = httpx.get(
            f"{base}/v1/artifacts/{art_id}/v/{version}/data",
            headers=_headers("team-a", "bob"),
        )
        assert data_resp.status_code == 200
        round_trip = ipc.open_stream(data_resp.content).read_all()
        assert round_trip.equals(dataset)

        # Other-team C cannot resolve team-a's name (tenant isolation).
        cross = httpx.get(
            f"{base}/v1/names/team/cleaned-events",
            headers=_headers("team-b", "carol"),
        )
        assert cross.status_code == 404

        # A team-a member WITHOUT the write scope cannot publish.
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
    """A protected alias (``champion``) queues for approval.

    Deciding it requires ``admin:registry`` and an approver other than the requester.
    """
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
        assert req.status_code == 202  # protected, so queued rather than applied
        assert req.json()["status"] == "pending"

        body = {"name": "team/model", "alias": "champion"}

        # (1) Approve without admin:registry: 403.
        no_scope = httpx.post(
            f"{base}/v1/registry/pending/approve", json=body, headers=_headers("team-a", "frank")
        )
        assert no_scope.status_code == 403

        # (2) The requester cannot self-approve, even with admin:registry: 403.
        self_app = httpx.post(
            f"{base}/v1/registry/pending/approve",
            json=body,
            headers=_headers("team-a", "alice", "admin:registry"),
        )
        assert self_app.status_code == 403
        assert "Separation of duty" in self_app.json()["detail"]

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

        resolved = httpx.get(
            f"{base}/v1/names/team/model/aliases/champion", headers=_headers("team-a", "bob")
        )
        assert resolved.status_code == 200

        # A follower of the feed sees the request and its approval; the refused
        # approvals above changed nothing and are not on it.
        assert _champion_events(base, "team-a") == [
            ("alias_request_set", "alice"),
            ("alias_approved", "frank"),
            ("alias_set", "frank"),
        ]
        assert _champion_events(base, "team-b") == []


def _champion_events(base: str, tenant: str) -> list[tuple[str, str]]:
    """The champion alias's events on the store's feed, as one tenant's member reads it."""
    resp = httpx.get(f"{base}/v1/events", headers=_headers(tenant, "reader"), timeout=30.0)
    assert resp.status_code == 200, resp.text
    return [(e["action"], e["actor"]) for e in resp.json()["events"] if e["alias"] == "champion"]


def test_protected_alias_admin_star_is_break_glass_self_approve(tmp_path):
    """``admin:*`` is break-glass: it satisfies admin:registry and allows self-approval."""
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

        # Self-approval is allowed here because admin:* is break-glass.
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
    """Rejecting is governance too: a member without ``admin:registry`` cannot drop a promotion."""
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

        ok = httpx.post(
            f"{base}/v1/registry/pending/reject",
            json=body,
            headers=_headers("team-a", "frank", "admin:registry"),
        )
        assert ok.status_code == 200
        assert ok.json()["status"] == "rejected"
        # The alias never resolves: the change was discarded.
        gone = httpx.get(
            f"{base}/v1/names/team/model/aliases/champion",
            headers=_headers("team-a", "alice"),
        )
        assert gone.status_code == 404
        assert _champion_events(base, "team-a") == [
            ("alias_request_set", "alice"),
            ("alias_rejected", "frank"),
        ]


def test_writes_to_another_tenants_artifact_answer_like_reads(tmp_path):
    """Naming, tagging or aliasing another tenant's artifact is a 404, as reading it is.

    A 400 naming the owning tenant told the caller the artifact exists and whose it is.
    """
    with run_server_with_context(
        tmp_path / "cache",
        tmp_path / "artifacts",
        "service",
        auth_mode="trusted_proxy",
        proxy_token=PROXY_TOKEN,
        multi_tenant_enabled=True,
        service_writes_enabled=True,
        registry_protected_aliases=["champion"],
    ) as ctx:
        base = ctx.base_url
        owner = _headers("team-a", "alice", "artifacts:write")
        other = _headers("team-b", "bob", "artifacts:write")
        assert _publish(base, pa.table({"x": [1]}), "team/model", owner).status_code == 200
        art_id, version = _name_ref(base, "team/model", owner)

        def writes(artifact_id: str) -> list[httpx.Response]:
            target = {"artifact_id": artifact_id, "version": int(version)}
            return [
                httpx.put(
                    f"{base}/v1/artifacts/{artifact_id}/v/{version}/tags",
                    json={"key": "k", "value": "v"},
                    headers=other,
                ),
                httpx.post(f"{base}/v1/names", json={"name": "mine", **target}, headers=other),
                httpx.put(f"{base}/v1/names/mine/aliases/candidate", json=target, headers=other),
                httpx.put(f"{base}/v1/names/mine/aliases/champion", json=target, headers=other),
            ]

        read = httpx.get(f"{base}/v1/artifacts/{art_id}/v/{version}", headers=other)
        assert read.status_code == 404
        for resp in writes(art_id):
            assert resp.status_code == 404, resp.text
            assert resp.json() == read.json()
            assert "team-a" not in resp.text
        # Indistinguishable from an artifact that does not exist.
        for resp in writes("no-such-artifact"):
            assert resp.status_code == 404, resp.text
            assert resp.json() == read.json()
