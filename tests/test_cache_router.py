"""Tests for the cache-plane HTTP router (``/v1/cache/*``).

Uses TestClient against a personal-mode ``ServerState`` (real DiskCache, no lifespan, so
``_cache_warmer`` is None), plus state patches for the error and warmer-present branches.
"""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from strata.types import WarmJobProgress, WarmJobStatus


def _progress(job_id: str = "job-1", status: WarmJobStatus = WarmJobStatus.RUNNING):
    return WarmJobProgress(
        job_id=job_id,
        status=status,
        tables_total=1,
        tables_completed=0,
        row_groups_total=0,
        row_groups_completed=0,
        row_groups_cached=0,
        row_groups_skipped=0,
        bytes_written=0,
        started_at=None,
        completed_at=None,
        elapsed_ms=0.0,
        current_table=None,
        errors=[],
    )


class _StubWarmer:
    """Stand-in for a started CacheWarmer, for the warmer-present paths."""

    async def start_job(self, request, authorize=None, tenant=None):
        return "job-1"

    def list_jobs(self, include_completed=False, tenant=None):
        return [_progress()]

    def get_progress(self, job_id, tenant=None):
        return _progress(job_id) if job_id == "job-1" else None

    async def cancel_job(self, job_id, tenant=None):
        return job_id == "job-1"


@pytest.fixture
def cache_client(tmp_path):
    import strata.server as server_module
    from strata.artifact_store import reset_artifact_store
    from strata.config import StrataConfig
    from strata.server import ServerState, app

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    config = StrataConfig(
        host="127.0.0.1",
        port=8765,
        deployment_mode="personal",
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    reset_artifact_store()
    original = server_module._state
    state = ServerState(config)
    server_module._state = state
    try:
        # No ``with``: lifespan never runs, so _cache_warmer stays None.
        yield TestClient(app), state
    finally:
        server_module._state = original
        reset_artifact_store()


@pytest.fixture
def warehouse_uri(tmp_path):
    """A real single-table Iceberg warehouse the planner can scan and warm."""
    import sys

    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")

    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField

    wh = tmp_path / "warehouse"
    wh.mkdir()
    catalog = SqlCatalog(
        "strata",
        uri=f"sqlite:///{wh / 'catalog.db'}",
        warehouse=str(wh),
    )
    catalog.create_namespace("test_db")
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
    )
    table = catalog.create_table("test_db.events", schema)
    table.append(
        pa.table(
            {
                "id": pa.array(range(100), type=pa.int64()),
                "value": pa.array([float(i) for i in range(100)], type=pa.float64()),
            }
        )
    )
    return f"file://{wh}#test_db.events"


class TestCacheStatsAndEntries:
    def test_stats_returns_disk_cache_stats(self, cache_client):
        client, _ = cache_client
        resp = client.get("/v1/cache/stats")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)

    def test_entries_lists_entries(self, cache_client):
        client, _ = cache_client
        resp = client.get("/v1/cache/entries")
        assert resp.status_code == 200
        assert "entries" in resp.json()
        assert isinstance(resp.json()["entries"], list)

    def test_evictions_with_events(self, cache_client):
        client, _ = cache_client
        resp = client.get("/v1/cache/evictions", params={"include_events": True, "limit": 5})
        assert resp.status_code == 200
        assert "recent_events" in resp.json()

    def test_histogram(self, cache_client):
        client, _ = cache_client
        resp = client.get("/v1/cache/histogram")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)


class TestClearCache:
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="DiskCache.clear() hits Windows file-locking on the just-created cache dir",
    )
    def test_clear_succeeds(self, cache_client):
        client, _ = cache_client
        resp = client.post("/v1/cache/clear")
        assert resp.status_code == 200
        assert resp.json() == {"status": "cleared"}

    def test_clear_error_is_500(self, cache_client):
        client, state = cache_client

        def boom():
            raise RuntimeError("disk gone")

        state.fetcher.cache.clear = boom
        resp = client.post("/v1/cache/clear")
        assert resp.status_code == 500
        assert "disk gone" in resp.json()["detail"]


class TestWarmSync:
    def test_warm_unplannable_table_reports_error(self, cache_client):
        client, _ = cache_client
        # A bogus URI fails planning as a per-table error, not a request failure.
        resp = client.post("/v1/cache/warm", json={"tables": ["file:///nope#bad.table"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["tables_warmed"] == 0
        assert len(body["errors"]) == 1

    def test_warm_empty_table_list(self, cache_client):
        client, _ = cache_client
        resp = client.post("/v1/cache/warm", json={"tables": []})
        assert resp.status_code == 200
        assert resp.json()["tables_warmed"] == 0

    def test_warm_real_table_caches_row_groups(self, cache_client, warehouse_uri):
        client, _ = cache_client
        resp = client.post("/v1/cache/warm", json={"tables": [warehouse_uri]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["tables_warmed"] == 1
        assert body["errors"] == []
        assert body["row_groups_cached"] >= 1
        assert body["bytes_written"] > 0

    def test_warm_twice_reports_skipped(self, cache_client, warehouse_uri):
        client, _ = cache_client
        client.post("/v1/cache/warm", json={"tables": [warehouse_uri]})
        resp = client.post("/v1/cache/warm", json={"tables": [warehouse_uri]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["row_groups_skipped"] >= 1
        assert body["row_groups_cached"] == 0

    def test_warm_respects_max_row_groups(self, cache_client, warehouse_uri):
        client, _ = cache_client
        resp = client.post(
            "/v1/cache/warm",
            json={"tables": [warehouse_uri], "max_row_groups": 1},
        )
        assert resp.status_code == 200
        assert resp.json()["tables_warmed"] == 1


class TestAsyncWarmerNotInitialized:
    """Without lifespan startup ``_cache_warmer`` is None: graceful 503/404/empty."""

    def test_async_warm_returns_503(self, cache_client):
        client, _ = cache_client
        resp = client.post("/v1/cache/warm/async", json={"tables": ["file:///x#a.b"]})
        assert resp.status_code == 503

    def test_list_jobs_empty(self, cache_client):
        client, _ = cache_client
        resp = client.get("/v1/cache/warm/jobs")
        assert resp.status_code == 200
        assert resp.json() == {"jobs": []}

    def test_get_job_404(self, cache_client):
        client, _ = cache_client
        resp = client.get("/v1/cache/warm/jobs/nope")
        assert resp.status_code == 404

    def test_cancel_job_404(self, cache_client):
        client, _ = cache_client
        resp = client.delete("/v1/cache/warm/jobs/nope")
        assert resp.status_code == 404


class TestNonDiskCache:
    """When the fetcher's cache isn't a DiskCache, stats/entries are 501."""

    def test_stats_501(self, cache_client):
        client, state = cache_client
        state.fetcher.cache = object()
        assert client.get("/v1/cache/stats").status_code == 501

    def test_entries_501(self, cache_client):
        client, state = cache_client
        state.fetcher.cache = object()
        assert client.get("/v1/cache/entries").status_code == 501


class TestAsyncWarmerPresent:
    @pytest.fixture
    def warmer_client(self, cache_client):
        client, state = cache_client
        state._cache_warmer = _StubWarmer()
        return client

    def test_async_warm_starts_job(self, warmer_client):
        resp = warmer_client.post("/v1/cache/warm/async", json={"tables": ["file:///x#a.b"]})
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-1"

    def test_list_jobs_returns_jobs(self, warmer_client):
        resp = warmer_client.get("/v1/cache/warm/jobs")
        assert resp.status_code == 200
        assert [j["job_id"] for j in resp.json()["jobs"]] == ["job-1"]

    def test_get_known_job(self, warmer_client):
        resp = warmer_client.get("/v1/cache/warm/jobs/job-1")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-1"

    def test_get_unknown_job_404(self, warmer_client):
        assert warmer_client.get("/v1/cache/warm/jobs/other").status_code == 404

    def test_cancel_known_job(self, warmer_client):
        resp = warmer_client.delete("/v1/cache/warm/jobs/job-1")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is True

    def test_cancel_unknown_job_404(self, warmer_client):
        assert warmer_client.delete("/v1/cache/warm/jobs/other").status_code == 404


# --- Table ACL on the warm endpoints ---


@pytest.fixture
def acl_cache_client(tmp_path):
    """Service mode + trusted proxy with a deny rule on ``*:denied.*``."""
    import strata.server as server_module
    from strata.artifact_store import reset_artifact_store
    from strata.config import AclConfig, AclRule, StrataConfig
    from strata.server import ServerState, app

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    config = StrataConfig(
        host="127.0.0.1",
        port=8765,
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
        acl_config=AclConfig(
            default="allow",
            deny_rules=[AclRule(principal="*", tables=["*:denied.*"])],
        ),
    )

    reset_artifact_store()
    original = server_module._state
    server_module._state = ServerState(config)
    try:
        yield TestClient(app)
    finally:
        server_module._state = original
        reset_artifact_store()


def _proxy_headers() -> dict[str, str]:
    return {
        "X-Strata-Proxy-Token": "test-token",
        "X-Strata-Principal": "analyst",
        "X-Strata-Scopes": "notebook:read",
    }


def test_warm_refuses_acl_denied_table(acl_cache_client):
    """Warming reads into the shared cache, so it clears the scan path's deny-first gate; otherwise
    a denied principal learns the table's size.
    """
    resp = acl_cache_client.post(
        "/v1/cache/warm",
        json={"tables": ["file:///wh#denied.salaries"]},
        headers=_proxy_headers(),
    )
    assert resp.status_code in (403, 404), resp.text
    assert "row_groups_cached" not in resp.text


def test_warm_async_refuses_acl_denied_table(acl_cache_client):
    """The background job must not be a way around the ACL."""
    resp = acl_cache_client.post(
        "/v1/cache/warm/async",
        json={"tables": ["file:///wh#denied.salaries"]},
        headers=_proxy_headers(),
    )
    assert resp.status_code in (403, 404), resp.text
    assert "job_id" not in resp.text


def test_warm_allows_permitted_table(acl_cache_client):
    """A permitted table reaches the handler, which fails on the missing warehouse, not on auth."""
    resp = acl_cache_client.post(
        "/v1/cache/warm",
        json={"tables": ["file:///wh#allowed.events"]},
        headers=_proxy_headers(),
    )
    assert resp.status_code == 200, resp.text
    # Planning fails (no such warehouse), but as a per-table error, not a 403.
    assert resp.json()["errors"]


class TestWarmTakesTheScanIdentity:
    """Warm authorizes the identity the scan path does, before planning and again after."""

    @pytest.fixture
    def lake_state(self, tmp_path):
        """Service mode with a named catalog ``lake`` whose ``denied`` namespace is denied."""
        import strata.server as server_module
        from strata.artifact_store import reset_artifact_store
        from strata.config import AclConfig, AclRule, StrataConfig
        from strata.server import ServerState

        config = StrataConfig(
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            catalogs={"lake": {"type": "rest", "uri": "http://catalog.invalid"}},
            acl_config=AclConfig(
                default="allow",
                deny_rules=[AclRule(principal="*", tables=["lake:denied.*", "file:denied.*"])],
            ),
        )
        reset_artifact_store()
        original = server_module._state
        state = ServerState(config)
        server_module._state = state
        try:
            yield state
        finally:
            server_module._state = original
            reset_artifact_store()

    @staticmethod
    def _planned_as(identity):
        """A planner whose catalog resolves every URI to *identity*."""
        from types import SimpleNamespace

        def plan(**kwargs):
            return SimpleNamespace(table_identity=identity, tasks=[])

        return plan

    @pytest.mark.parametrize("path", ["/v1/cache/warm", "/v1/cache/warm/async"])
    def test_a_named_catalog_deny_rule_applies(self, lake_state, path):
        from strata.server import app

        resp = TestClient(app).post(
            path, json={"tables": ["lake:denied.salaries"]}, headers=_proxy_headers()
        )
        assert resp.status_code == 404, resp.text
        assert "job_id" not in resp.text

    def test_the_planned_identity_is_checked(self, lake_state, monkeypatch):
        from strata.server import app
        from strata.types import TableIdentity

        denied = TableIdentity(catalog="strata", namespace="denied", table="salaries")
        monkeypatch.setattr(lake_state.planner, "plan", self._planned_as(denied))
        resp = TestClient(app).post(
            "/v1/cache/warm",
            json={"tables": ["file:///wh#allowed.events"]},
            headers=_proxy_headers(),
        )
        assert resp.status_code == 404, resp.text

    async def test_an_async_job_checks_the_planned_identity(self, lake_state):
        import asyncio
        from unittest.mock import MagicMock

        from httpx import ASGITransport, AsyncClient

        from strata.cache_warmer import CacheWarmer
        from strata.server import app
        from strata.types import TableIdentity

        denied = TableIdentity(catalog="strata", namespace="denied", table="salaries")
        planner = MagicMock()
        planner.plan.side_effect = self._planned_as(denied)
        lake_state._cache_warmer = CacheWarmer(
            planner=planner, fetcher=MagicMock(), metrics=MagicMock()
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            resp = await client.post(
                "/v1/cache/warm/async",
                json={"tables": ["file:///wh#allowed.events"]},
                headers=_proxy_headers(),
            )
            assert resp.status_code == 200, resp.text
            job_id = resp.json()["job_id"]
            for _ in range(100):
                job = await client.get(f"/v1/cache/warm/jobs/{job_id}", headers=_proxy_headers())
                if job.json()["status"] not in ("pending", "running"):
                    break
                await asyncio.sleep(0.01)
        progress = job.json()
        assert progress["status"] == "failed"
        assert progress["tables_completed"] == 0
        assert progress["errors"] == ["file:///wh#allowed.events: 404: Table not found"]


class TestWarmJobsAreTenantScoped:
    """A warm job names its tenant's tables, so another tenant cannot list, read or cancel it."""

    @pytest.fixture
    def tenant_state(self, tmp_path):
        import strata.server as server_module
        from strata.artifact_store import reset_artifact_store
        from strata.config import StrataConfig
        from strata.server import ServerState

        config = StrataConfig(
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            multi_tenant_enabled=True,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
        )
        reset_artifact_store()
        original = server_module._state
        state = ServerState(config)
        server_module._state = state
        try:
            yield state
        finally:
            server_module._state = original
            reset_artifact_store()

    @staticmethod
    def _headers(tenant: str, scopes: str = "") -> dict[str, str]:
        return {
            "X-Strata-Proxy-Token": "test-token",
            "X-Strata-Principal": "analyst",
            "X-Tenant-ID": tenant,
            "X-Strata-Scopes": scopes,
        }

    async def test_another_tenant_cannot_see_or_cancel_a_job(self, tenant_state):
        import threading
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from httpx import ASGITransport, AsyncClient

        from strata.cache_warmer import CacheWarmer
        from strata.server import app
        from strata.types import TableIdentity

        # One row group whose fetch blocks, so the job stays running while we look at it.
        release = threading.Event()
        planner = MagicMock()
        planner.plan.return_value = SimpleNamespace(
            table_identity=TableIdentity(catalog="strata", namespace="team", table="t"),
            tasks=[SimpleNamespace(cached=False, bytes_read=0, file_path="f", row_group_id=0)],
        )
        fetcher = MagicMock()
        fetcher.fetch_as_stream_bytes.side_effect = lambda task: release.wait(30)
        tenant_state._cache_warmer = CacheWarmer(
            planner=planner, fetcher=fetcher, metrics=MagicMock()
        )
        owner, other = self._headers("team-a"), self._headers("team-b")
        admin = self._headers("team-b", "admin:*")
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
                started = await client.post(
                    "/v1/cache/warm/async", json={"tables": ["file:///wh#team.t"]}, headers=owner
                )
                assert started.status_code == 200, started.text
                job_id = started.json()["job_id"]
                jobs = "/v1/cache/warm/jobs"

                def listed(resp):
                    return [j["job_id"] for j in resp.json()["jobs"]]

                assert listed(await client.get(jobs, headers=other)) == []
                assert (await client.get(f"{jobs}/{job_id}", headers=other)).status_code == 404
                assert (await client.delete(f"{jobs}/{job_id}", headers=other)).status_code == 404

                assert listed(await client.get(jobs, headers=owner)) == [job_id]
                assert listed(await client.get(jobs, headers=admin)) == [job_id]
                cancel = await client.delete(f"{jobs}/{job_id}", headers=owner)
                assert cancel.status_code == 200
        finally:
            release.set()
            await tenant_state._cache_warmer.stop()


@pytest.mark.parametrize("field,value", [("concurrent", 0), ("concurrent", -1)])
def test_warm_rejects_unusable_concurrency(cache_client, field, value):
    """``concurrent=0`` made a Semaphore(0) that hung the request and wedged an async job slot."""
    client, _ = cache_client
    resp = client.post("/v1/cache/warm", json={"tables": ["file:///wh#a.b"], field: value})
    assert resp.status_code == 422


class TestCachePlaneInformationDisclosure:
    """Cache introspection must not hand cross-tenant metadata to anyone.

    ``/v1/cache/entries`` and ``/v1/debug/cache/inspect`` walk every tenant's entries (table,
    snapshot, projection, path), so they need ``admin:cache`` like ``/v1/cache/clear``.
    """

    def test_cache_entries_requires_admin_scope(self, acl_cache_client):
        resp = acl_cache_client.get("/v1/cache/entries", headers=_proxy_headers())
        assert resp.status_code == 403

    def test_debug_inspect_requires_admin_scope(self, acl_cache_client):
        resp = acl_cache_client.get("/v1/debug/cache/inspect", headers=_proxy_headers())
        assert resp.status_code == 403

    def test_admin_scope_still_gets_through(self, acl_cache_client):
        headers = {**_proxy_headers(), "X-Strata-Scopes": "admin:cache"}
        assert acl_cache_client.get("/v1/cache/entries", headers=headers).status_code == 200

    def test_personal_mode_is_unchanged(self, cache_client):
        """No-auth deployments keep their open introspection."""
        client, _ = cache_client
        assert client.get("/v1/cache/entries").status_code == 200


class TestDebugInspectPrefixLayout:
    """``?prefix=`` searches the real layout: versioned_dir/{tenant_prefix}/hash[:2]/hash[2:4]."""

    def test_prefix_search_finds_a_real_entry(self, cache_client, tmp_path):
        import hashlib

        from strata.cache import CACHE_VERSION

        client, state = cache_client
        cache = state.fetcher.cache

        # The real on-disk shape: versioned/{tenant}/xx/yy/<hash>
        digest = "abcd1234" + "0" * 24
        tenant_prefix = hashlib.sha256(b"").hexdigest()[:8]
        entry_dir = cache.cache_dir / f"v{CACHE_VERSION}" / tenant_prefix / "ab" / "cd"
        entry_dir.mkdir(parents=True, exist_ok=True)
        (entry_dir / f"{digest}.arrow").write_bytes(b"x")

        resp = client.get("/v1/debug/cache/inspect", params={"prefix": "abcd"})
        assert resp.status_code == 200
        assert resp.json()["prefix_filter"] == "abcd"


class TestWarmDoesNotReportFailuresAsSuccess:
    """A row group that failed to fetch must not count as cached.

    ``fetch_task`` returned ``(False, 0)`` on error, the same value as "fetched and written", so a
    fully failed warm reported success with no errors.
    """

    def test_a_failing_fetch_is_not_counted_as_cached(self, cache_client, warehouse_uri):
        client, state = cache_client

        def boom(task):
            raise RuntimeError("storage unreachable")

        state.fetcher.fetch_as_stream_bytes = boom

        resp = client.post("/v1/cache/warm", json={"tables": [warehouse_uri]})
        assert resp.status_code == 200
        body = resp.json()

        assert body["row_groups_cached"] == 0, body
        assert body["bytes_written"] == 0, body

    def test_a_failing_fetch_is_surfaced(self, cache_client, warehouse_uri):
        client, state = cache_client

        def boom(task):
            raise RuntimeError("storage unreachable")

        state.fetcher.fetch_as_stream_bytes = boom

        body = client.post("/v1/cache/warm", json={"tables": [warehouse_uri]}).json()

        assert body["errors"], "a warm that cached nothing must say so"
        assert any("storage unreachable" in e for e in body["errors"]), body

    def test_a_healthy_warm_is_unchanged(self, cache_client, warehouse_uri):
        client, _ = cache_client
        body = client.post("/v1/cache/warm", json={"tables": [warehouse_uri]}).json()

        assert body["errors"] == []
        assert body["row_groups_cached"] >= 1
        assert body["bytes_written"] > 0
