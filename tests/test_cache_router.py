"""Tests for the cache-plane HTTP router (``/v1/cache/*``).

Drives the real FastAPI app in-process via TestClient against a personal-mode
``ServerState`` (real DiskCache, no lifespan so ``_cache_warmer`` stays None),
plus targeted state patches for the error / warmer-present branches.
"""

from __future__ import annotations

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
    """Stand-in for a started CacheWarmer — exercises the warmer-present paths."""

    async def start_job(self, request):
        return "job-1"

    def list_jobs(self, include_completed=False):
        return [_progress()]

    def get_progress(self, job_id):
        return _progress(job_id) if job_id == "job-1" else None

    async def cancel_job(self, job_id):
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
        # No ``with`` — lifespan never runs, so _cache_warmer stays None.
        yield TestClient(app), state
    finally:
        server_module._state = original
        reset_artifact_store()


@pytest.fixture
def warehouse_uri(tmp_path):
    """A real single-table Iceberg warehouse the planner can scan + warm."""
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
        # asdict(DiskCache.get_stats()) — a JSON object with cache fields.
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
        # A bogus URI fails planning → captured per-table in the errors list,
        # not a request failure.
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
        # First warm of a fresh cache writes the row group(s).
        assert body["row_groups_cached"] >= 1
        assert body["bytes_written"] > 0

    def test_warm_twice_reports_skipped(self, cache_client, warehouse_uri):
        client, _ = cache_client
        client.post("/v1/cache/warm", json={"tables": [warehouse_uri]})  # populate
        resp = client.post("/v1/cache/warm", json={"tables": [warehouse_uri]})
        assert resp.status_code == 200
        body = resp.json()
        # Second warm finds the row group(s) already cached.
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
    """Without lifespan startup, ``_cache_warmer`` is None → graceful 503/404/empty."""

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
    """With a started warmer, the async endpoints take their happy paths."""

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
