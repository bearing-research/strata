"""Tests for health check functionality."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest


class TestHealthStatus:
    def test_health_status_values(self):
        from strata.health import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestDependencyCheck:
    def test_to_dict_basic(self):
        from strata.health import DependencyCheck, HealthStatus

        check = DependencyCheck(
            name="test_check",
            status=HealthStatus.HEALTHY,
            latency_ms=1.5,
        )

        d = check.to_dict()
        assert d["name"] == "test_check"
        assert d["status"] == "healthy"
        assert d["latency_ms"] == 1.5
        assert "message" not in d
        assert "details" not in d

    def test_to_dict_with_message_and_details(self):
        from strata.health import DependencyCheck, HealthStatus

        check = DependencyCheck(
            name="test_check",
            status=HealthStatus.DEGRADED,
            latency_ms=2.5,
            message="Something is slow",
            details={"key": "value"},
        )

        d = check.to_dict()
        assert d["message"] == "Something is slow"
        assert d["details"] == {"key": "value"}


class TestHealthReport:
    def test_to_dict(self):
        from strata.health import DependencyCheck, HealthReport, HealthStatus

        checks = [
            DependencyCheck("check1", HealthStatus.HEALTHY, 1.0),
            DependencyCheck("check2", HealthStatus.DEGRADED, 2.0),
            DependencyCheck("check3", HealthStatus.HEALTHY, 1.5),
        ]

        report = HealthReport(
            status=HealthStatus.DEGRADED,
            checks=checks,
            timestamp=1234567890.0,
        )

        d = report.to_dict()
        assert d["status"] == "degraded"
        assert d["timestamp"] == 1234567890.0
        assert len(d["checks"]) == 3
        assert d["summary"]["total"] == 3
        assert d["summary"]["healthy"] == 2
        assert d["summary"]["degraded"] == 1
        assert d["summary"]["unhealthy"] == 0


class TestDiskCacheCheck:
    def test_healthy_cache(self, tmp_path):
        """May report degraded when the disk is full."""
        from strata.health import HealthStatus, check_disk_cache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        result = check_disk_cache(cache_dir, max_size_bytes=1024 * 1024 * 1024)

        assert result.name == "disk_cache"
        # DEGRADED if the disk is >90% full
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        assert result.latency_ms >= 0
        assert "path" in result.details
        assert "available_bytes" in result.details

    def test_missing_cache_dir(self, tmp_path):
        from strata.health import HealthStatus, check_disk_cache

        cache_dir = tmp_path / "nonexistent"

        result = check_disk_cache(cache_dir, max_size_bytes=1024 * 1024)

        assert result.status == HealthStatus.UNHEALTHY
        assert result.message is not None
        assert "does not exist" in result.message


class TestMetadataStoreCheck:
    def test_healthy_store(self, tmp_path):
        from strata.health import HealthStatus, check_metadata_store

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        result = check_metadata_store(cache_dir)

        assert result.name == "metadata_store"
        assert result.status == HealthStatus.HEALTHY
        assert "parquet_meta_entries" in result.details


class TestArrowMemoryCheck:
    def test_arrow_memory_check(self):
        from strata.health import HealthStatus, check_arrow_memory

        result = check_arrow_memory()

        assert result.name == "arrow_memory"
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        assert "backend" in result.details
        assert "bytes_allocated" in result.details


class TestThreadPoolsCheck:
    def test_thread_pools_check(self):
        from strata.health import HealthStatus, check_thread_pools
        from strata.pool_metrics import get_pool_tracker, reset_metrics

        reset_metrics()
        tracker = get_pool_tracker()

        planning = ThreadPoolExecutor(max_workers=4, thread_name_prefix="planning")
        fetch = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fetch")

        tracker.register_pool("planning", planning)
        tracker.register_pool("fetch", fetch)

        try:
            result = check_thread_pools(planning, fetch)

            assert result.name == "thread_pools"
            assert result.status == HealthStatus.HEALTHY
            assert "pools" in result.details
            assert "overall_utilization" in result.details
        finally:
            planning.shutdown(wait=False)
            fetch.shutdown(wait=False)


class TestRateLimiterCheck:
    def test_rate_limiter_not_initialized(self):
        from strata.health import HealthStatus, check_rate_limiter
        from strata.rate_limiter import reset_rate_limiter

        reset_rate_limiter()

        result = check_rate_limiter()

        assert result.name == "rate_limiter"
        assert result.status == HealthStatus.HEALTHY
        assert result.details.get("enabled") is False

    def test_rate_limiter_healthy(self):
        from strata.health import HealthStatus, check_rate_limiter
        from strata.rate_limiter import RateLimitConfig, init_rate_limiter, reset_rate_limiter

        reset_rate_limiter()
        init_rate_limiter(RateLimitConfig())

        result = check_rate_limiter()

        assert result.name == "rate_limiter"
        assert result.status == HealthStatus.HEALTHY
        assert result.details.get("enabled") is True

        reset_rate_limiter()


class TestCacheEvictionsCheck:
    def test_no_evictions(self):
        from strata.cache_metrics import reset_eviction_tracker
        from strata.health import HealthStatus, check_cache_evictions

        reset_eviction_tracker()

        result = check_cache_evictions()

        assert result.name == "cache_evictions"
        assert result.status == HealthStatus.HEALTHY
        assert result.details.get("pressure_level") == "low"


class TestRunHealthChecks:
    def test_all_healthy(self, tmp_path):
        """May report degraded when the disk is full."""
        from strata.cache_metrics import reset_eviction_tracker
        from strata.health import HealthStatus, run_health_checks
        from strata.pool_metrics import get_pool_tracker, reset_metrics
        from strata.rate_limiter import reset_rate_limiter

        reset_metrics()
        reset_rate_limiter()
        reset_eviction_tracker()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        planning = ThreadPoolExecutor(max_workers=4)
        fetch = ThreadPoolExecutor(max_workers=4)

        tracker = get_pool_tracker()
        tracker.register_pool("planning", planning)
        tracker.register_pool("fetch", fetch)

        try:
            report = run_health_checks(
                cache_dir=cache_dir,
                max_cache_size_bytes=1024 * 1024 * 1024,
                planning_executor=planning,
                fetch_executor=fetch,
            )

            # DEGRADED if the disk is >90% full, but never UNHEALTHY.
            assert report.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            assert len(report.checks) == 6
            assert not any(c.status == HealthStatus.UNHEALTHY for c in report.checks)
        finally:
            planning.shutdown(wait=False)
            fetch.shutdown(wait=False)


class TestHealthEndpointIntegration:
    @pytest.mark.asyncio
    async def test_health_dependencies_endpoint(self, tmp_path):
        """/health/dependencies endpoint."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.cache_metrics import reset_eviction_tracker
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.rate_limiter import reset_rate_limiter
        from strata.server import ServerState, app

        reset_metrics()
        reset_rate_limiter()
        reset_eviction_tracker()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = StrataConfig(cache_dir=cache_dir)
        server_module._state = ServerState(config)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health/dependencies")
                assert response.status_code == 200
                data = response.json()

                assert "status" in data
                assert "checks" in data
                assert "summary" in data
                assert "timestamp" in data

                assert data["summary"]["total"] == 6

                check_names = {c["name"] for c in data["checks"]}
                assert "disk_cache" in check_names
                assert "metadata_store" in check_names
                assert "arrow_memory" in check_names
                assert "thread_pools" in check_names
                assert "rate_limiter" in check_names
                assert "cache_evictions" in check_names
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None

    @pytest.mark.asyncio
    async def test_health_endpoint(self, tmp_path):
        """Basic /health endpoint."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.rate_limiter import reset_rate_limiter
        from strata.server import ServerState, app

        reset_metrics()
        reset_rate_limiter()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = StrataConfig(cache_dir=cache_dir)
        server_module._state = ServerState(config)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "ok"
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None


class TestReadinessProbesTheArtifactStore:
    """Every artifact, name and notebook route needs the artifact store's database."""

    @pytest.fixture
    def ready(self, tmp_path, monkeypatch):
        import strata.artifact_store as artifact_store_module
        import strata.server as server_module
        from strata.api.routers import metrics_health
        from strata.artifact_store import ArtifactStore
        from strata.config import StrataConfig
        from strata.server import ServerState

        state = ServerState(StrataConfig(cache_dir=tmp_path / "cache"))
        monkeypatch.setattr(server_module, "_state", state)
        store = ArtifactStore(tmp_path / "artifacts")
        monkeypatch.setattr(artifact_store_module, "_artifact_store", store)

        async def probe():
            import json

            response = await metrics_health.health_ready()
            return response.status_code, json.loads(response.body)

        yield store, probe
        state._planning_executor.shutdown(wait=False)
        state._fetch_executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_a_reachable_store_is_ready(self, ready):
        _store, probe = ready

        status, body = await probe()

        assert status == 200
        assert body["checks"]["artifact_store"] is True

    @pytest.mark.asyncio
    async def test_an_unreachable_store_is_not_ready(self, ready, monkeypatch):
        store, probe = ready

        def refuse():
            raise ConnectionError("connection refused")

        monkeypatch.setattr(store._dialect, "connect", refuse)

        status, body = await probe()

        assert status == 503
        assert body["checks"]["artifact_store"] is False
        # The unauthenticated probe names the failure but never echoes its message.
        assert body["checks"]["artifact_store_error"] == "ConnectionError"
        assert "connection refused" not in json.dumps(body)

    @pytest.mark.asyncio
    async def test_a_store_that_never_answers_is_not_ready(self, ready, monkeypatch, caplog):
        import logging
        import threading

        from strata.api.routers import metrics_health

        store, probe = ready
        release = threading.Event()
        monkeypatch.setattr(store, "ping", release.wait)
        monkeypatch.setattr(metrics_health, "ARTIFACT_STORE_PROBE_TIMEOUT_SECONDS", 0.01)

        try:
            with caplog.at_level(logging.WARNING, logger="strata.api.routers.metrics_health"):
                status, body = await probe()
        finally:
            release.set()

        assert status == 503
        assert body["checks"]["artifact_store"] is False
        # The log line names the reason, which for a timeout is only its type.
        assert "artifact store check failed: TimeoutError" in caplog.text


class TestReportedVersionIsTheInstalledOne:
    """``/health`` reports the installed version so an operator can confirm a rollout.

    A hardcoded version reads the same before and after a deploy and still looks authoritative.
    """

    def test_version_matches_package_metadata(self):
        import time
        from importlib.metadata import version as metadata_version

        from strata.health import HealthReport, HealthStatus

        report = HealthReport(status=HealthStatus.HEALTHY, checks=[], timestamp=time.time())
        assert report.version == metadata_version("strata-notebook")

    def test_version_is_not_the_stale_hardcoded_string(self):
        import time

        from strata.health import HealthReport, HealthStatus

        report = HealthReport(status=HealthStatus.HEALTHY, checks=[], timestamp=time.time())
        assert report.version != "0.2.0"
        assert report.to_dict()["version"] == report.version


def test_the_app_and_its_traces_report_the_installed_version():
    """The OpenAPI document and the trace resource report the installed version too."""
    from strata.health import _package_version
    from strata.server import app

    assert app.version == _package_version()
