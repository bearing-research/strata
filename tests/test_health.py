"""Tests for health check functionality."""

from concurrent.futures import ThreadPoolExecutor

import pytest


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        from strata.health import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestDependencyCheck:
    """Tests for DependencyCheck dataclass."""

    def test_to_dict_basic(self):
        """Test basic conversion to dict."""
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
        """Test conversion with message and details."""
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
    """Tests for HealthReport dataclass."""

    def test_to_dict(self):
        """Test report conversion to dict."""
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
    """Tests for disk cache health check."""

    def test_healthy_cache(self, tmp_path):
        """Test healthy disk cache check (may be degraded if disk is full)."""
        from strata.health import HealthStatus, check_disk_cache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        result = check_disk_cache(cache_dir, max_size_bytes=1024 * 1024 * 1024)

        assert result.name == "disk_cache"
        # May be DEGRADED if disk is >90% full
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        assert result.latency_ms >= 0
        assert "path" in result.details
        assert "available_bytes" in result.details

    def test_missing_cache_dir(self, tmp_path):
        """Test check with missing cache directory."""
        from strata.health import HealthStatus, check_disk_cache

        cache_dir = tmp_path / "nonexistent"

        result = check_disk_cache(cache_dir, max_size_bytes=1024 * 1024)

        assert result.status == HealthStatus.UNHEALTHY
        assert result.message is not None
        assert "does not exist" in result.message


class TestMetadataStoreCheck:
    """Tests for metadata store health check."""

    def test_healthy_store(self, tmp_path):
        """Test healthy metadata store check."""
        from strata.health import HealthStatus, check_metadata_store

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        result = check_metadata_store(cache_dir)

        assert result.name == "metadata_store"
        assert result.status == HealthStatus.HEALTHY
        assert "parquet_meta_entries" in result.details


class TestArrowMemoryCheck:
    """Tests for Arrow memory health check."""

    def test_arrow_memory_check(self):
        """Test Arrow memory pool check."""
        from strata.health import HealthStatus, check_arrow_memory

        result = check_arrow_memory()

        assert result.name == "arrow_memory"
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        assert "backend" in result.details
        assert "bytes_allocated" in result.details


class TestThreadPoolsCheck:
    """Tests for thread pools health check."""

    def test_thread_pools_check(self):
        """Test thread pools health check."""
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
    """Tests for rate limiter health check."""

    def test_rate_limiter_not_initialized(self):
        """Test check when rate limiter not initialized."""
        from strata.health import HealthStatus, check_rate_limiter
        from strata.rate_limiter import reset_rate_limiter

        reset_rate_limiter()

        result = check_rate_limiter()

        assert result.name == "rate_limiter"
        assert result.status == HealthStatus.HEALTHY
        assert result.details.get("enabled") is False

    def test_rate_limiter_healthy(self):
        """Test check with healthy rate limiter."""
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
    """Tests for cache evictions health check."""

    def test_no_evictions(self):
        """Test check with no evictions."""
        from strata.cache_metrics import reset_eviction_tracker
        from strata.health import HealthStatus, check_cache_evictions

        reset_eviction_tracker()

        result = check_cache_evictions()

        assert result.name == "cache_evictions"
        assert result.status == HealthStatus.HEALTHY
        assert result.details.get("pressure_level") == "low"


class TestRunHealthChecks:
    """Tests for run_health_checks function."""

    def test_all_healthy(self, tmp_path):
        """Test running all health checks (may be degraded if disk is full)."""
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

            # May be DEGRADED if disk is >90% full, but should not be UNHEALTHY
            assert report.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            assert len(report.checks) == 6
            # No check should be UNHEALTHY
            assert not any(c.status == HealthStatus.UNHEALTHY for c in report.checks)
        finally:
            planning.shutdown(wait=False)
            fetch.shutdown(wait=False)


class TestHealthEndpointIntegration:
    """Integration tests for health endpoint."""

    @pytest.mark.asyncio
    async def test_health_dependencies_endpoint(self, tmp_path):
        """Test /health/dependencies endpoint."""
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

                # Should have all 6 checks
                assert data["summary"]["total"] == 6

                # Check each dependency is present
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
        """Test basic /health endpoint."""
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


class TestReportedVersionIsTheInstalledOne:
    """``/health`` reports the running version so an operator can confirm what
    a rollout actually landed. It was hardcoded to ``"0.2.0"`` and stayed there
    through three releases, so the field read the same before and after a
    deploy — worse than absent, because it looks authoritative.
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
