"""Tests for timeout configuration."""

import pytest


class TestTimeoutConfig:
    def test_default_timeouts(self):
        from strata.config import StrataConfig

        config = StrataConfig()

        assert config.plan_timeout_seconds == 30.0
        assert config.scan_timeout_seconds == 300.0
        assert config.interactive_queue_timeout == 10.0
        assert config.bulk_queue_timeout == 30.0
        assert config.fetch_timeout_seconds == 60.0
        assert config.s3_connect_timeout_seconds == 10.0
        assert config.s3_request_timeout_seconds == 30.0

    def test_custom_timeouts(self):
        from strata.config import StrataConfig

        config = StrataConfig(
            plan_timeout_seconds=60.0,
            scan_timeout_seconds=600.0,
            interactive_queue_timeout=20.0,
            bulk_queue_timeout=5.0,
            fetch_timeout_seconds=120.0,
            s3_connect_timeout_seconds=20.0,
            s3_request_timeout_seconds=60.0,
        )

        assert config.plan_timeout_seconds == 60.0
        assert config.scan_timeout_seconds == 600.0
        assert config.interactive_queue_timeout == 20.0
        assert config.bulk_queue_timeout == 5.0
        assert config.fetch_timeout_seconds == 120.0
        assert config.s3_connect_timeout_seconds == 20.0
        assert config.s3_request_timeout_seconds == 60.0

    def test_get_timeout_config(self):
        from strata.config import StrataConfig

        config = StrataConfig()
        timeout_config = config.get_timeout_config()

        assert "planning" in timeout_config
        assert "scanning" in timeout_config
        assert "qos_queue" in timeout_config
        assert "fetching" in timeout_config
        assert "s3" in timeout_config

        assert timeout_config["planning"]["plan_timeout_seconds"] == 30.0
        assert timeout_config["scanning"]["scan_timeout_seconds"] == 300.0
        assert timeout_config["qos_queue"]["interactive_queue_timeout"] == 10.0
        assert timeout_config["qos_queue"]["bulk_queue_timeout"] == 30.0
        assert timeout_config["fetching"]["fetch_timeout_seconds"] == 60.0
        assert timeout_config["s3"]["s3_connect_timeout_seconds"] == 10.0
        assert timeout_config["s3"]["s3_request_timeout_seconds"] == 30.0


class TestTimeoutEndpointIntegration:
    """The timeout config endpoint."""

    @pytest.mark.asyncio
    async def test_timeout_endpoint(self, tmp_path):
        """/v1/config/timeouts."""
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
                response = await client.get("/v1/config/timeouts")
                assert response.status_code == 200
                data = response.json()

                assert "planning" in data
                assert "scanning" in data
                assert "qos_queue" in data
                assert "fetching" in data
                assert "s3" in data

                assert "plan_timeout_seconds" in data["planning"]
                assert data["planning"]["plan_timeout_seconds"] == 30.0

                assert "scan_timeout_seconds" in data["scanning"]
                assert data["scanning"]["scan_timeout_seconds"] == 300.0

                assert "interactive_queue_timeout" in data["qos_queue"]
                assert "bulk_queue_timeout" in data["qos_queue"]
                assert data["qos_queue"]["interactive_queue_timeout"] == 10.0
                assert data["qos_queue"]["bulk_queue_timeout"] == 30.0

                assert "fetch_timeout_seconds" in data["fetching"]
                assert data["fetching"]["fetch_timeout_seconds"] == 60.0

                assert "s3_connect_timeout_seconds" in data["s3"]
                assert "s3_request_timeout_seconds" in data["s3"]
                assert data["s3"]["s3_connect_timeout_seconds"] == 10.0
                assert data["s3"]["s3_request_timeout_seconds"] == 30.0
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None

    @pytest.mark.asyncio
    async def test_timeout_endpoint_custom_values(self, tmp_path):
        """/v1/config/timeouts with custom values."""
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
        config = StrataConfig(
            cache_dir=cache_dir,
            plan_timeout_seconds=45.0,
            scan_timeout_seconds=120.0,
            fetch_timeout_seconds=90.0,
        )
        server_module._state = ServerState(config)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/v1/config/timeouts")
                assert response.status_code == 200
                data = response.json()

                assert data["planning"]["plan_timeout_seconds"] == 45.0
                assert data["scanning"]["scan_timeout_seconds"] == 120.0
                assert data["fetching"]["fetch_timeout_seconds"] == 90.0
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None
