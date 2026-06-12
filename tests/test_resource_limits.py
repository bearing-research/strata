"""Tests for resource limits and backpressure."""

import sys
import threading
import time

import pyarrow as pa
import pytest
import uvicorn
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, LongType, NestedField
from strata_client.client import StrataClient

from strata.config import StrataConfig


@pytest.fixture
def temp_warehouse(tmp_path):
    """Create a temporary warehouse with a sample Iceberg table."""
    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
    warehouse_path = tmp_path / "warehouse"
    warehouse_path.mkdir()

    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )

    catalog.create_namespace("test_db")

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
    )

    table = catalog.create_table("test_db.events", schema)

    # Create sample data
    data = pa.table(
        {
            "id": pa.array(range(100), type=pa.int64()),
            "value": pa.array([float(i) for i in range(100)], type=pa.float64()),
        }
    )
    table.append(data)

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"file://{warehouse_path}#test_db.events",
        "catalog": catalog,
        "table": table,
    }


class TestResourceLimitConfig:
    """Tests for resource limit configuration."""

    def test_default_limits(self):
        """Test default resource limit values."""
        config = StrataConfig()

        assert config.max_concurrent_scans == 100
        assert config.max_tasks_per_scan == 1000
        assert config.plan_timeout_seconds == 30.0
        assert config.scan_timeout_seconds == 300.0
        assert config.max_response_bytes == 512 * 1024 * 1024  # 512 MB

    def test_custom_limits(self, tmp_path):
        """Test setting custom resource limits."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            max_concurrent_scans=10,
            max_tasks_per_scan=50,
            plan_timeout_seconds=10.0,
            scan_timeout_seconds=60.0,
            max_response_bytes=100 * 1024 * 1024,  # 100 MB
        )

        assert config.max_concurrent_scans == 10
        assert config.max_tasks_per_scan == 50
        assert config.plan_timeout_seconds == 10.0
        assert config.scan_timeout_seconds == 60.0
        assert config.max_response_bytes == 100 * 1024 * 1024


class TestServerResourceLimits:
    """Tests for server-side resource limit enforcement."""

    @pytest.fixture
    def server_with_client(self, temp_warehouse, tmp_path):
        """Start a server with custom limits and provide a client."""
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            max_concurrent_scans=2,  # Low limit for testing
            max_tasks_per_scan=10,  # Low limit for testing
            scan_timeout_seconds=5.0,  # Short timeout for testing
            max_response_bytes=1024 * 1024,  # 1 MB for testing
            deployment_mode="personal",
        )

        import strata.server as server_module
        from strata.server import ServerState, app

        server_module._state = ServerState(config)

        server_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={
                "app": app,
                "host": config.host,
                "port": config.port,
                "log_level": "error",
            },
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)

        client = StrataClient(base_url=f"http://127.0.0.1:{port}")

        yield {
            "client": client,
            "config": config,
            "warehouse": temp_warehouse,
        }

        client.close()

    def test_metrics_include_resource_limits(self, server_with_client):
        """Test that metrics endpoint includes resource limit info."""
        client = server_with_client["client"]

        metrics = client.metrics()

        # Check that resource_limits section exists and has expected fields
        assert "resource_limits" in metrics
        limits = metrics["resource_limits"]

        # These fields should always be present
        assert "max_concurrent_scans" in limits
        assert "max_tasks_per_scan" in limits
        assert "plan_timeout_seconds" in limits
        assert "scan_timeout_seconds" in limits
        assert "max_response_bytes" in limits
        assert "active_scans" in limits

        # Values should be reasonable types
        assert isinstance(limits["max_concurrent_scans"], int)
        assert isinstance(limits["max_tasks_per_scan"], int)
        assert isinstance(limits["plan_timeout_seconds"], (int, float))
        assert isinstance(limits["scan_timeout_seconds"], (int, float))
        assert isinstance(limits["max_response_bytes"], int)
        assert isinstance(limits["active_scans"], int)

    def test_fetch_completes_within_limits(self, server_with_client):
        """Test that a normal fetch completes successfully."""
        client = server_with_client["client"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        # Should complete without hitting limits
        artifact = client.materialize(
            inputs=[table_uri],
            transform={"executor": "scan@v1", "params": {}},
        )
        table = client.fetch(artifact.uri)
        assert table.num_rows == 100


class TestTaskLimitEnforcement:
    """Tests for max_tasks_per_scan enforcement."""

    def test_task_limit_config(self, tmp_path):
        """Test that task limit can be configured."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            max_tasks_per_scan=5,
        )

        assert config.max_tasks_per_scan == 5


class TestPlanTimeoutConfig:
    """Tests for plan timeout configuration."""

    def test_plan_timeout_config(self, tmp_path):
        """Test that plan timeout can be configured."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            plan_timeout_seconds=15.0,
        )

        assert config.plan_timeout_seconds == 15.0

    def test_default_plan_timeout(self):
        """Test default plan timeout is 30 seconds."""
        config = StrataConfig()
        assert config.plan_timeout_seconds == 30.0


class TestScanTimeoutConfig:
    """Tests for scan timeout configuration."""

    def test_timeout_config(self, tmp_path):
        """Test that timeout can be configured."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            scan_timeout_seconds=30.0,
        )

        assert config.scan_timeout_seconds == 30.0


class TestResponseSizeLimitConfig:
    """Tests for max_response_bytes configuration."""

    def test_response_size_config(self, tmp_path):
        """Test that response size limit can be configured."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            max_response_bytes=50 * 1024 * 1024,  # 50 MB
        )

        assert config.max_response_bytes == 50 * 1024 * 1024


class TestConcurrentScanLimitConfig:
    """Tests for max_concurrent_scans configuration."""

    def test_concurrent_scan_config(self, tmp_path):
        """Test that concurrent scan limit can be configured."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            max_concurrent_scans=50,
        )

        assert config.max_concurrent_scans == 50
