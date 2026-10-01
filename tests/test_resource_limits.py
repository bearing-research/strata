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
    def test_default_limits(self):
        config = StrataConfig()

        assert config.max_concurrent_scans == 100
        assert config.max_tasks_per_scan == 1000
        assert config.plan_timeout_seconds == 30.0
        assert config.scan_timeout_seconds == 300.0
        assert config.max_response_bytes == 512 * 1024 * 1024  # 512 MB

    def test_custom_limits(self, tmp_path):
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
    """Server-side resource limit enforcement."""

    @pytest.fixture
    def server_with_client(self, temp_warehouse, tmp_path):
        """A server with custom limits, and a client for it."""
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            max_concurrent_scans=2,
            max_tasks_per_scan=10,
            scan_timeout_seconds=5.0,
            max_response_bytes=1024 * 1024,
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
        """The metrics endpoint includes resource limit info."""
        client = server_with_client["client"]

        metrics = client.metrics()

        assert "resource_limits" in metrics
        limits = metrics["resource_limits"]

        assert "max_concurrent_scans" in limits
        assert "max_tasks_per_scan" in limits
        assert "plan_timeout_seconds" in limits
        assert "scan_timeout_seconds" in limits
        assert "max_response_bytes" in limits
        assert "active_scans" in limits

        assert isinstance(limits["max_concurrent_scans"], int)
        assert isinstance(limits["max_tasks_per_scan"], int)
        assert isinstance(limits["plan_timeout_seconds"], (int, float))
        assert isinstance(limits["scan_timeout_seconds"], (int, float))
        assert isinstance(limits["max_response_bytes"], int)
        assert isinstance(limits["active_scans"], int)

    def test_fetch_completes_within_limits(self, server_with_client):
        """A normal fetch completes."""
        client = server_with_client["client"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        artifact = client.materialize(
            inputs=[table_uri],
            transform={"executor": "scan@v1", "params": {}},
        )
        table = client.fetch(artifact.uri)
        assert table.num_rows == 100
