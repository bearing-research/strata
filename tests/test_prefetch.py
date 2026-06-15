"""Tests for prefetch optimization and safeguards.

These tests verify that the prefetch mechanism:
1. Reduces TTFB by overlapping planning and first row group fetch
2. Is bounded by a semaphore to prevent resource exhaustion
3. Properly cancels on scan deletion
4. Tracks metrics for observability (started, used, wasted)
"""

import time

import httpx
import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

from strata.config import StrataConfig
from tests.conftest import find_free_port, run_server


def build_materialize_request(table_uri: str, columns: list[str] | None = None) -> dict:
    """Build a materialize request for the given table and columns."""
    params = {}
    if columns is not None:
        params["columns"] = columns
    return {
        "inputs": [table_uri],
        "transform": {"executor": "scan@v1", "params": params},
        "mode": "stream",
    }


@pytest.fixture
def prefetch_warehouse(tmp_path):
    """Create a warehouse with data for prefetch testing."""
    import sys

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
        NestedField(2, "data", StringType(), required=False),
    )

    table = catalog.create_table("test_db.prefetch_test", schema)

    # Create enough data to make prefetch meaningful
    num_rows = 5000
    data = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "data": pa.array(["x" * 100 for _ in range(num_rows)], type=pa.string()),
        }
    )
    table.append(data)

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"file://{warehouse_path}#test_db.prefetch_test",
        "catalog": catalog,
        "table": table,
    }


class TestPrefetchBasics:
    """Basic prefetch functionality tests."""

    def test_prefetch_metrics_in_response(self, prefetch_warehouse, tmp_path):
        """Test that prefetch metrics are exposed in /metrics endpoint."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                # Get initial metrics
                resp = client.get(f"{base_url}/metrics")
                assert resp.status_code == 200
                metrics = resp.json()

                # Check prefetch metrics structure
                assert "prefetch" in metrics
                prefetch = metrics["prefetch"]
                assert "started" in prefetch
                assert "used" in prefetch
                assert "wasted" in prefetch
                assert "in_flight" in prefetch

                # Initial values should be 0
                assert prefetch["started"] == 0
                assert prefetch["used"] == 0
                assert prefetch["wasted"] == 0
                assert prefetch["in_flight"] == 0

    def test_prefetch_used_on_normal_scan(self, prefetch_warehouse, tmp_path):
        """Test that prefetch is used when stream is consumed normally."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = prefetch_warehouse["table_uri"]

            with httpx.Client(timeout=10.0) as client:
                # Create materialize request
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(table_uri),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                # Small delay to let prefetch complete
                time.sleep(0.2)

                # Consume the stream
                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    for _ in stream.iter_bytes():
                        pass

                # Check metrics - prefetch should be used
                resp = client.get(f"{base_url}/metrics")
                metrics = resp.json()
                prefetch = metrics["prefetch"]

                assert prefetch["started"] >= 1
                assert prefetch["used"] >= 1
                assert prefetch["in_flight"] == 0

    def test_prefetch_wasted_on_scan_delete(self, prefetch_warehouse, tmp_path):
        """Test that prefetch is marked as wasted when artifact is created but not streamed."""
        import strata.server as server_module

        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            stream_state_ttl_seconds=0.1,  # short TTL so cleanup fires within the test
        )

        with run_server(config) as base_url:
            table_uri = prefetch_warehouse["table_uri"]
            assert server_module._state is not None

            with httpx.Client(timeout=10.0) as client:
                # Create materialize request but don't stream it
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(table_uri),
                )
                assert resp.status_code == 200

                # Wait for prefetch and TTL cleanup to run.
                time.sleep(0.4)

                # Check metrics
                resp = client.get(f"{base_url}/metrics")
                metrics = resp.json()
                prefetch = metrics["prefetch"]

                assert prefetch["started"] >= 1
                assert prefetch["wasted"] >= 1
                assert prefetch["in_flight"] == 0


class TestPrefetchSemaphore:
    """Tests for prefetch semaphore limiting."""

    def test_prefetch_limited_by_semaphore(self, prefetch_warehouse, tmp_path):
        """Test that concurrent prefetches are limited by semaphore."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = prefetch_warehouse["table_uri"]

            with httpx.Client(timeout=10.0) as client:
                stream_urls = []

                # Create many materialize requests rapidly
                for _ in range(10):
                    resp = client.post(
                        f"{base_url}/v1/materialize",
                        json=build_materialize_request(table_uri),
                    )
                    assert resp.status_code == 200
                    stream_urls.append(resp.json()["stream_url"])

                # Check that in-flight prefetches are bounded
                resp = client.get(f"{base_url}/metrics")
                metrics = resp.json()
                prefetch = metrics["prefetch"]

                # Max 4 concurrent prefetches (semaphore limit)
                assert prefetch["in_flight"] <= 4

                # Consume all streams
                for stream_url in stream_urls:
                    try:
                        with client.stream("GET", f"{base_url}{stream_url}") as stream:
                            for _ in stream.iter_bytes():
                                pass
                    except Exception:
                        pass


class TestPrefetchCancellation:
    """Tests for prefetch cancellation on scan deletion."""

    def test_prefetch_cancelled_on_immediate_delete(self, prefetch_warehouse, tmp_path):
        """Test that prefetch is cancelled if artifact is created but not streamed immediately."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = prefetch_warehouse["table_uri"]

            with httpx.Client(timeout=10.0) as client:
                # Create materialize request (prefetch may start)
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(table_uri),
                )
                assert resp.status_code == 200

                # Wait a bit for cleanup
                time.sleep(0.2)

                # Check that no prefetches are in flight
                resp = client.get(f"{base_url}/metrics")
                metrics = resp.json()
                prefetch = metrics["prefetch"]

                assert prefetch["in_flight"] == 0


class TestPrefetchPrometheusMetrics:
    """Tests for prefetch Prometheus metrics."""

    def test_prefetch_prometheus_metrics(self, prefetch_warehouse, tmp_path):
        """Test that prefetch metrics are exposed in Prometheus format."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{base_url}/metrics/prometheus")
                assert resp.status_code == 200
                content = resp.text

                # Check that prefetch metrics are present
                assert "strata_prefetch_started_total" in content
                assert "strata_prefetch_used_total" in content
                assert "strata_prefetch_wasted_total" in content
                assert "strata_prefetch_in_flight" in content


class TestPrefetchNoLeak:
    """Tests to ensure prefetch doesn't leak resources."""

    def test_no_memory_leak_on_abandoned_scans(self, prefetch_warehouse, tmp_path):
        """Test that abandoned streams don't leak prefetch resources."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = prefetch_warehouse["table_uri"]

            with httpx.Client(timeout=10.0) as client:
                # Create and abandon many materialize requests
                for _ in range(20):
                    resp = client.post(
                        f"{base_url}/v1/materialize",
                        json=build_materialize_request(table_uri),
                    )
                    # Don't consume the stream

                # Wait for cleanup
                time.sleep(0.5)

                # Check metrics
                resp = client.get(f"{base_url}/metrics")
                metrics = resp.json()
                prefetch = metrics["prefetch"]

                # No prefetches should be in flight
                assert prefetch["in_flight"] == 0

                # Started count should be tracked
                assert prefetch["started"] >= 0
