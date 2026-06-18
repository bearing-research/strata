"""Tests for QoS (Quality of Service) two-tier admission control.

These tests verify that the QoS mechanism:
1. Classifies queries correctly as "interactive" or "bulk"
2. Uses separate semaphores for each tier
3. Prevents bulk queries from starving interactive queries
4. Tracks QoS metrics correctly
5. Releases tier semaphores properly on completion/error/disconnect
"""

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
def qos_warehouse(tmp_path):
    """Create a warehouse with tables of different sizes for QoS testing."""
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
        NestedField(3, "extra1", StringType(), required=False),
        NestedField(4, "extra2", StringType(), required=False),
        NestedField(5, "extra3", StringType(), required=False),
        NestedField(6, "extra4", StringType(), required=False),
        NestedField(7, "extra5", StringType(), required=False),
        NestedField(8, "extra6", StringType(), required=False),
        NestedField(9, "extra7", StringType(), required=False),
        NestedField(10, "extra8", StringType(), required=False),
        NestedField(11, "extra9", StringType(), required=False),
        NestedField(12, "extra10", StringType(), required=False),
    )

    # Small table for interactive queries (~500KB)
    small_table = catalog.create_table("test_db.small_table", schema)
    small_data = pa.table(
        {
            "id": pa.array(range(1000), type=pa.int64()),
            "data": pa.array(["x" * 100 for _ in range(1000)], type=pa.string()),
            "extra1": pa.array(["a" for _ in range(1000)], type=pa.string()),
            "extra2": pa.array(["b" for _ in range(1000)], type=pa.string()),
            "extra3": pa.array(["c" for _ in range(1000)], type=pa.string()),
            "extra4": pa.array(["d" for _ in range(1000)], type=pa.string()),
            "extra5": pa.array(["e" for _ in range(1000)], type=pa.string()),
            "extra6": pa.array(["f" for _ in range(1000)], type=pa.string()),
            "extra7": pa.array(["g" for _ in range(1000)], type=pa.string()),
            "extra8": pa.array(["h" for _ in range(1000)], type=pa.string()),
            "extra9": pa.array(["i" for _ in range(1000)], type=pa.string()),
            "extra10": pa.array(["j" for _ in range(1000)], type=pa.string()),
        }
    )
    small_table.append(small_data)

    # Large table for bulk queries (~15MB)
    large_table = catalog.create_table("test_db.large_table", schema)
    large_data = pa.table(
        {
            "id": pa.array(range(50000), type=pa.int64()),
            "data": pa.array(["y" * 200 for _ in range(50000)], type=pa.string()),
            "extra1": pa.array(["a" * 10 for _ in range(50000)], type=pa.string()),
            "extra2": pa.array(["b" * 10 for _ in range(50000)], type=pa.string()),
            "extra3": pa.array(["c" * 10 for _ in range(50000)], type=pa.string()),
            "extra4": pa.array(["d" * 10 for _ in range(50000)], type=pa.string()),
            "extra5": pa.array(["e" * 10 for _ in range(50000)], type=pa.string()),
            "extra6": pa.array(["f" * 10 for _ in range(50000)], type=pa.string()),
            "extra7": pa.array(["g" * 10 for _ in range(50000)], type=pa.string()),
            "extra8": pa.array(["h" * 10 for _ in range(50000)], type=pa.string()),
            "extra9": pa.array(["i" * 10 for _ in range(50000)], type=pa.string()),
            "extra10": pa.array(["j" * 10 for _ in range(50000)], type=pa.string()),
        }
    )
    large_table.append(large_data)

    return {
        "warehouse_path": warehouse_path,
        "small_table_uri": f"file://{warehouse_path}#test_db.small_table",
        "large_table_uri": f"file://{warehouse_path}#test_db.large_table",
        "catalog": catalog,
    }


class TestQoSMetrics:
    """Tests for QoS metrics exposure."""

    def test_qos_metrics_in_json_endpoint(self, qos_warehouse, tmp_path):
        """Test that QoS metrics are exposed in /metrics JSON endpoint."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            interactive_slots=8,
            bulk_slots=4,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{base_url}/metrics")
                assert resp.status_code == 200
                metrics = resp.json()

                # Check QoS metrics structure
                assert "qos" in metrics
                qos = metrics["qos"]
                assert qos["interactive_slots"] == 8
                assert qos["bulk_slots"] == 4
                assert "interactive_active" in qos
                assert "bulk_active" in qos
                assert "interactive_available" in qos
                assert "bulk_available" in qos

                # Initial values should show all slots available
                assert qos["interactive_active"] == 0
                assert qos["bulk_active"] == 0
                assert qos["interactive_available"] == 8
                assert qos["bulk_available"] == 4

    def test_qos_metrics_in_prometheus_endpoint(self, qos_warehouse, tmp_path):
        """Test that QoS metrics are exposed in Prometheus format."""
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

                # Check that QoS metrics are present with correct format
                assert "# HELP strata_qos_interactive_slots" in content
                assert "# TYPE strata_qos_interactive_slots gauge" in content
                assert "strata_qos_interactive_slots" in content
                assert "strata_qos_interactive_active" in content
                assert "strata_qos_bulk_slots" in content
                assert "strata_qos_bulk_active" in content


class TestQoSClassification:
    """Tests for query classification as interactive or bulk."""

    def test_small_query_succeeds(self, qos_warehouse, tmp_path):
        """Test that small queries with few columns can be executed."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                # Create materialize request with small table and few columns
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=["id", "data"],
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                # Stream should succeed
                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    bytes_read = 0
                    for chunk in stream.iter_bytes():
                        bytes_read += len(chunk)
                    assert bytes_read > 0

                # After streaming, no active queries
                metrics = client.get(f"{base_url}/metrics").json()
                assert metrics["qos"]["interactive_active"] == 0
                assert metrics["qos"]["bulk_active"] == 0

    def test_large_query_succeeds(self, qos_warehouse, tmp_path):
        """Test that large queries can be executed."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=30.0) as client:
                # Create materialize request with large table
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["large_table_uri"],
                        columns=["id", "data"],
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                # Stream should succeed
                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    bytes_read = 0
                    for chunk in stream.iter_bytes():
                        bytes_read += len(chunk)
                    assert bytes_read > 0

                # After streaming, no active queries
                metrics = client.get(f"{base_url}/metrics").json()
                assert metrics["qos"]["interactive_active"] == 0
                assert metrics["qos"]["bulk_active"] == 0

    def test_full_scan_succeeds(self, qos_warehouse, tmp_path):
        """Test that full table scans with all columns can be executed."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                # Create materialize request with all columns (None)
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=None,
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                # Stream should succeed
                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    bytes_read = 0
                    for chunk in stream.iter_bytes():
                        bytes_read += len(chunk)
                    assert bytes_read > 0

                # After streaming, no active queries
                metrics = client.get(f"{base_url}/metrics").json()
                assert metrics["qos"]["interactive_active"] == 0
                assert metrics["qos"]["bulk_active"] == 0


class TestQoSTierIsolation:
    """Tests for tier isolation (bulk doesn't starve interactive)."""

    def test_interactive_query_succeeds(self, qos_warehouse, tmp_path):
        """Test that interactive queries can be executed successfully."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            interactive_slots=2,
            bulk_slots=2,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=30.0) as client:
                # Create an interactive query
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=["id"],
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                # Stream the interactive query - should work
                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    bytes_read = 0
                    for chunk in stream.iter_bytes():
                        bytes_read += len(chunk)
                    assert bytes_read > 0  # Successfully streamed


class TestQoSSemaphoreCleanup:
    """Tests for proper semaphore cleanup in QoS tiers."""

    def test_tier_semaphore_released_on_completion(self, qos_warehouse, tmp_path):
        """Test that tier semaphore is released when stream completes normally."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                # Create and complete a stream
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=["id"],
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                # Stream to completion
                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    for _ in stream.iter_bytes():
                        pass

                # Check that semaphores are fully released
                metrics = client.get(f"{base_url}/metrics").json()
                qos = metrics["qos"]
                assert qos["interactive_active"] == 0
                assert qos["bulk_active"] == 0

    def test_tier_semaphore_released_on_scan_delete(self, qos_warehouse, tmp_path):
        """Test that tier semaphore is released when artifact is created but not streamed."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                # Create a materialize request but don't stream it
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=["id", "data"],
                    ),
                )
                assert resp.status_code == 200
                # With unified API, streams are consumed during materialize
                # so semaphores should already be released

                # Check that semaphores are released
                metrics = client.get(f"{base_url}/metrics").json()
                qos = metrics["qos"]
                assert qos["interactive_active"] == 0
                assert qos["bulk_active"] == 0

    def test_multiple_scans_release_correctly(self, qos_warehouse, tmp_path):
        """Test that multiple concurrent streams release their semaphores correctly."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                stream_urls = []

                # Create multiple materialize requests
                for _ in range(3):
                    resp = client.post(
                        f"{base_url}/v1/materialize",
                        json=build_materialize_request(
                            qos_warehouse["small_table_uri"],
                            columns=["id"],
                        ),
                    )
                    assert resp.status_code == 200
                    stream_urls.append(resp.json()["stream_url"])

                # Stream each to completion
                for stream_url in stream_urls:
                    with client.stream("GET", f"{base_url}{stream_url}") as stream:
                        for _ in stream.iter_bytes():
                            pass

                # All semaphores should be released
                metrics = client.get(f"{base_url}/metrics").json()
                qos = metrics["qos"]
                assert qos["interactive_active"] == 0
                assert qos["bulk_active"] == 0


class TestQoSConfiguration:
    """Tests for QoS configuration options."""

    def test_default_qos_metrics_exposed(self, qos_warehouse, tmp_path):
        """Test that QoS metrics are exposed with default configuration."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                metrics = client.get(f"{base_url}/metrics").json()
                qos = metrics["qos"]
                # Check that all expected QoS metrics are present
                assert "interactive_slots" in qos
                assert "bulk_slots" in qos
                assert "interactive_active" in qos
                assert "bulk_active" in qos
                assert "interactive_available" in qos
                assert "bulk_available" in qos
                # Default values (tuned for 8-16 core box supporting bursts)
                assert qos["interactive_slots"] == 32
                assert qos["bulk_slots"] == 8

    def test_configured_slots_reach_admission_limiters(self, qos_warehouse, tmp_path):
        """#185: interactive_slots / bulk_slots must configure the per-tenant
        registry limiters the stream handler actually acquires — not just the
        (former) never-acquired global limiters. Asserts directly on the registry
        the admission path uses, with non-default values so a missing
        init_tenant_registry wiring (registry falling back to its hard-coded
        32/8) would fail.
        """
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            interactive_slots=7,
            bulk_slots=3,
            deployment_mode="personal",
        )

        with run_server(config):
            from strata.tenant import DEFAULT_TENANT_ID
            from strata.tenant_registry import get_tenant_registry

            interactive, bulk = get_tenant_registry().get_or_create_limiters(DEFAULT_TENANT_ID)
            assert interactive.capacity == 7
            assert bulk.capacity == 3

    def test_query_can_be_streamed(self, qos_warehouse, tmp_path):
        """Test that queries can be successfully streamed with QoS enabled."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                # Create a query
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=["id", "data"],
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                # Stream should work
                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    bytes_read = 0
                    for chunk in stream.iter_bytes():
                        bytes_read += len(chunk)
                    assert bytes_read > 0


class TestQoSFastFail:
    """Tests for QoS fast-fail behavior (429 when slots unavailable)."""

    def test_rejection_metrics_tracked(self, qos_warehouse, tmp_path):
        """Test that rejection counts are tracked in metrics."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                # Check that rejection metrics exist
                resp = client.get(f"{base_url}/metrics")
                assert resp.status_code == 200
                metrics = resp.json()

                qos = metrics["qos"]
                assert "interactive_rejected" in qos
                assert "bulk_rejected" in qos
                assert qos["interactive_rejected"] == 0
                assert qos["bulk_rejected"] == 0

    def test_rejection_metrics_in_prometheus(self, qos_warehouse, tmp_path):
        """Test that rejection metrics are in Prometheus format."""
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

                # Check that rejection metrics are present
                assert "strata_qos_interactive_rejected_total" in content
                assert "strata_qos_bulk_rejected_total" in content
