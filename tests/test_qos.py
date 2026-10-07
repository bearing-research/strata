"""Two-tier QoS admission: interactive vs bulk classification, isolation and metrics.

Also checks tier slots are released on completion, error and disconnect.
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
    """A warehouse with tables of different sizes for QoS tests."""
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

    # ~500KB, for interactive queries
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

    # ~15MB, for bulk queries
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
    def test_qos_metrics_in_json_endpoint(self, qos_warehouse, tmp_path):
        """QoS metrics are exposed on the /metrics JSON endpoint."""
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

                assert "qos" in metrics
                qos = metrics["qos"]
                assert qos["interactive_slots"] == 8
                assert qos["bulk_slots"] == 4
                assert "interactive_active" in qos
                assert "bulk_active" in qos
                assert "interactive_available" in qos
                assert "bulk_available" in qos

                assert qos["interactive_active"] == 0
                assert qos["bulk_active"] == 0
                assert qos["interactive_available"] == 8
                assert qos["bulk_available"] == 4

    def test_qos_metrics_in_prometheus_endpoint(self, qos_warehouse, tmp_path):
        """QoS metrics are exposed in Prometheus format."""
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

                assert "# HELP strata_qos_interactive_slots" in content
                assert "# TYPE strata_qos_interactive_slots gauge" in content
                assert "strata_qos_interactive_slots" in content
                assert "strata_qos_interactive_active" in content
                assert "strata_qos_bulk_slots" in content
                assert "strata_qos_bulk_active" in content


class TestQoSClassification:
    """Classification as interactive or bulk."""

    def test_small_query_succeeds(self, qos_warehouse, tmp_path):
        """A small query with few columns runs."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=["id", "data"],
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    bytes_read = 0
                    for chunk in stream.iter_bytes():
                        bytes_read += len(chunk)
                    assert bytes_read > 0

                metrics = client.get(f"{base_url}/metrics").json()
                assert metrics["qos"]["interactive_active"] == 0
                assert metrics["qos"]["bulk_active"] == 0

    def test_large_query_succeeds(self, qos_warehouse, tmp_path):
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["large_table_uri"],
                        columns=["id", "data"],
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    bytes_read = 0
                    for chunk in stream.iter_bytes():
                        bytes_read += len(chunk)
                    assert bytes_read > 0

                metrics = client.get(f"{base_url}/metrics").json()
                assert metrics["qos"]["interactive_active"] == 0
                assert metrics["qos"]["bulk_active"] == 0

    def test_full_scan_succeeds(self, qos_warehouse, tmp_path):
        """A full scan with all columns runs."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                # All columns (None)
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=None,
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    bytes_read = 0
                    for chunk in stream.iter_bytes():
                        bytes_read += len(chunk)
                    assert bytes_read > 0

                metrics = client.get(f"{base_url}/metrics").json()
                assert metrics["qos"]["interactive_active"] == 0
                assert metrics["qos"]["bulk_active"] == 0


class TestQoSTierIsolation:
    """Bulk queries do not starve interactive ones."""

    def test_interactive_query_succeeds(self, qos_warehouse, tmp_path):
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
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=["id"],
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    bytes_read = 0
                    for chunk in stream.iter_bytes():
                        bytes_read += len(chunk)
                    assert bytes_read > 0


class TestQoSSemaphoreCleanup:
    """Tier semaphore cleanup."""

    def test_tier_semaphore_released_on_completion(self, qos_warehouse, tmp_path):
        """The tier semaphore is released when the stream completes normally."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=["id"],
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    for _ in stream.iter_bytes():
                        pass

                metrics = client.get(f"{base_url}/metrics").json()
                qos = metrics["qos"]
                assert qos["interactive_active"] == 0
                assert qos["bulk_active"] == 0

    def test_tier_semaphore_released_on_scan_delete(self, qos_warehouse, tmp_path):
        """The tier semaphore is released when the artifact is created but not streamed."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=["id", "data"],
                    ),
                )
                assert resp.status_code == 200
                # Materialize consumes the stream, so the semaphores are already released.

                metrics = client.get(f"{base_url}/metrics").json()
                qos = metrics["qos"]
                assert qos["interactive_active"] == 0
                assert qos["bulk_active"] == 0

    def test_multiple_scans_release_correctly(self, qos_warehouse, tmp_path):
        """Several concurrent streams each release their semaphore."""
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

                for stream_url in stream_urls:
                    with client.stream("GET", f"{base_url}{stream_url}") as stream:
                        for _ in stream.iter_bytes():
                            pass

                metrics = client.get(f"{base_url}/metrics").json()
                qos = metrics["qos"]
                assert qos["interactive_active"] == 0
                assert qos["bulk_active"] == 0


class TestQoSConfiguration:
    def test_default_qos_metrics_exposed(self, qos_warehouse, tmp_path):
        """QoS metrics are exposed under the default configuration."""
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
                assert "interactive_slots" in qos
                assert "bulk_slots" in qos
                assert "interactive_active" in qos
                assert "bulk_active" in qos
                assert "interactive_available" in qos
                assert "bulk_available" in qos
                # Defaults are tuned for an 8-16 core box with bursts.
                assert qos["interactive_slots"] == 32
                assert qos["bulk_slots"] == 8

    def test_configured_slots_reach_admission_limiters(self, qos_warehouse, tmp_path):
        """Configured slots reach the per-tenant limiters the stream handler acquires.

        Non-default values make a missing init_tenant_registry wiring (falling back to 32/8) fail.
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
        """A query streams with QoS enabled."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(
                        qos_warehouse["small_table_uri"],
                        columns=["id", "data"],
                    ),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    bytes_read = 0
                    for chunk in stream.iter_bytes():
                        bytes_read += len(chunk)
                    assert bytes_read > 0


class TestQoSFastFail:
    """Fast-fail with 429 when no slot is free."""

    def test_rejection_metrics_tracked(self, qos_warehouse, tmp_path):
        """Rejection counts are tracked in metrics."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{base_url}/metrics")
                assert resp.status_code == 200
                metrics = resp.json()

                qos = metrics["qos"]
                assert "interactive_rejected" in qos
                assert "bulk_rejected" in qos
                assert qos["interactive_rejected"] == 0
                assert qos["bulk_rejected"] == 0

    def test_rejection_metrics_in_prometheus(self, qos_warehouse, tmp_path):
        """Rejection metrics appear in Prometheus format."""
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

                assert "strata_qos_interactive_rejected_total" in content
                assert "strata_qos_bulk_rejected_total" in content


# These two tests pin the observable QoS admission contract: the full
# /metrics["qos"] key set, and no slot leak when a client disconnects mid-stream.


# Every key in the ``qos`` block of ``/metrics``. Dropping or renaming one silently
# breaks the observability dashboard, so pin the exact set.
_EXPECTED_QOS_METRIC_KEYS = {
    "interactive_slots",
    "interactive_active",
    "interactive_available",
    "interactive_rejected",
    "interactive_queue_timeout_seconds",
    "interactive_queue_wait_avg_ms",
    "interactive_queue_wait_total_ms",
    "interactive_queue_wait_count",
    "bulk_slots",
    "bulk_active",
    "bulk_available",
    "bulk_rejected",
    "bulk_queue_timeout_seconds",
    "bulk_queue_wait_avg_ms",
    "bulk_queue_wait_total_ms",
    "bulk_queue_wait_count",
    "per_client_interactive",
    "per_client_bulk",
    "client_rejected",
    "tracked_clients",
    "per_tenant",
}


class TestQoSCharacterization:
    """Characterization tests pinning QoS behaviour over HTTP."""

    def test_qos_metrics_golden_shape(self, qos_warehouse, tmp_path):
        """The ``/metrics`` qos block exposes exactly the documented key set."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
        )
        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                qos = client.get(f"{base_url}/metrics").json()["qos"]
        assert set(qos.keys()) == _EXPECTED_QOS_METRIC_KEYS
        # per_tenant is a nested map; the rest are scalars.
        assert isinstance(qos["per_tenant"], dict)

    def test_no_qos_slot_active_after_abandoned_stream(self, qos_warehouse, tmp_path):
        """After a client abandons a stream, no tier slot stays active.

        An end-state guard only: a small response is fully buffered before the client reads, so a
        mid-flight disconnect is racy over HTTP. test_qos_admission.py covers the cancel path.
        """
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            deployment_mode="service",  # no artifact_dir: pass-through streaming
            auth_mode="trusted_proxy",
            proxy_token="test-token",
        )
        headers = {"X-Strata-Proxy-Token": "test-token", "X-Strata-Principal": "user-1"}
        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0, headers=headers) as client:
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(qos_warehouse["large_table_uri"]),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                with client.stream("GET", f"{base_url}{stream_url}") as stream:
                    for _ in stream.iter_bytes():
                        break  # take one chunk, then abandon the connection

                # Release lands when the server observes the dropped reader; poll rather than
                # assume it is instant.
                deadline = time.monotonic() + 10.0
                qos = client.get(f"{base_url}/metrics").json()["qos"]
                while (qos["interactive_active"] or qos["bulk_active"]) and (
                    time.monotonic() < deadline
                ):
                    time.sleep(0.1)
                    qos = client.get(f"{base_url}/metrics").json()["qos"]
                assert qos["interactive_active"] == 0, "interactive slot left active"
                assert qos["bulk_active"] == 0, "bulk slot left active"
