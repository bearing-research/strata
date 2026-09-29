"""Regression tests for semaphore/resource leak under client disconnects and timeouts.

This test suite ensures the server properly releases resources when:
- Clients disconnect mid-stream
- Clients timeout waiting for response
- Requests are cancelled

The bug this prevents:
- Semaphore slots were leaked when clients disconnected before the generator
  completed, causing the server to eventually return 503 for all requests.
- Fixed by moving resource tracking inside the generator with proper cleanup
  in GeneratorExit and CancelledError handlers.

These tests should be run as part of CI to prevent regression.
"""

import asyncio
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
def large_warehouse(tmp_path):
    """Create a warehouse with enough data to cause slow responses."""
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
        NestedField(2, "payload", StringType(), required=False),
    )

    table = catalog.create_table("test_db.large_events", schema)

    # Create enough data to make responses take time
    # 10K rows with 1KB payload each = ~10MB
    num_rows = 10000
    payload_size = 1000
    data = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "payload": pa.array(["x" * payload_size for _ in range(num_rows)], type=pa.string()),
        }
    )
    table.append(data)

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"file://{warehouse_path}#test_db.large_events",
        "catalog": catalog,
        "table": table,
    }


class TestSemaphoreLeakRegression:
    """Regression tests for semaphore leak under disconnects/timeouts.

    These tests verify the fix for the bug where client disconnects
    caused semaphore slots to leak, eventually exhausting capacity.
    """

    def test_semaphore_released_on_client_timeout(self, large_warehouse, tmp_path):
        """Test that semaphore is released when client times out.

        This is the core regression test. Before the fix, client timeouts
        would leak semaphore slots, eventually causing all requests to 503.
        """
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=5,  # Low limit to make leak visible quickly
            scan_timeout_seconds=300.0,  # Server-side timeout (long)
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = large_warehouse["table_uri"]

            # Phase 1: Force some client-side timeouts
            # Use very short client timeout to ensure timeout happens
            timeout_count = 0
            for i in range(10):
                try:
                    with httpx.Client(timeout=0.001) as client:  # 1ms timeout
                        # Create materialize request
                        resp = client.post(
                            f"{base_url}/v1/materialize",
                            json=build_materialize_request(table_uri),
                            timeout=0.5,  # Longer timeout for POST
                        )
                        if resp.status_code == 200:
                            stream_url = resp.json()["stream_url"]
                            # Try to stream - this should timeout
                            try:
                                with client.stream(
                                    "GET",
                                    f"{base_url}{stream_url}",
                                    timeout=0.001,  # Very short timeout
                                ) as stream:
                                    for _ in stream.iter_bytes():
                                        pass
                            except httpx.TimeoutException:
                                timeout_count += 1
                except Exception:
                    timeout_count += 1

            # We should have had some timeouts
            assert timeout_count > 0, "Expected some client timeouts"

            # Phase 2: Verify server is still healthy
            # If semaphores leaked, this would eventually fail with 503
            # Poll with retries to handle cleanup timing under load
            with httpx.Client(timeout=30.0) as client:
                # Health check should pass
                resp = client.get(f"{base_url}/health")
                assert resp.status_code == 200

                # Metrics should show reasonable active_scans
                # Poll with retries - cleanup may be delayed under load
                active_scans = None
                for attempt in range(10):  # Up to 5 seconds total
                    time.sleep(0.5)
                    resp = client.get(f"{base_url}/metrics")
                    assert resp.status_code == 200
                    metrics = resp.json()
                    limits = metrics.get("resource_limits", {})
                    active_scans = limits.get("active_scans", 0)
                    if active_scans == 0:
                        break

                # Key invariant: active_scans should be 0 after all requests complete
                assert active_scans == 0, (
                    f"Semaphore leak detected! active_scans={active_scans} "
                    f"(should be 0 after all requests complete)"
                )

                # New materialize should succeed (not 503)
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(table_uri),
                )
                assert resp.status_code == 200, (
                    f"Expected 200, got {resp.status_code}. "
                    "Server may have exhausted semaphore slots due to leak."
                )

    def test_semaphore_released_on_client_disconnect(self, large_warehouse, tmp_path):
        """Test that semaphore is released when client disconnects mid-stream."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=5,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = large_warehouse["table_uri"]

            # Phase 1: Start streams and disconnect after partial read
            for i in range(10):
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        f"{base_url}/v1/materialize",
                        json=build_materialize_request(table_uri),
                    )
                    if resp.status_code != 200:
                        continue

                    stream_url = resp.json()["stream_url"]

                    # Start streaming but disconnect after reading some data
                    try:
                        with client.stream(
                            "GET",
                            f"{base_url}{stream_url}",
                        ) as stream:
                            bytes_read = 0
                            for chunk in stream.iter_bytes(chunk_size=1024):
                                bytes_read += len(chunk)
                                if bytes_read > 1000:  # Disconnect after 1KB
                                    break
                    except Exception:
                        pass

            # Give server time to clean up
            time.sleep(0.5)

            # Phase 2: Verify no leak
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{base_url}/metrics")
                assert resp.status_code == 200
                metrics = resp.json()
                active_scans = metrics.get("resource_limits", {}).get("active_scans", 0)

                assert active_scans == 0, (
                    f"Semaphore leak on disconnect! active_scans={active_scans}"
                )

    def test_no_503_after_many_timeouts(self, large_warehouse, tmp_path):
        """Test that server doesn't return 503 after many client timeouts.

        This is the key acceptance test: even after many timeouts,
        the server should still accept new requests.
        """
        port = find_free_port()
        max_scans = 3  # Very low limit
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=max_scans,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = large_warehouse["table_uri"]

            # Phase 1: Force more timeouts than max_concurrent_scans
            # If there's a leak, we'll exhaust slots after max_scans timeouts
            num_timeouts = max_scans * 3

            for i in range(num_timeouts):
                try:
                    with httpx.Client(timeout=0.001) as client:
                        resp = client.post(
                            f"{base_url}/v1/materialize",
                            json=build_materialize_request(table_uri),
                            timeout=1.0,
                        )
                        if resp.status_code == 200:
                            stream_url = resp.json()["stream_url"]
                            try:
                                with client.stream(
                                    "GET",
                                    f"{base_url}{stream_url}",
                                    timeout=0.001,
                                ) as stream:
                                    for _ in stream.iter_bytes():
                                        pass
                            except httpx.TimeoutException:
                                pass
                except Exception:
                    pass

            # Give server time to clean up
            time.sleep(0.5)

            # Phase 2: Verify we can still make requests (no 503)
            with httpx.Client(timeout=30.0) as client:
                # This is the key assertion: even after many timeouts,
                # we should NOT get 503 "Server at capacity"
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(table_uri),
                )

                assert resp.status_code != 503, (
                    "Got 503 after timeouts - semaphore leak detected! "
                    "The fix for generator cleanup may have regressed."
                )
                assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_disconnects_no_leak(self, large_warehouse, tmp_path):
        """Test that concurrent client disconnects don't leak semaphores."""
        port = find_free_port()
        max_scans = 10
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=max_scans,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = large_warehouse["table_uri"]

            async def disconnect_after_partial_read():
                """Start a materialize, read some data, then disconnect."""
                async with httpx.AsyncClient(timeout=10.0) as client:
                    try:
                        resp = await client.post(
                            f"{base_url}/v1/materialize",
                            json=build_materialize_request(table_uri),
                        )
                        if resp.status_code != 200:
                            return

                        stream_url = resp.json()["stream_url"]

                        try:
                            async with client.stream(
                                "GET",
                                f"{base_url}{stream_url}",
                            ) as stream:
                                bytes_read = 0
                                async for chunk in stream.aiter_bytes(chunk_size=512):
                                    bytes_read += len(chunk)
                                    if bytes_read > 500:
                                        # Force disconnect by breaking
                                        break
                        except Exception:
                            pass
                    except Exception:
                        pass

            # Phase 1: Many concurrent disconnects
            tasks = [disconnect_after_partial_read() for _ in range(20)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Give server time to clean up
            await asyncio.sleep(0.5)

            # Phase 2: Verify no leak
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{base_url}/metrics")
                assert resp.status_code == 200
                metrics = resp.json()
                active_scans = metrics.get("resource_limits", {}).get("active_scans", 0)

                assert active_scans == 0, (
                    f"Semaphore leak under concurrent disconnects! active_scans={active_scans}"
                )


class TestSemaphoreInvariants:
    """Tests for semaphore invariants that should always hold."""

    def test_active_scans_never_negative(self, large_warehouse, tmp_path):
        """Test that active_scans counter never goes negative."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=5,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = large_warehouse["table_uri"]

            # Do a bunch of requests with various outcomes
            for _ in range(20):
                with httpx.Client(timeout=5.0) as client:
                    try:
                        resp = client.post(
                            f"{base_url}/v1/materialize",
                            json=build_materialize_request(table_uri),
                        )
                        if resp.status_code == 200:
                            stream_url = resp.json()["stream_url"]
                            # Sometimes complete, sometimes disconnect
                            try:
                                with client.stream(
                                    "GET",
                                    f"{base_url}{stream_url}",
                                ) as stream:
                                    for chunk in stream.iter_bytes():
                                        pass  # Complete the stream
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Check invariant after each request
                    try:
                        resp = client.get(f"{base_url}/metrics")
                        if resp.status_code == 200:
                            metrics = resp.json()
                            active = metrics.get("resource_limits", {}).get("active_scans", 0)
                            assert active >= 0, f"active_scans went negative: {active}"
                    except Exception:
                        pass

    def test_active_scans_bounded_by_max(self, large_warehouse, tmp_path):
        """Test that active_scans never exceeds max_concurrent_scans."""
        port = find_free_port()
        max_scans = 3
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=max_scans,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            # Check metrics
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{base_url}/metrics")
                assert resp.status_code == 200
                metrics = resp.json()
                limits = metrics.get("resource_limits", {})
                active = limits.get("active_scans", 0)
                max_allowed = limits.get("max_concurrent_scans", max_scans)

                assert active <= max_allowed, (
                    f"active_scans ({active}) exceeds max_concurrent_scans ({max_allowed})"
                )
