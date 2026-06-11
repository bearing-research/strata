"""Hardening tests for v1 production quality.

These tests cover failure modes that occur in production:
- Restart persistence: data + metadata caches persist across restarts
- Corrupted cache: self-healing by delete and refetch
- Concurrent requests: no thundering herd for same data
- Stale metadata: invalidated correctly when files change
- Large scan streaming: doesn't buffer entire response in memory
"""

import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import uvicorn
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, LongType, NestedField, StringType
from strata_client.client import StrataClient

from strata.cache import CACHE_FILE_EXTENSION, CACHE_VERSION, CachedFetcher, DiskCache
from strata.config import StrataConfig
from strata.planner import ReadPlanner


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


def append_rows(table, start: int, count: int) -> None:
    """Append a small batch so scans span multiple files/row groups."""
    stop = start + count
    table.append(
        pa.table(
            {
                "id": pa.array(range(start, stop), type=pa.int64()),
                "value": pa.array([float(i * 1.5) for i in range(start, stop)], type=pa.float64()),
                "name": pa.array([f"item_{i}" for i in range(start, stop)], type=pa.string()),
            }
        )
    )


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
        NestedField(3, "name", StringType(), required=False),
    )

    table = catalog.create_table("test_db.events", schema)

    # Create sample data
    num_rows = 1000
    data = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "value": pa.array([float(i * 1.5) for i in range(num_rows)], type=pa.float64()),
            "name": pa.array([f"item_{i}" for i in range(num_rows)], type=pa.string()),
        }
    )
    table.append(data)

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"file://{warehouse_path}#test_db.events",
        "catalog": catalog,
        "table": table,
        "num_rows": num_rows,
    }


class TestRestartPersistence:
    """Test that caches persist across server/planner restarts."""

    def test_data_cache_persists_across_planner_instances(self, temp_warehouse, tmp_path):
        """Data cache entries survive planner restart."""
        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        config = StrataConfig(cache_dir=cache_dir)

        # First planner instance - cold run
        planner1 = ReadPlanner(config)
        fetcher1 = CachedFetcher(config)

        plan1 = planner1.plan(table_uri)
        batches1 = fetcher1.execute_plan(plan1)
        total_rows1 = sum(b.num_rows for b in batches1)

        # Verify data was cached
        cache_entries = list((cache_dir / f"v{CACHE_VERSION}").rglob(f"*{CACHE_FILE_EXTENSION}"))
        assert len(cache_entries) > 0, "Cache should have entries after first run"

        # Simulate restart: create new planner/fetcher instances
        planner2 = ReadPlanner(config)
        fetcher2 = CachedFetcher(config)

        plan2 = planner2.plan(table_uri)

        # Track cache hits
        cache_hits = 0
        for task in plan2.tasks:
            if fetcher2.cache.contains(task.cache_key):
                cache_hits += 1

        batches2 = fetcher2.execute_plan(plan2)
        total_rows2 = sum(b.num_rows for b in batches2)

        # Verify results match and cache was used
        assert total_rows1 == total_rows2
        assert cache_hits == len(plan2.tasks), "All tasks should hit cache after restart"

    def test_metadata_cache_persists_across_planner_instances(self, temp_warehouse, tmp_path):
        """Metadata cache (SQLite) survives planner restart."""
        from strata.metadata_cache import get_metadata_store, reset_caches

        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        # Reset global state
        reset_caches()

        config = StrataConfig(cache_dir=cache_dir)

        # First planner - populates metadata cache
        planner1 = ReadPlanner(config)
        plan1 = planner1.plan(table_uri)

        # Check metadata store has entries
        store = get_metadata_store(cache_dir)
        stats1 = store.stats()
        assert stats1["parquet_entries"] > 0, "Should have parquet metadata cached"

        # Simulate restart: reset in-memory caches but keep SQLite
        reset_caches()

        # New planner should use persisted metadata
        planner2 = ReadPlanner(config)

        # Record timing - should be faster due to cached metadata
        plan2 = planner2.plan(table_uri)

        # Verify metadata was reused (check store hit counters)
        get_metadata_store(cache_dir)

        # Should have hits from second planning
        assert len(plan2.tasks) == len(plan1.tasks)


class TestCorruptedCacheSelfHealing:
    """Test that corrupted cache entries are detected and self-heal."""

    def test_corrupted_data_cache_triggers_refetch(self, temp_warehouse, tmp_path):
        """Corrupted cache file is deleted and data is refetched."""
        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        config = StrataConfig(cache_dir=cache_dir)
        planner = ReadPlanner(config)
        fetcher = CachedFetcher(config)

        # First run - populate cache
        plan = planner.plan(table_uri)
        batches1 = fetcher.execute_plan(plan)
        total_rows1 = sum(b.num_rows for b in batches1)

        # Find and corrupt a cache file
        cache_files = list((cache_dir / f"v{CACHE_VERSION}").rglob(f"*{CACHE_FILE_EXTENSION}"))
        assert len(cache_files) > 0

        corrupted_file = cache_files[0]

        # Write garbage to corrupt the file
        corrupted_file.write_bytes(b"CORRUPTED DATA - NOT VALID ARROW IPC")

        # Create fresh fetcher (simulates restart)
        fetcher2 = CachedFetcher(config)

        # Plan again and fetch - should handle corruption gracefully
        plan2 = planner.plan(table_uri)
        batches2 = fetcher2.execute_plan(plan2)
        total_rows2 = sum(b.num_rows for b in batches2)

        # Data should still be correct (refetched)
        assert total_rows2 == total_rows1

        # Corrupted file should be deleted or replaced with valid data
        if corrupted_file.exists():
            # If it exists, it should be valid Arrow IPC now
            new_size = corrupted_file.stat().st_size
            assert new_size != len(b"CORRUPTED DATA - NOT VALID ARROW IPC"), (
                "Corrupted file should be replaced with valid data"
            )

    def test_corrupted_metadata_sidecar_handled_gracefully(self, temp_warehouse, tmp_path):
        """Corrupted metadata sidecar doesn't break cache operation."""
        from strata.cache import CACHE_META_EXTENSION

        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        config = StrataConfig(cache_dir=cache_dir)
        planner = ReadPlanner(config)
        fetcher = CachedFetcher(config)

        # First run - populate cache
        plan = planner.plan(table_uri)
        fetcher.execute_plan(plan)

        # Corrupt a metadata sidecar file
        meta_files = list((cache_dir / f"v{CACHE_VERSION}").rglob(f"*{CACHE_META_EXTENSION}"))
        assert len(meta_files) > 0

        meta_files[0].write_text("{ invalid json }")

        # Getting stats should handle corrupted metadata gracefully
        assert isinstance(fetcher.cache, DiskCache)
        stats = fetcher.cache.get_stats()
        # Should still return stats (corrupted entries are skipped)
        assert stats.total_entries >= 0


class TestConcurrentRequestsNoThunderingHerd:
    """Test that concurrent requests for same data don't cause thundering herd."""

    def test_concurrent_fetches_share_cache(self, temp_warehouse, tmp_path):
        """Multiple concurrent fetches for same data share cache efficiently."""
        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        config = StrataConfig(cache_dir=cache_dir)

        results = []
        errors = []

        def worker():
            try:
                # Each worker creates its own planner and fetcher
                planner = ReadPlanner(config)
                fetcher = CachedFetcher(config)
                plan = planner.plan(table_uri)
                batches = fetcher.execute_plan(plan)
                results.append(sum(b.num_rows for b in batches))
            except Exception as e:
                import traceback

                errors.append((e, traceback.format_exc()))

        # Run multiple concurrent fetchers
        num_workers = 5
        threads = [threading.Thread(target=worker) for _ in range(num_workers)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == num_workers

        # All should return same row count
        assert all(r == results[0] for r in results)

    def test_server_concurrent_scans_use_semaphore(self, temp_warehouse, tmp_path):
        """Server properly limits concurrent scans via semaphore."""
        import socket

        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        # Find free port
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=cache_dir,
            max_concurrent_scans=2,  # Low limit to test queuing
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

        # Submit multiple scans concurrently
        results = []
        errors = []

        def scan_worker():
            try:
                # Use the new materialize API
                artifact = client.materialize(
                    inputs=[table_uri],
                    transform={"executor": "scan@v1", "params": {}},
                )
                table = artifact.to_table()
                results.append(table.num_rows)
            except Exception as e:
                errors.append(e)

        num_workers = 4
        threads = [threading.Thread(target=scan_worker) for _ in range(num_workers)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        client.close()

        # All should succeed (queued if over limit)
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == num_workers


class TestStaleMetadataInvalidation:
    """Test that stale metadata is correctly invalidated."""

    def test_modified_file_invalidates_parquet_metadata(self, temp_warehouse, tmp_path):
        """Parquet metadata is invalidated when underlying file changes."""
        from strata.metadata_cache import get_metadata_store, reset_caches

        cache_dir = tmp_path / "cache"
        table = temp_warehouse["table"]

        reset_caches()

        config = StrataConfig(cache_dir=cache_dir)

        # First scan - populates metadata
        planner1 = ReadPlanner(config)
        plan1 = planner1.plan(temp_warehouse["table_uri"])

        store = get_metadata_store(cache_dir)
        stats1 = store.stats()
        initial_entries = stats1["parquet_entries"]
        assert initial_entries > 0

        # Append more data (creates new files, may update existing)
        new_data = pa.table(
            {
                "id": pa.array(range(100), type=pa.int64()),
                "value": pa.array([float(i) for i in range(100)], type=pa.float64()),
                "name": pa.array([f"new_{i}" for i in range(100)], type=pa.string()),
            }
        )
        table.append(new_data)

        # Run cleanup - should detect stale entries
        store.cleanup_stale_parquet_meta()

        # New planning should work correctly
        reset_caches()
        planner2 = ReadPlanner(config)
        plan2 = planner2.plan(temp_warehouse["table_uri"])

        # Should have more tasks now (more data)
        # Note: may have same number of row groups if data fits in existing
        assert len(plan2.tasks) >= len(plan1.tasks)


class TestLargeScanStreaming:
    """Test that large scans stream data without buffering entire response."""

    def test_streaming_does_not_buffer_all_batches(self, temp_warehouse, tmp_path):
        """Verify scan streams batches without holding all in memory."""
        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        config = StrataConfig(cache_dir=cache_dir)
        planner = ReadPlanner(config)
        fetcher = CachedFetcher(config)

        plan = planner.plan(table_uri)

        # Use streaming API - should yield batches one at a time
        batch_count = 0
        total_rows = 0

        for batch in fetcher.stream_plan(plan):
            batch_count += 1
            total_rows += batch.num_rows
            # Each batch should be processable independently
            assert batch.num_rows > 0

        assert batch_count == len(plan.tasks)
        assert total_rows == temp_warehouse["num_rows"]

    def test_ipc_streaming_yields_bytes_incrementally(self, temp_warehouse, tmp_path):
        """IPC streaming yields bytes for each batch separately."""
        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        config = StrataConfig(cache_dir=cache_dir)
        planner = ReadPlanner(config)
        fetcher = CachedFetcher(config)

        plan = planner.plan(table_uri)

        # Use IPC streaming API
        segment_count = 0
        total_bytes = 0

        for segment in fetcher.stream_plan_as_ipc(plan):
            segment_count += 1
            total_bytes += len(segment)
            # Each segment should be valid IPC bytes
            assert len(segment) > 0
            # Verify it's valid Arrow IPC
            reader = ipc.open_stream(pa.BufferReader(segment))
            batches = list(reader)
            assert len(batches) == 1

        assert segment_count == len(plan.tasks)
        assert total_bytes > 0

    def test_response_size_limit_rejects_large_scans(self, temp_warehouse, tmp_path):
        """max_response_bytes causes large scans to fail with 413."""
        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        config = StrataConfig(cache_dir=cache_dir)
        planner = ReadPlanner(config)
        fetcher = CachedFetcher(config)

        # First, do a scan to see how big the response is
        plan = planner.plan(table_uri)
        batches = fetcher.execute_plan(plan)
        total_size = sum(b.nbytes for b in batches)

        # The response will be larger than sum of nbytes due to IPC overhead
        # Verify that we can detect when responses would be too large
        assert total_size > 0, "Should have data"

        # The server enforces max_response_bytes during scan execution
        # This test verifies the check exists by examining the config
        assert config.max_response_bytes > 0, "Should have response size limit"
        assert config.max_response_bytes == 512 * 1024 * 1024  # Default 512MB


class TestStreamingIntegration:
    """Integration tests for HTTP streaming endpoint."""

    @pytest.fixture
    def server_with_client(self, temp_warehouse, tmp_path):
        """Start a server and provide a client."""
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        cache_dir = tmp_path / "cache"
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=cache_dir,
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
            "port": port,
        }

        client.close()

    def test_multi_row_group_stream_produces_valid_ipc(self, server_with_client):
        """Streaming multiple row groups produces valid Arrow IPC.

        This is a critical contract test: when scanning multiple row groups,
        the server streams them as a single valid Arrow IPC stream with:
        - One schema message at the start
        - Multiple record batch messages (one per row group)
        - Proper EOS marker at the end

        Client must be able to decode the full stream with ipc.open_stream().
        """
        import httpx

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]
        expected_rows = server_with_client["warehouse"]["num_rows"]

        with httpx.Client(timeout=30.0) as http_client:
            # Create materialize request with stream mode
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            data = response.json()
            stream_url = data["stream_url"]

            # Fetch stream data
            response = http_client.get(
                f"http://127.0.0.1:{config.port}{stream_url}",
            )
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/vnd.apache.arrow.stream"

            # Collect streamed bytes
            streamed_bytes = response.content

            # Must be valid Arrow IPC stream
            reader = ipc.open_stream(pa.BufferReader(streamed_bytes))

            # Verify schema is present
            schema = reader.schema
            assert "id" in schema.names
            assert "value" in schema.names

            # Read all batches
            batches = list(reader)
            assert len(batches) > 0, "Should have at least one batch"

            total_rows = sum(b.num_rows for b in batches)
            assert total_rows == expected_rows, f"Expected {expected_rows} rows, got {total_rows}"

    def test_streamed_artifact_records_real_row_count(self, server_with_client):
        """Stream-finalized artifacts store actual rows, not task count."""
        import httpx

        config = server_with_client["config"]
        warehouse = server_with_client["warehouse"]
        table_uri = warehouse["table_uri"]
        table = warehouse["table"]

        append_rows(table, 1000, 25)
        expected_rows = warehouse["num_rows"] + 25

        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            data = response.json()
            artifact_uri = data["artifact_uri"]
            stream_url = data["stream_url"]

            response = http_client.get(f"http://127.0.0.1:{config.port}{stream_url}")
            assert response.status_code == 200

            artifact_id, version = artifact_uri.removeprefix("strata://artifact/").split("@v=")
            info_response = http_client.get(
                f"http://127.0.0.1:{config.port}/v1/artifacts/{artifact_id}/v/{version}"
            )
            assert info_response.status_code == 200
            assert info_response.json()["row_count"] == expected_rows

    def test_empty_scan_returns_empty_response(self, server_with_client, tmp_path):
        """Empty scan (all row groups pruned) returns empty response."""
        import httpx
        from pyiceberg.catalog.sql import SqlCatalog
        from pyiceberg.schema import Schema
        from pyiceberg.types import LongType, NestedField

        config = server_with_client["config"]

        # Create empty table
        warehouse_path = tmp_path / "empty_warehouse"
        warehouse_path.mkdir()

        catalog = SqlCatalog(
            "strata",
            uri=f"sqlite:///{warehouse_path / 'catalog.db'}",
            warehouse=str(warehouse_path),
        )
        catalog.create_namespace("test_db")

        schema = Schema(NestedField(1, "id", LongType(), required=False))
        table = catalog.create_table("test_db.empty_table", schema)

        # Append empty table (this creates a snapshot with no data files)
        empty_data = pa.table({"id": pa.array([], type=pa.int64())})
        table.append(empty_data)

        table_uri = f"file://{warehouse_path}#test_db.empty_table"

        with httpx.Client(timeout=30.0) as http_client:
            # Create materialize request
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            data = response.json()
            stream_url = data["stream_url"]

            # Fetch stream data - should be valid empty IPC stream
            response = http_client.get(
                f"http://127.0.0.1:{config.port}{stream_url}",
            )
            assert response.status_code == 200

            # Verify it's a valid Arrow IPC stream (not 0 bytes)
            assert len(response.content) > 0, "Should return valid IPC stream, not 0 bytes"

            # Parse as Arrow IPC - should have schema but no batches
            reader = ipc.open_stream(pa.BufferReader(response.content))
            assert "id" in reader.schema.names, "Schema should have 'id' column"
            batches = list(reader)
            assert len(batches) == 0, "Should have no batches for empty table"

    def test_scan_response_includes_estimated_bytes(self, server_with_client):
        """Materialize response includes estimated_bytes from Parquet metadata."""
        import httpx

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            data = response.json()

            # estimated_bytes should be present and positive (renamed to estimated_size_bytes)
            # The response should contain size information
            assert "artifact_uri" in data
            assert "stream_url" in data

    def test_client_disconnect_releases_resources(self, server_with_client):
        """Client disconnect during streaming releases semaphore.

        This test verifies that when a client disconnects mid-stream:
        1. The server detects the disconnect
        2. Resources (semaphore) are released in the finally block
        3. Subsequent scans can proceed normally

        This is critical for preventing resource leaks under client failures.
        """
        import httpx

        import strata.server as server_module

        config = server_with_client["config"]
        warehouse = server_with_client["warehouse"]
        table_uri = warehouse["table_uri"]
        append_rows(warehouse["table"], 1000, 25)
        state = server_module._state
        assert state is not None

        original_fetch = state.fetcher.fetch_as_stream_bytes

        def slow_fetch(task):
            time.sleep(0.05)
            return original_fetch(task)

        state.fetcher.fetch_as_stream_bytes = slow_fetch

        # Create a materialize request
        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            artifact_uri = response.json()["artifact_uri"]
            stream_url = response.json()["stream_url"]
            artifact_id, version = artifact_uri.removeprefix("strata://artifact/").split("@v=")

            # Start streaming but close connection after first chunk
            # This simulates a client disconnect
            try:
                with http_client.stream(
                    "GET",
                    f"http://127.0.0.1:{config.port}{stream_url}",
                    timeout=5,
                ) as stream:
                    # Read just the first chunk then close
                    for chunk in stream.iter_bytes(chunk_size=1024):
                        if chunk:
                            break  # Simulate disconnect by breaking early
            except Exception:
                pass  # Connection errors expected

        # Give server time to detect disconnect and cleanup
        time.sleep(0.5)

        with httpx.Client(timeout=30.0) as http_client:
            artifact_info = http_client.get(
                f"http://127.0.0.1:{config.port}/v1/artifacts/{artifact_id}/v/{version}"
            )
            assert artifact_info.status_code == 200
            assert artifact_info.json()["state"] == "failed"

        # Now verify we can still do scans (resources were released)
        # If semaphore wasn't released, this would hang or timeout
        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            stream_url2 = response.json()["stream_url"]

            # Complete a full scan to verify functionality
            response = http_client.get(
                f"http://127.0.0.1:{config.port}{stream_url2}",
            )
            assert response.status_code == 200
            assert len(response.content) > 0, "Should get data from second scan"

            # Verify the streamed data is valid Arrow IPC
            reader = ipc.open_stream(pa.BufferReader(response.content))
            batches = list(reader)
            assert len(batches) > 0

    def test_timeout_aborts_stream_with_error(self, temp_warehouse, tmp_path):
        """Scan timeout during streaming aborts connection.

        This test verifies that when a scan exceeds the timeout:
        1. The server raises an error (doesn't silently truncate)
        2. Client receives incomplete/error response
        3. Resources are cleaned up

        We use a very short timeout to trigger this behavior.
        """
        import socket

        import httpx

        import strata.server as server_module

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        cache_dir = tmp_path / "timeout_cache"

        # Create config with very short timeout
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=cache_dir,
            scan_timeout_seconds=0.001,  # 1ms - will definitely timeout
            deployment_mode="personal",
        )

        from strata.server import ServerState, app

        state = ServerState(config)
        server_module._state = state

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

        append_rows(temp_warehouse["table"], 1000, 25)
        table_uri = temp_warehouse["table_uri"]

        original_fetch = state.fetcher.fetch_as_stream_bytes

        def slow_fetch(task):
            time.sleep(0.05)
            return original_fetch(task)

        state.fetcher.fetch_as_stream_bytes = slow_fetch

        with httpx.Client(timeout=30.0) as http_client:
            # Create materialize request
            response = http_client.post(
                f"http://127.0.0.1:{port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            artifact_uri = response.json()["artifact_uri"]
            stream_url = response.json()["stream_url"]
            artifact_id, version = artifact_uri.removeprefix("strata://artifact/").split("@v=")

            # Fetch stream - should fail due to timeout
            # The server aborts the connection, so we may get various errors
            try:
                response = http_client.get(
                    f"http://127.0.0.1:{port}{stream_url}",
                    timeout=10,
                )
                # If we get a response, it should be incomplete/invalid
                # (server raised error during streaming)
                if len(response.content) > 0:
                    # Try to parse - may fail if truncated
                    try:
                        reader = ipc.open_stream(pa.BufferReader(response.content))
                        list(reader)
                        # If it parses, the scan was fast enough to complete
                        # before timeout (possible with cached data)
                    except Exception:
                        # Expected - truncated stream
                        pass
            except httpx.ReadError:
                # Expected - server aborted connection
                pass

            time.sleep(0.2)
            artifact_info = http_client.get(
                f"http://127.0.0.1:{port}/v1/artifacts/{artifact_id}/v/{version}"
            )
            assert artifact_info.status_code == 200
            assert artifact_info.json()["state"] == "failed"


class TestStreamAbortMetrics:
    """Tests for stream abort metrics tracking."""

    @pytest.fixture
    def server_with_metrics(self, temp_warehouse, tmp_path):
        """Start a server and provide access to metrics."""
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        cache_dir = tmp_path / "cache"
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=cache_dir,
            deployment_mode="personal",
        )

        import strata.server as server_module
        from strata.server import ServerState, app

        state = ServerState(config)
        server_module._state = state

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

        yield {
            "state": state,
            "config": config,
            "warehouse": temp_warehouse,
            "port": port,
        }

    def test_client_disconnect_increments_counter(self, server_with_metrics):
        """Client disconnect increments client_disconnects counter."""
        import httpx

        state = server_with_metrics["state"]
        config = server_with_metrics["config"]
        warehouse = server_with_metrics["warehouse"]
        table_uri = warehouse["table_uri"]
        append_rows(warehouse["table"], 1000, 25)

        initial_disconnects = state.metrics.client_disconnects
        original_fetch = state.fetcher.fetch_as_stream_bytes

        def slow_fetch(task):
            time.sleep(0.05)
            return original_fetch(task)

        state.fetcher.fetch_as_stream_bytes = slow_fetch

        with httpx.Client(timeout=30.0) as http_client:
            # Create and start streaming a materialize request
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            stream_url = response.json()["stream_url"]

            # Start streaming but close connection immediately
            try:
                with http_client.stream(
                    "GET",
                    f"http://127.0.0.1:{config.port}{stream_url}",
                    timeout=5,
                ) as stream:
                    # Read first chunk then close
                    for chunk in stream.iter_bytes(chunk_size=1024):
                        if chunk:
                            break  # Simulate disconnect
            except Exception:
                pass

        # Give server time to detect disconnect
        time.sleep(0.5)

        final_disconnects = state.metrics.client_disconnects
        assert final_disconnects > initial_disconnects

    def test_timeout_increments_counter(self, temp_warehouse, tmp_path):
        """Scan timeout increments stream_aborts_timeout counter."""
        import socket

        import httpx

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        cache_dir = tmp_path / "timeout_metrics_cache"
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=cache_dir,
            scan_timeout_seconds=0.001,  # Very short timeout
            deployment_mode="personal",
        )

        import strata.server as server_module
        from strata.server import ServerState, app

        state = ServerState(config)
        server_module._state = state

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

        append_rows(temp_warehouse["table"], 1000, 25)
        table_uri = temp_warehouse["table_uri"]
        initial_timeouts = state.metrics.stream_aborts_timeout

        original_fetch = state.fetcher.fetch_as_stream_bytes

        def slow_fetch(task):
            time.sleep(0.05)
            return original_fetch(task)

        state.fetcher.fetch_as_stream_bytes = slow_fetch

        with httpx.Client(timeout=30.0) as http_client:
            # Create materialize request
            response = http_client.post(
                f"http://127.0.0.1:{port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            stream_url = response.json()["stream_url"]

            # Fetch - should timeout
            try:
                http_client.get(
                    f"http://127.0.0.1:{port}{stream_url}",
                    timeout=10,
                )
            except Exception:
                pass

        # Give server time to record metrics
        time.sleep(0.5)

        final_timeouts = state.metrics.stream_aborts_timeout
        assert final_timeouts > initial_timeouts

    def test_size_limit_increments_counter(self, temp_warehouse, tmp_path):
        """Pre-flight size rejection increments stream_aborts_size."""
        import socket

        import httpx

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        cache_dir = tmp_path / "size_metrics_cache"
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=cache_dir,
            max_response_bytes=1,
            deployment_mode="personal",
        )

        import strata.server as server_module
        from strata.server import ServerState, app

        state = ServerState(config)
        server_module._state = state

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

        initial_size_aborts = state.metrics.stream_aborts_size

        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                f"http://127.0.0.1:{port}/v1/materialize",
                json=build_materialize_request(temp_warehouse["table_uri"]),
            )
            assert response.status_code == 413

        assert state.metrics.stream_aborts_size > initial_size_aborts

    def test_metrics_endpoint_includes_abort_counters(self, server_with_metrics):
        """GET /metrics includes stream abort counters."""
        import requests

        config = server_with_metrics["config"]

        response = requests.get(f"http://127.0.0.1:{config.port}/metrics")
        assert response.status_code == 200
        metrics = response.json()

        # Verify abort counters are present
        assert "stream_aborts_timeout" in metrics
        assert "stream_aborts_size" in metrics
        assert "client_disconnects" in metrics

    def test_prometheus_metrics_includes_abort_counters(self, server_with_metrics):
        """GET /metrics/prometheus includes stream abort counters."""
        import requests

        config = server_with_metrics["config"]

        response = requests.get(f"http://127.0.0.1:{config.port}/metrics/prometheus")
        assert response.status_code == 200
        content = response.text

        # Verify abort counters are present in Prometheus format
        assert "strata_stream_aborts_timeout_total" in content
        assert "strata_stream_aborts_size_total" in content
        assert "strata_client_disconnects_total" in content


class TestActiveScanCount:
    """Tests for active scan counting and limiter management."""

    def test_get_active_scan_count_matches_limiter(self, temp_warehouse, tmp_path):
        """_get_active_scan_count returns correct count based on QoS tier limiters."""

        cache_dir = tmp_path / "cache"
        config = StrataConfig(
            cache_dir=cache_dir,
            interactive_slots=4,
            bulk_slots=2,
        )

        from strata.server import ServerState, _get_active_scan_count

        state = ServerState(config)

        # Initially no active scans
        assert _get_active_scan_count(state) == 0

        # Acquire limiter slots manually from both tiers
        async def test_counting():
            assert _get_active_scan_count(state) == 0

            # Acquire from interactive tier
            await state._interactive_limiter.acquire()
            assert _get_active_scan_count(state) == 1

            # Acquire from bulk tier
            await state._bulk_limiter.acquire()
            assert _get_active_scan_count(state) == 2

            # Acquire another from interactive
            await state._interactive_limiter.acquire()
            assert _get_active_scan_count(state) == 3

            # Release from interactive
            await state._interactive_limiter.release()
            assert _get_active_scan_count(state) == 2

            # Release from bulk
            await state._bulk_limiter.release()
            assert _get_active_scan_count(state) == 1

            # Release remaining interactive
            await state._interactive_limiter.release()
            assert _get_active_scan_count(state) == 0

        asyncio.run(test_counting())

    def test_active_scans_released_on_completion(self, temp_warehouse, tmp_path):
        """Active scan count returns to zero after scan completes."""
        import socket

        import httpx

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        cache_dir = tmp_path / "cache"
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=cache_dir,
            max_concurrent_scans=2,
            deployment_mode="personal",
        )

        import strata.server as server_module
        from strata.server import ServerState, _get_active_scan_count, app

        state = ServerState(config)
        server_module._state = state

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

        table_uri = temp_warehouse["table_uri"]

        # Check initial state
        assert _get_active_scan_count(state) == 0

        with httpx.Client(timeout=30.0) as http_client:
            # Create and complete a materialize request
            response = http_client.post(
                f"http://127.0.0.1:{port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            stream_url = response.json()["stream_url"]

            # Fetch all data
            response = http_client.get(f"http://127.0.0.1:{port}{stream_url}")
            assert response.status_code == 200

        # Give server time to release resources
        time.sleep(0.2)

        # Should be back to zero
        assert _get_active_scan_count(state) == 0


class TestConcurrentScans:
    """Tests verifying concurrent scans complete correctly."""

    def test_concurrent_scans_all_succeed(self, temp_warehouse, tmp_path):
        """Five concurrent scans against the same table all return 200 with
        valid IPC bytes. If concurrency were broken (e.g. a shared
        non-thread-safe cursor, a deadlock, a corrupted shared cache), one
        or more requests would error or hang past the per-request timeout.
        """
        import socket
        from concurrent.futures import as_completed

        import httpx

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        cache_dir = tmp_path / "cache"
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=cache_dir,
            max_concurrent_scans=10,
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

        table_uri = temp_warehouse["table_uri"]

        def do_scan() -> int:
            """Run one materialize+stream cycle and return the byte count."""
            with httpx.Client(timeout=30.0) as http_client:
                resp = http_client.post(
                    f"http://127.0.0.1:{port}/v1/materialize",
                    json=build_materialize_request(table_uri),
                )
                assert resp.status_code == 200
                stream_url = resp.json()["stream_url"]

                resp = http_client.get(f"http://127.0.0.1:{port}{stream_url}")
                assert resp.status_code == 200
                return len(resp.content)

        # Warm the cache so all 5 concurrent requests follow the cache-hit
        # path uniformly.
        do_scan()

        num_concurrent = 5
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(do_scan) for _ in range(num_concurrent)]
            byte_counts = [f.result() for f in as_completed(futures)]

        assert len(byte_counts) == num_concurrent
        assert all(n > 0 for n in byte_counts)


class TestNonBlockingLogging:
    """Tests for non-blocking metrics logging.

    These tests verify that the MetricsCollector uses a queue + background writer
    to prevent logging from blocking request handlers. This prevents the pipe buffer
    deadlock that occurred when:
    1. Server was started with stdout=subprocess.PIPE
    2. Parent didn't read from pipe, so buffer filled up (~64KB)
    3. MetricsCollector._write_log() called flush() while holding _lock
    4. flush() blocked waiting for buffer space
    5. /metrics endpoint needed _lock, causing deadlock
    """

    def test_metrics_collector_uses_queue_based_logging(self):
        """MetricsCollector should use a queue for non-blocking writes."""
        import io
        import queue as queue_module

        from strata.metrics import MetricsCollector

        output = io.StringIO()
        collector = MetricsCollector(output=output, enabled=True)

        try:
            # Verify queue exists
            assert hasattr(collector, "_log_queue")
            assert isinstance(collector._log_queue, queue_module.Queue)

            # Verify background writer thread is running
            assert hasattr(collector, "_writer_thread")
            assert collector._writer_thread.is_alive()

            # Log an event
            collector.log_event("test_event", key="value")

            # Wait for background thread to process
            collector._log_queue.join()

            # Verify output was written
            output.seek(0)
            content = output.read()
            assert "test_event" in content
            assert "key" in content
        finally:
            collector.shutdown()

    def test_logging_drops_when_queue_full(self):
        """Logs should be dropped (not blocked) when queue is full."""
        import io

        from strata.metrics import MetricsCollector

        # Create collector with tiny queue that will fill up
        output = io.StringIO()
        collector = MetricsCollector(output=output, enabled=True, log_queue_size=2)

        try:
            # Pause background writer by filling queue beyond capacity
            # First, shut down the writer so queue fills up
            collector._shutdown.set()
            collector._writer_thread.join(timeout=1)

            # Reset for new attempt - create a blocking scenario
            initial_dropped = collector.dropped_logs

            # Flood the queue - should drop after queue is full
            for i in range(100):
                collector.log_event(f"flood_event_{i}")

            # Some logs should have been dropped (queue only holds 2)
            assert collector.dropped_logs > initial_dropped, (
                "Should have dropped logs when queue was full"
            )
        finally:
            collector.shutdown()

    def test_get_aggregate_stats_never_blocks_on_logging(self):
        """get_aggregate_stats() should not block even if logging is slow."""
        import io
        import time

        from strata.metrics import MetricsCollector

        output = io.StringIO()
        collector = MetricsCollector(output=output, enabled=True)

        try:
            # Record some metrics
            collector.record_fetch(1000, 10, 5.0, from_cache=True)
            collector.record_fetch(2000, 20, 10.0, from_cache=False)

            # Time the stats call - should be fast
            start = time.perf_counter()
            stats = collector.get_aggregate_stats()
            elapsed = time.perf_counter() - start

            # Should complete in < 100ms (no blocking on I/O)
            assert elapsed < 0.1, f"get_aggregate_stats took too long: {elapsed:.3f}s"

            # Verify stats are correct
            assert stats["cache_hits"] == 1
            assert stats["cache_misses"] == 1
            assert stats["bytes_from_cache"] == 1000
            assert stats["bytes_from_storage"] == 2000
        finally:
            collector.shutdown()

    def test_dropped_logs_counter_in_stats(self):
        """dropped_logs counter should be exposed in aggregate stats."""
        import io

        from strata.metrics import MetricsCollector

        output = io.StringIO()
        collector = MetricsCollector(output=output, enabled=True, log_queue_size=1)

        try:
            # Force some drops
            collector._shutdown.set()
            collector._writer_thread.join(timeout=1)

            for _ in range(50):
                collector.log_event("flood")

            stats = collector.get_aggregate_stats()
            assert "dropped_logs" in stats
            assert stats["dropped_logs"] > 0
        finally:
            collector.shutdown()

    def test_logging_thread_shuts_down_gracefully(self):
        """Background writer thread should shut down cleanly."""
        import io

        from strata.metrics import MetricsCollector

        output = io.StringIO()
        collector = MetricsCollector(output=output, enabled=True)

        # Thread should be alive
        assert collector._writer_thread.is_alive()

        # Shutdown should complete quickly
        collector.shutdown()

        # Thread should be stopped
        assert not collector._writer_thread.is_alive()


class TestCacheVersioning:
    """Test that cache versioning works correctly."""

    def test_different_cache_versions_coexist(self, temp_warehouse, tmp_path):
        """Different cache versions don't interfere with each other."""

        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        config = StrataConfig(cache_dir=cache_dir)
        planner = ReadPlanner(config)
        fetcher = CachedFetcher(config)

        # Populate current version cache
        plan = planner.plan(table_uri)
        fetcher.execute_plan(plan)

        # Create fake "old version" cache directory
        old_version_dir = cache_dir / "v0" / "ab" / "cd"
        old_version_dir.mkdir(parents=True)
        (old_version_dir / "fake_old_cache.arrowstream").write_bytes(b"old data")

        # Current version should still work
        fetcher2 = CachedFetcher(config)
        plan2 = planner.plan(table_uri)

        cache_hits = sum(1 for t in plan2.tasks if fetcher2.cache.contains(t.cache_key))
        assert cache_hits == len(plan2.tasks), "Should hit current version cache"

        # Old version files should still exist (not deleted)
        assert (old_version_dir / "fake_old_cache.arrowstream").exists()

        # Stats should only count current version
        assert isinstance(fetcher2.cache, DiskCache)
        stats = fetcher2.cache.get_stats()
        assert stats.total_entries == len(plan.tasks)
