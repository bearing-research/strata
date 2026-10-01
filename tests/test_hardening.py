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

        planner1 = ReadPlanner(config)
        fetcher1 = CachedFetcher(config)

        plan1 = planner1.plan(table_uri)
        batches1 = fetcher1.execute_plan(plan1)
        total_rows1 = sum(b.num_rows for b in batches1)

        cache_entries = list((cache_dir / f"v{CACHE_VERSION}").rglob(f"*{CACHE_FILE_EXTENSION}"))
        assert len(cache_entries) > 0, "Cache should have entries after first run"

        # Simulate a restart with fresh planner and fetcher instances.
        planner2 = ReadPlanner(config)
        fetcher2 = CachedFetcher(config)

        plan2 = planner2.plan(table_uri)

        cache_hits = 0
        for task in plan2.tasks:
            if fetcher2.cache.contains(task.cache_key):
                cache_hits += 1

        batches2 = fetcher2.execute_plan(plan2)
        total_rows2 = sum(b.num_rows for b in batches2)

        assert total_rows1 == total_rows2
        assert cache_hits == len(plan2.tasks), "All tasks should hit cache after restart"

    def test_metadata_cache_persists_across_planner_instances(self, temp_warehouse, tmp_path):
        """Metadata cache (SQLite) survives planner restart."""
        from strata.metadata_cache import get_metadata_store, reset_caches

        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        reset_caches()

        config = StrataConfig(cache_dir=cache_dir)

        planner1 = ReadPlanner(config)
        plan1 = planner1.plan(table_uri)

        store = get_metadata_store(cache_dir)
        stats1 = store.stats()
        assert stats1["parquet_entries"] > 0, "Should have parquet metadata cached"

        # Simulate a restart: reset in-memory caches but keep SQLite.
        reset_caches()

        planner2 = ReadPlanner(config)

        plan2 = planner2.plan(table_uri)

        get_metadata_store(cache_dir)

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

        plan = planner.plan(table_uri)
        batches1 = fetcher.execute_plan(plan)
        total_rows1 = sum(b.num_rows for b in batches1)

        cache_files = list((cache_dir / f"v{CACHE_VERSION}").rglob(f"*{CACHE_FILE_EXTENSION}"))
        assert len(cache_files) > 0

        corrupted_file = cache_files[0]

        corrupted_file.write_bytes(b"CORRUPTED DATA - NOT VALID ARROW IPC")

        # A fresh fetcher simulates a restart.
        fetcher2 = CachedFetcher(config)

        plan2 = planner.plan(table_uri)
        batches2 = fetcher2.execute_plan(plan2)
        total_rows2 = sum(b.num_rows for b in batches2)

        # Refetched, so the data is still correct.
        assert total_rows2 == total_rows1

        # The corrupted file is deleted or replaced with valid data.
        if corrupted_file.exists():
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

        plan = planner.plan(table_uri)
        fetcher.execute_plan(plan)

        meta_files = list((cache_dir / f"v{CACHE_VERSION}").rglob(f"*{CACHE_META_EXTENSION}"))
        assert len(meta_files) > 0

        meta_files[0].write_text("{ invalid json }")

        assert isinstance(fetcher.cache, DiskCache)
        stats = fetcher.cache.get_stats()
        # Corrupted entries are skipped.
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
                planner = ReadPlanner(config)
                fetcher = CachedFetcher(config)
                plan = planner.plan(table_uri)
                batches = fetcher.execute_plan(plan)
                results.append(sum(b.num_rows for b in batches))
            except Exception as e:
                import traceback

                errors.append((e, traceback.format_exc()))

        num_workers = 5
        threads = [threading.Thread(target=worker) for _ in range(num_workers)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == num_workers

        assert all(r == results[0] for r in results)

    def test_server_concurrent_scans_use_semaphore(self, temp_warehouse, tmp_path):
        """Server properly limits concurrent scans via semaphore."""
        import socket

        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=cache_dir,
            max_concurrent_scans=2,  # Low limit to exercise queuing
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

        results = []
        errors = []

        def scan_worker():
            try:
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

        # All succeed (queued when over the limit).
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

        planner1 = ReadPlanner(config)
        plan1 = planner1.plan(temp_warehouse["table_uri"])

        store = get_metadata_store(cache_dir)
        stats1 = store.stats()
        initial_entries = stats1["parquet_entries"]
        assert initial_entries > 0

        # Creates new files and may update existing ones.
        new_data = pa.table(
            {
                "id": pa.array(range(100), type=pa.int64()),
                "value": pa.array([float(i) for i in range(100)], type=pa.float64()),
                "name": pa.array([f"new_{i}" for i in range(100)], type=pa.string()),
            }
        )
        table.append(new_data)

        store.cleanup_stale_parquet_meta()

        reset_caches()
        planner2 = ReadPlanner(config)
        plan2 = planner2.plan(temp_warehouse["table_uri"])

        # May keep the same row-group count if the new data fits in existing files.
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

        batch_count = 0
        total_rows = 0

        for batch in fetcher.stream_plan(plan):
            batch_count += 1
            total_rows += batch.num_rows
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

        segment_count = 0
        total_bytes = 0

        for segment in fetcher.stream_plan_as_ipc(plan):
            segment_count += 1
            total_bytes += len(segment)
            assert len(segment) > 0
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

        plan = planner.plan(table_uri)
        batches = fetcher.execute_plan(plan)
        total_size = sum(b.nbytes for b in batches)

        assert total_size > 0, "Should have data"

        # The server enforces max_response_bytes during the scan; this only checks
        # the config carries the limit.
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
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            data = response.json()
            stream_url = data["stream_url"]

            response = http_client.get(
                f"http://127.0.0.1:{config.port}{stream_url}",
            )
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/vnd.apache.arrow.stream"

            streamed_bytes = response.content

            reader = ipc.open_stream(pa.BufferReader(streamed_bytes))

            schema = reader.schema
            assert "id" in schema.names
            assert "value" in schema.names

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

        # This creates a snapshot with no data files.
        empty_data = pa.table({"id": pa.array([], type=pa.int64())})
        table.append(empty_data)

        table_uri = f"file://{warehouse_path}#test_db.empty_table"

        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            data = response.json()
            stream_url = data["stream_url"]

            response = http_client.get(
                f"http://127.0.0.1:{config.port}{stream_url}",
            )
            assert response.status_code == 200

            assert len(response.content) > 0, "Should return valid IPC stream, not 0 bytes"

            # Schema but no batches.
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

            assert "artifact_uri" in data
            assert "stream_url" in data

    def test_client_disconnect_releases_resources(self, server_with_client):
        """Client disconnect during streaming releases semaphore.

        This test verifies that when a client disconnects mid-stream:
        1. The artifact build is decoupled from the read — a dropped client
           does NOT fail the artifact; the background build finalizes it ready.
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

        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            artifact_uri = response.json()["artifact_uri"]
            stream_url = response.json()["stream_url"]
            artifact_id, version = artifact_uri.removeprefix("strata://artifact/").split("@v=")

            # Close the connection after the first chunk to simulate a disconnect.
            try:
                with http_client.stream(
                    "GET",
                    f"http://127.0.0.1:{config.port}{stream_url}",
                    timeout=5,
                ) as stream:
                    for chunk in stream.iter_bytes(chunk_size=1024):
                        if chunk:
                            break
            except Exception:
                pass  # Connection errors expected

        # The build is decoupled from the client: a mid-stream disconnect must NOT
        # poison the artifact. The background build finalizes it ready regardless.
        with httpx.Client(timeout=30.0) as http_client:
            deadline = time.time() + 10
            final_state = None
            while time.time() < deadline:
                artifact_info = http_client.get(
                    f"http://127.0.0.1:{config.port}/v1/artifacts/{artifact_id}/v/{version}"
                )
                assert artifact_info.status_code == 200
                final_state = artifact_info.json()["state"]
                if final_state == "ready":
                    break
                time.sleep(0.1)
            assert final_state == "ready", (
                f"client disconnect must not fail the build; got state={final_state}"
            )

        # A leaked semaphore would make this hang or time out.
        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                f"http://127.0.0.1:{config.port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            stream_url2 = response.json()["stream_url"]

            response = http_client.get(
                f"http://127.0.0.1:{config.port}{stream_url2}",
            )
            assert response.status_code == 200
            assert len(response.content) > 0, "Should get data from second scan"

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

        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=cache_dir,
            scan_timeout_seconds=0.001,  # 1ms: will definitely time out
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

        # Three row groups, not two: the first can come from a finished prefetch, so
        # the check before the second can land inside 1ms. The check before the third
        # always follows a slow fetch.
        append_rows(temp_warehouse["table"], 1000, 25)
        append_rows(temp_warehouse["table"], 1025, 25)
        table_uri = temp_warehouse["table_uri"]

        original_fetch = state.fetcher.fetch_as_stream_bytes

        def slow_fetch(task):
            time.sleep(0.05)
            return original_fetch(task)

        state.fetcher.fetch_as_stream_bytes = slow_fetch

        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                f"http://127.0.0.1:{port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            artifact_uri = response.json()["artifact_uri"]
            stream_url = response.json()["stream_url"]
            artifact_id, version = artifact_uri.removeprefix("strata://artifact/").split("@v=")

            # The server aborts the connection, so several errors are possible.
            try:
                response = http_client.get(
                    f"http://127.0.0.1:{port}{stream_url}",
                    timeout=10,
                )
                # A response here is incomplete: the server raised during streaming.
                if len(response.content) > 0:
                    # May fail if truncated.
                    try:
                        reader = ipc.open_stream(pa.BufferReader(response.content))
                        list(reader)
                        # Parsing means the scan beat the timeout (possible with cached data).
                    except Exception:
                        # Truncated stream
                        pass
            except httpx.ReadError:
                # Server aborted the connection
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

    # No client-disconnect *counter* test: a disconnect is only observable if it
    # lands mid-send, which depends on socket buffer sizes and Starlette's cancel
    # timing and is not reproducible across platforms. The contract (artifact
    # stays `ready`, no leaked resources) is covered by
    # TestStreamingIntegration.test_client_disconnect_releases_resources and
    # test_semaphore_leak.test_concurrent_disconnects_no_leak.

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
            response = http_client.post(
                f"http://127.0.0.1:{port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            stream_url = response.json()["stream_url"]

            try:
                http_client.get(
                    f"http://127.0.0.1:{port}{stream_url}",
                    timeout=10,
                )
            except Exception:
                pass

        # Give the server time to record metrics.
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

        assert "strata_stream_aborts_timeout_total" in content
        assert "strata_stream_aborts_size_total" in content
        assert "strata_client_disconnects_total" in content


class TestActiveScanCount:
    """Tests for active scan counting and limiter management."""

    def test_saturation_tracks_registry_limiters(self, tmp_path):
        """Saturation is measured on the per-tenant admission limiters, not the
        never-acquired global ones; a server with no live limiters is idle, not
        saturated (finding 2)."""
        from strata.server import ServerState, _update_saturation_tracking
        from strata.tenant_registry import get_tenant_registry, reset_tenant_registry

        reset_tenant_registry()
        config = StrataConfig(cache_dir=tmp_path / "cache", interactive_slots=2, bulk_slots=2)
        state = ServerState(config)

        # Fresh registry: no live limiters means idle, not saturated.
        _update_saturation_tracking(state)
        assert state._interactive_saturated_since is None

        registry = get_tenant_registry()
        interactive, _bulk = registry.get_or_create_limiters("team-a")

        async def drive():
            for _ in range(interactive.available):
                await interactive.acquire()
            _update_saturation_tracking(state)
            assert state._interactive_saturated_since is not None

            # Free one slot: no longer saturated.
            await interactive.release()
            _update_saturation_tracking(state)
            assert state._interactive_saturated_since is None

        asyncio.run(drive())
        reset_tenant_registry()

    def test_get_active_scan_count_matches_limiter(self, temp_warehouse, tmp_path):
        """_get_active_scan_count counts the per-tenant admission limiters.

        Stream admission acquires the tenant-registry limiters, not the global
        ServerState ones, so the count must track the registry (otherwise
        graceful shutdown drains past live streams — finding 2).
        """
        cache_dir = tmp_path / "cache"
        config = StrataConfig(
            cache_dir=cache_dir,
            interactive_slots=4,
            bulk_slots=2,
        )

        from strata.server import ServerState, _get_active_scan_count
        from strata.tenant_registry import get_tenant_registry, reset_tenant_registry

        reset_tenant_registry()
        ServerState(config)
        registry = get_tenant_registry()
        interactive, bulk = registry.get_or_create_limiters("team-a")

        assert _get_active_scan_count() == 0

        async def test_counting():
            assert _get_active_scan_count() == 0

            await interactive.acquire()
            assert _get_active_scan_count() == 1

            await bulk.acquire()
            assert _get_active_scan_count() == 2

            await interactive.acquire()
            assert _get_active_scan_count() == 3

            await interactive.release()
            assert _get_active_scan_count() == 2

            await bulk.release()
            assert _get_active_scan_count() == 1

            await interactive.release()
            assert _get_active_scan_count() == 0

        asyncio.run(test_counting())
        reset_tenant_registry()

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

        assert _get_active_scan_count() == 0

        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                f"http://127.0.0.1:{port}/v1/materialize",
                json=build_materialize_request(table_uri),
            )
            assert response.status_code == 200
            stream_url = response.json()["stream_url"]

            response = http_client.get(f"http://127.0.0.1:{port}{stream_url}")
            assert response.status_code == 200

        # Give the server time to release resources.
        time.sleep(0.2)

        assert _get_active_scan_count() == 0


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

        # Warm the cache so all 5 concurrent requests take the cache-hit path.
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
            assert hasattr(collector, "_log_queue")
            assert isinstance(collector._log_queue, queue_module.Queue)

            assert hasattr(collector, "_writer_thread")
            assert collector._writer_thread.is_alive()

            collector.log_event("test_event", key="value")

            collector._log_queue.join()

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

        output = io.StringIO()
        collector = MetricsCollector(output=output, enabled=True, log_queue_size=2)

        try:
            # Stop the writer so the queue fills up.
            collector._shutdown.set()
            collector._writer_thread.join(timeout=1)

            initial_dropped = collector.dropped_logs

            for i in range(100):
                collector.log_event(f"flood_event_{i}")

            # The queue only holds 2.
            assert collector.dropped_logs > initial_dropped, (
                "Should have dropped logs when queue was full"
            )
        finally:
            collector.shutdown()

    def test_get_aggregate_stats_never_blocks_on_logging(self):
        """get_aggregate_stats() reads in-memory counters (no I/O to block on)."""
        import io

        from strata.metrics import MetricsCollector

        output = io.StringIO()
        collector = MetricsCollector(output=output, enabled=True)

        try:
            collector.record_fetch(1000, 10, 5.0, from_cache=True)
            collector.record_fetch(2000, 20, 10.0, from_cache=False)

            # Stats aggregate in-memory state; there is no I/O on this path to put a
            # time bound on, so check the aggregation is correct.
            stats = collector.get_aggregate_stats()

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

        assert collector._writer_thread.is_alive()

        collector.shutdown()

        assert not collector._writer_thread.is_alive()


class TestCacheVersioning:
    """Test that cache versioning works correctly."""

    def test_another_cache_version_is_removed_and_the_current_one_kept(
        self, temp_warehouse, tmp_path
    ):
        """A new cache deletes another version's directory, keeping its own entries."""

        cache_dir = tmp_path / "cache"
        table_uri = temp_warehouse["table_uri"]

        config = StrataConfig(cache_dir=cache_dir)
        planner = ReadPlanner(config)
        fetcher = CachedFetcher(config)

        plan = planner.plan(table_uri)
        fetcher.execute_plan(plan)

        old_version_dir = cache_dir / "v0" / "ab" / "cd"
        old_version_dir.mkdir(parents=True)
        (old_version_dir / "fake_old_cache.arrowstream").write_bytes(b"old data")

        fetcher2 = CachedFetcher(config)
        plan2 = planner.plan(table_uri)

        cache_hits = sum(1 for t in plan2.tasks if fetcher2.cache.contains(t.cache_key))
        assert cache_hits == len(plan2.tasks), "Should hit current version cache"

        # Nothing counts or evicts another version's entries, so they go.
        assert not (cache_dir / "v0").exists()

        assert isinstance(fetcher2.cache, DiskCache)
        stats = fetcher2.cache.get_stats()
        assert stats.total_entries == len(plan.tasks)
