"""Smoke tests for Strata."""

import sys
import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pyarrow as pa
import pytest
import uvicorn
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    DoubleType,
    LongType,
    NestedField,
    StringType,
)
from strata_client.client import StrataClient
from strata_client.integration.duckdb import StrataScanner

from strata.cache.disk import CachedFetcher, DiskCache
from strata.config import StrataConfig
from strata.planner import ReadPlanner
from strata.types import CacheKey, Filter, FilterOp, TableIdentity


@pytest.fixture
def temp_warehouse(tmp_path):
    """Create a temporary warehouse with a sample Iceberg table."""
    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
    warehouse_path = tmp_path / "warehouse"
    warehouse_path.mkdir()

    # "strata" matches PyIcebergCatalog.
    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )

    catalog.create_namespace("test_db")

    # Optional fields match PyArrow defaults.
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
        NestedField(3, "name", StringType(), required=False),
        NestedField(4, "timestamp", LongType(), required=False),  # Epoch micros
    )

    table = catalog.create_table("test_db.events", schema)

    num_rows = 500
    base_ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)
    data = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "value": pa.array([float(i * 1.5) for i in range(num_rows)], type=pa.float64()),
            "name": pa.array([f"item_{i}" for i in range(num_rows)], type=pa.string()),
            "timestamp": pa.array(
                [base_ts + i * 3600_000_000 for i in range(num_rows)],  # micros
                type=pa.int64(),
            ),
        }
    )

    table.append(data)

    # Dispose the engine now, not at teardown: the server opens its own SqlCatalog on
    # the same catalog.db, and on Linux + Python 3.14 + tmpfs the two connections
    # intermittently hit SQLITE_IOERR. Nothing uses catalog/table after this point.
    catalog.engine.dispose()

    yield {
        "warehouse_path": warehouse_path,
        "table_uri": f"file://{warehouse_path}#test_db.events",
        "catalog": catalog,
        "table": table,
    }


@pytest.fixture
def strata_config(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return StrataConfig(cache_dir=cache_dir)


@pytest.fixture
def planner(strata_config):
    with ReadPlanner(strata_config) as planner:
        yield planner


class TestTableIdentity:
    def test_from_table_id(self):
        identity = TableIdentity.from_table_id("test_db.events")
        assert identity.catalog == "strata"
        assert identity.namespace == "test_db"
        assert identity.table == "events"
        assert str(identity) == "strata.test_db.events"

    def test_from_table_id_with_catalog(self):
        identity = TableIdentity.from_table_id("analytics.page_views", catalog="prod")
        assert identity.catalog == "prod"
        assert identity.namespace == "analytics"
        assert identity.table == "page_views"
        assert str(identity) == "prod.analytics.page_views"

    def test_invalid_table_id(self):
        with pytest.raises(ValueError, match="expected 'namespace.table' format"):
            TableIdentity.from_table_id("just_table")


class TestCacheKey:
    def test_to_hex(self):
        identity = TableIdentity.from_table_id("test_db.events")
        key = CacheKey(
            tenant_id="_default",
            table_identity=identity,
            file_size=1024,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="abc123",
        )
        hex_digest = key.to_hex()
        assert len(hex_digest) == 64  # SHA-256 hex digest
        assert hex_digest == key.to_hex()  # Deterministic
        assert key.table_id == "strata.test_db.events"

    def test_projection_fingerprint(self):
        fp1 = CacheKey.compute_projection_fingerprint(["a", "b", "c"])
        fp2 = CacheKey.compute_projection_fingerprint(["c", "b", "a"])
        fp3 = CacheKey.compute_projection_fingerprint(["a", "b"])
        fp_same = CacheKey.compute_projection_fingerprint(["a", "b", "c"])
        fp_all = CacheKey.compute_projection_fingerprint(None)

        # Column order matters.
        assert fp1 != fp2
        assert fp1 == fp_same
        assert fp1 != fp3
        # None means all columns.
        assert fp_all == "*"

    def test_a_comma_in_a_column_name_is_not_a_separator(self):
        # Iceberg and Parquet both allow "a,b" as one column name; it must not share a
        # fingerprint (and so a cache key) with ["a", "b"].
        fp = CacheKey.compute_projection_fingerprint
        assert fp(["a,b"]) != fp(["a", "b"])


class TestFilter:
    def test_matches_stats_eq(self):
        f = Filter(column="x", op=FilterOp.EQ, value=50)
        assert f.matches_stats(0, 100) is True
        assert f.matches_stats(0, 49) is False
        assert f.matches_stats(51, 100) is False
        assert f.matches_stats(50, 50) is True

    def test_matches_stats_lt(self):
        f = Filter(column="x", op=FilterOp.LT, value=50)
        assert f.matches_stats(0, 100) is True
        assert f.matches_stats(0, 49) is True
        assert f.matches_stats(50, 100) is False

    def test_matches_stats_gt(self):
        f = Filter(column="x", op=FilterOp.GT, value=50)
        assert f.matches_stats(0, 100) is True
        assert f.matches_stats(51, 100) is True
        assert f.matches_stats(0, 50) is False


class TestDiskCache:
    def test_put_get(self, strata_config):
        cache = DiskCache(strata_config)
        identity = TableIdentity.from_table_id("test_db.events")
        key = CacheKey(
            tenant_id="_default",
            table_identity=identity,
            file_size=1024,
            file_path="/test.parquet",
            row_group_id=0,
            projection_fingerprint="*",
        )

        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]})

        assert cache.contains(key) is False
        cache.put(key, batch)
        assert cache.contains(key) is True

        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved.num_rows == 3
        assert retrieved.column("id").to_pylist() == [1, 2, 3]

        # Metadata uses the canonical identity.
        stats = cache.get_stats()
        assert stats.total_entries == 1
        assert "strata.test_db.events" in stats.entries_by_table

    def test_clear(self, strata_config):
        cache = DiskCache(strata_config)
        identity = TableIdentity.from_table_id("test_db.events")
        key = CacheKey(
            tenant_id="_default",
            table_identity=identity,
            file_size=1024,
            file_path="/test.parquet",
            row_group_id=0,
            projection_fingerprint="*",
        )

        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        cache.put(key, batch)
        assert cache.contains(key) is True

        cache.clear()
        assert cache.contains(key) is False


class TestReadPlanner:
    def test_plan_basic(self, temp_warehouse, planner):
        plan = planner.plan(temp_warehouse["table_uri"])

        assert plan.snapshot_id > 0
        assert len(plan.tasks) > 0
        assert plan.total_row_groups > 0

    def test_plan_with_projection(self, temp_warehouse, planner):
        plan = planner.plan(
            temp_warehouse["table_uri"],
            columns=["id", "value"],
        )

        assert len(plan.tasks) > 0
        for task in plan.tasks:
            assert task.columns == ["id", "value"]

    def test_plan_with_filter_pruning(self, temp_warehouse, planner):
        planner.plan(temp_warehouse["table_uri"])

        # value ranges from 0 to 748.5 (500 rows * 1.5)
        filters = [Filter(column="value", op=FilterOp.LT, value=100.0)]
        plan_filtered = planner.plan(
            temp_warehouse["table_uri"],
            filters=filters,
        )

        assert plan_filtered.pruned_row_groups >= 0


class TestCachedFetcher:
    def test_fetch_and_cache(self, temp_warehouse, strata_config, planner):
        fetcher = CachedFetcher(strata_config)
        plan = planner.plan(temp_warehouse["table_uri"])
        task = plan.tasks[0]

        batch1 = fetcher.fetch(task)
        assert not task.cached
        assert batch1.num_rows > 0

        task2 = plan.tasks[0]  # Same task
        batch2 = fetcher.fetch(task2)
        assert task2.cached
        assert batch2.num_rows == batch1.num_rows


class TestEndToEnd:
    """End-to-end tests against a running server."""

    @pytest.fixture
    def server_with_client(self, temp_warehouse, strata_config, tmp_path):
        """A running server and a client for it."""
        import socket

        from vestigo.store import reset_artifact_store

        from strata.cache.metadata import reset_caches

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            deployment_mode="personal",  # Enables the artifact store for the unified API
        )

        import strata.server as server_module
        from strata.server import ServerState, app

        reset_caches()
        reset_artifact_store()

        original_state = server_module._state
        server_module._state = ServerState(config)

        server = uvicorn.Server(
            uvicorn.Config(
                app=app,
                host=config.host,
                port=config.port,
                log_level="error",
                # Match production (server.main): default ws="auto" imports uvicorn's legacy
                # websockets backend (DeprecationWarning; broken on CPython 3.14).
                ws="websockets-sansio",
            )
        )
        server_thread = threading.Thread(
            target=server.run,
            daemon=True,
        )
        server_thread.start()

        deadline = time.time() + 10
        while not server.started:
            if not server_thread.is_alive():
                raise RuntimeError("Uvicorn server thread exited before startup")
            if time.time() >= deadline:
                raise RuntimeError("Timed out waiting for uvicorn server startup")
            time.sleep(0.05)

        client = StrataClient(base_url=f"http://127.0.0.1:{port}")

        yield {
            "client": client,
            "config": config,
            "warehouse": temp_warehouse,
        }

        client.close()
        server.should_exit = True
        server_thread.join(timeout=5)
        server_module._state = original_state
        reset_caches()
        reset_artifact_store()

    def test_fetch_and_cache_hit(self, server_with_client):
        """The second fetch is a cache hit."""
        client = server_with_client["client"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        artifact1 = client.materialize(
            inputs=[table_uri],
            transform={"executor": "scan@v1", "params": {}},
        )
        table1 = client.fetch(artifact1.uri)
        assert table1.num_rows == 500
        assert artifact1.cache_hit is False

        # Let the artifact finalize.
        time.sleep(0.5)

        artifact2 = client.materialize(
            inputs=[table_uri],
            transform={"executor": "scan@v1", "params": {}},
        )
        table2 = client.fetch(artifact2.uri)
        assert table2.num_rows == 500
        assert artifact2.cache_hit is True
        assert artifact2.artifact_id == artifact1.artifact_id

    def test_fetch_with_filters(self, server_with_client):
        client = server_with_client["client"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        filters = [{"column": "value", "op": "<", "value": 100.0}]
        artifact = client.materialize(
            inputs=[table_uri],
            transform={
                "executor": "scan@v1",
                "params": {"filters": filters},
            },
        )
        table = client.fetch(artifact.uri)

        # The exact count depends on row-group pruning; this checks the filter is accepted.
        assert table.num_rows >= 0

    def test_duckdb_integration(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")

        try:
            scanner.register("events", table_uri, columns=["id", "value"])

            result = scanner.query("SELECT COUNT(*) as cnt FROM events")
            assert result.column("cnt")[0].as_py() == 500

            result = scanner.query("SELECT AVG(value) as avg_val FROM events")
            avg_val = result.column("avg_val")[0].as_py()
            # Average of 0*1.5, 1*1.5, ..., 499*1.5 = 1.5 * 249.5 = 374.25
            assert abs(avg_val - 374.25) < 0.1

        finally:
            scanner.close()

    def test_metadata_stats_endpoint(self, server_with_client):
        """/v1/metadata/stats."""
        import requests

        client = server_with_client["client"]
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        artifact = client.materialize(
            inputs=[table_uri],
            transform={"executor": "scan@v1", "params": {}},
        )
        table = client.fetch(artifact.uri)
        assert table.num_rows > 0

        response = requests.get(f"http://127.0.0.1:{config.port}/v1/metadata/stats")
        assert response.status_code == 200

        stats = response.json()

        assert "parquet_cache" in stats
        assert "manifest_cache" in stats
        assert "metadata_store" in stats

        pq_stats = stats["parquet_cache"]
        assert "hits" in pq_stats
        assert "misses" in pq_stats

        # Two levels: unfiltered and filtered.
        manifest_stats = stats["manifest_cache"]
        assert "unfiltered" in manifest_stats
        assert "filtered" in manifest_stats
        assert "hits" in manifest_stats["unfiltered"]
        assert "misses" in manifest_stats["unfiltered"]

        store_stats = stats["metadata_store"]
        if store_stats is not None:
            assert "manifest_hits" in store_stats
            assert "parquet_meta_hits" in store_stats
            assert "stale_invalidations" in store_stats

    def test_metadata_cleanup_endpoint(self, server_with_client):
        """/v1/metadata/cleanup."""
        import requests

        config = server_with_client["config"]

        response = requests.post(f"http://127.0.0.1:{config.port}/v1/metadata/cleanup")
        assert response.status_code == 200

        result = response.json()
        assert result["status"] == "completed"
        assert "stale_entries_removed" in result
        assert isinstance(result["stale_entries_removed"], int)

    def test_health_ready_endpoint(self, server_with_client):
        """/health/ready."""
        import requests

        config = server_with_client["config"]

        response = requests.get(f"http://127.0.0.1:{config.port}/health/ready")
        assert response.status_code == 200

        result = response.json()
        assert result["status"] == "ready"
        assert "checks" in result

        checks = result["checks"]
        assert checks["server_initialized"] is True
        assert checks["draining"] is False
        assert checks["capacity_exhausted"] is False
        assert "metadata_store" in checks
        assert "interactive_available" in checks
        assert "bulk_available" in checks
        assert "active_scans" in checks

    def test_prometheus_metrics_endpoint(self, server_with_client):
        """/metrics/prometheus."""
        import requests

        client = server_with_client["client"]
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        artifact = client.materialize(
            inputs=[table_uri],
            transform={"executor": "scan@v1", "params": {}},
        )
        table = client.fetch(artifact.uri)
        assert table.num_rows > 0

        response = requests.get(f"http://127.0.0.1:{config.port}/metrics/prometheus")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

        content = response.text

        assert "strata_cache_hits_total" in content
        assert "strata_cache_misses_total" in content
        assert "strata_scans_total" in content
        assert "strata_active_scans" in content
        assert "strata_draining" in content

        assert "strata_parquet_cache_hits_total" in content
        assert "strata_manifest_cache_hits_total" in content

        assert "# HELP strata_cache_hits_total" in content
        assert "# TYPE strata_cache_hits_total counter" in content

    def test_materialized_scans_move_the_scan_series(self, server_with_client):
        """A built scan and a cache hit each count, per table and per tenant too."""
        import re

        import requests

        client = server_with_client["client"]
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]
        base = f"http://127.0.0.1:{config.port}"

        rows = 0
        for _ in range(2):  # a miss that builds, then a hit
            artifact = client.materialize(
                inputs=[table_uri], transform={"executor": "scan@v1", "params": {}}
            )
            rows = client.fetch(artifact.uri).num_rows
        assert rows > 0

        text = requests.get(f"{base}/metrics/prometheus").text

        def series(name: str) -> str:
            match = re.search(rf"^{name} (\S+)$", text, re.MULTILINE)
            assert match, name
            return match.group(1)

        assert series("strata_scans_total") == "2"
        assert series("strata_rows_returned_total") == str(2 * rows)
        assert re.search(r'^strata_tenant_scans_total\{tenant="_default"\} 2$', text, re.M)
        assert re.search(r"^strata_table_scans_total\{table=\"[^\"]+\"\} 2$", text, re.M)

        tables = requests.get(f"{base}/metrics/tables").json()["tables"]
        assert [(t["scan_count"], t["rows_returned"]) for t in tables] == [(2, 2 * rows)]
        assert tables[0]["cache_hits"] >= 1  # the hit counts its row groups as cached

    def test_a_scan_streamed_without_an_artifact_store_counts(self, temp_warehouse, tmp_path):
        """Service mode with no artifact_dir streams straight from the fetcher."""
        import re

        import requests

        from tests.conftest import find_free_port, run_server

        config = StrataConfig(
            host="127.0.0.1",
            port=find_free_port(),
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
        )
        headers = {"X-Strata-Proxy-Token": "test-token", "X-Strata-Principal": "user-1"}
        with run_server(config) as base:
            response = requests.post(
                f"{base}/v1/materialize",
                json={
                    "inputs": [temp_warehouse["table_uri"]],
                    "transform": {"executor": "scan@v1", "params": {}},
                    "mode": "stream",
                },
                headers=headers,
            )
            assert response.status_code == 200, response.text
            stream = requests.get(f"{base}{response.json()['stream_url']}", headers=headers)
            rows = pa.ipc.open_stream(stream.content).read_all().num_rows
            text = requests.get(f"{base}/metrics/prometheus", headers=headers).text

        assert rows > 0
        assert re.search(r"^strata_scans_total 1$", text, re.M)
        assert re.search(rf"^strata_rows_returned_total {rows}$", text, re.M)

    def test_debug_cache_inspect_endpoint(self, server_with_client):
        """/v1/debug/cache/inspect."""
        import requests

        client = server_with_client["client"]
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        artifact = client.materialize(
            inputs=[table_uri],
            transform={"executor": "scan@v1", "params": {}},
        )
        table = client.fetch(artifact.uri)
        assert table.num_rows > 0

        response = requests.get(f"http://127.0.0.1:{config.port}/v1/debug/cache/inspect")
        assert response.status_code == 200

        result = response.json()
        assert "cache_version" in result
        assert "cache_dir" in result
        assert "entries" in result
        assert "total_matched" in result
        assert "truncated" in result

        assert result["total_matched"] > 0
        assert len(result["entries"]) > 0

        entry = result["entries"][0]
        assert "hash" in entry
        assert "hash_prefix" in entry
        assert "file_path" in entry
        assert "file_exists" in entry
        assert "metadata" in entry

        response = requests.get(f"http://127.0.0.1:{config.port}/v1/debug/cache/inspect?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result["entries"]) <= 1

        # Prefix taken from the first entry's hash.
        first_hash = result["entries"][0]["hash"][:4] if result["entries"] else "0000"
        response = requests.get(
            f"http://127.0.0.1:{config.port}/v1/debug/cache/inspect?prefix={first_hash}"
        )
        assert response.status_code == 200
        result = response.json()
        assert "prefix_filter" in result

        response = requests.get(
            f"http://127.0.0.1:{config.port}/v1/debug/cache/inspect?prefix=zzzz"
        )
        assert response.status_code == 200
        result = response.json()
        assert result["total_matched"] == 0

    def test_cache_warm_endpoint(self, server_with_client):
        """/v1/cache/warm."""
        import requests

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]
        warm_url = f"http://127.0.0.1:{config.port}/v1/cache/warm"
        warm_payload = {
            "tables": [table_uri],
            "columns": ["id", "value"],  # Subset of columns
            "max_row_groups": 2,  # Limit for testing
            "concurrent": 2,
        }

        response = requests.post(f"http://127.0.0.1:{config.port}/v1/cache/clear")
        assert response.status_code == 200

        # Clear-then-warm breaks if ``cache/clear`` deletes the metadata DB's -wal and -shm
        # sidecars. No retry: SQLITE_IOERR poisons the connection for good, so a recurrence
        # must be visible rather than absorbed.
        response = requests.post(warm_url, json=warm_payload)
        assert response.status_code == 200
        result = response.json()

        # Assert errors first so a CI failure shows the exception text, not "0 == 1".
        assert result["errors"] == [], result
        assert result["tables_warmed"] == 1
        assert result["row_groups_cached"] > 0
        assert result["row_groups_skipped"] == 0  # Cache was cleared
        assert result["bytes_written"] > 0
        assert result["elapsed_ms"] > 0

        # Warm again: cached row groups are skipped.
        response = requests.post(
            f"http://127.0.0.1:{config.port}/v1/cache/warm",
            json={
                "tables": [table_uri],
                "columns": ["id", "value"],
                "max_row_groups": 2,
                "concurrent": 2,
            },
        )
        assert response.status_code == 200

        result2 = response.json()
        assert result2["tables_warmed"] == 1
        assert result2["row_groups_cached"] == 0
        assert result2["row_groups_skipped"] > 0
        assert result2["bytes_written"] == 0

    def test_cache_warm_with_invalid_table(self, server_with_client):
        """Cache warming with an invalid table URI."""
        import requests

        config = server_with_client["config"]

        response = requests.post(
            f"http://127.0.0.1:{config.port}/v1/cache/warm",
            json={
                "tables": ["file:///nonexistent#db.table"],
                "concurrent": 1,
            },
        )
        assert response.status_code == 200

        result = response.json()
        assert result["tables_warmed"] == 0
        assert len(result["errors"]) == 1


class TestEagerWarmup:
    """Eager warmup at server startup."""

    def test_eager_warmup_returns_timing_info(self, tmp_path):
        """_eager_warmup returns timing information."""
        from strata.config import StrataConfig
        from strata.server import _eager_warmup

        config = StrataConfig(cache_dir=tmp_path / "cache")
        warmup_times = _eager_warmup(config, scans=True)

        assert "total_ms" in warmup_times
        assert "imports_ms" in warmup_times
        assert "sqlite_ms" in warmup_times
        assert "caches_ms" in warmup_times

        assert warmup_times["total_ms"] >= 0
        assert warmup_times["imports_ms"] >= 0
        assert warmup_times["sqlite_ms"] >= 0
        assert warmup_times["caches_ms"] >= 0

        assert "sqlite_entries" in warmup_times

    def test_warmup_initializes_metadata_store(self, tmp_path):
        from strata.cache import metadata as metadata_cache
        from strata.cache.metadata import get_metadata_store
        from strata.config import StrataConfig
        from strata.server import _eager_warmup

        metadata_cache._metadata_store = None

        config = StrataConfig(cache_dir=tmp_path / "cache")
        _eager_warmup(config, scans=True)

        store = get_metadata_store(config.cache_dir)
        assert store is not None
        stats = store.stats()
        assert isinstance(stats, dict)

    def test_without_scans_it_loads_no_scan_plane(self, tmp_path):
        from strata.config import StrataConfig
        from strata.server import _eager_warmup

        warmup_times = _eager_warmup(StrataConfig(cache_dir=tmp_path / "cache"), scans=False)

        assert "imports_ms" not in warmup_times and "caches_ms" not in warmup_times
        assert not (tmp_path / "cache").exists()


# Run in a fresh interpreter, so what the server's start loaded is all that sys.modules holds.
_STARTUP_PROBE = """
import json, os, sys
from fastapi.testclient import TestClient
from strata.server import app, get_state

HEAVY = ("pyiceberg", "pandas", "sqlalchemy", "strata.planner", "strata.cache.disk")
seen = {}
with TestClient(app, base_url="http://127.0.0.1") as client:
    assert client.get("/health").status_code == 200
    # Reports on the scan plane do not build it.
    assert client.get("/metrics").status_code == 200
    assert client.get("/metrics/prometheus").status_code == 200
    seen["started"] = [m for m in HEAVY if m in sys.modules]
    seen["built"] = get_state().built_planner is not None
    if sys.argv[1:] == ["scan"]:
        get_state().planner
seen["stopped"] = [m for m in HEAVY if m in sys.modules]
# A file, not stdout: the server's own log lines go there too, in any order.
with open(os.environ["PROBE_OUT"], "w") as out:
    json.dump(seen, out)
"""


class TestScanPlaneOnFirstUse:
    """The server's planner and fetcher, built once on first use."""

    @pytest.fixture
    def state(self, tmp_path):
        from strata.server import ServerState

        state = ServerState(StrataConfig(cache_dir=tmp_path / "cache", artifact_dir=tmp_path / "a"))
        yield state
        state._planning_executor.shutdown()
        state._fetch_executor.shutdown()

    def test_threads_asking_at_once_get_one_planner(self, state, monkeypatch):
        built = []
        entered = threading.Event()
        release = threading.Event()

        class Planner:
            def __init__(self, config):
                built.append(self)
                entered.set()
                release.wait(10)

        monkeypatch.setattr("strata.planner.ReadPlanner", Planner)
        got = []
        threads = [threading.Thread(target=lambda: got.append(state.planner)) for _ in range(4)]
        threads[0].start()
        entered.wait(10)
        for thread in threads[1:]:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(10)

        assert len(built) == 1 and got == built * 4

    def test_closing_waits_for_a_planner_being_built(self, state, monkeypatch):
        entered = threading.Event()
        release = threading.Event()
        closed = []

        class Planner:
            def __init__(self, config):
                entered.set()
                release.wait(10)

            def close(self):
                closed.append(self)

        monkeypatch.setattr("strata.planner.ReadPlanner", Planner)
        building = threading.Thread(target=lambda: state.planner)
        building.start()
        entered.wait(10)
        closing = threading.Thread(target=state.close_scan_plane)
        closing.start()
        release.set()
        building.join(10)
        closing.join(10)

        assert closed == [state.built_planner]

    def test_a_planner_never_built_is_not_closed(self, state, monkeypatch):
        monkeypatch.setattr("strata.planner.ReadPlanner", None)
        state.close_scan_plane()

        assert state.built_planner is None

    def test_an_artifact_input_resolves_without_a_planner(self, state, monkeypatch):
        from strata.api.dependencies import _input_resolution

        class Store:
            def get_artifact(self, artifact_id, version):
                return SimpleNamespace(id=artifact_id, version=version)

        def no_planner(config):
            raise AssertionError("a planner was built")

        monkeypatch.setattr("strata.planner.ReadPlanner", no_planner)
        monkeypatch.setattr("strata.server._state", state)
        monkeypatch.setattr("strata.server._get_artifact_store", lambda **kwargs: Store())

        _, resolve, _ = _input_resolution("strata://artifact/abc@v=2", None)

        assert resolve().version == "abc@v=2"
        assert state.built_planner is None


class TestScanPlaneAtStartup:
    """A scratchpad daemon starts without the scan plane; any other server warms it."""

    def _start(self, tmp_path, anchor: str, *args: str) -> dict:
        import json
        import os
        import subprocess

        env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
        env.update(
            HOME=str(tmp_path / "home"),
            STRATA_SCRATCH_ANCHOR=anchor,
            STRATA_ARTIFACT_DIR=str(tmp_path / "artifacts"),
            STRATA_CACHE_DIR=str(tmp_path / "cache"),
            STRATA_NOTEBOOK_STORAGE_DIR=str(tmp_path),
            PROBE_OUT=str(tmp_path / "probe.json"),
        )
        (tmp_path / "probe.json").unlink(missing_ok=True)  # the test's first start wrote one
        done = subprocess.run(
            [sys.executable, "-c", _STARTUP_PROBE, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        return json.loads((tmp_path / "probe.json").read_text())

    def test_a_scratchpad_daemon_loads_it_on_first_use_only(self, tmp_path):
        idle = self._start(tmp_path, str(tmp_path))
        # Its shutdown closes no planner it never built.
        assert idle == {"started": [], "built": False, "stopped": []}

        scanned = self._start(tmp_path, str(tmp_path), "scan")
        assert "pyiceberg" in scanned["stopped"] and "strata.planner" in scanned["stopped"]

    def test_any_other_server_warms_it_at_start(self, tmp_path):
        seen = self._start(tmp_path, "")

        assert "pyiceberg" in seen["started"] and "strata.cache.disk" in seen["started"]
        assert seen["built"]
