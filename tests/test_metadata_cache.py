"""Tests for metadata caching (Parquet metadata and manifest resolution)."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata.metadata_cache import (
    LRUCache,
    ManifestCache,
    ManifestEntry,
    ManifestResolution,
    ParquetMetadata,
    ParquetMetadataCache,
    clear_all_caches,
    get_manifest_cache,
    get_parquet_cache,
    reset_caches,
)


class TestLRUCache:
    """Tests for the base LRU cache."""

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache[str, int](max_size=10)
        cache.put("a", 1)
        cache.put("b", 2)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") is None

    def test_lru_eviction(self):
        """Test that oldest entries are evicted when at capacity."""
        cache = LRUCache[str, int](max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # This should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_updates_lru_order(self):
        """Test that accessing an entry updates its LRU position."""
        cache = LRUCache[str, int](max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)

        # Access "a" to make it most recently used
        cache.get("a")

        # Now add "c", which should evict "b" (not "a")
        cache.put("c", 3)

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_update_existing_key(self):
        """Test that updating an existing key works."""
        cache = LRUCache[str, int](max_size=2)
        cache.put("a", 1)
        cache.put("a", 2)  # Update

        assert cache.get("a") == 2
        assert len(cache) == 1

    def test_stats(self):
        """Test cache statistics."""
        cache = LRUCache[str, int](max_size=10)
        cache.put("a", 1)
        cache.put("b", 2)

        cache.get("a")  # Hit
        cache.get("b")  # Hit
        cache.get("c")  # Miss

        stats = cache.stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3

    def test_clear(self):
        """Test clearing the cache."""
        cache = LRUCache[str, int](max_size=10)
        cache.put("a", 1)
        cache.put("b", 2)

        cache.clear()

        assert len(cache) == 0
        assert cache.get("a") is None

    def test_contains(self):
        """Test contains check (doesn't update LRU order)."""
        cache = LRUCache[str, int](max_size=2)
        cache.put("a", 1)

        assert "a" in cache
        assert "b" not in cache

    def test_get_with_default(self):
        """Test get with default value."""
        cache = LRUCache[str, int](max_size=10)
        cache.put("a", 1)

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("b", 42) == 42
        assert cache.get("a", 99) == 1  # Existing value, not default

    def test_get_or_put(self):
        """Test get_or_put computes only on miss."""
        cache = LRUCache[str, int](max_size=10)
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return 42

        # First call should invoke factory
        result1 = cache.get_or_put("a", factory)
        assert result1 == 42
        assert call_count == 1

        # Second call should return cached value, not invoke factory
        result2 = cache.get_or_put("a", factory)
        assert result2 == 42
        assert call_count == 1  # Still 1

    def test_resize_shrink(self):
        """Test resizing cache smaller evicts entries."""
        cache = LRUCache[str, int](max_size=5)
        for i in range(5):
            cache.put(str(i), i)

        assert len(cache) == 5

        cache.resize(2)
        assert len(cache) == 2
        assert cache.stats()["max_size"] == 2
        # Oldest entries (0, 1, 2) should be evicted
        assert "0" not in cache
        assert "1" not in cache
        assert "2" not in cache
        assert "3" in cache
        assert "4" in cache

    def test_resize_grow(self):
        """Test resizing cache larger allows more entries."""
        cache = LRUCache[str, int](max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)

        cache.resize(5)
        cache.put("c", 3)
        cache.put("d", 4)

        assert len(cache) == 4
        assert cache.get("a") == 1  # Not evicted

    def test_max_size_zero_disables_cache(self):
        """Test max_size=0 disables caching."""
        cache = LRUCache[str, int](max_size=0)
        cache.put("a", 1)

        assert len(cache) == 0
        assert cache.get("a") is None

    def test_evictions_counter(self):
        """Test evictions are tracked in stats."""
        cache = LRUCache[str, int](max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Evicts "a"
        cache.put("d", 4)  # Evicts "b"

        stats = cache.stats()
        assert stats["evictions"] == 2

    def test_updates_counter(self):
        """Test updates (overwrites) are tracked in stats."""
        cache = LRUCache[str, int](max_size=10)
        cache.put("a", 1)
        cache.put("a", 2)  # Update
        cache.put("a", 3)  # Update
        cache.put("b", 1)  # New entry

        stats = cache.stats()
        assert stats["updates"] == 2


class TestParquetMetadataCache:
    """Tests for Parquet metadata caching."""

    @pytest.fixture
    def sample_parquet_file(self, tmp_path):
        """Create a sample Parquet file."""
        table = pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
                "name": ["a", "b", "c", "d", "e"],
            }
        )
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path, row_group_size=2)
        return str(file_path)

    def test_get_or_load_caches_metadata(self, sample_parquet_file):
        """Test that get_or_load caches Parquet metadata."""
        cache = ParquetMetadataCache(max_size=10)

        # First call should load
        stats_before = cache.stats()
        assert stats_before["misses"] == 0

        meta1 = cache.get_or_load(sample_parquet_file)

        stats_after = cache.stats()
        assert stats_after["misses"] == 1  # Cache miss
        assert stats_after["size"] == 1

        # Second call should hit cache
        meta2 = cache.get_or_load(sample_parquet_file)

        stats_final = cache.stats()
        assert stats_final["hits"] == 1  # Cache hit
        assert stats_final["misses"] == 1  # No new misses

        # Same metadata object
        assert meta1 is meta2

    def test_metadata_contains_expected_fields(self, sample_parquet_file):
        """Test that cached metadata has all expected fields."""
        cache = ParquetMetadataCache(max_size=10)
        meta = cache.get_or_load(sample_parquet_file)

        assert isinstance(meta, ParquetMetadata)
        assert meta.arrow_schema is not None
        assert meta.num_row_groups == 3  # 5 rows / 2 per group = 3 groups
        assert len(meta.row_group_metadata) == 3
        assert meta.parquet_schema is not None

    def test_row_group_metadata_accessible(self, sample_parquet_file):
        """Test that row group metadata is accessible from cache."""
        cache = ParquetMetadataCache(max_size=10)
        meta = cache.get_or_load(sample_parquet_file)

        # Check we can access row group metadata
        for i, rg_meta in enumerate(meta.row_group_metadata):
            assert rg_meta.num_rows > 0
            # First two groups have 2 rows, last has 1
            if i < 2:
                assert rg_meta.num_rows == 2
            else:
                assert rg_meta.num_rows == 1

    def test_lru_eviction_works(self, tmp_path):
        """Test that LRU eviction works for Parquet cache."""
        cache = ParquetMetadataCache(max_size=2)

        # Create 3 Parquet files
        files = []
        for i in range(3):
            table = pa.table({"x": [i]})
            file_path = tmp_path / f"test_{i}.parquet"
            pq.write_table(table, file_path)
            files.append(str(file_path))

        # Load all 3 (should evict first)
        cache.get_or_load(files[0])
        cache.get_or_load(files[1])
        cache.get_or_load(files[2])  # Evicts files[0]

        assert cache.get(files[0]) is None
        assert cache.get(files[1]) is not None
        assert cache.get(files[2]) is not None

    def test_get_or_load_many_persists_without_rereading_file(
        self, sample_parquet_file, tmp_path, monkeypatch
    ):
        """Batch loads should persist from loaded metadata, not reopen the file."""
        import strata.metadata_store as metadata_store

        store = metadata_store.MetadataStore(tmp_path / "metadata.sqlite")
        cache = ParquetMetadataCache(max_size=10, store=store)

        def fail_extract(*args, **kwargs):
            raise AssertionError("extract_parquet_meta should not be called")

        monkeypatch.setattr(metadata_store, "extract_parquet_meta", fail_extract)

        result = cache.get_or_load_many([sample_parquet_file])

        assert sample_parquet_file in result
        assert store.get_parquet_meta(sample_parquet_file) is not None


class TestManifestCache:
    """Tests for manifest resolution caching."""

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = ManifestCache(max_size=10)

        resolution = ManifestResolution(
            data_files=[
                ManifestEntry(file_path="/data/file1.parquet", actual_path="/abs/file1.parquet"),
                ManifestEntry(file_path="/data/file2.parquet", actual_path="/abs/file2.parquet"),
            ]
        )

        cache.put("default", "strata.ns.table", 123, resolution)

        # Same catalog+table+snapshot should hit
        cached = cache.get("default", "strata.ns.table", 123)
        assert cached is not None
        assert len(cached.data_files) == 2
        assert cached.data_files[0].file_path == "/data/file1.parquet"

        # Different snapshot should miss
        assert cache.get("default", "strata.ns.table", 124) is None

        # Different table should miss
        assert cache.get("default", "strata.ns.other", 123) is None

        # Different catalog should miss
        assert cache.get("other_catalog", "strata.ns.table", 123) is None

    def test_cache_key_includes_snapshot_id(self):
        """Test that different snapshots have different cache entries."""
        cache = ManifestCache(max_size=10)

        res1 = ManifestResolution(
            data_files=[ManifestEntry(file_path="/data/v1.parquet", actual_path="/data/v1.parquet")]
        )
        res2 = ManifestResolution(
            data_files=[ManifestEntry(file_path="/data/v2.parquet", actual_path="/data/v2.parquet")]
        )

        cache.put("default", "strata.ns.table", 1, res1)
        cache.put("default", "strata.ns.table", 2, res2)

        cached1 = cache.get("default", "strata.ns.table", 1)
        cached2 = cache.get("default", "strata.ns.table", 2)
        assert cached1 is not None
        assert cached2 is not None

        assert cached1.data_files[0].file_path == "/data/v1.parquet"
        assert cached2.data_files[0].file_path == "/data/v2.parquet"

    def test_lru_eviction(self):
        """Test that LRU eviction works for manifest cache."""
        cache = ManifestCache(max_size=2)

        res1 = ManifestResolution(data_files=[])
        res2 = ManifestResolution(data_files=[])
        res3 = ManifestResolution(data_files=[])

        cache.put("default", "table1", 1, res1)
        cache.put("default", "table2", 1, res2)
        cache.put("default", "table3", 1, res3)  # Evicts table1

        assert cache.get("default", "table1", 1) is None
        assert cache.get("default", "table2", 1) is not None
        assert cache.get("default", "table3", 1) is not None

    def test_filtered_queries_fall_back_to_unfiltered_cache(self):
        """Filtered lookups should reuse the unfiltered resolution when needed."""
        cache = ManifestCache(max_size=10)
        resolution = ManifestResolution(
            data_files=[
                ManifestEntry(
                    file_path="/data/file1.parquet",
                    actual_path="/abs/file1.parquet",
                )
            ]
        )

        cache.put("default", "strata.ns.table", 123, resolution)

        cached = cache.get("default", "strata.ns.table", 123, filter_fingerprint="f1")

        assert cached is not None
        assert cached.data_files[0].file_path == "/data/file1.parquet"

    def test_filtered_queries_fall_back_to_persisted_unfiltered(self, tmp_path):
        """Filtered lookups should use persisted unfiltered manifest results after restart."""
        from strata.metadata_store import MetadataStore

        store = MetadataStore(tmp_path / "metadata.sqlite")
        cache = ManifestCache(max_size=10, store=store)
        resolution = ManifestResolution(
            data_files=[
                ManifestEntry(
                    file_path="/data/file1.parquet",
                    actual_path="/abs/file1.parquet",
                )
            ]
        )
        cache.put("default", "strata.ns.table", 123, resolution)

        restarted_cache = ManifestCache(max_size=10, store=store)
        cached = restarted_cache.get(
            "default",
            "strata.ns.table",
            123,
            filter_fingerprint="f1",
        )

        assert cached is not None
        assert cached.data_files[0].actual_path == "/abs/file1.parquet"


class TestGlobalCaches:
    """Tests for global cache singletons."""

    def setup_method(self):
        """Reset global caches before each test."""
        reset_caches()

    def test_get_parquet_cache_creates_singleton(self):
        """Test that get_parquet_cache returns a singleton."""
        cache1 = get_parquet_cache()
        cache2 = get_parquet_cache()
        assert cache1 is cache2

    def test_get_manifest_cache_creates_singleton(self):
        """Test that get_manifest_cache returns a singleton."""
        cache1 = get_manifest_cache()
        cache2 = get_manifest_cache()
        assert cache1 is cache2

    def test_clear_all_caches(self, tmp_path):
        """Test that clear_all_caches clears both caches."""
        # Create a Parquet file
        table = pa.table({"x": [1]})
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        # Populate caches
        pq_cache = get_parquet_cache()
        manifest_cache = get_manifest_cache()

        pq_cache.get_or_load(str(file_path))
        manifest_cache.put("default", "table", 1, ManifestResolution(data_files=[]))

        assert len(pq_cache._cache) == 1
        assert len(manifest_cache._cache) == 1

        # Clear all
        clear_all_caches()

        assert len(pq_cache._cache) == 0
        assert len(manifest_cache._cache) == 0

    def test_reset_caches(self):
        """Test that reset_caches recreates new instances."""
        cache1 = get_parquet_cache()
        reset_caches()
        cache2 = get_parquet_cache()
        assert cache1 is not cache2


class TestPlannerWithMetadataCache:
    """Integration tests for planner with metadata caching."""

    @pytest.fixture
    def warehouse_with_table(self, tmp_path):
        """Create a warehouse with an Iceberg table."""
        import sys

        if sys.platform == "win32":
            pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")

        from pyiceberg.catalog.sql import SqlCatalog
        from pyiceberg.schema import Schema
        from pyiceberg.types import LongType, NestedField, StringType

        warehouse_path = tmp_path / "warehouse"
        warehouse_path.mkdir()

        # Use "strata" as catalog name to match what the planner expects
        catalog = SqlCatalog(
            "strata",
            **{
                "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
                "warehouse": str(warehouse_path),
            },
        )

        catalog.create_namespace("test_ns")

        # Use LongType to match PyArrow's default int64
        schema = Schema(
            NestedField(1, "id", LongType()),
            NestedField(2, "name", StringType()),
        )
        table = catalog.create_table("test_ns.events", schema)

        # Write some data
        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        df = pa.Table.from_batches([batch])
        table.append(df)

        return {
            "warehouse_path": warehouse_path,
            "table_uri": f"file://{warehouse_path}#test_ns.events",
            "catalog": catalog,
        }

    def test_planner_uses_parquet_cache(self, warehouse_with_table):
        """Test that planner uses Parquet metadata cache."""
        reset_caches()

        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        config = StrataConfig()
        planner = ReadPlanner(config)

        table_uri = warehouse_with_table["table_uri"]

        # First plan - cache miss
        plan1 = planner.plan(table_uri)
        assert len(plan1.tasks) > 0

        pq_cache_stats = planner.parquet_cache.stats()
        assert pq_cache_stats["misses"] >= 1

        # Second plan - should use cache
        plan2 = planner.plan(table_uri)
        assert len(plan2.tasks) == len(plan1.tasks)

        pq_cache_stats = planner.parquet_cache.stats()
        assert pq_cache_stats["hits"] >= 1

    def test_planner_uses_manifest_cache(self, warehouse_with_table):
        """Test that planner uses manifest resolution cache."""
        reset_caches()

        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        config = StrataConfig()
        planner = ReadPlanner(config)

        table_uri = warehouse_with_table["table_uri"]

        # First plan - cache miss
        plan1 = planner.plan(table_uri)
        assert len(plan1.tasks) > 0

        manifest_stats = planner.manifest_cache.stats()
        # Stats are now nested: {"unfiltered": {...}, "filtered": {...}}
        assert manifest_stats["unfiltered"]["misses"] >= 1

        # Second plan - should use cache
        plan2 = planner.plan(table_uri)
        assert len(plan2.tasks) == len(plan1.tasks)

        manifest_stats = planner.manifest_cache.stats()
        assert manifest_stats["unfiltered"]["hits"] >= 1

    def test_manifest_cache_isolated_per_warehouse(self, tmp_path):
        """Different warehouses with the same table name should not share manifests."""
        import sys

        if sys.platform == "win32":
            pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
        reset_caches()

        from pyiceberg.catalog.sql import SqlCatalog
        from pyiceberg.schema import Schema
        from pyiceberg.types import LongType, NestedField, StringType

        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        schema = Schema(
            NestedField(1, "id", LongType()),
            NestedField(2, "name", StringType()),
        )

        def _create_warehouse(base: Path, values: list[int]) -> str:
            base.mkdir(parents=True, exist_ok=True)
            warehouse_path = base / "warehouse"
            warehouse_path.mkdir()
            warehouse_uri = warehouse_path.as_uri()
            catalog_db = (warehouse_path / "catalog.db").as_posix()
            catalog = SqlCatalog(
                "strata",
                **{
                    "uri": f"sqlite:///{catalog_db}",
                    "warehouse": warehouse_uri,
                },
            )
            catalog.create_namespace("test_ns")
            table = catalog.create_table("test_ns.events", schema)
            batch = pa.RecordBatch.from_pydict(
                {
                    "id": values,
                    "name": [f"item_{value}" for value in values],
                }
            )
            table.append(pa.Table.from_batches([batch]))
            return f"{warehouse_uri}#test_ns.events"

        table_uri_one = _create_warehouse(tmp_path / "one", [1, 2, 3])
        table_uri_two = _create_warehouse(tmp_path / "two", [10, 11, 12])

        planner = ReadPlanner(StrataConfig())

        plan_one = planner.plan(table_uri_one)
        plan_two = planner.plan(table_uri_two)

        warehouse_two_path = str((tmp_path / "two" / "warehouse").resolve())
        assert len(plan_one.tasks) > 0
        assert len(plan_two.tasks) > 0
        assert all(task.file_path.startswith(warehouse_two_path) for task in plan_two.tasks)

        manifest_stats = planner.manifest_cache.stats()
        assert manifest_stats["unfiltered"]["misses"] >= 2

    def test_different_snapshots_use_different_cache_entries(self, warehouse_with_table):
        """Test that different snapshots don't share manifest cache entries."""
        reset_caches()

        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        catalog = warehouse_with_table["catalog"]
        table = catalog.load_table("test_ns.events")

        # Add a second snapshot
        batch = pa.RecordBatch.from_pydict({"id": [4, 5, 6], "name": ["d", "e", "f"]})
        df = pa.Table.from_batches([batch])
        table.append(df)

        # Get both snapshot IDs
        snapshots = list(table.history())
        assert len(snapshots) >= 2
        snap1_id = snapshots[0].snapshot_id
        snap2_id = snapshots[1].snapshot_id

        config = StrataConfig()
        planner = ReadPlanner(config)
        table_uri = warehouse_with_table["table_uri"]

        # Plan for snapshot 1
        planner.plan(table_uri, snapshot_id=snap1_id)

        # Plan for snapshot 2
        planner.plan(table_uri, snapshot_id=snap2_id)

        # Both should be cache misses (different snapshots)
        manifest_stats = planner.manifest_cache.stats()
        assert manifest_stats["unfiltered"]["misses"] >= 2

        # Repeat - should hit cache
        planner.plan(table_uri, snapshot_id=snap1_id)
        planner.plan(table_uri, snapshot_id=snap2_id)

        manifest_stats = planner.manifest_cache.stats()
        assert manifest_stats["unfiltered"]["hits"] >= 2


class TestMetadataStore:
    """Tests for SQLite-backed metadata store."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a MetadataStore with a temp database."""
        from strata.metadata_store import MetadataStore

        db_path = tmp_path / "test_metadata.sqlite"
        return MetadataStore(db_path)

    @pytest.fixture
    def sample_parquet_files(self, tmp_path):
        """Create sample Parquet files for testing."""
        files = []
        for i in range(3):
            file_path = tmp_path / f"test_{i}.parquet"
            table = pa.table(
                {
                    "id": [i * 10 + j for j in range(5)],
                    "name": [f"row_{j}" for j in range(5)],
                }
            )
            pq.write_table(table, file_path)
            files.append(str(file_path))
        return files

    def test_manifest_put_and_get(self, store):
        """Test basic manifest cache operations."""
        data_files = [
            ("/data/f1.parquet", "/abs/f1.parquet"),
            ("/data/f2.parquet", "/abs/f2.parquet"),
        ]

        store.put_manifest("default", "ns.table", 123, data_files)

        result = store.get_manifest("default", "ns.table", 123)
        assert result is not None
        assert len(result) == 2
        assert result[0] == ("/data/f1.parquet", "/abs/f1.parquet")

    def test_manifest_miss(self, store):
        """Test manifest cache miss."""
        result = store.get_manifest("default", "ns.table", 999)
        assert result is None
        assert store.manifest_misses == 1

    def test_manifest_hit_counter(self, store):
        """Test manifest hit counter."""
        store.put_manifest("default", "ns.table", 1, [])
        store.get_manifest("default", "ns.table", 1)
        assert store.manifest_hits == 1

    def test_parquet_meta_put_and_get(self, store, sample_parquet_files):
        """Test basic parquet metadata operations."""
        from strata.metadata_store import (
            extract_parquet_meta,
        )

        file_path = sample_parquet_files[0]
        meta = extract_parquet_meta(file_path)

        store.put_parquet_meta(file_path, meta)

        result = store.get_parquet_meta(file_path)
        assert result is not None
        assert result.num_row_groups == meta.num_row_groups
        assert result.column_names == meta.column_names

    def test_parquet_meta_stale_detection(self, store, tmp_path):
        """Test that stale entries are detected."""
        import time

        from strata.metadata_store import extract_parquet_meta

        file_path = tmp_path / "stale_test.parquet"
        table = pa.table({"x": [1, 2, 3]})
        pq.write_table(table, file_path)

        meta = extract_parquet_meta(str(file_path))
        store.put_parquet_meta(str(file_path), meta)

        # Verify it's cached
        assert store.get_parquet_meta(str(file_path)) is not None
        initial_stale = store.stale_invalidations

        # Modify the file. Sleep covers the coarser mtime resolution on
        # Windows/FAT32 (~1s) so the second write definitively bumps
        # the timestamp; POSIX tmpfs has sub-second precision so this
        # is just generous padding.
        time.sleep(1.1)
        table2 = pa.table({"x": [4, 5, 6, 7]})
        pq.write_table(table2, file_path)

        # Should detect staleness
        result = store.get_parquet_meta(str(file_path))
        assert result is None
        assert store.stale_invalidations == initial_stale + 1

    def test_get_parquet_meta_many(self, store, sample_parquet_files):
        """Test batch get for parquet metadata."""
        from strata.metadata_store import extract_parquet_meta

        # Store metadata for all files
        for file_path in sample_parquet_files:
            meta = extract_parquet_meta(file_path)
            store.put_parquet_meta(file_path, meta)

        # Batch get
        result = store.get_parquet_meta_many(sample_parquet_files)

        assert len(result) == 3
        for file_path in sample_parquet_files:
            assert file_path in result
            assert result[file_path].num_row_groups >= 1

    def test_get_parquet_meta_many_partial(self, store, sample_parquet_files):
        """Test batch get with some missing entries."""
        from strata.metadata_store import extract_parquet_meta

        # Only store first file
        meta = extract_parquet_meta(sample_parquet_files[0])
        store.put_parquet_meta(sample_parquet_files[0], meta)

        # Batch get all three
        result = store.get_parquet_meta_many(sample_parquet_files)

        assert len(result) == 1
        assert sample_parquet_files[0] in result

    def test_get_parquet_meta_many_empty(self, store):
        """Test batch get with empty input."""
        result = store.get_parquet_meta_many([])
        assert result == {}

    def test_put_parquet_meta_many(self, store, sample_parquet_files):
        """Test batch put for parquet metadata."""
        from strata.metadata_store import extract_parquet_meta

        # Extract metadata for all files
        items = [(fp, extract_parquet_meta(fp)) for fp in sample_parquet_files]

        # Batch put
        store.put_parquet_meta_many(items)

        # Verify all stored
        for file_path in sample_parquet_files:
            result = store.get_parquet_meta(file_path)
            assert result is not None

    def test_put_parquet_meta_many_empty(self, store):
        """Test batch put with empty input."""
        store.put_parquet_meta_many([])  # Should not raise

    def test_stats_includes_counters(self, store, sample_parquet_files):
        """Test that stats() includes all counters."""
        from strata.metadata_store import extract_parquet_meta

        # Generate some hits and misses
        store.get_manifest("default", "ns.table", 1)  # miss
        store.put_manifest("default", "ns.table", 1, [])
        store.get_manifest("default", "ns.table", 1)  # hit

        meta = extract_parquet_meta(sample_parquet_files[0])
        store.get_parquet_meta(sample_parquet_files[0])  # miss
        store.put_parquet_meta(sample_parquet_files[0], meta)
        store.get_parquet_meta(sample_parquet_files[0])  # hit

        stats = store.stats()

        assert stats["manifest_hits"] == 1
        assert stats["manifest_misses"] == 1
        assert stats["parquet_meta_hits"] == 1
        assert stats["parquet_meta_misses"] == 1
        assert "stale_invalidations" in stats
        assert "db_path" in stats

    def test_cleanup_stale_parquet_meta(self, store, tmp_path):
        """Test cleanup of stale entries."""
        from strata.metadata_store import extract_parquet_meta

        # Create a file and cache its metadata
        file_path = tmp_path / "cleanup_test.parquet"
        table = pa.table({"x": [1, 2, 3]})
        pq.write_table(table, file_path)

        meta = extract_parquet_meta(str(file_path))
        store.put_parquet_meta(str(file_path), meta)

        # Delete the file
        file_path.unlink()

        # Cleanup should remove the stale entry
        removed = store.cleanup_stale_parquet_meta()
        assert removed == 1

        # Entry should be gone
        stats = store.stats()
        assert stats["parquet_entries"] == 0

    def test_remote_parquet_meta_is_not_treated_as_stale(self, store, tmp_path):
        """Remote parquet metadata should survive lookup and stale cleanup."""
        from strata.metadata_store import extract_parquet_meta

        source_file = tmp_path / "remote_source.parquet"
        table = pa.table({"x": [1, 2, 3]})
        pq.write_table(table, source_file)

        meta = extract_parquet_meta(str(source_file))
        remote_path = "s3://bucket/path/remote_source.parquet"
        store.put_parquet_meta(remote_path, meta)

        assert store.get_parquet_meta(remote_path) is not None
        assert remote_path in store.get_parquet_meta_many([remote_path])
        assert store.cleanup_stale_parquet_meta() == 0
        assert store.stats()["parquet_entries"] == 1

    def test_schema_migration(self, tmp_path):
        """Test that schema migration works for old databases."""
        import sqlite3

        from strata.metadata_store import MetadataStore

        db_path = tmp_path / "old_schema.sqlite"

        # Create old schema without catalog_name and file_size
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE manifest_cache (
                table_identity TEXT NOT NULL,
                snapshot_id INTEGER NOT NULL,
                data_files_json TEXT NOT NULL,
                PRIMARY KEY (table_identity, snapshot_id)
            );
            CREATE TABLE parquet_meta (
                file_path TEXT PRIMARY KEY,
                schema_ipc BLOB NOT NULL,
                num_row_groups INTEGER NOT NULL,
                row_groups_json TEXT NOT NULL,
                column_names_json TEXT NOT NULL,
                file_mtime REAL
            );
        """)
        conn.close()

        # Opening with MetadataStore should migrate
        store = MetadataStore(db_path)

        # Should work with new schema
        store.put_manifest("default", "ns.table", 1, [])
        result = store.get_manifest("default", "ns.table", 1)
        assert result == []


class TestNestedColumnStatsUsePaths:
    """Persisted Parquet stats must be keyed by the column's dotted PATH.

    A struct field ``user.id`` has Parquet leaf name ``id``, colliding with a
    top-level ``id``. Keying persisted stats by leaf name (a) let the nested
    column's stats overwrite the top-level column's and (b) stripped the dot
    that the planner's ``"." in col.path`` guard uses to skip nested columns —
    so after metadata round-tripped through SQLite (i.e. after any restart)
    row-group pruning compared a filter against the WRONG column's min/max and
    silently dropped matching rows. That breaks the conservative-pruning
    invariant in the unsafe direction.
    """

    def _nested_file(self, tmp_path: Path) -> Path:
        fp = tmp_path / "nested.parquet"
        pq.write_table(
            pa.table(
                {
                    "id": pa.array([1, 2, 3]),
                    "user": pa.array([{"id": 100}, {"id": 200}, {"id": 300}]),
                }
            ),
            fp,
        )
        return fp

    def test_persisted_column_names_are_unique_paths(self, tmp_path):
        from strata.metadata_store import extract_parquet_meta

        persisted = extract_parquet_meta(str(self._nested_file(tmp_path)))
        assert persisted.column_names == ["id", "user.id"]

    def test_top_level_stats_survive_a_nested_name_collision(self, tmp_path):
        from strata.metadata_store import extract_parquet_meta

        persisted = extract_parquet_meta(str(self._nested_file(tmp_path)))
        stats = persisted.row_groups[0].column_stats
        # Both columns keep their own stats (the nested one used to clobber).
        assert stats["id"]["min"] == 1 and stats["id"]["max"] == 3
        assert stats["user.id"]["min"] == 100 and stats["user.id"]["max"] == 300

    def test_restored_schema_excludes_nested_from_the_column_map(self, tmp_path):
        from strata.metadata_cache import ParquetSchema
        from strata.metadata_store import extract_parquet_meta
        from strata.planner import _build_column_index_map

        persisted = extract_parquet_meta(str(self._nested_file(tmp_path)))
        col_map = _build_column_index_map(ParquetSchema(_column_names=persisted.column_names))
        # The nested column must not shadow the top-level one.
        assert col_map == {"id": 0}

    def test_restored_stats_do_not_prune_a_matching_row_group(self, tmp_path):
        """The end-to-end symptom: `id = 2` must not prune a group holding 1-3."""
        from strata.metadata_cache import ParquetSchema
        from strata.metadata_store import extract_parquet_meta
        from strata.planner import _build_column_index_map

        persisted = extract_parquet_meta(str(self._nested_file(tmp_path)))
        col_map = _build_column_index_map(ParquetSchema(_column_names=persisted.column_names))
        stats = persisted.row_groups[0].column_stats[persisted.column_names[col_map["id"]]]
        assert stats["min"] <= 2 <= stats["max"]

    def test_schema_shim_splits_name_from_path(self):
        from strata.metadata_cache import ParquetSchema

        col = ParquetSchema(_column_names=["id", "user.id"]).column(1)
        assert col.path == "user.id"
        assert col.name == "id"

    def test_legacy_leaf_named_rows_are_treated_as_a_miss(self):
        """Rows persisted before this fix hold leaf names; duplicates are the
        signature. They must be re-read rather than trusted — flat-schema rows
        (no duplicates) stay valid."""
        from strata.metadata_cache import _persisted_meta_is_legacy_leaf_named
        from strata.metadata_store import PersistedParquetMeta

        def _meta(names):
            return PersistedParquetMeta(
                arrow_schema_bytes=b"", num_row_groups=1, row_groups=[], column_names=names
            )

        assert _persisted_meta_is_legacy_leaf_named(_meta(["id", "id"])) is True
        assert _persisted_meta_is_legacy_leaf_named(_meta(["id", "user.id"])) is False
        assert _persisted_meta_is_legacy_leaf_named(_meta(["a", "b", "c"])) is False


class TestPreflightSizeRespectsProjection:
    """The pre-flight 413 compares an estimate against ``max_response_bytes``,
    so the estimate has to describe the response the limit governs.

    It was ``total_byte_size`` — the size of the WHOLE row group — regardless
    of the projection. Scanning two columns of a forty-column table was
    estimated as if all forty were read, so a legitimate projected scan was
    rejected as oversized and the only workaround (raising the limit) defeats
    the guard.
    """

    def test_a_projected_scan_is_estimated_far_smaller(self, temp_warehouse, tmp_path):
        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        planner = ReadPlanner(StrataConfig(cache_dir=tmp_path / "c"))
        uri = temp_warehouse["table_uri"]

        full = planner.plan(uri).estimated_bytes
        projected = planner.plan(uri, columns=["id"]).estimated_bytes

        assert projected < full

    def test_the_estimate_still_covers_the_real_response(self, temp_warehouse, tmp_path):
        """Over-estimating is the safe direction for a guard; under-estimating
        lets an oversized response through."""
        from strata.cache import CachedFetcher
        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        config = StrataConfig(cache_dir=tmp_path / "c")
        planner, fetcher = ReadPlanner(config), CachedFetcher(config)

        plan = planner.plan(temp_warehouse["table_uri"], columns=["id"])
        actual = sum(b.nbytes for b in fetcher.execute_plan(plan))

        assert plan.estimated_bytes >= actual

    def test_it_survives_a_restart(self, temp_warehouse, tmp_path):
        """The sizes go through the persisted metadata cache, so a second
        planner over the same cache dir must still see them. Persisted
        metadata is exactly where the column-keying bug in #533 hid."""
        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        config = StrataConfig(cache_dir=tmp_path / "c")
        uri = temp_warehouse["table_uri"]

        cold = ReadPlanner(config).plan(uri, columns=["id"]).estimated_bytes
        warm = ReadPlanner(config).plan(uri, columns=["id"]).estimated_bytes

        assert warm == cold

    def test_a_projection_below_the_limit_is_no_longer_rejected(self, temp_warehouse, tmp_path):
        """The user-visible symptom: the pre-flight 413 fires on
        ``plan.estimated_bytes > max_response_bytes``. With a limit set
        between the projected and unprojected sizes, the projected scan used
        to trip it because it was measured as the whole row group."""
        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        planner = ReadPlanner(StrataConfig(cache_dir=tmp_path / "c"))
        uri = temp_warehouse["table_uri"]

        full = planner.plan(uri).estimated_bytes
        projected = planner.plan(uri, columns=["id"]).estimated_bytes

        # A limit the narrow scan fits under and the wide scan does not.
        limit = (full + projected) // 2
        assert projected <= limit < full

    def test_an_unprojected_scan_is_unchanged(self, temp_warehouse, tmp_path):
        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        plan = ReadPlanner(StrataConfig(cache_dir=tmp_path / "c")).plan(temp_warehouse["table_uri"])

        meta = pq.ParquetFile(plan.tasks[0].file_path).metadata
        assert plan.estimated_bytes == sum(
            meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)
        )


class TestRowGroupEstimateFallsBack:
    """Whole-row-group size is the safe answer whenever per-column sizes
    cannot be trusted, since over-estimating only makes the guard stricter."""

    def _rg(self, sizes):
        from strata.metadata_cache import ColumnChunkMeta, RowGroupMeta

        return RowGroupMeta(
            num_rows=10,
            total_byte_size=1000,
            _columns={
                i: ColumnChunkMeta(is_stats_set=False, statistics=None, total_uncompressed_size=s)
                for i, s in enumerate(sizes)
            },
        )

    def test_sums_only_the_projected_columns(self):
        from strata.planner import _estimate_row_group_bytes

        rg = self._rg([100, 200, 300])
        assert _estimate_row_group_bytes(rg, ["a", "c"], {"a": 0, "b": 1, "c": 2}) == 400

    def test_no_projection_uses_the_whole_row_group(self):
        from strata.planner import _estimate_row_group_bytes

        assert _estimate_row_group_bytes(self._rg([100, 200]), None, {"a": 0, "b": 1}) == 1000

    def test_a_column_outside_the_index_map_falls_back(self):
        """A nested projection: ``_build_column_index_map`` skips dotted paths."""
        from strata.planner import _estimate_row_group_bytes

        rg = self._rg([100, 200])
        assert _estimate_row_group_bytes(rg, ["user.id"], {"a": 0, "b": 1}) == 1000

    def test_a_legacy_entry_without_recorded_sizes_falls_back(self):
        """Cache entries written before the sizes existed report 0, which must
        read as "unknown", not as "this column is free"."""
        from strata.planner import _estimate_row_group_bytes

        rg = self._rg([0, 0])
        assert _estimate_row_group_bytes(rg, ["a"], {"a": 0, "b": 1}) == 1000


class TestPersistedColumnSizes:
    """Sizes ride along in the row-group JSON, so old rows stay readable."""

    def test_round_trips(self, tmp_path):
        from strata.metadata_store import (
            MetadataStore,
            PersistedParquetMeta,
            PersistedRowGroupMeta,
        )

        store = MetadataStore(tmp_path / "meta.sqlite")
        target = tmp_path / "x.parquet"
        target.write_bytes(b"not really parquet, only stat'd")
        meta = PersistedParquetMeta(
            arrow_schema_bytes=b"",
            num_row_groups=1,
            row_groups=[
                PersistedRowGroupMeta(
                    num_rows=5,
                    total_byte_size=900,
                    column_stats={},
                    column_sizes={"a": 100, "b": 200},
                )
            ],
            column_names=["a", "b"],
        )
        store.put_parquet_meta(str(target), meta)

        loaded = store.get_parquet_meta(str(target))
        assert loaded.row_groups[0].column_sizes == {"a": 100, "b": 200}

    def test_a_row_written_without_sizes_still_loads(self, tmp_path):
        import json

        from strata.metadata_store import MetadataStore

        store = MetadataStore(tmp_path / "meta.sqlite")
        target = tmp_path / "legacy.parquet"
        target.write_bytes(b"not really parquet, only stat'd")
        legacy = json.dumps([{"num_rows": 5, "total_byte_size": 900, "column_stats": {}}])
        conn = store._get_conn()
        conn.execute(
            """INSERT INTO parquet_meta
               (file_path, schema_ipc, num_row_groups, row_groups_json,
                column_names_json, file_mtime, file_size)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(target), b"", 1, legacy, json.dumps(["a"]), None, None),
        )
        conn.commit()

        loaded = store.get_parquet_meta(str(target))
        assert loaded.row_groups[0].column_sizes == {}


class TestNoArgLookupKeepsTheConfiguredStore:
    """``get_metadata_store()`` must mean "the store in use", not "the home one".

    ``/health/ready`` and the metadata routes call it with no argument. That
    used to resolve to ``~/.strata/cache``, and because the path-mismatch
    branch rebuilds the singleton for a different path, the readiness probe
    swapped the global store out from under a server configured with any other
    cache_dir — on every probe, so the store thrashed between the two while the
    readiness check reported on a database nothing was serving from.
    """

    @pytest.fixture(autouse=True)
    def _isolate_singleton(self, monkeypatch, tmp_path):
        import strata.metadata_cache as metadata_cache

        monkeypatch.setattr(metadata_cache, "_metadata_store", None)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
        yield
        metadata_cache._metadata_store = None

    def test_no_arg_returns_the_configured_store(self, tmp_path):
        from strata.metadata_cache import get_metadata_store

        configured = tmp_path / "configured"
        served = get_metadata_store(configured)

        assert get_metadata_store() is served
        # And a later configured lookup is still the same object, so the
        # singleton never churned.
        assert get_metadata_store(configured) is served

    def test_no_arg_does_not_create_a_home_cache_dir(self, tmp_path):
        from strata.metadata_cache import get_metadata_store

        get_metadata_store(tmp_path / "configured")
        get_metadata_store()

        assert not (tmp_path / "home" / ".strata" / "cache").exists()

    def test_no_arg_still_falls_back_when_nothing_is_initialized(self, tmp_path):
        """Personal-mode and CLI callers with no server-initialized store keep
        the home default."""
        from strata.metadata_cache import get_metadata_store

        store = get_metadata_store()

        assert store.db_path == tmp_path / "home" / ".strata" / "cache" / "metadata.sqlite"
