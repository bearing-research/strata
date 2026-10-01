"""Tests for cache granularity configuration."""

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

from strata.cache import CachedFetcher, DiskCache
from strata.config import StrataConfig
from strata.types import CacheGranularity, CacheKey, TableIdentity, Task


@pytest.fixture
def sample_batch():
    """A sample record batch."""
    return pa.RecordBatch.from_pydict(
        {
            "id": [1, 2, 3],
            "value": [1.0, 2.0, 3.0],
            "name": ["a", "b", "c"],
        }
    )


@pytest.fixture
def table_identity():
    """A sample table identity."""
    return TableIdentity.from_table_id("test_db.events")


class TestCacheGranularityConfig:
    def test_default_granularity_is_row_group_projection(self, tmp_path):
        config = StrataConfig(cache_dir=tmp_path / "cache")
        assert config.cache_granularity == CacheGranularity.ROW_GROUP_PROJECTION

    def test_can_configure_row_group_granularity(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            cache_granularity=CacheGranularity.ROW_GROUP,
        )
        assert config.cache_granularity == CacheGranularity.ROW_GROUP


class TestCacheKeyGranularity:
    def test_row_group_projection_includes_projection(self, table_identity):
        key1 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="abc123",
        )
        key2 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="def456",
        )

        hex1 = key1.to_hex(CacheGranularity.ROW_GROUP_PROJECTION)
        hex2 = key2.to_hex(CacheGranularity.ROW_GROUP_PROJECTION)
        assert hex1 != hex2

    def test_row_group_ignores_projection(self, table_identity):
        key1 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="abc123",
        )
        key2 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="def456",
        )

        hex1 = key1.to_hex(CacheGranularity.ROW_GROUP)
        hex2 = key2.to_hex(CacheGranularity.ROW_GROUP)
        assert hex1 == hex2

    def test_row_group_still_differentiates_row_groups(self, table_identity):
        key1 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="abc123",
        )
        key2 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=1,
            projection_fingerprint="abc123",
        )

        # Row groups always split the key, even at ROW_GROUP granularity.
        hex1 = key1.to_hex(CacheGranularity.ROW_GROUP)
        hex2 = key2.to_hex(CacheGranularity.ROW_GROUP)
        assert hex1 != hex2


class TestDiskCacheGranularity:
    def test_row_group_projection_caches_separately(self, tmp_path, sample_batch, table_identity):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            cache_granularity=CacheGranularity.ROW_GROUP_PROJECTION,
        )
        cache = DiskCache(config)

        key1 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="proj1",
        )
        key2 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="proj2",
        )

        cache.put(key1, sample_batch)

        assert cache.get(key1) is not None
        assert cache.get(key2) is None

    def test_row_group_shares_cache_across_projections(
        self, tmp_path, sample_batch, table_identity
    ):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            cache_granularity=CacheGranularity.ROW_GROUP,
        )
        cache = DiskCache(config)

        key1 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="proj1",
        )
        key2 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="proj2",
        )

        cache.put(key1, sample_batch)

        assert cache.get(key1) is not None
        assert cache.get(key2) is not None

    def test_row_group_still_separates_different_row_groups(
        self, tmp_path, sample_batch, table_identity
    ):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            cache_granularity=CacheGranularity.ROW_GROUP,
        )
        cache = DiskCache(config)

        key1 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="proj1",
        )
        key2 = CacheKey(
            tenant_id="_default",
            table_identity=table_identity,
            snapshot_id=123,
            file_path="/data/file.parquet",
            row_group_id=1,
            projection_fingerprint="proj1",
        )

        cache.put(key1, sample_batch)

        assert cache.get(key1) is not None
        assert cache.get(key2) is None

    def test_row_group_cached_fetcher_refetches_full_row_group_for_broader_projection(
        self, tmp_path
    ):
        """ROW_GROUP mode caches full row groups instead of pinning the first projection."""
        parquet_path = tmp_path / "data.parquet"
        pq.write_table(
            pa.table(
                {
                    "id": [1, 2],
                    "value": [10.0, 20.0],
                    "name": ["a", "b"],
                }
            ),
            parquet_path,
        )

        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            cache_granularity=CacheGranularity.ROW_GROUP,
        )
        fetcher = CachedFetcher(config)
        table_identity = TableIdentity.from_table_id("test_db.events")

        subset_task = Task(
            file_path=str(parquet_path),
            row_group_id=0,
            cache_key=CacheKey(
                tenant_id="_default",
                table_identity=table_identity,
                snapshot_id=123,
                file_path=str(parquet_path),
                row_group_id=0,
                projection_fingerprint=CacheKey.compute_projection_fingerprint(["id"]),
            ),
            num_rows=2,
            columns=["id"],
        )
        subset_batch = fetcher.fetch(subset_task)
        assert subset_batch.schema.names == ["id"]

        full_task = Task(
            file_path=str(parquet_path),
            row_group_id=0,
            cache_key=CacheKey(
                tenant_id="_default",
                table_identity=table_identity,
                snapshot_id=123,
                file_path=str(parquet_path),
                row_group_id=0,
                projection_fingerprint=CacheKey.compute_projection_fingerprint(None),
            ),
            num_rows=2,
            columns=None,
        )
        full_batch = fetcher.fetch(full_task)

        assert full_task.cached is True
        assert full_batch.schema.names == ["id", "value", "name"]

        projected_stream_task = Task(
            file_path=str(parquet_path),
            row_group_id=0,
            cache_key=CacheKey(
                tenant_id="_default",
                table_identity=table_identity,
                snapshot_id=123,
                file_path=str(parquet_path),
                row_group_id=0,
                projection_fingerprint=CacheKey.compute_projection_fingerprint(["id"]),
            ),
            num_rows=2,
            columns=["id"],
        )
        stream_bytes = fetcher.fetch_as_stream_bytes(projected_stream_task)
        streamed_batch = list(ipc.open_stream(pa.BufferReader(stream_bytes)))[0]
        assert projected_stream_task.cached is True
        assert streamed_batch.schema.names == ["id"]
