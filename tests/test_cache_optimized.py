"""Tests for optimized cache methods and Rust acceleration integration."""

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from strata import fast_io
from strata.cache import CACHE_META_EXTENSION, CachedFetcher, DiskCache
from strata.config import StrataConfig
from strata.types import CacheKey, TableIdentity


@pytest.fixture
def strata_config(tmp_path):
    """Create a test configuration."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return StrataConfig(cache_dir=cache_dir)


@pytest.fixture
def sample_batch():
    """Create a sample record batch."""
    return pa.RecordBatch.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "name": ["a", "b", "c", "d", "e"],
        }
    )


@pytest.fixture
def cache_key():
    """Create a sample cache key."""
    identity = TableIdentity.from_table_id("test_db.events")
    return CacheKey(
        tenant_id="_default",
        table_identity=identity,
        snapshot_id=123,
        file_path="/data/file.parquet",
        row_group_id=0,
        projection_fingerprint="abc123",
    )


class TestDiskCacheGetAsStreamBytes:
    """Tests for DiskCache.get_as_stream_bytes() method."""

    def test_returns_none_for_missing_key(self, strata_config, cache_key):
        """Test that get_as_stream_bytes returns None for missing keys."""
        cache = DiskCache(strata_config)

        result = cache.get_as_stream_bytes(cache_key)

        assert result is None

    def test_returns_stream_bytes_for_cached_data(self, strata_config, cache_key, sample_batch):
        """Test that get_as_stream_bytes returns valid stream bytes for cached data."""
        cache = DiskCache(strata_config)

        # Store data in cache
        cache.put(cache_key, sample_batch)

        # Get as stream bytes
        stream_bytes = cache.get_as_stream_bytes(cache_key)

        # Should return bytes
        assert isinstance(stream_bytes, bytes)
        assert len(stream_bytes) > 0

        # Should be valid Arrow IPC stream format
        reader = ipc.open_stream(pa.BufferReader(stream_bytes))
        batches = list(reader)
        assert len(batches) == 1
        assert batches[0].num_rows == sample_batch.num_rows

    def test_stream_bytes_data_matches_original(self, strata_config, cache_key, sample_batch):
        """Test that stream bytes contain the same data as the original batch."""
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)

        stream_bytes = cache.get_as_stream_bytes(cache_key)

        # Parse and verify data
        reader = ipc.open_stream(pa.BufferReader(stream_bytes))
        result_batch = list(reader)[0]

        assert result_batch.column("id").to_pylist() == [1, 2, 3, 4, 5]
        assert result_batch.column("value").to_pylist() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert result_batch.column("name").to_pylist() == ["a", "b", "c", "d", "e"]


class TestDiskCacheGetPath:
    """Tests for DiskCache.get_path() method."""

    def test_returns_none_for_missing_key(self, strata_config, cache_key):
        """Test that get_path returns None for missing keys."""
        cache = DiskCache(strata_config)

        result = cache.get_path(cache_key)

        assert result is None

    def test_returns_path_for_cached_data(self, strata_config, cache_key, sample_batch):
        """Test that get_path returns a valid path for cached data."""
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)

        path = cache.get_path(cache_key)

        assert path is not None
        assert path.exists()
        assert path.suffix == ".arrowstream"


class TestDiskCacheStatsAndCleanup:
    """Tests for cache stats and corruption cleanup behavior."""

    def test_stats_use_actual_stream_file_size(self, strata_config, cache_key, sample_batch):
        """Reported cache size should match the stored Arrow stream bytes."""
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)

        path = cache.get_path(cache_key)
        assert path is not None

        stats = cache.get_stats()

        assert stats.total_entries == 1
        assert stats.total_size_bytes == path.stat().st_size
        assert stats.total_size_bytes == cache.get_size_bytes()

    def test_corrupted_data_removes_sidecar_and_disappears_from_stats(
        self, strata_config, cache_key, sample_batch
    ):
        """Corrupted cache reads should remove both data and metadata files."""
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)

        path = cache.get_path(cache_key)
        assert path is not None
        meta_path = path.with_suffix(CACHE_META_EXTENSION)
        assert meta_path.exists()

        path.write_bytes(b"CORRUPTED DATA - NOT VALID ARROW IPC")

        assert cache.get(cache_key) is None
        assert not path.exists()
        assert not meta_path.exists()
        assert cache.get_stats().total_entries == 0


class TestCachedFetcherFetchAsStreamBytes:
    """Tests for CachedFetcher.fetch_as_stream_bytes() method."""

    def test_returns_stream_bytes_for_cache_hit(self, strata_config, cache_key, sample_batch):
        """Test that fetch_as_stream_bytes returns stream bytes for cache hits."""
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)

        fetcher = CachedFetcher(strata_config, cache=cache)

        # Create a mock task
        from strata.types import Task

        task = Task(
            file_path="/data/file.parquet",
            row_group_id=0,
            columns=None,
            cache_key=cache_key,
            num_rows=sample_batch.num_rows,
        )

        # Fetch as stream bytes
        stream_bytes = fetcher.fetch_as_stream_bytes(task)

        # Should return valid stream bytes
        assert isinstance(stream_bytes, bytes)
        assert len(stream_bytes) > 0

        # Task should be marked as cached
        assert task.cached is True
        assert task.bytes_read > 0

        # Should be valid Arrow IPC stream
        reader = ipc.open_stream(pa.BufferReader(stream_bytes))
        batches = list(reader)
        assert len(batches) == 1
        assert batches[0].num_rows == sample_batch.num_rows


class TestRustAccelerationIntegration:
    """Tests for Rust acceleration integration with cache."""

    def test_rust_path_produces_valid_output(self, strata_config, cache_key, sample_batch):
        """Test that Rust acceleration produces valid output when available."""
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)

        # Get stream bytes (uses Rust if available)
        stream_bytes = cache.get_as_stream_bytes(cache_key)

        # Should work regardless of Rust availability
        assert stream_bytes is not None

        # Verify the data is correct
        reader = ipc.open_stream(pa.BufferReader(stream_bytes))
        result_batch = list(reader)[0]

        assert result_batch.num_rows == sample_batch.num_rows
        assert result_batch.column("id").to_pylist() == sample_batch.column("id").to_pylist()

    def test_rust_availability_is_reported(self):
        """Test that Rust availability is correctly reported."""
        # This should not raise
        is_available = fast_io.is_rust_available()
        assert isinstance(is_available, bool)

        # If CI has Rust, it should be available
        # This is informational - test passes either way
        print(f"Rust acceleration available: {is_available}")


class TestMmapFileReading:
    """Tests for memory-mapped file reading."""

    def test_read_file_mmap_returns_bytes(self, tmp_path):
        """Test that read_file_mmap returns file contents as bytes."""
        test_file = tmp_path / "test.txt"
        test_data = b"Hello, memory-mapped world!"
        test_file.write_bytes(test_data)

        result = fast_io.read_file_mmap(str(test_file))

        assert result == test_data

    def test_read_file_mmap_with_binary_data(self, tmp_path):
        """Test that read_file_mmap handles binary data correctly."""
        test_file = tmp_path / "binary.bin"
        # Binary data with null bytes and various byte values
        test_data = bytes(range(256)) * 10
        test_file.write_bytes(test_data)

        result = fast_io.read_file_mmap(str(test_file))

        assert result == test_data
        assert len(result) == 2560

    def test_read_file_mmap_large_file(self, tmp_path):
        """Test that read_file_mmap handles larger files."""
        test_file = tmp_path / "large.bin"
        # 1 MB of data
        test_data = b"x" * (1024 * 1024)
        test_file.write_bytes(test_data)

        result = fast_io.read_file_mmap(str(test_file))

        assert result == test_data
        assert len(result) == 1024 * 1024

    def test_read_file_mmap_arrow_stream(self, strata_config, cache_key, sample_batch):
        """Test that read_file_mmap correctly reads Arrow IPC stream files."""
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)

        # Get the cache file path
        path = cache.get_path(cache_key)
        assert path is not None

        # Read using mmap
        stream_bytes = fast_io.read_file_mmap(str(path))

        # Verify it's valid Arrow IPC data
        reader = ipc.open_stream(pa.BufferReader(stream_bytes))
        result_batch = list(reader)[0]

        assert result_batch.num_rows == sample_batch.num_rows
        assert result_batch.column("id").to_pylist() == sample_batch.column("id").to_pylist()

    def test_read_file_mmap_matches_read_bytes(self, strata_config, cache_key, sample_batch):
        """Test that read_file_mmap produces same result as Path.read_bytes()."""
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)

        path = cache.get_path(cache_key)
        assert path is not None

        # Read both ways
        mmap_result = fast_io.read_file_mmap(str(path))
        read_bytes_result = path.read_bytes()

        # Results should be identical
        assert mmap_result == read_bytes_result

    def test_read_file_mmap_nonexistent_file(self, tmp_path):
        """Test that read_file_mmap raises on nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.txt"

        # Should raise an exception (IOError from Rust or FileNotFoundError from Python)
        with pytest.raises(Exception):
            fast_io.read_file_mmap(str(nonexistent))


class TestDamagedEntriesSelfHeal:
    """A damaged entry on the zero-parse path used to poison a key forever.

    ``get`` parses Arrow and drops an entry that fails, so it recovers on its
    own. ``get_as_stream_bytes`` deliberately skips parsing — that is the point
    of the hot path — and so noticed nothing: the damaged bytes went straight
    to the network, and because cache keys are immutable snapshot IDs that are
    never invalidated, the *same* damaged entry was served again on every later
    request for that row group. There is no expiry to eventually clear it.

    The stream's fixed head and end-of-stream markers make the check free: the
    bytes are already in memory.
    """

    @pytest.mark.parametrize(
        "label,damage",
        [
            ("truncated", lambda raw: raw[: len(raw) // 2]),
            ("zeroed", lambda raw: b"\x00" * len(raw)),
            ("tail_lost", lambda raw: raw[:64] + b"\x00" * (len(raw) - 64)),
            ("emptied", lambda raw: b""),
        ],
    )
    def test_damaged_entry_is_a_miss_and_is_dropped(
        self, strata_config, sample_batch, cache_key, label, damage
    ):
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)
        path = cache._key_path(cache_key)
        path.write_bytes(damage(path.read_bytes()))

        assert cache.get_as_stream_bytes(cache_key) is None, label
        # Dropped, so the next fetch repopulates it instead of re-serving damage.
        assert not path.exists(), label
        assert not path.with_suffix(CACHE_META_EXTENSION).exists(), label

    def test_an_intact_entry_round_trips(self, strata_config, sample_batch, cache_key):
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)

        raw = cache.get_as_stream_bytes(cache_key)
        assert raw is not None
        assert ipc.open_stream(pa.py_buffer(raw)).read_all().num_rows == sample_batch.num_rows

    def test_writes_are_flushed_before_the_rename(
        self, strata_config, sample_batch, cache_key, monkeypatch
    ):
        # os.replace is atomic for observers but not durable: without the flush
        # a crash can apply the rename and lose the data blocks behind it.
        import strata.cache as cache_module

        synced = []
        real_fsync = cache_module.os.fsync
        monkeypatch.setattr(
            cache_module.os,
            "fsync",
            lambda fd: (synced.append(fd), real_fsync(fd))[1],
        )

        DiskCache(strata_config).put(cache_key, sample_batch)
        # The data file and its metadata sidecar.
        assert len(synced) == 2


class TestClearPreservesTheMetadataDatabase:
    """``clear`` documents itself as preserving ``metadata.sqlite``, but it
    matched that name exactly and so deleted the ``-wal`` and ``-shm``
    sidecars beside it.

    Under WAL mode — which ``MetadataStore`` enables — those two files are
    part of the database, not scratch next to it. The ``-wal`` holds committed
    transactions not yet checkpointed into the main file, and the ``-shm`` is
    the shared index into it that every live connection maps (the store keeps
    one per thread). Removing them from under an open database can discard
    committed metadata and can leave a connection raising SQLITE_IOERR.

    SQLite checkpoints and removes the sidecars itself when the *last*
    connection closes, so the live connection here is what makes the
    difference observable — and it is also the state a running server is
    always in.
    """

    def _live_store(self, cache_dir):
        import sqlite3

        from strata.metadata_store import MetadataStore

        db = cache_dir / "metadata.sqlite"
        store = MetadataStore(db)
        store.put_manifest("cat", "db.t", 1, [("a.parquet", "/w/a.parquet")])

        held = sqlite3.connect(str(db))
        held.execute("PRAGMA journal_mode=WAL")
        held.execute("SELECT count(*) FROM manifest_cache").fetchone()
        return store, held

    def test_the_wal_sidecars_survive_a_clear(self, strata_config):
        cache_dir = strata_config.cache_dir
        store, held = self._live_store(cache_dir)
        try:
            assert {"metadata.sqlite-wal", "metadata.sqlite-shm"} <= {
                p.name for p in cache_dir.iterdir()
            }

            DiskCache(strata_config).clear()

            names = {p.name for p in cache_dir.iterdir()}
            assert "metadata.sqlite" in names
            assert "metadata.sqlite-wal" in names
            assert "metadata.sqlite-shm" in names
        finally:
            held.close()

    def test_the_metadata_is_still_readable_after_a_clear(self, strata_config):
        cache_dir = strata_config.cache_dir
        store, held = self._live_store(cache_dir)
        try:
            DiskCache(strata_config).clear()
            assert store.get_manifest("cat", "db.t", 1) is not None
        finally:
            held.close()

    def test_clear_still_removes_cached_row_groups(self, strata_config, sample_batch, cache_key):
        """The preservation must not cost ``clear`` its actual job."""
        cache = DiskCache(strata_config)
        cache.put(cache_key, sample_batch)
        assert cache.get_as_stream_bytes(cache_key) is not None

        cache.clear()
        assert cache.get_as_stream_bytes(cache_key) is None
