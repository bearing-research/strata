"""Concurrency tests for the Fetcher's per-file cache.

One Fetcher serves the whole fetch pool, which reads row groups of one file in parallel. It caches
parsed footers and opens a handle per read: pyarrow 25 segfaults on a shared ParquetFile.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata.fetcher import PyArrowFetcher
from strata.types import Task


def _write(tmp_path, name: str):
    fp = tmp_path / f"{name}.parquet"
    pq.write_table(pa.table({"id": pa.array(range(10))}), fp)
    return str(fp)


def _task(file_path: str) -> Task:
    return Task(
        file_path=file_path,
        row_group_id=0,
        cache_key=None,  # type: ignore[arg-type] - unused by PyArrowFetcher.fetch
        num_rows=10,
    )


def test_eviction_does_not_close_a_handle_another_thread_is_reading(tmp_path):
    """Thread B evicts a handle thread A is reading, mid-stream after the 200 is sent."""
    # Cache of 1 makes every new file evict the previous one.
    fetcher = PyArrowFetcher(max_file_cache_size=1)
    paths = [_write(tmp_path, f"f{i}") for i in range(8)]

    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def read(path: str):
        try:
            barrier.wait(timeout=10)
            for _ in range(25):
                fetcher.fetch(_task(path))
        except BaseException as exc:  # noqa: BLE001 - recorded and re-raised below
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(read, paths))

    assert not errors, f"concurrent fetch raised: {errors[:3]}"


def test_handle_cache_stays_within_its_bound(tmp_path):
    fetcher = PyArrowFetcher(max_file_cache_size=3)
    for i in range(10):
        fetcher.fetch(_task(_write(tmp_path, f"g{i}")))
    assert len(fetcher._file_cache) <= 3


def test_concurrent_reads_of_one_path_keep_one_footer(tmp_path):
    """Threads reading one path at once leave a single cached footer."""
    fetcher = PyArrowFetcher(max_file_cache_size=8)
    path = _write(tmp_path, "shared")
    barrier = threading.Barrier(6)

    def read(_i: int):
        barrier.wait(timeout=10)
        fetcher.fetch(_task(path))

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(read, range(6)))

    assert list(fetcher._file_cache) == [path]


def test_close_forgets_the_footers(tmp_path):
    fetcher = PyArrowFetcher(max_file_cache_size=4)
    fetcher.fetch(_task(_write(tmp_path, "h0")))
    assert fetcher._file_cache
    fetcher.close()
    assert not fetcher._file_cache


class _RecordingParquetFile:
    """Stands in for a ParquetFile and records how it is used."""

    created: list[_RecordingParquetFile] = []
    fail_reads = False

    def __init__(self, path, filesystem=None, metadata=None):
        self.path = path
        self.given_metadata = metadata
        self.metadata = metadata or object()
        self.closed = False
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        _RecordingParquetFile.created.append(self)

    def read_row_group(self, i, columns=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.005)  # widen the window two readers would share
            if _RecordingParquetFile.fail_reads:
                raise OSError("connection reset")
            return pa.table({"id": pa.array(range(3))})
        finally:
            with self._lock:
                self.active -= 1

    def close(self):
        self.closed = True


def _recording(monkeypatch, *, fail_reads=False):
    _RecordingParquetFile.created = []
    _RecordingParquetFile.fail_reads = fail_reads
    monkeypatch.setattr(pq, "ParquetFile", _RecordingParquetFile)


def test_no_two_threads_read_one_handle_at_once(tmp_path, monkeypatch):
    """pyarrow 25 segfaults when two threads read one ParquetFile."""
    _recording(monkeypatch)
    fetcher = PyArrowFetcher(max_file_cache_size=8)
    path = str(tmp_path / "hot.parquet")
    barrier = threading.Barrier(12)

    def read(_i: int):
        barrier.wait(timeout=10)
        for _ in range(5):
            fetcher.fetch(_task(path))

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(read, range(12)))

    # The reads did overlap in time: twelve threads, sixty reads, one handle each.
    assert len(_RecordingParquetFile.created) == 60
    assert all(pf.max_active == 1 for pf in _RecordingParquetFile.created)
    assert all(pf.closed for pf in _RecordingParquetFile.created)


def test_the_footer_is_parsed_once_per_file(tmp_path, monkeypatch):
    _recording(monkeypatch)
    fetcher = PyArrowFetcher(max_file_cache_size=8)
    path = str(tmp_path / "one.parquet")

    for _ in range(3):
        fetcher.fetch(_task(path))

    assert len(_RecordingParquetFile.created) == 3  # a handle per read
    first, *later = _RecordingParquetFile.created
    assert first.given_metadata is None
    assert all(pf.given_metadata is first.metadata for pf in later)


def test_a_failed_read_still_closes_its_handle(tmp_path, monkeypatch):
    """Every read opens and closes its own handle, even when the read fails."""
    _recording(monkeypatch, fail_reads=True)
    fetcher = PyArrowFetcher(max_file_cache_size=8)

    with pytest.raises(OSError):
        fetcher.fetch(_task(str(tmp_path / "flaky.parquet")))

    assert [pf.closed for pf in _RecordingParquetFile.created] == [True]
