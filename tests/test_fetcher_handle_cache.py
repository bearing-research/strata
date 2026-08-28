"""Concurrency tests for the Fetcher's ParquetFile handle cache.

One ``Fetcher`` is shared by the whole fetch thread pool
(``max_fetch_workers``, 32 by default), so its LRU handle cache is mutated
concurrently. It previously had no lock, and evicting a handle *closed* it —
even while another thread was inside ``read_row_group`` with that same handle.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

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
    """The original failure: thread A is inside read_row_group on a handle that
    thread B evicts and closes, so A dies with 'I/O operation on closed file'
    mid-stream — after the 200 has already gone out."""
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


def test_concurrent_opens_of_one_path_keep_a_single_handle(tmp_path):
    """Two threads racing to open the same path must converge on one cached
    handle rather than leaking the loser."""
    fetcher = PyArrowFetcher(max_file_cache_size=8)
    path = _write(tmp_path, "shared")
    barrier = threading.Barrier(6)

    def read(_i: int):
        barrier.wait(timeout=10)
        fetcher.fetch(_task(path))

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(read, range(6)))

    assert list(fetcher._file_cache) == [path]


def test_close_releases_handles(tmp_path):
    fetcher = PyArrowFetcher(max_file_cache_size=4)
    fetcher.fetch(_task(_write(tmp_path, "h0")))
    assert fetcher._file_cache
    fetcher.close()
    assert not fetcher._file_cache
