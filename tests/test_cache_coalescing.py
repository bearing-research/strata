"""Concurrent cache misses on one row group share a single storage read.

Scans of one table that miss at the same time used to read each row group once
per scan: sixteen concurrent cold scans fetched 37 row groups 218 times.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

from strata.cache import CachedFetcher
from strata.config import StrataConfig
from strata.types import CacheKey, TableIdentity, Task


class _CountingFetcher:
    """A storage fetcher that counts reads and holds each one open briefly."""

    def __init__(self, fail_first: bool = False):
        self.calls = 0
        self._fail_first = fail_first
        self._lock = threading.Lock()

    def fetch(self, task: Task) -> pa.RecordBatch:
        with self._lock:
            self.calls += 1
            call = self.calls
        time.sleep(0.05)  # the window concurrent misses land in
        if self._fail_first and call == 1:
            raise OSError("connection reset")
        return pa.RecordBatch.from_pydict({"id": [task.row_group_id] * 3})


def _task(row_group_id: int = 0) -> Task:
    key = CacheKey(
        tenant_id="_default",
        table_identity=TableIdentity(catalog="strata", namespace="db", table="events"),
        snapshot_id=1,
        file_path="/data/file.parquet",
        row_group_id=row_group_id,
        projection_fingerprint="all",
    )
    return Task(
        file_path="/data/file.parquet", row_group_id=row_group_id, cache_key=key, num_rows=3
    )


def _fetcher(tmp_path, storage) -> CachedFetcher:
    return CachedFetcher(StrataConfig(cache_dir=tmp_path / "cache"), fetcher=storage)


def _concurrently(n: int, fn):
    barrier = threading.Barrier(n)

    def run(i: int):
        barrier.wait(timeout=10)
        return fn(i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(run, range(n)))


def test_concurrent_misses_on_one_row_group_read_storage_once(tmp_path):
    storage = _CountingFetcher()
    fetcher = _fetcher(tmp_path, storage)
    tasks = [_task() for _ in range(16)]

    batches = _concurrently(16, lambda i: fetcher.fetch(tasks[i]))

    assert storage.calls == 1
    assert all(b.equals(batches[0]) for b in batches)
    # One task read storage; the rest were answered from its read.
    assert sorted(t.cached for t in tasks) == [False] + [True] * 15
    assert not fetcher._flights


def test_different_row_groups_are_read_separately(tmp_path):
    storage = _CountingFetcher()
    fetcher = _fetcher(tmp_path, storage)

    batches = _concurrently(16, lambda i: fetcher.fetch(_task(row_group_id=i % 4)))

    assert storage.calls == 4
    assert sorted({b.column("id")[0].as_py() for b in batches}) == [0, 1, 2, 3]


def test_a_failed_shared_read_lets_the_waiters_read_for_themselves(tmp_path):
    """One transient error fails the read that hit it, not every scan waiting."""
    storage = _CountingFetcher(fail_first=True)
    fetcher = _fetcher(tmp_path, storage)
    outcomes: list[str] = []
    lock = threading.Lock()

    def read(_i: int):
        try:
            fetcher.fetch(_task())
            result = "ok"
        except OSError:
            result = "failed"
        with lock:
            outcomes.append(result)

    _concurrently(4, read)

    assert sorted(outcomes) == ["failed", "ok", "ok", "ok"]
    assert not fetcher._flights


def test_a_miss_after_a_read_landed_is_served_from_the_cache(tmp_path):
    storage = _CountingFetcher()
    fetcher = _fetcher(tmp_path, storage)
    fetcher.fetch(_task())

    task = _task()
    fetcher.fetch(task)

    assert storage.calls == 1
    assert task.cached is True


@pytest.mark.parametrize("n", [2, 32])
def test_no_flight_outlives_its_read(tmp_path, n):
    fetcher = _fetcher(tmp_path, _CountingFetcher())
    _concurrently(n, lambda i: fetcher.fetch(_task(row_group_id=i % 3)))
    assert not fetcher._flights
