"""Regression tests for subprocess line-limit handling (code-review fix).

asyncio's StreamReader defaults to a 64 KiB line limit, and readline()
raises ValueError past it. Harness frames and warm-pool result lines
legitimately embed full stdout captures and base64 display payloads, so
the notebook's subprocess readers must (a) carry a far larger limit and
(b) fail a run cleanly, not crash the caller, when even that is exceeded.
Also covers the warm-pool timeout contract: a cell timeout must surface
as TimeoutError, not silently trigger a cold re-execution.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from strata.notebook.pool import PooledCellExecutor
from strata.notebook.process_tree import SUBPROCESS_LINE_LIMIT


@pytest.mark.asyncio
async def test_stream_reader_limit_accepts_large_frames():
    """A 1 MiB frame line (a cached PNG display) must survive readline —
    it raised ValueError under the 64 KiB default."""
    reader = asyncio.StreamReader(limit=SUBPROCESS_LINE_LIMIT)
    payload = b"x" * (1024 * 1024) + b"\n"
    reader.feed_data(payload)
    reader.feed_eof()
    line = await reader.readline()
    assert len(line) == len(payload)


class _FakeStdin:
    def write(self, data):
        return None

    async def drain(self):
        return None


class _FakeProcess:
    def __init__(self, stdout):
        self.stdin = _FakeStdin()
        self.stdout = stdout
        self.pid = 12345
        self.returncode = 0


class _FakePool:
    def __init__(self, stdout):
        self._stdout = stdout
        self.replaced = False

    async def acquire(self):
        return SimpleNamespace(process=_FakeProcess(self._stdout))

    async def release_and_replace(self, warm_proc):
        self.replaced = True


@pytest.mark.asyncio
async def test_pool_timeout_raises_instead_of_cold_fallback(tmp_path):
    """A cell exceeding its timeout in the warm worker must raise
    TimeoutError (surfaced as a cell timeout by the executor) — returning
    None meant the caller re-ran the whole cell body cold: paying the
    timeout twice and repeating side effects."""

    class _NeverStdout:
        async def readline(self):
            await asyncio.sleep(3600)

    pool = _FakePool(_NeverStdout())
    with pytest.raises(TimeoutError):
        await PooledCellExecutor.execute_with_pool(
            pool, tmp_path / "manifest.json", tmp_path, timeout_seconds=0.05
        )
    assert pool.replaced  # the hung worker was still recycled


@pytest.mark.asyncio
async def test_batch_service_loop_survives_oversized_frame(tmp_path):
    """A frame exceeding even the raised limit must end the batch as
    subprocess_died — previously the ValueError propagated uncaught,
    aborting run-all and leaking the harness subprocess."""
    from strata.notebook.executor import CellExecutor

    class _OverflowReader:
        async def readline(self):
            raise ValueError("Separator is found, but chunk is longer than limit")

    end_reason, failed_cell_id = await CellExecutor._batch_service_loop(
        SimpleNamespace(),  # self is untouched before the overflow abort
        _OverflowReader(),
        lambda payload: None,
        {},
        tmp_path,
        use_cache=True,
    )
    assert end_reason == "subprocess_died"
    assert failed_cell_id is None
