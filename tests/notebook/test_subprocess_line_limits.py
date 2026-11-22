"""Subprocess line-limit handling.

asyncio's StreamReader raises ValueError past 64 KiB, but harness frames and warm-pool result lines
embed stdout captures and base64 displays. Readers need a far larger limit, and must fail the run
cleanly when even that is exceeded. Also: a warm-pool cell timeout raises TimeoutError instead of a
cold re-execution, and a warm run whose hand-over raises still releases its worker.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from strata.notebook.pool import PooledCellExecutor
from strata.notebook.process_tree import SUBPROCESS_LINE_LIMIT
from strata.notebook.run_channel import EVENT_KEY


@pytest.mark.asyncio
async def test_stream_reader_limit_accepts_large_frames():
    """A 1 MiB frame line (a cached PNG display) must survive readline."""
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
    def __init__(self, stdout, returncode=0):
        self.stdin = _FakeStdin()
        self.stdout = stdout
        self.pid = 12345
        self.returncode = returncode

    async def wait(self):
        return self.returncode


class _FakePool:
    def __init__(self, stdout, returncode=0):
        self._stdout = stdout
        self._returncode = returncode
        self.releases = 0

    async def acquire(self):
        return SimpleNamespace(process=_FakeProcess(self._stdout, self._returncode))

    async def release_and_replace(self, warm_proc):
        self.releases += 1


@pytest.mark.asyncio
async def test_pool_timeout_raises_instead_of_cold_fallback(tmp_path):
    """Returning None would rerun the whole cell cold: the timeout paid twice and side effects
    repeated.
    """

    class _NeverStdout:
        async def readline(self):
            await asyncio.sleep(3600)

    pool = _FakePool(_NeverStdout())
    with pytest.raises(TimeoutError):
        await PooledCellExecutor.execute_with_pool(
            pool, tmp_path / "manifest.json", tmp_path, timeout_seconds=0.05
        )
    assert pool.releases == 1  # the hung worker was still recycled


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [IndexError("a stat that does not parse"), KeyboardInterrupt()])
async def test_a_hand_over_that_raises_still_releases_the_worker(tmp_path, monkeypatch, error):
    """Its sweep can fail (a /proc it cannot read), or be interrupted: the run's error, and
    the pool keeps its slot."""

    def refused(path, user):
        raise error

    monkeypatch.setattr("strata.notebook.pool.hand_over", refused)
    pool = _FakePool(None)

    with pytest.raises(type(error)):
        await PooledCellExecutor.execute_with_pool(
            pool, tmp_path / "manifest.json", tmp_path, timeout_seconds=1
        )
    assert pool.releases == 1


@pytest.mark.asyncio
async def test_batch_service_loop_survives_oversized_frame(tmp_path):
    """A frame past even the raised limit ends the batch as subprocess_died instead of aborting
    run-all and leaking the harness subprocess.
    """
    from strata.notebook.cell_paths import batch

    class _OverflowReader:
        async def readline(self):
            raise ValueError("Separator is found, but chunk is longer than limit")

    end_reason, failed_cell_id = await batch.batch_service_loop(
        SimpleNamespace(),  # the host and context are untouched before the overflow abort
        SimpleNamespace(),
        _OverflowReader(),
        lambda payload: None,
        {},
        tmp_path,
        executed_sources={},
        provenance={},
        use_cache=True,
    )
    assert end_reason == "subprocess_died"
    assert failed_cell_id is None


def _lines(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader(limit=SUBPROCESS_LINE_LIMIT)
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def _pooled(tmp_path, data: bytes, returncode: int = 0, on_event=None):
    return await PooledCellExecutor.execute_with_pool(
        _FakePool(_lines(data), returncode),
        tmp_path / "manifest.json",
        tmp_path,
        timeout_seconds=30,
        on_event=on_event,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "torn",
    [b'{"success": true, "varia', b'{"' + EVENT_KEY.encode() + b'": "checkpoint", "par'],
    ids=["result", "event"],
)
async def test_a_torn_line_is_skipped_for_the_next(tmp_path, torn):
    """An interrupt mid-write leaves a line that is not JSON; the worker starts the next
    on a line of its own. None ("pool unavailable") would run the cell again cold."""
    events = []

    async def on_event(event):
        events.append(event)

    event = b'{"' + EVENT_KEY.encode() + b'": "head", "name": "best", "sequence": 1}'
    result = await _pooled(
        tmp_path,
        torn + b"\n" + event + b'\n{"success": false, "error": "interrupted"}\n',
        on_event=on_event,
    )

    assert result == {"success": False, "error": "interrupted"}
    # The event after the torn one still counts.
    assert events == [json.loads(event)]


@pytest.mark.asyncio
async def test_a_line_a_child_wrote_is_dropped_never_the_result(tmp_path):
    """R's worker shares its fd 1 with the cell's children: their lines are dropped, as
    the cold path drops them, and none passes for the result."""
    data = (
        b'42\nnull\n{"items": [1]}\n{"success": tr\n'
        b'\n{"success": true, "variables": {}, "stdout": "own\\n"}\n'
    )

    result = await _pooled(tmp_path, data)

    assert result == {"success": True, "variables": {}, "stdout": "own\n"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returncode", "interrupted"),
    # A kill as itself and under the lifeline wrapper; SIGKILL is 9, which Windows' signal
    # module does not name.
    [
        (-signal.SIGINT, True),
        (128 + signal.SIGINT, True),
        (0, False),
        (1, False),
        (-9, False),
        (137, False),
    ],
)
async def test_a_worker_an_interrupt_ended_is_a_failed_cell_not_a_pool_miss(
    tmp_path, returncode, interrupted
):
    result = await _pooled(tmp_path, b'{"success": true, "va', returncode)

    if interrupted:
        assert result is not None
        assert result["success"] is False
        assert "KeyboardInterrupt" in result["error"]
    else:
        # Died some other way (a crash, a kill): the pool is unavailable, run it cold.
        assert result is None


def test_an_uncaught_interrupt_ends_python_with_the_status_the_pool_reads():
    from strata.notebook import pool

    done = subprocess.run([sys.executable, "-c", "raise KeyboardInterrupt"], capture_output=True)

    assert done.returncode in pool._INTERRUPT_EXITS


@pytest.mark.asyncio
async def test_the_blank_line_before_a_result_is_not_logged(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="strata.notebook.pool"):
        result = await _pooled(tmp_path, b'\n{"success": true}\n')

    assert result == {"success": True}
    assert [r for r in caplog.records if r.name == "strata.notebook.pool"] == []


@pytest.mark.asyncio
async def test_the_cold_relay_reads_past_a_torn_event_line(caplog):
    from strata.notebook.local_runs import relay_events

    read_fd, write_fd = os.pipe()
    event = {EVENT_KEY: "head", "name": "best", "sequence": 1}
    with os.fdopen(write_fd, "wb") as pipe:
        pipe.write(b'{"' + EVENT_KEY.encode() + b'": "checkp\n\n' + json.dumps(event).encode())
    events = []

    async def on_event(payload):
        events.append(payload)

    with caplog.at_level("WARNING", logger="strata.notebook.local_runs"):
        assert await relay_events(read_fd, on_event, None) is None

    assert events == [event]
    # The torn line is said; the blank one before every event is not.
    said = [r for r in caplog.records if r.name == "strata.notebook.local_runs"]
    assert len(said) == 1


def test_an_event_starts_its_own_line_after_a_torn_one(tmp_path):
    from strata.notebook.run_channel import RunChannel

    written = ['{"event": "checkp']
    channel = RunChannel(
        written.append, tmp_path, serialize=lambda *a: {}, deserialize=lambda *a: None
    )
    channel.head("best", 1)

    assert json.loads("".join(written).splitlines()[-1])["name"] == "best"
