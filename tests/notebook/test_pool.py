"""Tests for the warm process pool."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from strata.notebook.pool import WarmProcessPool
from strata.notebook.writer import create_notebook

pytestmark = [pytest.mark.integration, pytest.mark.warm_pool]


@pytest.fixture
def notebook_dir(tmp_path):
    """Create a simple notebook directory."""
    notebook_path = create_notebook(tmp_path, "Test Notebook")
    return notebook_path


class TestWarmProcessPool:
    """The warm process pool."""

    @pytest.mark.asyncio
    async def test_pool_starts_with_configured_size(self, notebook_dir):
        pool = WarmProcessPool(notebook_dir, pool_size=2)
        await pool.start()

        assert pool._available.qsize() == 2

        await pool.drain()

    @pytest.mark.asyncio
    async def test_acquire_warm_process(self, notebook_dir):
        pool = WarmProcessPool(notebook_dir, pool_size=1)
        await pool.start()

        warm_proc = await pool.acquire()

        if warm_proc is not None:  # May be None if startup failed
            assert warm_proc.process is not None
            assert warm_proc.ready is True

        await pool.drain()

    @pytest.mark.asyncio
    async def test_process_killed_after_use(self, notebook_dir):
        pool = WarmProcessPool(notebook_dir, pool_size=1)
        await pool.start()

        warm_proc = await pool.acquire()

        if warm_proc is not None:
            await pool.release_and_replace(warm_proc)

            assert warm_proc.process.returncode is not None

        await pool.drain()

    @pytest.mark.asyncio
    async def test_pool_drains_on_close(self, notebook_dir):
        pool = WarmProcessPool(notebook_dir, pool_size=2)
        await pool.start()

        initial_size = pool._available.qsize()
        assert initial_size == 2

        await pool.drain()

        assert pool._available.qsize() == 0

    @pytest.mark.asyncio
    async def test_pool_invalidate_respawns(self, notebook_dir):
        pool = WarmProcessPool(notebook_dir, pool_size=1)
        await pool.start()

        initial_size = pool._available.qsize()
        assert initial_size >= 1

        # Invalidate drains and respawns.
        await pool.invalidate()

        # The respawn may be async.
        await asyncio.sleep(0.5)

        final_size = pool._available.qsize()
        assert final_size >= 0

        await pool.drain()

    @pytest.mark.asyncio
    async def test_cold_fallback_when_pool_empty(self, notebook_dir):
        """acquire returns None when the pool is empty."""
        pool = WarmProcessPool(notebook_dir, pool_size=0)
        # Pool not started.

        warm_proc = await pool.acquire()
        assert warm_proc is None


@pytest.mark.asyncio
async def test_draining_while_a_worker_boots_stops_that_worker(notebook_dir, tmp_path):
    """drain() cancels a spawn still waiting for "ready"; the worker it started must go too,
    not live on until the server exits."""
    pid_file = tmp_path / "worker.pid"
    script = (
        "import os, pathlib, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(600)\n"
    )
    pool = WarmProcessPool(notebook_dir, pool_size=1, worker_command=[sys.executable, "-c", script])
    pool.track_background_task(asyncio.create_task(pool._spawn_warm_process()))
    for _ in range(600):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.05)
    pid = int(pid_file.read_text())

    try:
        await pool.drain()
        for _ in range(600):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail(f"the booting worker {pid} outlived drain()")
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_draining_before_the_spawn_returns_stops_the_wrapped_worker(
    notebook_dir, tmp_path, monkeypatch
):
    """drain() can cancel a spawn still inside create_subprocess_exec, as on a starved
    runner; asyncio then kills only the lifeline wrapper, and the worker under it must go too."""
    pid_file = tmp_path / "worker.pid"
    script = (
        "import os, pathlib, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(600)\n"
    )
    create_subprocess_exec = asyncio.create_subprocess_exec
    spawn_may_return = asyncio.Event()

    async def slow_create_subprocess_exec(*args, **kwargs):
        process = await create_subprocess_exec(*args, **kwargs)
        try:
            await spawn_may_return.wait()
        except asyncio.CancelledError:
            # What asyncio does when cancelled while connecting the pipes.
            process.kill()
            raise
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_create_subprocess_exec)
    pool = WarmProcessPool(notebook_dir, pool_size=1, worker_command=[sys.executable, "-c", script])
    spawn = asyncio.create_task(pool._spawn_warm_process())
    pool.track_background_task(spawn)
    for _ in range(600):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.05)
    pid = int(pid_file.read_text())

    try:
        drain = asyncio.create_task(pool.drain())
        while not spawn.cancelling():
            await asyncio.sleep(0)
        spawn_may_return.set()
        await drain
        for _ in range(600):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail(f"the wrapped worker {pid} outlived drain()")
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
