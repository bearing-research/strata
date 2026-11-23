"""A scratchpad's warm workers and their base image (design 4.2)."""

from __future__ import annotations

import asyncio
import sys

import pytest


@pytest.mark.asyncio
async def test_a_claimed_worker_whose_base_image_no_longer_holds_is_replaced(tmp_path, monkeypatch):
    """A scratchpad worker built on an older environment never runs a cell (design 4.2)."""
    from strata.notebook.pool import WarmProcess, WarmProcessPool
    from strata.notebook.process_tree import (
        subprocess_kwargs_for_new_group,
        terminate_subprocess_tree,
    )

    pool = WarmProcessPool(
        tmp_path,
        base_image=lambda: ("new", []),
        base_image_holds=lambda stamp: stamp == "new",
    )
    spawned: list[int] = []

    async def spawn() -> None:
        spawned.append(1)

    monkeypatch.setattr(pool, "_spawn_warm_process", spawn)
    pool._started = True
    processes = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            **subprocess_kwargs_for_new_group(),
        )
        for _ in range(2)
    ]
    for process, stamp in zip(processes, ("old", "new"), strict=True):
        await pool._available.put(
            WarmProcess(process=process, created_at=0.0, ready=True, stamp=stamp)
        )
    claimed = await pool.acquire()
    await asyncio.gather(*pool._background_tasks)
    assert claimed is not None and claimed.stamp == "new"
    assert processes[0].returncode is not None
    assert spawned == [1]
    await terminate_subprocess_tree(processes[1])


async def _sleeper():
    from strata.notebook.process_tree import subprocess_kwargs_for_new_group

    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)", **subprocess_kwargs_for_new_group()
    )


@pytest.mark.asyncio
async def test_a_worker_checked_while_the_pool_drains_is_not_handed_out(tmp_path, monkeypatch):
    from strata.notebook.pool import WarmProcess, WarmProcessPool

    spawned: list[int] = []

    def drained_meanwhile(stamp):
        pool._started = False
        return True

    pool = WarmProcessPool(tmp_path, base_image=lambda: None, base_image_holds=drained_meanwhile)

    async def spawn() -> None:
        spawned.append(1)

    monkeypatch.setattr(pool, "_spawn_warm_process", spawn)
    pool._started = True
    process = await _sleeper()
    await pool._available.put(WarmProcess(process=process, created_at=0.0, ready=True))
    assert await pool.acquire() is None
    assert process.returncode is not None
    assert spawned == []


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["raised", "cancelled"])
async def test_a_worker_whose_check_did_not_finish_is_replaced(tmp_path, monkeypatch, ending):
    """Off the queue when the check raises or the run is cancelled during it: kept, it
    would run on outside the pool, which would stay one worker short. Unchecked, it runs
    nothing: the cell goes to the cold path."""
    import threading

    from strata.notebook.pool import WarmProcess, WarmProcessPool

    entered, release = threading.Event(), threading.Event()

    def check(stamp):
        if ending == "raised":
            raise OSError("the environment record is unreadable")
        entered.set()
        release.wait(30)
        return True

    pool = WarmProcessPool(tmp_path, base_image=lambda: None, base_image_holds=check)
    spawned: list[int] = []

    async def spawn() -> None:
        spawned.append(1)

    monkeypatch.setattr(pool, "_spawn_warm_process", spawn)
    pool._started = True
    process, behind = await _sleeper(), await _sleeper()
    await pool._available.put(WarmProcess(process=process, created_at=0.0, ready=True))
    await pool._available.put(WarmProcess(process=behind, created_at=0.0, ready=True))

    if ending == "raised":
        assert await pool.acquire() is None
    else:
        claim = asyncio.create_task(pool.acquire())
        await asyncio.to_thread(entered.wait, 30)
        claim.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await claim
    await asyncio.gather(*pool._background_tasks)

    assert process.returncode is not None
    assert spawned == [1]
    # Only the one checked: a check that keeps failing would fail every queued worker.
    assert behind.returncode is None and pool._available.qsize() == 1
    await pool.drain()


@pytest.mark.asyncio
async def test_a_discard_a_cancel_interrupts_still_finishes(tmp_path):
    """A worker whose base image no longer holds is released even when the run is
    cancelled while that happens: cut short, it would run on, and no replacement come."""
    from strata.notebook.pool import WarmProcess, WarmProcessPool

    pool = WarmProcessPool(tmp_path, base_image=lambda: None, base_image_holds=lambda s: False)
    started, release = asyncio.Event(), asyncio.Event()
    released: list[WarmProcess] = []

    async def release_and_replace(warm_proc):
        started.set()
        await release.wait()
        released.append(warm_proc)

    pool.release_and_replace = release_and_replace  # type: ignore[method-assign]
    pool._started = True
    worker = WarmProcess(process=None, created_at=0.0, ready=True)  # type: ignore[arg-type]
    await pool._available.put(worker)

    claim = asyncio.create_task(pool.acquire())
    await started.wait()
    claim.cancel()
    with pytest.raises(asyncio.CancelledError):
        await claim
    release.set()
    await asyncio.sleep(0)

    assert released == [worker]


@pytest.mark.asyncio
async def test_a_drained_pool_spawns_no_replacement(tmp_path, monkeypatch):
    from strata.notebook.pool import WarmProcess, WarmProcessPool

    pool = WarmProcessPool(tmp_path)
    spawned: list[int] = []

    async def spawn() -> None:
        spawned.append(1)

    monkeypatch.setattr(pool, "_spawn_warm_process", spawn)
    process = await _sleeper()
    await pool.release_and_replace(WarmProcess(process=process, created_at=0.0, ready=True))
    await asyncio.gather(*pool._background_tasks)
    assert process.returncode is not None
    assert spawned == []
