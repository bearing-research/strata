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
