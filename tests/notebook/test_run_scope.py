"""RunScope: the harness stops while its readers still read, and no helper outlives the scope."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from strata.notebook.run_scope import RunScope

pytestmark = pytest.mark.asyncio


async def _forever() -> None:
    await asyncio.Event().wait()


async def test_a_normal_exit_cancels_helpers_and_leaves_the_harness(monkeypatch):
    stopped = []

    async def terminate(proc):
        stopped.append(proc)

    monkeypatch.setattr("strata.notebook.process_tree.terminate_subprocess_tree", terminate)
    async with RunScope("Test") as scope:
        scope.own(SimpleNamespace(pid=1))
        helper = scope.start(_forever())

    assert stopped == []
    assert helper.cancelled()


async def test_the_harness_stops_while_its_helpers_still_read(monkeypatch):
    """A cell writing as it shuts down must not block on a pipe nobody reads."""
    helper_running_at_stop = []

    async def terminate(proc):
        helper_running_at_stop.append(not helper.done())

    monkeypatch.setattr("strata.notebook.process_tree.terminate_subprocess_tree", terminate)
    with pytest.raises(RuntimeError, match="callback"):
        async with RunScope("Test") as scope:
            scope.own(SimpleNamespace(pid=1))
            helper = scope.start(_forever())
            await asyncio.sleep(0)
            raise RuntimeError("callback")

    assert helper_running_at_stop == [True]
    assert helper.cancelled()


async def test_a_second_cancel_during_the_stop_still_ends_the_helpers(monkeypatch):
    stopping = asyncio.Event()
    release = asyncio.Event()

    async def terminate(proc):
        stopping.set()
        await release.wait()

    monkeypatch.setattr("strata.notebook.process_tree.terminate_subprocess_tree", terminate)
    helpers: list[asyncio.Task[None]] = []

    async def run() -> None:
        async with RunScope("Test") as scope:
            scope.own(SimpleNamespace(pid=1))
            helpers.append(scope.start(_forever()))
            raise RuntimeError("first")

    task = asyncio.create_task(run())
    await asyncio.wait_for(stopping.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert helpers[0].cancelled()
    release.set()
