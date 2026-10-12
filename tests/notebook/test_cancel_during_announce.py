"""A cancel that lands while a run is being announced leaves the cell idle, not running.

The announce marks the cell running, then reads the run's worker policy and broadcasts.
A cancel in that window used to escape the handler that sets the cell idle, so the cell
stayed running and every environment change was refused as "execution running".
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from strata.notebook.models import CellStatus
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from tests.notebook.test_cli import _build_notebook

# A timeout keeps Run All from batching the cell, so it takes the single-cell path.
_CELLS = [("up", "# @timeout 60\nx = 1", None), ("down", "# @timeout 60\ny = x + 1", "up")]


def _session(tmp_path: Path) -> NotebookSession:
    nb = _build_notebook(tmp_path, cells=_CELLS)
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


def _run(entry: str, session: NotebookSession, state):
    from strata.notebook.cascade import CascadePlanner
    from strata.notebook.ws import _execute_cascade, _execute_run_all, execute_cell_and_broadcast

    if entry == "single":
        return execute_cell_and_broadcast(session, "up", state, session.id, mode="normal")
    if entry == "cascade":
        plan = CascadePlanner(session).plan("down")
        assert plan is not None and plan.steps[0].cell_id == "up"
        return _execute_cascade(None, session, plan, state, session.id)
    return _execute_run_all(None, session, ["up", "down"], state, session.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["single", "cascade", "run_all"])
async def test_a_cancel_during_the_announce_leaves_the_cell_idle(tmp_path, monkeypatch, entry):
    from strata.notebook import ws
    from strata.notebook.ws import NotebookExecutionState

    session = _session(tmp_path)
    announcing = asyncio.Event()

    async def held_policy_read(*args, **kwargs):
        announcing.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ws, "_read_run_policy", held_policy_read)
    run = asyncio.create_task(_run(entry, session, NotebookExecutionState()))
    await asyncio.wait_for(announcing.wait(), timeout=30)
    assert session.notebook_state.get_cell("up").status == CellStatus.RUNNING

    run.cancel()
    await asyncio.gather(run, return_exceptions=True)

    assert session.notebook_state.get_cell("up").status == CellStatus.IDLE
    assert not session._has_active_execution()
