"""A batched Run All shows each cell running while it executes, as single-cell runs do.

Each batch cell used to be announced running only when its result arrived, so the
browser never showed it running and never offered its stop button.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from strata.notebook.executor import BatchCellResult, BatchExecutionResult, CellExecutor
from strata.notebook.models import CellStatus
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.ws_payloads import cell_status_payload
from tests.notebook.test_cli import _build_notebook


def _session(tmp_path: Path, cells: list[tuple[str, str, str | None]]) -> NotebookSession:
    nb = _build_notebook(tmp_path, cells=cells)
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


def _capture(monkeypatch) -> list[dict[str, Any]]:
    from strata.notebook import ws

    sent: list[dict[str, Any]] = []

    async def record(notebook_id: str, message: dict[str, Any], **kwargs: Any) -> None:
        sent.append(message)

    monkeypatch.setattr(ws, "_broadcast_message", record)
    return sent


def _statuses(sent: list[dict[str, Any]], cell_id: str) -> list[str]:
    return [
        m["payload"]["status"]
        for m in sent
        if m["type"] == "cell_status" and m["payload"]["cell_id"] == cell_id
    ]


@pytest.mark.asyncio
async def test_a_batch_announces_each_cell_running_while_it_executes(tmp_path, monkeypatch):
    from strata.notebook import ws
    from strata.notebook.executor import partition_batchable_runs
    from strata.notebook.ws import NotebookExecutionState, _execute_run_all

    gate = tmp_path / "announced"
    # The cell finishes only once the server has announced it running.
    waits = (
        "import pathlib, time\n"
        f"_gate = pathlib.Path({str(gate)!r})\n"
        "for _ in range(200):\n"
        "    if _gate.exists():\n"
        "        break\n"
        "    time.sleep(0.05)\n"
        "assert _gate.exists(), 'not announced running'\n"
        "x = 1\n"
    )
    session = _session(tmp_path, [("first", waits, None), ("second", "y = x + 1\n", "first")])
    cells = session.notebook_state.cells
    assert [kind for kind, _ in partition_batchable_runs(CellExecutor(session), cells)] == ["batch"]
    state = NotebookExecutionState()
    sent: list[dict[str, Any]] = []
    seen_running: list[tuple[str | None, CellStatus]] = []

    async def record(notebook_id: str, message: dict[str, Any], **kwargs: Any) -> None:
        sent.append(message)
        payload = message["payload"]
        if message["type"] == "cell_status" and payload == cell_status_payload("first", "running"):
            seen_running.append(
                (state.running_cell, session.notebook_state.get_cell("first").status)
            )
            gate.touch()

    monkeypatch.setattr(ws, "_broadcast_message", record)

    await _execute_run_all(None, session, [c.id for c in cells], state, session.id)

    assert session.notebook_state.get_cell("first").status == CellStatus.READY
    assert seen_running == [("first", CellStatus.RUNNING)]
    # One running frame per cell, then its result.
    assert _statuses(sent, "first") == ["running", "ready"]
    assert _statuses(sent, "second") == ["running", "ready"]
    order = [(m["type"], m["payload"].get("cell_id")) for m in sent]
    assert order.index(("cell_status", "second")) > order.index(("cell_output", "first"))


def _fake_batch(monkeypatch, run):
    async def execute_batch(self, cell_specs, *, on_cell_start=None, **kwargs):
        return await run(on_cell_start)

    monkeypatch.setattr(CellExecutor, "execute_batch", execute_batch)


_TWO = [("first", "x = 1\n", None), ("second", "y = x + 1\n", "first")]


@pytest.mark.asyncio
async def test_a_cancel_mid_batch_leaves_the_running_cell_idle(tmp_path, monkeypatch):
    from strata.notebook.ws import NotebookExecutionState, _execute_run_all

    session = _session(tmp_path, _TWO)
    started = asyncio.Event()

    async def run(on_cell_start):
        await on_cell_start("first")
        started.set()
        await asyncio.Event().wait()

    _fake_batch(monkeypatch, run)
    sent = _capture(monkeypatch)
    task = asyncio.create_task(
        _execute_run_all(None, session, ["first", "second"], NotebookExecutionState(), session.id)
    )
    await asyncio.wait_for(started.wait(), timeout=30)
    assert session.notebook_state.get_cell("first").status == CellStatus.RUNNING

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert session.notebook_state.get_cell("first").status == CellStatus.IDLE
    assert _statuses(sent, "first") == ["running", "idle"]


@pytest.mark.asyncio
async def test_a_harness_that_dies_mid_cell_leaves_it_idle(tmp_path, monkeypatch):
    from strata.notebook.ws import NotebookExecutionState, _execute_run_all

    session = _session(tmp_path, _TWO)

    async def run(on_cell_start):
        await on_cell_start("first")
        return BatchExecutionResult(
            cell_results=[
                BatchCellResult(cell_id="first", status="not_run"),
                BatchCellResult(cell_id="second", status="not_run"),
            ],
            completed=False,
            end_reason="subprocess_died",
        )

    _fake_batch(monkeypatch, run)
    sent = _capture(monkeypatch)
    await _execute_run_all(
        None,
        session,
        ["first", "second"],
        NotebookExecutionState(),
        session.id,
        continue_on_error=False,
    )

    assert session.notebook_state.get_cell("first").status == CellStatus.IDLE
    assert _statuses(sent, "first") == ["running", "idle"]
