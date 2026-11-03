"""Run All records what each cell cost, so its cache hits can say what they saved.

The batch path used to record 0 for every cell: the profiler read "0ms" after a Run All and
every later hit was priced at nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from strata.notebook.executor import BatchCellResult, BatchExecutionResult, CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.ws_payloads import profiling_summary_payload
from tests.notebook.test_cli import _build_notebook

_NAP_MS = 50
_CELLS = [
    ("load", f"import time\ntime.sleep({_NAP_MS / 1000})\nrows = [1, 2, 3]\n", None),
    ("total", "result = sum(rows)\n", "load"),
    ("report", "text = str(result)\n", "total"),
]


def _open(notebook_dir: Path) -> NotebookSession:
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def _capture(monkeypatch, session: NotebookSession) -> list[dict[str, Any]]:
    from strata.notebook import ws

    sent: list[dict[str, Any]] = []

    async def record(notebook_id: str, message: dict[str, Any], **kwargs: Any) -> None:
        sent.append(message)

    class _Manager:
        def get_session(self, notebook_id: str) -> NotebookSession:
            return session

    monkeypatch.setattr(ws, "_broadcast_message", record)
    monkeypatch.setattr(ws, "_get_session_manager", lambda: _Manager())
    return sent


async def _run_all(session: NotebookSession) -> None:
    from strata.notebook.ws import NotebookExecutionState, _execute_run_all

    cell_ids = [c.id for c in session.notebook_state.cells]
    await _execute_run_all(None, session, cell_ids, NotebookExecutionState(), session.id)


def _outputs(sent: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {m["payload"]["cell_id"]: m["payload"] for m in sent if m["type"] == "cell_output"}


@pytest.mark.asyncio
async def test_a_cold_run_all_records_each_cells_own_time(tmp_path, monkeypatch):
    session = _open(_build_notebook(tmp_path, cells=_CELLS))
    sent = _capture(monkeypatch, session)

    await _run_all(session)

    durations = {cid: session.execution_history[cid][-1].duration_ms for cid, _, _ in _CELLS}
    assert all(duration and duration > 0 for duration in durations.values()), durations
    # A sleep never returns early: this is the body's time, not a placeholder.
    assert durations["load"] >= _NAP_MS
    assert _outputs(sent)["load"]["duration_ms"] >= _NAP_MS
    assert "saved_ms" not in _outputs(sent)["load"], "a computed cell saved nothing"
    summary = session.get_profiling_summary()
    assert summary["total_execution_ms"] >= _NAP_MS
    assert summary["cache_savings_ms"] == 0


@pytest.mark.asyncio
async def test_a_second_run_all_saves_what_the_first_one_cost(tmp_path, monkeypatch):
    notebook_dir = _build_notebook(tmp_path, cells=_CELLS)
    session = _open(notebook_dir)
    sent = _capture(monkeypatch, session)

    await _run_all(session)
    first = {cid: int(session.execution_history[cid][-1].duration_ms) for cid, _, _ in _CELLS}
    sent.clear()

    await _run_all(session)

    outputs = _outputs(sent)
    assert all(outputs[cid]["cache_hit"] for cid, _, _ in _CELLS), outputs
    assert {cid: outputs[cid]["saved_ms"] for cid, _, _ in _CELLS} == first
    # The hit's own time is its load, which the tooltip reports.
    assert all(session.execution_history[cid][-1].duration_ms for cid, _, _ in _CELLS)
    summary = session.get_profiling_summary()
    assert summary["cache_savings_ms"] == sum(first.values())
    assert summary["cache_hits"] == len(_CELLS)
    assert summary["cache_misses"] == len(_CELLS)

    # The figure comes back from .strata/runtime.json after a restart.
    restarted = _open(notebook_dir)
    assert restarted.get_profiling_summary()["cache_savings_ms"] == sum(first.values())
    assert restarted.last_hit_saved_ms("load") == first["load"]
    assert restarted.get_estimated_duration("load") == first["load"]


@pytest.mark.asyncio
async def test_a_failing_cell_still_records_how_long_it_ran(tmp_path, monkeypatch):
    source = f"import time\ntime.sleep({_NAP_MS / 1000})\nraise ValueError('no')\n"
    # Two: a lone cell takes the single-cell path.
    session = _open(
        _build_notebook(tmp_path, cells=[("boom", source, None), ("next", "y = 1\n", "boom")])
    )
    _capture(monkeypatch, session)

    await _run_all(session)

    duration = session.execution_history["boom"][-1].duration_ms
    assert duration is not None and duration >= _NAP_MS


@pytest.mark.asyncio
async def test_a_cell_with_no_known_time_records_none_not_zero(tmp_path, monkeypatch):
    """A cell the batch reported on without a time, as a watchdog kill or mount failure does."""
    # Two: a lone cell takes the single-cell path.
    notebook_dir = _build_notebook(
        tmp_path, cells=[("only", "x = 1\n", None), ("next", "y = x\n", "only")]
    )
    session = _open(notebook_dir)
    _capture(monkeypatch, session)
    session.record_execution("only", 1500.0, cache_hit=False)

    async def execute_batch(self, cell_specs, *, on_cell_event=None, **kwargs):
        result = BatchCellResult(cell_id="only", status="cell_error", error="Timed out")
        await on_cell_event(result)
        return BatchExecutionResult(
            cell_results=[result], completed=False, failed_cell_id="only", end_reason="cell_timeout"
        )

    monkeypatch.setattr(CellExecutor, "execute_batch", execute_batch)
    await _run_all(session)

    assert session.execution_history["only"][-1].duration_ms is None
    summary = session.get_profiling_summary()
    assert summary["total_execution_ms"] == 1500, "an unknown time adds nothing to the total"
    assert summary["cell_profiles"][0]["duration_ms"] is None, "shown as -, not 0ms"
    assert profiling_summary_payload(summary)["cell_profiles"][0]["duration_ms"] is None
    # The last time that is known still prices estimates and hits.
    assert session.get_estimated_duration("only") == 1500
    session.record_execution("only", 3.0, cache_hit=True)
    assert session.last_hit_saved_ms("only") == 1500
    assert session.get_profiling_summary()["cache_savings_ms"] == 1500

    # None survives the file and the restart: it does not come back as 0.
    restarted = _open(notebook_dir)
    assert restarted.execution_history["only"][-2].duration_ms is None
    assert restarted.get_profiling_summary()["cache_savings_ms"] == 1500


def test_a_hit_with_nothing_to_price_it_against_saves_nothing(tmp_path):
    """The badge reads "cached", never "saved 0ms"; the frontend hides a 0 and tests that."""
    session = _open(_build_notebook(tmp_path, cells=[("only", "x = 1\n", None)]))
    assert session.last_hit_saved_ms("only") == 0
    session.record_execution("only", 2.0, cache_hit=True)
    assert session.last_hit_saved_ms("only") == 0
    assert session.get_profiling_summary()["cache_savings_ms"] == 0


def test_a_team_hit_is_priced_by_the_publisher_over_a_local_run(tmp_path):
    session = _open(_build_notebook(tmp_path, cells=[("only", "x = 1\n", None)]))
    session.record_execution("only", 1000.0, cache_hit=False)
    session.record_execution("only", 2.0, cache_hit=True, from_team=True, team_saved_ms=38000)
    assert session.last_hit_saved_ms("only") == 38000
    summary = session.get_profiling_summary()
    assert summary["cache_savings_ms"] == 38000
    assert summary["team_cache_savings_ms"] == 38000
