"""Run All's plan and failure policy, over steps that only record (no socket, no harness)."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from strata.notebook import run_all
from strata.notebook.cell_paths.batch import BatchCellResult, BatchExecutionResult
from strata.notebook.models import CellStatus

pytestmark = pytest.mark.asyncio


def _session(upstream: dict[str, list[str]]):
    cells = {
        cell_id: SimpleNamespace(id=cell_id, status=None, staleness=None) for cell_id in upstream
    }
    return SimpleNamespace(
        dag=SimpleNamespace(cell_upstream=upstream),
        notebook_state=SimpleNamespace(get_cell=cells.get),
    )


class _Executor:
    def __init__(self) -> None:
        self.runs = 0

    @contextlib.contextmanager
    def one_run(self):
        self.runs += 1
        yield


class _Steps:
    """Records what Run All asked for; a batch fails at *batch_fails*, a cell in *fails*."""

    def __init__(self, *, batch_fails: str | None = None, fails: set[str] = frozenset()):
        self.calls: list[tuple[str, object]] = []
        self.batch_fails = batch_fails
        self.fails = fails

    async def run_batch(self, cells):
        ids = [c.id for c in cells]
        self.calls.append(("batch", ids))
        if self.batch_fails is None:
            return BatchExecutionResult(
                cell_results=[BatchCellResult(cell_id=i, status="ok") for i in ids], completed=True
            )
        results, seen = [], False
        for i in ids:
            if i == self.batch_fails:
                results.append(BatchCellResult(cell_id=i, status="cell_error"))
                seen = True
            else:
                results.append(BatchCellResult(cell_id=i, status="not_run" if seen else "ok"))
        return BatchExecutionResult(
            cell_results=results, completed=False, failed_cell_id=self.batch_fails
        )

    async def run_one(self, cell):
        self.calls.append(("one", cell.id))
        return cell.id not in self.fails

    async def blocked(self, cell_id):
        self.calls.append(("blocked", cell_id))


def _plan(monkeypatch, plan: list[tuple[str, list[str]]]):
    def partition(_executor, _cells):
        return [(kind, [SimpleNamespace(id=i) for i in ids]) for kind, ids in plan]

    monkeypatch.setattr(run_all, "partition_batchable_runs", partition)


async def test_batchable_cells_run_together_and_the_rest_alone(monkeypatch):
    _plan(monkeypatch, [("batch", ["a", "b"]), ("single", ["c"])])
    steps, executor = _Steps(), _Executor()

    await run_all.run_all(_session({"a": [], "b": [], "c": []}), executor, [], steps)

    assert steps.calls == [("batch", ["a", "b"]), ("one", "c")]
    assert executor.runs == 1


async def test_a_batch_of_one_runs_alone(monkeypatch):
    """Nothing to amortize: with batching on, a batchable run of one still runs alone."""
    _plan(monkeypatch, [("batch", ["a"]), ("batch", ["b", "c"])])
    steps = _Steps()

    await run_all.run_all(_session({"a": [], "b": [], "c": []}), _Executor(), [], steps)

    assert steps.calls == [("one", "a"), ("batch", ["b", "c"])]


async def test_a_batch_left_with_one_cell_after_blocking_runs_alone(monkeypatch):
    _plan(monkeypatch, [("single", ["p"]), ("batch", ["q", "r"])])
    steps = _Steps(fails={"p"})

    await run_all.run_all(_session({"p": [], "q": ["p"], "r": []}), _Executor(), [], steps)

    assert steps.calls == [("one", "p"), ("blocked", "q"), ("one", "r")]


async def test_with_batching_off_every_cell_runs_alone(monkeypatch):
    _plan(monkeypatch, [("batch", ["a", "b"]), ("batch", ["c", "d"])])
    steps = _Steps()

    await run_all.run_all(
        _session({"a": [], "b": [], "c": [], "d": []}), _Executor(), [], steps, batching=False
    )

    assert steps.calls == [("one", "a"), ("one", "b"), ("one", "c"), ("one", "d")]


async def test_after_a_batch_fails_its_rest_run_alone_and_dependents_are_blocked(monkeypatch):
    _plan(monkeypatch, [("batch", ["a", "b", "c", "d"]), ("single", ["e"])])
    session = _session({"a": [], "b": [], "c": ["b"], "d": [], "e": ["c"]})
    steps = _Steps(batch_fails="b")

    await run_all.run_all(session, _Executor(), [], steps)

    assert steps.calls == [
        ("batch", ["a", "b", "c", "d"]),
        ("blocked", "c"),
        ("one", "d"),
        ("blocked", "e"),
    ]
    blocked = session.notebook_state.get_cell("e")
    assert blocked.status == CellStatus.STALE
    assert [r.value for r in blocked.staleness.reasons] == ["upstream"]


async def test_a_failure_stops_the_rest_without_continue_on_error(monkeypatch):
    _plan(monkeypatch, [("single", ["a"]), ("batch", ["b", "c"])])
    steps = _Steps(fails={"a"})

    await run_all.run_all(
        _session({"a": [], "b": [], "c": []}), _Executor(), [], steps, continue_on_error=False
    )

    assert steps.calls == [("one", "a")]


async def test_a_consumer_above_its_failed_producer_is_blocked_too(monkeypatch):
    """Display order is not topological: the failed cell is found through others."""
    _plan(monkeypatch, [("single", ["p"]), ("single", ["q"]), ("single", ["r"])])
    steps = _Steps(fails={"p"})

    await run_all.run_all(_session({"p": [], "q": ["p"], "r": ["q"]}), _Executor(), [], steps)

    assert steps.calls == [("one", "p"), ("blocked", "q"), ("blocked", "r")]
