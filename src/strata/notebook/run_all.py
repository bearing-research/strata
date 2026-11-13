"""Run All, as a plan and a failure policy, with no socket.

Which cells run together in one batch and which alone, which a failure blocks, and when
a failure stops the rest, decided here. How a batch or a single cell is run and told
about is the caller's: :class:`RunAllSteps`. The WebSocket's Run All
(``ws._execute_run_all``) supplies steps that broadcast; another frontend can supply its
own and keep the same policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from strata.notebook.executor import partition_batchable_runs
from strata.notebook.harness_user import LocalExecutionRefused, resolve_harness_user
from strata.notebook.models import CellStaleness, CellStatus, StalenessReason

if TYPE_CHECKING:
    from strata.notebook.cell_paths.batch import BatchExecutionResult
    from strata.notebook.dag import NotebookDag
    from strata.notebook.executor import CellExecutor
    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)

# A batch result's statuses that are neither a failure nor a cell the batch left.
_CLEAN = ("ok", "cache_hit", "not_run")


class RunAllSteps(Protocol):
    """How one Run All runs its cells and says what happened to each."""

    async def run_batch(self, cells: list[Any]) -> BatchExecutionResult:
        """Run *cells* in one harness."""
        ...

    async def run_one(self, cell: Any) -> bool:
        """Run *cell* alone; whether it succeeded."""
        ...

    async def blocked(self, cell_id: str) -> None:
        """Say *cell_id* did not run because what it reads from failed (already marked)."""
        ...


def upstream_that_failed(dag: NotebookDag | None, cell_id: str, failed: set[str]) -> str | None:
    """The failed cell this one reads from, directly or through others.

    Run All walks display order, not topological order, so a consumer can precede
    its producer.
    """
    if dag is None or not failed:
        return None
    seen: set[str] = set()
    queue = list(dag.cell_upstream.get(cell_id, []))
    while queue:
        upstream_id = queue.pop()
        if upstream_id in seen:
            continue
        seen.add(upstream_id)
        if upstream_id in failed:
            return upstream_id
        queue.extend(dag.cell_upstream.get(upstream_id, []))
    return None


def mark_blocked(session: NotebookSession, cell_id: str) -> None:
    """Mark a cell that did not run because what it reads from failed.

    Stale rather than error: nothing went wrong in this cell, and its last result is
    still the last thing it computed.
    """
    cell = session.notebook_state.get_cell(cell_id)
    if cell is not None:
        cell.status = CellStatus.STALE
        cell.staleness = CellStaleness(status=CellStatus.STALE, reasons=[StalenessReason.UPSTREAM])


def runnable_cells(session: NotebookSession, cell_ids: list[str]) -> list[Any]:
    """The requested cells that have source, in notebook order."""
    requested = set(cell_ids)
    return [c for c in session.notebook_state.cells if c.id in requested and c.source.strip()]


def batching_allowed() -> bool:
    """Whether this host runs cell code at all; a refused batch would leave the rest idle."""
    try:
        resolve_harness_user()
    except LocalExecutionRefused:
        return False
    return True


async def run_all(
    session: NotebookSession,
    executor: CellExecutor,
    runnable: list[Any],
    steps: RunAllSteps,
    *,
    continue_on_error: bool = True,
    batching: bool = True,
) -> None:
    """Run *runnable* in notebook order: consecutive batchable cells together, the rest
    (workers, loops, explicit timeouts, read-write mounts) one at a time.

    After a failure, cells that read from the failed one are blocked, or the run stops
    when *continue_on_error* is False. Single-cell for size-1 batches (nothing to
    amortize) and when *batching* is off.
    """
    had_failure = False
    # Failed cells and their dependents; a dependent must not run on the
    # pre-failure artifacts and report success.
    failed: set[str] = set()

    async def blocked(cell_id: str) -> bool:
        if upstream_that_failed(session.dag, cell_id, failed) is None:
            return False
        mark_blocked(session, cell_id)
        await steps.blocked(cell_id)
        failed.add(cell_id)
        return True

    # Each cell executes at most once: in display order a consumer above its
    # producer would materialise it, then the producer's row would run again.
    with executor.one_run():
        partition = partition_batchable_runs(executor, runnable)
        logger.info("Run All: %d cells in %d partitioned runs", len(runnable), len(partition))
        for kind, cells_in_run in partition:
            if had_failure and not continue_on_error:
                break
            # A batch never checks for a failed producer, so take those cells out first.
            cells_in_run = [cell for cell in cells_in_run if not await blocked(cell.id)]

            if kind == "batch" and len(cells_in_run) >= 2 and batching:
                result = await steps.run_batch(cells_in_run)
                if result.completed:
                    continue
                had_failure = True
                # Cells after the failure are not_run. With continue_on_error, run
                # them single-cell; any with a failed upstream is marked blocked.
                not_run = {r.cell_id for r in result.cell_results if r.status == "not_run"}
                failed.update(r.cell_id for r in result.cell_results if r.status not in _CLEAN)
                for cell in cells_in_run:
                    if cell.id not in not_run:
                        continue
                    if not continue_on_error:
                        break
                    if await blocked(cell.id):
                        continue
                    if not await steps.run_one(cell):
                        failed.add(cell.id)
                continue

            for cell in cells_in_run:
                if had_failure and not continue_on_error:
                    break
                if await blocked(cell.id):
                    continue
                if not await steps.run_one(cell):
                    had_failure = True
                    failed.add(cell.id)
