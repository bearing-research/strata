"""Tests for staleness detection."""

import pytest

from strata.notebook.models import CellState, NotebookState
from strata.notebook.session import NotebookSession


@pytest.fixture
def three_cell_notebook(tmp_path):
    """A 3-cell notebook with dependencies."""
    notebook_dir = tmp_path / "notebook"
    notebook_dir.mkdir()

    cells_dir = notebook_dir / "cells"
    cells_dir.mkdir()

    (cells_dir / "load.py").write_text("df = [1, 2, 3]")
    (cells_dir / "clean.py").write_text("cleaned = [x for x in df]")
    (cells_dir / "explore.py").write_text("print(cleaned)")

    (notebook_dir / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    notebook_state = NotebookState(
        id="test_nb",
        name="Test",
        cells=[
            CellState(
                id="load",
                source="df = [1, 2, 3]",
                language="python",
                order=0,
            ),
            CellState(
                id="clean",
                source="cleaned = [x for x in df]",
                language="python",
                order=1,
            ),
            CellState(
                id="explore",
                source="print(cleaned)",
                language="python",
                order=2,
            ),
        ],
    )

    return notebook_dir, notebook_state


def test_fresh_notebook_all_idle(three_cell_notebook):
    notebook_dir, notebook_state = three_cell_notebook

    session = NotebookSession(notebook_state, notebook_dir)
    staleness = session.compute_staleness()

    # All cells should be idle initially (no cached artifacts), with no reasons.
    for cell_id, status in staleness.items():
        assert status.status == "idle"
        assert len(status.reasons) == 0


def test_staleness_updates_cells_when_dag_is_invalid(tmp_path):
    """With ``session.dag`` None, cells are still marked idle and causality data is cleared.

    Plain sources no longer produce cycles, so the test nulls the DAG directly.
    """
    notebook_dir = tmp_path / "null_dag_notebook"
    notebook_dir.mkdir()
    (notebook_dir / "cells").mkdir()
    (notebook_dir / "pyproject.toml").write_text("[project]\nname = 'null_dag'\n")

    notebook_state = NotebookState(
        id="null_dag_nb",
        name="NullDag",
        cells=[
            CellState(id="a", source="x = y + 1", language="python", order=0),
            CellState(id="b", source="y = x + 1", language="python", order=1),
        ],
    )

    session = NotebookSession(notebook_state, notebook_dir)
    session.dag = None  # simulate failed DAG build

    for cell in session.notebook_state.cells:
        cell.status = "ready"
        cell.cache_hit = True
    session.causality_map = {"a": object()}  # prove compute_staleness clears stale data

    staleness = session.compute_staleness()

    assert staleness["a"].status == "idle"
    assert staleness["b"].status == "idle"
    assert session.causality_map == {}
    for cell in session.notebook_state.cells:
        assert cell.status == "idle"
        assert cell.cache_hit is False
        assert cell.staleness is not None
        assert cell.staleness.status == "idle"


class TestStalenessOffTheEventLoop:
    """Staleness reads @fetch URLs, @dataset registries and @table catalogs through blocking calls
    with long timeouts, so async callers run it on a thread to keep the event loop free.
    """

    def test_the_lock_is_free_while_the_outside_world_is_read(
        self, three_cell_notebook, monkeypatch
    ):
        """The broadcast path takes this lock on the event loop, so holding it across a slow fetch
        would stall every socket in the process.
        """
        import threading

        notebook_dir, notebook_state = three_cell_notebook
        session = NotebookSession(notebook_state, notebook_dir)
        free_while_reading: list[bool] = []
        original = session._collect_fetch_fingerprints

        def _probe(cell):
            # From another thread: the lock is reentrant, so the thread doing
            # the reading can always take it and would learn nothing.
            answer: list[bool] = []

            def _try() -> None:
                got = session._staleness_lock.acquire(timeout=0.5)
                if got:
                    session._staleness_lock.release()
                answer.append(got)

            probe = threading.Thread(target=_try)
            probe.start()
            probe.join()
            free_while_reading.append(answer[0])
            return original(cell)

        monkeypatch.setattr(session, "_collect_fetch_fingerprints", _probe)

        session.compute_staleness()

        assert free_while_reading, "the outside world was never read"
        assert all(free_while_reading), (
            "the staleness lock was held while reading the outside world, so a "
            "slow fetch blocks every caller -- including the event loop"
        )

    @pytest.mark.asyncio
    async def test_the_work_does_not_run_on_the_loop_thread(self, three_cell_notebook, monkeypatch):
        import threading

        notebook_dir, notebook_state = three_cell_notebook
        session = NotebookSession(notebook_state, notebook_dir)
        ran_on: list[int] = []
        original = session.compute_staleness

        def _record(executing=None):
            ran_on.append(threading.get_ident())
            return original(executing)

        monkeypatch.setattr(session, "compute_staleness", _record)

        await session.compute_staleness_async()

        assert ran_on, "the sync computation never ran"
        assert ran_on[0] != threading.get_ident(), (
            "staleness ran on the event loop thread, so its network calls block everything else"
        )

    @pytest.mark.asyncio
    async def test_a_caller_that_stayed_on_the_loop_does_not_interleave(
        self, three_cell_notebook, monkeypatch
    ):
        """The broadcast path stays on the loop (an await would reorder result and status frames),
        so the thread and loop callers walk and write the same cells the cascade planner reads.
        """
        import asyncio
        import threading

        notebook_dir, notebook_state = three_cell_notebook
        session = NotebookSession(notebook_state, notebook_dir)
        inside = 0
        overlapped = False
        guard = threading.Lock()
        original = session._compute_staleness_locked

        def _watch(prefetched, executing=None):
            nonlocal inside, overlapped
            with guard:
                inside += 1
                if inside > 1:
                    overlapped = True
            try:
                return original(prefetched, executing)
            finally:
                with guard:
                    inside -= 1

        monkeypatch.setattr(session, "_compute_staleness_locked", _watch)

        offloaded = asyncio.create_task(session.compute_staleness_async())
        await asyncio.sleep(0)
        session.compute_staleness()  # the on-loop caller, as ws.py still makes
        await offloaded

        assert not overlapped

    @pytest.mark.asyncio
    async def test_two_at_once_do_not_overlap(self, three_cell_notebook, monkeypatch):
        """The walk mutates the cells it visits, so two concurrent walks must serialize."""
        import asyncio
        import threading

        notebook_dir, notebook_state = three_cell_notebook
        session = NotebookSession(notebook_state, notebook_dir)
        inside = 0
        overlapped = False
        guard = threading.Lock()
        # The body inside the lock, not the entry point: threads queue at the
        # lock, so a wrapper around the entry point sees them arrive together
        # whether or not the serialization works.
        original = session._compute_staleness_locked

        def _watch(prefetched, executing=None):
            nonlocal inside, overlapped
            with guard:
                inside += 1
                if inside > 1:
                    overlapped = True
            try:
                return original(prefetched, executing)
            finally:
                with guard:
                    inside -= 1

        monkeypatch.setattr(session, "_compute_staleness_locked", _watch)

        await asyncio.gather(*(session.compute_staleness_async() for _ in range(4)))

        assert not overlapped
