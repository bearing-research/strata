"""Tests for staleness detection."""

import pytest

from strata.notebook.models import CellState, NotebookState
from strata.notebook.session import NotebookSession


@pytest.fixture
def three_cell_notebook(tmp_path):
    """Create a 3-cell notebook with dependencies."""
    notebook_dir = tmp_path / "notebook"
    notebook_dir.mkdir()

    # Create cells directory
    cells_dir = notebook_dir / "cells"
    cells_dir.mkdir()

    # Create cell files
    (cells_dir / "load.py").write_text("df = [1, 2, 3]")
    (cells_dir / "clean.py").write_text("cleaned = [x for x in df]")
    (cells_dir / "explore.py").write_text("print(cleaned)")

    # Create pyproject.toml
    (notebook_dir / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    # Create NotebookState
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
    """Fresh notebook → all cells should be idle."""
    notebook_dir, notebook_state = three_cell_notebook

    session = NotebookSession(notebook_state, notebook_dir)
    staleness = session.compute_staleness()

    # All cells should be idle initially (no cached artifacts), with no reasons.
    for cell_id, status in staleness.items():
        assert status.status == "idle"
        assert len(status.reasons) == 0


def test_staleness_updates_cells_when_dag_is_invalid(tmp_path):
    """When session.dag is None, compute_staleness still authoritatively
    marks cells idle and clears cached causality data.

    The single-pass DAG builder no longer produces cycles from plain
    cell sources (forward references simply leave the cell without an
    upstream edge), so we exercise the ``dag is None`` branch directly
    by nulling it after construction — simulating any future failure
    mode that leaves the session without a DAG.
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
    """Deciding whether a cell is stale reads the outside world -- an @fetch
    URL, a @dataset registry, an @table catalog -- through synchronous calls
    with timeouts measured in tens of seconds. On the event loop one
    unreachable host stalled every notebook's socket and every stream in
    flight, so the async callers hand the work to a thread."""

    def test_the_lock_is_free_while_the_outside_world_is_read(
        self, three_cell_notebook, monkeypatch
    ):
        """Serializing the walk is not licence to hold the lock across a
        sixty-second fetch. The broadcast path takes this lock on the event
        loop, so a thread holding it out there is every socket in the process
        waiting on someone else's network."""
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

        def _record():
            ran_on.append(threading.get_ident())
            return original()

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
        """Not every caller was moved off the loop -- the broadcast path runs
        between a cell's result and the frames describing it, where an await
        reorders them. So the one in the thread and the one on the loop walk
        the same cells, writing each cell's status, and the cascade planner
        reads exactly that."""
        import asyncio
        import threading

        notebook_dir, notebook_state = three_cell_notebook
        session = NotebookSession(notebook_state, notebook_dir)
        inside = 0
        overlapped = False
        guard = threading.Lock()
        original = session._compute_staleness_locked

        def _watch(prefetched):
            nonlocal inside, overlapped
            with guard:
                inside += 1
                if inside > 1:
                    overlapped = True
            try:
                return original(prefetched)
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
        """It mutates the cells it walks. On the loop that was free."""
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

        def _watch(prefetched):
            nonlocal inside, overlapped
            with guard:
                inside += 1
                if inside > 1:
                    overlapped = True
            try:
                return original(prefetched)
            finally:
                with guard:
                    inside -= 1

        monkeypatch.setattr(session, "_compute_staleness_locked", _watch)

        await asyncio.gather(*(session.compute_staleness_async() for _ in range(4)))

        assert not overlapped
