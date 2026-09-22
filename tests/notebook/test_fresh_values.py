"""A value that is fresh on every run, and the cells that read it.

``# @nocache`` says a cell's result is not determined by its source, inputs and
environment: a clock, a counter, a file being edited. Two things went wrong
downstream of such a cell.

Its artifacts carried the provenance hash every other artifact does, and that
hash is the same on every run. A consumer's cache key is built from its
inputs' hashes, so the producer could re-run with a new value and the consumer
would still hit its cache and hand back what it computed from the old one.

And every multi-cell run (headless ``strata run``, the browser cascade, Run
All) asked each cell to materialise its upstreams, which for a normal producer
is a cache hit and for a ``@nocache`` one is another execution. One run of a
producer with two consumers executed it three times, and the two consumers
read different values in the same run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus, StalenessReason
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from tests.notebook.test_cli import _build_notebook, _mk_fake_venv


def _producer(counter: Path) -> str:
    """Bump a file counter and expose the new count, the way a fresh read does."""
    return (
        "# @nocache\n"
        "from pathlib import Path as _P\n"
        f"_c = _P({str(counter)!r})\n"
        "_c.write_text(str(int(_c.read_text()) + 1) if _c.exists() else '1')\n"
        "run_count = int(_c.read_text())\n"
    )


CONSUMER = "seen = run_count\nseen\n"


def _session(notebook_dir: Path) -> NotebookSession:
    return NotebookSession(parse_notebook(notebook_dir), notebook_dir)


def _run(executor: CellExecutor, session: NotebookSession, cell_id: str):
    cell = session.notebook_state.get_cell(cell_id)
    return asyncio.run(executor.execute_cell(cell_id, cell.source))


# -- finding 1: a consumer's cache follows the value it read ---------------


def test_a_consumer_recomputes_when_its_fresh_input_changed(tmp_path: Path):
    counter = tmp_path / "count.txt"
    nb = _build_notebook(tmp_path, cells=[("p", _producer(counter), None), ("c", CONSUMER, "p")])
    session = _session(nb)

    first = _run(CellExecutor(session), session, "c")
    assert first.display_output["preview"] == 1

    # Running the consumer again re-reads its @nocache source (count 2). The
    # consumer has to follow: a cache hit here returns the value from run 1.
    second = _run(CellExecutor(session), session, "c")
    assert counter.read_text() == "2"
    assert second.cache_hit is False
    assert second.display_output["preview"] == 2


def test_an_unchanged_fresh_value_still_lets_the_consumer_hit(tmp_path: Path):
    """The fix must key on the value, not on the fact of a re-run.

    A @nocache producer that happens to produce the same bytes again has not
    changed anything its consumer read, so the consumer's cache still applies.
    """
    steady = "# @nocache\nrun_count = 7\n"
    nb = _build_notebook(tmp_path, cells=[("p", steady, None), ("c", CONSUMER, "p")])
    session = _session(nb)

    assert _run(CellExecutor(session), session, "c").display_output["preview"] == 7
    again = _run(CellExecutor(session), session, "c")
    assert again.cache_hit is True
    assert again.display_output["preview"] == 7


def test_staleness_sees_a_changed_fresh_input(tmp_path: Path):
    """Execution and staleness compute the same key, so they must move together.

    The report's consumer read ``ready`` with no reason while holding a value
    computed from an older read. Here ``c`` holds a stored artifact (``d``
    consumes it) and ``d`` is a leaf; neither may claim ``ready`` once the
    fresh value underneath them has changed.

    Which not-ready word they get follows the walk's existing rule for a cell
    whose provenance no longer matches its stored result, and is not what this
    pins.
    """
    counter = tmp_path / "count.txt"
    nb = _build_notebook(
        tmp_path,
        cells=[
            ("p", _producer(counter), None),
            ("c", "seen = run_count\n", "p"),
            ("d", "seen\n", "c"),
        ],
    )
    session = _session(nb)
    _run(CellExecutor(session), session, "d")
    assert session.compute_staleness()["c"].status == CellStatus.READY

    # The producer runs on its own and reads a new value.
    _run(CellExecutor(session), session, "p")
    staleness = session.compute_staleness()

    assert staleness["c"].status != CellStatus.READY
    assert staleness["d"].status != CellStatus.READY


# -- the label: what a cell says once its fresh input moved ----------------


def test_a_consumer_of_a_moved_fresh_value_reads_stale_upstream(tmp_path: Path):
    """``stale · upstream changed``, not ``idle``, which reads as never run.

    ``c`` holds a stored artifact (``d`` consumes it). The producer re-runs on
    its own with a new value, so ``c``'s result was made from a version of the
    upstream that is no longer current: the definition of stale-upstream.
    """
    counter = tmp_path / "count.txt"
    nb = _build_notebook(
        tmp_path,
        cells=[
            ("p", _producer(counter), None),
            ("c", "seen = run_count\n", "p"),
            ("d", "seen\n", "c"),
        ],
    )
    session = _session(nb)
    _run(CellExecutor(session), session, "d")
    _run(CellExecutor(session), session, "p")

    staleness = session.compute_staleness()
    assert staleness["c"].status == CellStatus.STALE
    assert staleness["c"].reasons == [StalenessReason.UPSTREAM]


def test_the_reported_leaf_consumer_reads_stale_upstream(tmp_path: Path):
    """The round-3 shape exactly: the consumer is a leaf that displays.

    A leaf stores no variable artifact, only its display output, which records
    the same inputs and the same source and environment hashes.
    """
    counter = tmp_path / "count.txt"
    nb = _build_notebook(tmp_path, cells=[("p", _producer(counter), None), ("c", CONSUMER, "p")])
    session = _session(nb)
    _run(CellExecutor(session), session, "c")
    _run(CellExecutor(session), session, "p")

    staleness = session.compute_staleness()
    assert staleness["c"].status == CellStatus.STALE
    assert staleness["c"].reasons == [StalenessReason.UPSTREAM]


def test_an_edit_is_not_called_an_upstream_change(tmp_path: Path):
    """The label says what moved. An edited cell did not see its upstream move,
    even if the producer also re-ran; that keeps its existing classification."""
    counter = tmp_path / "count.txt"
    nb = _build_notebook(tmp_path, cells=[("p", _producer(counter), None), ("c", CONSUMER, "p")])
    session = _session(nb)
    _run(CellExecutor(session), session, "c")
    _run(CellExecutor(session), session, "p")
    session.notebook_state.get_cell("c").source = "seen = run_count * 2\nseen\n"

    staleness = session.compute_staleness()
    assert StalenessReason.UPSTREAM not in staleness["c"].reasons


def test_a_cell_that_never_ran_is_still_idle(tmp_path: Path):
    counter = tmp_path / "count.txt"
    nb = _build_notebook(tmp_path, cells=[("p", _producer(counter), None), ("c", CONSUMER, "p")])
    session = _session(nb)
    _run(CellExecutor(session), session, "p")

    assert session.compute_staleness()["c"].status == CellStatus.IDLE


# -- finding 2: one run executes each cell once ----------------------------


THREE_CELLS = [
    ("first", "a = run_count\na\n", "p"),
    ("second", "b = run_count\nb\n", "first"),
]


def test_one_run_executes_a_fresh_producer_once(tmp_path: Path):
    counter = tmp_path / "count.txt"
    nb = _build_notebook(tmp_path, cells=[("p", _producer(counter), None), *THREE_CELLS])
    session = _session(nb)
    executor = CellExecutor(session)

    with executor.one_run():
        for cell_id in session.dag.topological_order:
            _run(executor, session, cell_id)

    assert counter.read_text() == "1"
    first = session.notebook_state.get_cell("first").display_output.preview
    second = session.notebook_state.get_cell("second").display_output.preview
    assert first == second == 1


def test_a_run_in_display_order_still_materialises_what_it_has_not_run(tmp_path: Path):
    """Run All walks notebook order, which need not be topological.

    A consumer placed above its producer must still get the producer run for
    it. Only what this run has already executed is skipped.
    """
    counter = tmp_path / "count.txt"
    nb = _build_notebook(tmp_path, cells=[("p", _producer(counter), None), *THREE_CELLS])
    session = _session(nb)
    executor = CellExecutor(session)

    with executor.one_run():
        for cell_id in ["second", "first", "p"]:  # consumers first
            _run(executor, session, cell_id)

    # "second" materialised p for itself; "first" and then p's own turn both
    # found p already run in this run.
    assert counter.read_text() == "1"


def test_outside_a_run_a_single_cell_still_refreshes_its_fresh_source(tmp_path: Path):
    """Standalone semantics are unchanged: running a consumer re-reads @nocache."""
    counter = tmp_path / "count.txt"
    nb = _build_notebook(tmp_path, cells=[("p", _producer(counter), None), ("c", CONSUMER, "p")])
    session = _session(nb)
    executor = CellExecutor(session)  # one long-lived executor, no run scope

    _run(executor, session, "c")
    _run(executor, session, "c")
    assert counter.read_text() == "2"


def test_the_headless_run_executes_a_fresh_producer_once(tmp_path: Path, capsys):
    import json as _json

    from strata.cli import main

    counter = tmp_path / "count.txt"
    nb = _build_notebook(tmp_path, cells=[("p", _producer(counter), None), *THREE_CELLS])
    _mk_fake_venv(nb)

    assert main(["run", str(nb), "--no-sync", "--format", "json"]) == 0
    payload = _json.loads(capsys.readouterr().out)

    assert counter.read_text() == "1"
    assert all(cell["status"] == "ok" for cell in payload["cells"])


def test_a_forced_rerun_is_not_satisfied_by_an_earlier_cache_hit(tmp_path: Path):
    """Rerun-all bypasses every cell's cache, including one a consumer above it
    already materialised as a cache hit earlier in the same run."""
    counter = tmp_path / "count.txt"
    deterministic = (
        "from pathlib import Path as _P\n"
        f"_c = _P({str(counter)!r})\n"
        "_c.write_text(str(int(_c.read_text()) + 1) if _c.exists() else '1')\n"
        "value = 5\n"
    )
    nb = _build_notebook(tmp_path, cells=[("p", deterministic, None), ("c", "value\n", "p")])
    session = _session(nb)
    _run(CellExecutor(session), session, "c")  # warm the cache: p ran once
    assert counter.read_text() == "1"

    executor = CellExecutor(session)
    with executor.one_run():
        # c materialises p: a cache hit, p's body does not run.
        asyncio.run(executor.execute_cell_rerun("c", session.notebook_state.get_cell("c").source))
        assert counter.read_text() == "1"
        # p's own rerun must still really execute.
        p_source = session.notebook_state.get_cell("p").source
        asyncio.run(executor.execute_cell_rerun("p", p_source))

    assert counter.read_text() == "2"


def test_the_batch_path_does_not_serve_a_stale_consumer(tmp_path: Path):
    """Browser Run All batches cells into one harness and checks each cell's
    cache on its own, so it computes the same key and needs the same fix."""
    from tests.notebook.test_executor_batch import _cell_spec, _populate_consumed_vars

    counter = tmp_path / "count.txt"
    producer = _producer(counter)
    nb = _build_notebook(tmp_path, cells=[("p", producer, None), ("c", CONSUMER, "p")])
    session = _session(nb)

    def batch():
        specs = _populate_consumed_vars(
            [_cell_spec("p", producer), _cell_spec("c", CONSUMER)], session
        )
        return asyncio.run(CellExecutor(session).execute_batch(specs))

    assert batch().completed
    assert session.notebook_state.get_cell("c").display_output.preview == 1

    second = batch()
    assert second.completed
    assert counter.read_text() == "2"
    assert session.notebook_state.get_cell("c").display_output.preview == 2


# -- the browser's two multi-cell drivers ----------------------------------

# An explicit timeout keeps a cell out of Run All's batch, so these exercise
# the per-cell path that re-materialised upstreams (the batch path runs the
# producer once in a shared namespace and never had this problem).
UNBATCHED = [
    ("first", "# @timeout 60\na = run_count\na\n", "p"),
    ("second", "# @timeout 60\nb = run_count\nb\n", "first"),
]


def _unbatched_notebook(tmp_path: Path, counter: Path) -> NotebookSession:
    producer = "# @timeout 60\n" + _producer(counter)
    nb = _build_notebook(tmp_path, cells=[("p", producer, None), *UNBATCHED])
    return _session(nb)


def test_run_all_executes_a_fresh_producer_once(tmp_path: Path):
    from strata.notebook.executor import partition_batchable_runs
    from strata.notebook.ws import NotebookExecutionState, _execute_run_all

    counter = tmp_path / "count.txt"
    session = _unbatched_notebook(tmp_path, counter)
    cells = session.notebook_state.cells
    # The path under test: every cell runs single-cell, none in a batch.
    kinds = [kind for kind, _ in partition_batchable_runs(CellExecutor(session), cells)]
    assert "batch" not in kinds

    asyncio.run(
        _execute_run_all(None, session, [c.id for c in cells], NotebookExecutionState(), session.id)
    )

    assert counter.read_text() == "1"
    first = session.notebook_state.get_cell("first").display_output.preview
    second = session.notebook_state.get_cell("second").display_output.preview
    assert first == second == 1


def test_a_cascade_executes_a_fresh_producer_once(tmp_path: Path):
    from strata.notebook.cascade import CascadePlanner
    from strata.notebook.ws import NotebookExecutionState, _execute_cascade

    counter = tmp_path / "count.txt"
    session = _unbatched_notebook(tmp_path, counter)
    plan = CascadePlanner(session).plan("second")
    assert plan is not None
    # p runs as its own step, then "second" materialises its upstream p: the
    # second request is the one that used to execute it again.
    assert [step.cell_id for step in plan.steps] == ["p", "second"]

    asyncio.run(_execute_cascade(None, session, plan, NotebookExecutionState(), session.id))

    assert counter.read_text() == "1"
    assert session.notebook_state.get_cell("second").display_output.preview == 1
