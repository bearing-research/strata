"""A SQL cell must not hold the server, and a failure must name the cell it hit.

Round 8. A SQL query ran its driver work directly on the event loop, so for as
long as the query lasted nothing else on the server was served: an independent
health check took 2.9 seconds against 0.003 idle, and a `cell_cancel` could not
even be delivered until the query it meant to stop had finished on its own.

Separately, a SQL cell that failed while being materialized for a downstream
consumer said nothing about it. The consumer reported the failure; the cell that
caused it stayed `idle`, with no error and its last successful table still
showing, which is the state of a cell that never ran.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("adbc_driver_sqlite")

from strata.notebook.models import CellStatus  # noqa: E402
from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402
from strata.notebook.writer import (  # noqa: E402
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)

# Bounded, and long enough that the loop would visibly stall if the driver ran
# on it. Nothing here asserts how long it takes, only that other work ran.
SLOW = (
    "# @sql connection=db\n"
    "# @name slow_total\n"
    "WITH RECURSIVE counter(n) AS (\n"
    "  VALUES(0) UNION ALL SELECT n+1 FROM counter WHERE n < 20000000\n"
    ") SELECT SUM(n) AS total FROM counter\n"
)
GOOD = (
    "# @sql connection=db\n"
    "# @name rev\n"
    "SELECT region, SUM(gross) AS net FROM orders GROUP BY region\n"
)
CONSUMER = "rows = len(rev)\n{'rows': rows}\n"


def _notebook(tmp_path: Path, cells: list[tuple[str, str, str]]) -> Path:
    db = tmp_path / "orders.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, gross REAL)")
        conn.executemany(
            "INSERT INTO orders VALUES (?,?,?)", [(1, "North", 100.0), (2, "South", 500.0)]
        )
        conn.commit()
    nb = create_notebook(tmp_path / "nb", "round8")
    after = None
    for cell_id, language, source in cells:
        add_cell_to_notebook(nb, cell_id, after_cell_id=after, language=language)
        write_cell(nb, cell_id, source)
        after = cell_id
    toml = nb / "notebook.toml"
    toml.write_text(toml.read_text() + f'\n[connections.db]\ndriver="sqlite"\npath="{db}"\n')
    return nb


def _session(nb: Path) -> Any:
    session = NotebookSession(parse_notebook(nb), nb)
    session._analyze_and_build_dag()
    session.environment_sync_state = "ready"
    return session


async def _run(session: Any, cell_id: str, mode: str = "normal") -> Any:
    from strata.notebook.ws import _ensure_execution_state, execute_cell_and_broadcast

    return await execute_cell_and_broadcast(
        session, cell_id, _ensure_execution_state(session.id), session.id, mode=mode
    )


@pytest.mark.asyncio
async def test_the_driver_does_not_run_on_the_event_loop(tmp_path, monkeypatch):
    """Asserted as the mechanism, not as a duration.

    Counting how much other work got scheduled proves little: the surrounding
    async machinery yields either way, so a loop-blocking query still lets a
    heartbeat tick. What decides whether the rest of the server is served is
    plainly where the driver call runs, so that is what this checks.
    """
    import threading

    from strata.notebook.sql import cell_executor

    loop_thread = threading.get_ident()
    ran_on: dict[str, int] = {}
    real_query = cell_executor._execute_query

    def recording_query(*args: Any, **kwargs: Any) -> Any:
        ran_on["query"] = threading.get_ident()
        return real_query(*args, **kwargs)

    monkeypatch.setattr(cell_executor, "_execute_query", recording_query)

    session = _session(_notebook(tmp_path, [("q", "sql", GOOD)]))
    result = await _run(session, "q")

    assert result is not None and result.success
    assert "query" in ran_on, "the query never ran"
    assert ran_on["query"] != loop_thread, "the driver ran on the event loop"


@pytest.mark.asyncio
async def test_a_write_cell_does_not_run_on_the_event_loop(tmp_path, monkeypatch):
    """The write path blocks the same way the read path did."""
    import threading

    from strata.notebook.sql import cell_executor

    loop_thread = threading.get_ident()
    ran_on: dict[str, int] = {}
    real_write = cell_executor._execute_write_statements

    def recording_write(*args: Any, **kwargs: Any) -> Any:
        ran_on["write"] = threading.get_ident()
        return real_write(*args, **kwargs)

    monkeypatch.setattr(cell_executor, "_execute_write_statements", recording_write)

    seed = (
        "# @sql connection=db write=true\n"
        "# @name seeded\n"
        "INSERT INTO orders VALUES (9, 'West', 50.0);\n"
    )
    session = _session(_notebook(tmp_path, [("w", "sql", seed)]))
    result = await _run(session, "w")

    assert result is not None and result.success
    assert ran_on.get("write") not in (None, loop_thread), "the write ran on the event loop"


@pytest.mark.asyncio
async def test_a_cancelled_query_stops_waiting_and_publishes_nothing(tmp_path):
    """Cancel could not be delivered at all until the query returned, and the
    run then published its result as though nothing had been asked."""
    from strata.notebook.ws import _ensure_execution_state, _handle_cell_cancel

    session = _session(_notebook(tmp_path, [("q", "sql", SLOW)]))
    execution_state = _ensure_execution_state(session.id)

    run = asyncio.create_task(_run(session, "q"))
    async with execution_state.control_lock:
        execution_state.execution_task = run
        execution_state.running_cell = "q"
    # Let the query get under way before asking for it to stop.
    await asyncio.sleep(0.5)

    await _handle_cell_cancel(session, {"cell_id": "q"}, execution_state, session.id)

    assert run.cancelled()
    cell = session.notebook_state.get_cell("q")
    assert cell.status != CellStatus.READY
    assert not (cell.display_outputs or []), "a cancelled run published its result"


@pytest.mark.asyncio
async def test_the_notebook_runs_again_after_a_cancelled_query(tmp_path):
    """The reservation the cancelled run held has to come back."""
    from strata.notebook.ws import _ensure_execution_state, _handle_cell_cancel

    nb = _notebook(tmp_path, [("q", "sql", SLOW), ("fast", "sql", GOOD)])
    session = _session(nb)
    execution_state = _ensure_execution_state(session.id)

    run = asyncio.create_task(_run(session, "q"))
    async with execution_state.control_lock:
        execution_state.execution_task = run
        execution_state.running_cell = "q"
    await asyncio.sleep(0.5)
    await _handle_cell_cancel(session, {"cell_id": "q"}, execution_state, session.id)

    result = await _run(session, "fast")
    assert result is not None and result.success
    assert session.notebook_state.get_cell("fast").status == CellStatus.READY


@pytest.mark.asyncio
async def test_a_failure_while_materializing_names_the_cell_it_hit(tmp_path):
    """Run only the consumer, with SQL that has never failed before.

    A source that failed here previously can restore the recorded error and
    hide the gap, so the broken query has to be one this notebook has not seen.
    """
    nb = _notebook(tmp_path, [("q", "sql", GOOD), ("py", "python", CONSUMER)])
    session = _session(nb)
    await _run(session, "py")
    assert session.notebook_state.get_cell("q").display_outputs, "not warm to begin with"

    write_cell(nb, "q", GOOD.replace("SUM(gross)", "SUM(never_seen_round8)"))
    session.reload()
    session._analyze_and_build_dag()
    result = await _run(session, "py")

    assert result is not None and not result.success
    broken = session.notebook_state.get_cell("q")
    assert broken.status == CellStatus.ERROR
    assert broken.error and "never_seen_round8" in broken.error
    assert not (broken.display_outputs or []), "the failed cell still shows its old table"


@pytest.mark.asyncio
async def test_both_cells_recover_when_the_query_is_fixed(tmp_path):
    nb = _notebook(tmp_path, [("q", "sql", GOOD), ("py", "python", CONSUMER)])
    session = _session(nb)
    await _run(session, "py")
    write_cell(nb, "q", GOOD.replace("SUM(gross)", "SUM(never_seen_round8)"))
    session.reload()
    session._analyze_and_build_dag()
    await _run(session, "py")

    write_cell(nb, "q", GOOD)
    session.reload()
    session._analyze_and_build_dag()
    result = await _run(session, "py")

    assert result is not None and result.success
    assert session.notebook_state.get_cell("q").status == CellStatus.READY
    assert session.notebook_state.get_cell("q").error is None
    assert session.notebook_state.get_cell("py").status == CellStatus.READY


@pytest.mark.asyncio
async def test_every_outbound_frame_gets_its_own_sequence(tmp_path):
    """The protocol reference promises one counter that increments on every
    server-to-client message, and tells clients to dedupe on ``seq``.

    A whole batch of staleness frames shared one number, and a sync reply and
    an agent note were hard-coded to 0, so a client following that advice threw
    away legitimate frames -- including the status saying a cell had finished --
    and read the reply to its own sync as a gap.
    """
    from typing import cast

    from strata.notebook.mcp_server import _agent_note
    from strata.notebook.ws import (
        _ensure_execution_state,
        _handle_notebook_sync,
        _notebook_connections,
    )
    from tests.notebook.e2e_fixtures import FakeNotebookWebSocket

    nb = _notebook(tmp_path, [("q", "sql", GOOD), ("py", "python", CONSUMER)])
    session = _session(nb)

    observer = FakeNotebookWebSocket()
    _notebook_connections.setdefault(session.id, []).append(cast(Any, observer))
    _ensure_execution_state(session.id)
    try:
        # A run (status + output + the staleness batch behind it), a sync reply
        # and an agent note: the three paths the trace showed repeating.
        await _run(session, "py")
        await _handle_notebook_sync(cast(Any, observer), session, session.id)
        await _agent_note(session.id, "mcp", "widget minimum_amount=200")
    finally:
        _notebook_connections.get(session.id, []).remove(cast(Any, observer))

    seqs = [frame["seq"] for frame in observer.sent]
    assert len(seqs) > 3, f"too few frames to prove anything: {len(seqs)}"
    assert len(set(seqs)) == len(seqs), f"repeated sequence numbers: {seqs}"
    assert seqs == sorted(seqs), f"sequences went backwards: {seqs}"
    assert 0 not in seqs, f"a frame was sent with a hard-coded 0: {seqs}"
