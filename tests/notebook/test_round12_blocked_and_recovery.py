"""What a failed run publishes, and what a recovered one says.

A cell downstream of a failure must not run against the artifacts from before
it, and a rebuilt upstream must send the result that replaces a shown error.
"""

from __future__ import annotations

import asyncio
import json
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

GOOD = "# @sql connection=db\n# @name inventory\nSELECT sku, qty FROM stock\n"
BAD = GOOD.replace("FROM stock", "FROM missing_stock")
MID = "shortage = len(inventory)\nshortage\n"
REPORT = "report = shortage * 2\n{'report': report}\n"
INDEPENDENT = "audit = 99\naudit\n"


class Observer:
    def __init__(self) -> None:
        self.raw: list[str] = []

    async def send_text(self, text: str) -> None:
        self.raw.append(text)

    @property
    def sent(self) -> list[dict[str, Any]]:
        return [json.loads(text) for text in self.raw]

    def for_cell(self, cell_id: str) -> list[dict[str, Any]]:
        return [f for f in self.sent if (f.get("payload") or {}).get("cell_id") == cell_id]


@pytest.fixture
def inventory(tmp_path) -> Path:
    """SQL, two Python cells reading it in turn, and one that reads nothing.

    The SQL cell cannot batch, so the Python cells form their own batch, which
    ends early when the first fails.
    """
    db = tmp_path / "stock.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE stock (sku TEXT, qty INTEGER)")
        conn.executemany("INSERT INTO stock VALUES (?,?)", [("a", 10), ("b", 20)])
        conn.commit()
    nb = create_notebook(tmp_path / "nb", "inventory")
    after = None
    for cell_id, language, source in (
        ("q", "sql", GOOD),
        ("mid", "python", MID),
        ("rep", "python", REPORT),
        ("indep", "python", INDEPENDENT),
    ):
        add_cell_to_notebook(nb, cell_id, after_cell_id=after, language=language)
        write_cell(nb, cell_id, source)
        after = cell_id
    toml = nb / "notebook.toml"
    toml.write_text(toml.read_text() + f'\n[connections.db]\ndriver="sqlite"\npath="{db}"\n')
    return nb


def _session(nb: Path) -> Any:
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    session._analyze_and_build_dag()
    session.environment_sync_state = "ready"
    return session


def _watch(session: Any) -> Observer:
    from strata.notebook.ws import _notebook_connections

    observer = Observer()
    _notebook_connections[session.id] = [observer]
    return observer


async def _run_all(session: Any, observer: Observer, *, rerun: bool) -> None:
    from strata.notebook.ws import (
        _ensure_execution_state,
        _handle_notebook_rerun_all,
        _handle_notebook_run_all,
    )

    state = _ensure_execution_state(session.id)
    handler = _handle_notebook_rerun_all if rerun else _handle_notebook_run_all
    await handler(observer, session, state, session.id, {"continue_on_error": True})
    task = state.execution_task
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)


async def _run(session: Any, cell_id: str) -> Any:
    from strata.notebook.ws import _ensure_execution_state, execute_cell_and_broadcast

    return await execute_cell_and_broadcast(
        session, cell_id, _ensure_execution_state(session.id), session.id, mode="normal"
    )


def _break_sql(nb: Path, session: Any) -> None:
    write_cell(nb, "q", BAD)
    session.reload()
    session._analyze_and_build_dag()


def _fix_sql(nb: Path, session: Any) -> None:
    write_cell(nb, "q", GOOD)
    session.reload()
    session._analyze_and_build_dag()


@pytest.mark.asyncio
async def test_a_cell_behind_a_failure_publishes_nothing(inventory):
    """A cell behind a failure does not run, so nothing is published for it."""
    session = _session(inventory)
    await _run_all(session, _watch(session), rerun=False)
    assert all(c.status == CellStatus.READY for c in session.notebook_state.cells)

    _break_sql(inventory, session)
    observer = _watch(session)
    await _run_all(session, observer, rerun=True)

    assert not [f for f in observer.for_cell("rep") if f["type"] == "cell_output"], (
        "a cell behind the failure published a result"
    )
    assert session.notebook_state.get_cell("rep").status == CellStatus.STALE


@pytest.mark.asyncio
async def test_a_branch_that_reads_nothing_broken_still_runs(inventory):
    """Blocking is for what the failure reached, not for everything after it."""
    session = _session(inventory)
    await _run_all(session, _watch(session), rerun=False)
    _break_sql(inventory, session)

    observer = _watch(session)
    await _run_all(session, observer, rerun=True)

    assert [f for f in observer.for_cell("indep") if f["type"] == "cell_output"], (
        "the independent branch was blocked too"
    )
    assert session.notebook_state.get_cell("indep").status == CellStatus.READY


@pytest.mark.asyncio
async def test_a_rebuilt_upstream_sends_the_result_that_clears_its_error(inventory):
    """A status alone leaves a client showing the error it was told about."""
    session = _session(inventory)
    await _run(session, "rep")
    _break_sql(inventory, session)
    await _run(session, "rep")
    assert session.notebook_state.get_cell("q").error is not None

    _fix_sql(inventory, session)
    observer = _watch(session)
    result = await _run(session, "rep")

    assert result is not None and result.success, result and result.error
    for cell_id in ("q", "mid"):
        outputs = [f for f in observer.for_cell(cell_id) if f["type"] == "cell_output"]
        assert outputs, f"{cell_id} recovered with nothing to replace its error"
        assert session.notebook_state.get_cell(cell_id).error is None


@pytest.mark.asyncio
async def test_an_ordinary_upstream_is_not_announced_again(inventory):
    """A cache hit behind an already-green cell has nothing to correct, so no frame."""
    session = _session(inventory)
    await _run(session, "rep")

    observer = _watch(session)
    await _run(session, "rep")

    assert not observer.for_cell("q"), "a healthy upstream was announced for no reason"


FLAKY = (
    "# @nocache\n"
    "from pathlib import Path\n"
    "p = Path(MARKER)\n"
    "n = len(p.read_text().splitlines()) if p.exists() else 0\n"
    "p.write_text('x\\n' * (n + 1))\n"
    "if n >= 1:\n"
    "    raise ValueError('fails from the second run on')\n"
    "seed = 7\n"
    "seed\n"
)


@pytest.fixture
def flaky_chain(tmp_path) -> Path:
    """A cell that works once and then fails, with a consumer.

    Failing after a clean run gives the failure a ready downstream to invalidate.
    Editing a cell to break it would mark the downstream stale before the run.
    """
    marker = tmp_path / "runs.txt"
    nb = create_notebook(tmp_path / "nb", "flaky")
    after = None
    for cell_id, source in (
        ("flaky", FLAKY.replace("MARKER", repr(str(marker)))),
        ("down", "doubled = seed * 2\ndoubled\n"),
    ):
        add_cell_to_notebook(nb, cell_id, after_cell_id=after, language="python")
        write_cell(nb, cell_id, source)
        after = cell_id
    return nb


@pytest.mark.asyncio
async def test_a_batch_failure_blocks_what_read_from_it(flaky_chain):
    """A batch reports failure as ``cell_error``; the blocked set must match that spelling."""
    session = _session(flaky_chain)
    await _run(session, "down")
    assert session.notebook_state.get_cell("down").status == CellStatus.READY

    observer = _watch(session)
    await _run_all(session, observer, rerun=False)

    assert not [f for f in observer.for_cell("down") if f["type"] == "cell_output"], (
        "the consumer of a batch failure published a result"
    )
    assert session.notebook_state.get_cell("down").status == CellStatus.STALE


@pytest.mark.asyncio
async def test_the_cells_a_failure_invalidates_are_numbered_apart(flaky_chain):
    """The frames marking a failure's downstream stale are sent as their own batch."""
    session = _session(flaky_chain)
    await _run(session, "down")

    observer = _watch(session)
    await _run_all(session, observer, rerun=False)

    stale = [
        f
        for f in observer.sent
        if f["type"] == "cell_status" and (f.get("payload") or {}).get("status") == "stale"
    ]
    assert stale, "the failure invalidated nothing, so this proves nothing"
    seqs = [f["seq"] for f in observer.sent]
    assert len(set(seqs)) == len(seqs), f"frames share a sequence: {seqs}"
    assert seqs == sorted(seqs), f"frames went out of order: {seqs}"


@pytest.mark.asyncio
async def test_both_broken_upstreams_are_reported_in_one_run(tmp_path):
    """Both broken upstreams are reported in one run, not one per attempt."""
    nb = create_notebook(tmp_path / "nb", "siblings")
    after = None
    for cell_id, source in (
        ("left", "raise ValueError('left broken')\nleft = 1\n"),
        ("right", "raise ValueError('right broken')\nright = 2\n"),
        ("both", "total = left + right\ntotal\n"),
    ):
        add_cell_to_notebook(nb, cell_id, after_cell_id=after, language="python")
        write_cell(nb, cell_id, source)
        after = cell_id
    session = _session(nb)

    observer = _watch(session)
    result = await _run(session, "both")

    assert result is not None and not result.success
    errored = {
        (f.get("payload") or {}).get("cell_id") for f in observer.sent if f["type"] == "cell_error"
    }
    assert {"left", "right"} <= errored, f"a broken sibling went unmentioned: {errored}"
    for cell_id in ("left", "right"):
        assert session.notebook_state.get_cell(cell_id).status == CellStatus.ERROR
