"""A SQL cell's result table is a display like any other.

Round 7 found that it was the one display in the notebook backed by nothing.
The rows were always right and so was every Python calculation over them; what
went wrong was everything built on the display *record*. The preview was inline
only, so:

- a SQL cell reached as an upstream stayed ``idle`` showing nothing, because
  staleness resolves a cell's display from its cached artifacts and there were
  none;
- one whose consumer had recomputed at a new parameter went on showing the
  table from before the change;
- ``save_cell_output`` refused it as "not backed by an artifact";
- and an export rendered the query and dropped its result, because
  ``markdown_text`` is stripped at persist time and re-fetched through the
  artifact uri.

One cause, four symptoms. These tests pin each of them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("adbc_driver_sqlite")

from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.runtime_state import load_runtime_state  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402
from strata.notebook.writer import (  # noqa: E402
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)

QUERY = (
    "# @sql connection=db\n"
    "# @name regional_revenue\n"
    "SELECT region, SUM(gross) AS net FROM orders "
    "WHERE gross >= :minimum_amount GROUP BY region ORDER BY net DESC\n"
)
CONSUMER = "total = float(regional_revenue['net'].sum())\n{'total': total}\n"


@pytest.fixture
def orders(tmp_path) -> Path:
    """A widget feeding a SQL query feeding a Python cell.

    The rows are chosen so the two filter settings differ in the table's *first*
    row: at minimum 0 there are three regions, at 200 only two. A fixture whose
    top row is the same either way cannot tell a refreshed display from a stale
    one.
    """
    db = tmp_path / "orders.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, gross REAL)")
        conn.executemany(
            "INSERT INTO orders VALUES (?,?,?)",
            [(1, "North", 100.0), (2, "North", 150.0), (3, "South", 500.0), (4, "West", 250.0)],
        )
        conn.commit()

    nb = create_notebook(tmp_path / "nb", "orders")
    after = None
    for cell_id, language, source in (
        ("w", "widget", "minimum_amount = number(default=0, min=0, max=1000)\n"),
        ("q", "sql", QUERY),
        ("py", "python", CONSUMER),
    ):
        add_cell_to_notebook(nb, cell_id, after_cell_id=after, language=language)
        write_cell(nb, cell_id, source)
        after = cell_id
    toml = nb / "notebook.toml"
    toml.write_text(toml.read_text() + f'\n[connections.db]\ndriver = "sqlite"\npath = "{db}"\n')
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


def _display(session: Any, cell_id: str) -> Any:
    cell = session.notebook_state.get_cell(cell_id)
    return (cell.display_outputs or [None])[-1]


@pytest.mark.asyncio
async def test_a_sql_cell_reached_as_an_upstream_shows_what_it_produced(orders):
    """Running only the consumer materializes the query. The query cell used to
    stay idle with no display, which reads as "never run" for a cell whose value
    the consumer just used."""
    from strata.notebook.models import CellStatus

    session = _session(orders)
    result = await _run(session, "py")
    assert result is not None and result.success, (
        result and result.error,
        result and result.stderr,
    )
    session.compute_staleness()

    assert session.notebook_state.get_cell("py").status == CellStatus.READY
    assert session.notebook_state.get_cell("q").status == CellStatus.READY
    display = _display(session, "q")
    assert display is not None, "the SQL cell shows nothing it produced"
    assert "South" in display.preview


@pytest.mark.asyncio
async def test_the_table_follows_the_parameter_it_was_run_at(orders):
    """After the widget moves and the consumer recomputes, the query cell must
    not still be showing the table from before the change."""
    from strata.notebook.runtime_state import persist_cell_widget_values

    session = _session(orders)
    await _run(session, "py")
    before = _display(session, "q").preview
    assert "North" in before and "3 rows" in before

    persist_cell_widget_values(orders, "w", {"minimum_amount": 200})
    await _run(session, "w", mode="force")
    result = await _run(session, "py")
    session.compute_staleness()

    after = _display(session, "q").preview
    assert result is not None and result.display_outputs[-1]["preview"] == {"total": 750.0}
    # Two regions clear 200; North's orders are 100 and 150, so it drops out.
    assert "2 rows" in after
    assert "North" not in after


@pytest.mark.asyncio
async def test_the_table_is_an_artifact_a_caller_can_fetch(orders, tmp_path):
    """``save_cell_output`` refused a SQL display: it had no artifact behind it."""
    from strata.notebook.ops import _save_blob

    session = _session(orders)
    await _run(session, "q")

    display = _display(session, "q")
    assert display.artifact_uri, "the SQL display is backed by no artifact"
    assert display.bytes > 0

    dest = tmp_path / "table.md"
    saved = _save_blob(session, "q", dest, -1)
    assert saved.bytes > 0
    assert "South" in dest.read_text()


@pytest.mark.asyncio
async def test_an_export_reads_the_table_back_off_disk(orders):
    """``markdown_text`` is stripped when the display is persisted and re-fetched
    through the uri, so a display with no uri exported as a query and no result."""
    from strata.notebook.export import export_notebook

    session = _session(orders)
    await _run(session, "q")

    entry = load_runtime_state(orders).cells.get("q")
    assert entry is not None and entry.display_outputs, "nothing persisted for a reopen"

    body = export_notebook(orders)
    assert "| region | net |" in body
    assert "South" in body


@pytest.mark.asyncio
async def test_a_cache_hit_reuses_the_stored_table(orders):
    """The query was not re-issued, so the display it rebuilt is the stored one.
    Writing an identical blob under a new version on every run would grow the
    store for nothing."""
    session = _session(orders)
    await _run(session, "q")
    first = _display(session, "q").artifact_uri
    # Without the stored artifact both runs report ``None`` and comparing them
    # proves nothing, so pin that there is a version to reuse before reusing it.
    assert first and first.endswith("@v=1")

    result = await _run(session, "q")
    assert result is not None and result.cache_hit is True
    assert _display(session, "q").artifact_uri == first


@pytest.mark.asyncio
async def test_the_chain_behind_a_result_names_the_value_it_was_run_at(orders):
    """A query binding ``:minimum_amount`` came back as a single step.

    The bound variable's *hash* went into the cache key, which is what makes a
    change recompute, but a hash identifies no artifact: with no input
    reference recorded, the walk had nothing to follow and the parameter behind
    the number was unrecoverable.
    """
    from strata.notebook.mcp_server import _lineage
    from strata.notebook.session import SessionManager

    session = _session(orders)
    await _run(session, "py")

    manager = SessionManager()
    manager._sessions[session.id] = session
    chain = _lineage(manager, session.id, "q", "regional_revenue", max_depth=10)

    steps = chain["steps"]
    assert len(steps) > 1, "the query is the whole chain; its inputs are not in it"
    assert any("minimum_amount" in str(step.get("artifact_id", "")) for step in steps), steps


@pytest.mark.asyncio
async def test_an_agent_setting_a_live_widget_gets_the_cascade_a_drag_gets(orders):
    """``set_widget_value`` is documented as the same act as moving the slider.

    It ran the widget and stopped there, so a ``# @live`` notebook that
    auto-computes for a person left an agent looking at stale downstream cells
    and the previous total.
    """
    from strata.notebook.mcp_server import _set_widget_value
    from strata.notebook.models import CellStatus
    from strata.notebook.session import SessionManager
    from strata.notebook.writer import write_cell

    write_cell(orders, "w", "# @live\nminimum_amount = number(default=0, min=0, max=1000)\n")
    session = _session(orders)
    await _run(session, "py")
    assert session.notebook_state.get_cell("py").display_outputs[-1].preview == {"total": 1000.0}

    manager = SessionManager()
    manager._sessions[session.id] = session
    await _set_widget_value(manager, session.id, "w", {"minimum_amount": 200})

    consumer = session.notebook_state.get_cell("py")
    assert consumer.status == CellStatus.READY
    assert consumer.display_outputs[-1].preview == {"total": 750.0}


def test_an_untouched_control_reports_the_value_the_cell_runs_at(orders):
    """``value: null`` for a control nobody has moved described the storage, not
    the notebook: the widget executor falls back to the declared default, so
    that default *is* the input the result came from."""
    from strata.notebook.ops import _cell_view

    session = _session(orders)
    control = _cell_view(session.notebook_state.get_cell("w")).controls[0]

    assert control.default == 0
    assert control.value == 0
