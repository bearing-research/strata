"""A SQL cell's result table is a display backed by an artifact, like any other.

Without one, a SQL upstream stayed idle with no display, showed a table from a
previous parameter, could not be saved by ``save_cell_output``, and exported
without its result.
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

    The two filter settings differ in the table's first row (three regions at 0, two
    at 200), so a stale display is distinguishable from a refreshed one.
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
    session.refresh_environment_runtime()
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
    """Running only the consumer leaves the query cell showing what it produced, not idle."""
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
    """After the widget moves and the consumer recomputes, the query shows the new table."""
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
    """``save_cell_output`` can save a SQL display because an artifact backs it."""
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
    """An export re-fetches the table through its uri; ``markdown_text`` is not persisted."""
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
    """A cache hit reuses the stored table rather than writing an identical new version."""
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
    """The lineage behind a result names the bound parameter's artifact.

    A hash of the bound value makes a change recompute, but identifies no artifact,
    so without a recorded input reference the walk stops at one step.
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
    """``set_widget_value`` on a ``# @live`` notebook cascades as a slider drag does."""
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
    """An untouched control reports its declared default, the value the cell actually runs at."""
    from strata.notebook.ops import _cell_view

    session = _session(orders)
    control = _cell_view(session.notebook_state.get_cell("w")).controls[0]

    assert control.default == 0
    assert control.value == 0
