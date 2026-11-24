"""The sql_orders_report example reopens as it was left after a run.

Its seed is idempotent setup, cached by its source: a reopen finds it ready, and so the
cells ordered after it, rather than a seed that must run again and every reader stale.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("adbc_driver_sqlite")

from strata.notebook.executor import CellExecutor  # noqa: E402
from strata.notebook.models import CellStatus  # noqa: E402
from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "sql_orders_report"


def _open(notebook_dir: Path) -> NotebookSession:
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


async def test_a_reopen_after_a_run_finds_every_cell_ready(tmp_path):
    notebook_dir = tmp_path / "sql_orders_report"
    shutil.copytree(EXAMPLE_DIR, notebook_dir, ignore=shutil.ignore_patterns(".strata", ".venv"))
    session = _open(notebook_dir)
    executor = CellExecutor(session)
    assert session.dag is not None
    for cell_id in session.dag.topological_order:
        cell = session.notebook_state.get_cell(cell_id)
        assert cell is not None
        result = await executor.execute_cell(cell_id, cell.source)
        assert result.success, (cell_id, result.error)

    reopened = _open(notebook_dir)
    staleness = reopened.compute_staleness()

    assert {cell_id: s.status for cell_id, s in staleness.items()} == {
        cell.id: CellStatus.READY for cell in reopened.notebook_state.cells
    }
