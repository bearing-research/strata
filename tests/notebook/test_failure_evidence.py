"""What a failed cell leaves behind for whoever looks at it next.

A run that fails reports its traceback and its prints to whoever started it.
An agent that comes back a moment later -- through ``cell show``, ``get_cell``,
or simply a reopen -- was getting an empty console, no error, and a status of
``idle`` that reads as "never run". The only way to find out what happened was
to run the failure again, side effects and all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus
from strata.notebook.ops import LocalNotebookOps
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

FAILING = 'print("before the failure")\n1 / 0\n'


def _notebook(tmp_path: Path, source: str = FAILING) -> Path:
    notebook_dir = create_notebook(tmp_path, "FailureEvidence", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", source)
    return notebook_dir


def _run(notebook_dir: Path, source: str) -> tuple[NotebookSession, object]:
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    executor = CellExecutor(session)
    result = asyncio.run(executor.execute_cell("c1", source))
    return session, result


def test_failed_cell_keeps_its_console_and_error(tmp_path: Path):
    notebook_dir = _notebook(tmp_path)
    session, result = _run(notebook_dir, FAILING)
    assert result.success is False

    cell = session.notebook_state.get_cell("c1")
    assert "before the failure" in cell.console_stdout
    assert "ZeroDivisionError" in (cell.error or "")


def test_the_failure_is_still_there_after_a_reopen(tmp_path: Path):
    notebook_dir = _notebook(tmp_path)
    _run(notebook_dir, FAILING)

    reopened = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    cell = reopened.notebook_state.get_cell("c1")
    assert "before the failure" in cell.console_stdout
    assert "ZeroDivisionError" in (cell.error or "")

    # And the curated agent view carries it, which is the surface the CLI and
    # the MCP server both project through.
    view = LocalNotebookOps(notebook_dir).get_cell("c1")
    assert "ZeroDivisionError" in (view.error or "")
    assert "before the failure" in view.console_stdout
    assert view.status == CellStatus.ERROR.value


def test_an_unrelated_change_does_not_erase_the_failure(tmp_path: Path):
    """Adding a cell elsewhere recomputes staleness for the whole notebook.

    The failed cell stored no artifact, so that walk can only call it idle.
    Its source has not changed and it has not been re-run, so the failure is
    still what is true about it.
    """
    notebook_dir = _notebook(tmp_path)
    session, _ = _run(notebook_dir, FAILING)
    session.mark_cell_error("c1")  # what the server does when a run comes back failed
    assert session.notebook_state.get_cell("c1").status == CellStatus.ERROR

    # Any structural edit (adding a cell, reordering) recomputes staleness
    # for the whole notebook, which is where the status was being lost.
    session.compute_staleness()

    assert session.notebook_state.get_cell("c1").status == CellStatus.ERROR


def test_editing_the_cell_drops_the_error(tmp_path: Path):
    """An error is about one source. Change it and the claim expires."""
    notebook_dir = _notebook(tmp_path)
    session, _ = _run(notebook_dir, FAILING)

    cell = session.notebook_state.get_cell("c1")
    cell.source = 'print("fixed")\n'
    session.compute_staleness()

    assert session.notebook_state.get_cell("c1").status != CellStatus.ERROR


def test_a_successful_run_clears_the_error(tmp_path: Path):
    notebook_dir = _notebook(tmp_path)
    session, _ = _run(notebook_dir, FAILING)

    executor = CellExecutor(session)
    good = 'print("fixed")\n'
    session.notebook_state.get_cell("c1").source = good
    result = asyncio.run(executor.execute_cell("c1", good))
    assert result.success is True

    assert session.notebook_state.get_cell("c1").error is None
    reopened = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    assert reopened.notebook_state.get_cell("c1").error is None
