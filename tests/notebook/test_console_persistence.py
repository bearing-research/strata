"""Tests for stdout/stderr persistence across notebook reopens."""

from __future__ import annotations

import asyncio
from pathlib import Path

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def test_stdout_persists_across_reopen(tmp_path: Path):
    """Print output should survive closing and reopening a notebook."""
    notebook_dir = create_notebook(tmp_path, "ConsoleTest", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", 'print("hello from cell")\nx = 42')

    state = parse_notebook(notebook_dir)
    session = NotebookSession(state, notebook_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)

    result = asyncio.run(executor.execute_cell("c1", 'print("hello from cell")\nx = 42'))
    assert result.success
    assert "hello from cell" in result.stdout

    cell = next(c for c in session.notebook_state.cells if c.id == "c1")
    assert "hello from cell" in cell.console_stdout

    # Reopen from scratch (new session, new parse).
    state2 = parse_notebook(notebook_dir)
    session2 = NotebookSession(state2, notebook_dir)

    cell2 = next(c for c in session2.notebook_state.cells if c.id == "c1")
    assert "hello from cell" in cell2.console_stdout


def test_stderr_persists_across_reopen(tmp_path: Path):
    """Stderr should also survive reopens."""
    notebook_dir = create_notebook(tmp_path, "StderrTest", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1")
    source = 'import sys\nprint("warning!", file=sys.stderr)\nx = 1'
    write_cell(notebook_dir, "c1", source)

    state = parse_notebook(notebook_dir)
    session = NotebookSession(state, notebook_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)

    result = asyncio.run(executor.execute_cell("c1", source))
    assert result.success
    assert "warning!" in result.stderr

    state2 = parse_notebook(notebook_dir)
    session2 = NotebookSession(state2, notebook_dir)

    cell2 = next(c for c in session2.notebook_state.cells if c.id == "c1")
    assert "warning!" in cell2.console_stderr


def test_console_output_in_serialized_state(tmp_path: Path):
    """Serialized notebook state should include console_stdout/stderr."""
    notebook_dir = create_notebook(tmp_path, "SerializeTest", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", 'print("serialized")\nx = 1')

    state = parse_notebook(notebook_dir)
    session = NotebookSession(state, notebook_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)

    asyncio.run(executor.execute_cell("c1", 'print("serialized")\nx = 1'))

    data = session.serialize_notebook_state()
    cell_data = next(c for c in data["cells"] if c["id"] == "c1")
    assert "serialized" in cell_data["console_stdout"]


def test_console_cleared_on_new_execution(tmp_path: Path):
    """Re-executing a cell should replace old console output."""
    notebook_dir = create_notebook(tmp_path, "ClearTest", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", 'print("first")\nx = 1')

    state = parse_notebook(notebook_dir)
    session = NotebookSession(state, notebook_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)

    asyncio.run(executor.execute_cell("c1", 'print("first")\nx = 1'))
    cell = next(c for c in session.notebook_state.cells if c.id == "c1")
    assert "first" in cell.console_stdout

    write_cell(notebook_dir, "c1", 'print("second")\nx = 2')
    session.reload()
    executor2 = CellExecutor(session)
    asyncio.run(executor2.execute_cell("c1", 'print("second")\nx = 2'))

    cell = next(c for c in session.notebook_state.cells if c.id == "c1")
    assert "second" in cell.console_stdout
    assert "first" not in cell.console_stdout
