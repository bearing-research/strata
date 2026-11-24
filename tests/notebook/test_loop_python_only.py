"""``@loop`` runs Python cells only.

The loop path runs the cell's source in the Python harness, iteration after iteration, so a
cell of another language that declares a loop is refused rather than run as Python.
"""

from __future__ import annotations

import pytest

from strata.notebook.annotation_validation import blocking_diagnostic, validate_cell_annotations
from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellState, ConnectionSpec, NotebookState
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

R_LOOP = "# @loop max_iter=2 carry=x\nx <- x + 1\n"


@pytest.mark.parametrize(
    ("language", "source", "line"),
    [
        ("r", R_LOOP, 1),
        ("sql", "-- @sql connection=db\n-- @loop max_iter=2 carry=x\nSELECT 1\n", 2),
        ("prompt", "# @loop max_iter=2 carry=x\nSay hi.\n", 1),
        ("widget", "# @loop max_iter=2 carry=x\nslider('k', 0, 10)\n", 1),
    ],
)
def test_a_loop_outside_python_blocks_the_cell(language, source, line):
    cell = CellState(id="c", source=source, language=language)
    state = NotebookState(
        id="nb", cells=[cell], connections=[ConnectionSpec(name="db", driver="sqlite")]
    )

    errors = [
        (d.code, d.line) for d in validate_cell_annotations(cell, state) if d.severity == "error"
    ]

    assert errors == [("loop_not_python", line)]
    assert blocking_diagnostic(cell, state) is not None


def test_a_loop_until_alone_outside_python_blocks_the_cell():
    cell = CellState(id="c", source="# @loop_until x > 3\nx <- x + 1\n", language="r")

    diagnostic = blocking_diagnostic(cell, NotebookState(id="nb", cells=[cell]))

    assert diagnostic is not None and diagnostic.code == "loop_not_python"
    assert "`@loop_until`" in diagnostic.message


@pytest.mark.parametrize("key", ["loop max_iter=2 carry=x", "loop_until x > 3"])
def test_a_sql_loop_reports_its_error_alone(key):
    """Not also ``annotation_not_for_sql``: the error already says why."""
    cell = CellState(id="c", source=f"-- @sql connection=db\n-- @{key}\nSELECT 1\n", language="sql")
    state = NotebookState(
        id="nb", cells=[cell], connections=[ConnectionSpec(name="db", driver="sqlite")]
    )

    codes = [d.code for d in validate_cell_annotations(cell, state)]

    assert "loop_not_python" in codes and "annotation_not_for_sql" not in codes


def test_a_python_loop_is_not_refused():
    cell = CellState(id="c", source="# @loop max_iter=2 carry=x\nx = x + 1\n", language="python")

    assert blocking_diagnostic(cell, NotebookState(id="nb", cells=[cell])) is None


async def test_an_r_loop_is_refused_before_anything_runs(tmp_path, monkeypatch):
    notebook_dir = create_notebook(tmp_path, "Loop")
    add_cell_to_notebook(notebook_dir, "c", None, language="r")
    write_cell(notebook_dir, "c", R_LOOP)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    executor = CellExecutor(session)

    async def no_loop(*_args, **_kwargs):
        raise AssertionError("the Python loop path ran an R cell")

    monkeypatch.setattr(executor, "_execute_loop_cell", no_loop)

    result = await executor.execute_cell("c", R_LOOP)

    assert not result.success
    assert result.error_code == "loop_not_python"
