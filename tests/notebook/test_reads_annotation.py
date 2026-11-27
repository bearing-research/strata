"""``# @reads``: names a cell reads where the analysis cannot see them (#1706)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from strata.notebook.annotation_validation import validate_cell_annotations
from strata.notebook.annotations import parse_annotations
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

_SETUP = "import pandas as pd\nmu = 2\ndf = pd.DataFrame({'a': [1, 2, 3]})\n"
_QUERY = 'out = df.query("a > @mu")["a"].tolist()\nprint(out)\n'
_HINT = (
    "`mu` is defined by cell `c1` above, but this cell reads it where Strata cannot see it "
    "(inside a string, for example), so it was not loaded. Add `# @reads mu` at the top of "
    "the cell."
)


def _session(tmp_path: Path, cells: list[tuple[str, str]]) -> NotebookSession:
    nb = create_notebook(tmp_path, "Reads", initialize_environment=False)
    after = None
    for cell_id, source in cells:
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    return session


def _run(session: NotebookSession, cell_id: str):
    source = session.notebook_state.get_cell(cell_id).source
    return asyncio.run(CellExecutor(session).execute_cell(cell_id, source))


def _codes(session: NotebookSession, cell_id: str) -> list[str]:
    cell = session.notebook_state.get_cell(cell_id)
    return [
        d.code for d in validate_cell_annotations(cell, session.notebook_state, session.analysis)
    ]


def test_lines_stack_into_one_list_of_names():
    source = "# @reads mu, sig\n# @reads mu\nx = 1\n"

    assert parse_annotations(source, "python").reads == ["mu", "sig"]


@pytest.mark.parametrize("value", ["", ",", "1x", "mu sig-ma", "class", "mu None"])
def test_a_value_that_is_not_names_is_an_error(tmp_path, value):
    session = _session(tmp_path, [("c1", f"# @reads {value}\nx = 1\n")])

    assert "reads_unreadable" in _codes(session, "c1")


def test_the_names_are_references(tmp_path):
    session = _session(tmp_path, [("c1", _SETUP), ("c2", "# @reads mu\n" + _QUERY)])

    assert "mu" in session.analysis.cells["c2"].references
    assert "c1" in session.dag.cell_upstream["c2"]
    assert "mu" in session.dag.consumed_variables["c1"]


def test_a_declared_read_runs_and_follows_its_producer(tmp_path):
    session = _session(tmp_path, [("c1", _SETUP), ("c2", "# @reads mu\n" + _QUERY)])

    first = _run(session, "c2")
    assert first.success, first.error
    assert first.stdout.strip() == "[3]"

    write_cell(session.path, "c1", _SETUP.replace("mu = 2", "mu = 0"))
    session.reload()
    again = _run(session, "c2")

    assert again.success, again.error
    assert again.cache_hit is False
    assert again.stdout.strip() == "[1, 2, 3]"


def test_an_undeclared_read_names_the_annotation(tmp_path):
    session = _session(tmp_path, [("c1", _SETUP), ("c2", _QUERY)])

    result = _run(session, "c2")

    assert not result.success
    assert "UndefinedVariableError" in result.error
    assert result.error.endswith(_HINT)
    assert session.notebook_state.get_cell("c2").error.endswith(_HINT)
    # The WebSocket path applies a result again: the hint is there once.
    session.apply_execution_result_metadata("c2", result)
    assert result.error.count(_HINT) == 1


def test_a_name_error_for_a_name_above_names_it_too(tmp_path):
    session = _session(tmp_path, [("c1", "mu = 2\n"), ("c2", 'print(eval("mu"))\n')])

    result = _run(session, "c2")

    assert result.error.startswith("NameError: name 'mu' is not defined")
    assert result.error.endswith(_HINT)


@pytest.mark.parametrize(
    "cells",
    [
        # Nothing defines it.
        [("c1", "x = 1\n"), ("c2", 'print(eval("mu"))\n')],
        # Only a cell below defines it.
        [("c2", 'print(eval("mu"))\n'), ("c1", "mu = 2\n")],
    ],
)
def test_no_cell_above_defines_it_so_no_hint(tmp_path, cells):
    session = _session(tmp_path, cells)

    result = _run(session, "c2")

    assert not result.success
    assert "@reads" not in result.error


def test_run_all_follows_a_declared_read(tmp_path):
    """The case #1732 gets wrong without the annotation: a run-all, then an edit upstream."""
    cells = [("c1", "mu = 2\n"), ("c2", '# @reads mu\nout = eval("mu")\n'), ("c3", "print(out)\n")]
    session = _session(tmp_path, cells)
    executor = CellExecutor(session)
    consumed = session.dag.consumed_variables
    specs = [
        {
            "cell_id": cell_id,
            "source": source,
            "consumed_vars": sorted(consumed.get(cell_id, set())),
            "env": {},
            "mount_manifest": {},
            "source_hash": "",
            "env_hash": "",
        }
        for cell_id, source in cells
    ]
    assert asyncio.run(executor.execute_batch(specs)).completed

    write_cell(session.path, "c1", "mu = 3\n")
    session.reload()
    after = asyncio.run(executor.execute_cell("c3", "print(out)\n"))

    assert after.success, after.error
    assert after.stdout.strip() == "3"


def test_a_declared_read_holds_when_the_cell_assigns_the_name_after(tmp_path):
    session = _session(
        tmp_path, [("c1", "mu = 2\n"), ("c2", '# @reads mu\nprint(eval("mu"))\nmu = 5\n')]
    )

    assert "c1" in session.dag.cell_upstream["c2"]
    result = _run(session, "c2")
    assert result.success, result.error
    assert result.stdout.strip() == "2"


def _other_language_session(tmp_path: Path, language: str, source: str) -> NotebookSession:
    nb = create_notebook(tmp_path, "Reads", initialize_environment=False)
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", "mu = 2\n")
    add_cell_to_notebook(nb, "c2", "c1", language=language)
    write_cell(nb, "c2", source)
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    return session


@pytest.mark.parametrize(
    ("language", "source", "code"),
    [
        ("sql", "-- @sql connection=db\n-- @reads mu\nSELECT 1\n", "annotation_not_for_sql"),
        ("prompt", "# @reads mu\nhello\n", "reads_not_python"),
        ("r", "# @reads my.var\nx <- get('my.var')\n", "reads_not_python"),
    ],
)
def test_on_another_language_it_adds_no_edge_and_warns(tmp_path, language, source, code):
    session = _other_language_session(tmp_path, language, source)

    assert "mu" not in session.analysis.cells["c2"].references
    assert "c1" not in session.dag.cell_upstream.get("c2", [])
    codes = _codes(session, "c2")
    assert code in codes
    assert "reads_unreadable" not in codes


def test_a_missing_query_column_names_no_annotation(tmp_path):
    """pandas' bare ``name 'b'`` is a column the frame lacks; ``@reads b`` cannot fix it."""
    session = _session(tmp_path, [("c1", _SETUP + "b = 3\n"), ("c2", 'print(df.query("b > 1"))\n')])

    result = _run(session, "c2")

    assert not result.success
    assert "UndefinedVariableError" in result.error
    assert "@reads" not in result.error
