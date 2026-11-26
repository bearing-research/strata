"""A name no cell defines, below a SQL cell that outputs the default ``result`` (#1709)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from strata.notebook.annotation_validation import validate_cell_annotations
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

_SQL = "-- @sql connection=db\nSELECT 1 AS x\n"
_HINT = (
    "`top_routes` is not defined by any cell. The SQL cell `sql00001` above outputs "
    "`result`; add `-- @name top_routes` to it."
)


def _session(tmp_path: Path, cells: list[tuple[str, str, str]]) -> NotebookSession:
    nb = create_notebook(tmp_path, "Hint", initialize_environment=False)
    after = None
    for cell_id, language, source in cells:
        add_cell_to_notebook(nb, cell_id, after, language=language)
        write_cell(nb, cell_id, source)
        after = cell_id
    toml = nb / "notebook.toml"
    toml.write_text(toml.read_text() + '\n[connections.db]\ndriver = "duckdb"\npath = ":memory:"\n')
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    return session


def _warnings(session: NotebookSession, cell_id: str) -> list[str]:
    cell = session.notebook_state.get_cell(cell_id)
    return [
        d.message
        for d in validate_cell_annotations(cell, session.notebook_state, session.analysis)
        if d.code == "sql_result_unnamed"
    ]


def test_the_name_the_sql_cell_was_meant_to_have_is_named(tmp_path):
    session = _session(
        tmp_path,
        [("sql00001", "sql", _SQL), ("py000001", "python", "print(top_routes, strata)\n")],
    )

    assert _warnings(session, "py000001") == [_HINT]


@pytest.mark.parametrize(
    ("sql", "python"),
    [
        ("-- @sql connection=db\n-- @name top_routes\nSELECT 1 AS x\n", "print(top_routes)\n"),
        (_SQL, "print(result)\n"),
        (_SQL, "# @mount top_routes file:///tmp ro\nprint(top_routes)\n"),
        (_SQL, "from math import *\nprint(sqrt(2))\n"),
        (_SQL, "print(top_routes, pritn)\n"),
    ],
    ids=["named", "uses-result", "a-mount", "star-import", "two-names"],
)
def test_a_name_something_gives_is_left_alone(tmp_path, sql, python):
    session = _session(tmp_path, [("sql00001", "sql", sql), ("py000001", "python", python)])

    assert _warnings(session, "py000001") == []


def test_only_the_nearest_sql_cell_above_counts(tmp_path):
    named = "-- @sql connection=db\n-- @name routes\nSELECT 1 AS x\n"
    session = _session(
        tmp_path,
        [
            ("sql00001", "sql", _SQL),
            ("sql00002", "sql", named),
            ("py000001", "python", "print(top_route)\n"),
        ],
    )

    assert _warnings(session, "py000001") == []


def test_a_sql_cell_below_is_no_hint(tmp_path):
    session = _session(
        tmp_path, [("py000001", "python", "print(top_routes)\n"), ("sql00001", "sql", _SQL)]
    )

    assert _warnings(session, "py000001") == []


def test_the_run_error_says_it_too(tmp_path):
    session = _session(
        tmp_path, [("sql00001", "sql", _SQL), ("py000001", "python", "print(top_routes)\n")]
    )
    source = session.notebook_state.get_cell("py000001").source

    result = asyncio.run(CellExecutor(session).execute_cell("py000001", source))

    assert not result.success
    assert result.error.startswith("NameError: name 'top_routes' is not defined")
    assert result.error.endswith(_HINT)
    assert session.notebook_state.get_cell("py000001").error.endswith(_HINT)
    # The WebSocket path applies a result again: the hint is there once.
    session.apply_execution_result_metadata("py000001", result)
    assert result.error.count(_HINT) == 1


def test_the_run_reads_the_source_it_ran(tmp_path):
    """An edit the analysis has not caught up with still gets its hint."""
    session = _session(tmp_path, [("sql00001", "sql", _SQL), ("py000001", "python", "print(1)\n")])
    cell = session.notebook_state.get_cell("py000001")
    cell.source = "print(top_routes)\n"

    result = asyncio.run(CellExecutor(session).execute_cell("py000001", cell.source))

    assert result.error.endswith(_HINT)
