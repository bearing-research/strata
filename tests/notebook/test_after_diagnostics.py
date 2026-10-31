"""An ``@after`` that adds no edge is reported, and an error when ordering is lost."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from strata.notebook.annotation_validation import validate_cell_annotations
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _session(tmp_path: Path, cells: dict[str, str]) -> NotebookSession:
    nb = create_notebook(tmp_path, "After", initialize_environment=False)
    after = None
    for cell_id, source in cells.items():
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    return session


def _diagnostics(session: NotebookSession, cell_id: str) -> dict[str, str]:
    cell = session.notebook_state.get_cell(cell_id)
    return {
        d.code: d.severity
        for d in validate_cell_annotations(cell, session.notebook_state, session.dag)
    }


def _refused(session: NotebookSession, cell_id: str) -> str | None:
    source = session.notebook_state.get_cell(cell_id).source
    result = asyncio.run(CellExecutor(session).execute_cell(cell_id, source))
    return None if result.success else result.error_code


def test_an_unknown_cell_is_an_error_that_refuses_the_run(tmp_path):
    session = _session(tmp_path, {"c1": "# @after nosuchcell\nx = 1\n"})

    assert _diagnostics(session, "c1") == {"after_unknown_cell": "error"}
    assert _refused(session, "c1") == "after_unknown_cell"


def test_a_name_is_not_an_id(tmp_path):
    session = _session(
        tmp_path, {"setup": "# @name Setup\nx = 1\n", "q": "# @after Setup\ny = 1\n"}
    )

    assert "after_unknown_cell" in _diagnostics(session, "q")


def test_an_inactive_variant_is_an_error_that_refuses_the_run(tmp_path):
    session = _session(
        tmp_path,
        {
            "va": "# @variant model a\nm = 1\n",
            "vb": "# @variant model b\nm = 2\n",
            "q": "# @after vb\ny = 1\n",
        },
    )
    assert session.dag.inactive_cells == {"vb"}

    assert _diagnostics(session, "q") == {"after_inactive_variant": "error"}
    assert _refused(session, "q") == "after_inactive_variant"


def test_the_cell_itself_is_a_warning_and_still_runs(tmp_path):
    session = _session(tmp_path, {"c1": "# @after c1\nx = 1\n"})

    assert _diagnostics(session, "c1") == {"after_self": "warn"}
    assert session.dag.cell_upstream["c1"] == []
    assert _refused(session, "c1") is None


def test_a_valid_after_adds_its_ordering_edge_and_no_variable(tmp_path):
    session = _session(tmp_path, {"setup": "x = 1\n", "q": "# @after setup\ny = 1\n"})

    assert _diagnostics(session, "q") == {}
    assert session.dag.cell_upstream["q"] == ["setup"]
    assert session.dag.wired_variables("q", "setup") == set()
    assert session.dag.consumed_variables["setup"] == set()
