"""A cell whose source file is gone is an error, not an empty cell.

``notebook.toml`` lists the cell, but ``cells/<file>`` was deleted or never committed: the
parser read it as empty, ``strata validate`` passed, and a run "succeeded" with no code.
"""

from __future__ import annotations

import argparse
import json

from strata.notebook.annotation_validation import blocking_diagnostic
from strata.notebook.cli import validate_main
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _notebook(tmp_path):
    notebook_dir = create_notebook(tmp_path, "Gone", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1", None)
    write_cell(notebook_dir, "c1", "x = 1\n")
    (notebook_dir / "cells" / "c1.py").unlink()
    return notebook_dir


def test_validate_reports_the_missing_file(tmp_path, capsys):
    notebook_dir = _notebook(tmp_path)

    assert validate_main(argparse.Namespace(path=str(notebook_dir), format="json")) == 1

    cell = json.loads(capsys.readouterr().out)["cells"][0]
    assert [d["code"] for d in cell["diagnostics"]] == ["cell_file_missing"]
    assert "cells/c1.py" in cell["diagnostics"][0]["message"]


async def test_a_run_refuses_the_cell(tmp_path):
    notebook_dir = _notebook(tmp_path)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)

    result = await CellExecutor(session).execute_cell("c1", "")

    assert not result.success
    assert result.error_code == "cell_file_missing"


def test_writing_the_cell_clears_it(tmp_path):
    notebook_dir = _notebook(tmp_path)
    state = parse_notebook(notebook_dir)
    assert blocking_diagnostic(state.cells[0], state) is not None

    write_cell(notebook_dir, "c1", "x = 2\n")
    state = parse_notebook(notebook_dir)

    assert blocking_diagnostic(state.cells[0], state) is None


def test_an_empty_file_is_an_empty_cell(tmp_path):
    notebook_dir = _notebook(tmp_path)
    (notebook_dir / "cells" / "c1.py").write_text("")
    state = parse_notebook(notebook_dir)

    assert blocking_diagnostic(state.cells[0], state) is None


def test_a_rest_edit_clears_it_for_every_tab(tmp_path, monkeypatch):
    """An agent's edit goes over REST; the cell it sends back is no longer missing."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from strata.notebook.routes import router
    from tests.notebook.test_routes import open_session_id

    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    notebook_dir = _notebook(tmp_path)
    session_id = open_session_id(client, notebook_dir)

    response = client.put(f"/v1/notebooks/{session_id}/cells/c1", json={"source": "x = 1\n"})

    assert response.status_code == 200, response.text
    assert response.json()["cell"]["annotation_diagnostics"] == []
    assert response.json()["cell"]["file_missing"] is None


def test_an_agent_sees_the_missing_file(tmp_path):
    from strata.notebook.ops import _cell_view

    state = parse_notebook(_notebook(tmp_path))

    assert _cell_view(state.cells[0]).file_missing == "cells/c1.py"


def test_a_directory_where_the_file_belongs_is_reported_too(tmp_path):
    notebook_dir = _notebook(tmp_path)
    (notebook_dir / "cells" / "c1.py").mkdir()

    state = parse_notebook(notebook_dir)

    assert blocking_diagnostic(state.cells[0], state).code == "cell_file_missing"
