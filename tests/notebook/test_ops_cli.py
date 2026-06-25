"""Tests for the NotebookOps core + the `strata cell|dag|status` inspect CLI (P0).

Read-only, local backend — no server, no env sync. Builds a tiny two-cell
notebook with a real upstream→downstream edge and asserts the operation shapes
(which match the server's REST API) plus the CLI exit-code contract.
"""

from __future__ import annotations

import json

import pytest

from strata.cli import main
from strata.notebook.ops import CellView, LocalNotebookOps, NotebookOpsError
from tests.notebook.test_cli import _build_notebook


@pytest.fixture
def chain_nb(tmp_path):
    # a defines `x`; b consumes it → one DAG edge a→b.
    return _build_notebook(
        tmp_path,
        cells=[("a", "x = 1", None), ("b", "y = x + 1", "a")],
    )


def test_local_ops_list_cells(chain_nb):
    ops = LocalNotebookOps(chain_nb)
    cells = ops.list_cells()
    assert [c.id for c in cells] == ["a", "b"]
    assert all(isinstance(c, CellView) for c in cells)
    assert cells[0].source == "x = 1"


def test_local_ops_get_cell_and_unknown(chain_nb):
    ops = LocalNotebookOps(chain_nb)
    cell = ops.get_cell("a")
    assert isinstance(cell, CellView)
    assert cell.id == "a"
    assert cell.source == "x = 1"
    # Curated view drops internal bookkeeping — no provenance hashes leak through.
    assert "last_provenance_hash" not in cell.model_dump()
    with pytest.raises(NotebookOpsError):
        ops.get_cell("ghost")


def test_local_ops_dag_has_the_edge(chain_nb):
    dag = LocalNotebookOps(chain_nb).dag()
    assert any(
        e.from_cell_id == "a" and e.to_cell_id == "b" and e.variable == "x" for e in dag.edges
    )
    assert dag.topological_order.index("a") < dag.topological_order.index("b")


def test_local_ops_status_summary(chain_nb):
    status = LocalNotebookOps(chain_nb).status()
    assert status.name
    assert [c.id for c in status.cells] == ["a", "b"]
    assert all(isinstance(c.staleness_reasons, list) for c in status.cells)


def test_cli_cell_list_json(chain_nb, capsys):
    rc = main(["cell", "list", str(chain_nb), "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert [c["id"] for c in data] == ["a", "b"]
    assert data[0]["source"] == "x = 1"


def test_cli_cell_show_unknown_is_exit_1(chain_nb, capsys):
    rc = main(["cell", "show", str(chain_nb), "ghost", "--format", "json"])
    assert rc == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_cli_not_a_notebook_is_exit_2(tmp_path, capsys):
    rc = main(["cell", "list", str(tmp_path), "--format", "json"])
    assert rc == 2
    assert "not a Strata notebook" in capsys.readouterr().err


def test_cli_dag_and_status_json(chain_nb, capsys):
    assert main(["dag", str(chain_nb), "--format", "json"]) == 0
    dag = json.loads(capsys.readouterr().out)
    assert dag["edges"][0]["from_cell_id"] == "a"

    assert main(["status", str(chain_nb), "--format", "json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["notebook_id"]
