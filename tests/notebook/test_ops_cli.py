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


class _FakeExecutor:
    """Records which run mode was used and returns canned results (no subprocess)."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def _result(self, cell_id, *, success=True, cache_hit=False, error=None):
        from strata.notebook.executor import CellExecutionResult

        return CellExecutionResult(
            cell_id=cell_id,
            success=success,
            cache_hit=cache_hit,
            error=error,
            duration_ms=5.0,
            stdout="hi\n",
            execution_method="cold",
        )

    async def execute_cell(self, cell_id, source):
        self.calls.append(("normal", cell_id))
        return self._result(cell_id, cache_hit=True)

    async def execute_cell_rerun(self, cell_id, source):
        self.calls.append(("rerun", cell_id))
        return self._result(cell_id)

    async def execute_cell_force(self, cell_id, source):
        self.calls.append(("force", cell_id))
        return self._result(cell_id, success=False, error="boom")

    async def run_cell_tests(self, cell_id, test_source):
        from strata.notebook.models import CellTestCase, CellTestResult

        return CellTestResult(
            passed=1,
            failed=1,
            tests=[
                CellTestCase(name="t_ok", outcome="passed"),
                CellTestCase(name="t_bad", outcome="failed", message="assert 1 == 2"),
            ],
        )


@pytest.mark.asyncio
async def test_run_cell_dispatches_modes_and_maps(chain_nb, monkeypatch):
    ops = LocalNotebookOps(chain_nb)
    fake = _FakeExecutor()
    monkeypatch.setattr(ops, "_ensure_executor", lambda: fake)

    normal = await ops.run_cell("a")
    assert normal.cell_id == "a" and normal.status == "ok" and normal.stdout == "hi\n"
    await ops.run_cell("a", mode="rerun")
    forced = await ops.run_cell("a", mode="force")
    assert forced.status == "error" and forced.error == "boom"  # success=False → "error"
    assert [mode for mode, _ in fake.calls] == ["normal", "rerun", "force"]


@pytest.mark.asyncio
async def test_run_cell_errors(chain_nb, monkeypatch):
    ops = LocalNotebookOps(chain_nb)
    monkeypatch.setattr(ops, "_ensure_executor", lambda: _FakeExecutor())
    with pytest.raises(NotebookOpsError):
        await ops.run_cell("ghost")
    with pytest.raises(NotebookOpsError):
        await ops.run_cell("a", mode="bogus")


@pytest.mark.asyncio
async def test_run_tests_maps_and_requires_test_source(chain_nb, monkeypatch):
    ops = LocalNotebookOps(chain_nb)
    monkeypatch.setattr(ops, "_ensure_executor", lambda: _FakeExecutor())
    # Cell 'a' ships no cells/a.test.py → run_tests refuses.
    with pytest.raises(NotebookOpsError):
        await ops.run_tests("a")
    # Give it a test source; now it maps the executor's CellTestResult.
    ops._session.notebook_state.get_cell("a").test_source = "def test_x(cell): pass"
    result = await ops.run_tests("a")
    assert result.passed == 1 and result.failed == 1
    assert [c.name for c in result.cases] == ["t_ok", "t_bad"]
    assert result.cases[1].outcome == "failed" and "assert 1 == 2" in result.cases[1].message


def test_cli_dag_and_status_json(chain_nb, capsys):
    assert main(["dag", str(chain_nb), "--format", "json"]) == 0
    dag = json.loads(capsys.readouterr().out)
    assert dag["edges"][0]["from_cell_id"] == "a"

    assert main(["status", str(chain_nb), "--format", "json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["notebook_id"]
