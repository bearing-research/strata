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


def test_add_cell_inserts_and_validates(chain_nb):
    ops = LocalNotebookOps(chain_nb)
    cell = ops.add_cell("z = 9", after="a", language="python")
    assert cell.source == "z = 9"
    ids = [c.id for c in ops.list_cells()]
    assert ids[ids.index("a") + 1] == cell.id  # inserted right after a
    with pytest.raises(NotebookOpsError):
        ops.add_cell("x", after="ghost")
    with pytest.raises(NotebookOpsError):
        ops.add_cell("x", language="cobol")


def test_set_cell_tests_persists(chain_nb):
    ops = LocalNotebookOps(chain_nb)
    src = "def test_x(cell):\n    assert cell.x == 1\n"
    cell = ops.set_cell_tests("a", src)
    assert cell.id == "a"
    # written to its committed sibling file + reflected in the live session
    assert (chain_nb / "cells" / "a.test.py").read_text() == src
    assert ops._session.notebook_state.get_cell("a").test_source == src
    with pytest.raises(NotebookOpsError):
        ops.set_cell_tests("ghost", "x")


def test_edit_cell_persists(chain_nb):
    ops = LocalNotebookOps(chain_nb)
    updated = ops.edit_cell("a", "x = 999")
    assert updated.source == "x = 999"
    # A fresh open sees the written source (it went to disk, not just memory).
    assert LocalNotebookOps(chain_nb).get_cell("a").source == "x = 999"
    with pytest.raises(NotebookOpsError):
        ops.edit_cell("ghost", "y")


def test_remove_and_move_cell(chain_nb):
    ops = LocalNotebookOps(chain_nb)
    moved = ops.move_cell("a", 1)
    assert [c.id for c in moved] == ["b", "a"]
    ops.remove_cell("b")
    assert [c.id for c in ops.list_cells()] == ["a"]
    with pytest.raises(NotebookOpsError):
        ops.remove_cell("ghost")
    with pytest.raises(NotebookOpsError):
        ops.move_cell("ghost", 0)


@pytest.mark.asyncio
async def test_add_dependency_maps_result(chain_nb, monkeypatch):
    ops = LocalNotebookOps(chain_nb)

    class _Result:
        success, package, action, error, lockfile_changed = True, "pandas", "add", None, True

    class _Outcome:
        result = _Result()

    async def _fake(package, *, action):
        assert package == "pandas" and action == "add"
        return _Outcome()

    monkeypatch.setattr(ops._session, "mutate_dependency", _fake)
    res = await ops.add_dependency("pandas")
    assert res.success and res.package == "pandas" and res.action == "add"
    assert res.lockfile_changed and res.error is None


def test_cli_cell_add_then_rm(chain_nb, tmp_path, capsys):
    src = tmp_path / "s.py"
    src.write_text("w = 5")
    assert main(["cell", "add", str(chain_nb), "--file", str(src), "--format", "json"]) == 0
    new = json.loads(capsys.readouterr().out)
    assert new["source"] == "w = 5"
    assert main(["cell", "rm", str(chain_nb), new["id"], "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"removed": new["id"]}


def test_cli_cell_show_var_defined(chain_nb, capsys):
    # chain_nb: cell `a` defines x, cell `b` defines y (= x + 1).
    assert main(["cell", "show", str(chain_nb), "--var", "x", "--format", "json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["defined"] is True and d["defined_in"] == "a"
    assert d["cell"]["source"] == "x = 1"


def test_cli_cell_show_var_undefined_lists_available(chain_nb, capsys):
    assert main(["cell", "show", str(chain_nb), "--var", "nope", "--format", "json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["defined"] is False
    assert d["available"] == ["x", "y"]


def test_cell_show_var_sweep_vs_dangling_producer(capsys):
    from strata.notebook.cli import _cell_show_var
    from strata.notebook.ops import NotebookOpsError

    class _Dag:
        def __init__(self, vp):
            self.variable_producer = vp

    class _FakeOps:
        def __init__(self, vp, cells):
            self._vp, self._cells = vp, cells

        def dag(self):
            return _Dag(self._vp)

        def get_cell(self, cid):
            if cid not in self._cells:
                raise NotebookOpsError(f"no cell {cid!r}")
            return self._cells[cid]

    # A sweep-group producer has no single cell — reported as a pointer, no get_cell.
    assert _cell_show_var(_FakeOps({"m": "sweep:grp"}, {}), "m", "json") == 0
    assert json.loads(capsys.readouterr().out) == {
        "variable": "m",
        "defined": True,
        "defined_in": "sweep:grp",
    }
    # A plain producer id get_cell can't fetch is a real error, not masked as defined.
    assert _cell_show_var(_FakeOps({"g": "ghost"}, {}), "g", "json") == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_cli_cell_show_requires_exactly_one_of_id_or_var(chain_nb, capsys):
    # Neither a cell id nor --var.
    assert main(["cell", "show", str(chain_nb), "--format", "json"]) == 2
    # Both.
    assert main(["cell", "show", str(chain_nb), "a", "--var", "x", "--format", "json"]) == 2


def test_cli_cell_add_run_includes_post_run_outputs(chain_nb, monkeypatch, capsys):
    # add --run must return the POST-run cell view so its rendered outputs (a
    # trailing bare expression's value) ride along — not the pre-run view.
    from strata.notebook import cli as cli_mod
    from strata.notebook.ops import CellView, OutputView, RunResult

    pre = CellView(
        id="new",
        name="",
        language="python",
        status="ready",
        source="1 + 1",
        staleness_reasons=[],
        upstream_ids=[],
        downstream_ids=[],
        outputs=[],
        console_stdout="",
        console_stderr="",
    )
    post = pre.model_copy(update={"outputs": [OutputView(content_type="json/object", preview=2)]})

    class _FakeOps:
        def add_cell(self, source, after=None, language="python"):
            return pre

        async def run_cell(self, cid, mode="normal"):
            return RunResult(
                cell_id=cid,
                status="ok",
                cache_hit=False,
                execution_method="subprocess",
                duration_ms=1.0,
                stdout="",
            )

        def get_cell(self, cid):
            return post

        async def aclose(self):
            pass

    monkeypatch.setattr(cli_mod, "_open_read_ops", lambda args: _FakeOps())

    async def _no_env(ops, args):
        return 0

    monkeypatch.setattr(cli_mod, "_prepare_env_for_ops", _no_env)

    rc = main(["cell", "add", str(chain_nb), "-c", "1 + 1", "--run", "--format", "json"])
    assert rc == 0
    d = json.loads(capsys.readouterr().out)
    assert d["outputs"] == [o.model_dump(mode="json") for o in post.outputs]
    assert d["run"]["status"] == "ok"


def test_cli_cell_add_inline_c(chain_nb, capsys):
    # `-c` supplies the cell source inline, as an alternative to `--file`.
    assert main(["cell", "add", str(chain_nb), "-c", "w = 7", "--format", "json"]) == 0
    new = json.loads(capsys.readouterr().out)
    assert new["source"] == "w = 7"


def test_cli_cell_add_c_and_file_are_mutually_exclusive(chain_nb, tmp_path):
    src = tmp_path / "s.py"
    src.write_text("w = 5")
    with pytest.raises(SystemExit):
        main(["cell", "add", str(chain_nb), "-c", "w = 7", "--file", str(src)])


def test_cli_cell_add_requires_a_source(chain_nb):
    # Neither -c nor --file → the required mutually-exclusive group rejects it.
    with pytest.raises(SystemExit):
        main(["cell", "add", str(chain_nb)])


def test_cli_cell_annotate_set_and_unset(chain_nb, capsys):
    # Set two annotations on cell `a`; the @name is reflected back in the view.
    rc = main(
        [
            "cell",
            "annotate",
            str(chain_nb),
            "a",
            "--set",
            "name=loader",
            "--set",
            "worker=gpu",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    cell = json.loads(capsys.readouterr().out)
    assert cell["name"] == "loader"
    assert "# @worker gpu" in cell["source"] and cell["source"].endswith("x = 1")

    # Unset worker; it persists to disk and the body survives.
    assert (
        main(["cell", "annotate", str(chain_nb), "a", "--unset", "worker", "--format", "json"]) == 0
    )
    from strata.notebook.ops import LocalNotebookOps

    reopened = LocalNotebookOps(chain_nb).get_cell("a")
    assert reopened.name == "loader" and "# @worker" not in reopened.source


def test_cli_cell_annotate_requires_an_op(chain_nb, capsys):
    rc = main(["cell", "annotate", str(chain_nb), "a", "--format", "json"])
    assert rc == 2
    assert "--set" in capsys.readouterr().err


def test_cli_cell_annotate_bad_set_is_exit_2(chain_nb, capsys):
    rc = main(["cell", "annotate", str(chain_nb), "a", "--set", "noequals", "--format", "json"])
    assert rc == 2
    assert "KEY=VALUE" in capsys.readouterr().err


def test_cli_cell_annotate_repeatable_key_is_exit_2(chain_nb, capsys):
    # @env is repeatable — splicing one line would clobber others, so it's refused.
    rc = main(["cell", "annotate", str(chain_nb), "a", "--set", "env=A=1", "--format", "json"])
    assert rc == 2
    assert "repeatable" in capsys.readouterr().err


def test_cli_dag_and_status_json(chain_nb, capsys):
    assert main(["dag", str(chain_nb), "--format", "json"]) == 0
    dag = json.loads(capsys.readouterr().out)
    assert dag["edges"][0]["from_cell_id"] == "a"

    assert main(["status", str(chain_nb), "--format", "json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["notebook_id"]


class TestMoveCellDoesNotLoseConcurrentEdits:
    """``move_cell`` built its order from the snapshot taken when the ops
    object was constructed, and ``reorder_cells`` then wrote only those ids.

    The documented ``strata agent`` workflow has a server session open on the
    notebook while an agent drives the CLI, so this is the normal case, not a
    corner: the human adds a cell, the agent's next reorder erases it from
    committed config and orphans ``cells/<id>.py``.
    """

    def _notebook(self, tmp_path):
        from strata.notebook.writer import create_notebook

        return create_notebook(tmp_path, name="nb", initialize_environment=False)

    def test_a_cell_added_by_someone_else_survives_a_move(self, tmp_path):
        import tomllib

        from strata.notebook.ops import LocalNotebookOps

        notebook_dir = self._notebook(tmp_path)
        agent = LocalNotebookOps(notebook_dir)
        ids = [agent.add_cell(source=f"x{i} = {i}").id for i in range(3)]

        # A live session / TUI / second CLI process adds a cell.
        other = LocalNotebookOps(notebook_dir)
        added = other.add_cell(source="added_elsewhere = 1").id

        # The agent reorders using its now-stale view.
        agent.move_cell(ids[2], 0)

        with open(notebook_dir / "notebook.toml", "rb") as handle:
            on_disk = [c["id"] for c in tomllib.load(handle)["cells"]]
        assert added in on_disk, "an unrelated reorder deleted a cell"
        assert (notebook_dir / "cells" / f"{added}.py").exists()

    def test_the_requested_move_still_happens(self, tmp_path):
        from strata.notebook.ops import LocalNotebookOps

        notebook_dir = self._notebook(tmp_path)
        ops = LocalNotebookOps(notebook_dir)
        ids = [ops.add_cell(source=f"x{i} = {i}").id for i in range(3)]

        ops.move_cell(ids[2], 0)

        assert [c.id for c in ops.list_cells()][0] == ids[2]
