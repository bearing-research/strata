"""A scratchpad keeps every variable its cells define, so the first cell to read one
loads it instead of rerunning the cell that made it. These run real cells."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.ops import LocalNotebookOps
from strata.notebook.parser import parse_notebook
from strata.notebook.writer import create_notebook, update_notebook_worker


def _loads(log: Path) -> str:
    # The cell that made the value appends to *log* each time it runs.
    return f"with open({str(log)!r}, 'a') as f:\n    f.write('ran\\n')\nrows = list(range(5))\n"


async def _add_and_run(ops: LocalNotebookOps, source: str):
    cell = ops.add_cell(source)
    result = await ops.run_cell(cell.id)
    assert result.status == "ok", result.error
    return result


@pytest.mark.parametrize(("scratchpad", "runs"), [(True, 1), (False, 2)])
async def test_the_first_reader_loads_a_scratchpads_value(tmp_path, scratchpad, runs):
    log = tmp_path / "ran.log"
    path = create_notebook(tmp_path, "nb", scratchpad=scratchpad)
    with LocalNotebookOps(path) as ops:
        ops._session.refresh_environment_runtime()
        await _add_and_run(ops, _loads(log))
        first = await _add_and_run(ops, "total = sum(rows)\nprint(total)")
        second = await _add_and_run(ops, "print(len(rows))")
    assert "10" in first.stdout
    assert "5" in second.stdout
    # A plain notebook stores only what is read, so its first reader reran the maker.
    assert log.read_text().splitlines() == ["ran"] * runs


def test_a_scratchpad_says_so_in_its_config_and_keeps_saying_it(tmp_path):
    path = create_notebook(tmp_path, "nb", scratchpad=True)
    assert "scratchpad = true" in (path / "notebook.toml").read_text()
    update_notebook_worker(path, "local")
    assert parse_notebook(path).scratchpad

    plain = create_notebook(tmp_path, "plain")
    assert "scratchpad" not in (plain / "notebook.toml").read_text()
    assert not parse_notebook(plain).scratchpad


async def test_a_cell_with_one_value_read_keeps_the_rest(tmp_path):
    # Edited once a cell reads `rows`, the maker runs with `rows` consumed: `size` is kept
    # too, so a cell reading it later loads it.
    log = tmp_path / "ran.log"
    path = create_notebook(tmp_path, "nb", scratchpad=True)
    with LocalNotebookOps(path) as ops:
        maker = ops.add_cell(_loads(log))
        ops._session.refresh_environment_runtime()
        assert (await ops.run_cell(maker.id)).status == "ok"
        await _add_and_run(ops, "print(sum(rows))")
        ops.edit_cell(maker.id, _loads(log) + "size = len(rows)\n")
        assert (await ops.run_cell(maker.id)).status == "ok"
        late = await _add_and_run(ops, "print(size * 2)")
    assert "10" in late.stdout
    assert log.read_text().splitlines() == ["ran", "ran"]


async def test_a_kept_def_is_its_module_export(tmp_path):
    log = tmp_path / "ran.log"
    path = create_notebook(tmp_path, "nb", scratchpad=True)
    with LocalNotebookOps(path) as ops:
        ops._session.refresh_environment_runtime()
        maker = await _add_and_run(ops, _loads(log) + "def double(x):\n    return 2 * x\n")
        used = await _add_and_run(ops, "print(double(len(rows)))")
        kinds = {
            json.loads(version.transform_spec)["params"]["content_type"]
            for version in ops._session.artifact_manager.artifact_store.list_latest_by_id_prefix(
                f"nb_{ops._session.notebook_state.id}_cell_{maker.cell_id}_var_double"
            )
        }
    assert "10" in used.stdout
    assert log.read_text().splitlines() == ["ran"]
    assert kinds == {"module/cell"}


async def test_a_kept_def_that_cannot_be_shared_is_left_for_its_reader_to_find_out(tmp_path):
    # A lambda pickles, but no module export ships it. Kept as a pickle, its reader would
    # get it from the cache; it reruns the cell and is told, as in a plain notebook.
    path = create_notebook(tmp_path, "nb", scratchpad=True)
    with LocalNotebookOps(path) as ops:
        ops._session.refresh_environment_runtime()
        await _add_and_run(ops, "square = lambda x: x * x")
        cell = ops.add_cell("print(square(3))")
        refused = await ops.run_cell(cell.id)
    assert refused.status == "error"
    assert "cannot be shared across cells" in (refused.error or "")


async def test_run_all_keeps_what_a_single_run_keeps(tmp_path):
    from strata.notebook import ws
    from strata.notebook.writer import add_cell_to_notebook, write_cell

    log = tmp_path / "ran.log"
    path = create_notebook(tmp_path, "nb", scratchpad=True)
    maker = _loads(log) + "def double(x):\n    return 2 * x\n"
    for cell_id, source, after in (("a", maker, None), ("b", "other = 1", "a")):
        add_cell_to_notebook(path, cell_id, after)
        write_cell(path, cell_id, source)
    specs: list[dict] = []
    real_batch = CellExecutor.execute_batch

    async def execute_batch(self, cell_specs, **kwargs):
        specs.extend(cell_specs)
        return await real_batch(self, cell_specs, **kwargs)

    with (
        LocalNotebookOps(path) as ops,
        mock.patch.object(CellExecutor, "execute_batch", execute_batch),
    ):
        session = ops._session
        session.refresh_environment_runtime()
        await ws._execute_run_all(
            None, session, ["a", "b"], ws._ensure_execution_state(session.id), session.id
        )
    # Code goes as its module export: the harness does not pickle it.
    (kept,) = [spec["kept_vars"] for spec in specs if spec["cell_id"] == "a"]
    assert "rows" in kept
    assert "double" not in kept
    assert log.read_text().splitlines() == ["ran"]
    with LocalNotebookOps(path) as ops:
        ops._session.refresh_environment_runtime()
        late = await _add_and_run(ops, "print(len(rows))")
    assert "5" in late.stdout
    assert log.read_text().splitlines() == ["ran"]


def _kinds(ops: LocalNotebookOps, cell_id: str, name: str) -> set[str]:
    store = ops._session.artifact_manager.artifact_store
    prefix = f"nb_{ops._session.notebook_state.id}_cell_{cell_id}_var_{name}"
    return {
        json.loads(version.transform_spec)["params"]["content_type"]
        for version in store.list_latest_by_id_prefix(prefix)
    }


async def test_a_read_constant_beside_kept_code_stays_a_value(tmp_path):
    # A prompt or R cell can read it only as a value; it rides a module only beside code
    # something reads.
    path = create_notebook(tmp_path, "nb", scratchpad=True)
    with LocalNotebookOps(path) as ops:
        ops._session.refresh_environment_runtime()
        maker = ops.add_cell("LIMIT = 10\ndef double(x):\n    return 2 * x\n")
        ops._session.refresh_environment_runtime()
        # The maker runs with LIMIT read and `double` not.
        assert (await ops.run_cell(ops.add_cell("print(LIMIT)").id)).status == "ok"
        assert _kinds(ops, maker.id, "LIMIT") == {"json/object"}
        assert _kinds(ops, maker.id, "double") == {"module/cell"}


async def test_a_plain_notebook_still_refuses_a_cell_it_cannot_share(tmp_path):
    path = create_notebook(tmp_path, "nb")
    with LocalNotebookOps(path) as ops:
        ops._session.refresh_environment_runtime()
        await _add_and_run(ops, "LIMIT = 10\nsquare = lambda x: x * x")
        reader = ops.add_cell("print(LIMIT)")
        refused = await ops.run_cell(reader.id)
    assert refused.status == "error"
    assert "cannot be shared across cells" in (refused.error or "")


async def test_a_stochastic_run_keeps_only_what_is_read(tmp_path):
    # Keyed by the run, which no later reader's key matches.
    path = create_notebook(tmp_path, "nb", scratchpad=True)
    with LocalNotebookOps(path) as ops:
        ops._session.refresh_environment_runtime()
        maker = await _add_and_run(ops, "# @stochastic\nimport random\nx = random.random()\n")
        assert _kinds(ops, maker.cell_id, "x") == set()
