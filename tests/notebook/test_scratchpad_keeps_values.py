"""A scratchpad keeps every variable its cells define, so the first cell to read one
loads it instead of rerunning the cell that made it. These run real cells."""

from __future__ import annotations

from pathlib import Path

import pytest

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
