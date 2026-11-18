"""A scratchpad cell is keyed on what its run read: a file it read changing makes it miss,
and the value it then makes gives the cells reading it a new key too. These run real cells."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from strata.notebook.code_keys import INDEX
from strata.notebook.ops import LocalNotebookOps
from strata.notebook.writer import create_notebook


def _maker(log: Path, read: str) -> str:
    # Appends to *log* each time it runs, then makes x from what it reads.
    return f"with open({str(log)!r}, 'a') as f:\n    f.write('ran\\n')\n{read}\n"


def _status(ops: LocalNotebookOps, cell_id: str):
    return next(row for row in ops.status().cells if row.id == cell_id)


@pytest.fixture
def scratch(tmp_path):
    path = create_notebook(tmp_path, "nb", scratchpad=True)
    with LocalNotebookOps(path) as ops:
        ops._session.refresh_environment_runtime()
        yield ops


async def _run(ops: LocalNotebookOps, cell_id: str):
    result = await ops.run_cell(cell_id)
    assert result.status == "ok", result.error
    return result


async def test_a_changed_file_reruns_the_cell_and_rekeys_its_readers(tmp_path, scratch):
    data = tmp_path / "data.txt"
    data.write_text("2")
    log = tmp_path / "ran.log"
    maker = scratch.add_cell(_maker(log, f"x = int(open({str(data)!r}).read())"))
    await _run(scratch, maker.id)
    user = scratch.add_cell("y = x * 10\nprint(y)")
    assert "20" in (await _run(scratch, user.id)).stdout

    data.write_text("3")
    stale = _status(scratch, maker.id)
    assert stale.status == "stale"
    assert stale.staleness_reasons == ["code"]
    assert stale.changed_code == str(data)

    # The reader reruns the maker, and its own key moved with the maker's.
    rerun = await _run(scratch, user.id)
    assert "30" in rerun.stdout
    assert not rerun.cache_hit

    # Put back, both find what they made before: what they show is from 3 until then.
    data.write_text("2")
    assert _status(scratch, maker.id).changed_code == str(data)
    back = await _run(scratch, user.id)
    assert "20" in back.stdout
    assert back.cache_hit
    assert log.read_text().splitlines() == ["ran", "ran"]


async def test_a_project_file_read_through_the_pinned_mount_restales_it(tmp_path):
    # The project mount is pinned, so its tree is never hashed; what a cell read through it
    # still keys the cell, as ``strata scratch`` sets the scratchpad up.
    config = tmp_path / "config.txt"
    config.write_text("a")
    path = create_notebook(
        tmp_path, "scratch", directory=".strata-scratch", project_mount="project", scratchpad=True
    )
    with LocalNotebookOps(path) as ops:
        ops._session.refresh_environment_runtime()
        cell = ops.add_cell('print((project / "config.txt").read_text())')
        assert "a" in (await _run(ops, cell.id)).stdout

        config.write_text("b")

        assert _status(ops, cell.id).changed_code == str(config.resolve())
        again = await _run(ops, cell.id)
    assert "b" in again.stdout
    assert not again.cache_hit


async def test_editing_a_module_it_imported_reruns_it(tmp_path, scratch):
    lib = tmp_path / "lib"
    lib.mkdir()
    module = lib / "geom.py"
    module.write_text("def area(side):\n    return side * side\n")
    log = tmp_path / "ran.log"
    cell = scratch.add_cell(
        _maker(
            log, f"import sys\nsys.path.insert(0, {str(lib)!r})\nimport geom\nprint(geom.area(3))"
        )
    )
    assert "9" in (await _run(scratch, cell.id)).stdout
    assert (await _run(scratch, cell.id)).cache_hit

    # Same size, its mtime put back: still a change.
    before = module.stat()
    module.write_text("def area(side):\n    return side + side\n")
    os.utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert _status(scratch, cell.id).changed_code == str(module)
    again = await _run(scratch, cell.id)
    assert "6" in again.stdout
    assert not again.cache_hit


async def test_a_module_that_appeared_where_an_import_looked_reruns_it(tmp_path, scratch):
    cell = scratch.add_cell(
        f"import sys\nsys.path.insert(0, {str(tmp_path)!r})\n"
        "try:\n    import fastpath\n    print('fast')\nexcept ImportError:\n    print('slow')\n"
    )
    assert "slow" in (await _run(scratch, cell.id)).stdout

    (tmp_path / "fastpath.py").write_text("")

    assert _status(scratch, cell.id).changed_code == str(tmp_path / "fastpath.py")
    assert "fast" in (await _run(scratch, cell.id)).stdout


async def test_losing_the_index_costs_a_rerun_only(tmp_path, scratch):
    data = tmp_path / "data.txt"
    data.write_text("2")
    log = tmp_path / "ran.log"
    cell = scratch.add_cell(_maker(log, f"print(open({str(data)!r}).read())"))
    await _run(scratch, cell.id)

    (scratch._session.path / INDEX).unlink()

    again = await _run(scratch, cell.id)
    assert "2" in again.stdout
    assert not again.cache_hit
    assert (await _run(scratch, cell.id)).cache_hit


async def test_a_run_whose_file_changed_as_it_ran_is_kept_for_itself(tmp_path, scratch):
    data = tmp_path / "data.txt"
    data.write_text("2")
    cell = scratch.add_cell(f"x = open({str(data)!r}).read()\nopen({str(data)!r}, 'a').write('!')")

    result = await _run(scratch, cell.id)

    assert f"cache: skipped, the cell wrote {data}, which it had read" in result.stderr
    assert not (await _run(scratch, cell.id)).cache_hit


async def test_a_notebook_records_nothing_and_keeps_its_keys(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("2")
    path = create_notebook(tmp_path, "nb")
    with LocalNotebookOps(path) as ops:
        ops._session.refresh_environment_runtime()
        cell = ops.add_cell(f"print(open({str(data)!r}).read())")
        await _run(ops, cell.id)
        data.write_text("3")

        assert _status(ops, cell.id).status == "ready"
        assert (await _run(ops, cell.id)).cache_hit
    assert not (path / INDEX).exists()
