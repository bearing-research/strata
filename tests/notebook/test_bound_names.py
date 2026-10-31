"""A name the run binds from an annotation is not an input from an earlier cell."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.models import MountMode, MountSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    update_notebook_mounts,
    write_cell,
)


def _session(tmp_path: Path, cells: dict[str, str], mounts=()) -> NotebookSession:
    nb = create_notebook(tmp_path, "Bound")
    after = None
    for cell_id, source in cells.items():
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    if mounts:
        update_notebook_mounts(nb, list(mounts))
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    return session


def _data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("a")
    return data


@pytest.mark.asyncio
async def test_editing_a_cell_that_defines_a_mount_name_leaves_the_mount_cell_ready(tmp_path):
    data = _data(tmp_path)
    mounted = f"# @mount data {data.as_uri()} ro\nn = len(list(data.iterdir()))\n"
    session = _session(tmp_path, {"c1": "data = 1\n", "c2": mounted, "c3": "m = n\n"})
    executor = CellExecutor(session)

    result = await executor.execute_cell("c2", mounted)
    assert result.success, result.error
    assert session.dag.cell_upstream["c2"] == []

    cell = session.notebook_state.get_cell("c1")
    cell.source = "data = 2\n"
    session.re_analyze_cell("c1")
    assert session.compute_staleness()["c2"].status.value == "ready"


@pytest.mark.parametrize(
    "annotation, names",
    [
        ("# @table trips file:///data/warehouse#nyc.trips", ["trips", "trips_snapshot"]),
        ("# @fetch zones https://example.org/taxi_zones.csv", ["zones"]),
        ("# @dataset model taxi/model@champion", ["model"]),
    ],
)
def test_an_annotation_bound_name_wires_no_edge(tmp_path, annotation, names):
    upstream = "".join(f"{name} = 1\n" for name in names)
    reader = f"{annotation}\nprint({', '.join(names)})\n"
    session = _session(tmp_path, {"c1": upstream, "c2": reader})

    assert session.dag.cell_upstream["c2"] == []
    assert session.notebook_state.get_cell("c2").references == []


def test_a_notebook_mount_name_wires_no_edge(tmp_path):
    mount = MountSpec(name="raw", uri=_data(tmp_path).as_uri(), mode=MountMode.READ_ONLY)
    session = _session(tmp_path, {"c1": "raw = 1\n", "c2": "print(raw)\n"}, mounts=[mount])

    assert session.dag.cell_upstream["c2"] == []


def test_a_plain_reference_still_wires_its_edge(tmp_path):
    data = _data(tmp_path)
    reader = f"# @mount data {data.as_uri()} ro\nprint(data, x)\n"
    session = _session(tmp_path, {"c1": "data = 1\nx = 2\n", "c2": reader})

    assert session.dag.cell_upstream["c2"] == ["c1"]
    assert session.notebook_state.get_cell("c2").references == ["x"]


@pytest.mark.parametrize(
    "annotation",
    ["# @mount input {uri} ro", "# @output_dir input"],
)
def test_a_bound_name_that_shadows_a_builtin_wires_no_edge(tmp_path, annotation):
    reader = f"{annotation.format(uri=_data(tmp_path).as_uri())}\nprint(input)\n"
    session = _session(tmp_path, {"c1": "input = 1\n", "c2": reader})

    assert session.dag.cell_upstream["c2"] == []
    assert session.notebook_state.get_cell("c2").builtin_references == ["print"]


def test_a_loop_until_on_a_mount_name_wires_no_edge(tmp_path):
    loop = (
        f"# @loop max_iter=2 carry=state\n# @loop_until len(list(data.iterdir())) > state\n"
        f"# @mount data {_data(tmp_path).as_uri()} ro\nstate = state + 1\n"
    )
    session = _session(tmp_path, {"seed": "state = 0\ndata = 1\n", "loop": loop})

    assert session.dag.wired_variables("loop", "seed") == {"state"}
