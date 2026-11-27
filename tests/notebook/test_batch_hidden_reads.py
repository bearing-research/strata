"""A read the analysis cannot see behaves in run-all as it does in a cell run alone."""

from __future__ import annotations

from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

# ``mu`` reaches c2 only through a string, so c2 has no edge from c1.
CELLS = [("c1", "mu = 2\n"), ("c2", 'out = eval("mu")\n'), ("c3", "print(out)\n")]


def _session(tmp_path: Path, cells: list[tuple[str, str]] = CELLS) -> NotebookSession:
    notebook_dir = create_notebook(tmp_path, "HiddenReads")
    prev: str | None = None
    for cell_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=prev)
        write_cell(notebook_dir, cell_id, source)
        prev = cell_id
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def _specs(session: NotebookSession, cells: list[tuple[str, str]] = CELLS) -> list[dict]:
    consumed = session.dag.consumed_variables if session.dag else {}
    return [
        {
            "cell_id": cell_id,
            "source": source,
            "consumed_vars": sorted(consumed.get(cell_id, set())),
            "references": sorted(session.notebook_state.get_cell(cell_id).references),
            "env": {},
            "mount_manifest": {},
            "source_hash": "",
            "env_hash": "",
        }
        for cell_id, source in cells
    ]


@pytest.mark.asyncio
async def test_alone_a_hidden_read_fails(tmp_path: Path):
    session = _session(tmp_path)
    executor = CellExecutor(session)
    assert (await executor.execute_cell("c1", CELLS[0][1])).success
    alone = await executor.execute_cell("c2", CELLS[1][1])
    assert not alone.success
    assert "mu" in (alone.error or "")


@pytest.mark.asyncio
async def test_run_all_does_not_cache_a_value_read_through_a_hidden_name(tmp_path: Path):
    session = _session(tmp_path)
    executor = CellExecutor(session)
    batch = await executor.execute_batch(_specs(session))

    # As alone: c2 is not given ``mu``, so it fails and nothing is cached for it.
    by_id = {r.cell_id: r for r in batch.cell_results}
    assert by_id["c2"].status == "cell_error"
    assert "mu" in (by_id["c2"].error or "")
    assert "out" not in session.notebook_state.get_cell("c2").artifact_uris

    write_cell(session.path, "c1", "mu = 3\n")
    session.reload()
    assert (await executor.execute_cell("c1", "mu = 3\n")).success

    after = await executor.execute_cell("c3", CELLS[2][1])
    assert "2" not in after.stdout.split(), (
        f"c3 printed the value c2 read before mu changed: {after.stdout!r}"
    )


@pytest.mark.asyncio
async def test_a_function_keeps_the_names_of_the_cell_that_defined_it(tmp_path: Path):
    """c2 reads ``f`` and not ``helper``; ``f`` still finds the ``helper`` of its own cell."""
    cells = [
        ("c1", "helper = 10\ndef f():\n    return helper\n"),
        ("c2", "out = f()\n"),
        ("c3", "print(out)\n"),
    ]
    session = _session(tmp_path, cells)
    result = await CellExecutor(session).execute_batch(_specs(session, cells))

    assert result.completed, result.cell_results
    stdout = {r.cell_id: r.stdout for r in result.cell_results}
    assert stdout["c3"].strip() == "10"


@pytest.mark.asyncio
async def test_a_later_binding_reaches_the_cells_after_it(tmp_path: Path):
    cells = [("c1", "x = 1\n"), ("c2", "x = x + 1\n"), ("c3", "print(x)\n")]
    session = _session(tmp_path, cells)
    result = await CellExecutor(session).execute_batch(_specs(session, cells))

    assert result.completed, result.cell_results
    stdout = {r.cell_id: r.stdout for r in result.cell_results}
    assert stdout["c3"].strip() == "2"


async def _run_all(tmp_path: Path, cells: list[tuple[str, str]]):
    session = _session(tmp_path, cells)
    result = await CellExecutor(session).execute_batch(_specs(session, cells))
    assert result.completed, result.cell_results
    return session, {r.cell_id: r for r in result.cell_results}


@pytest.mark.asyncio
async def test_an_input_restored_after_an_in_place_change_reaches_earlier_functions(tmp_path):
    """c2 changes ``data`` where the analysis cannot see. A function in a container keeps
    the names of c1, which defined it, and sees the stored ``data``, as a single run of c3
    would (a function passed on alone is its stored copy already)."""
    _, by_id = await _run_all(
        tmp_path,
        [
            ("c1", "data = [1, 2]\nops = {'total': lambda: sum(data)}\n"),
            ("c2", "alias = data\nalias.append(3)\n"),
            ("c3", "print(ops['total']())\n"),
        ],
    )

    assert by_id["c3"].stdout.strip() == "3"


@pytest.mark.asyncio
async def test_an_output_named_like_a_display_helper_is_stored(tmp_path):
    session, by_id = await _run_all(tmp_path, [("c1", "display = 5\n"), ("c2", "print(display)\n")])

    assert by_id["c2"].stdout.strip() == "5"
    assert "display" in session.notebook_state.get_cell("c1").artifact_uris


@pytest.mark.asyncio
async def test_a_cell_that_binds_nothing_stores_no_other_cells_value(tmp_path):
    """As alone, c2 leaving ``y`` unset fails; c1's ``y`` is not stored as c2's."""
    cells = [("c1", "y = 1\n"), ("c2", "flag = False\nif flag:\n    y = 2\n"), ("c3", "print(y)\n")]
    session = _session(tmp_path, cells)
    result = await CellExecutor(session).execute_batch(_specs(session, cells))

    c2 = next(r for r in result.cell_results if r.cell_id == "c2")
    assert c2.status == "persist_failed"
    assert "y is not set when it finishes" in (c2.error or "")
    assert "y" not in session.notebook_state.get_cell("c2").artifact_uris


@pytest.mark.asyncio
async def test_deleting_an_input_warns_as_alone(tmp_path):
    _, by_id = await _run_all(tmp_path, [("c1", "df = [1]\n"), ("c2", "print(df)\ndel df\n")])

    assert any(w["var_name"] == "df" for w in by_id["c2"].mutation_warnings), by_id["c2"]
