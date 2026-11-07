"""An inactive switch member: cellgraph neither wires nor keys it, so it is refused."""

from __future__ import annotations

from pathlib import Path

import cellgraph as cg
import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.keys import InputNotFingerprinted, cell_key
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

SWITCH = [("a", "# @variant g a\nx = 1\n"), ("b", "# @variant g b\nx = 2\n"), ("r", "y = x\n")]
REFUSAL = "'b' is not the active variant of group 'g'; make it active to run it"


def _session(tmp_path: Path, cells: list[tuple[str, str]], language: str = "python"):
    notebook_dir = create_notebook(tmp_path, "Variants")
    previous: str | None = None
    for cell_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=previous, language=language)
        write_cell(notebook_dir, cell_id, source)
        previous = cell_id
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


@pytest.mark.asyncio
async def test_running_an_inactive_variant_is_refused_and_not_kept(tmp_path: Path):
    session = _session(tmp_path, SWITCH)
    assert session.dag is not None and "b" in session.dag.inactive_cells

    result = await CellExecutor(session).execute_cell("b", SWITCH[1][1])

    assert (result.success, result.error) == (False, REFUSAL)
    # Making it active, not an edit, clears the refusal, so it is not the cell's error.
    cell = session.notebook_state.get_cell("b")
    assert cell is not None and cell.current_error() is None


@pytest.mark.asyncio
async def test_the_source_being_run_decides_what_is_inactive(tmp_path: Path):
    # Made a member of g before the editor saved it: refused, not keyed (which raises).
    session = _session(tmp_path, [SWITCH[0], ("b", "z = 3\n")])
    result = await CellExecutor(session).execute_cell("b", "# @variant g b\nz = 3\n")
    assert (result.success, result.error) == (False, REFUSAL)

    # Taken out of the group the same way: it runs.
    session = _session(tmp_path / "out", SWITCH)
    result = await CellExecutor(session).execute_cell("b", "x = 2\n")
    assert result.success, result.error


@pytest.mark.asyncio
async def test_an_inactive_variants_tests_are_refused(tmp_path: Path):
    session = _session(tmp_path, SWITCH)

    result = await CellExecutor(session).run_cell_tests("b", "def test_x():\n    assert x == 2\n")

    assert result.errored == 1
    assert [(t.name, t.message) for t in result.tests] == [("<refused>", REFUSAL)]


def test_a_fallback_key_takes_an_item_key_with_a_colon(tmp_path: Path):
    # A cycle leaves no DAG, and the fallback key derives each item's key from its own.
    session = _session(tmp_path, [("c0", "# @after c1\na = 1\n"), ("c1", "b = a\n")])
    analysis = session.analysis
    assert analysis is not None and analysis.structural_errors
    keys = {
        cell_key(
            session,
            analysis,
            "c0",
            mounts=[],
            tables=[],
            fetches=[],
            datasets=[],
            env_hash="e",
            instance=cg.Item(key, "d"),
        ).provenance_hash
        for key in ("https://a", "https://b")
    }
    assert len(keys) == 2


def test_a_host_keyed_cell_with_an_unreadable_upstream_has_no_key(tmp_path: Path):
    session = _session(tmp_path, [("c0", "x = 1\n")])
    add_cell_to_notebook(session.path, "q", after_cell_id="c0", language="sql")
    write_cell(session.path, "q", "-- @sql connection=db\nSELECT :x\n")
    session = NotebookSession(parse_notebook(session.path), session.path)
    analysis = session.analysis
    assert analysis is not None
    # c0 never ran, so the artifact its URI names is not in the store.
    c0 = session.notebook_state.get_cell("c0")
    assert c0 is not None
    c0.artifact_uris = {"x": f"strata://artifact/nb_{session.notebook_state.id}_cell_c0_var_x@v=1"}
    with pytest.raises(InputNotFingerprinted, match="output:c0.x"):
        cell_key(
            session, analysis, "q", mounts=[], tables=[], fetches=[], datasets=[], env_hash="e"
        )
