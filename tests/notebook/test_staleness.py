"""Tests for staleness detection."""

import pytest

from strata.notebook.models import CellState, NotebookState
from strata.notebook.session import NotebookSession


@pytest.fixture
def three_cell_notebook(tmp_path):
    """Create a 3-cell notebook with dependencies."""
    notebook_dir = tmp_path / "notebook"
    notebook_dir.mkdir()

    # Create cells directory
    cells_dir = notebook_dir / "cells"
    cells_dir.mkdir()

    # Create cell files
    (cells_dir / "load.py").write_text("df = [1, 2, 3]")
    (cells_dir / "clean.py").write_text("cleaned = [x for x in df]")
    (cells_dir / "explore.py").write_text("print(cleaned)")

    # Create pyproject.toml
    (notebook_dir / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    # Create NotebookState
    notebook_state = NotebookState(
        id="test_nb",
        name="Test",
        cells=[
            CellState(
                id="load",
                source="df = [1, 2, 3]",
                language="python",
                order=0,
            ),
            CellState(
                id="clean",
                source="cleaned = [x for x in df]",
                language="python",
                order=1,
            ),
            CellState(
                id="explore",
                source="print(cleaned)",
                language="python",
                order=2,
            ),
        ],
    )

    return notebook_dir, notebook_state


def test_fresh_notebook_all_idle(three_cell_notebook):
    """Fresh notebook → all cells should be idle."""
    notebook_dir, notebook_state = three_cell_notebook

    session = NotebookSession(notebook_state, notebook_dir)
    staleness = session.compute_staleness()

    # All cells should be idle initially (no cached artifacts), with no reasons.
    for cell_id, status in staleness.items():
        assert status.status == "idle"
        assert len(status.reasons) == 0


def test_staleness_updates_cells_when_dag_is_invalid(tmp_path):
    """When session.dag is None, compute_staleness still authoritatively
    marks cells idle and clears cached causality data.

    The single-pass DAG builder no longer produces cycles from plain
    cell sources (forward references simply leave the cell without an
    upstream edge), so we exercise the ``dag is None`` branch directly
    by nulling it after construction — simulating any future failure
    mode that leaves the session without a DAG.
    """
    notebook_dir = tmp_path / "null_dag_notebook"
    notebook_dir.mkdir()
    (notebook_dir / "cells").mkdir()
    (notebook_dir / "pyproject.toml").write_text("[project]\nname = 'null_dag'\n")

    notebook_state = NotebookState(
        id="null_dag_nb",
        name="NullDag",
        cells=[
            CellState(id="a", source="x = y + 1", language="python", order=0),
            CellState(id="b", source="y = x + 1", language="python", order=1),
        ],
    )

    session = NotebookSession(notebook_state, notebook_dir)
    session.dag = None  # simulate failed DAG build

    for cell in session.notebook_state.cells:
        cell.status = "ready"
        cell.cache_hit = True
    session.causality_map = {"a": object()}  # prove compute_staleness clears stale data

    staleness = session.compute_staleness()

    assert staleness["a"].status == "idle"
    assert staleness["b"].status == "idle"
    assert session.causality_map == {}
    for cell in session.notebook_state.cells:
        assert cell.status == "idle"
        assert cell.cache_hit is False
        assert cell.staleness is not None
        assert cell.staleness.status == "idle"
