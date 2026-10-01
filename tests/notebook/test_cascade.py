"""Tests for cascade execution planner."""

import tempfile
from pathlib import Path

import pytest

from strata.notebook.cascade import CascadePlan, CascadePlanner, CascadeReason, CascadeStep
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.fixture
def temp_pipeline():
    """Create a 4-cell pipeline for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir, "test_pipeline")

        # root -> middle1 -> middle2 -> leaf
        cells_data = [
            ("root", "x = 1"),
            ("middle1", "y = x + 1"),
            ("middle2", "z = y + 1"),
            ("leaf", "w = z + 1"),
        ]

        for cell_id, source in cells_data:
            add_cell_to_notebook(notebook_dir, cell_id)
            write_cell(notebook_dir, cell_id, source)

        notebook_state = parse_notebook(notebook_dir)
        session = NotebookSession(notebook_state, notebook_dir)

        yield session, notebook_dir


def test_cascade_planner_no_cascade_needed(temp_pipeline):
    """Test that no cascade is needed when upstream is ready."""
    session, _ = temp_pipeline

    session.notebook_state.cells[0].status = "ready"

    planner = CascadePlanner(session)

    plan = planner.plan(session.notebook_state.cells[1].id)

    # Root is ready, so no cascade needed for the second cell
    assert plan is None


def test_cascade_planner_cascade_needed(temp_pipeline):
    """Test that cascade is detected when upstream is stale."""
    session, _ = temp_pipeline

    session.notebook_state.cells[0].status = "stale"

    planner = CascadePlanner(session)

    plan = planner.plan(session.notebook_state.cells[2].id)  # middle2

    assert plan is not None, "Expected cascade plan for stale upstream"
    assert plan.target_cell_id == session.notebook_state.cells[2].id
    assert len(plan.steps) > 0


def test_cascade_plan_structure(temp_pipeline):
    """Test that cascade plan has correct structure."""
    session, _ = temp_pipeline

    planner = CascadePlanner(session)

    session.notebook_state.cells[0].status = "stale"

    plan = planner.plan(session.notebook_state.cells[2].id)

    assert plan is not None, "Expected cascade plan for stale upstream"
    assert plan.plan_id
    assert plan.target_cell_id
    assert isinstance(plan.steps, list)
    assert plan.estimated_duration_ms >= 0

    for step in plan.steps:
        assert step.cell_id
        assert step.cell_name
        assert step.reason in ["stale", "missing", "target"]
        assert isinstance(step.skip, bool)


def test_cascade_plan_topological_order(temp_pipeline):
    """Test that cascade plan steps are in topological order."""
    session, _ = temp_pipeline

    session.notebook_state.cells[0].status = "stale"

    planner = CascadePlanner(session)
    plan = planner.plan(session.notebook_state.cells[-1].id)  # leaf

    assert plan is not None, "Expected cascade plan for stale upstream"
    if len(plan.steps) > 1:
        step_indices = {step.cell_id: i for i, step in enumerate(plan.steps)}

        for i, step in enumerate(plan.steps):
            cell = next(
                (c for c in session.notebook_state.cells if c.id == step.cell_id),
                None,
            )
            if cell:
                for upstream_id in cell.upstream_ids:
                    if upstream_id in step_indices:
                        assert step_indices[upstream_id] < i, (
                            f"{upstream_id} should come before {step.cell_id}"
                        )


def test_cascade_plan_includes_target(temp_pipeline):
    """Test that cascade plan always includes the target cell."""
    session, _ = temp_pipeline

    planner = CascadePlanner(session)

    session.notebook_state.cells[0].status = "stale"

    target_cell_id = session.notebook_state.cells[-1].id
    plan = planner.plan(target_cell_id)

    assert plan is not None, "Expected cascade plan for stale upstream"
    cell_ids = [step.cell_id for step in plan.steps]
    assert target_cell_id in cell_ids, "Target cell should be included in cascade plan"

    target_step = next((s for s in plan.steps if s.cell_id == target_cell_id), None)
    assert target_step is not None
    assert target_step.reason == "target"


def test_cascade_plan_skip_ready_cells(temp_pipeline):
    """Test that ready (cached) cells are marked to skip."""
    session, _ = temp_pipeline

    session.notebook_state.cells[0].status = "ready"

    planner = CascadePlanner(session)

    session.notebook_state.cells[1].status = "stale"

    plan = planner.plan(session.notebook_state.cells[-1].id)

    assert plan is not None, "Expected cascade plan for stale upstream"
    # The ready cell may be marked skip or left out of the plan entirely.
    ready_cell_step = next(
        (s for s in plan.steps if s.cell_id == session.notebook_state.cells[0].id),
        None,
    )
    if ready_cell_step is not None:
        assert ready_cell_step.skip, "Ready cell step should be marked to skip"


def test_cascade_uses_last_non_cached_duration(temp_pipeline):
    """Cached reruns should not collapse future cascade duration estimates."""
    session, _ = temp_pipeline

    session.record_execution("root", 250, cache_hit=False)
    session.record_execution("root", 5, cache_hit=True)
    session.notebook_state.cells[0].status = "stale"

    planner = CascadePlanner(session)
    plan = planner.plan(session.notebook_state.cells[1].id)

    assert plan is not None
    root_step = next(s for s in plan.steps if s.cell_id == "root")
    assert root_step.estimated_ms == 250


def test_cascade_planner_no_dag(temp_pipeline):
    """Test cascade planner behavior when DAG is None."""
    session, _ = temp_pipeline
    session.dag = None  # Simulate no DAG

    planner = CascadePlanner(session)
    plan = planner.plan(session.notebook_state.cells[0].id)

    assert plan is None


def test_cascade_step_initialization():
    """Test CascadeStep initialization."""
    step = CascadeStep(
        cell_id="test_cell",
        cell_name="Test Cell",
        reason=CascadeReason.STALE,
        skip=False,
        estimated_ms=100,
    )

    assert step.cell_id == "test_cell"
    assert step.cell_name == "Test Cell"
    assert step.reason == CascadeReason.STALE
    assert step.skip is False
    assert step.estimated_ms == 100


def test_cascade_plan_initialization():
    """Test CascadePlan initialization."""
    steps = [
        CascadeStep(cell_id="c1", cell_name="C1"),
        CascadeStep(cell_id="c2", cell_name="C2"),
    ]

    plan = CascadePlan(
        plan_id="test_plan",
        target_cell_id="c2",
        steps=steps,
        estimated_duration_ms=500,
    )

    assert plan.plan_id == "test_plan"
    assert plan.target_cell_id == "c2"
    assert len(plan.steps) == 2
    assert plan.estimated_duration_ms == 500


def test_cascade_plan_auto_generated_id():
    """Test that CascadePlan generates ID if not provided."""
    plan = CascadePlan(plan_id="", target_cell_id="test")

    assert plan.plan_id
    assert len(plan.plan_id) > 0
