"""Cascade execution planner for notebooks.

When a user runs a cell with stale upstream inputs, we offer to run
the required upstream cells automatically (cascade execution).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from strata.notebook.dag import reachable_via
from strata.notebook.models import CellStatus

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession


class CascadeReason(StrEnum):
    """Why a cell is included in a cascade plan."""

    STALE = "stale"
    MISSING = "missing"
    TARGET = "target"


@dataclass
class CascadeStep:
    """A single cell in a cascade plan.

    Attributes:
        cell_id: ID of the cell to run
        cell_name: Display name of the cell (for the UI)
        reason: Why this cell needs to run (stale, missing, or target)
        skip: If True, cell can be skipped (e.g., cache hit)
        estimated_ms: Estimated execution time
    """

    cell_id: str
    cell_name: str
    reason: CascadeReason = CascadeReason.MISSING
    skip: bool = False
    estimated_ms: int = 0


@dataclass
class CascadePlan:
    """Plan for cascading execution.

    Attributes:
        plan_id: Unique ID for this plan
        target_cell_id: The cell the user wants to run
        steps: Cells to run, in topological order
        estimated_duration_ms: Total estimated duration
    """

    plan_id: str
    target_cell_id: str
    steps: list[CascadeStep] = field(default_factory=list)
    estimated_duration_ms: int = 0

    def __post_init__(self):
        """Generate plan_id if not provided."""
        if not self.plan_id:
            self.plan_id = str(uuid.uuid4())[:8]


class CascadePlanner:
    """Plans and executes cascades.

    A cascade is triggered when a user tries to run a cell whose inputs
    are not all ready (some are stale or missing). The planner determines
    which upstream cells need to run first.
    """

    def __init__(self, session: NotebookSession):
        """Initialize planner for a session.

        Args:
            session: NotebookSession instance
        """
        self.session = session

    def plan(self, cell_id: str) -> CascadePlan | None:
        """Check if a cell needs upstream execution.

        Args:
            cell_id: ID of the cell to run

        Returns:
            CascadePlan if upstream cells need to run, None if cell can run immediately
        """
        if not self.session.dag:
            # No DAG — can't plan cascade
            return None

        # Find the target cell
        target_cell = next((c for c in self.session.notebook_state.cells if c.id == cell_id), None)
        if not target_cell:
            return None

        # Check if any upstream inputs are stale
        upstream_cells = self.session.dag.cell_upstream.get(cell_id, [])
        if not upstream_cells:
            # No upstream — cell can run immediately
            return None

        # Check staleness of upstream cells
        has_stale_upstream = False
        for upstream_id in upstream_cells:
            upstream_cell = next(
                (c for c in self.session.notebook_state.cells if c.id == upstream_id),
                None,
            )
            if not upstream_cell:
                continue

            # Check if upstream is ready (artifact exists and provenance matches)
            if upstream_cell.status != CellStatus.READY:
                has_stale_upstream = True
                break

        if not has_stale_upstream:
            # All upstream cells are ready — can run immediately
            return None

        # Build the cascade plan: get all cells that need to run
        # in topological order
        plan = self._build_plan(cell_id)
        return plan

    def _build_plan(self, target_cell_id: str) -> CascadePlan | None:
        """Build a cascade plan for a target cell.

        Args:
            target_cell_id: The cell the user wants to run

        Returns:
            CascadePlan or None if no cascade needed
        """
        if not self.session.dag:
            return None

        # Shared BFS-upstream helper — same one ``get_cascade_plan`` uses,
        # so both call sites agree on what "reachable upstream" means.
        visited = reachable_via(self.session.dag.cell_upstream, target_cell_id)

        # Order by the DAG's topological sort if available; otherwise the
        # raw set order is fine (visited as a fallback).
        if self.session.dag.topological_order:
            step_cells = [cid for cid in self.session.dag.topological_order if cid in visited]
        else:
            step_cells = list(visited)

        # Build steps for each cell
        steps: list[CascadeStep] = []
        for step_cell_id in step_cells:
            cell = next(
                (c for c in self.session.notebook_state.cells if c.id == step_cell_id),
                None,
            )
            if not cell:
                continue

            # Determine reason
            if step_cell_id == target_cell_id:
                reason = CascadeReason.TARGET
            elif cell.status == CellStatus.STALE:
                reason = CascadeReason.STALE
            else:
                reason = CascadeReason.MISSING

            # Check if can skip (already cached/ready)
            skip = cell.status == CellStatus.READY

            step = CascadeStep(
                cell_id=step_cell_id,
                cell_name=cell.defines[0] if cell.defines else cell.id,
                reason=reason,
                skip=skip,
                estimated_ms=self.session.get_estimated_duration(step_cell_id),
            )
            steps.append(step)

        if not steps:
            return None

        plan = CascadePlan(
            plan_id="",  # Will be auto-generated
            target_cell_id=target_cell_id,
            steps=steps,
            estimated_duration_ms=sum(s.estimated_ms for s in steps if not s.skip),
        )
        return plan
