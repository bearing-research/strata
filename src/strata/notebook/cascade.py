"""Cascade planner: which upstream cells must run before a target cell."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

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
    """A single cell in a cascade plan; ``skip`` marks one that can be skipped."""

    cell_id: str
    cell_name: str
    reason: CascadeReason = CascadeReason.MISSING
    skip: bool = False
    estimated_ms: int = 0


@dataclass
class CascadePlan:
    """Plan for cascading execution; ``steps`` are in topological order."""

    plan_id: str
    target_cell_id: str
    steps: list[CascadeStep] = field(default_factory=list)
    estimated_duration_ms: int = 0

    def __post_init__(self):
        """Generate plan_id if not provided."""
        if not self.plan_id:
            self.plan_id = str(uuid.uuid4())[:8]


class CascadePlanner:
    """Plans the upstream cells to run before a cell whose inputs are not all ready."""

    def __init__(self, session: NotebookSession):
        """Initialize planner for a session."""
        self.session = session

    def plan(self, cell_id: str) -> CascadePlan | None:
        """Return the cascade plan for *cell_id*, or None if it can run immediately."""
        if not self.session.dag:
            return None

        target_cell = self.session.notebook_state.get_cell(cell_id)
        if not target_cell:
            return None

        upstream_cells = self.session.dag.cell_upstream.get(cell_id, [])
        if not upstream_cells:
            return None

        has_stale_upstream = False
        for upstream_id in upstream_cells:
            upstream_cell = self.session.notebook_state.get_cell(upstream_id)
            if not upstream_cell:
                continue

            if upstream_cell.status != CellStatus.READY:
                has_stale_upstream = True
                break

        if not has_stale_upstream:
            return None

        plan = self._build_plan(cell_id)
        return plan

    def _build_plan(self, target_cell_id: str) -> CascadePlan | None:
        """Build a cascade plan for *target_cell_id*, or None if none is needed."""
        if not self.session.dag:
            return None

        # Same helper as ``cascade_plan``, so both agree on "reachable upstream".
        visited = self.session.dag.upstream_reachable(target_cell_id)

        if self.session.dag.topological_order:
            step_cells = [cid for cid in self.session.dag.topological_order if cid in visited]
        else:
            step_cells = list(visited)

        steps: list[CascadeStep] = []
        for step_cell_id in step_cells:
            cell = self.session.notebook_state.get_cell(step_cell_id)
            if not cell:
                continue

            if step_cell_id == target_cell_id:
                reason = CascadeReason.TARGET
            elif cell.status == CellStatus.STALE:
                reason = CascadeReason.STALE
            else:
                reason = CascadeReason.MISSING

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
            plan_id="",  # auto-generated
            target_cell_id=target_cell_id,
            steps=steps,
            estimated_duration_ms=sum(s.estimated_ms for s in steps if not s.skip),
        )
        return plan
