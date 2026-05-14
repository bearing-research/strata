"""Run Impact Preview — shows upstream + downstream consequences before execution.

When a user is about to run a cell, the impact preview tells them:
1. Which upstream cells need to run first (reuses CascadePlanner)
2. Which downstream cells will become stale (forward walk from target)
3. Estimated total execution time

This extends the existing cascade prompt with downstream analysis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from strata.notebook.cascade import CascadePlanner, CascadeStep
from strata.notebook.models import CellStatus

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession


@dataclass
class DownstreamImpact:
    """A downstream cell that will be invalidated.

    Attributes:
        cell_id: ID of the affected cell
        cell_name: Display name of the affected cell
        current_status: Cell's current status
        new_status: Status after target cell runs (always 'stale:upstream')
    """

    cell_id: str
    cell_name: str
    current_status: str
    new_status: str = "stale:upstream"

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "cell_id": self.cell_id,
            "cell_name": self.cell_name,
            "current_status": self.current_status,
            "new_status": self.new_status,
        }


@dataclass
class ImpactPreview:
    """Full impact preview for running a cell.

    Attributes:
        target_cell_id: The cell the user wants to run
        upstream: Cells that need to run first (from cascade planner)
        downstream: Cells that will become stale
        estimated_ms: Total estimated execution time
    """

    target_cell_id: str
    upstream: list[CascadeStep] = field(default_factory=list)
    downstream: list[DownstreamImpact] = field(default_factory=list)
    estimated_ms: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "target_cell_id": self.target_cell_id,
            "upstream": [
                {
                    "cell_id": s.cell_id,
                    "cell_name": s.cell_name,
                    "reason": s.reason,
                    "skip": s.skip,
                    "estimated_ms": s.estimated_ms,
                }
                for s in self.upstream
            ],
            "downstream": [d.to_dict() for d in self.downstream],
            "estimated_ms": self.estimated_ms,
        }

    @property
    def has_impact(self) -> bool:
        """Whether there is any upstream or downstream impact.

        If False, the UI should skip the preview and just run the cell.
        """
        # Filter out the target cell itself from upstream
        upstream_non_target = [s for s in self.upstream if s.cell_id != self.target_cell_id]
        return len(upstream_non_target) > 0 or len(self.downstream) > 0


class ImpactAnalyzer:
    """Analyzes the impact of running a cell.

    Combines upstream cascade analysis with downstream invalidation
    analysis to produce a complete picture of what will happen.
    """

    def __init__(self, session: NotebookSession):
        """Initialize analyzer for a session.

        Args:
            session: NotebookSession instance
        """
        self.session = session

    def preview(self, cell_id: str) -> ImpactPreview:
        """Compute the impact of running a cell.

        Args:
            cell_id: ID of the cell to run

        Returns:
            ImpactPreview with upstream and downstream effects
        """
        # Upstream: reuse cascade planner
        upstream_steps = self._compute_upstream(cell_id)

        # Downstream: forward walk from target cell
        downstream = self._compute_downstream(cell_id)

        # Estimate total time
        estimated_ms = sum(s.estimated_ms for s in upstream_steps if not s.skip)

        return ImpactPreview(
            target_cell_id=cell_id,
            upstream=upstream_steps,
            downstream=downstream,
            estimated_ms=estimated_ms,
        )

    def _compute_upstream(self, cell_id: str) -> list[CascadeStep]:
        """Compute upstream cells that need to run.

        Args:
            cell_id: Target cell ID

        Returns:
            List of CascadeStep in topological order
        """
        planner = CascadePlanner(self.session)
        plan = planner.plan(cell_id)
        if plan is None:
            return []
        return plan.steps

    def _compute_downstream(self, cell_id: str) -> list[DownstreamImpact]:
        """Compute downstream cells that will become stale.

        Performs a forward BFS from the target cell through the DAG,
        collecting all cells that will be invalidated.

        Args:
            cell_id: Target cell ID

        Returns:
            List of DownstreamImpact for affected cells
        """
        if not self.session.dag:
            return []

        impacts: list[DownstreamImpact] = []
        visited: set[str] = set()
        queue: deque[str] = deque(self.session.dag.cell_downstream.get(cell_id, []))

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            cell = self.session.notebook_state.get_cell(current)
            if cell is None:
                continue

            # Only report cells that are currently ready — they'll become stale
            if cell.status == CellStatus.READY:
                cell_name = cell.defines[0] if cell.defines else cell.id
                impacts.append(
                    DownstreamImpact(
                        cell_id=current,
                        cell_name=cell_name,
                        current_status=cell.status,
                    )
                )

            # Continue walking downstream
            for downstream_id in self.session.dag.cell_downstream.get(current, []):
                if downstream_id not in visited:
                    queue.append(downstream_id)

        return impacts
