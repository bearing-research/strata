"""Run impact preview: what running a cell will cost and invalidate.

Lists the upstream cells that must run first (via ``CascadePlanner``), the
downstream cells that will go stale, and the estimated upstream run time.
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
    """A downstream cell that running the target will invalidate.

    ``new_status`` is always ``"stale:upstream"``.
    """

    cell_id: str
    cell_name: str
    current_status: str
    new_status: str = "stale:upstream"


@dataclass
class ImpactPreview:
    """Full impact preview for running a cell.

    ``estimated_ms`` sums the upstream steps that are not skipped.
    """

    target_cell_id: str
    upstream: list[CascadeStep] = field(default_factory=list)
    downstream: list[DownstreamImpact] = field(default_factory=list)
    estimated_ms: int = 0

    @property
    def has_impact(self) -> bool:
        """Whether there is any upstream or downstream impact; if not, the UI skips the preview."""
        upstream_non_target = [s for s in self.upstream if s.cell_id != self.target_cell_id]
        return len(upstream_non_target) > 0 or len(self.downstream) > 0


class ImpactAnalyzer:
    """Combines upstream cascade planning with downstream invalidation."""

    def __init__(self, session: NotebookSession):
        self.session = session

    def preview(self, cell_id: str) -> ImpactPreview:
        """Compute the impact of running a cell."""
        upstream_steps = self._compute_upstream(cell_id)

        downstream = self._compute_downstream(cell_id)

        estimated_ms = sum(s.estimated_ms for s in upstream_steps if not s.skip)

        return ImpactPreview(
            target_cell_id=cell_id,
            upstream=upstream_steps,
            downstream=downstream,
            estimated_ms=estimated_ms,
        )

    def _compute_upstream(self, cell_id: str) -> list[CascadeStep]:
        """Upstream cells that need to run, in topological order."""
        planner = CascadePlanner(self.session)
        plan = planner.plan(cell_id)
        if plan is None:
            return []
        return plan.steps

    def _compute_downstream(self, cell_id: str) -> list[DownstreamImpact]:
        """Cells that will become stale, found by a forward BFS from the target."""
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

            # Only ready cells can become stale.
            if cell.status == CellStatus.READY:
                cell_name = cell.defines[0] if cell.defines else cell.id
                impacts.append(
                    DownstreamImpact(
                        cell_id=current,
                        cell_name=cell_name,
                        current_status=cell.status,
                    )
                )

            for downstream_id in self.session.dag.cell_downstream.get(current, []):
                if downstream_id not in visited:
                    queue.append(downstream_id)

        return impacts
