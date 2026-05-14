"""DAG construction and analysis for notebook cells."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass
class DagEdge:
    """An edge in the DAG representing a variable dependency.

    Attributes:
        from_cell_id: Cell that defines the variable
        to_cell_id: Cell that references the variable
        variable: Variable name that flows along this edge
    """

    from_cell_id: str
    to_cell_id: str
    variable: str


@dataclass
class VariantGroupResolution:
    """Resolved state for a single variant group.

    ``members`` is in source order; ``active_cell_id`` is the cell whose
    defines flow into the producer map. Inactive members are tracked here
    so the frontend can render them as tabs but they are excluded from
    everything DAG-related (producer map, edges, consumed_variables).
    """

    group: str
    active_name: str
    active_cell_id: str
    members: list[tuple[str, str]] = field(default_factory=list)
    """List of (cell_id, variant_name) in source order."""


@dataclass
class NotebookDag:
    """The complete DAG for a notebook.

    Attributes:
        edges: All variable-level edges in the DAG
        cell_upstream: For each cell, list of upstream cell IDs it depends on
        cell_downstream: For each cell, list of downstream cell IDs that depend on it
        leaves: Set of cell IDs with no downstream consumers
        roots: Set of cell IDs with no upstream dependencies
        topological_order: Cells in valid execution order
        variable_producer: For each variable, which cell produces it (last in cell order wins)
        consumed_variables: For each cell, set of variable names consumed by downstream cells
        variant_groups: Resolved variant groups, in source-order of first member
        inactive_cells: Cell IDs that are inactive variants (excluded from edges/producer map)
    """

    edges: list[DagEdge] = field(default_factory=list)
    cell_upstream: dict[str, list[str]] = field(default_factory=dict)
    cell_downstream: dict[str, list[str]] = field(default_factory=dict)
    leaves: set[str] = field(default_factory=set)
    roots: set[str] = field(default_factory=set)
    topological_order: list[str] = field(default_factory=list)
    variable_producer: dict[str, str] = field(default_factory=dict)
    consumed_variables: dict[str, set[str]] = field(default_factory=dict)
    shadow_warnings: dict[str, list[str]] = field(default_factory=dict)
    """Maps cell_id -> list of warning messages about shadowed variables."""
    variant_groups: list[VariantGroupResolution] = field(default_factory=list)
    inactive_cells: set[str] = field(default_factory=set)


@dataclass
class CellAnalysisWithId:
    """Cell analysis result paired with cell ID.

    Attributes:
        id: Cell ID
        defines: Variables defined by this cell
        references: Variables referenced by this cell
        after: Explicit ordering dependencies (``# @after <cell-id>``).
            Each entry is an upstream cell ID; the DAG edge is
            ordering-only (no variable flows along it).
        variant_group: Variant group ID parsed from ``# @variant``, or None
        variant_name: Variant name within the group, or None
    """

    id: str
    defines: list[str]
    references: list[str]
    after: list[str] = field(default_factory=list)
    variant_group: str | None = None
    variant_name: str | None = None


class VariantNameCollisionError(ValueError):
    """Raised when two cells claim the same (variant_group, variant_name).

    Recovery requires the user to rename one of the variants — there is
    no defensible default the system can pick.
    """


def build_dag(
    cells: list[CellAnalysisWithId],
    variant_active_selections: Mapping[str, str] | None = None,
) -> NotebookDag:
    """Build the DAG from cell analyses.

    Args:
        cells: List of cells with their analysis results, in source order
        variant_active_selections: Per-group active variant name from
            notebook.toml. If a group is missing here, the first variant
            in source order is implicitly active.

    Returns:
        NotebookDag with edges, upstream/downstream relations, and metadata.
        Inactive variants are entirely shadowed: they're recorded in
        ``inactive_cells`` and ``variant_groups`` for the frontend, but
        they don't produce edges or appear in the producer map.

    Raises:
        VariantNameCollisionError: If two cells share (group, variant_name).
        ValueError: If the resulting DAG (over active cells) contains a cycle.
    """
    dag = NotebookDag()
    cell_ids = [c.id for c in cells]

    # Resolve variant groups and figure out which cells to skip in the
    # variable-producer pass. Groups are derived from source annotations;
    # the active selection comes from notebook.toml.
    selections = dict(variant_active_selections or {})
    dag.variant_groups, dag.inactive_cells = _resolve_variant_groups(cells, selections)
    inactive = dag.inactive_cells

    # Initialize structures (every cell gets entries — even inactive ones,
    # so the frontend can index into the maps without special-casing).
    for cell_id in cell_ids:
        dag.cell_upstream[cell_id] = []
        dag.cell_downstream[cell_id] = []
        dag.consumed_variables[cell_id] = set()

    # Single pass: walk cells in order, wiring each cell's references
    # to whoever produced that variable *before* this cell, then record
    # this cell's own defines as the new producer for cells that come
    # after. This lets a mutating cell (``sales["col"] = ...``) both
    # reference the prior ``sales`` and become the producer for
    # downstream cells, without a spurious self-cycle error.
    #
    # Inactive variants are skipped entirely: they don't resolve
    # references against the producer map and they don't update it.
    # This keeps them out of the executable graph while leaving them
    # visible to the frontend through ``variant_groups``.
    for cell in cells:
        if cell.id in inactive:
            continue
        # Resolve references against the producer map as it stands before
        # this cell's defines are applied.
        for var in cell.references:
            producer_id = dag.variable_producer.get(var)
            if not producer_id or producer_id == cell.id:
                # Either the variable is external (no prior producer) or
                # a producer for this cell hasn't been set yet — nothing
                # to wire up. A mutating cell whose reference has no
                # upstream producer simply has no edge; the runtime will
                # raise NameError which is the right signal.
                continue
            edge = DagEdge(
                from_cell_id=producer_id,
                to_cell_id=cell.id,
                variable=var,
            )
            dag.edges.append(edge)
            if producer_id not in dag.cell_upstream[cell.id]:
                dag.cell_upstream[cell.id].append(producer_id)
            if cell.id not in dag.cell_downstream[producer_id]:
                dag.cell_downstream[producer_id].append(cell.id)
            dag.consumed_variables[producer_id].add(var)

        # ``# @after <cell-id>`` adds an ordering-only edge (no
        # variable flows along it). Used by SQL cells whose dependency
        # is on an upstream side-effect, like a setup cell that seeds
        # a SQLite file the connection points at. The edge participates
        # in upstream/downstream wiring and topological order, but does
        # not appear in consumed_variables (there's no variable to
        # consume), so per-variable provenance is unaffected.
        cell_id_set = set(cell_ids)
        for upstream_id in cell.after:
            if upstream_id == cell.id or upstream_id not in cell_id_set:
                # Self-references and dangling IDs are silently dropped
                # here; annotation_validation surfaces them as
                # diagnostics so the user sees the issue without the
                # DAG build crashing.
                continue
            edge = DagEdge(
                from_cell_id=upstream_id,
                to_cell_id=cell.id,
                variable="",
            )
            dag.edges.append(edge)
            if upstream_id not in dag.cell_upstream[cell.id]:
                dag.cell_upstream[cell.id].append(upstream_id)
            if cell.id not in dag.cell_downstream[upstream_id]:
                dag.cell_downstream[upstream_id].append(cell.id)

        # Now apply this cell's defines so later cells see it as the
        # producer. Shadow warnings still fire when a later cell
        # overwrites a prior producer.
        for var in cell.defines:
            previous_producer = dag.variable_producer.get(var)
            if previous_producer is not None and previous_producer != cell.id:
                warning = f"Variable '{var}' shadows definition from cell {previous_producer[:8]}"
                dag.shadow_warnings.setdefault(cell.id, []).append(warning)
            dag.variable_producer[var] = cell.id

    # Inactive variants are excluded from leaves / roots / topological
    # order — they're shadow cells, not real graph members. Frontend
    # discovers them via ``variant_groups`` instead.
    active_cell_ids = [cid for cid in cell_ids if cid not in inactive]

    # Identify leaves (cells with no downstream consumers)
    for cell_id in active_cell_ids:
        if not dag.cell_downstream[cell_id]:
            dag.leaves.add(cell_id)

    # Identify roots (cells with no upstream dependencies)
    for cell_id in active_cell_ids:
        if not dag.cell_upstream[cell_id]:
            dag.roots.add(cell_id)

    # Topological sort over the active subgraph
    dag.topological_order = topological_sort(dag, active_cell_ids)

    return dag


def _resolve_variant_groups(
    cells: list[CellAnalysisWithId],
    selections: Mapping[str, str],
) -> tuple[list[VariantGroupResolution], set[str]]:
    """Group cells by ``variant_group``, resolve active per group.

    Returns the resolved groups (in source order of first member) and the
    set of cell IDs that are inactive variants. Cells without a variant
    group always count as active.

    Raises:
        VariantNameCollisionError: If two cells share (group, variant_name).
    """
    # Walk cells in source order, collecting groups.
    grouped: dict[str, list[tuple[str, str]]] = {}
    group_order: list[str] = []
    for cell in cells:
        if cell.variant_group is None or cell.variant_name is None:
            continue
        members = grouped.setdefault(cell.variant_group, [])
        for existing_id, existing_name in members:
            if existing_name == cell.variant_name:
                raise VariantNameCollisionError(
                    f"Variant name '{cell.variant_name}' is used by both "
                    f"cell {existing_id[:8]} and cell {cell.id[:8]} in "
                    f"group '{cell.variant_group}'"
                )
        if not members:
            group_order.append(cell.variant_group)
        members.append((cell.id, cell.variant_name))

    resolutions: list[VariantGroupResolution] = []
    inactive: set[str] = set()
    for group_id in group_order:
        members = grouped[group_id]
        wanted_name = selections.get(group_id)
        # Pick the active member: toml selection if it points at a real
        # variant, otherwise the first variant in source order. The
        # ``variant_active_unknown`` diagnostic surfaces toml drift.
        active_cell_id, active_name = members[0]
        if wanted_name is not None:
            for cid, name in members:
                if name == wanted_name:
                    active_cell_id, active_name = cid, name
                    break

        resolutions.append(
            VariantGroupResolution(
                group=group_id,
                active_name=active_name,
                active_cell_id=active_cell_id,
                members=list(members),
            )
        )
        for cid, _ in members:
            if cid != active_cell_id:
                inactive.add(cid)

    return resolutions, inactive


def topological_sort(dag: NotebookDag, cell_ids: list[str]) -> list[str]:
    """Return cells in topological (execution) order.

    Uses Kahn's algorithm with cycle detection.

    Args:
        dag: The DAG
        cell_ids: List of all cell IDs in original order

    Returns:
        Cells in topological order

    Raises:
        ValueError: If a cycle is detected
    """
    # Build in-degree map
    in_degree = {cell_id: len(dag.cell_upstream[cell_id]) for cell_id in cell_ids}

    # Find all nodes with in-degree 0. deque (not list) so popleft is O(1) —
    # this path runs on every keystroke-driven DAG rebuild, and list.pop(0)
    # is O(n) per call, giving O(n²) on the hot interactive path.
    queue: deque[str] = deque(cell_id for cell_id in cell_ids if in_degree[cell_id] == 0)
    result = []

    while queue:
        # Process a node with in-degree 0
        current = queue.popleft()
        result.append(current)

        # For each downstream cell, reduce in-degree
        for downstream_id in dag.cell_downstream[current]:
            in_degree[downstream_id] -= 1
            if in_degree[downstream_id] == 0:
                queue.append(downstream_id)

    # Check for cycles
    if len(result) != len(cell_ids):
        cycles = detect_cycles(dag, cell_ids)
        cycle_str = " → ".join(cycles[0]) if cycles else "unknown"
        raise ValueError(f"Cycle detected in DAG: {cycle_str}")

    return result


def detect_cycles(dag: NotebookDag, cell_ids: list[str]) -> list[list[str]]:
    """Find all cycles in the DAG using DFS.

    Args:
        dag: The DAG
        cell_ids: List of all cell IDs

    Returns:
        List of cycles (each cycle is a list of cell IDs)
    """
    # Colors: 0=white, 1=gray, 2=black
    color = {cell_id: 0 for cell_id in cell_ids}
    cycles: list[list[str]] = []

    def dfs(node: str, path: list[str]) -> None:
        """DFS to find cycles."""
        color[node] = 1  # Gray
        path.append(node)

        for downstream in dag.cell_downstream[node]:
            if color[downstream] == 1:
                # Back edge — found a cycle
                cycle_start = path.index(downstream)
                cycle = path[cycle_start:] + [downstream]
                cycles.append(cycle)
            elif color[downstream] == 0:
                # White — recurse
                dfs(downstream, path)

        path.pop()
        color[node] = 2  # Black

    for cell_id in cell_ids:
        if color[cell_id] == 0:
            dfs(cell_id, [])

    return cycles


def reachable_via(
    adjacency: Mapping[str, list[str]],
    start: str,
) -> set[str]:
    """BFS from ``start`` over ``adjacency``; return reachable set (incl. start).

    Single shared helper for both upstream and downstream walks across the
    DAG, cascade planner, and impact previewer — pass ``dag.cell_upstream``
    for an upstream walk, ``dag.cell_downstream`` for a downstream one.
    Uses ``deque.popleft`` (O(1)) rather than ``list.pop(0)`` (O(n)) since
    these walks run on every keystroke-driven DAG rebuild and once per
    impact-preview request.
    """
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return visited


def get_cascade_plan(dag: NotebookDag, target_cell_id: str, cell_ids: list[str]) -> list[str]:
    """Get all upstream cells needed before executing a target cell.

    Args:
        dag: The DAG
        target_cell_id: The cell to execute
        cell_ids: List of all cell IDs in execution order

    Returns:
        List of cell IDs in execution order that need to run before the target.
        If the target is a root cell, includes the target cell itself.
        Otherwise, includes only upstream cells.
    """
    visited = reachable_via(dag.cell_upstream, target_cell_id)
    # Target stays in the plan only if it's a root (no upstreams to run).
    if dag.cell_upstream[target_cell_id]:
        visited.discard(target_cell_id)
    return [cid for cid in cell_ids if cid in visited]
