"""DAG construction and analysis for notebook cells."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass
class DagEdge:
    """An edge in the DAG representing a variable dependency.

    Attributes
    ----------
    from_cell_id : str
        Cell that defines the variable.
    to_cell_id : str
        Cell that references the variable.
    variable : str
        Variable name that flows along this edge.
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

    Attributes
    ----------
    group : str
        Variant group ID parsed from ``# @variant`` annotations.
    active_name : str
        Name of the variant currently selected as active.
    active_cell_id : str
        Cell ID of the active variant — its defines flow into the producer map.
    members : list of tuple of (str, str)
        ``(cell_id, variant_name)`` pairs in source order, including inactive ones.
    """

    group: str
    active_name: str
    active_cell_id: str
    members: list[tuple[str, str]] = field(default_factory=list)
    # "switch" (one active member) or "sweep" (all members run; downstream
    # consumes a {variant: value} dict). Sweep groups have no inactive members.
    mode: str = "switch"


@dataclass(frozen=True)
class SweepProducer:
    """Producer-map entry for a variable produced by a *sweep* variant group.

    In switch mode a variable maps to a single producing cell id. In sweep mode
    every variant produces the same name, so the variable maps to this — the
    group plus all ``(variant_name, cell_id)`` members (sorted, so it's a stable
    value) — and a downstream reference fans out to one edge per member.
    """

    group: str
    variants: tuple[tuple[str, str], ...]  # sorted ((variant_name, cell_id), ...)


@dataclass
class CellAnalysisWithId:
    """Cell analysis result paired with cell ID.

    Attributes
    ----------
    id : str
        Cell ID.
    defines : list of str
        Variables defined by this cell.
    references : list of str
        Variables referenced by this cell.
    after : list of str
        Explicit ordering dependencies (``# @after <cell-id>``). Each entry
        is an upstream cell ID; the DAG edge is ordering-only (no variable
        flows along it).
    variant_group : str or None
        Variant group ID parsed from ``# @variant``, or ``None``.
    variant_name : str or None
        Variant name within the group, or ``None``.
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


@dataclass
class NotebookDag:
    """The complete DAG for a notebook.

    Attributes
    ----------
    edges : list of DagEdge
        All variable-level edges in the DAG.
    cell_upstream : dict of {str : list of str}
        For each cell, the list of upstream cell IDs it depends on.
    cell_downstream : dict of {str : list of str}
        For each cell, the list of downstream cell IDs that depend on it.
    leaves : set of str
        Cell IDs with no downstream consumers.
    roots : set of str
        Cell IDs with no upstream dependencies.
    topological_order : list of str
        Cells in valid execution order.
    variable_producer : dict of {str : str}
        For each variable, which cell produces it (last in cell order wins).
    consumed_variables : dict of {str : set of str}
        For each cell, the set of variable names consumed by downstream cells.
    shadow_warnings : dict of {str : list of str}
        Maps ``cell_id`` to a list of warning messages about shadowed variables.
    variant_groups : list of VariantGroupResolution
        Resolved variant groups, in source-order of first member.
    inactive_cells : set of str
        Cell IDs that are inactive variants (excluded from edges/producer map).
    """

    edges: list[DagEdge] = field(default_factory=list)
    cell_upstream: dict[str, list[str]] = field(default_factory=dict)
    cell_downstream: dict[str, list[str]] = field(default_factory=dict)
    leaves: set[str] = field(default_factory=set)
    roots: set[str] = field(default_factory=set)
    topological_order: list[str] = field(default_factory=list)
    variable_producer: dict[str, str | SweepProducer] = field(default_factory=dict)
    consumed_variables: dict[str, set[str]] = field(default_factory=dict)
    shadow_warnings: dict[str, list[str]] = field(default_factory=dict)
    variant_groups: list[VariantGroupResolution] = field(default_factory=list)
    inactive_cells: set[str] = field(default_factory=set)

    @classmethod
    def from_cells(
        cls,
        cells: list[CellAnalysisWithId],
        variant_active_selections: Mapping[str, str] | None = None,
        variant_modes: Mapping[str, str] | None = None,
    ) -> NotebookDag:
        """Build the DAG from cell analyses.

        Parameters
        ----------
        cells : list of CellAnalysisWithId
            Cells with their analysis results, in source order.
        variant_active_selections : Mapping of {str : str}, optional
            Per-group active variant name from ``notebook.toml``. If a group
            is missing here, the first variant in source order is implicitly
            active.

        Returns
        -------
        NotebookDag
            DAG with edges, upstream/downstream relations, and metadata.
            Inactive variants are entirely shadowed: they are recorded in
            ``inactive_cells`` and ``variant_groups`` for the frontend, but
            they do not produce edges or appear in the producer map.

        Raises
        ------
        VariantNameCollisionError
            If two cells share the same ``(group, variant_name)``.
        ValueError
            If the resulting DAG (over active cells) contains a cycle.
        """
        dag = cls()
        cell_ids = [c.id for c in cells]

        # Resolve variant groups and figure out which cells to skip in the
        # variable-producer pass. Groups are derived from source annotations;
        # the active selection comes from notebook.toml.
        selections = dict(variant_active_selections or {})
        modes = dict(variant_modes or {})
        sweep_groups = {g for g, m in modes.items() if m == "sweep"}
        dag.variant_groups, dag.inactive_cells = _resolve_variant_groups(
            cells, selections, sweep_groups
        )
        inactive = dag.inactive_cells

        # Sweep groups: a SweepProducer (all members) per group, keyed for the
        # defines pass and the reference fan-out below.
        sweep_producers: dict[str, SweepProducer] = {}
        sweep_member_group: dict[str, str] = {}
        for res in dag.variant_groups:
            if res.mode != "sweep":
                continue
            sweep_producers[res.group] = SweepProducer(
                group=res.group,
                variants=tuple(sorted((name, cid) for cid, name in res.members)),
            )
            for cid, _ in res.members:
                sweep_member_group[cid] = res.group

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
        cell_id_set = set(cell_ids)
        for cell in cells:
            if cell.id in inactive:
                continue
            # Resolve references against the producer map as it stands before
            # this cell's defines are applied.
            for var in cell.references:
                producer = dag.variable_producer.get(var)
                if producer is None or producer == cell.id:
                    # Either the variable is external (no prior producer) or
                    # a producer for this cell hasn't been set yet — nothing
                    # to wire up. A mutating cell whose reference has no
                    # upstream producer simply has no edge; the runtime will
                    # raise NameError which is the right signal.
                    continue
                if isinstance(producer, SweepProducer):
                    # Sweep group: fan out to one edge per variant member, all
                    # flowing into this cell. Each member's output is consumed
                    # (stored) so the harness can assemble the {variant: value}
                    # dict; topological order then requires all members first.
                    for _name, member_id in producer.variants:
                        if member_id != cell.id:
                            _wire_variable_edge(dag, member_id, cell.id, var)
                else:
                    _wire_variable_edge(dag, producer, cell.id, var)

            # ``# @after <cell-id>`` adds an ordering-only edge (no
            # variable flows along it). Used by SQL cells whose dependency
            # is on an upstream side-effect, like a setup cell that seeds
            # a SQLite file the connection points at. The edge participates
            # in upstream/downstream wiring and topological order, but does
            # not appear in consumed_variables (there's no variable to
            # consume), so per-variable provenance is unaffected.
            for upstream_id in cell.after:
                if upstream_id == cell.id or upstream_id not in cell_id_set:
                    # Self-references and dangling IDs are silently dropped
                    # here; annotation_validation surfaces them as
                    # diagnostics so the user sees the issue without the
                    # DAG build crashing.
                    continue
                dag.edges.append(DagEdge(from_cell_id=upstream_id, to_cell_id=cell.id, variable=""))
                if upstream_id not in dag.cell_upstream[cell.id]:
                    dag.cell_upstream[cell.id].append(upstream_id)
                if cell.id not in dag.cell_downstream[upstream_id]:
                    dag.cell_downstream[upstream_id].append(cell.id)

            # Now apply this cell's defines so later cells see it as the
            # producer. Shadow warnings still fire when a later cell
            # overwrites a prior producer.
            # A sweep variant member produces the group's SweepProducer (all
            # members), not itself — so every sibling sets the same value and
            # downstream sees the whole group. Setting the same value for each
            # sibling is idempotent and doesn't trip the shadow check.
            group_for_cell = sweep_member_group.get(cell.id)
            new_producer: str | SweepProducer = (
                sweep_producers[group_for_cell] if group_for_cell is not None else cell.id
            )
            for var in cell.defines:
                previous_producer = dag.variable_producer.get(var)
                if previous_producer is not None and previous_producer != new_producer:
                    short_id = (
                        previous_producer.group
                        if isinstance(previous_producer, SweepProducer)
                        else previous_producer
                    )[:8]
                    warning = f"Variable '{var}' shadows definition from cell {short_id}"
                    dag.shadow_warnings.setdefault(cell.id, []).append(warning)
                dag.variable_producer[var] = new_producer

        # Inactive variants are excluded from leaves / roots / topological
        # order — they're shadow cells, not real graph members. Frontend
        # discovers them via ``variant_groups`` instead.
        active_cell_ids = [cid for cid in cell_ids if cid not in inactive]

        for cell_id in active_cell_ids:
            if not dag.cell_downstream[cell_id]:
                dag.leaves.add(cell_id)
            if not dag.cell_upstream[cell_id]:
                dag.roots.add(cell_id)

        dag.topological_order = dag.topological_sort(active_cell_ids)

        return dag

    def topological_sort(self, cell_ids: list[str]) -> list[str]:
        """Return cells in topological (execution) order.

        Uses Kahn's algorithm with cycle detection.

        Parameters
        ----------
        cell_ids : list of str
            Cell IDs to sort (typically active cells only).

        Returns
        -------
        list of str
            Cells in topological order.

        Raises
        ------
        ValueError
            If a cycle is detected.
        """
        in_degree = {cell_id: len(self.cell_upstream[cell_id]) for cell_id in cell_ids}

        # deque (not list) so popleft is O(1) — this path runs on every
        # keystroke-driven DAG rebuild, and list.pop(0) is O(n) per call,
        # giving O(n²) on the hot interactive path.
        queue: deque[str] = deque(cell_id for cell_id in cell_ids if in_degree[cell_id] == 0)
        result: list[str] = []

        while queue:
            current = queue.popleft()
            result.append(current)
            for downstream_id in self.cell_downstream[current]:
                in_degree[downstream_id] -= 1
                if in_degree[downstream_id] == 0:
                    queue.append(downstream_id)

        if len(result) != len(cell_ids):
            cycles = self.detect_cycles(cell_ids)
            cycle_str = " → ".join(cycles[0]) if cycles else "unknown"
            raise ValueError(f"Cycle detected in DAG: {cycle_str}")

        return result

    def detect_cycles(self, cell_ids: list[str]) -> list[list[str]]:
        """Find all cycles in the DAG using DFS.

        Parameters
        ----------
        cell_ids : list of str
            Cell IDs to scan.

        Returns
        -------
        list of list of str
            One inner list per cycle, each containing the cell IDs along that cycle.
        """
        # Colors: 0=white, 1=gray, 2=black
        color = {cell_id: 0 for cell_id in cell_ids}
        cycles: list[list[str]] = []

        def dfs(node: str, path: list[str]) -> None:
            color[node] = 1
            path.append(node)
            for downstream in self.cell_downstream[node]:
                if color[downstream] == 1:
                    # Back edge — found a cycle
                    cycle_start = path.index(downstream)
                    cycles.append(path[cycle_start:] + [downstream])
                elif color[downstream] == 0:
                    dfs(downstream, path)
            path.pop()
            color[node] = 2

        for cell_id in cell_ids:
            if color[cell_id] == 0:
                dfs(cell_id, [])

        return cycles

    def upstream_reachable(self, start: str) -> set[str]:
        """Return the set of cells reachable upstream from ``start`` (inclusive).

        Uses ``deque.popleft`` (O(1)) rather than ``list.pop(0)`` (O(n)) since
        this walk runs on every keystroke-driven DAG rebuild and once per
        impact-preview / cascade-plan request.

        Parameters
        ----------
        start : str
            Cell ID to BFS from over ``cell_upstream``.

        Returns
        -------
        set of str
            All cell IDs reachable upstream from ``start``, including ``start`` itself.
        """
        visited: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.cell_upstream.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        return visited

    def cascade_plan(self, target_cell_id: str, cell_ids: list[str]) -> list[str]:
        """Get all upstream cells needed before executing a target cell.

        Parameters
        ----------
        target_cell_id : str
            The cell to execute.
        cell_ids : list of str
            All cell IDs in execution order.

        Returns
        -------
        list of str
            Cell IDs in execution order that need to run before the target.
            If the target is a root cell, includes the target cell itself;
            otherwise, includes only upstream cells.
        """
        visited = self.upstream_reachable(target_cell_id)
        # Target stays in the plan only if it's a root (no upstreams to run).
        if self.cell_upstream[target_cell_id]:
            visited.discard(target_cell_id)
        return [cid for cid in cell_ids if cid in visited]

    def serialize_edges(self) -> list[dict[str, str]]:
        """Serialize edges in the wire format the frontend expects.

        The frontend's ``applyBackendDag`` keys off ``from_cell_id`` /
        ``to_cell_id``; every broadcast site must use this so the field
        names can't drift apart. (Pre-fix, three near-identical loops
        diverged and the agent-edit path silently emitted ``from`` /
        ``to``, stranding every edge until the user hard-refreshed.)

        Returns
        -------
        list of dict of {str : str}
            One entry per edge with ``from_cell_id``, ``to_cell_id``,
            and ``variable`` keys.
        """
        return [
            {
                "from_cell_id": edge.from_cell_id,
                "to_cell_id": edge.to_cell_id,
                "variable": edge.variable,
            }
            for edge in self.edges
        ]


def producer_cell_label(producer: str | SweepProducer) -> str:
    """Flatten a producer-map value to one label for display / JSON surfaces.

    Switch producers are a cell id; a sweep group has no single producer, so it
    renders as ``"sweep:<group>"``.
    """
    if isinstance(producer, SweepProducer):
        return f"sweep:{producer.group}"
    return producer


def _wire_variable_edge(dag: NotebookDag, from_id: str, to_id: str, var: str) -> None:
    """Add a variable edge ``from_id -> to_id`` and update derived structures."""
    dag.edges.append(DagEdge(from_cell_id=from_id, to_cell_id=to_id, variable=var))
    if from_id not in dag.cell_upstream[to_id]:
        dag.cell_upstream[to_id].append(from_id)
    if to_id not in dag.cell_downstream[from_id]:
        dag.cell_downstream[from_id].append(to_id)
    dag.consumed_variables[from_id].add(var)


def _resolve_variant_groups(
    cells: list[CellAnalysisWithId],
    selections: Mapping[str, str],
    sweep_groups: set[str] | None = None,
) -> tuple[list[VariantGroupResolution], set[str]]:
    """Group cells by ``variant_group`` and resolve the active member per group.

    Cells without a variant group always count as active.

    Parameters
    ----------
    cells : list of CellAnalysisWithId
        Cells in source order; only those with both ``variant_group`` and
        ``variant_name`` set participate in grouping.
    selections : Mapping of {str : str}
        Per-group active variant name (e.g. from ``notebook.toml``).
        Missing groups fall back to the first variant in source order.

    Returns
    -------
    resolutions : list of VariantGroupResolution
        Resolved groups, in source order of their first member.
    inactive : set of str
        Cell IDs that are inactive variants.

    Raises
    ------
    VariantNameCollisionError
        If two cells share the same ``(group, variant_name)``.
    """
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

    sweep = sweep_groups or set()
    resolutions: list[VariantGroupResolution] = []
    inactive: set[str] = set()
    for group_id in group_order:
        members = grouped[group_id]
        is_sweep = group_id in sweep
        wanted_name = selections.get(group_id)
        # Pick the active member: toml selection if it points at a real
        # variant, otherwise the first variant in source order. The
        # ``variant_active_unknown`` diagnostic surfaces toml drift. In sweep
        # mode the active pointer is ignored (every member runs); the first
        # member is kept only as a default display cell for the frontend.
        active_cell_id, active_name = members[0]
        if wanted_name is not None and not is_sweep:
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
                mode="sweep" if is_sweep else "switch",
            )
        )
        # Switch mode shadows the non-active members; sweep mode runs them all.
        if not is_sweep:
            for cid, _ in members:
                if cid != active_cell_id:
                    inactive.add(cid)

    return resolutions, inactive
