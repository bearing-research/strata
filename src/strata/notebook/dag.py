"""DAG construction and analysis for notebook cells."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass
class DagEdge:
    """An edge in the DAG: ``variable`` flows from ``from_cell_id`` to ``to_cell_id``."""

    from_cell_id: str
    to_cell_id: str
    variable: str


@dataclass
class VariantGroupResolution:
    """Resolved state for a single variant group.

    ``members`` is ``(cell_id, variant_name)`` pairs in source order, inactive ones
    included so the frontend can render tabs. Only ``active_cell_id`` takes part in
    the DAG (producer map, edges, consumed_variables).
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
    """Producer-map entry for a variable produced across a set of variants.

    - **Sweep group** (``fanout_cell is None``): ``variants`` maps
      ``variant_name -> member_cell_id`` and a downstream reference fans out to one
      edge per member.
    - **Fan-out cell** (``fanout_cell`` set): a ``# @per_variant`` cell runs once
      per variant of an upstream sweep, so every entry points at ``fanout_cell``;
      per-instance artifacts use an ``@variant=<name>`` subkey.

    ``variants`` is sorted so the value is stable and hashable.
    """

    group: str
    variants: tuple[tuple[str, str], ...]  # sorted ((variant_name, cell_id), ...)
    fanout_cell: str | None = None


@dataclass
class CellAnalysisWithId:
    """Cell analysis result paired with cell ID.

    ``after`` holds upstream cell IDs from ``# @after``; those edges are
    ordering-only and carry no variable.
    """

    id: str
    defines: list[str]
    references: list[str]
    # Free names shadowing a builtin: hidden from ``references`` but resolved
    # against the producer map the same way.
    builtin_references: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    variant_group: str | None = None
    variant_name: str | None = None
    # ``# @per_variant [group]``: outputs become a fan-out ``SweepProducer`` over
    # ``per_variant_group`` (or the single sweep group it reads when None).
    per_variant: bool = False
    per_variant_group: str | None = None


class VariantNameCollisionError(ValueError):
    """Raised when two cells claim the same (variant_group, variant_name).

    There is no defensible default, so the user must rename one variant.
    """


@dataclass
class NotebookDag:
    """The complete DAG for a notebook.

    ``variable_producer`` maps a variable to its producing cell (last in cell order
    wins) or a ``SweepProducer``. ``consumed_variables`` is, per cell, the variables
    downstream cells reference. Inactive variants appear only in ``variant_groups``
    and ``inactive_cells``.
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
        """Build the DAG from cell analyses given in source order.

        A group missing from ``variant_active_selections`` uses its first variant in
        source order. Inactive variants produce no edges and no producer-map entries.

        Raises
        ------
        VariantNameCollisionError
            If two cells share the same ``(group, variant_name)``.
        ValueError
            If the DAG over active cells contains a cycle.
        """
        dag = cls()
        cell_ids = [c.id for c in cells]

        # Groups come from source annotations; the active selection from notebook.toml.
        selections = dict(variant_active_selections or {})
        modes = dict(variant_modes or {})
        sweep_groups = {g for g, m in modes.items() if m == "sweep"}
        dag.variant_groups, dag.inactive_cells = _resolve_variant_groups(
            cells, selections, sweep_groups
        )
        inactive = dag.inactive_cells

        # One SweepProducer per produced variable, holding only members that define it:
        # define-sets can diverge, and fanning onto a member without the var would wire
        # a phantom consumed variable and fail _store_outputs.
        cell_by_id = {c.id: c for c in cells}
        sweep_producer_for_var: dict[str, SweepProducer] = {}
        sweep_member_group: dict[str, str] = {}
        for res in dag.variant_groups:
            if res.mode != "sweep":
                continue
            for cid, _ in res.members:
                sweep_member_group[cid] = res.group
            var_members: dict[str, list[tuple[str, str]]] = {}
            for cid, name in res.members:
                member = cell_by_id.get(cid)
                if member is None:
                    continue
                for var in member.defines:
                    var_members.setdefault(var, []).append((name, cid))
            for var, members in var_members.items():
                sweep_producer_for_var[var] = SweepProducer(
                    group=res.group, variants=tuple(sorted(members))
                )

        # ``var_to_sweep_group`` is seeded from sweep members and extended in the main
        # loop with fan-out outputs, so a chained ``@per_variant`` resolves. Group choice
        # mirrors annotation validation: named group, else the single sweep group read;
        # zero or ambiguous keeps ordinary producer semantics.
        var_to_sweep_group = {var: sp.group for var, sp in sweep_producer_for_var.items()}
        sweep_group_variant_names: dict[str, tuple[str, ...]] = {
            res.group: tuple(sorted(name for _, name in res.members))
            for res in dag.variant_groups
            if res.mode == "sweep"
        }

        # Every cell gets entries, even inactive ones, so the frontend can index freely.
        for cell_id in cell_ids:
            dag.cell_upstream[cell_id] = []
            dag.cell_downstream[cell_id] = []
            dag.consumed_variables[cell_id] = set()

        # Single pass: wire each reference to the producer *before* this cell, then
        # register this cell's defines. A mutating cell (``sales["col"] = ...``) thus
        # reads the prior ``sales`` and produces the new one without a self-cycle.
        # Inactive variants neither resolve nor update the map; the frontend still
        # sees them via ``variant_groups``.
        cell_id_set = set(cell_ids)
        for cell in cells:
            if cell.id in inactive:
                continue
            # Builtin-shadowing names ride along: ``input = load_data()`` upstream is a real
            # producer for a consumer's ``builtin_references``.
            for var in (*cell.references, *cell.builtin_references):
                producer = dag.variable_producer.get(var)
                if producer is None or producer == cell.id:
                    # External variable or no prior producer: no edge. A mutating cell with no
                    # upstream producer raises NameError at runtime, which is the right signal.
                    continue
                if isinstance(producer, SweepProducer):
                    member_ids = {cid for _, cid in producer.variants}
                    if cell.id in member_ids:
                        # A member refining its own group's var (``preds = f(preds)``) must
                        # not depend on its siblings.
                        continue
                    # One edge per member; each member's output is stored so the harness can
                    # assemble the {variant: value} dict.
                    for _name, member_id in producer.variants:
                        _wire_variable_edge(dag, member_id, cell.id, var)
                else:
                    _wire_variable_edge(dag, producer, cell.id, var)

            # ``# @after <cell-id>``: ordering-only edge (e.g. a setup cell seeding a SQLite
            # file). It affects wiring and topological order but not consumed_variables, so
            # per-variable provenance is unaffected.
            for upstream_id in cell.after:
                if upstream_id == cell.id or upstream_id not in cell_id_set:
                    # Self-references and dangling ids are dropped so a typo can't crash the build.
                    # No diagnostic reports them yet.
                    continue
                dag.edges.append(DagEdge(from_cell_id=upstream_id, to_cell_id=cell.id, variable=""))
                if upstream_id not in dag.cell_upstream[cell.id]:
                    dag.cell_upstream[cell.id].append(upstream_id)
                if cell.id not in dag.cell_downstream[upstream_id]:
                    dag.cell_downstream[upstream_id].append(cell.id)

            # A sweep member registers, for each var it defines, that var's SweepProducer;
            # every sibling sets the same value, so downstream sees the whole group.
            is_sweep_member = cell.id in sweep_member_group
            # Includes upstream fan-out outputs, so chained @per_variant works.
            fanout_group: str | None = None
            if cell.per_variant:
                read_groups = {
                    var_to_sweep_group[ref]
                    for ref in (*cell.references, *cell.builtin_references)
                    if ref in var_to_sweep_group
                }
                named = cell.per_variant_group
                if named is not None:
                    fanout_group = named if named in read_groups else None
                else:
                    fanout_group = next(iter(read_groups)) if len(read_groups) == 1 else None
            for var in cell.defines:
                if is_sweep_member and var in sweep_producer_for_var:
                    new_producer: str | SweepProducer = sweep_producer_for_var[var]
                elif fanout_group is not None:
                    # Register the output as sweep-sourced so a chained @per_variant resolves.
                    new_producer = SweepProducer(
                        group=fanout_group,
                        variants=tuple(
                            sorted(
                                (name, cell.id) for name in sweep_group_variant_names[fanout_group]
                            )
                        ),
                        fanout_cell=cell.id,
                    )
                    var_to_sweep_group[var] = fanout_group
                else:
                    new_producer = cell.id
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

        # Inactive variants are shadow cells; the frontend finds them via ``variant_groups``.
        active_cell_ids = [cid for cid in cell_ids if cid not in inactive]

        for cell_id in active_cell_ids:
            if not dag.cell_downstream[cell_id]:
                dag.leaves.add(cell_id)
            if not dag.cell_upstream[cell_id]:
                dag.roots.add(cell_id)

        dag.topological_order = dag.topological_sort(active_cell_ids)

        return dag

    def topological_sort(self, cell_ids: list[str]) -> list[str]:
        """Return cells in topological (execution) order; raise ``ValueError`` on a cycle."""
        in_degree = {cell_id: len(self.cell_upstream[cell_id]) for cell_id in cell_ids}

        # deque: this runs on every keystroke-driven DAG rebuild, and list.pop(0) is O(n).
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
        """Return every cycle among ``cell_ids``, each as the list of cell IDs along it."""
        # Colors: 0=white, 1=gray, 2=black
        color = {cell_id: 0 for cell_id in cell_ids}
        cycles: list[list[str]] = []

        def dfs(node: str, path: list[str]) -> None:
            color[node] = 1
            path.append(node)
            for downstream in self.cell_downstream[node]:
                if color[downstream] == 1:
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
        """Return the cells reachable upstream from ``start``, including ``start``."""
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
        """Return the cells, in ``cell_ids`` order, to run before executing the target.

        Includes the target itself only when it is a root.
        """
        visited = self.upstream_reachable(target_cell_id)
        # Target stays in the plan only if it's a root (no upstreams to run).
        if self.cell_upstream[target_cell_id]:
            visited.discard(target_cell_id)
        return [cid for cid in cell_ids if cid in visited]

    def serialize_edges(self) -> list[dict[str, str]]:
        """Serialize edges as ``from_cell_id``/``to_cell_id``/``variable`` dicts.

        Every broadcast site must use this so the field names the frontend's
        ``applyBackendDag`` reads cannot drift between paths.
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
        if producer.fanout_cell is not None:
            return f"fanout:{producer.group}"
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

    Cells without a variant group always count as active. A group missing from
    ``selections`` uses its first variant in source order. Returns the resolutions
    (source order of first member) and the set of inactive cell IDs.

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
        # Active member: the toml selection if it names a real variant, else the first
        # in source order. In sweep mode every member runs; the first is only the
        # default display cell.
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
