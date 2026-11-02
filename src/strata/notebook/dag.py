"""The notebook DAG: cellgraph's, under Strata's names.

``cellgraph.analyze`` wires the graph (``strata.notebook.graph``); ``NotebookDag`` is its
``Dag`` with the two things Strata adds: the edge list the WebSocket and REST payloads
carry, and ``from_cells``, which wires names already analyzed (the Jupyter import and
the DAG tests have no notebook to analyze).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import cellgraph as cg
from cellgraph import DagEdge, SweepProducer, VariantGroupResolution

__all__ = [
    "CellAnalysisWithId",
    "DagEdge",
    "NotebookDag",
    "SweepProducer",
    "VariantGroupResolution",
    "VariantNameCollisionError",
    "producer_cell_label",
]


@dataclass
class CellAnalysisWithId:
    """A cell's names and graph annotations, for ``NotebookDag.from_cells``.

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
    # ``# @loop start_from=<cell>@iter=k``: ``(cell, carry)``. The seed is that cell's
    # iteration, so the loop depends on it whichever cell last defined the carry.
    seed_from: tuple[str, str] | None = None
    # Variables kept as if a downstream cell read them: ``# @write_table`` ones, which the
    # server writes, and ``# @output_dir`` folders, which are outputs even with no reader.
    kept: list[str] = field(default_factory=list)


class VariantNameCollisionError(ValueError):
    """Raised when two cells claim the same (variant_group, variant_name).

    There is no defensible default, so the user must rename one variant.
    """


@dataclass
class NotebookDag(cg.Dag):
    """cellgraph's ``Dag``: inactive variants appear only in ``variant_groups`` and
    ``inactive_cells``; ``variable_producer`` maps a variable to its last producer."""

    @classmethod
    def of(cls, dag: cg.Dag) -> NotebookDag:
        """*dag* as a ``NotebookDag``; its fields are shared, not copied."""
        return cls(**{f.name: getattr(dag, f.name) for f in dataclasses.fields(cg.Dag)})

    @classmethod
    def from_cells(
        cls,
        cells: list[CellAnalysisWithId],
        variant_active_selections: Mapping[str, str] | None = None,
        variant_modes: Mapping[str, str] | None = None,
    ) -> NotebookDag:
        """Wire cells given in source order, with their names already analyzed.

        Raises
        ------
        VariantNameCollisionError
            If two cells share the same ``(group, variant_name)``.
        ValueError
            If the DAG over active cells contains a cycle.
        """
        analysis = cg.analyze(
            [cg.Cell(c.id, "", _GIVEN, meta={"cell": c}) for c in cells],
            cg.Vocabulary([], cell_effects=_given_effects),
            cg.GroupConfig(
                modes=dict(variant_modes or {}), active=dict(variant_active_selections or {})
            ),
            analyzers={_GIVEN: _GivenNames()},
        )
        for code, error in (
            ("variant_collision", VariantNameCollisionError),
            ("cycle", ValueError),
        ):
            found = next((d for d in analysis.structural_errors if d.code == code), None)
            if found is not None:
                raise error(found.message)
        return cls.of(analysis.graph)

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


# A language for cells whose names ``from_cells`` was given.
_GIVEN = "strata-given-names"


class _GivenNames:
    """The names a ``CellAnalysisWithId`` carries, as a language analyzer."""

    owns_key = True
    directives = False

    def analyze(self, cell: cg.Cell) -> cg.Analyzed:
        given: CellAnalysisWithId = cell.meta["cell"]
        return cg.Analyzed(
            defines=list(given.defines),
            references=list(given.references),
            builtin_references=list(given.builtin_references),
        )

    def normalize(self, body: str) -> str:
        return body


def _given_effects(meta: Mapping[str, Any], parsed: Mapping[str, Any]) -> list[cg.Effect]:
    given: CellAnalysisWithId = meta["cell"]
    effects: list[cg.Effect] = [cg.After(given.after), cg.Keeps(given.kept)]
    if given.variant_group is not None and given.variant_name is not None:
        effects.append(cg.Member(given.variant_group, given.variant_name))
    if given.per_variant:
        effects.append(cg.FanOut(given.per_variant_group))
    if given.seed_from is not None:
        effects.append(cg.Seeds(*given.seed_from))
    return effects
