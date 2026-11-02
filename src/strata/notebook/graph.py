"""The notebook, analyzed by cellgraph with Strata's vocabulary.

One ``cellgraph.analyze`` call gives each cell's annotations, names and diagnostics,
the DAG, and which cells can run. The session keeps the result
(``NotebookSession.analysis``).
"""

from __future__ import annotations

import dataclasses

import cellgraph as cg

from strata.notebook import vocabulary
from strata.notebook.models import CellLanguage, CellState, NotebookState


class _RunsWithSyntaxErrors:
    """cellgraph's Python analyzer, without its ``syntax_error``.

    Strata runs a cell that does not parse: the harness reports the error, with the
    traceback, where the cell's output goes.
    """

    def __init__(self, inner: cg.LanguageAnalyzer) -> None:
        self._inner = inner
        self.owns_key = inner.owns_key
        self.directives = getattr(inner, "directives", True)

    def analyze(self, cell: cg.Cell) -> cg.Analyzed:
        return dataclasses.replace(self._inner.analyze(cell), error=None)

    def normalize(self, body: str) -> str:
        return self._inner.normalize(body)


def _analyzers() -> dict[str, cg.LanguageAnalyzer]:
    table = {lang.value: cg.get_language_analyzer(lang.value) for lang in CellLanguage}
    table["python"] = _RunsWithSyntaxErrors(table["python"])
    return table


def analyze_notebook(state: NotebookState, cells: list[CellState] | None = None) -> cg.Analysis:
    """Analyze *state*'s notebook, or *cells* in its place."""
    cg_cells = [
        cg.Cell(
            id=cell.id,
            source=cell.source,
            language=cell.language.value,
            meta={
                "language": cell.language.value,
                "mounts": list(cell.mounts),
                "notebook_mounts": list(state.mounts),
                "connections": list(state.connections),
                "malformed_connections": list(state.malformed_connections),
            },
        )
        for cell in (state.cells if cells is None else cells)
    ]
    groups = cg.GroupConfig(
        modes=dict(state.variant_modes), active=dict(state.variant_active_selections)
    )
    return cg.analyze(
        cg_cells,
        vocabulary.strata_vocabulary(),
        groups,
        normalization=cg.NORMALIZE_LEGACY_1,
        digest=cg.DIGEST_LEGACY_1,
        analyzers=_analyzers(),
    )


def analysis_with(
    cell: CellState, state: NotebookState, analysis: cg.Analysis | None = None
) -> cg.Analysis:
    """*analysis* when it read *cell* as it is; otherwise the notebook analyzed again with
    *cell* in place of its namesake (or after the others, when it has none)."""
    current = state.get_cell(cell.id)
    if (
        analysis is not None
        and current is not None
        and current.source == cell.source
        and cell.id in analysis.cells
    ):
        return analysis
    cells = [cell if c.id == cell.id else c for c in state.cells]
    if current is None:
        cells.append(cell)
    return analyze_notebook(state, cells)
