"""The notebook, analyzed by cellgraph with Strata's vocabulary.

One ``cellgraph.analyze`` call gives each cell's annotations, names and diagnostics,
the DAG, and which cells can run. The session keeps the result
(``NotebookSession.analysis``).
"""

from __future__ import annotations

import dataclasses
import functools
import re

import cellgraph as cg

from strata.notebook import vocabulary
from strata.notebook.annotations import parse_annotations
from strata.notebook.models import CellLanguage, CellState, MountSpec, NotebookState


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


# Read a cell's namespace without naming what they read.
_DYNAMIC_LOOKUPS = frozenset({"dir", "eval", "exec", "globals", "locals", "vars"})
_R_DYNAMIC_LOOKUPS = frozenset(
    {
        ".GlobalEnv",
        "as.environment",
        "caller_env",
        "current_env",
        "dynGet",
        "environment",
        "eval",
        "evalq",
        "exists",
        "get",
        "get0",
        "global_env",
        "globalenv",
        "ls",
        "mget",
        "objects",
        "parent.env",
        "parent.frame",
        "parse",
        "source",
        "str2expression",
        "str2lang",
        "sys.frame",
        "sys.frames",
        "sys.source",
        "topenv",
    }
)
# Over comments and strings too: glue("{data}") reads by name, and @each is a comment.
_R_WORD = re.compile(r"[A-Za-z.][\w.]*")


def mounts_of(state: NotebookState, cell: CellState, source: str) -> list[MountSpec]:
    """The mounts *cell* gets: its own, and the notebook's that *source* reads.

    A notebook-level mount costs every cell a listing, a download and a place in its key.
    A cell can only read one by name: code from an upstream cell is refused if it reads a
    global, and a function nested in a value holds the upstream cell's path.
    """
    names = {mount.name for mount in cell.mount_overrides}
    if cell.language == CellLanguage.R:
        words = set(_R_WORD.findall(source))
        if words & _R_DYNAMIC_LOOKUPS:
            return list(cell.mounts)
        names |= words
    elif cell.language == CellLanguage.PYTHON:
        analyzed = cg.get_language_analyzer("python").analyze(
            cg.Cell(id=cell.id, source=source, language="python", meta={})
        )
        names |= {*analyzed.references, *analyzed.builtin_references}
        if names & _DYNAMIC_LOOKUPS:
            return list(cell.mounts)
        each = parse_annotations(source, cell.language).each
        if each is not None:
            names.add(each["source"])
    elif cell.language == CellLanguage.SQL:
        sql = parse_annotations(source, cell.language).sql
        for connection in state.connections:
            if sql is not None and connection.name == sql.connection:
                names.update(getattr(connection, "mounts", None) or [])
    return [mount for mount in cell.mounts if mount.name in names]


def analyze_notebook(state: NotebookState, cells: list[CellState] | None = None) -> cg.Analysis:
    """Analyze *state*'s notebook, or *cells* in its place."""
    cg_cells = [
        cg.Cell(
            id=cell.id,
            source=cell.source,
            language=cell.language.value,
            meta={
                "language": cell.language.value,
                "mounts": mounts_of(state, cell, cell.source),
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
        analyzers=_analyzers(),
    )


def analysis_with(
    cell: CellState, state: NotebookState, analysis: cg.Analysis | None = None
) -> cg.Analysis:
    """*analysis* when it read *cell* as it is; otherwise the notebook analyzed again with
    *cell* in place of its namesake (or after the others, when it has none).

    The source *analysis* read is checked, not the state's: a source set on the state
    and not yet analyzed is the cell as it is.
    """
    current = state.get_cell(cell.id)
    read = analysis.cells.get(cell.id) if analysis is not None else None
    if current is not None and read is not None and read.source == cell.source:
        assert analysis is not None
        return analysis
    cells = [cell if c.id == cell.id else c for c in state.cells]
    if current is None:
        cells.append(cell)
    return analyze_notebook(state, cells)


@functools.lru_cache(maxsize=4096)
def definition_key(source: str, language: str = "python") -> str:
    """The definition key of a *language* cell with *source*: its own lines decide it."""
    cell = cg.Cell(id="cell", source=source, language=language, meta={"language": language})
    analysis = cg.analyze(
        [cell],
        vocabulary.strata_vocabulary(),
        analyzers=_analyzers(),
    )
    key = analysis.cells[cell.id].definition_key
    assert key is not None
    return key
