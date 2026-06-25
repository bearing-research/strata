"""NotebookOps — the operation core an agent drives a notebook through.

A single operation set, exposed today by the ``strata`` CLI and (later) the MCP
server, so an agent has a full-feature tool for a notebook without re-deriving
the verbs in each surface. Two backends implement the protocol:

- :class:`LocalNotebookOps` (here) — an in-process ``NotebookSession``, offline,
  the same path ``strata run`` takes. No server required.
- ``RemoteNotebookOps`` (a later phase) — an httpx REST client against a running
  ``strata-notebook``, so the same commands drive a session a human can watch in
  the TUI / web UI.

The verbs return small **curated view models** (``CellView`` / ``DagView`` /
``NotebookStatus``) — agent-facing projections of the internal ``CellState`` /
``NotebookDag`` domain models, fully typed (no ``Any``) and free of internal
bookkeeping. They are *the* contract: both backends return them and the MCP
server wraps them, so all three surfaces agree. (The raw server ``serialize()``
wire dict — ``CellState`` + ~7 computed overlays — is deliberately not modelled;
see ``docs/internal/design-cli-hardening.md``.)

This module is import-light: it pulls ``parse_notebook`` / ``NotebookSession``
lazily and never imports the FastAPI server tree, so ``strata cell …`` stays a
fast CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, JsonValue

if TYPE_CHECKING:
    from strata.notebook.dag import NotebookDag
    from strata.notebook.models import CellOutput, CellState, CellTestResult


# ---------------------------------------------------------------------------
# Curated view models — the agent-facing contract.
# ---------------------------------------------------------------------------


class OutputView(BaseModel):
    """One of a cell's display outputs, agent-facing."""

    content_type: str | None = None
    preview: JsonValue = None
    rows: int | None = None
    columns: list[str] | None = None


class TestCaseView(BaseModel):
    """One pytest test case in a :class:`CellTestView`."""

    name: str
    outcome: str
    message: str = ""


class CellTestView(BaseModel):
    """A cell's last unit-test run, agent-facing."""

    passed: int
    failed: int
    errored: int
    skipped: int
    cases: list[TestCaseView]


class CellView(BaseModel):
    """An agent-facing view of one cell, projected from ``CellState``.

    Carries what an agent needs to read a cell — source, status, dependency
    links, outputs (with previews), console, and any test result — without the
    internal bookkeeping fields (provenance hashes, remote-build state, …).
    """

    id: str
    name: str
    language: str
    status: str
    source: str
    staleness_reasons: list[str]
    upstream_ids: list[str]
    downstream_ids: list[str]
    outputs: list[OutputView]
    console_stdout: str
    console_stderr: str
    test: CellTestView | None = None


class DagEdgeView(BaseModel):
    """One variable-level dependency edge in a :class:`DagView`."""

    from_cell_id: str
    to_cell_id: str
    variable: str


class DagView(BaseModel):
    """A notebook's dependency graph, agent-facing (JSON view of ``NotebookDag``)."""

    edges: list[DagEdgeView]
    topological_order: list[str]
    leaves: list[str]
    roots: list[str]
    variable_producer: dict[str, str]


class CellStatusRow(BaseModel):
    """One cell's row in a :class:`NotebookStatus` summary."""

    id: str
    name: str
    language: str
    status: str
    staleness_reasons: list[str]


class NotebookStatus(BaseModel):
    """A compact per-cell status + staleness summary for a notebook."""

    notebook_id: str
    name: str
    cells: list[CellStatusRow]


class NotebookOpsError(Exception):
    """An operation failed (unknown cell, DAG cycle, …).

    Distinct from an invocation error (bad path): the CLI maps this to a
    structured ``{"error": …}`` on stdout + exit 1, not the exit-2 usage path.
    """


@runtime_checkable
class NotebookOps(Protocol):
    """The notebook operation set.

    Read-only verbs first (P0); run / test / author / env verbs land in later
    phases on this same protocol. Both the local and remote backends implement
    it, and the MCP server wraps it, so all three surfaces share one contract.
    """

    def list_cells(self) -> list[CellView]:
        """Return every cell, in notebook order.

        Returns
        -------
        list of CellView
            One curated view per cell, in display order.
        """
        ...

    def get_cell(self, cell_id: str) -> CellView:
        """Return one cell's curated view.

        Parameters
        ----------
        cell_id : str
            Identifier of the cell to fetch.

        Returns
        -------
        CellView
            The cell's agent-facing view (source, status, outputs, …).

        Raises
        ------
        NotebookOpsError
            If no cell with ``cell_id`` exists in the notebook.
        """
        ...

    def dag(self) -> DagView:
        """Return the dependency graph.

        Returns
        -------
        DagView
            Variable-level ``edges``, a ``topological_order``, the ``leaves`` and
            ``roots``, and the ``variable_producer`` map.
        """
        ...

    def status(self) -> NotebookStatus:
        """Return a compact per-cell status + staleness summary.

        Returns
        -------
        NotebookStatus
            ``notebook_id``, ``name``, and one :class:`CellStatusRow` per cell.
        """
        ...


class LocalNotebookOps:
    """:class:`NotebookOps` over an in-process session — offline, no server.

    Parameters
    ----------
    notebook_dir : Path
        Path to a notebook directory (must contain ``notebook.toml``).
    """

    def __init__(self, notebook_dir: Path) -> None:
        # Lazy heavy imports so ``--help`` / path errors stay cheap.
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession

        state = parse_notebook(notebook_dir)
        self._session = NotebookSession(state, notebook_dir)

    def list_cells(self) -> list[CellView]:
        """List every cell in order (see :meth:`NotebookOps.list_cells`)."""
        return [_cell_view(cell) for cell in self._session.notebook_state.cells]

    def get_cell(self, cell_id: str) -> CellView:
        """Project one cell (see :meth:`NotebookOps.get_cell`)."""
        cell = self._session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        return _cell_view(cell)

    def dag(self) -> DagView:
        """Build the DAG view (see :meth:`NotebookOps.dag`)."""
        return _dag_view(self._session.dag)

    def status(self) -> NotebookStatus:
        """Summarize per-cell status (see :meth:`NotebookOps.status`)."""
        state = self._session.notebook_state
        return NotebookStatus(
            notebook_id=state.id,
            name=state.name,
            cells=[_status_row(cell) for cell in state.cells],
        )


# ---------------------------------------------------------------------------
# Projections: domain models → view models.
# ---------------------------------------------------------------------------


def _cell_name(cell: CellState) -> str:
    """The cell's display name — its ``# @name`` annotation, else empty."""
    from strata.notebook.annotations import parse_annotations

    return parse_annotations(cell.source).name or ""


def _staleness_reasons(cell: CellState) -> list[str]:
    if cell.staleness is None:
        return []
    return [reason.value for reason in cell.staleness.reasons]


def _output_view(output: CellOutput) -> OutputView:
    return OutputView(
        content_type=output.content_type,
        preview=output.preview,
        rows=output.rows,
        columns=output.columns,
    )


def _test_view(result: CellTestResult) -> CellTestView:
    return CellTestView(
        passed=result.passed,
        failed=result.failed,
        errored=result.errored,
        skipped=result.skipped,
        cases=[
            TestCaseView(name=case.name, outcome=case.outcome, message=case.message)
            for case in result.tests
        ],
    )


def _cell_view(cell: CellState) -> CellView:
    return CellView(
        id=cell.id,
        name=_cell_name(cell),
        language=cell.language.value,
        status=cell.status.value,
        source=cell.source,
        staleness_reasons=_staleness_reasons(cell),
        upstream_ids=list(cell.upstream_ids),
        downstream_ids=list(cell.downstream_ids),
        outputs=[_output_view(output) for output in cell.display_outputs],
        console_stdout=cell.console_stdout,
        console_stderr=cell.console_stderr,
        test=_test_view(cell.test_result) if cell.test_result is not None else None,
    )


def _status_row(cell: CellState) -> CellStatusRow:
    return CellStatusRow(
        id=cell.id,
        name=_cell_name(cell),
        language=cell.language.value,
        status=cell.status.value,
        staleness_reasons=_staleness_reasons(cell),
    )


def _dag_view(dag: NotebookDag | None) -> DagView:
    if dag is None:
        return DagView(
            edges=[],
            topological_order=[],
            leaves=[],
            roots=[],
            variable_producer={},
        )
    return DagView(
        edges=[
            DagEdgeView(
                from_cell_id=edge.from_cell_id,
                to_cell_id=edge.to_cell_id,
                variable=edge.variable,
            )
            for edge in dag.edges
        ],
        topological_order=list(dag.topological_order),
        leaves=list(dag.leaves),
        roots=list(dag.roots),
        variable_producer=dict(dag.variable_producer),
    )
