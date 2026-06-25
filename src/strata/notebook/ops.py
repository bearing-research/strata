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


class RunResult(BaseModel):
    """The outcome of running a single cell, agent-facing."""

    cell_id: str
    status: str  # "ok" | "error"
    cache_hit: bool
    execution_method: str
    duration_ms: float
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


class TestRunResult(BaseModel):
    """The outcome of running a cell's unit tests, agent-facing."""

    cell_id: str
    passed: int
    failed: int
    errored: int
    skipped: int
    pytest_unavailable: bool
    cases: list[TestCaseView]


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


class DependencyResult(BaseModel):
    """The outcome of adding or removing a notebook dependency, agent-facing."""

    package: str
    action: str  # "add" | "remove"
    success: bool
    lockfile_changed: bool
    error: str | None = None


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

    async def run_cell(self, cell_id: str, *, mode: str = "normal") -> RunResult:
        """Execute one cell and return its outcome.

        Parameters
        ----------
        cell_id : str
            Identifier of the cell to run.
        mode : {"normal", "rerun", "force"}, optional
            ``normal`` uses the cache and materializes stale upstreams; ``rerun``
            bypasses the target's cache but still materializes upstreams;
            ``force`` runs against whatever upstream artifacts already exist.

        Returns
        -------
        RunResult
            Execution metadata (status, cache hit, duration, method) plus the
            cell's captured stdout / stderr and any error.

        Raises
        ------
        NotebookOpsError
            If no such cell exists, or ``mode`` is not recognized.
        """
        ...

    async def run_tests(self, cell_id: str) -> TestRunResult:
        """Run a cell's unit tests and return per-test outcomes.

        Parameters
        ----------
        cell_id : str
            Identifier of the (Python) cell whose ``cells/{id}.test.py`` to run.

        Returns
        -------
        TestRunResult
            Pass / fail / error / skip counts and per-test cases.

        Raises
        ------
        NotebookOpsError
            If no such cell exists or it has no test source.
        """
        ...

    def add_cell(
        self, source: str, *, after: str | None = None, language: str = "python"
    ) -> CellView:
        """Add a new cell with backend-minted id.

        Parameters
        ----------
        source : str
            The new cell's source.
        after : str or None, optional
            Insert after this cell id (``None`` appends at the end).
        language : str, optional
            One of ``python``, ``markdown``, ``sql``, ``r``, ``prompt``.

        Returns
        -------
        CellView
            The newly created cell.

        Raises
        ------
        NotebookOpsError
            If ``after`` names a missing cell, or ``language`` is unsupported.
        """
        ...

    def edit_cell(self, cell_id: str, source: str) -> CellView:
        """Replace a cell's source, returning the updated cell.

        Raises
        ------
        NotebookOpsError
            If no such cell exists.
        """
        ...

    def remove_cell(self, cell_id: str) -> None:
        """Delete a cell (and its source / test files).

        Raises
        ------
        NotebookOpsError
            If no such cell exists.
        """
        ...

    def move_cell(self, cell_id: str, index: int) -> list[CellView]:
        """Move a cell to ``index`` in notebook order; returns the new order.

        Raises
        ------
        NotebookOpsError
            If no such cell exists.
        """
        ...

    async def add_dependency(self, package: str) -> DependencyResult:
        """Add a Python dependency to the notebook (``uv add``)."""
        ...

    async def remove_dependency(self, package: str) -> DependencyResult:
        """Remove a Python dependency from the notebook (``uv remove``)."""
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

        self.notebook_dir = notebook_dir
        state = parse_notebook(notebook_dir)
        self._session = NotebookSession(state, notebook_dir)
        self._executor: object | None = None

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

    # -- execution (P1) ------------------------------------------------------

    async def sync_environment(self) -> None:
        """Sync the notebook venv (``uv sync``) before executing.

        Raises
        ------
        NotebookOpsError
            If the environment sync fails.
        """
        from strata.notebook.cli import _sync_environment

        ok, err = await _sync_environment(self._session)
        if not ok:
            raise NotebookOpsError(err or "environment sync failed")

    async def run_cell(self, cell_id: str, *, mode: str = "normal") -> RunResult:
        """Execute one cell (see :meth:`NotebookOps.run_cell`).

        Assumes the environment is ready — call :meth:`sync_environment` first
        (the CLI does, unless ``--no-sync``).
        """
        cell = self._session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        executor = self._ensure_executor()
        if mode == "normal":
            result = await executor.execute_cell(cell_id, cell.source)
        elif mode == "rerun":
            result = await executor.execute_cell_rerun(cell_id, cell.source)
        elif mode == "force":
            result = await executor.execute_cell_force(cell_id, cell.source)
        else:
            raise NotebookOpsError(f"unknown run mode {mode!r} (normal|rerun|force)")
        return RunResult(
            cell_id=result.cell_id,
            status="ok" if result.success else "error",
            cache_hit=result.cache_hit,
            execution_method=result.execution_method,
            duration_ms=result.duration_ms,
            error=result.error,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    async def run_tests(self, cell_id: str) -> TestRunResult:
        """Run a cell's unit tests (see :meth:`NotebookOps.run_tests`)."""
        cell = self._session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        if not cell.test_source.strip():
            raise NotebookOpsError(f"cell {cell_id!r} has no tests (cells/{cell_id}.test.py)")
        executor = self._ensure_executor()
        result = await executor.run_cell_tests(cell_id, cell.test_source)
        return TestRunResult(
            cell_id=cell_id,
            passed=result.passed,
            failed=result.failed,
            errored=result.errored,
            skipped=result.skipped,
            pytest_unavailable=result.pytest_unavailable,
            cases=[
                TestCaseView(name=case.name, outcome=case.outcome, message=case.message)
                for case in result.tests
            ],
        )

    async def aclose(self) -> None:
        """Release the warm process pool, if one was started (cleanup on exit)."""
        from strata.notebook.cli import _drain_warm_pool

        await _drain_warm_pool(self._session)

    def _ensure_executor(self):
        if self._executor is None:
            from strata.notebook.executor import CellExecutor

            self._executor = CellExecutor(self._session)
        return self._executor

    # -- authoring + env (P2) ------------------------------------------------

    _LANGUAGES = ("python", "markdown", "sql", "r", "prompt")

    def add_cell(
        self, source: str, *, after: str | None = None, language: str = "python"
    ) -> CellView:
        """Add a new cell (see :meth:`NotebookOps.add_cell`)."""
        import uuid

        from strata.notebook.writer import add_cell_to_notebook, write_cell

        if language not in self._LANGUAGES:
            raise NotebookOpsError(
                f"unsupported language {language!r} ({'|'.join(self._LANGUAGES)})"
            )
        if after is not None and self._session.notebook_state.get_cell(after) is None:
            raise NotebookOpsError(f"no cell with id {after!r} to insert after")
        cell_id = str(uuid.uuid4())[:8]
        add_cell_to_notebook(self.notebook_dir, cell_id, after, language=language)
        write_cell(self.notebook_dir, cell_id, source)
        self._reload()
        return self.get_cell(cell_id)

    def edit_cell(self, cell_id: str, source: str) -> CellView:
        """Replace a cell's source (see :meth:`NotebookOps.edit_cell`)."""
        from strata.notebook.writer import write_cell

        if self._session.notebook_state.get_cell(cell_id) is None:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        write_cell(self.notebook_dir, cell_id, source)
        self._reload()
        return self.get_cell(cell_id)

    def remove_cell(self, cell_id: str) -> None:
        """Delete a cell (see :meth:`NotebookOps.remove_cell`)."""
        from strata.notebook.writer import remove_cell_from_notebook

        if self._session.notebook_state.get_cell(cell_id) is None:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        remove_cell_from_notebook(self.notebook_dir, cell_id)
        self._reload()

    def move_cell(self, cell_id: str, index: int) -> list[CellView]:
        """Reorder a cell (see :meth:`NotebookOps.move_cell`)."""
        from strata.notebook.writer import reorder_cells

        order = [cell.id for cell in self._session.notebook_state.cells]
        if cell_id not in order:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        order.remove(cell_id)
        order.insert(max(0, index), cell_id)
        reorder_cells(self.notebook_dir, order)
        self._reload()
        return self.list_cells()

    async def add_dependency(self, package: str) -> DependencyResult:
        """Add a dependency (see :meth:`NotebookOps.add_dependency`)."""
        return await self._mutate_dependency(package, "add")

    async def remove_dependency(self, package: str) -> DependencyResult:
        """Remove a dependency (see :meth:`NotebookOps.remove_dependency`)."""
        return await self._mutate_dependency(package, "remove")

    async def _mutate_dependency(self, package: str, action: str) -> DependencyResult:
        outcome = await self._session.mutate_dependency(package, action=action)
        result = outcome.result
        return DependencyResult(
            package=result.package,
            action=result.action,
            success=result.success,
            lockfile_changed=result.lockfile_changed,
            error=result.error,
        )

    def _reload(self) -> None:
        """Re-parse the notebook after a file mutation so reads see fresh state."""
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession

        state = parse_notebook(self.notebook_dir)
        self._session = NotebookSession(state, self.notebook_dir)
        self._executor = None


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
