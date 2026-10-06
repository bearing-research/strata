"""NotebookOps: the one operation set an agent drives a notebook through.

``LocalNotebookOps`` runs an in-process session offline; ``RemoteNotebookOps``
drives a running server's session over HTTP. Both return the same curated view
models, built by one wire-dict mapper (``_cell_view_from_wire``) so the CLI, the
MCP server and the remote path cannot drift. Import-light: no FastAPI tree, so
the CLI stays fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, JsonValue, field_validator

if TYPE_CHECKING:
    import httpx

    from strata.notebook.dag import NotebookDag
    from strata.notebook.models import CellOutput, CellState, WorkerSpec
    from strata.notebook.session import NotebookSession


# --- View models: the agent-facing contract ---


class OutputView(BaseModel):
    """One of a cell's display outputs, agent-facing."""

    content_type: str | None = None
    preview: JsonValue = None
    rows: int | None = None
    columns: list[str] | None = None
    # For images and blobs, which have no useful ``preview``; ``save_output``
    # writes the artifact to a file.
    artifact_uri: str | None = None
    bytes: int = 0


class SavedOutput(BaseModel):
    """Where a cell's display output was written, agent-facing."""

    cell_id: str
    index: int
    path: str
    content_type: str
    bytes: int


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


class WidgetControlView(BaseModel):
    """One control of a widget cell, and what it is currently set to.

    The selection lives in runtime state, not the source, so without this an agent
    could not tell the current value from the declared default.
    """

    name: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    default: Any = None
    # The selection, else the declared default: what the cell computes with.
    value: Any = None


class CellView(BaseModel):
    """An agent-facing view of one cell, projected from ``CellState``.

    Omits internal bookkeeping such as provenance hashes and remote-build state.
    """

    id: str
    name: str
    language: str
    status: str
    source: str
    staleness_reasons: list[str]
    # Promised by the MCP `get_notebook` description.
    defines: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    upstream_ids: list[str]
    downstream_ids: list[str]
    outputs: list[OutputView]
    console_stdout: str
    console_stderr: str
    # Last failure, only while the source is unchanged since.
    error: str | None = None
    test: CellTestView | None = None
    # Empty for older cells and on a personal server with no declared author.
    created_by: str = ""
    updated_by: str = ""
    # Widget cells only.
    controls: list[WidgetControlView] = Field(default_factory=list)


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
    # Why the graph could not be built, or None. Otherwise a failed build
    # looks identical to a notebook with no dependencies.
    error: str | None = None


# Per-stream cap: stdout arrives uncapped, and one print-heavy cell would flood
# an agent's context. Matches the CLI's own cap.
MAX_AGENT_CONSOLE_CHARS = 10_000


def _cap_console(text: str) -> str:
    if len(text) <= MAX_AGENT_CONSOLE_CHARS:
        return text
    omitted = len(text) - MAX_AGENT_CONSOLE_CHARS
    return text[:MAX_AGENT_CONSOLE_CHARS] + f"… [+{omitted} chars truncated]"


class RunResult(BaseModel):
    """The outcome of running a single cell, agent-facing."""

    cell_id: str
    status: str  # "ok" | "error"
    cache_hit: bool
    execution_method: str
    duration_ms: float
    error: str | None = None
    # A stable name for the failure, e.g. ``fetch_pin_mismatch``; ``None`` when it has none.
    error_code: str | None = None
    stdout: str = ""
    stderr: str = ""

    # Every agent-facing run result, local or remote, passes through here.
    @field_validator("stdout", "stderr")
    @classmethod
    def _truncate_console(cls, value: str) -> str:
        return _cap_console(value)


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
    # Answers "do I already have this?" without a call per cell.
    defines: list[str] = Field(default_factory=list)


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


class WorkerView(BaseModel):
    """An agent-facing view of one registered worker, projected from ``WorkerSpec``."""

    name: str
    backend: str  # "local" | "executor"
    transport: str  # "local" | "direct" | "signed" | "embedded" | "executor"
    url: str | None = None
    runtime_id: str | None = None
    token_env: str | None = None
    is_default: bool = False


class WorkerListView(BaseModel):
    """The notebook's registered workers plus which one is the default."""

    default: str | None  # configured notebook default; ``None`` ⇒ the built-in local worker
    editable: bool  # False in service mode (definitions are server-managed)
    workers: list[WorkerView]


class NotebookOpsError(Exception):
    """An operation failed (unknown cell, DAG cycle, ...).

    The CLI maps this to a structured ``{"error": ...}`` on stdout and exit 1, not
    the exit-2 usage path.
    """


@runtime_checkable
class NotebookOps(Protocol):
    """The notebook operation set, shared by the local and remote backends and the MCP server."""

    def list_cells(self) -> list[CellView]:
        """Return every cell, in notebook order."""
        ...

    def get_cell(self, cell_id: str) -> CellView:
        """Return one cell's curated view.

        Raises
        ------
        NotebookOpsError
            If no cell with ``cell_id`` exists.
        """
        ...

    def save_output(self, cell_id: str, dest: Path, *, index: int = -1) -> SavedOutput:
        """Write one of a cell's display outputs to *dest* and say where.

        This is the only way to get an output's bytes (a plot, an image) out. The
        caller chooses the destination; *dest*'s parent must exist and an existing
        file is overwritten. ``index`` counts in emission order, ``-1`` being the last.

        Raises
        ------
        NotebookOpsError
            If the cell does not exist, has no display output at ``index``, or its
            stored bytes cannot be read back.
        """
        ...

    def dag(self) -> DagView:
        """Return the dependency graph."""
        ...

    def status(self) -> NotebookStatus:
        """Return a compact per-cell status and staleness summary."""
        ...

    async def run_cell(self, cell_id: str, *, mode: str = "normal") -> RunResult:
        """Execute one cell and return its outcome.

        ``mode``: ``normal`` uses the cache and materializes stale upstreams; ``rerun``
        bypasses the target's cache but still materializes upstreams; ``force`` runs
        against whatever upstream artifacts already exist.

        Raises
        ------
        NotebookOpsError
            If no such cell exists, or ``mode`` is not recognized.
        """
        ...

    async def run_tests(self, cell_id: str) -> TestRunResult:
        """Run a cell's unit tests (``cells/{id}.test.py``) and return per-test outcomes.

        Raises
        ------
        NotebookOpsError
            If no such cell exists or it has no test source.
        """
        ...

    def set_cell_tests(self, cell_id: str, test_source: str) -> CellView:
        """Set a Python cell's unit-test source; an empty string clears the tests.

        Raises
        ------
        NotebookOpsError
            If no such cell exists or it is not a Python cell.
        """
        ...

    def add_cell(
        self, source: str, *, after: str | None = None, language: str = "python"
    ) -> CellView:
        """Add a new cell with a backend-minted id, after ``after`` or at the end.

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
    """:class:`NotebookOps` over an in-process session; offline, no server."""

    def __init__(self, notebook_dir: Path, author: str | None = None) -> None:
        # Lazy imports keep ``--help`` and path errors cheap.
        from strata.notebook.authorship import resolve_author
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession

        self.notebook_dir = notebook_dir
        # Resolved once: one handle is one caller.
        self.author = resolve_author(author)
        state = parse_notebook(notebook_dir)
        self._session = NotebookSession(state, notebook_dir)
        self._executor: object | None = None
        self._staleness_computed = False

    @classmethod
    def from_session(cls, session: NotebookSession, author: str | None = None) -> LocalNotebookOps:
        """Wrap an already-open ``NotebookSession`` instead of opening a new one.

        The in-process MCP server uses this to share the server's live session (warm
        artifact cache, current cell state) so tools see what the UI sees. ``author``
        is credited for edits made through this handle.
        """
        from strata.notebook.authorship import resolve_author

        ops = cls.__new__(cls)
        ops.notebook_dir = session.path
        ops.author = resolve_author(author)
        ops._session = session
        ops._executor = None
        # The server keeps staleness current; this handle only views it.
        ops._staleness_computed = True
        return ops

    def _ensure_staleness(self) -> None:
        """Compute staleness once per handle before any status is reported.

        A session starts every cell IDLE (status is not persisted). The server's
        ``SessionManager`` computes staleness on open; an offline handle must do the
        same or every cell reads back ``idle`` with no outputs. Once is enough: one
        handle is one command, and the computation can reach ``@fetch`` URLs and
        ``@table`` catalogs.
        """
        if self._staleness_computed:
            return
        self._staleness_computed = True
        self._session.compute_staleness()

    def list_cells(self) -> list[CellView]:
        """List every cell in order (see :meth:`NotebookOps.list_cells`)."""
        self._ensure_staleness()
        return [_cell_view(cell) for cell in self._session.notebook_state.cells]

    def get_cell(self, cell_id: str) -> CellView:
        """Project one cell (see :meth:`NotebookOps.get_cell`)."""
        self._ensure_staleness()
        cell = self._session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        return _cell_view(cell)

    def save_output(self, cell_id: str, dest: Path, *, index: int = -1) -> SavedOutput:
        """Write a display output to *dest* (see :meth:`NotebookOps.save_output`)."""
        return _save_blob(self._session, cell_id, dest, index)

    def dag(self) -> DagView:
        """Build the DAG view (see :meth:`NotebookOps.dag`)."""
        return _dag_view(self._session.dag, getattr(self._session, "dag_error", None))

    def status(self) -> NotebookStatus:
        """Summarize per-cell status (see :meth:`NotebookOps.status`)."""
        self._ensure_staleness()
        state = self._session.notebook_state
        return NotebookStatus(
            notebook_id=state.id,
            name=state.name,
            cells=[_status_row(cell) for cell in state.cells],
        )

    # -- execution -----------------------------------------------------------

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

    def use_existing_environment(self) -> None:
        """Take the prepared notebook venv as the interpreter (``--no-sync``).

        Raises
        ------
        NotebookOpsError
            If the venv has no interpreter.
        """
        from strata.notebook.cli import _use_existing_environment

        ok, err = _use_existing_environment(self._session)
        if not ok:
            raise NotebookOpsError(err or "notebook venv is not usable")

    async def run_cell(self, cell_id: str, *, mode: str = "normal") -> RunResult:
        """Execute one cell (see :meth:`NotebookOps.run_cell`).

        Assumes the environment is ready: call :meth:`sync_environment` first.
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
            error_code=result.error_code,
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

    def set_cell_tests(self, cell_id: str, test_source: str) -> CellView:
        """Set a cell's unit-test source (see :meth:`NotebookOps.set_cell_tests`)."""
        from strata.notebook.writer import write_cell_tests

        cell = self._session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        if cell.language.value != "python":
            raise NotebookOpsError(f"cell {cell_id!r} is not a Python cell")
        try:
            write_cell_tests(self.notebook_dir, cell_id, test_source)
        except FileNotFoundError as exc:
            raise NotebookOpsError(str(exc)) from exc
        cell.test_source = test_source
        return _cell_view(cell)

    async def aclose(self) -> None:
        """Release the warm process pool, if one was started (cleanup on exit)."""
        from strata.notebook.cli import _drain_warm_pool

        await _drain_warm_pool(self._session)

    def _ensure_executor(self):
        if self._executor is None:
            from strata.notebook.executor import CellExecutor

            self._executor = CellExecutor(self._session)
        return self._executor

    # --- authoring + env ---

    _LANGUAGES = ("python", "markdown", "sql", "r", "prompt", "widget")

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
        add_cell_to_notebook(
            self.notebook_dir, cell_id, after, language=language, author=self.author
        )
        write_cell(self.notebook_dir, cell_id, source, author=self.author)
        self._reload()
        return self.get_cell(cell_id)

    def edit_cell(self, cell_id: str, source: str) -> CellView:
        """Replace a cell's source (see :meth:`NotebookOps.edit_cell`).

        Subject to the same soft lock as REST and WebSocket edits: on a live session a
        recent edit of this cell by someone else refuses this one. An offline session
        never refuses.
        """
        from strata.notebook.presence import lock_window_seconds
        from strata.notebook.writer import write_cell

        if self._session.notebook_state.get_cell(cell_id) is None:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        held_by = self._session.presence.holder(cell_id, self.author, lock_window_seconds())
        if held_by is not None:
            raise NotebookOpsError(
                f"{held_by} changed cell {cell_id!r} moments ago; retry in a few seconds"
            )
        write_cell(self.notebook_dir, cell_id, source, author=self.author)
        self._session.presence.record_edit(cell_id, self.author)
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
        import tomllib

        from strata.notebook.writer import reorder_cells

        # Read the current order from disk: a server, the TUI or another CLI
        # may have added or removed cells since this object was built.
        with open(self.notebook_dir / "notebook.toml", "rb") as handle:
            on_disk = tomllib.load(handle)
        order = [
            cell["id"]
            for cell in sorted(on_disk.get("cells", []), key=lambda c: c.get("order", 0))
            if cell.get("id")
        ]
        if cell_id not in order:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        order.remove(cell_id)
        order.insert(max(0, index), cell_id)
        reorder_cells(self.notebook_dir, order)
        self._reload()
        return self.list_cells()

    # -- workers -------------------------------------------------------------

    def list_workers(self) -> WorkerListView:
        """Return the notebook's workers and the default; the built-in ``local`` is listed first."""
        from strata.notebook.workers import (
            get_builtin_local_worker,
            notebook_worker_definitions_editable,
        )

        state = self._session.notebook_state
        default = state.worker
        specs = [get_builtin_local_worker(), *state.workers]
        return WorkerListView(
            default=default,
            editable=notebook_worker_definitions_editable(state),
            workers=[_worker_view(spec, default) for spec in specs],
        )

    def add_worker(
        self,
        name: str,
        *,
        url: str | None = None,
        transport: str = "direct",
        backend: str = "executor",
        runtime_id: str | None = None,
        token_env: str | None = None,
        set_default: bool = False,
    ) -> WorkerListView:
        """Register (or replace by name) a notebook-scoped worker.

        An ``executor`` worker requires ``url`` (its ``/v1/execute`` endpoint). With
        ``set_default`` the notebook's default worker is pointed at it.

        Raises
        ------
        NotebookOpsError
            If worker definitions aren't editable (service mode), the backend,
            transport or fields are invalid, or an ``executor`` worker is missing ``url``.
        """
        from pydantic import ValidationError

        from strata.notebook.models import WorkerBackendType, WorkerConfig, WorkerSpec
        from strata.notebook.workers import (
            check_worker_transport,
            notebook_worker_definitions_editable,
        )
        from strata.notebook.writer import update_notebook_worker, update_notebook_workers

        state = self._session.notebook_state
        if not notebook_worker_definitions_editable(state):
            raise NotebookOpsError("worker definitions are managed by the server in service mode")
        if backend == WorkerBackendType.EXECUTOR.value and not (url or "").strip():
            raise NotebookOpsError(f"executor worker {name!r} requires a url")
        try:
            check_worker_transport(transport)
            spec = WorkerSpec(
                name=name,
                backend=WorkerBackendType(backend),
                runtime_id=runtime_id,
                config=WorkerConfig(
                    url=url or None,
                    transport=transport,
                    token_env=token_env,
                ),
            )
        except (ValidationError, ValueError) as exc:
            raise NotebookOpsError(f"invalid worker {name!r}: {exc}") from exc

        others = [w for w in state.workers if w.name != spec.name]
        update_notebook_workers(self.notebook_dir, [*others, spec])
        if set_default:
            update_notebook_worker(self.notebook_dir, spec.name)
        self._reload()
        return self.list_workers()

    def remove_worker(self, name: str) -> WorkerListView:
        """Remove a notebook-scoped worker; clears the default if it named it.

        Raises
        ------
        NotebookOpsError
            If ``name`` is the built-in ``local`` worker, isn't defined, or
            definitions aren't editable.
        """
        from strata.notebook.workers import notebook_worker_definitions_editable
        from strata.notebook.writer import update_notebook_worker, update_notebook_workers

        state = self._session.notebook_state
        if not notebook_worker_definitions_editable(state):
            raise NotebookOpsError("worker definitions are managed by the server in service mode")
        if name == "local":
            raise NotebookOpsError("cannot remove the built-in 'local' worker")
        remaining = [w for w in state.workers if w.name != name]
        if len(remaining) == len(state.workers):
            raise NotebookOpsError(f"no worker named {name!r}")
        update_notebook_workers(self.notebook_dir, remaining)
        if state.worker == name:
            update_notebook_worker(self.notebook_dir, None)
        self._reload()
        return self.list_workers()

    def set_default_worker(self, name: str | None) -> WorkerListView:
        """Set (or clear, with ``None``/``"local"``) the notebook default worker.

        Raises
        ------
        NotebookOpsError
            If ``name`` isn't the built-in ``local`` worker or a defined worker.
        """
        from strata.notebook.writer import update_notebook_worker

        normalized = (name or "").strip() or None
        if normalized is not None and normalized != "local":
            known = {w.name for w in self._session.notebook_state.workers}
            if normalized not in known:
                raise NotebookOpsError(
                    f"no worker named {normalized!r} (add it first, or use 'local')"
                )
        # None for the implicit local default keeps notebook.toml clean.
        update_notebook_worker(self.notebook_dir, None if normalized == "local" else normalized)
        self._reload()
        return self.list_workers()

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
        """Re-read the notebook after a file mutation so reads see fresh state.

        In place rather than a fresh session: a fresh one logs every annotation
        diagnostic a second time. The reload computes staleness, so it counts as
        this handle's one computation.
        """
        self._session.reload()
        self._staleness_computed = True
        self._executor = None


_EMPTY_DAG: dict[str, Any] = {
    "edges": [],
    "topological_order": [],
    "leaves": [],
    "roots": [],
    "variable_producer": {},
}


class RemoteNotebookOps:
    """:class:`NotebookOps` over a running ``strata-notebook`` server.

    Drives a live session by ``session_id`` (the route ``{id}``, not the
    ``notebook.toml`` id), projecting the server's JSON through the shared wire
    mapper so views match the local backend. The session-state endpoint it reads
    is personal-mode only. A ``client`` passed in is not closed by :meth:`close`.
    """

    def __init__(
        self,
        base_url: str,
        session_id: str,
        *,
        client: httpx.Client | None = None,
        author: str | None = None,
    ) -> None:
        import httpx

        from strata.notebook.authorship import clean_author

        self._base_url = base_url.rstrip("/")
        self._session_id = session_id
        # The server decides authorship (ignoring this when it can authenticate
        # the caller). Bounded here because the routes cap the field: an
        # over-long name would 422 where the local path succeeds.
        self._author = clean_author(author)

        self._owns_client = client is None
        self._client: httpx.Client = client if client is not None else httpx.Client(timeout=30.0)

    def _credit(self) -> dict[str, str]:
        """The author field, omitted rather than sent as null when unset."""
        return {"author": self._author} if self._author else {}

    def _send(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Issue one request, turning a connection failure into an ops error.

        ``timeout`` overrides the client default for one call (SSH-worker verbs may
        install ``strata-worker`` and run past the 30s default).
        """
        import httpx

        url = f"{self._base_url}{path}"
        kwargs: dict[str, Any] = {"json": json, "params": params}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            return self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise NotebookOpsError(f"cannot reach {self._base_url}: {exc}") from exc

    def _state(self) -> dict[str, Any]:
        """Fetch the live session snapshot (name, cells, dag) or raise."""
        resp = self._send("GET", f"/v1/notebooks/sessions/{self._session_id}")
        if resp.status_code == 404:
            raise NotebookOpsError(f"no session {self._session_id!r} on {self._base_url}")
        if resp.status_code >= 400:
            raise NotebookOpsError(
                f"server returned {resp.status_code} for session {self._session_id!r}"
            )
        return resp.json()

    def _cell_op(
        self,
        method: str,
        path: str,
        *,
        cell_id: str | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Issue a cell-scoped request and return the JSON body, mapping errors.

        ``404`` becomes "no cell ..." when *cell_id* is given (else the server's
        detail), ``409`` an environment-busy error, any other error status the
        server's detail message.
        """
        resp = self._send(method, path, json=json, params=params)
        if resp.status_code == 404:
            raise NotebookOpsError(
                f"no cell with id {cell_id!r}" if cell_id else _error_detail(resp)
            )
        if resp.status_code == 409:
            raise NotebookOpsError(f"environment busy: {_error_detail(resp)}")
        if resp.status_code >= 400:
            raise NotebookOpsError(_error_detail(resp))
        return resp.json() if resp.content else {}

    def list_cells(self) -> list[CellView]:
        """List every cell (see :meth:`NotebookOps.list_cells`)."""
        return [_cell_view_from_wire(cell) for cell in self._state().get("cells") or []]

    def get_cell(self, cell_id: str) -> CellView:
        """Project one cell (see :meth:`NotebookOps.get_cell`)."""
        for cell in self._state().get("cells") or []:
            if cell.get("id") == cell_id:
                return _cell_view_from_wire(cell)
        raise NotebookOpsError(f"no cell with id {cell_id!r}")

    def save_output(self, cell_id: str, dest: Path, *, index: int = -1) -> SavedOutput:
        """Fetch a display output's bytes and write them locally.

        The client writes the file, not the server, so a caller cannot make the
        server write to an arbitrary path.
        """
        resp = self._send(
            "GET",
            f"/v1/notebooks/{self._session_id}/cells/{cell_id}/outputs/{index}/blob",
        )
        if resp.status_code == 404:
            raise NotebookOpsError(_error_detail(resp))
        if resp.status_code >= 400:
            raise NotebookOpsError(_error_detail(resp))
        blob = resp.content
        dest.write_bytes(blob)
        resolved = int(resp.headers.get("X-Strata-Output-Index", index))
        return SavedOutput(
            cell_id=cell_id,
            index=resolved,
            path=str(dest),
            # Starlette appends `; charset=utf-8` to text/* types; drop it so
            # remote matches local.
            content_type=resp.headers.get("content-type", "application/octet-stream")
            .split(";", 1)[0]
            .strip(),
            bytes=len(blob),
        )

    def dag(self) -> DagView:
        """Build the DAG view (see :meth:`NotebookOps.dag`)."""
        return DagView.model_validate(self._state().get("dag") or _EMPTY_DAG)

    def status(self) -> NotebookStatus:
        """Summarize per-cell status (see :meth:`NotebookOps.status`)."""
        state = self._state()
        return NotebookStatus(
            notebook_id=state.get("id") or "",
            name=state.get("name") or "",
            cells=[_status_row_from_wire(cell) for cell in state.get("cells") or []],
        )

    # -- execution -----------------------------------------------------------

    async def run_cell(self, cell_id: str, *, mode: str = "normal") -> RunResult:
        """Execute one cell on the server; the server owns its venv, so there is no client sync."""
        import asyncio

        data = await asyncio.to_thread(
            self._cell_op,
            "POST",
            f"/v1/notebooks/{self._session_id}/cells/{cell_id}/execute",
            cell_id=cell_id,
            params={"mode": mode},
        )
        return _run_result_from_wire(data)

    async def run_tests(self, cell_id: str) -> TestRunResult:
        """Run a cell's unit tests on the server (see :meth:`NotebookOps.run_tests`)."""
        import asyncio

        data = await asyncio.to_thread(
            self._cell_op,
            "POST",
            f"/v1/notebooks/{self._session_id}/cells/{cell_id}/tests",
            cell_id=cell_id,
        )
        return _test_run_result_from_wire(data, cell_id)

    def set_cell_tests(self, cell_id: str, test_source: str) -> CellView:
        """Set a cell's unit-test source (see :meth:`NotebookOps.set_cell_tests`)."""
        data = self._cell_op(
            "PUT",
            f"/v1/notebooks/{self._session_id}/cells/{cell_id}/tests",
            cell_id=cell_id,
            json={"source": test_source},
        )
        return _cell_view_from_wire(data)

    # -- authoring -----------------------------------------------------------

    def add_cell(
        self, source: str, *, after: str | None = None, language: str = "python"
    ) -> CellView:
        """Add a new cell (see :meth:`NotebookOps.add_cell`).

        The add endpoint creates an empty cell, so this POSTs then PUTs the source.
        """
        base = f"/v1/notebooks/{self._session_id}/cells"
        created = self._cell_op(
            "POST", base, json={"after_cell_id": after, "language": language, **self._credit()}
        )
        cell_id = _require_field(created, "id")
        updated = self._cell_op(
            "PUT", f"{base}/{cell_id}", cell_id=cell_id, json={"source": source, **self._credit()}
        )
        return _cell_view_from_wire(_require_field(updated, "cell"))

    def edit_cell(self, cell_id: str, source: str) -> CellView:
        """Replace a cell's source (see :meth:`NotebookOps.edit_cell`)."""
        updated = self._cell_op(
            "PUT",
            f"/v1/notebooks/{self._session_id}/cells/{cell_id}",
            cell_id=cell_id,
            json={"source": source, **self._credit()},
        )
        return _cell_view_from_wire(_require_field(updated, "cell"))

    def remove_cell(self, cell_id: str) -> None:
        """Delete a cell (see :meth:`NotebookOps.remove_cell`)."""
        self._cell_op(
            "DELETE", f"/v1/notebooks/{self._session_id}/cells/{cell_id}", cell_id=cell_id
        )

    def move_cell(self, cell_id: str, index: int) -> list[CellView]:
        """Reorder a cell (see :meth:`NotebookOps.move_cell`)."""
        order = [cell.get("id") for cell in self._state().get("cells") or []]
        if cell_id not in order:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        order.remove(cell_id)
        order.insert(max(0, index), cell_id)
        result = self._cell_op(
            "PUT", f"/v1/notebooks/{self._session_id}/cells/reorder", json={"cell_ids": order}
        )
        return [_cell_view_from_wire(cell) for cell in result.get("cells") or []]

    async def add_dependency(self, package: str) -> DependencyResult:
        """Add a dependency on the server (see :meth:`NotebookOps.add_dependency`)."""
        import asyncio

        return await asyncio.to_thread(self._mutate_dependency, package, "add")

    async def remove_dependency(self, package: str) -> DependencyResult:
        """Remove a dependency on the server (see :meth:`NotebookOps.remove_dependency`)."""
        import asyncio

        return await asyncio.to_thread(self._mutate_dependency, package, "remove")

    def _mutate_dependency(self, package: str, action: str) -> DependencyResult:
        base = f"/v1/notebooks/{self._session_id}/dependencies"
        if action == "add":
            resp = self._send("POST", base, json={"package": package})
        else:
            resp = self._send("DELETE", f"{base}/{package}")
        if resp.status_code == 404:
            raise NotebookOpsError(f"no session {self._session_id!r} on {self._base_url}")
        if resp.status_code == 409:
            raise NotebookOpsError(f"environment busy: {_error_detail(resp)}")
        if resp.status_code == 400:
            # A failed resolve is an outcome, not an error, as in the local backend.
            return DependencyResult(
                package=package,
                action=action,
                success=False,
                lockfile_changed=False,
                error=_error_detail(resp),
            )
        if resp.status_code >= 400:
            raise NotebookOpsError(_error_detail(resp))
        data = resp.json()
        return DependencyResult(
            package=data.get("package") or package,
            action=action,
            success=True,
            lockfile_changed=data.get("lockfile_changed", False),
            error=None,
        )

    # -- ssh workers ---------------------------------------------------------

    def add_ssh_worker(
        self,
        ssh_target: str,
        *,
        name: str | None = None,
        set_default: bool = True,
        install: bool = True,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Provision, tunnel and register a worker over SSH on the server.

        The server owns the tunnel, so this only works against a running server.
        Returns the tunnel ``worker`` record plus the worker catalog.
        """
        body: dict[str, Any] = {
            "ssh_target": ssh_target,
            "set_default": set_default,
            "install": install,
        }
        if name:
            body["name"] = name
        resp = self._send(
            "POST",
            f"/v1/notebooks/{self._session_id}/workers/ssh",
            json=body,
            timeout=timeout,
        )
        if resp.status_code == 404:
            raise NotebookOpsError(f"no session {self._session_id!r} on {self._base_url}")
        if resp.status_code >= 400:
            raise NotebookOpsError(_error_detail(resp))
        return resp.json()

    def remove_ssh_worker(self, name: str, *, stop_remote: bool = False) -> dict[str, Any]:
        """Close a worker's SSH tunnel and remove its registration."""
        resp = self._send(
            "DELETE",
            f"/v1/notebooks/{self._session_id}/workers/ssh/{name}",
            params={"stop_remote": "true" if stop_remote else "false"},
        )
        if resp.status_code >= 400:
            raise NotebookOpsError(_error_detail(resp))
        return resp.json()

    def close(self) -> None:
        """Close the httpx client if this instance created it."""
        if self._owns_client:
            self._client.close()


# --- Projections: domain models to view models ---


# Both backends project the same wire dict (``CellState.serialize()`` locally,
# the server's JSON remotely), so their view models match.


def display_output_at(cell: CellState, index: int) -> tuple[CellOutput, int]:
    """Return one of *cell*'s display outputs and its resolved index.

    Negative indices count from the end; ``-1`` is the value of a trailing bare
    expression. Raises ``NotebookOpsError`` when there is no output at ``index``.
    """
    outputs = cell.display_outputs or (
        [cell.display_output] if cell.display_output is not None else []
    )
    if not outputs:
        raise NotebookOpsError(
            f"cell {cell.id!r} has no display output to save "
            "(run it, and end it in an expression that renders)"
        )
    resolved = index if index >= 0 else len(outputs) + index
    if not 0 <= resolved < len(outputs):
        raise NotebookOpsError(
            f"cell {cell.id!r} has {len(outputs)} display output(s); no index {index}"
        )
    return outputs[resolved], resolved


def _save_blob(session: NotebookSession, cell_id: str, dest: Path, index: int) -> SavedOutput:
    """Write a cell's display output to *dest*; shared by the local backend and the route."""
    cell = session.notebook_state.get_cell(cell_id)
    if cell is None:
        raise NotebookOpsError(f"no cell with id {cell_id!r}")
    output, resolved = display_output_at(cell, index)
    try:
        blob = session.read_display_blob(output)
    except ValueError as exc:
        raise NotebookOpsError(f"cell {cell_id!r} output {resolved}: {exc}") from exc
    dest.write_bytes(blob)
    return SavedOutput(
        cell_id=cell_id,
        index=resolved,
        path=str(dest),
        content_type=output.content_type,
        bytes=len(blob),
    )


def _output_view_from_wire(data: dict[str, Any]) -> OutputView:
    return OutputView(
        content_type=data.get("content_type"),
        preview=data.get("preview"),
        rows=data.get("rows"),
        columns=data.get("columns"),
        artifact_uri=data.get("artifact_uri"),
        bytes=int(data.get("bytes") or 0),
    )


def _test_view_from_wire(data: dict[str, Any]) -> CellTestView:
    return CellTestView(
        passed=data.get("passed", 0),
        failed=data.get("failed", 0),
        errored=data.get("errored", 0),
        skipped=data.get("skipped", 0),
        cases=[
            TestCaseView(
                name=case["name"], outcome=case["outcome"], message=case.get("message", "")
            )
            for case in data.get("tests", [])
        ],
    )


def _cell_view_from_wire(data: dict[str, Any]) -> CellView:
    """Project a serialized-cell wire dict into a :class:`CellView`."""
    annotations = data.get("annotations") or {}
    test = data.get("test_result")
    return CellView(
        id=data["id"],
        name=annotations.get("name") or "",
        language=data["language"],
        status=data["status"],
        source=data.get("source") or "",
        staleness_reasons=list(data.get("staleness_reasons") or []),
        defines=list(data.get("defines") or []),
        references=list(data.get("references") or []),
        upstream_ids=list(data.get("upstream_ids") or []),
        downstream_ids=list(data.get("downstream_ids") or []),
        outputs=[_output_view_from_wire(output) for output in data.get("display_outputs") or []],
        console_stdout=data.get("console_stdout") or "",
        console_stderr=data.get("console_stderr") or "",
        error=_cap_console(data["error"]) if data.get("error") else None,
        test=_test_view_from_wire(test) if test else None,
        created_by=data.get("created_by") or "",
        updated_by=data.get("updated_by") or "",
        controls=_control_views_from_wire(data.get("widget")),
    )


def _control_views_from_wire(widget: dict[str, Any] | None) -> list[WidgetControlView]:
    """The widget block of a serialized cell, or nothing for a non-widget cell."""
    if not widget:
        return []
    values = widget.get("values") or {}
    return [
        WidgetControlView(
            name=descriptor["name"],
            kind=descriptor.get("kind") or "",
            params=dict(descriptor.get("params") or {}),
            default=descriptor.get("default"),
            # The effective value: untouched controls use the declared default.
            value=values.get(descriptor["name"], descriptor.get("default")),
        )
        for descriptor in widget.get("descriptors") or []
    ]


def _status_row_from_wire(data: dict[str, Any]) -> CellStatusRow:
    annotations = data.get("annotations") or {}
    return CellStatusRow(
        id=data["id"],
        name=annotations.get("name") or "",
        language=data["language"],
        status=data["status"],
        staleness_reasons=list(data.get("staleness_reasons") or []),
        defines=list(data.get("defines") or []),
    )


def _run_result_from_wire(data: dict[str, Any]) -> RunResult:
    """Project the server's execute-result wire dict into a :class:`RunResult`.

    The server reports ``status`` as ``"ready"``/``"error"``; :class:`RunResult`
    uses ``"ok"``/``"error"``.
    """
    return RunResult(
        cell_id=data["cell_id"],
        status="ok" if data.get("status") == "ready" else "error",
        cache_hit=data.get("cache_hit", False),
        execution_method=data.get("execution_method") or "",
        duration_ms=data.get("duration_ms", 0.0),
        error=data.get("error"),
        error_code=data.get("error_code"),
        stdout=data.get("stdout") or "",
        stderr=data.get("stderr") or "",
    )


def _test_run_result_from_wire(data: dict[str, Any], cell_id: str) -> TestRunResult:
    return TestRunResult(
        cell_id=data.get("cell_id") or cell_id,
        passed=data.get("passed", 0),
        failed=data.get("failed", 0),
        errored=data.get("errored", 0),
        skipped=data.get("skipped", 0),
        pytest_unavailable=data.get("pytest_unavailable", False),
        cases=[
            TestCaseView(
                name=case["name"], outcome=case["outcome"], message=case.get("message", "")
            )
            for case in data.get("tests", [])
        ],
    )


def _error_detail(resp: Any) -> str:
    """Pull a human message out of a FastAPI error response body."""
    try:
        body = resp.json()
    except ValueError:
        return resp.text or f"server returned {resp.status_code}"
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return detail.get("message") or str(detail)
    if isinstance(detail, str):
        return detail
    return f"server returned {resp.status_code}"


def _require_field(data: dict[str, Any], key: str) -> Any:
    """Return ``data[key]``, or raise because a response missing it is malformed."""
    value = data.get(key)
    if value is None:
        raise NotebookOpsError(f"malformed server response: missing {key!r}")
    return value


def _cell_view(cell: CellState) -> CellView:
    """Project a cell locally through the same wire mapper the remote path uses."""
    return _cell_view_from_wire(cell.serialize())


def _status_row(cell: CellState) -> CellStatusRow:
    return _status_row_from_wire(cell.serialize())


def _worker_view(worker: WorkerSpec, default: str | None) -> WorkerView:
    """Project a :class:`WorkerSpec` into an agent-facing :class:`WorkerView`.

    A ``default`` of ``None`` means the notebook falls back to the built-in
    ``local`` worker, so that row is the one flagged ``is_default``.
    """
    from strata.notebook.workers import worker_transport

    return WorkerView(
        name=worker.name,
        backend=worker.backend.value,
        transport=worker_transport(worker),
        url=worker.config.url,
        runtime_id=worker.runtime_id,
        token_env=worker.config.token_env,
        is_default=default == worker.name or (default is None and worker.name == "local"),
    )


def _dag_view(dag: NotebookDag | None, error: str | None = None) -> DagView:
    from strata.notebook.dag import producer_cell_label

    if dag is None:
        return DagView(
            edges=[],
            topological_order=[],
            leaves=[],
            roots=[],
            variable_producer={},
            error=error or "notebook DAG could not be built",
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
        variable_producer={v: producer_cell_label(p) for v, p in dag.variable_producer.items()},
    )
