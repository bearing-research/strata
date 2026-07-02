"""NotebookOps — the operation core an agent drives a notebook through.

A single operation set, exposed today by the ``strata`` CLI and (later) the MCP
server, so an agent has a full-feature tool for a notebook without re-deriving
the verbs in each surface. Two backends implement the protocol:

- :class:`LocalNotebookOps` (here) — an in-process ``NotebookSession``, offline,
  the same path ``strata run`` takes. No server required.
- :class:`RemoteNotebookOps` (here) — an httpx client against a running
  ``strata-notebook``, so the same commands drive a session a human can watch in
  the TUI / web UI. Read verbs land first (P3b); run / author verbs follow.

The verbs return small **curated view models** (``CellView`` / ``DagView`` /
``NotebookStatus``) — agent-facing projections of the internal ``CellState`` /
``NotebookDag`` domain models, fully typed (no ``Any``) and free of internal
bookkeeping. They are *the* contract: both backends return them and the MCP
server wraps them, so all three surfaces agree. A single wire-dict mapper
(``_cell_view_from_wire``) builds them — the local backend feeds it
``CellState.serialize()``, the remote backend feeds it the server's JSON, so the
two paths cannot drift. (The raw server ``serialize()`` wire dict — ``CellState``
+ ~7 computed overlays — is deliberately not modelled as a view; see
``docs/internal/design-cli-hardening.md``.)

This module is import-light: it pulls ``parse_notebook`` / ``NotebookSession``
lazily and never imports the FastAPI server tree, so ``strata cell …`` stays a
fast CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, JsonValue

if TYPE_CHECKING:
    import httpx

    from strata.notebook.dag import NotebookDag
    from strata.notebook.models import CellState
    from strata.notebook.session import NotebookSession


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

    @classmethod
    def from_session(cls, session: NotebookSession) -> LocalNotebookOps:
        """Wrap an already-open ``NotebookSession`` instead of opening a new one.

        The CLI constructs one offline session per invocation; the in-process
        MCP server instead reuses the server's warm live session (its populated
        artifact cache, current cell state) so tools see exactly what the UI
        sees. ``notebook_dir`` is taken from the live session's path.

        Parameters
        ----------
        session : NotebookSession
            An open session, typically from the server's ``SessionManager``.
        """
        ops = cls.__new__(cls)
        ops.notebook_dir = session.path
        ops._session = session
        ops._executor = None
        return ops

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


_EMPTY_DAG: dict[str, Any] = {
    "edges": [],
    "topological_order": [],
    "leaves": [],
    "roots": [],
    "variable_producer": {},
}


class RemoteNotebookOps:
    """:class:`NotebookOps` reads over a running ``strata-notebook`` server.

    Drives a live session — the same one a human can watch in the TUI / web UI —
    by its ``session_id``, reading the session-state endpoint and projecting the
    server's JSON through the shared wire mapper, so a remote ``CellView`` is
    byte-for-byte what the local backend would return for that notebook.

    Read-only today (P3b); run / author verbs land in a later phase. The
    session-state endpoint it reads is personal-mode only, which matches the
    intended use — driving the session you're watching locally.

    Parameters
    ----------
    base_url : str
        Server root, e.g. ``http://localhost:8765``.
    session_id : str
        The open session to drive — the route ``{id}`` (a session id, *not* the
        ``notebook.toml`` id).
    client : httpx.Client or None, optional
        An httpx client to reuse; one is created (and owned) when omitted.
    """

    def __init__(
        self, base_url: str, session_id: str, *, client: httpx.Client | None = None
    ) -> None:
        import httpx

        self._base_url = base_url.rstrip("/")
        self._session_id = session_id
        self._owns_client = client is None
        self._client: httpx.Client = client if client is not None else httpx.Client(timeout=30.0)

    def _send(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Issue one request, turning a connection failure into an ops error."""
        import httpx

        url = f"{self._base_url}{path}"
        try:
            return self._client.request(method, url, json=json, params=params)
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

        ``404`` becomes "no cell …" when *cell_id* is known (else the server's
        detail), ``409`` an environment-busy error, any other ``4xx``/``5xx`` the
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
        """Execute one cell on the server (see :meth:`NotebookOps.run_cell`).

        The server owns its venv, so there is no client-side environment sync —
        unlike the local backend, which calls ``uv sync`` first.
        """
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

    # -- authoring -----------------------------------------------------------

    def add_cell(
        self, source: str, *, after: str | None = None, language: str = "python"
    ) -> CellView:
        """Add a new cell (see :meth:`NotebookOps.add_cell`).

        Two calls: POST to mint the cell (server assigns the id), then PUT its
        source — the add endpoint creates an empty cell.
        """
        base = f"/v1/notebooks/{self._session_id}/cells"
        created = self._cell_op("POST", base, json={"after_cell_id": after, "language": language})
        cell_id = _require_field(created, "id")
        updated = self._cell_op(
            "PUT", f"{base}/{cell_id}", cell_id=cell_id, json={"source": source}
        )
        return _cell_view_from_wire(_require_field(updated, "cell"))

    def edit_cell(self, cell_id: str, source: str) -> CellView:
        """Replace a cell's source (see :meth:`NotebookOps.edit_cell`)."""
        updated = self._cell_op(
            "PUT",
            f"/v1/notebooks/{self._session_id}/cells/{cell_id}",
            cell_id=cell_id,
            json={"source": source},
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
            # A failed `uv` resolve is a structured outcome, not an ops error —
            # parity with the local backend's DependencyResult(success=False).
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

    def close(self) -> None:
        """Close the httpx client if this instance created it."""
        if self._owns_client:
            self._client.close()


# ---------------------------------------------------------------------------
# Projections: domain models → view models.
# ---------------------------------------------------------------------------


# Both backends map from the same wire dict — ``CellState.serialize()`` locally,
# the server's JSON remotely — so the two paths produce identical view models.


def _output_view_from_wire(data: dict[str, Any]) -> OutputView:
    return OutputView(
        content_type=data.get("content_type"),
        preview=data.get("preview"),
        rows=data.get("rows"),
        columns=data.get("columns"),
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
    """Project a serialized-cell wire dict into a :class:`CellView`.

    Reads only the agent-facing fields; the wire dict's internal bookkeeping
    (provenance hashes, remote-build state, …) is simply not consulted.
    """
    annotations = data.get("annotations") or {}
    test = data.get("test_result")
    return CellView(
        id=data["id"],
        name=annotations.get("name") or "",
        language=data["language"],
        status=data["status"],
        source=data.get("source") or "",
        staleness_reasons=list(data.get("staleness_reasons") or []),
        upstream_ids=list(data.get("upstream_ids") or []),
        downstream_ids=list(data.get("downstream_ids") or []),
        outputs=[_output_view_from_wire(output) for output in data.get("display_outputs") or []],
        console_stdout=data.get("console_stdout") or "",
        console_stderr=data.get("console_stderr") or "",
        test=_test_view_from_wire(test) if test else None,
    )


def _status_row_from_wire(data: dict[str, Any]) -> CellStatusRow:
    annotations = data.get("annotations") or {}
    return CellStatusRow(
        id=data["id"],
        name=annotations.get("name") or "",
        language=data["language"],
        status=data["status"],
        staleness_reasons=list(data.get("staleness_reasons") or []),
    )


def _run_result_from_wire(data: dict[str, Any]) -> RunResult:
    """Project the server's execute-result wire dict into a :class:`RunResult`.

    The server renames ``success`` → ``status`` (``"ready"`` / ``"error"``); the
    agent-facing :class:`RunResult` uses ``"ok"`` / ``"error"``.
    """
    return RunResult(
        cell_id=data["cell_id"],
        status="ok" if data.get("status") == "ready" else "error",
        cache_hit=data.get("cache_hit", False),
        execution_method=data.get("execution_method") or "",
        duration_ms=data.get("duration_ms", 0.0),
        error=data.get("error"),
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
    """Return ``data[key]`` or raise — a server response missing it is malformed."""
    value = data.get(key)
    if value is None:
        raise NotebookOpsError(f"malformed server response: missing {key!r}")
    return value


def _cell_view(cell: CellState) -> CellView:
    """Local projection: serialize the cell to the wire dict, then map it."""
    return _cell_view_from_wire(cell.serialize())


def _status_row(cell: CellState) -> CellStatusRow:
    return _status_row_from_wire(cell.serialize())


def _dag_view(dag: NotebookDag | None) -> DagView:
    from strata.notebook.dag import producer_cell_label

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
        variable_producer={v: producer_cell_label(p) for v, p in dag.variable_producer.items()},
    )
