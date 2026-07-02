"""MCP server — expose the live notebook session to an external coding agent.

Mounts at ``/mcp`` (streamable HTTP) inside the FastAPI app when
``mcp_enabled`` is set (personal mode only). This is P4 of the CLI-hardening
phase: the same :class:`~strata.notebook.ops.NotebookOps` contract the ``strata``
CLI drives, wrapped as MCP tools so a coding agent (Claude Code, etc.) can
operate a warm session — its populated artifact cache and current cell state —
rather than an offline copy.

Gated on the ``[mcp]`` extra: :func:`build_mcp_app` returns ``None`` when the
``mcp`` package is not installed, so the server runs fine without it
(core-deps-only rule). The tool *logic* lives in module-level ``_*`` functions
that take a ``SessionManager`` so it is unit-testable without an MCP client or a
live socket; :func:`build_mcp_app` registers thin wrappers whose docstrings are
the agent-facing tool descriptions.

Phase 1 = read tools (``list_notebooks`` / ``get_notebook`` / ``get_cell`` /
``dag`` / ``status``). Run, authoring, and dependency tools land in later phases
on the same surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strata.notebook.ops import LocalNotebookOps

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from strata.notebook.session import SessionManager


def _resolve_ops(session_manager: SessionManager, session_id: str) -> LocalNotebookOps:
    """Wrap the server's warm session for *session_id* in ``LocalNotebookOps``.

    Raises ``ValueError`` (surfaced to the agent as a tool error) when no such
    session is open — MCP operates on sessions the UI/CLI already opened, not
    arbitrary paths.
    """
    return LocalNotebookOps.from_session(_live_session(session_manager, session_id))


def _list_notebooks(session_manager: SessionManager) -> list[dict[str, Any]]:
    """Return one ``{session_id, name, path}`` entry per open session."""
    notebooks: list[dict[str, Any]] = []
    for session_id in session_manager.list_sessions():
        session = session_manager.get_session(session_id)
        if session is None:
            continue
        notebooks.append(
            {
                "session_id": session_id,
                "name": session.notebook_state.name,
                "path": str(session.path),
            }
        )
    return notebooks


def _get_notebook(session_manager: SessionManager, session_id: str) -> dict[str, Any]:
    """Return every cell's curated view for an open session, in order."""
    ops = _resolve_ops(session_manager, session_id)
    return {"cells": [cell.model_dump(mode="json") for cell in ops.list_cells()]}


def _get_cell(session_manager: SessionManager, session_id: str, cell_id: str) -> dict[str, Any]:
    """Return one cell's curated view (source, status, outputs, …)."""
    return _resolve_ops(session_manager, session_id).get_cell(cell_id).model_dump(mode="json")


def _dag(session_manager: SessionManager, session_id: str) -> dict[str, Any]:
    """Return the dependency graph (edges, topological order, roots, leaves)."""
    return _resolve_ops(session_manager, session_id).dag().model_dump(mode="json")


def _status(session_manager: SessionManager, session_id: str) -> dict[str, Any]:
    """Return a compact per-cell status + staleness summary."""
    return _resolve_ops(session_manager, session_id).status().model_dump(mode="json")


async def _run_cell(
    session_manager: SessionManager,
    session_id: str,
    cell_id: str,
    mode: str = "normal",
) -> dict[str, Any]:
    """Execute a cell in the warm session, broadcasting live frames to spectators.

    Unlike the read tools this does not go through ``LocalNotebookOps.run_cell``
    (which runs the cell silently): it calls the same ``execute_cell_and_broadcast``
    path the WS/REST drives use, so a browser or TUI attached to the session sees
    the agent's run as ``cell_status`` → result → staleness frames in real time.
    The result is mapped to the agent-facing ``RunResult`` view.
    """
    from strata.notebook.ops import NotebookOpsError, _run_result_from_wire
    from strata.notebook.ws import _ensure_execution_state, execute_cell_and_broadcast

    if mode not in ("normal", "rerun", "force"):
        raise ValueError(f"unknown run mode {mode!r} (normal|rerun|force)")

    session = session_manager.get_session(session_id)
    if session is None:
        raise ValueError(f"no open notebook session {session_id!r}; call list_notebooks first")
    if session.notebook_state.get_cell(cell_id) is None:
        raise NotebookOpsError(f"no cell with id {cell_id!r}")

    block_reason = session.environment_execution_block_message()
    if block_reason:
        raise ValueError(block_reason)

    execution_state = _ensure_execution_state(session_id)
    result = await execute_cell_and_broadcast(
        session,
        cell_id,
        execution_state,
        session_id,
        mode=mode,  # type: ignore[arg-type]
    )
    if result is None:
        raise NotebookOpsError(f"cell {cell_id!r} could not be executed")
    return _run_result_from_wire(result.to_dict()).model_dump(mode="json")


async def _run_tests(
    session_manager: SessionManager, session_id: str, cell_id: str
) -> dict[str, Any]:
    """Run a cell's unit tests in the warm session and return per-test outcomes."""
    ops = _resolve_ops(session_manager, session_id)
    result = await ops.run_tests(cell_id)
    return result.model_dump(mode="json")


async def _broadcast_notebook(session_id: str, session: Any) -> None:
    """Push a full notebook_state to the session's WS spectators after a mutation.

    Authoring over MCP is a file mutation the offline ``LocalNotebookOps`` verbs
    do not broadcast; sending the state sync (as the REST CRUD routes do) is what
    makes an agent's edits appear live in an attached browser / TUI.
    """
    from strata.notebook.ws import broadcast_notebook_sync

    await broadcast_notebook_sync(session_id, session)


def _live_session(session_manager: SessionManager, session_id: str):
    """Return the server's warm session for *session_id*, or raise ``ValueError``."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise ValueError(f"no open notebook session {session_id!r}; call list_notebooks first")
    return session


async def _sync_and_broadcast(session_id: str, session: Any) -> None:
    """Reload the live session from disk after a file mutation, then broadcast.

    The ``LocalNotebookOps`` authoring verbs write ``cells/*.py`` + ``notebook.toml``
    and reload a *detached* copy — they never mutate the server's live session
    (that is fine for the offline CLI). So we ``reload()`` the live session in
    place (the same call the REST CRUD routes make: re-parse, rebuild the DAG,
    restore execution history, recompute staleness) and broadcast the result, so
    the server's warm state and any attached viewer reflect the edit.
    """
    session.reload()
    await _broadcast_notebook(session_id, session)


async def _add_cell(
    session_manager: SessionManager,
    session_id: str,
    source: str,
    after: str | None = None,
    language: str = "python",
) -> dict[str, Any]:
    """Add a new cell (backend-minted id), then sync + broadcast the new state."""
    session = _live_session(session_manager, session_id)
    view = LocalNotebookOps.from_session(session).add_cell(source, after=after, language=language)
    await _sync_and_broadcast(session_id, session)
    return view.model_dump(mode="json")


async def _edit_cell(
    session_manager: SessionManager, session_id: str, cell_id: str, source: str
) -> dict[str, Any]:
    """Replace a cell's source, then sync + broadcast the new state."""
    session = _live_session(session_manager, session_id)
    view = LocalNotebookOps.from_session(session).edit_cell(cell_id, source)
    await _sync_and_broadcast(session_id, session)
    return view.model_dump(mode="json")


async def _remove_cell(
    session_manager: SessionManager, session_id: str, cell_id: str
) -> dict[str, Any]:
    """Delete a cell (and its source / test files), then sync + broadcast."""
    session = _live_session(session_manager, session_id)
    LocalNotebookOps.from_session(session).remove_cell(cell_id)
    await _sync_and_broadcast(session_id, session)
    return {"removed": cell_id}


async def _move_cell(
    session_manager: SessionManager, session_id: str, cell_id: str, index: int
) -> dict[str, Any]:
    """Move a cell to ``index`` in notebook order, then sync + broadcast."""
    session = _live_session(session_manager, session_id)
    cells = LocalNotebookOps.from_session(session).move_cell(cell_id, index)
    await _sync_and_broadcast(session_id, session)
    return {"cells": [cell.model_dump(mode="json") for cell in cells]}


def build_mcp_app(session_manager: SessionManager) -> Starlette | None:
    """Build the streamable-HTTP MCP ASGI app, or ``None`` if ``[mcp]`` is absent.

    The returned Starlette app is meant to be mounted at ``/mcp``; its own
    lifespan (which starts the MCP session manager) must be entered by the host
    app's lifespan — see ``server.py``.

    Parameters
    ----------
    session_manager : SessionManager
        The server's live session registry; tools resolve warm sessions from it.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError:
        return None

    # streamable_http_path="/" so mounting the app at "/mcp" yields the endpoint
    # at exactly "/mcp" (the default "/mcp" would nest it at "/mcp/mcp").
    mcp = FastMCP("strata-notebook", streamable_http_path="/")

    @mcp.tool()
    def list_notebooks() -> list[dict[str, Any]]:
        """List the notebook sessions currently open on the Strata server.

        Returns one entry per session with its ``session_id`` (use it as the
        ``session_id`` argument to the other tools), human-readable ``name``,
        and on-disk ``path``. Sessions are opened by the notebook UI or the
        ``strata`` CLI; this tool does not open them.
        """
        return _list_notebooks(session_manager)

    @mcp.tool()
    def get_notebook(session_id: str) -> dict[str, Any]:
        """Return every cell of an open notebook session, in notebook order.

        Each cell view includes its id, source, language, status, defines /
        references, and rendered outputs.
        """
        return _get_notebook(session_manager, session_id)

    @mcp.tool()
    def get_cell(session_id: str, cell_id: str) -> dict[str, Any]:
        """Return one cell's full curated view: source, status, and outputs.

        Errors if the notebook session or the cell id does not exist.
        """
        return _get_cell(session_manager, session_id, cell_id)

    @mcp.tool()
    def dag(session_id: str) -> dict[str, Any]:
        """Return the notebook's dependency graph.

        Includes variable-level ``edges``, a ``topological_order``, and the
        ``roots`` / ``leaves`` — how cell outputs feed downstream cells.
        """
        return _dag(session_manager, session_id)

    @mcp.tool()
    def status(session_id: str) -> dict[str, Any]:
        """Return a compact per-cell status + staleness summary for a session.

        Use it to see which cells are ready, stale (and why), or idle before
        deciding what to run.
        """
        return _status(session_manager, session_id)

    @mcp.tool()
    async def run_cell(session_id: str, cell_id: str, mode: str = "normal") -> dict[str, Any]:
        """Execute a cell in a warm session and return the run outcome.

        ``mode`` is one of: ``normal`` (use the cache, re-run stale upstreams
        first), ``rerun`` (bypass this cell's cache, still refresh upstreams), or
        ``force`` ("run this only" — run against whatever upstream artifacts
        already exist). The run is broadcast live, so a browser or terminal
        viewer attached to the session watches it happen. Returns status
        (ok / error), cache hit, duration, and captured stdout / stderr; call
        get_cell afterwards for the rendered outputs.
        """
        return await _run_cell(session_manager, session_id, cell_id, mode)

    @mcp.tool()
    async def run_tests(session_id: str, cell_id: str) -> dict[str, Any]:
        """Run a Python cell's unit tests (``cells/{cell_id}.test.py``).

        Returns pass / fail / error / skip counts and per-test cases. Errors if
        the cell has no test file.
        """
        return await _run_tests(session_manager, session_id, cell_id)

    @mcp.tool()
    async def add_cell(
        session_id: str,
        source: str,
        after: str | None = None,
        language: str = "python",
    ) -> dict[str, Any]:
        """Add a new cell and return it (the server mints the cell id).

        ``after`` inserts the cell after that cell id (omit to append at the
        end). ``language`` is one of python, markdown, sql, r, prompt. The new
        cell appears live in any attached viewer.
        """
        return await _add_cell(session_manager, session_id, source, after, language)

    @mcp.tool()
    async def edit_cell(session_id: str, cell_id: str, source: str) -> dict[str, Any]:
        """Replace a cell's source and return the updated cell.

        Downstream cells that consumed the old output become stale; use status /
        run_cell to re-materialize them.
        """
        return await _edit_cell(session_manager, session_id, cell_id, source)

    @mcp.tool()
    async def remove_cell(session_id: str, cell_id: str) -> dict[str, Any]:
        """Delete a cell and its source / test files. Returns the removed id."""
        return await _remove_cell(session_manager, session_id, cell_id)

    @mcp.tool()
    async def move_cell(session_id: str, cell_id: str, index: int) -> dict[str, Any]:
        """Move a cell to a new 0-based position and return the new cell order."""
        return await _move_cell(session_manager, session_id, cell_id, index)

    return mcp.streamable_http_app()
