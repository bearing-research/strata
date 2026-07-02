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

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from strata.notebook.session import SessionManager


def _resolve_ops(session_manager: SessionManager, session_id: str):
    """Wrap the server's warm session for *session_id* in ``LocalNotebookOps``.

    Raises ``ValueError`` (surfaced to the agent as a tool error) when no such
    session is open — MCP operates on sessions the UI/CLI already opened, not
    arbitrary paths.
    """
    from strata.notebook.ops import LocalNotebookOps

    session = session_manager.get_session(session_id)
    if session is None:
        raise ValueError(
            f"no open notebook session {session_id!r}; call list_notebooks to see "
            "the sessions currently open on the server"
        )
    return LocalNotebookOps.from_session(session)


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

    return mcp.streamable_http_app()
