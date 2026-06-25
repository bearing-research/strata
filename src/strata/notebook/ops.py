"""NotebookOps — the operation core an agent drives a notebook through.

A single operation set, exposed today by the ``strata`` CLI and (later) the MCP
server, so an agent has a full-feature tool for a notebook without re-deriving
the verbs in each surface. Two backends implement the protocol:

- :class:`LocalNotebookOps` (here) — an in-process ``NotebookSession``, offline,
  the same path ``strata run`` takes. No server required.
- ``RemoteNotebookOps`` (a later phase) — an httpx REST client against a running
  ``strata-notebook``, so the same commands drive a session a human can watch in
  the TUI / web UI.

Return shapes deliberately match the server's REST API (``GET /{id}/cells``,
``GET /{id}/dag``) so the CLI, the remote backend, and MCP all agree on one JSON
contract. See ``docs/internal/design-cli-hardening.md``.

This module is import-light: it pulls ``parse_notebook`` / ``NotebookSession``
lazily and never imports the FastAPI server tree, so ``strata cell …`` stays a
fast CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

# The full per-cell wire view (``CellState.serialize()`` — source, status,
# outputs, annotations, staleness, …). It has ~50 dynamic keys whose value types
# vary per content type, so it is genuinely a string-keyed mapping rather than a
# fixed schema; ``object`` (not ``Any``) keeps callers honest about narrowing.
SerializedCell = dict[str, object]


class CellList(TypedDict):
    """Return of :meth:`NotebookOps.list_cells`."""

    notebook_id: str
    cells: list[SerializedCell]


class DagEdgeDict(TypedDict):
    """One variable-level dependency edge in a :class:`DagResult`."""

    from_cell_id: str
    to_cell_id: str
    variable: str


class DagResult(TypedDict):
    """Return of :meth:`NotebookOps.dag` (mirrors ``GET /{id}/dag``)."""

    edges: list[DagEdgeDict]
    topological_order: list[str]
    leaves: list[str]
    roots: list[str]
    variable_producer: dict[str, str]


class CellStatusRow(TypedDict):
    """One cell's row in a :class:`NotebookStatus` summary."""

    id: str
    name: str
    language: str | None
    status: str | None
    staleness_reasons: list[str]


class NotebookStatus(TypedDict):
    """Return of :meth:`NotebookOps.status`."""

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

    def list_cells(self) -> CellList:
        """Return every cell, in notebook order.

        Returns
        -------
        CellList
            ``{notebook_id, cells}`` where each cell is the full serialized wire
            view — the same shape as ``GET /{id}/cells``.
        """
        ...

    def get_cell(self, cell_id: str) -> SerializedCell:
        """Return one cell's full serialized view.

        Parameters
        ----------
        cell_id : str
            Identifier of the cell to fetch.

        Returns
        -------
        SerializedCell
            The cell's wire view (source, status, outputs, annotations, …) — one
            element of ``GET /{id}/cells``.

        Raises
        ------
        NotebookOpsError
            If no cell with ``cell_id`` exists in the notebook.
        """
        ...

    def dag(self) -> DagResult:
        """Return the dependency graph.

        Returns
        -------
        DagResult
            Variable-level ``edges``, a ``topological_order``, the ``leaves`` and
            ``roots``, and the ``variable_producer`` map — mirrors ``GET /{id}/dag``.
        """
        ...

    def status(self) -> NotebookStatus:
        """Return a compact per-cell status + staleness summary.

        Returns
        -------
        NotebookStatus
            ``{notebook_id, name, cells}`` with one :class:`CellStatusRow` per
            cell (id, name, language, status, staleness reasons).
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

    def list_cells(self) -> CellList:
        """List every cell in order (see :meth:`NotebookOps.list_cells`)."""
        session = self._session
        cells: list[SerializedCell] = list(session.serialize_cells())
        return {"notebook_id": session.notebook_state.id, "cells": cells}

    def get_cell(self, cell_id: str) -> SerializedCell:
        """Serialize one cell (see :meth:`NotebookOps.get_cell`)."""
        session = self._session
        cell = session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        serialized: SerializedCell = session.serialize_cell(cell)
        return serialized

    def dag(self) -> DagResult:
        """Build the DAG view (see :meth:`NotebookOps.dag`).

        Inlines ``routes._format_dag`` to keep the server tree out of the CLI
        import path; the shape is identical to ``GET /{id}/dag``.
        """
        dag = self._session.dag
        if dag is None:
            return {
                "edges": [],
                "topological_order": [],
                "leaves": [],
                "roots": [],
                "variable_producer": {},
            }
        edges: list[DagEdgeDict] = [
            {
                "from_cell_id": edge.from_cell_id,
                "to_cell_id": edge.to_cell_id,
                "variable": edge.variable,
            }
            for edge in dag.edges
        ]
        return {
            "edges": edges,
            "topological_order": dag.topological_order,
            "leaves": list(dag.leaves),
            "roots": list(dag.roots),
            "variable_producer": dag.variable_producer,
        }

    def status(self) -> NotebookStatus:
        """Summarize per-cell status (see :meth:`NotebookOps.status`)."""
        session = self._session
        rows: list[CellStatusRow] = [
            {
                "id": str(cell["id"]),
                "name": str(cell.get("name") or ""),
                "language": _opt_str(cell.get("language")),
                "status": _opt_str(cell.get("status")),
                "staleness_reasons": [str(r) for r in (cell.get("staleness_reasons") or [])],
            }
            for cell in session.serialize_cells()
        ]
        return {
            "notebook_id": session.notebook_state.id,
            "name": session.notebook_state.name,
            "cells": rows,
        }


def _opt_str(value: object) -> str | None:
    """Coerce a serialized field to ``str`` or ``None`` (drops the upstream ``Any``)."""
    return None if value is None else str(value)
