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
from typing import Any, Protocol, runtime_checkable


class NotebookOpsError(Exception):
    """An operation failed (unknown cell, DAG cycle, …).

    Distinct from an invocation error (bad path): the CLI maps this to a
    structured ``{"error": …}`` on stdout + exit 1, not the exit-2 usage path.
    """


@runtime_checkable
class NotebookOps(Protocol):
    """The notebook operation set. Read-only verbs first (P0); run / test /
    author / env verbs land in later phases on this same protocol."""

    def list_cells(self) -> dict[str, Any]:
        """All cells in order: ``{notebook_id, cells: [serialized cell, …]}``."""
        ...

    def get_cell(self, cell_id: str) -> dict[str, Any]:
        """One cell's full serialized view; raises :class:`NotebookOpsError`
        if no such cell."""
        ...

    def dag(self) -> dict[str, Any]:
        """The dependency graph: edges, topological order, leaves, roots."""
        ...

    def status(self) -> dict[str, Any]:
        """A compact per-cell status + staleness summary for the notebook."""
        ...


class LocalNotebookOps:
    """:class:`NotebookOps` over an in-process session — offline, no server."""

    def __init__(self, notebook_dir: Path) -> None:
        # Lazy heavy imports so ``--help`` / path errors stay cheap.
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession

        state = parse_notebook(notebook_dir)
        self._session = NotebookSession(state, notebook_dir)

    def list_cells(self) -> dict[str, Any]:
        session = self._session
        return {
            "notebook_id": session.notebook_state.id,
            "cells": session.serialize_cells(),
        }

    def get_cell(self, cell_id: str) -> dict[str, Any]:
        session = self._session
        cell = session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise NotebookOpsError(f"no cell with id {cell_id!r}")
        return session.serialize_cell(cell)

    def dag(self) -> dict[str, Any]:
        # Mirrors routes._format_dag, inlined to keep the server tree out of the
        # CLI import path. Same shape as ``GET /{id}/dag``.
        dag = self._session.dag
        if dag is None:
            return {
                "edges": [],
                "topological_order": [],
                "leaves": [],
                "roots": [],
                "variable_producer": {},
            }
        return {
            "edges": [
                {
                    "from_cell_id": edge.from_cell_id,
                    "to_cell_id": edge.to_cell_id,
                    "variable": edge.variable,
                }
                for edge in dag.edges
            ],
            "topological_order": dag.topological_order,
            "leaves": list(dag.leaves),
            "roots": list(dag.roots),
            "variable_producer": dag.variable_producer,
        }

    def status(self) -> dict[str, Any]:
        session = self._session
        cells = [
            {
                "id": cell["id"],
                "name": cell.get("name") or "",
                "language": cell.get("language"),
                "status": cell.get("status"),
                "staleness_reasons": cell.get("staleness_reasons", []),
            }
            for cell in session.serialize_cells()
        ]
        return {
            "notebook_id": session.notebook_state.id,
            "name": session.notebook_state.name,
            "cells": cells,
        }
