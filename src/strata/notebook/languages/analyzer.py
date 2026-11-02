"""Strata's language adapters that cellgraph's analyzers wrap.

The analyzers the notebook runs are cellgraph's, registered in
``strata.notebook.vocabulary``; ``cellgraph.register_language_analyzer`` is the
extension point.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata.notebook.models import CellState
    from strata.notebook.session import NotebookSession


@dataclass(frozen=True)
class AnalyzedCell:
    """An adapter's result; only Python and R set ``mutation_defines``."""

    defines: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    mutation_defines: list[str] = field(default_factory=list)
    # Free names that shadow a builtin (``input``, ``type``). Hidden from
    # ``references`` but resolved by the DAG so the edge still wires.
    builtin_references: list[str] = field(default_factory=list)


_SQL_EXTRA_MISSING = (
    "SQL cells need the sql extra, which this server does not have: install "
    "strata-notebook[sql] plus a driver extra for each database (for example sql-duckdb, "
    "sql-sqlite)."
)


def require_sql_extra() -> None:
    """Raise ``ValueError`` naming the sql extra when this server does not have it.

    Checked before a SQL cell or connection is written: once one is in notebook.toml,
    every open of the notebook fails.
    """
    if importlib.util.find_spec("sqlglot") is None:
        raise ValueError(_SQL_EXTRA_MISSING)


class _SqlAnalyzer:
    """Adapter over ``strata.notebook.sql.analyzer.analyze_sql_cell``.

    The DAG needs only the output name and ``:name`` placeholders, so no
    dialect is passed and sqlglot never parses the body: a query sqlglot
    cannot read must not keep the notebook from opening. The placeholder scan
    does take the connection's dialect, which decides where a string ends.
    """

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        try:
            from strata.notebook.sql.analyzer import _extract_placeholders, analyze_sql_cell
        except ModuleNotFoundError as exc:
            if exc.name != "sqlglot":
                raise
            raise ValueError(f"Cell {cell.id} is a SQL cell, and {_SQL_EXTRA_MISSING}") from exc

        result = analyze_sql_cell(cell.source)
        return AnalyzedCell(
            defines=list(result.defines),
            references=_extract_placeholders(
                result.sql_body, _connection_dialect(result.connection, session)
            ),
        )


def _connection_dialect(connection: str | None, session: NotebookSession) -> str | None:
    """The sqlglot dialect of the notebook connection named *connection*, if known."""
    from strata.notebook.sql.registry import get_adapter

    if connection is None:
        return None
    spec = next((c for c in session.notebook_state.connections if c.name == connection), None)
    if spec is None:
        return None
    try:
        return get_adapter(spec.driver).sqlglot_dialect
    except KeyError:
        return None
