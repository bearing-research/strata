"""The server-managed notebook worker registry, in the artifact database.

Strata's own table, beside the ledger's: it shares the store's dialect and so
its connection pool (as ``auth.api_keys`` and ``streaming.ownership`` do) but
the ledger package knows nothing about it. Until the registry is first set, the
configured ``[tool.strata.transforms]`` table applies; after that, editing the
configured table has no effect, so an emptied registry stays empty.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vestigo.sql_backend import SqlDialect, SqliteDialect, StoreConnection, create_tables

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notebook_workers (
    name TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    spec TEXT NOT NULL,  -- JSON: the WorkerSpec plus "enabled"
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS notebook_worker_registry (
    id INTEGER PRIMARY KEY,  -- always 1
    updated_at REAL NOT NULL
);
"""

TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("notebook_workers", ("name",)),
    ("notebook_worker_registry", ("id",)),
)
"""The registry's tables with their primary keys, for ``strata migrate``."""


class NotebookWorkerStore:
    """Reads and atomic rewrites of the worker registry."""

    def __init__(self, db_path: Path, dialect: SqlDialect | None = None):
        self.db_path = db_path
        self._dialect: SqlDialect = dialect if dialect is not None else SqliteDialect(db_path)
        create_tables(self._dialect, "notebook_workers", _SCHEMA_SQL)

    def _get_connection(self) -> StoreConnection:
        return self._dialect.connect()

    def entries(self) -> list[dict[str, Any]] | None:
        """The registry in order, or ``None`` if never set.

        ``None`` means the configured table applies; ``[]`` means every worker was removed.
        """
        conn = self._get_connection()
        try:
            return self._entries(conn)
        finally:
            conn.close()

    @staticmethod
    def _entries(conn: StoreConnection) -> list[dict[str, Any]] | None:
        if conn.execute("SELECT 1 FROM notebook_worker_registry WHERE id = 1").fetchone() is None:
            return None
        rows = conn.execute("SELECT spec FROM notebook_workers ORDER BY ordinal").fetchall()
        return [json.loads(row["spec"]) for row in rows]

    def update(
        self,
        change: Callable[[list[dict[str, Any]] | None], list[dict[str, Any]] | None],
    ) -> list[dict[str, Any]] | None:
        """Rewrite the registry as ``change(current)`` returns it, atomically.

        ``change`` gets the current entries (``None`` if never set) and returns the new
        list in order, or ``None`` to leave it alone; anything it raises rolls back. Held
        under a write lock so two nodes editing at once cannot lose either change. Rows
        are keyed by ``name``, and an unchanged one keeps its ``updated_at``.
        """
        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, "__notebook_workers__")
            stored = {
                row["name"]: (row["ordinal"], row["spec"])
                for row in conn.execute("SELECT name, ordinal, spec FROM notebook_workers")
            }
            entries = change(self._entries(conn))
            if entries is None:
                conn.rollback()
                return None
            now = time.time()
            for name in stored.keys() - {entry["name"] for entry in entries}:
                conn.execute("DELETE FROM notebook_workers WHERE name = ?", (name,))
            for ordinal, entry in enumerate(entries):
                spec = json.dumps(entry, sort_keys=True)
                previous = stored.get(entry["name"])
                if previous is None:
                    conn.execute(
                        "INSERT INTO notebook_workers "
                        "(name, ordinal, spec, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (entry["name"], ordinal, spec, now, now),
                    )
                elif previous != (ordinal, spec):
                    conn.execute(
                        "UPDATE notebook_workers SET ordinal = ?, spec = ?, updated_at = ? "
                        "WHERE name = ?",
                        (ordinal, spec, now, entry["name"]),
                    )
            conn.execute(
                "INSERT INTO notebook_worker_registry (id, updated_at) VALUES (1, ?) "
                "ON CONFLICT (id) DO UPDATE SET updated_at = excluded.updated_at",
                (now,),
            )
            conn.commit()
            return entries
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


_worker_store: NotebookWorkerStore | None = None


def open_notebook_worker_store(
    db_path: Path, dialect: SqlDialect | None = None
) -> NotebookWorkerStore:
    """Open the process-wide registry store on the artifact database.

    The server does this once at startup, so no request ever creates the schema.
    """
    global _worker_store
    if _worker_store is None:
        _worker_store = NotebookWorkerStore(db_path, dialect=dialect)
    return _worker_store


def get_notebook_worker_store() -> NotebookWorkerStore | None:
    """The store startup opened, or ``None`` on a server without an artifact store."""
    return _worker_store


def reset_notebook_worker_store() -> None:
    """Forget the singleton (for tests)."""
    global _worker_store
    _worker_store = None
