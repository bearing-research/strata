"""Record which node serves each stream, so another node can point there instead of 404ing.

Stream state (a live task, slot and in-memory plan) cannot move between nodes, so
only the ``stream_id -> node url`` pointer is shared, via the artifact store's
database. Opt-in: nothing is read or written without ``node_advertised_url``.
"""

from __future__ import annotations

import time
from pathlib import Path

from strata.sql_backend import SqlDialect, SqliteDialect, StoreConnection

_OWNERSHIP_SCHEMA_SQL = """
-- Stream ownership: which node can serve a given stream, and until when.
-- Rows are short-lived; they expire with the stream's TTL.
CREATE TABLE IF NOT EXISTS stream_owners (
    stream_id TEXT PRIMARY KEY,
    node_url TEXT NOT NULL,
    registered_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stream_owners_expires ON stream_owners(expires_at);
"""


class StreamOwnershipStore:
    """Records and resolves the node serving each stream, in the artifact store's database."""

    def __init__(self, db_path: Path, dialect: SqlDialect | None = None):
        self.db_path = db_path
        self._dialect: SqlDialect = dialect if dialect is not None else SqliteDialect(db_path)
        self._init_schema()

    def _get_connection(self) -> StoreConnection:
        return self._dialect.connect()

    def _init_schema(self) -> None:
        conn = self._get_connection()
        try:
            if not self._dialect.supports_legacy_migration:
                # CREATE TABLE IF NOT EXISTS races in Postgres, and every node runs this at startup.
                if not self._dialect.schema_exists(conn, "stream_owners"):
                    self._dialect.begin_write(conn, "__stream_owner_schema__")
                    conn.executescript(self._dialect.adapt_ddl(_OWNERSHIP_SCHEMA_SQL))
                    conn.commit()
                return

            conn.executescript(self._dialect.adapt_ddl(_OWNERSHIP_SCHEMA_SQL))
            conn.commit()
        finally:
            conn.close()

    def claim(self, stream_id: str, node_url: str, ttl_seconds: float) -> None:
        """Record this node as the one serving ``stream_id``; the newest claim wins.

        ``stream_id`` is often the artifact id, which a refresh may re-stream elsewhere.
        """
        now = time.time()
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO stream_owners (stream_id, node_url, registered_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (stream_id) DO UPDATE SET
                    node_url = excluded.node_url,
                    registered_at = excluded.registered_at,
                    expires_at = excluded.expires_at
                """,
                (stream_id, node_url, now, now + ttl_seconds),
            )
            conn.commit()
        finally:
            conn.close()

    def resolve(self, stream_id: str, exclude_node_url: str) -> str | None:
        """Return the URL of another node serving ``stream_id``.

        ``None`` for no claim, an expired claim, or one held by ``exclude_node_url``;
        the caller should then 404.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT node_url, expires_at FROM stream_owners WHERE stream_id = ?",
                (stream_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None or row["expires_at"] <= time.time():
            return None
        if row["node_url"] == exclude_node_url:
            return None
        return row["node_url"]

    def release(self, stream_id: str) -> None:
        """Drop a claim once the stream is finished or expired locally."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM stream_owners WHERE stream_id = ?", (stream_id,))
            conn.commit()
        finally:
            conn.close()

    def sweep_expired(self) -> int:
        """Delete expired claims, including a dead node's, and return how many."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM stream_owners WHERE expires_at <= ?", (time.time(),))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()


_ownership_store: StreamOwnershipStore | None = None


def get_stream_ownership_store(
    db_path: Path | None = None,
    dialect: SqlDialect | None = None,
) -> StreamOwnershipStore | None:
    """Get the ownership store singleton, or None when not configured."""
    global _ownership_store
    if _ownership_store is None and db_path is not None:
        _ownership_store = StreamOwnershipStore(db_path, dialect=dialect)
    return _ownership_store


def reset_stream_ownership_store() -> None:
    """Reset the singleton (for testing)."""
    global _ownership_store
    _ownership_store = None
