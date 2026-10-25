"""Adding a column to a store that already holds data, on SQLite and Postgres."""

from __future__ import annotations

import sqlite3

from vestigo.store import (
    _LATEST_SCHEMA_VERSION,
    ArtifactStore,
)

from strata.notebook.worker_store import NotebookWorkerStore


def _version(db_path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    finally:
        conn.close()


class TestTheWorkerRegistryMigration:
    def test_a_database_from_before_it_gains_the_tables(self, tmp_path):
        db_path = ArtifactStore(tmp_path / "old").db_path
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM schema_version WHERE version >= 9")
        conn.commit()
        conn.close()

        # Migration 9 is a tombstone: Strata's worker store creates the tables itself.
        reopened = ArtifactStore(tmp_path / "old")
        workers = NotebookWorkerStore(reopened.db_path, dialect=reopened.dialect)
        entry = {"name": "box", "backend": "executor", "enabled": True}
        workers.update(lambda current: [entry])

        assert workers.entries() == [entry]
        assert _version(db_path) == _LATEST_SCHEMA_VERSION
