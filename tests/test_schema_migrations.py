"""Adding a column to a store that already holds data.

Before this existed, the migrations were SQLite-only by construction — PRAGMA
and sqlite_master, guarded by ``supports_legacy_migration`` — and the Postgres
path returned as soon as the schema existed. So there was no way at all to
evolve a Postgres store that held anything. Fine while Postgres was new; not
fine once one holds something worth keeping.
"""

from __future__ import annotations

import sqlite3

import pytest

from strata.artifact_store import (
    _BASELINE_SCHEMA_VERSION,
    _LATEST_SCHEMA_VERSION,
    _MIGRATIONS,
    ArtifactStore,
)


def _columns(db_path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(artifact_versions)")}
    finally:
        conn.close()


def _version(db_path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    finally:
        conn.close()


class TestMigrationList:
    def test_versions_are_unique_and_ordered(self):
        """Applying by ``version`` means duplicates would silently skip one."""
        versions = [m.version for m in _MIGRATIONS]
        assert versions == sorted(versions)
        assert len(versions) == len(set(versions))

    def test_every_version_is_above_the_baseline(self):
        """A migration at or below the baseline never runs on an old database."""
        assert all(m.version > _BASELINE_SCHEMA_VERSION for m in _MIGRATIONS)


class TestFreshDatabase:
    def test_is_stamped_at_the_latest_version(self, tmp_path):
        store = ArtifactStore(tmp_path / "fresh")

        assert _version(store.db_path) == _LATEST_SCHEMA_VERSION

    def test_has_every_migrated_column(self, tmp_path):
        store = ArtifactStore(tmp_path / "fresh")

        assert "content_sha256" in _columns(store.db_path)


class TestExistingDatabase:
    def test_a_store_predating_the_version_table_is_migrated(self, tmp_path):
        """The case the mechanism exists for."""
        store = ArtifactStore(tmp_path / "old")
        db_path = store.db_path

        # Rewind it to what a pre-migration database looked like.
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE schema_version")
        conn.execute("ALTER TABLE artifact_versions DROP COLUMN content_sha256")
        conn.commit()
        conn.close()
        assert "content_sha256" not in _columns(db_path)

        ArtifactStore(tmp_path / "old")  # reopening applies what is pending

        assert "content_sha256" in _columns(db_path)
        assert _version(db_path) == _LATEST_SCHEMA_VERSION

    def test_reopening_a_current_database_applies_nothing(self, tmp_path):
        """Idempotent, and the common path: every construction runs this."""
        db_path = ArtifactStore(tmp_path / "s").db_path
        before = _version(db_path)

        for _ in range(3):
            ArtifactStore(tmp_path / "s")

        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        finally:
            conn.close()
        assert _version(db_path) == before
        assert rows == 1, "a no-op reopen must not stamp another row"

    def test_data_survives_a_migration(self, tmp_path):
        """The whole point: the rows are why the store could not be recreated."""
        import hashlib

        store = ArtifactStore(tmp_path / "old")
        version = store.create_artifact("keeper", hashlib.sha256(b"x").hexdigest())
        with store.open_blob_writer("keeper", version) as writer:
            writer.write(b"x")
        store.finalize_artifact("keeper", version, schema_json="", row_count=0, byte_size=1)

        conn = sqlite3.connect(store.db_path)
        conn.execute("DROP TABLE schema_version")
        conn.execute("ALTER TABLE artifact_versions DROP COLUMN content_sha256")
        conn.commit()
        conn.close()

        reopened = ArtifactStore(tmp_path / "old")

        artifact = reopened.get_artifact("keeper", version)
        assert artifact is not None
        assert artifact.state == "ready"


class TestConstantsAndMigrationsAgree:
    def test_a_migrated_database_matches_a_fresh_one(self, tmp_path):
        """The drift this mechanism is most likely to develop.

        The schema constants describe the latest shape and the migrations
        describe the path to it. Someone adding a column to the constants and
        forgetting the migration gets a fresh database that works and an
        existing one that does not — and nothing else in the suite would
        notice, because every test starts from a fresh database.
        """
        fresh = ArtifactStore(tmp_path / "fresh")

        aged = ArtifactStore(tmp_path / "aged")
        conn = sqlite3.connect(aged.db_path)
        conn.execute("DROP TABLE schema_version")
        conn.execute("ALTER TABLE artifact_versions DROP COLUMN content_sha256")
        conn.commit()
        conn.close()
        migrated = ArtifactStore(tmp_path / "aged")

        assert _columns(migrated.db_path) == _columns(fresh.db_path)


@pytest.mark.parametrize("dialect_name", ["sqlite"])
def test_column_exists_reports_truthfully(tmp_path, dialect_name):
    """The primitive every migration is built from."""
    from strata.sql_backend import SqliteDialect

    store = ArtifactStore(tmp_path / "s")
    dialect = SqliteDialect(store.db_path)
    conn = dialect.connect()
    try:
        assert dialect.column_exists(conn, "artifact_versions", "content_sha256")
        assert not dialect.column_exists(conn, "artifact_versions", "no_such_column")
    finally:
        conn.close()
