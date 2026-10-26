"""Adding a column to a store that already holds data, on SQLite.

The Postgres half lives in ``test_artifact_store_postgres.py``.
"""

from __future__ import annotations

import sqlite3

import pytest

from vestigo.store import (
    _BASELINE_SCHEMA_VERSION,
    _LATEST_SCHEMA_VERSION,
    _MIGRATIONS,
    ArtifactStore,
    StoreSchemaMismatch,
)


def _columns(db_path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(artifact_versions)")}
    finally:
        conn.close()


def _schema(db_path) -> set[tuple[str, str]]:
    """Every table and index with its DDL."""
    conn = sqlite3.connect(db_path)
    try:
        return set(conn.execute("SELECT name, sql FROM sqlite_master").fetchall())
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
        """Applying by ``version`` means a duplicate would silently skip one."""
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
        """The rows are why the store could not be recreated."""
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

    def test_a_store_a_newer_vestigo_migrated_is_refused(self, tmp_path):
        """An older vestigo on a migrated store would write rows the newer one does not expect."""
        db_path = ArtifactStore(tmp_path / "s").db_path
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, 0)",
            (_LATEST_SCHEMA_VERSION + 1,),
        )
        conn.commit()
        conn.close()

        with pytest.raises(RuntimeError, match="newer than this vestigo supports"):
            ArtifactStore(tmp_path / "s")


class TestTwoFirstBoots:
    """Two processes creating one SQLite store together both come out at the latest version."""

    def test_the_fresh_stamp_runs_under_the_write_lock(self, tmp_path):
        """Interleaved at the stamp: the first boot has read 'absent' and not yet
        inserted when the second boot finds the tables. Under the lock the second
        waits (its no-wait connection reports the lock instead); without it the
        second stamps every version first and the first boot's insert collides."""
        import threading
        import time

        from vestigo.sql_backend import SqliteDialect

        class NoWait(SqliteDialect):
            def connect(self):
                conn = sqlite3.connect(str(self.db_path), timeout=0.05)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.row_factory = sqlite3.Row
                return conn

        target = tmp_path / "artifacts"
        second: dict[str, object] = {}

        def boot_second():
            try:
                second["store"] = ArtifactStore(target, dialect=NoWait(target / "artifacts.sqlite"))
            except sqlite3.OperationalError as exc:
                second["error"] = exc

        original = ArtifactStore._stamp_schema_version

        def interleaved(self, conn, version):
            if version != _LATEST_SCHEMA_VERSION or "ran" in second:
                return original(self, conn, version)
            second["ran"] = True
            stamped = conn.execute(
                "SELECT 1 FROM schema_version WHERE version = ?", (version,)
            ).fetchone()
            if stamped is not None:
                return
            other = threading.Thread(target=boot_second)
            other.start()
            other.join(timeout=10)
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, time.time()),
            )

        ArtifactStore._stamp_schema_version = interleaved
        try:
            first = ArtifactStore(target)
        finally:
            ArtifactStore._stamp_schema_version = original

        assert "store" not in second, "the second boot got in between the read and the insert"
        assert "locked" in str(second["error"])
        assert _version(first.db_path) == _LATEST_SCHEMA_VERSION
        assert ArtifactStore(target).stats()["total_versions"] == 0
        conn = sqlite3.connect(first.db_path)
        try:
            assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
        finally:
            conn.close()


class TestAReadOnlyOpen:
    """For commands that only inspect: the store stays exactly as it was."""

    def test_an_older_database_is_refused_untouched(self, tmp_path):
        db_path = ArtifactStore(tmp_path / "old").db_path
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM schema_version WHERE version >= 9")
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (8, 0)")
        conn.commit()
        conn.close()

        before = _schema(db_path)

        with pytest.raises(StoreSchemaMismatch, match="version 8, older"):
            ArtifactStore(tmp_path / "old", read_only=True)

        assert _version(db_path) == 8
        assert _schema(db_path) == before

    def test_a_missing_store_is_refused_before_anything_is_created(self, tmp_path):
        target = tmp_path / "nope" / "artifacts"

        with pytest.raises(StoreSchemaMismatch, match="No artifact store"):
            ArtifactStore(target, read_only=True)

        assert not target.exists()
        assert not target.parent.exists()

    def test_a_database_from_before_versioning_is_refused(self, tmp_path):
        db_path = ArtifactStore(tmp_path / "old").db_path
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE schema_version")
        conn.commit()
        conn.close()

        with pytest.raises(StoreSchemaMismatch, match="older"):
            ArtifactStore(tmp_path / "old", read_only=True)

    def test_a_newer_database_is_refused_as_before(self, tmp_path):
        db_path = ArtifactStore(tmp_path / "s").db_path
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, 0)",
            (_LATEST_SCHEMA_VERSION + 1,),
        )
        conn.commit()
        conn.close()

        with pytest.raises(StoreSchemaMismatch, match="newer than this vestigo supports"):
            ArtifactStore(tmp_path / "s", read_only=True)

    def test_a_current_database_opens(self, tmp_path):
        ArtifactStore(tmp_path / "s")

        assert ArtifactStore(tmp_path / "s", read_only=True).stats()["total_versions"] == 0


class TestConstantsAndMigrationsAgree:
    def test_a_migrated_database_matches_a_fresh_one(self, tmp_path):
        """A column added to the schema constants without a migration breaks only old stores.

        Every other test starts from a fresh database, so nothing else would notice.
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


def _ready(store, artifact_id, executor, inputs, input_versions=None) -> None:
    from vestigo.provenance import TransformSpec

    spec = TransformSpec(executor=executor, params={"sql": "SELECT 1"}, inputs=inputs)
    version = store.create_artifact(
        artifact_id, f"hash-{artifact_id}", transform_spec=spec, input_versions=input_versions
    )
    store.finalize_artifact(artifact_id, version, "{}", 1, 1)


def test_column_exists_reports_truthfully(tmp_path):
    """The primitive every migration is built from."""
    from vestigo.sql_backend import SqliteDialect

    store = ArtifactStore(tmp_path / "s")
    dialect = SqliteDialect(store.db_path)
    conn = dialect.connect()
    try:
        assert dialect.column_exists(conn, "artifact_versions", "content_sha256")
        assert not dialect.column_exists(conn, "artifact_versions", "no_such_column")
    finally:
        conn.close()
