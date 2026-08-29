"""The artifact store running on Postgres instead of SQLite.

These exercise the dialect seam end to end: the same ``ArtifactStore`` methods
personal mode uses, against a real server, so the port is proven by the store's
own behavior rather than by asserting on rendered SQL.

Requirements:
    - Docker must be running
    - The ``postgres`` extra (psycopg)
"""

from __future__ import annotations

import threading
import time

import docker
import pytest
from testcontainers.postgres import PostgresContainer

from strata.artifact_store import ArtifactStore, TransformSpec
from strata.sql_backend import PostgresDialect, advisory_lock_id


def _docker_daemon_reachable() -> bool:
    """Skip when the daemon is unreachable. ``docker`` itself is a transitive
    dev dep via testcontainers, but the daemon may not be running on a
    contributor's laptop. CI always has Docker; this only triggers locally.
    """
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


if not _docker_daemon_reachable():
    pytest.skip("Docker daemon is not running", allow_module_level=True)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def postgres_dsn():
    """A live Postgres, shared across this module (container startup is slow)."""
    with PostgresContainer("postgres:16-alpine") as container:
        yield container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")


@pytest.fixture
def store(postgres_dsn, tmp_path):
    """A store whose metadata is on Postgres and whose blobs are local.

    Each test gets a clean schema: the module-scoped container is reused for
    speed, so state has to be dropped between tests rather than recreated.
    """
    dialect = PostgresDialect(postgres_dsn)
    conn = dialect.connect()
    try:
        conn.executescript(
            "DROP TABLE IF EXISTS artifact_versions, artifact_names, artifact_aliases, "
            "artifact_tags, registry_audit, registry_pending CASCADE;"
        )
        conn.commit()
    finally:
        conn.close()

    return ArtifactStore(tmp_path / "artifacts", dialect=dialect)


def _spec() -> TransformSpec:
    return TransformSpec(executor="duckdb_sql_v1", params={"sql": "SELECT 1"}, inputs=[])


class TestSchemaInitialization:
    def test_schema_is_created_without_the_sqlite_migration_path(self, store):
        # supports_legacy_migration is False for Postgres, so _init_schema
        # takes the fresh-schema branch. Reaching a working store at all
        # proves the multi-statement DDL executed.
        assert store.stats()["total_versions"] == 0

    def test_reopening_an_existing_database_is_idempotent(self, postgres_dsn, tmp_path, store):
        # Re-running the CREATE TABLE IF NOT EXISTS script against a populated
        # database must not disturb it.
        version = store.create_artifact("a1", "prov-1", _spec())
        store.finalize_artifact("a1", version, "{}", row_count=0, byte_size=0)

        reopened = ArtifactStore(tmp_path / "artifacts", dialect=PostgresDialect(postgres_dsn))
        assert reopened.get_latest_version("a1") is not None


class TestRoundTrip:
    def test_create_finalize_and_find_by_provenance(self, store):
        version = store.create_artifact("a1", "prov-1", _spec())
        store.write_blob("a1", version, b"payload")
        store.finalize_artifact("a1", version, '{"f": []}', row_count=1, byte_size=7)

        found = store.find_by_provenance("prov-1")
        assert found is not None
        assert found.id == "a1"
        assert store.read_blob("a1", version) == b"payload"

    def test_versions_increment(self, store):
        assert store.create_artifact("a1", "p1", _spec()) == 1
        assert store.create_artifact("a1", "p2", _spec()) == 2
        assert store.create_artifact("a1", "p3", _spec()) == 3

    def test_name_pointer_resolves(self, store):
        version = store.create_artifact("a1", "prov-1", _spec())
        store.finalize_artifact("a1", version, "{}", row_count=0, byte_size=0)
        store.set_name("daily_revenue", "a1", version)

        resolved = store.resolve_name("daily_revenue")
        assert resolved is not None
        assert resolved.id == "a1"

    def test_tags_and_aliases_round_trip(self, store):
        # Exercises _REGISTRY_SCHEMA_SQL, which carries the one AUTOINCREMENT
        # column in the schema (registry_audit.seq).
        version = store.create_artifact("a1", "prov-1", _spec())
        store.finalize_artifact("a1", version, "{}", row_count=0, byte_size=0)
        store.set_name("model", "a1", version)
        store.set_alias("model", "champion", "a1", version)
        store.set_tag("a1", version, "auc", "0.91")

        assert store.get_tags("a1", version)["auc"] == "0.91"
        resolved = store.resolve_alias("model", "champion")
        assert resolved is not None and resolved.id == "a1"

    def test_audit_rows_are_dict_convertible(self, store):
        # read_audit does `dict(row)`, which needs the row factory to behave
        # like a Mapping rather than a tuple.
        version = store.create_artifact("a1", "prov-1", _spec())
        store.finalize_artifact("a1", version, "{}", row_count=0, byte_size=0)
        store.set_name("model", "a1", version)

        audit = store.read_audit()
        assert audit
        assert all(isinstance(entry, dict) for entry in audit)
        assert any(entry.get("name") == "model" for entry in audit)


class TestTimestampPrecision:
    """The REAL-vs-DOUBLE PRECISION trap, checked against a live server."""

    def test_created_at_keeps_sub_second_resolution(self, store):
        before = time.time()
        version = store.create_artifact("a1", "prov-1", _spec())
        after = time.time()

        artifact = store.get_artifact("a1", version)
        assert artifact is not None
        # Under a single-precision column this lands seconds-to-minutes away
        # from the true value, so the window would not hold.
        assert before <= artifact.created_at <= after


class TestWriterSerialization:
    """What replaced BEGIN IMMEDIATE."""

    def test_lock_id_is_stable_and_in_range(self):
        # Recomputed on every call site; drift would silently stop serializing.
        assert advisory_lock_id("a1") == advisory_lock_id("a1")
        assert advisory_lock_id("a1") != advisory_lock_id("a2")
        assert -(2**63) <= advisory_lock_id("a1") < 2**63

    def test_concurrent_creates_get_distinct_versions(self, store):
        # The bug the lock exists to prevent: two writers both read MAX=N and
        # collide on the (id, version) primary key. Without serialization this
        # raises or duplicates; with it every writer gets its own version.
        versions: list[int] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(8)

        def create(i: int) -> None:
            barrier.wait()
            try:
                versions.append(store.create_artifact("contended", f"prov-{i}", _spec()))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert sorted(versions) == list(range(1, 9))

    def test_duplicate_provenance_finalize_is_idempotent(self, store):
        # Exercises the dialect's integrity_error: the partial unique index on
        # (tenant, provenance_hash) rejects the second ready row, and the store
        # catches it and returns the existing artifact.
        first = store.create_artifact("a1", "same-prov", _spec())
        store.finalize_artifact("a1", first, "{}", row_count=0, byte_size=0)

        second = store.create_artifact("a2", "same-prov", _spec())
        result = store.finalize_artifact("a2", second, "{}", row_count=0, byte_size=0)

        assert result is not None
        found = store.find_by_provenance("same-prov")
        assert found is not None
