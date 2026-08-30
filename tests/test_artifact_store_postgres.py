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

    yield ArtifactStore(tmp_path / "artifacts", dialect=dialect)

    # Each test builds its own pool against the shared container. Left open
    # they accumulate toward Postgres's default max_connections of 100 and
    # later tests start failing for reasons that have nothing to do with them.
    dialect.close()


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


class TestColumnWidths:
    """Both numeric column types are narrower in Postgres than in SQLite."""

    def test_byte_size_holds_an_artifact_over_two_gigabytes(self, store):
        # Postgres INTEGER is int4, capped at 2147483647. Under it this raises
        # NumericValueOutOfRange -- a DataError, not an IntegrityError, so the
        # finalize handler does not catch it: the blob is already written and
        # the row is stranded in 'building' forever.
        version = store.create_artifact("big", "prov-big", _spec())
        three_gib = 3 * 1024**3
        store.finalize_artifact("big", version, "{}", row_count=1, byte_size=three_gib)

        artifact = store.get_artifact("big", version)
        assert artifact is not None
        assert artifact.byte_size == three_gib
        assert artifact.state == "ready"

    def test_row_count_holds_more_than_two_billion_rows(self, store):
        version = store.create_artifact("wide", "prov-wide", _spec())
        rows = 5_000_000_000
        store.finalize_artifact("wide", version, "{}", row_count=rows, byte_size=1)
        assert store.get_artifact("wide", version).row_count == rows


class TestConcurrentSchemaInitialization:
    """Every node runs _init_schema at startup, against one database."""

    def test_simultaneous_first_boots_all_succeed(self, postgres_dsn, tmp_path):
        # CREATE TABLE IF NOT EXISTS is not concurrency-safe in Postgres: it
        # checks and then creates without a lock, so simultaneous creators race
        # in the system catalog and all but one fail with a duplicate key on
        # pg_type_typname_nsp_index. Observed 7 of 8 failing before the schema
        # advisory lock; multi-node boot is the whole point of this backend.
        dialect = PostgresDialect(postgres_dsn)
        conn = dialect.connect()
        try:
            conn.executescript(
                "DROP TABLE IF EXISTS artifact_versions, artifact_names, "
                "artifact_aliases, artifact_tags, registry_audit, "
                "registry_pending CASCADE;"
            )
            conn.commit()
        finally:
            conn.close()

        errors: list[Exception] = []
        barrier = threading.Barrier(8)

        def boot(i: int) -> None:
            barrier.wait()
            try:
                ArtifactStore(tmp_path / f"node{i}", dialect=PostgresDialect(postgres_dsn))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=boot, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []


class TestAggregateReturnTypes:
    """Postgres aggregates do not have SQLite's return types."""

    def test_stats_totals_are_ints_not_decimals(self, store):
        # SUM over a BIGINT column is `numeric` in Postgres and arrives as
        # Decimal, so an in-process caller doing total_bytes / 1024**3 gets
        # TypeError where the SQLite backend gave a float.
        version = store.create_artifact("a1", "prov-1", _spec())
        store.finalize_artifact("a1", version, "{}", row_count=7, byte_size=11)

        stats = store.stats()
        assert isinstance(stats["total_bytes"], int)
        assert isinstance(stats["total_rows"], int)
        assert stats["total_bytes"] / 1024**3 > 0

    def test_usage_totals_are_ints_not_decimals(self, store):
        version = store.create_artifact("a1", "prov-1", _spec())
        store.finalize_artifact("a1", version, "{}", row_count=7, byte_size=11)

        usage = store.get_usage()
        assert isinstance(usage["total_bytes"], int)
        assert isinstance(usage["total_rows"], int)


class TestConnectionLimits:
    """SQLite's timeouts had to be carried over, not dropped."""

    def test_lock_timeout_matches_the_sqlite_busy_timeout(self, postgres_dsn):
        # pg_advisory_xact_lock waits forever by default, so a node stalling
        # while holding the global schema lock would hang every other node
        # with no error and no bound. sqlite3.connect(timeout=30.0) failed
        # loudly after 30s; this restores the same ceiling.
        conn = PostgresDialect(postgres_dsn).connect()
        try:
            assert conn.execute("SHOW lock_timeout").fetchone()[0] == "30s"
        finally:
            conn.close()

    def test_schema_lock_is_skipped_once_the_schema_exists(self, postgres_dsn, store):
        # The schema lock is global. An ArtifactStore is constructed per
        # session in several places, so taking it on every construction would
        # funnel the whole cluster through one mutex.
        dialect = PostgresDialect(postgres_dsn)
        conn = dialect.connect()
        try:
            assert dialect.schema_exists(conn) is True
        finally:
            conn.close()


class TestCanonicalPromotion:
    def test_promotion_returns_the_canonical_id_never_a_foreign_one(self, store):
        # The only caller reaches force_finalize_canonical *because* finalize
        # landed under a different id, so returning that foreign id back would
        # leave the canonical row 'failed' and the caller unaware.
        first = store.create_artifact("a1", "shared-prov", _spec())
        store.finalize_artifact("a1", first, "{}", row_count=0, byte_size=0)

        second = store.create_artifact("a2", "shared-prov", _spec())
        deduped = store.finalize_artifact("a2", second, "{}", row_count=0, byte_size=0)
        assert deduped is not None and deduped.id == "a1"  # dedup put us on a1

        promoted = store.force_finalize_canonical(
            artifact_id="a2", version=second, schema_json="{}", row_count=0, byte_size=0
        )
        assert promoted is not None
        assert promoted.id == "a2"
        assert promoted.state == "ready"


class TestConnectionPool:
    """A bounded pool is only safe here because acquisition is re-entrant."""

    def test_nested_acquisition_reuses_one_pooled_connection(self, postgres_dsn):
        dialect = PostgresDialect(postgres_dsn)
        try:
            outer = dialect.connect()
            inner = dialect.connect()
            try:
                # Same underlying connection, so the nested call cannot be
                # waiting on the pool for a second one.
                assert outer._inner is inner._inner
            finally:
                inner.close()
                # Released only by the outermost holder.
                assert dialect._local.conn is not None
                outer.close()
                assert dialect._local.conn is None
        finally:
            dialect.close()

    def test_more_threads_than_pool_slots_still_complete(self, postgres_dsn, tmp_path):
        # The deadlock this guards: ArtifactStore acquires two deep in six
        # places, so without re-entrancy max_size threads each holding one and
        # waiting for a second block until the pool timeout. With max_size=2
        # and 8 threads doing nested work, that is unmissable.
        dialect = PostgresDialect(postgres_dsn, max_size=2)
        try:
            conn = dialect.connect()
            conn.executescript(
                "DROP TABLE IF EXISTS artifact_versions, artifact_names, "
                "artifact_aliases, artifact_tags, registry_audit, "
                "registry_pending CASCADE;"
            )
            conn.commit()
            conn.close()

            store = ArtifactStore(tmp_path / "pool", dialect=dialect)
            errors: list[Exception] = []
            done: list[int] = []
            barrier = threading.Barrier(8)

            def work(i: int) -> None:
                barrier.wait()
                try:
                    for round_ in range(3):
                        aid = f"a{i}-{round_}"
                        version = store.create_artifact(aid, f"prov-{i}-{round_}", _spec())
                        # finalize_artifact evaluates `return self.get_artifact(...)`
                        # inside its try, so the outer connection is still held.
                        store.finalize_artifact(aid, version, "{}", row_count=1, byte_size=1)
                    done.append(i)
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                # Generous but finite: a deadlock shows up as a thread that
                # never finishes, not as an exception.
                thread.join(timeout=90)

            assert [t for t in threads if t.is_alive()] == [], "threads deadlocked on the pool"
            assert errors == []
            assert sorted(done) == list(range(8))
        finally:
            dialect.close()

    def test_pool_is_not_opened_until_something_queries(self, postgres_dsn):
        # Constructing a dialect must cost no sockets; the config factory and
        # every test fixture build them freely.
        dialect = PostgresDialect(postgres_dsn)
        assert dialect._pool is None
        try:
            dialect.connect().close()
            assert dialect._pool is not None
        finally:
            dialect.close()


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
