"""The artifact store running on Postgres instead of SQLite.

Exercises the same ``ArtifactStore`` methods personal mode uses against a real server, so the port
is proven by behaviour rather than rendered SQL. Requires Docker and the ``postgres`` extra.
"""

from __future__ import annotations

import threading
import time

import docker
import pytest
from testcontainers.community.postgres import PostgresContainer

from strata.artifact_cli import _server_store as _configured_server_store
from strata.artifact_store import ArtifactStore, TransformSpec
from strata.sql_backend import PostgresDialect, advisory_lock_id


def _docker_daemon_reachable() -> bool:
    """Skip when the Docker daemon is unreachable (only possible locally; CI always has Docker)."""
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
    with PostgresContainer("postgres:18-alpine") as container:
        yield container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")


@pytest.fixture
def store(postgres_dsn, tmp_path):
    """A store with metadata on Postgres and local blobs; each test gets a clean schema."""
    dialect = PostgresDialect(postgres_dsn)
    conn = dialect.connect()
    try:
        conn.executescript(
            "DROP TABLE IF EXISTS artifact_versions, artifact_names, artifact_aliases, "
            "artifact_tags, registry_audit, registry_pending, artifact_publications, "
            "notebook_workers, notebook_worker_registry, schema_version CASCADE;"
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


class TestPing:
    """The readiness probe's query runs on Postgres."""

    def test_ping_answers_on_a_live_server(self, store):
        store.ping()


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
        try:
            assert reopened.get_latest_version("a1") is not None
        finally:
            reopened.close()


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

    def test_ancestors_are_read_through_their_input_edges(self, store):
        """The ACL's lineage walk: rows matched on (id, version) pairs, read by column name."""
        scan = TransformSpec(executor="scan@v1", params={}, inputs=["file:///wh#secret.events"])
        store.create_artifact("scan", "p-scan", scan)
        mid_input = "strata://artifact/scan@v=1"
        store.create_artifact("mid", "p-mid", _spec(), input_versions={mid_input: "scan@v=1"})
        top_input = "strata://name/mid"
        store.create_artifact("top", "p-top", _spec(), input_versions={top_input: "mid@v=1"})

        specs = store.ancestor_transform_specs(store.get_artifact("top", 1), max_depth=10)

        assert sorted(specs) == sorted([scan.to_json(), _spec().to_json()])

    def test_name_reads_are_listed_per_tenant(self, store):
        edges = {"strata://name/taxi/model@champion": "m@v=1"}
        for artifact_id, tenant in (("ours", "team-a"), ("theirs", "team-b")):
            version = store.create_artifact(
                artifact_id, f"p-{artifact_id}", _spec(), input_versions=edges, tenant=tenant
            )
            store.finalize_artifact(artifact_id, version, "{}", row_count=0, byte_size=0)

        assert store.list_name_reads(tenant="team-a") == [("team-a", "ours", "taxi/model@champion")]
        assert [read[:2] for read in store.list_name_reads()] == [
            ("team-a", "ours"),
            ("team-b", "theirs"),
        ]

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

    def test_a_credit_change_is_one_event(self, store):
        version = store.create_artifact("a1", "prov-1", _spec())
        store.write_blob("a1", version, b"payload")
        store.finalize_artifact("a1", version, "{}", row_count=0, byte_size=7)
        publication = store.publish_artifact("a1", version)

        store.update_publication_credits(publication.token, authors=[{"name": "F. Li"}])

        assert [(e["action"], e["value"]) for e in store.read_events()] == [
            ("publish", publication.id),
            ("credit", publication.id),
        ]


class TestColumnWidths:
    """Both numeric column types are narrower in Postgres than in SQLite."""

    def test_byte_size_holds_an_artifact_over_two_gigabytes(self, store):
        # Postgres INTEGER is int4, capped at 2147483647. Overflow raises
        # NumericValueOutOfRange, a DataError the finalize handler does not catch, so the
        # blob is written and the row is stranded in 'building' forever.
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
        # CREATE TABLE IF NOT EXISTS is not concurrency-safe in Postgres: simultaneous
        # creators race in the system catalog and all but one fail with a duplicate key on
        # pg_type_typname_nsp_index. Multi-node boot is the whole point of this backend.
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
            dialect.close()

        errors: list[Exception] = []
        booted: list[ArtifactStore] = []
        barrier = threading.Barrier(8)

        def boot(i: int) -> None:
            barrier.wait()
            try:
                booted.append(
                    ArtifactStore(tmp_path / f"node{i}", dialect=PostgresDialect(postgres_dsn))
                )
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
        # pg_advisory_xact_lock waits forever by default, so a node stalling while holding
        # the global schema lock would hang every other node with no error. The bound
        # matches SQLite's 30s connect timeout.
        dialect = PostgresDialect(postgres_dsn)
        try:
            conn = dialect.connect()
            try:
                assert conn.execute("SHOW lock_timeout").fetchone()[0] == "30s"
            finally:
                conn.close()
        finally:
            dialect.close()

    def test_schema_lock_is_skipped_once_the_schema_exists(self, postgres_dsn, store):
        # The schema lock is global. An ArtifactStore is constructed per
        # session in several places, so taking it on every construction would
        # funnel the whole cluster through one mutex.
        dialect = PostgresDialect(postgres_dsn)
        try:
            conn = dialect.connect()
            try:
                assert dialect.schema_exists(conn) is True
            finally:
                conn.close()
        finally:
            dialect.close()


class TestCanonicalPromotion:
    def test_a_runs_outputs_finalize_together_or_not_at_all(self, store):
        from strata.artifact_store import StagedVersion

        first = store.create_artifact("a1", "shared-prov", _spec())
        store.finalize_artifact("a1", first, "{}", row_count=0, byte_size=0)
        x = StagedVersion(
            "a2", store.create_artifact("a2", "shared-prov", _spec()), "{}", 0, 0, "d"
        )
        y = StagedVersion("a3", store.create_artifact("a3", "prov-y", _spec()), "{}", 0, 0, "d")
        z = StagedVersion("a4", store.create_artifact("a4", "prov-z", _spec()), "{}", 0, 0, "d")
        store.fail_artifact("a4", z.version)

        with pytest.raises(ValueError, match="building"):
            store.finalize_canonical_together([y, z])
        assert store.get_artifact("a3", y.version).state == "building"

        finalized = store.finalize_canonical_together([x, y])

        assert [(v.id, v.state) for v in finalized] == [("a2", "ready"), ("a3", "ready")]
        assert store.get_artifact("a1", first).state == "superseded"


class TestGarbageCollection:
    def test_the_current_value_survives_a_rebuild_in_flight(self, store):
        # The spare-the-current-value clause is a correlated NOT EXISTS; run
        # it on the dialect that has never seen it in a unit test.
        first = store.create_artifact("nb_x_cell_c1_var_df", "prov-v1", _spec())
        store.finalize_artifact("nb_x_cell_c1_var_df", first, "{}", row_count=0, byte_size=0)
        store.create_artifact("nb_x_cell_c1_var_df", "prov-v2", _spec())

        assert store.garbage_collect(max_idle_days=0)["deleted_count"] == 0
        assert store.get_latest_version("nb_x_cell_c1_var_df").version == first

    @staticmethod
    def _ready(store, artifact_id: str, provenance: str, *, minted: bool = False) -> int:
        version = store.create_artifact(artifact_id, provenance, _spec(), minted=minted)
        store.finalize_artifact(artifact_id, version, "{}", row_count=0, byte_size=100)
        return version

    @staticmethod
    def _last_used(store, artifact_id: str, seconds_ago: float) -> None:
        conn = store._get_connection()
        try:
            conn.execute(
                "UPDATE artifact_versions SET created_at = 0, last_used_at = ? WHERE id = ?",
                (time.time() - seconds_ago, artifact_id),
            )
            conn.commit()
        finally:
            conn.close()

    def test_retention_runs_its_queries_on_postgres(self, store):
        """The minted clause, LRU order under a cap, the version-gap GROUP BY and keep-superseded,
        on the only dialect that runs them here.
        """
        oldest = "0b6f2a4e-6d1a-4c0e-9b1e-2f8d6a3c1e01"
        newer = "0b6f2a4e-6d1a-4c0e-9b1e-2f8d6a3c1e02"
        self._ready(store, oldest, "prov-old", minted=True)
        self._ready(store, newer, "prov-new", minted=True)
        for version in range(1, 4):
            self._ready(store, "nb_x_cell_c1_var_df", f"prov-nb-{version}")
        self._last_used(store, oldest, 7200)
        self._last_used(store, newer, 3600)
        self._last_used(store, "nb_x_cell_c1_var_df", 7200)

        # 500 bytes over a 300 cap: down to 240, least recently used first. The
        # notebook output's current value is never a candidate.
        result = store.garbage_collect(max_bytes=300, keep_superseded=1)

        assert result["store_bytes"] == 500
        assert store.get_artifact(oldest, 1) is None
        assert store.get_artifact("nb_x_cell_c1_var_df", 1) is None
        assert store.get_artifact("nb_x_cell_c1_var_df", 3) is not None

        provenance = store.get_artifact("nb_x_cell_c1_var_df", 3).provenance_hash
        store.find_by_provenance(provenance)
        conn = store._get_connection()
        try:
            row = conn.execute(
                "SELECT last_used_at FROM artifact_versions WHERE id = ? AND version = 3",
                ("nb_x_cell_c1_var_df",),
            ).fetchone()
        finally:
            conn.close()
        assert row["last_used_at"] == pytest.approx(time.time(), abs=60)

    def test_a_hit_during_the_sweep_keeps_the_version(self, store, monkeypatch):
        """Another node's cache hit commits while this sweep is between choosing and deleting."""
        artifact = "0b6f2a4e-6d1a-4c0e-9b1e-2f8d6a3c1e03"
        self._ready(store, artifact, "prov-hit", minted=True)
        self._last_used(store, artifact, 7200)
        chosen_then = ArtifactStore._without_version_gaps

        def hit_after_choosing(conn, chosen):
            # Its own thread, so its own pooled connection and transaction.
            hit = threading.Thread(target=store.find_by_provenance, args=("prov-hit",))
            hit.start()
            hit.join()
            return chosen_then(conn, chosen)

        monkeypatch.setattr(
            ArtifactStore, "_without_version_gaps", staticmethod(hit_after_choosing)
        )

        result = store.garbage_collect(max_idle_days=0, min_idle_seconds=3600)

        assert result["deleted_count"] == 0
        assert store.get_artifact(artifact, 1) is not None

    def test_an_overtaken_version_goes_with_its_canonical(self, store):
        canonical = "0b6f2a4e-6d1a-4c0e-9b1e-2f8d6a3c1e04"
        overtaken = "0b6f2a4e-6d1a-4c0e-9b1e-2f8d6a3c1e05"
        for artifact_id in (canonical, overtaken):
            version = store.create_artifact(artifact_id, "prov-dup", _spec(), minted=True)
            store.write_blob(artifact_id, version, b"x" * 100)
            store.finalize_artifact(artifact_id, version, "{}", row_count=0, byte_size=100)
        # The canonical the older, as finalize leaves it: the cap stops right after it.
        self._last_used(store, canonical, 7200)
        self._last_used(store, overtaken, 3600)

        result = store.garbage_collect(max_bytes=50)

        assert result["deleted_count"] == 2
        assert store.get_artifact(canonical, 1) is None
        assert store.get_artifact(overtaken, 1) is None

    def test_deleting_a_canonical_hands_its_bytes_to_a_pinned_reader(self, store):
        for artifact_id in ("a1", "a2"):
            version = store.create_artifact(artifact_id, "prov-dup", _spec())
            store.write_blob(artifact_id, version, b"bytes")
            store.finalize_artifact(artifact_id, version, "{}", row_count=0, byte_size=5)
        store.pin_artifact("a2", 1, "review")

        assert store.delete_artifact("a1", 1)

        assert store.read_blob("a2", 1) == b"bytes"
        assert store.blob_store.blob_exists("a2", 1)

    @pytest.mark.parametrize("hold", ["publish", "pin"])
    def test_deleting_a_canonical_whose_blob_is_gone_under_a_held_reader_refuses(self, store, hold):
        for artifact_id in ("a1", "a2"):
            version = store.create_artifact(artifact_id, "prov-dup", _spec())
            store.write_blob(artifact_id, version, b"bytes")
            store.finalize_artifact(artifact_id, version, "{}", row_count=0, byte_size=5)
        if hold == "publish":
            store.publish_artifact("a2", 1)
        else:
            store.pin_artifact("a2", 1, "review")
        store.blob_store.delete_blob("a1", 1)

        with pytest.raises(ValueError, match="whose blob is gone"):
            store.delete_artifact("a1", 1)

        assert store.get_artifact("a1", 1) is not None
        conn = store._get_connection()
        try:
            held = conn.execute(
                "SELECT state, superseded_by FROM artifact_versions WHERE id = ? AND version = 1",
                ("a2",),
            ).fetchone()
        finally:
            conn.close()
        assert (held["state"], held["superseded_by"]) == ("superseded", "a1@v=1")

    @pytest.mark.parametrize("raced", [1, 2])
    def test_a_name_landing_mid_sweep_skips_only_that_version(self, store, raced):
        """A failed DELETE aborts a Postgres transaction; without a savepoint the closing commit
        rolled back every row while their blobs were still deleted.
        """
        ids = [f"0b6f2a4e-6d1a-4c0e-9b1e-2f8d6a3c1e1{i}" for i in range(3)]
        for i, artifact_id in enumerate(ids):
            version = store.create_artifact(artifact_id, f"prov-race-{i}", _spec(), minted=True)
            store.write_blob(artifact_id, version, b"x" * 100)
            store.finalize_artifact(artifact_id, version, "{}", row_count=0, byte_size=100)
            self._last_used(store, artifact_id, 7200 - i)
        children = store._delete_version_children

        def name_lands(conn, artifact_id, version):
            # Its own thread, so its own pooled connection; the claim does not block a name.
            if artifact_id == ids[raced]:
                setter = threading.Thread(target=store.set_name, args=("raced", artifact_id, 1))
                setter.start()
                setter.join()
            children(conn, artifact_id, version)

        store._delete_version_children = name_lands
        try:
            result = store.garbage_collect(max_idle_days=0)
        finally:
            store._delete_version_children = children

        assert result["deleted_count"] == 2
        for i, artifact_id in enumerate(ids):
            assert (store.get_artifact(artifact_id, 1) is not None) == (i == raced)
            assert store.blob_store.blob_exists(artifact_id, 1) == (i == raced)

    def test_cleanup_failed_takes_a_failed_build_row_with_it(self, store, tmp_path):
        """The build row's foreign key refused deleting the failed version it references."""
        from strata.transforms.build_store import BuildStore

        # The fixture's CASCADE drop leaves a prior test's build table without its foreign key.
        conn = store._get_connection()
        try:
            conn.executescript("DROP TABLE IF EXISTS artifact_builds;")
            conn.commit()
        finally:
            conn.close()
        builds = BuildStore(tmp_path / "builds.sqlite", dialect=store.dialect)
        version = store.create_artifact("built", "prov-built", _spec())
        builds.create_build("build-1", "built", version, "exec@v1")
        builds.fail_build("build-1", "boom")
        store.fail_artifact("built", version)

        assert store.cleanup_failed(max_age_seconds=-10) == 1
        assert store.get_artifact("built", version) is None
        assert builds.get_build("build-1") is None


class TestConnectionPool:
    """A bounded pool is only safe here because acquisition is re-entrant."""

    def test_reads_return_their_connection_without_a_pool_warning(self, store, caplog):
        """psycopg_pool warns and rolls back whenever a connection comes back mid-transaction,
        which every read did, burying real warnings under one per call.
        """
        import logging

        version = store.create_artifact("warn-a", "warn-p", _spec())
        store.write_blob("warn-a", version, b"x")
        store.finalize_artifact("warn-a", version, "{}", row_count=0, byte_size=1)

        with caplog.at_level(logging.WARNING, logger="psycopg.pool"):
            store.get_artifact("warn-a", version)
            store.get_latest_version("warn-a")
            store.find_by_provenance("warn-p")
            store.read_blob("warn-a", version)
            store.list_artifacts()
            store.stats()

        assert [r.getMessage() for r in caplog.records if r.name.startswith("psycopg")] == []

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

    def test_releasing_after_close_does_not_resurrect_a_pool(self, postgres_dsn):
        # Releasing a still-checked-out connection after close() must not rebuild the
        # pool through _get_pool(): that opens fresh sockets and then raises "can't return
        # connection to pool 'pool-2', it comes from 'pool-1'".
        dialect = PostgresDialect(postgres_dsn)
        conn = dialect.connect()
        conn.execute("SELECT 1").fetchone()
        dialect.close()

        conn.close()  # must dispose the connection, not rebuild the pool
        assert dialect._pool is None

    def test_close_is_terminal(self, postgres_dsn):
        # A closed dialect quietly reopening defeats the connection bound that
        # close() exists to enforce.
        dialect = PostgresDialect(postgres_dsn)
        dialect.connect().close()
        dialect.close()

        with pytest.raises(RuntimeError, match="closed"):
            dialect.connect()

    def test_release_from_a_non_owning_thread_is_ignored(self, postgres_dsn):
        # A release from a thread that never acquired must not raise AttributeError or
        # putconn(None), which would lose the real connection from a bounded pool.
        dialect = PostgresDialect(postgres_dsn)
        try:
            conn = dialect.connect()
            errors: list[Exception] = []

            def release_elsewhere() -> None:
                try:
                    conn.close()
                except Exception as exc:  # noqa: BLE001 - recorded, then asserted
                    errors.append(exc)

            thread = threading.Thread(target=release_elsewhere)
            thread.start()
            thread.join()

            assert errors == []
            # The owning thread still holds it, so the pool did not lose it.
            assert dialect._local.conn is not None
            conn._closed = False  # the foreign close flipped the wrapper's guard
            conn.close()
            assert dialect._local.conn is None
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


class TestConfigurationSelectsTheBackend:
    """The wiring end to end: a DSN in config produces a Postgres-backed store."""

    def test_a_configured_dsn_produces_a_postgres_backed_store(self, postgres_dsn, tmp_path):
        from strata.artifact_store import get_artifact_store, reset_artifact_store
        from strata.config import StrataConfig
        from strata.server import _init_configured_artifact_store

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_metadata_dsn=postgres_dsn,
        )

        reset_artifact_store()
        try:
            # The same call the server's lifespan makes.
            _init_configured_artifact_store(config)
            store = get_artifact_store(config.artifact_dir)
            assert store is not None
            assert store._dialect.name == "postgres"

            # And it actually works against the server, not just constructs.
            version = store.create_artifact("wired", "prov-wired", _spec())
            store.finalize_artifact("wired", version, "{}", row_count=1, byte_size=1)
            assert store.get_latest_version("wired") is not None

            # No SQLite file was created alongside it.
            assert not (config.artifact_dir / "artifacts.sqlite").exists()
        finally:
            store = get_artifact_store()
            if store is not None:
                store.close()
            reset_artifact_store()

    def test_no_dsn_still_produces_a_sqlite_store(self, tmp_path):
        from strata.artifact_store import get_artifact_store, reset_artifact_store
        from strata.config import StrataConfig
        from strata.server import _init_configured_artifact_store

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )
        reset_artifact_store()
        try:
            _init_configured_artifact_store(config)
            store = get_artifact_store(config.artifact_dir)
            assert store is not None
            assert store._dialect.name == "sqlite"
            assert (config.artifact_dir / "artifacts.sqlite").exists()
        finally:
            reset_artifact_store()


class TestBuildStoreSharesTheBackend:
    """Build rows live in the artifact store's database, so they follow it.

    On a node-local SQLite file, a build claimed on one node would be invisible to ``GET
    /v1/builds/{id}`` on another.
    """

    def test_build_created_on_one_node_is_visible_on_another(self, postgres_dsn, tmp_path):
        from strata.transforms.build_store import BuildStore

        dialect_a = PostgresDialect(postgres_dsn)
        dialect_b = PostgresDialect(postgres_dsn)
        try:
            conn = dialect_a.connect()
            conn.executescript(
                "DROP TABLE IF EXISTS artifact_builds, artifact_versions, "
                "artifact_names, artifact_aliases, artifact_tags, "
                "registry_audit, registry_pending CASCADE;"
            )
            conn.commit()
            conn.close()

            # The artifact store first: artifact_builds has a FOREIGN KEY to
            # artifact_versions, and Postgres requires the referenced table at CREATE time and
            # enforces it on insert (SQLite never turns foreign_keys on). The server's call
            # sites build the artifact store first for this reason.
            store_a = ArtifactStore(tmp_path / "a", dialect=dialect_a)
            version = store_a.create_artifact("art-1", "prov-1", _spec())

            node_a = BuildStore(tmp_path / "a.sqlite", dialect=dialect_a)
            node_b = BuildStore(tmp_path / "b.sqlite", dialect=dialect_b)

            build_id = "build-cross-node"
            node_a.create_build(
                build_id=build_id,
                artifact_id="art-1",
                version=version,
                executor_ref="duckdb_sql_v1",
            )
            # The node that did not create it can still see it.
            seen = node_b.get_build(build_id)
            assert seen is not None
            assert seen.artifact_id == "art-1"

            # And a claim made on B is observed by A.
            assert node_b.start_build(build_id) is True
            assert node_a.get_build(build_id).state == "building"

            # No SQLite file was created for either node.
            assert not (tmp_path / "a.sqlite").exists()
            assert not (tmp_path / "b.sqlite").exists()
        finally:
            dialect_a.close()
            dialect_b.close()

    def test_console_left_by_one_node_is_taken_by_the_other(self, postgres_dsn, tmp_path):
        """A remote cell's console chunk can land on a node that did not dispatch it."""
        from strata.transforms.build_store import BuildStore

        dialect_a = PostgresDialect(postgres_dsn)
        dialect_b = PostgresDialect(postgres_dsn)
        try:
            conn = dialect_a.connect()
            conn.executescript(
                "DROP TABLE IF EXISTS build_console_chunks, artifact_builds, artifact_versions, "
                "artifact_names, artifact_aliases, artifact_tags, "
                "registry_audit, registry_pending CASCADE;"
            )
            conn.commit()
            conn.close()

            version = ArtifactStore(tmp_path / "a", dialect=dialect_a).create_artifact(
                "art-c", "prov-c", _spec()
            )
            node_a = BuildStore(tmp_path / "a.sqlite", dialect=dialect_a)
            node_b = BuildStore(tmp_path / "b.sqlite", dialect=dialect_b)
            node_a.create_build(
                build_id="b-c", artifact_id="art-c", version=version, executor_ref="x"
            )
            node_a.start_build("b-c")

            assert node_b.append_console_chunk("b-c", "stdout", 1, "epoch 2\n")
            assert node_b.append_console_chunk("b-c", "stdout", 0, "epoch 1\n")
            assert not node_b.append_console_chunk("b-c", "stdout", 0, "epoch 1\n")
            assert not node_b.append_console_chunk("missing", "stdout", 0, "stray\n")

            assert node_a.take_console_chunks("b-c") == [
                ("stdout", 0, "epoch 1\n"),
                ("stdout", 1, "epoch 2\n"),
            ]
            assert node_a.take_console_chunks("b-c") == []
            node_b.append_console_chunk("b-c", "stderr", 0, "late\n")
            node_a.delete_console_chunks("b-c")
            assert node_a.take_console_chunks("b-c") == []
        finally:
            dialect_a.close()
            dialect_b.close()

    def test_build_columns_survive_postgres_widths(self, postgres_dsn, tmp_path):
        # Same INTEGER/REAL traps as the artifact store: byte counts are
        # INTEGER and timestamps are REAL in the shared schema.
        from strata.transforms.build_store import BuildStore

        dialect = PostgresDialect(postgres_dsn)
        try:
            conn = dialect.connect()
            conn.executescript(
                "DROP TABLE IF EXISTS artifact_builds, artifact_versions, "
                "artifact_names, artifact_aliases, artifact_tags, "
                "registry_audit, registry_pending CASCADE;"
            )
            conn.commit()
            conn.close()

            artifacts = ArtifactStore(tmp_path / "w", dialect=dialect)
            version = artifacts.create_artifact("big", "prov-big", _spec())

            store = BuildStore(tmp_path / "x.sqlite", dialect=dialect)
            before = time.time()
            build_id = "build-widths"
            store.create_build(
                build_id=build_id,
                artifact_id="big",
                version=version,
                executor_ref="duckdb_sql_v1",
            )
            store.start_build(build_id)
            store.complete_build(build_id, output_byte_count=3 * 1024**3)
            after = time.time()

            build = store.get_build(build_id)
            assert build.output_byte_count == 3 * 1024**3
            assert before <= build.created_at <= after
        finally:
            dialect.close()


class TestTheAttemptLedgerOnPostgres:
    """``build_attempts`` is created apart from ``artifact_builds``, so an existing database gains
    it, and deadlines keep sub-second precision (REAL would round them to minutes).
    """

    def test_an_existing_database_gains_the_ledger(self, postgres_dsn, tmp_path):
        from strata.transforms.build_store import BuildStore

        dialect = PostgresDialect(postgres_dsn)
        try:
            conn = dialect.connect()
            conn.executescript(
                "DROP TABLE IF EXISTS build_attempts, artifact_builds, artifact_versions, "
                "artifact_names, artifact_aliases, artifact_tags, "
                "registry_audit, registry_pending CASCADE;"
            )
            conn.commit()
            conn.close()

            artifacts = ArtifactStore(tmp_path / "w", dialect=dialect)
            version = artifacts.create_artifact("att", "prov-att", _spec())
            BuildStore(tmp_path / "x.sqlite", dialect=dialect)

            # A database from before the ledger: builds, but no attempts.
            conn = dialect.connect()
            conn.executescript("DROP TABLE build_attempts;")
            conn.commit()
            conn.close()

            store = BuildStore(tmp_path / "x.sqlite", dialect=dialect)
            store.create_build(
                build_id="b-att", artifact_id="att", version=version, executor_ref="x"
            )
            deadline = time.time() - 0.25
            store.record_attempt("b-att", "att", version, "ab" * 16, writable_until=deadline)
            store.record_attempt("b-att", "att", version, "ab" * 16, writable_until=deadline - 60)
            store.start_build("b-att")
            assert store.fail_build("b-att", "gave up")

            assert store.settled_attempts() == [("b-att", "att", version, "ab" * 16, False)]
            conn = dialect.connect()
            kept = conn.execute("SELECT writable_until FROM build_attempts").fetchone()
            conn.close()
            assert kept["writable_until"] == pytest.approx(deadline, abs=1e-3)

            store.forget_attempt("b-att", "ab" * 16)
            assert store.settled_attempts() == []
        finally:
            dialect.close()


class TestTheCliOnAPostgresStore:
    """With a service store's settings, the artifact CLI must open Postgres, not an empty SQLite
    file.
    """

    def test_archive_by_token_writes_the_zip_the_route_serves(
        self, postgres_dsn, tmp_path, monkeypatch
    ):
        import argparse

        from strata.api.publication_bundle import bundle_zip
        from strata.artifact_cli import cmd_archive

        dialect = PostgresDialect(postgres_dsn)
        try:
            conn = dialect.connect()
            conn.executescript(
                "DROP TABLE IF EXISTS artifact_builds, artifact_versions, artifact_names, "
                "artifact_aliases, artifact_tags, artifact_publications, artifact_pins, "
                "registry_audit, registry_pending CASCADE;"
            )
            conn.commit()
            conn.close()

            artifact_dir = tmp_path / "store"
            store = ArtifactStore(artifact_dir, dialect=dialect)
            version = store.create_artifact("fig", "prov-fig", _spec())
            with store.open_blob_writer("fig", version) as writer:
                writer.write(b"figure bytes")
            store.finalize_artifact("fig", version, schema_json="", row_count=1, byte_size=12)
            publication = store.publish_artifact("fig", version, title="Figure")

            monkeypatch.setenv("STRATA_ARTIFACT_METADATA_DSN", postgres_dsn)
            monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(artifact_dir))
            out = tmp_path / "deposit.zip"
            args = argparse.Namespace(
                ref=None,
                token=publication.token,
                artifact_dir=None,
                to=str(out),
                force=False,
                title=None,
                author=None,
                tenant=None,
                max_depth=10,
            )
            assert cmd_archive(args) == 0

            artifact = store.get_artifact("fig", version)
            bundle_zip(store, artifact, tmp_path / "direct.zip", publication=publication)
            assert out.read_bytes() == (tmp_path / "direct.zip").read_bytes()
        finally:
            dialect.close()

    def test_publish_mints_the_grant_in_the_store_the_server_serves(
        self, store, tmp_path, monkeypatch, postgres_dsn
    ):
        """The publish target is built like the server's store, not from ``artifact_dir`` alone.

        Built from the directory, it was an empty SQLite file: "not found", and a stray database.
        """
        import argparse

        from strata.artifact_cli import cmd_publish

        version = store.create_artifact("fig", "prov-fig", _spec())
        with store.open_blob_writer("fig", version) as writer:
            writer.write(b"figure bytes")
        store.finalize_artifact("fig", version, schema_json="", row_count=1, byte_size=12)

        monkeypatch.setenv("STRATA_ARTIFACT_METADATA_DSN", postgres_dsn)
        monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
        # The suite stubs this out so no test publishes into a developer's own store.
        monkeypatch.setattr("strata.artifact_cli._server_store", _configured_server_store)
        args = argparse.Namespace(
            ref=f"fig@v={version}",
            artifact_dir=None,
            format="json",
            title="Figure",
            author=None,
            tenant=None,
            here=False,
            into=None,
            to_url=None,
            max_depth=10,
        )
        assert cmd_publish(args) == 0

        assert [p.artifact_id for p in store.list_publications()] == ["fig"]
        assert not (tmp_path / "artifacts" / "artifacts.sqlite").exists()


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

    def test_an_events_follower_never_skips_an_event_that_commits_late(self, store):
        """``seq`` is drawn at insert: a writer that inserts first and commits last was skipped.

        The follower saw the later writer's event, advanced its cursor past the earlier ``seq``,
        and never got the earlier event once it committed.
        """
        version = store.create_artifact("m1", "prov-m1", _spec())
        store.finalize_artifact("m1", version, "{}", row_count=0, byte_size=0)

        inserted, release = threading.Event(), threading.Event()

        def slow_writer() -> None:
            # Its own thread: connections are shared per thread, so a commit elsewhere on this
            # thread would commit this insert too.
            conn = store._get_connection()
            try:
                store._audit_in_connection(conn, action="name_set", name="slow")
                inserted.set()
                release.wait()
                conn.commit()
            finally:
                conn.close()

        slow = threading.Thread(target=slow_writer)
        slow.start()
        inserted.wait()
        tagging = threading.Thread(target=store.set_tag, args=("m1", version, "k", "v"))
        tagging.start()
        probe = store._get_connection()
        try:
            # Until the tag writer waits behind the slow one (or, unserialized, has committed).
            while tagging.is_alive():
                row = probe.execute(
                    "SELECT count(*) AS n FROM pg_locks WHERE locktype = 'advisory' AND NOT granted"
                ).fetchone()
                probe.commit()
                if row["n"]:
                    break
                tagging.join(timeout=0.05)
        finally:
            probe.close()
        seen = store.read_events()
        cursor = seen[-1]["seq"] if seen else 0
        release.set()
        slow.join()
        tagging.join()

        later = store.read_events(since=cursor)
        assert sorted(e["action"] for e in seen + later) == ["name_set", "tag_set"]

    def test_deleting_an_artifact_while_its_name_is_deleted_does_not_deadlock(self, store):
        """Delete took the audit lock after deleting the name rows; ``delete_name`` the reverse.

        Each then waited on the lock the other held, and Postgres aborted one of them.
        """
        version = store.create_artifact("d1", "prov-d1", _spec())
        store.finalize_artifact("d1", version, "{}", row_count=0, byte_size=0)
        store.set_name("current", "d1", version)
        store.set_alias("current", "prod", "d1", version)

        locked, release = threading.Event(), threading.Event()
        errors: list[BaseException] = []

        def name_deleter() -> None:
            # delete_name's order: the audit lock, then the name row.
            conn = store._get_connection()
            try:
                store._serialize_audit(conn)
                locked.set()
                release.wait()
                conn.execute("DELETE FROM artifact_names WHERE name = ?", ("current",))
                store._audit_in_connection(conn, action="name_delete", name="current")
                conn.commit()
            except BaseException as exc:
                errors.append(exc)
            finally:
                conn.close()

        def artifact_deleter() -> None:
            try:
                store.delete_artifact("d1", version)
            except BaseException as exc:
                errors.append(exc)

        names = threading.Thread(target=name_deleter)
        names.start()
        locked.wait()
        deleting = threading.Thread(target=artifact_deleter)
        deleting.start()
        probe = store._get_connection()
        try:
            # Until the artifact delete waits on the audit lock the name deleter holds.
            while deleting.is_alive():
                row = probe.execute(
                    "SELECT count(*) AS n FROM pg_locks WHERE locktype = 'advisory' AND NOT granted"
                ).fetchone()
                probe.commit()
                if row["n"]:
                    break
                deleting.join(timeout=0.05)
        finally:
            probe.close()
        release.set()
        names.join()
        deleting.join()

        assert errors == []
        assert store.get_artifact("d1", version) is None

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


def _worker_entry(name: str, enabled: bool = True) -> dict:
    return {
        "name": name,
        "backend": "executor",
        "config": {"url": "http://x:1"},
        "enabled": enabled,
    }


class TestTheNotebookWorkerRegistry:
    """The service-mode worker registry, shared by every node on the database."""

    def test_unset_then_written_then_emptied(self, store):
        assert store.notebook_worker_entries() is None

        store.update_notebook_workers(lambda current: [_worker_entry("a"), _worker_entry("b")])
        assert [e["name"] for e in store.notebook_worker_entries()] == ["a", "b"]

        store.update_notebook_workers(lambda current: [_worker_entry("b", False)])
        assert store.notebook_worker_entries() == [_worker_entry("b", False)]

        store.update_notebook_workers(lambda current: [])
        assert store.notebook_worker_entries() == []

    def test_an_unchanged_row_keeps_its_timestamps(self, store):
        store.update_notebook_workers(lambda current: [_worker_entry("a"), _worker_entry("b")])
        conn = store._get_connection()
        try:
            before = {
                row["name"]: row["updated_at"]
                for row in conn.execute("SELECT name, updated_at FROM notebook_workers")
            }
        finally:
            conn.close()

        store.update_notebook_workers(
            lambda current: [_worker_entry("a"), _worker_entry("b", False)]
        )

        conn = store._get_connection()
        try:
            after = {
                row["name"]: row["updated_at"]
                for row in conn.execute("SELECT name, updated_at FROM notebook_workers")
            }
        finally:
            conn.close()
        assert after["a"] == before["a"]
        assert after["b"] > before["b"]

    def test_a_raising_change_rolls_back(self, store):
        store.update_notebook_workers(lambda current: [_worker_entry("a")])

        def refuse(current):
            raise KeyError("missing")

        with pytest.raises(KeyError):
            store.update_notebook_workers(refuse)

        assert store.notebook_worker_entries() == [_worker_entry("a")]

    def test_two_nodes_on_one_database_see_each_others_writes(self, postgres_dsn, tmp_path, store):
        other_dialect = PostgresDialect(postgres_dsn)
        try:
            other_node = ArtifactStore(tmp_path / "node2", dialect=other_dialect)

            store.update_notebook_workers(lambda current: [_worker_entry("box")])
            assert other_node.notebook_worker_entries() == [_worker_entry("box")]

            other_node.update_notebook_workers(lambda current: [*current, _worker_entry("gpu")])
            assert [e["name"] for e in store.notebook_worker_entries()] == ["box", "gpu"]
        finally:
            other_dialect.close()


class TestSchemaMigrations:
    """Evolving a Postgres store that already holds data.

    ``_init_schema`` must carry new columns to a deployed database, not only to fresh ones.
    """

    def test_a_fresh_database_is_stamped_and_complete(self, store):
        from strata.artifact_store import _LATEST_SCHEMA_VERSION

        conn = store._get_connection()
        try:
            version = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()["v"]
            has_column = conn.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'artifact_versions' AND column_name = 'content_sha256'"
            ).fetchone()
        finally:
            conn.close()

        assert version == _LATEST_SCHEMA_VERSION
        assert has_column is not None

    def test_an_existing_database_is_carried_forward(self, postgres_dsn, tmp_path, store):
        """A new column reaches a store that already has rows."""
        from strata.artifact_store import _LATEST_SCHEMA_VERSION
        from strata.sql_backend import PostgresDialect

        version = store.create_artifact("keeper", "prov-keep", _spec())
        store.finalize_artifact("keeper", version, schema_json="", row_count=1, byte_size=1)

        # Rewind to a pre-migration database that still holds its rows.
        conn = store._get_connection()
        try:
            conn.execute("DROP TABLE schema_version")
            conn.execute("ALTER TABLE artifact_versions DROP COLUMN content_sha256")
            conn.commit()
        finally:
            conn.close()

        dialect = PostgresDialect(postgres_dsn)
        try:
            reopened = ArtifactStore(tmp_path / "artifacts", dialect=dialect)

            conn = reopened._get_connection()
            try:
                stamped = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()[
                    "v"
                ]
                has_column = conn.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'artifact_versions' AND column_name = 'content_sha256'"
                ).fetchone()
            finally:
                conn.close()

            assert has_column is not None, "the migration never reached an existing database"
            assert stamped == _LATEST_SCHEMA_VERSION
            assert reopened.get_artifact("keeper", version) is not None, "rows must survive"
        finally:
            dialect.close()

    def test_existing_publication_tokens_are_hashed_and_still_resolve(
        self, postgres_dsn, tmp_path, store
    ):
        from strata.artifact_store import publication_id
        from strata.sql_backend import PostgresDialect

        version = store.create_artifact("fig", "prov-fig", _spec())
        with store.open_blob_writer("fig", version) as writer:
            writer.write(b"figure bytes")
        store.finalize_artifact("fig", version, schema_json="", row_count=1, byte_size=12)
        publication = store.publish_artifact("fig", version, title="Figure")

        # Back to a store from before the hashing migration: raw tokens in the row and audit.
        conn = store._get_connection()
        try:
            conn.execute(
                "UPDATE artifact_publications SET token = ? WHERE token = ?",
                (publication.token, publication.id),
            )
            conn.execute(
                "UPDATE registry_audit SET value = ? WHERE value = ?",
                (publication.token, publication.id),
            )
            conn.execute("DELETE FROM schema_version WHERE version >= 8")
            conn.commit()
        finally:
            conn.close()
        assert store.get_publication(publication.token) is None

        dialect = PostgresDialect(postgres_dsn)
        try:
            reopened = ArtifactStore(tmp_path / "artifacts", dialect=dialect)

            found = reopened.get_publication(publication.token)
            assert found is not None and found.id == publication_id(publication.token)
            assert [e["value"] for e in reopened.read_events() if e["key"] == "token"] == [
                publication.id
            ]
            conn = reopened._get_connection()
            try:
                stored = [
                    row["token"]
                    for row in conn.execute("SELECT token FROM artifact_publications").fetchall()
                ]
            finally:
                conn.close()
            assert stored == [publication.id]
        finally:
            dialect.close()

    def test_a_read_only_open_leaves_an_older_database_alone(self, postgres_dsn, tmp_path, store):
        """A shared database older nodes still serve: inspecting it must not migrate it."""
        from strata.artifact_store import StoreSchemaMismatch

        conn = store._get_connection()
        try:
            conn.execute("DROP TABLE notebook_workers")
            conn.execute("DROP TABLE notebook_worker_registry")
            conn.execute("UPDATE schema_version SET version = 8")
            conn.commit()
        finally:
            conn.close()

        dialect = PostgresDialect(postgres_dsn)
        try:
            with pytest.raises(StoreSchemaMismatch, match="version 8, older"):
                ArtifactStore(tmp_path / "artifacts", dialect=dialect, read_only=True)

            conn = dialect.connect()
            try:
                stamped = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
                assert stamped["v"] == 8
                assert not dialect.schema_exists(conn, "notebook_workers")
            finally:
                conn.close()
        finally:
            dialect.close()

    def test_an_existing_database_gains_the_worker_registry(self, postgres_dsn, tmp_path, store):
        from strata.artifact_store import _LATEST_SCHEMA_VERSION

        conn = store._get_connection()
        try:
            conn.execute("DROP TABLE notebook_workers")
            conn.execute("DROP TABLE notebook_worker_registry")
            conn.execute("DELETE FROM schema_version WHERE version >= 9")
            conn.commit()
        finally:
            conn.close()

        dialect = PostgresDialect(postgres_dsn)
        try:
            reopened = ArtifactStore(tmp_path / "artifacts", dialect=dialect)
            reopened.update_notebook_workers(lambda current: [_worker_entry("box")])

            assert reopened.notebook_worker_entries() == [_worker_entry("box")]
            conn = reopened._get_connection()
            try:
                stamped = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            finally:
                conn.close()
            assert stamped["v"] == _LATEST_SCHEMA_VERSION
        finally:
            dialect.close()

    def test_column_exists_reports_truthfully(self, store):
        conn = store._get_connection()
        try:
            assert store._dialect.column_exists(conn, "artifact_versions", "content_sha256")
            assert not store._dialect.column_exists(conn, "artifact_versions", "nope")
        finally:
            conn.close()
