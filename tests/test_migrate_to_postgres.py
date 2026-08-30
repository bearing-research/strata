"""Migrating a populated SQLite artifact store onto Postgres.

Driven against a real server, because the things that break a migration are
things SQLite does not have: enforced foreign keys, a sequence standing beside
a generated key, and narrower column types.
"""

from __future__ import annotations

import docker
import pytest
from testcontainers.postgres import PostgresContainer

from strata.artifact_store import ArtifactStore, TransformSpec
from strata.migrate import MIGRATED_TABLES, migrate, plan_migration
from strata.sql_backend import PostgresDialect, SqliteDialect


def _docker_daemon_reachable() -> bool:
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
    with PostgresContainer("postgres:16-alpine") as container:
        yield container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")


def _spec() -> TransformSpec:
    return TransformSpec(executor="duckdb_sql_v1", params={"sql": "SELECT 1"}, inputs=[])


@pytest.fixture
def populated_sqlite(tmp_path):
    """A SQLite store with an artifact, a name, an alias, a tag, and audit rows."""
    store = ArtifactStore(tmp_path / "src")
    version = store.create_artifact("a1", "prov-1", _spec())
    store.write_blob("a1", version, b"payload")
    store.finalize_artifact("a1", version, '{"f": []}', row_count=3, byte_size=7)
    store.set_name("daily", "a1", version)
    store.set_alias("daily", "champion", "a1", version)
    store.set_tag("a1", version, "auc", "0.91")
    return store


@pytest.fixture
def target(postgres_dsn, tmp_path):
    """An empty Postgres store with the schema already created.

    Mirrors the documented flow: boot Strata against the target once so the
    stores create their own schema, then migrate.
    """
    dialect = PostgresDialect(postgres_dsn)
    conn = dialect.connect()
    try:
        conn.executescript(
            "DROP TABLE IF EXISTS artifact_builds, api_keys, stream_owners, "
            "artifact_versions, artifact_names, artifact_aliases, artifact_tags, "
            "registry_audit, registry_pending CASCADE;"
        )
        conn.commit()
    finally:
        conn.close()

    ArtifactStore(tmp_path / "tgt", dialect=dialect)
    yield dialect
    dialect.close()


class TestPlanning:
    def test_plan_reports_both_sides_without_writing(self, populated_sqlite, target, tmp_path):
        plan = plan_migration(SqliteDialect(tmp_path / "src" / "artifacts.sqlite"), target)

        assert plan.source_counts["artifact_versions"] == 1
        assert plan.source_counts["artifact_names"] == 1
        assert plan.target_is_empty
        assert plan.total_rows > 0

    def test_plan_names_tables_the_target_lacks(self, populated_sqlite, postgres_dsn, tmp_path):
        # A database Strata has never booted against has no schema, and the
        # migration must say so rather than invent the DDL from a second place.
        dialect = PostgresDialect(postgres_dsn)
        try:
            conn = dialect.connect()
            conn.executescript("DROP TABLE IF EXISTS artifact_versions CASCADE;")
            conn.commit()
            conn.close()

            plan = plan_migration(SqliteDialect(tmp_path / "src" / "artifacts.sqlite"), dialect)
            assert "artifact_versions" in plan.missing_in_target
        finally:
            dialect.close()


class TestMigration:
    def _source(self, tmp_path):
        return SqliteDialect(tmp_path / "src" / "artifacts.sqlite")

    def test_artifacts_names_aliases_and_tags_all_arrive(self, populated_sqlite, target, tmp_path):
        result = migrate(self._source(tmp_path), target)
        assert result.total_copied > 0

        migrated = ArtifactStore(tmp_path / "tgt", dialect=target)
        artifact = migrated.get_latest_version("a1")
        assert artifact is not None
        assert artifact.row_count == 3
        assert artifact.byte_size == 7

        assert migrated.resolve_name("daily").id == "a1"
        assert migrated.resolve_alias("daily", "champion").id == "a1"
        assert migrated.get_tags("a1", artifact.version)["auc"] == "0.91"

    def test_timestamps_survive_the_move(self, populated_sqlite, target, tmp_path):
        original = populated_sqlite.get_latest_version("a1")
        migrate(self._source(tmp_path), target)

        migrated = ArtifactStore(tmp_path / "tgt", dialect=target).get_latest_version("a1")
        # Would fail under a single-precision column: created_at is an epoch
        # value and rounds to whole minutes in 32 bits.
        assert migrated.created_at == original.created_at

    def test_rerunning_copies_nothing_twice(self, populated_sqlite, target, tmp_path):
        first = migrate(self._source(tmp_path), target)
        second = migrate(self._source(tmp_path), target, allow_nonempty_target=True)

        # Idempotent, so an interrupted migration can simply be run again.
        assert second.total_copied == 0
        assert second.total_skipped == first.total_copied

    def test_a_populated_target_is_refused_by_default(self, populated_sqlite, target, tmp_path):
        migrate(self._source(tmp_path), target)
        with pytest.raises(ValueError, match="already holds rows"):
            migrate(self._source(tmp_path), target)

    def test_the_audit_sequence_is_moved_past_the_migrated_rows(
        self, populated_sqlite, target, tmp_path
    ):
        # registry_audit.seq is BIGSERIAL on Postgres. Inserting explicit ids
        # does not advance the sequence, so without a resync the next audited
        # mutation collides on the primary key.
        migrate(self._source(tmp_path), target)

        migrated = ArtifactStore(tmp_path / "tgt", dialect=target)
        before = len(migrated.read_audit())
        migrated.set_alias("daily", "candidate", "a1", 1)
        assert len(migrated.read_audit()) == before + 1

    def test_the_audit_sequence_is_resynced_on_a_resumed_run(
        self, populated_sqlite, target, tmp_path
    ):
        # A run killed between the last commit and the resync leaves the rows
        # in place with the sequence still at 1. The resumed run copies
        # nothing, so gating the resync on "did this run copy" would skip it
        # forever and reintroduce the collision.
        migrate(self._source(tmp_path), target)
        conn = target.connect()
        try:
            # Put the sequence back where an interrupted run would have left it.
            conn.execute("SELECT setval(pg_get_serial_sequence('registry_audit','seq'), 1, false)")
            conn.commit()
        finally:
            conn.close()

        second = migrate(self._source(tmp_path), target, allow_nonempty_target=True)
        assert second.total_copied == 0

        migrated = ArtifactStore(tmp_path / "tgt", dialect=target)
        before = len(migrated.read_audit())
        migrated.set_alias("daily", "candidate", "a1", 1)
        assert len(migrated.read_audit()) == before + 1

    def test_a_dangling_build_row_is_reported_not_fatal(self, populated_sqlite, target, tmp_path):
        # garbage_collect, cleanup_failed, and delete_artifact all remove
        # artifact_versions rows without touching artifact_builds, so a real
        # store accumulates references Postgres will refuse and SQLite never
        # checked. One bad row must not strand the whole migration.
        from strata.transforms.build_store import BuildStore

        builds = BuildStore(tmp_path / "src" / "artifacts.sqlite")
        builds.create_build(
            build_id="orphan",
            artifact_id="does-not-exist",
            version=1,
            executor_ref="duckdb_sql_v1",
        )
        # The target needs the table too: a deployment with builds has booted
        # its build store on both sides.
        BuildStore(tmp_path / "tgt" / "artifacts.sqlite", dialect=target)

        result = migrate(self._source(tmp_path), target)

        assert result.total_copied > 0, "the good rows still moved"
        assert result.total_rejected == 1
        table, _key, _reason = result.rejected[0]
        assert table == "artifact_builds"
        # And the artifact itself arrived regardless.
        assert ArtifactStore(tmp_path / "tgt", dialect=target).get_latest_version("a1") is not None

    def test_a_target_missing_only_unused_tables_is_fine(
        self, populated_sqlite, postgres_dsn, tmp_path
    ):
        # The documented flow: boot the artifact store against the target.
        # Personal mode forbids auth_mode='api_key', so api_keys is never
        # created -- and artifact_builds only appears when the build store is
        # constructed. Refusing on those made the documented path exit 1.
        dialect = PostgresDialect(postgres_dsn)
        try:
            conn = dialect.connect()
            conn.executescript(
                "DROP TABLE IF EXISTS artifact_builds, api_keys, artifact_versions, "
                "artifact_names, artifact_aliases, artifact_tags, registry_audit, "
                "registry_pending CASCADE;"
            )
            conn.commit()
            conn.close()
            ArtifactStore(tmp_path / "tgt2", dialect=dialect)

            plan = plan_migration(self._source(tmp_path), dialect)
            assert "api_keys" in plan.missing_in_target
            assert plan.blocking_tables == [], "no source rows need those tables"

            result = migrate(self._source(tmp_path), dialect)
            assert result.total_copied > 0
        finally:
            dialect.close()

    def test_a_blocking_table_is_refused_before_anything_is_written(
        self, populated_sqlite, postgres_dsn, tmp_path
    ):
        # Discovering this mid-loop would leave earlier tables committed, so
        # the retry would then need allow_nonempty_target for a mistake the
        # caller never made.
        dialect = PostgresDialect(postgres_dsn)
        try:
            conn = dialect.connect()
            conn.executescript(
                "DROP TABLE IF EXISTS artifact_builds, api_keys, artifact_versions, "
                "artifact_names, artifact_aliases, artifact_tags, registry_audit, "
                "registry_pending CASCADE;"
            )
            conn.commit()
            conn.close()
            ArtifactStore(tmp_path / "tgt3", dialect=dialect)

            # Drop the one table the source definitely has rows for.
            conn = dialect.connect()
            conn.executescript("DROP TABLE artifact_versions CASCADE;")
            conn.commit()
            conn.close()

            with pytest.raises(ValueError, match="missing"):
                migrate(self._source(tmp_path), dialect)

            # Nothing was written on the way to that refusal.
            conn = dialect.connect()
            try:
                remaining = conn.execute("SELECT COUNT(*) FROM artifact_names").fetchone()[0]
            finally:
                conn.close()
            assert remaining == 0
        finally:
            dialect.close()

    def test_a_rejected_row_does_not_take_its_batch_with_it(
        self, populated_sqlite, target, tmp_path
    ):
        """The worst failure this tool can have: reporting success while losing rows.

        Without a per-row savepoint, the rollback for one bad row discards
        every uncommitted insert in the same batch, and ``copied`` has already
        counted them. Measured at 99 of 100 lost, reported as copied.

        The earlier dangling-build test used a single row -- the rejected one
        -- so it could not see this.
        """
        from strata.transforms.build_store import BuildStore

        builds = BuildStore(tmp_path / "src" / "artifacts.sqlite")
        version = populated_sqlite.get_latest_version("a1").version
        for i in range(99):
            builds.create_build(
                build_id=f"good-{i}", artifact_id="a1", version=version, executor_ref="x"
            )
        builds.create_build(
            build_id="orphan", artifact_id="does-not-exist", version=1, executor_ref="x"
        )
        BuildStore(tmp_path / "tgt" / "artifacts.sqlite", dialect=target)

        result = migrate(self._source(tmp_path), target)

        conn = target.connect()
        try:
            actual = conn.execute("SELECT COUNT(*) FROM artifact_builds").fetchone()[0]
        finally:
            conn.close()

        assert result.total_rejected == 1
        assert result.copied["artifact_builds"] == 99
        # The count reported has to be the count that arrived.
        assert actual == result.copied["artifact_builds"]

    def test_a_value_postgres_refuses_is_rejected_not_fatal(
        self, populated_sqlite, target, tmp_path
    ):
        # A NUL byte inside TEXT: SQLite stores it, Postgres refuses it. That
        # is a DataError, not an IntegrityError, so catching only the latter
        # let it escape and abort the run with earlier tables committed.
        conn = SqliteDialect(tmp_path / "src" / "artifacts.sqlite").connect()
        try:
            conn.execute(
                "INSERT INTO artifact_tags (tenant, artifact_id, version, key, value, updated_at) "
                "VALUES ('', 'a1', 1, 'bad', ?, 1.0)",
                ("has\x00nul",),
            )
            conn.commit()
        finally:
            conn.close()

        result = migrate(self._source(tmp_path), target)
        assert result.total_rejected == 1
        assert result.rejected[0][0] == "artifact_tags"
        # And the run still finished the tables after it.
        assert ArtifactStore(tmp_path / "tgt", dialect=target).get_latest_version("a1") is not None

    def test_stream_owners_is_not_migrated(self):
        # It records which node serves a live stream; none survive the move.
        assert "stream_owners" not in {table for table, _ in MIGRATED_TABLES}

    def test_artifact_versions_is_copied_before_its_dependents(self):
        # Postgres enforces the foreign keys from artifact_builds and
        # artifact_names into artifact_versions; SQLite never did.
        order = [table for table, _ in MIGRATED_TABLES]
        assert order.index("artifact_versions") < order.index("artifact_names")
        assert order.index("artifact_versions") < order.index("artifact_builds")
