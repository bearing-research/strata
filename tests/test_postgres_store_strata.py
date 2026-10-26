"""The artifact store running on Postgres instead of SQLite.

Exercises the same ``ArtifactStore`` methods personal mode uses against a real server, so the port
is proven by behaviour rather than rendered SQL. Requires Docker and the ``postgres`` extra.
"""

from __future__ import annotations

import time

import docker
import pytest
from testcontainers.community.postgres import PostgresContainer
from vestigo.provenance import TransformSpec
from vestigo.sql_backend import PostgresDialect
from vestigo.store import ArtifactStore

from strata.artifact.cli import _server_store as _configured_server_store
from strata.notebook.worker_store import NotebookWorkerStore


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
            "notebook_workers, notebook_worker_registry, store_upgrades, nodes, node_inputs, "
            "schema_version CASCADE;"
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


def _workers(store):
    """Strata's worker registry store on the same database as *store*."""
    return NotebookWorkerStore(store.db_path, dialect=store.dialect)


def _worker_entry(name: str, enabled: bool = True) -> dict:
    return {
        "name": name,
        "backend": "executor",
        "config": {"url": "http://x:1"},
        "enabled": enabled,
    }


class TestConfigurationSelectsTheBackend:
    """The wiring end to end: a DSN in config produces a Postgres-backed store."""

    def test_a_configured_dsn_produces_a_postgres_backed_store(self, postgres_dsn, tmp_path):
        from vestigo.store import get_artifact_store, reset_artifact_store

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
        from vestigo.store import get_artifact_store, reset_artifact_store

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


class TestTheCliOnAPostgresStore:
    """With a service store's settings, the artifact CLI must open Postgres, not an empty SQLite
    file.
    """

    def test_archive_by_token_writes_the_zip_the_route_serves(
        self, postgres_dsn, tmp_path, monkeypatch
    ):
        import argparse

        from strata.api.publication_bundle import bundle_zip
        from strata.artifact.cli import cmd_archive

        dialect = PostgresDialect(postgres_dsn)
        try:
            conn = dialect.connect()
            conn.executescript(
                "DROP TABLE IF EXISTS artifact_builds, artifact_versions, artifact_names, "
                "artifact_aliases, artifact_tags, artifact_publications, retention_roots, "
                "retention_edges, "
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

        from strata.artifact.cli import cmd_publish

        version = store.create_artifact("fig", "prov-fig", _spec())
        with store.open_blob_writer("fig", version) as writer:
            writer.write(b"figure bytes")
        store.finalize_artifact("fig", version, schema_json="", row_count=1, byte_size=12)

        monkeypatch.setenv("STRATA_ARTIFACT_METADATA_DSN", postgres_dsn)
        monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
        # The suite stubs this out so no test publishes into a developer's own store.
        monkeypatch.setattr("strata.artifact.cli._server_store", _configured_server_store)
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


class TestTheNotebookWorkerRegistry:
    """The service-mode worker registry, shared by every node on the database."""

    def test_unset_then_written_then_emptied(self, store):
        assert _workers(store).entries() is None

        _workers(store).update(lambda current: [_worker_entry("a"), _worker_entry("b")])
        assert [e["name"] for e in _workers(store).entries()] == ["a", "b"]

        _workers(store).update(lambda current: [_worker_entry("b", False)])
        assert _workers(store).entries() == [_worker_entry("b", False)]

        _workers(store).update(lambda current: [])
        assert _workers(store).entries() == []

    def test_an_unchanged_row_keeps_its_timestamps(self, store):
        _workers(store).update(lambda current: [_worker_entry("a"), _worker_entry("b")])
        conn = store._get_connection()
        try:
            before = {
                row["name"]: row["updated_at"]
                for row in conn.execute("SELECT name, updated_at FROM notebook_workers")
            }
        finally:
            conn.close()

        _workers(store).update(lambda current: [_worker_entry("a"), _worker_entry("b", False)])

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
        _workers(store).update(lambda current: [_worker_entry("a")])

        def refuse(current):
            raise KeyError("missing")

        with pytest.raises(KeyError):
            _workers(store).update(refuse)

        assert _workers(store).entries() == [_worker_entry("a")]

    def test_two_nodes_on_one_database_see_each_others_writes(self, postgres_dsn, tmp_path, store):
        other_dialect = PostgresDialect(postgres_dsn)
        try:
            other_node = ArtifactStore(tmp_path / "node2", dialect=other_dialect)

            _workers(store).update(lambda current: [_worker_entry("box")])
            assert _workers(other_node).entries() == [_worker_entry("box")]

            _workers(other_node).update(lambda current: [*current, _worker_entry("gpu")])
            assert [e["name"] for e in _workers(store).entries()] == ["box", "gpu"]
        finally:
            other_dialect.close()


class TestGarbageCollection:
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

    def test_a_published_folder_keeps_its_files(self, store, tmp_path):
        # The walk follows the edges the import wrote; run it on Postgres.
        from strata.artifact.transfer import copy_chain
        from tests.test_folder_transfer import FILES, _folder, _rebuilt

        manager, folder = _folder(tmp_path / "alice", FILES)
        copy_chain(manager.artifact_store, store, folder, 10)
        store.publish_artifact(folder.id, folder.version)

        assert store.garbage_collect(max_idle_days=0, collect_latest=True)["deleted_count"] == 0
        assert _rebuilt(store, folder.id, folder.version, tmp_path / "out") == FILES

    def test_a_folder_goes_whole_or_stays_whole(self, store):
        # The requirement join and the wholeness query, on Postgres.
        from tests.test_artifact_retention import _team_folder

        idle, idle_files = _team_folder(store, tenant="a")
        used, used_files = _team_folder(store, tenant="a")
        for artifact_id, _ in (idle, *idle_files, *used_files):
            self._last_used(store, artifact_id, 40 * 86400)
        self._last_used(store, used[0], 86400)

        dry = store.garbage_collect(max_idle_days=30, dry_run=True)
        store.garbage_collect(max_idle_days=30)

        reported = {(entry["artifact_id"], entry["version"]) for entry in dry["collected"]}
        assert reported == {idle, *idle_files}
        assert all(store.get_artifact(*key) is None for key in reported)
        assert store.requirements_met(store.get_artifact(*used))
        store.delete_artifact(*used_files[0])
        assert not store.requirements_met(store.get_artifact(*used))


class TestSchemaMigrations:
    """Evolving a Postgres store that already holds data.

    ``_init_schema`` must carry new columns to a deployed database, not only to fresh ones.
    """

    def test_an_existing_database_gains_the_worker_registry(self, postgres_dsn, tmp_path, store):
        from vestigo.store import _LATEST_SCHEMA_VERSION

        conn = store._get_connection()
        try:
            conn.execute("DELETE FROM schema_version WHERE version >= 9")
            conn.commit()
        finally:
            conn.close()

        dialect = PostgresDialect(postgres_dsn)
        try:
            # Migration 9 is a tombstone: Strata's worker store creates the tables itself.
            reopened = ArtifactStore(tmp_path / "artifacts", dialect=dialect)
            _workers(reopened).update(lambda current: [_worker_entry("box")])

            assert _workers(reopened).entries() == [_worker_entry("box")]
            conn = reopened._get_connection()
            try:
                stamped = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            finally:
                conn.close()
            assert stamped["v"] == _LATEST_SCHEMA_VERSION
        finally:
            dialect.close()
