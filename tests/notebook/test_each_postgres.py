"""``# @each`` with the notebook's ledger on Postgres.

The same cells as ``test_each``, their metadata in a real Postgres server and their blobs
under the notebook directory. Requires Docker and the ``postgres`` extra; the unit job
ignores this file and the integration workflow runs it.
"""

from __future__ import annotations

import sys

import pytest
from testcontainers.community.postgres import PostgresContainer
from vestigo.sql_backend import PostgresDialect

from tests.docker_helpers import docker_daemon_reachable
from tests.notebook import test_each

if not docker_daemon_reachable():
    pytest.skip("Docker daemon is not running", allow_module_level=True)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform == "win32", reason="pyiceberg's local warehouse paths"),
]


@pytest.fixture(scope="module")
def postgres_dsn():
    with PostgresContainer("postgres:18-alpine") as container:
        yield container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")


@pytest.fixture(autouse=True)
def ledger_on_postgres(postgres_dsn, monkeypatch):
    """Each test opens its notebook's ledger on a clean Postgres schema, one pool per test."""
    # The integration job's Iceberg catalog must not claim the test warehouse's tables.
    monkeypatch.delenv("STRATA_CATALOG_URI", raising=False)
    dialect = PostgresDialect(postgres_dsn)
    conn = dialect.connect()
    try:
        conn.executescript("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(test_each, "_dialect", lambda: dialect)

    def configured_store(artifact_dir):
        # --artifact-dir names a SQLite store; the CLI reaches this one by its config.
        monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(artifact_dir))
        monkeypatch.setenv("STRATA_ARTIFACT_METADATA_DSN", postgres_dsn)
        return None

    monkeypatch.setattr(test_each, "_cli_store", configured_store)
    yield
    dialect.close()


class TestFilesOnPostgres(test_each.TestFiles):
    pass


class TestVariablesOnPostgres(test_each.TestVariables):
    pass


class TestFailureOnPostgres(test_each.TestFailure):
    pass


class TestRetentionOnPostgres(test_each.TestRetention):
    pass


class TestSyncOnPostgres(test_each.TestSync):
    pass


class TestASnapshotOnPostgres(test_each.TestASnapshot):
    pass


class TestAnotherStoreOnPostgres(test_each.TestAnotherStore):
    pass


class TestOtherSurfacesOnPostgres(test_each.TestOtherSurfaces):
    pass


class TestTheSweepDuringARunOnPostgres(test_each.TestTheSweepDuringARun):
    pass


class TestDisplaysOnPostgres(test_each.TestDisplays):
    pass
