"""Gate 1 (ADR 002 §8) with the notebook's ledger on Postgres.

The same cell runs as ``test_local_runs``, their metadata in a real Postgres server
and their blobs under the notebook directory. Requires Docker and the ``postgres``
extra; the unit job ignores this file and the integration workflow runs it.
"""

from __future__ import annotations

import sys

import docker
import pytest
from testcontainers.community.postgres import PostgresContainer
from vestigo.sql_backend import PostgresDialect

from tests.notebook import test_local_runs


def _docker_daemon_reachable() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


if not _docker_daemon_reachable():
    pytest.skip("Docker daemon is not running", allow_module_level=True)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform == "win32", reason="no event pipe on Windows"),
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
    monkeypatch.setattr(test_local_runs, "_dialect", lambda: dialect)
    yield
    dialect.close()


class TestGate1OnPostgres(test_local_runs.TestGate1):
    """All six conditions, on the second backend."""
