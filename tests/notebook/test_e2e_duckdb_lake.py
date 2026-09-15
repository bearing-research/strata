"""DuckDB SQL cells against a real Iceberg REST catalog and an S3 mount. Item 26.

Iceberg's REST catalog fixture and MinIO in containers. A DuckDB cell reads
``lake.taxi.trips`` at its current snapshot, a new snapshot makes the cell stale
and the next run reads it, and the query reads the snapshot its provenance
names even when the table moves in between.
"""

from __future__ import annotations

import io
import socket
from pathlib import Path
from typing import Any

import docker
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata.config import StrataConfig
from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus, MountSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    update_notebook_mounts,
    write_cell,
)
from tests.conftest import start_container_or_skip


def _docker_daemon_reachable() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


if not _docker_daemon_reachable():
    pytest.skip("Docker daemon is not running", allow_module_level=True)

from testcontainers.community.minio import MinioContainer  # noqa: E402
from testcontainers.core.container import DockerContainer  # noqa: E402
from testcontainers.core.waiting_utils import wait_for_logs  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def rest_catalog(tmp_path):
    """A REST catalog with ``taxi.trips`` (ids 1 and 2), and its properties."""
    from pyiceberg.catalog import load_catalog

    warehouse = tmp_path / "rest-warehouse"
    warehouse.mkdir()
    warehouse.chmod(0o777)
    # The catalog, root in the container, writes metadata where this process
    # writes data files; Hadoop creates directories 0755, so they exist first.
    for directory in ("taxi", "taxi/trips", "taxi/trips/data", "taxi/trips/metadata"):
        (warehouse / directory).mkdir()
        (warehouse / directory).chmod(0o777)
    port = _free_port()
    container = DockerContainer("apache/iceberg-rest-fixture:1.9.2")
    container.with_volume_mapping(str(warehouse), str(warehouse), "rw")
    container.with_env("CATALOG_WAREHOUSE", warehouse.as_uri())
    container.with_bind_ports(8181, port)
    start_container_or_skip(
        container,
        label="iceberg-rest-fixture",
        ready=lambda c: wait_for_logs(c, "Started", timeout=120),
    )
    try:
        properties = {"type": "rest", "uri": f"http://127.0.0.1:{port}"}
        catalog = load_catalog("lake", **properties)
        catalog.create_namespace("taxi")
        table = catalog.create_table("taxi.trips", schema=pa.schema([("id", pa.int64())]))
        table.append(pa.table({"id": [1, 2]}))
        yield catalog, properties
    finally:
        container.stop()


def _notebook(tmp_path: Path, source: str, connection: str) -> Path:
    nb_dir = create_notebook(tmp_path, "duckdb_lake_e2e")
    add_cell_to_notebook(nb_dir, "c1", language="sql")
    write_cell(nb_dir, "c1", source)
    toml = nb_dir / "notebook.toml"
    toml.write_text(toml.read_text() + f"\n[connections.lake]\n{connection}\n")
    return nb_dir


def _configure(monkeypatch, config: StrataConfig) -> None:
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)


async def _run(nb_dir: Path, session: NotebookSession) -> tuple[Any, list[dict[str, Any]]]:
    source = (nb_dir / "cells" / "c1.py").read_text()
    result = await CellExecutor(session).execute_cell("c1", source)
    assert result.success, result.error
    session.compute_staleness()
    session.mark_executed_ready("c1")
    art_id, version = result.artifact_uri.removeprefix("strata://artifact/").rsplit("@v=", 1)
    blob = session.get_artifact_manager().load_artifact_data(art_id, int(version))
    return result, pa.ipc.open_stream(blob).read_all().to_pylist()


@pytest.mark.asyncio
async def test_a_duckdb_cell_reads_the_catalog_and_goes_stale_on_a_new_snapshot(
    tmp_path, monkeypatch, rest_catalog
):
    catalog, properties = rest_catalog
    _configure(
        monkeypatch, StrataConfig(cache_dir=tmp_path / "cache", catalogs={"lake": properties})
    )
    nb_dir = _notebook(
        tmp_path,
        "# @sql connection=lake\nSELECT id FROM lake.taxi.trips ORDER BY id\n",
        'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"',
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    cell = session.notebook_state.get_cell("c1")
    first_snapshot = catalog.load_table("taxi.trips").current_snapshot().snapshot_id

    result, rows = await _run(nb_dir, session)
    assert rows == [{"id": 1}, {"id": 2}]
    assert [f.rsplit(":", 1)[1] for f in session._collect_table_fingerprints(cell)] == [
        str(first_snapshot)
    ]
    assert session.compute_staleness()["c1"].status == CellStatus.READY

    catalog.load_table("taxi.trips").append(pa.table({"id": [3]}))
    assert session.compute_staleness()["c1"].status != CellStatus.READY

    result, rows = await _run(nb_dir, session)
    assert result.cache_hit is False
    assert rows == [{"id": 1}, {"id": 2}, {"id": 3}]
    result, _ = await _run(nb_dir, session)
    assert result.cache_hit is True


def test_the_query_reads_the_snapshot_its_provenance_names(tmp_path, monkeypatch, rest_catalog):
    from strata.notebook.sql.analyzer import analyze_sql_cell
    from strata.notebook.sql.cell_executor import _execute_query
    from strata.notebook.sql.lake import resolve_lake
    from strata.notebook.sql.registry import get_adapter

    catalog, properties = rest_catalog
    _configure(
        monkeypatch, StrataConfig(cache_dir=tmp_path / "cache", catalogs={"lake": properties})
    )
    source = "# @sql connection=lake\nSELECT id FROM lake.taxi.trips ORDER BY id\n"
    nb_dir = _notebook(tmp_path, source, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    spec = session.notebook_state.connections[0]
    analysis = analyze_sql_cell(source, dialect="duckdb")

    lake = resolve_lake(session, "c1", source, spec, analysis.tables)
    catalog.load_table("taxi.trips").append(pa.table({"id": [3]}))
    table = _execute_query(get_adapter("duckdb"), lake.spec, analysis, (), lake=lake)

    assert table.column("id").to_pylist() == [1, 2]


@pytest.mark.asyncio
async def test_an_s3_mount_is_a_view_read_with_its_storage_options(tmp_path):
    with MinioContainer("quay.io/minio/minio:RELEASE.2024-11-07T00-52-20Z") as minio:
        minio_config = minio.get_config()
        client = minio.get_client()
        client.make_bucket("raw")
        buffer = io.BytesIO()
        pq.write_table(pa.table({"k": [4, 5]}), buffer)
        client.put_object(
            "raw", "events/part-0.parquet", io.BytesIO(buffer.getvalue()), len(buffer.getvalue())
        )
        nb_dir = _notebook(
            tmp_path,
            "# @sql connection=lake\nSELECT sum(k) AS total FROM events\n",
            'driver = "duckdb"\npath = ":memory:"\nmounts = ["events"]',
        )
        update_notebook_mounts(
            nb_dir,
            [
                MountSpec(
                    name="events",
                    uri="s3://raw/events",
                    options={
                        "endpoint_url": f"http://{minio_config['endpoint']}",
                        "key": minio_config["access_key"],
                        "secret": minio_config["secret_key"],
                    },
                )
            ],
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)

        _, rows = await _run(nb_dir, session)

        assert rows == [{"total": 9}]


@pytest.mark.asyncio
async def test_a_read_cell_cannot_write_to_the_catalog(tmp_path, monkeypatch, rest_catalog):
    catalog, properties = rest_catalog
    _configure(
        monkeypatch, StrataConfig(cache_dir=tmp_path / "cache", catalogs={"lake": properties})
    )
    source = "# @sql connection=lake\nCOMMIT; INSERT INTO lake.taxi.trips VALUES (9)\n"
    nb_dir = _notebook(tmp_path, source, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await CellExecutor(session).execute_cell("c1", source)

    assert result.success is False
    rows = catalog.load_table("taxi.trips").scan().to_arrow().column("id").to_pylist()
    assert sorted(rows) == [1, 2]
