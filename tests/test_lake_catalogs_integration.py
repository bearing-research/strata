"""Scanning tables from a REST catalog, a Glue catalog and GCS and Azure warehouses.

Uses containers for the REST catalog, fake-gcs-server and Azurite, and moto Glue over MinIO data.
Each test reads a table by catalog name, as ``@table`` and a scan do, or by a warehouse URI, and
reads a pinned snapshot.
"""

from __future__ import annotations

import io
import shutil
import socket
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

import docker
import pyarrow as pa
import pytest

from strata.config import StrataConfig
from strata.fetcher import create_fetcher
from strata.notebook.models import TableSpec
from strata.notebook.tables import resolve_table_snapshot
from strata.planner import ReadPlanner
from tests.conftest import MINIO_IMAGE, start_container_or_skip


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

SCHEMA = pa.schema([("id", pa.int64())])


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _rows(config: StrataConfig, uri: str, snapshot_id: int | None = None) -> list[int]:
    s3 = config.get_s3_filesystem() if config.s3_endpoint_url else None
    plan = ReadPlanner(config).plan(uri, snapshot_id=snapshot_id)
    fetcher = create_fetcher(s3_filesystem=s3)
    return sorted(v for task in plan.tasks for v in fetcher.fetch(task).column("id").to_pylist())


def _two_snapshots(catalog) -> int:
    """``taxi.trips`` with ids 1, 2 and then 3; the first snapshot's id."""
    catalog.create_namespace("taxi")
    table = catalog.create_table("taxi.trips", schema=SCHEMA)
    table.append(pa.table({"id": [1, 2]}))
    first = catalog.load_table("taxi.trips").current_snapshot().snapshot_id
    catalog.load_table("taxi.trips").append(pa.table({"id": [3]}))
    return first


def _reads_by_name_and_pin(config: StrataConfig, name: str, catalog, first: int) -> None:
    uri = f"{name}:taxi.trips"
    current = catalog.load_table("taxi.trips").current_snapshot().snapshot_id

    assert resolve_table_snapshot(TableSpec(name="trips", uri=uri), config) == current
    assert _rows(config, uri) == [1, 2, 3]
    pinned = TableSpec(name="trips", uri=uri, snapshot_pin=first)
    assert _rows(config, uri, snapshot_id=resolve_table_snapshot(pinned, config)) == [1, 2]


class RestCatalog(NamedTuple):
    catalog: Any
    properties: dict[str, str]
    container: Any  # the testcontainers container, for joining its network
    warehouse: Path


@pytest.fixture
def rest_catalog(tmp_path):
    """Iceberg's REST catalog fixture: the catalog, its properties, container and warehouse."""
    from pyiceberg.catalog import load_catalog

    warehouse = tmp_path / "rest-warehouse"
    warehouse.mkdir()
    warehouse.chmod(0o777)
    port = _free_port()
    container = DockerContainer("apache/iceberg-rest-fixture:1.9.2")
    # Same path inside and out: the catalog writes metadata files where the client,
    # and then the scan, reads them.
    container.with_volume_mapping(str(warehouse), str(warehouse), "rw")
    container.with_env("CATALOG_WAREHOUSE", warehouse.as_uri())
    # The catalog (root in the container) writes table metadata where this process
    # then writes data files. Hadoop creates directories 0755 whatever the umask,
    # so create them open first. (Running the container as this uid fails: Hadoop
    # cannot log in a user the image has no name for.)
    for directory in ("taxi", "taxi/trips", "taxi/trips/data", "taxi/trips/metadata"):
        (warehouse / directory).mkdir()
        (warehouse / directory).chmod(0o777)
    container.with_bind_ports(8181, port)
    start_container_or_skip(
        container,
        label="iceberg-rest-fixture",
        ready=lambda c: wait_for_logs(c, "Started", timeout=120),
    )
    try:
        properties = {"type": "rest", "uri": f"http://127.0.0.1:{port}"}
        yield RestCatalog(load_catalog("rest", **properties), properties, container, warehouse)
    finally:
        container.stop()


def test_a_rest_catalog_table_is_read_by_name(tmp_path, rest_catalog):
    catalog, properties, _, _ = rest_catalog
    first = _two_snapshots(catalog)
    config = StrataConfig(cache_dir=tmp_path / "cache", catalogs={"rest": properties})

    _reads_by_name_and_pin(config, "rest", catalog, first)


# DuckDB deletes merge-on-read: a positional delete file on v2, a deletion vector
# (Puffin) on v3. Its manifests leave snapshot ids and sequence numbers to be
# inherited, which pyiceberg before 0.12 misread, silently dropping the deletes.
@pytest.mark.parametrize(
    ("format_version", "delete_format"),
    [("2", "PARQUET"), ("3", "PUFFIN")],
    ids=["v2-position-deletes", "v3-deletion-vector"],
)
def test_rows_another_engine_deleted_are_not_scanned(
    tmp_path, rest_catalog, format_version, delete_format
):
    import duckdb

    catalog, properties, _, _ = rest_catalog
    catalog.create_namespace("taxi")
    catalog.create_table("taxi.trips", schema=SCHEMA, properties={"format-version": format_version})
    conn = duckdb.connect()
    conn.execute("INSTALL iceberg; LOAD iceberg")
    conn.execute(
        f"ATTACH 'warehouse' AS lake "
        f"(TYPE iceberg, ENDPOINT '{properties['uri']}', AUTHORIZATION_TYPE 'none')"
    )
    conn.execute("INSERT INTO lake.taxi.trips SELECT range FROM range(10)")
    before = catalog.load_table("taxi.trips").current_snapshot().snapshot_id
    conn.execute("DELETE FROM lake.taxi.trips WHERE id IN (2, 5)")

    (task,) = catalog.load_table("taxi.trips").scan().plan_files()
    assert [d.file_format.value for d in task.delete_files] == [delete_format]
    config = StrataConfig(cache_dir=tmp_path / "cache", catalogs={"rest": properties})
    assert _rows(config, "rest:taxi.trips") == [0, 1, 3, 4, 6, 7, 8, 9]
    assert _rows(config, "rest:taxi.trips", snapshot_id=before) == list(range(10))


def test_a_glue_catalog_table_is_read_by_name(tmp_path, monkeypatch):
    from moto import mock_aws
    from pyiceberg.catalog import load_catalog

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with MinioContainer(MINIO_IMAGE) as minio:
        minio_config = minio.get_config()
        endpoint = f"http://{minio_config['endpoint']}"
        minio.get_client().make_bucket("lake")
        properties = {
            "type": "glue",
            "glue.region": "us-east-1",
            "warehouse": "s3://lake/glue",
            "s3.endpoint": endpoint,
            "s3.region": "us-east-1",
            "s3.access-key-id": minio_config["access_key"],
            "s3.secret-access-key": minio_config["secret_key"],
        }
        with mock_aws():
            catalog = load_catalog("glue", **properties)
            first = _two_snapshots(catalog)
            # No Strata S3 settings: data files are read with the credentials the catalog
            # gave the table.
            config = StrataConfig(cache_dir=tmp_path / "cache", catalogs={"glue": properties})

            _reads_by_name_and_pin(config, "glue", catalog, first)


@pytest.fixture
def fake_gcs():
    """fake-gcs-server with a ``lake`` bucket; its endpoint."""
    port = _free_port()
    container = DockerContainer("fsouza/fake-gcs-server:1.52.2")
    # Resumable uploads redirect to the external URL, so it must be the address this
    # process reaches the container on.
    container.with_command(f"-scheme http -port 4443 -external-url http://127.0.0.1:{port}")
    container.with_bind_ports(4443, port)
    start_container_or_skip(
        container, label="fake-gcs-server", ready=lambda c: wait_for_logs(c, "server started at")
    )
    try:
        endpoint = f"http://127.0.0.1:{port}"
        urllib.request.urlopen(
            urllib.request.Request(
                f"{endpoint}/storage/v1/b",
                data=b'{"name": "lake"}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        ).read()
        yield endpoint
    finally:
        container.stop()


def _gcs_token() -> dict[str, str]:
    """A bearer token for fake-gcs-server, which accepts any."""
    import pyiceberg.io as io

    return {
        io.GCS_TOKEN: "emulator",
        io.GCS_TOKEN_EXPIRES_AT_MS: str(int((time.time() + 86400) * 1000)),
    }


def test_a_gcs_warehouse_scans(tmp_path, fake_gcs):
    import pyiceberg.io as io
    from pyiceberg.catalog.sql import SqlCatalog

    properties = {
        "type": "sql",
        "uri": f"sqlite:///{tmp_path / 'catalog.db'}",
        "warehouse": "gs://lake/wh",
        io.GCS_SERVICE_HOST: fake_gcs,
        **_gcs_token(),
    }
    catalog = SqlCatalog("lake", **{k: v for k, v in properties.items() if k != "type"})
    first = _two_snapshots(catalog)
    files = [task.file.file_path for task in catalog.load_table("taxi.trips").scan().plan_files()]
    assert all(path.startswith("gs://") for path in files)
    config = StrataConfig(
        cache_dir=tmp_path / "cache",
        catalogs={"lake": properties},
        gcs_anonymous=True,
        gcs_endpoint_override=fake_gcs,
    )

    _reads_by_name_and_pin(config, "lake", catalog, first)


def _reads_request_warehouse(config: StrataConfig, uri: str, first: int) -> None:
    assert config.deployment_mode == "personal" and "uri" not in config.catalog_properties
    assert _rows(config, uri) == [1, 2, 3]
    assert _rows(config, uri, snapshot_id=first) == [1, 2]


def test_a_gcs_request_warehouse_reads_with_the_gcs_settings(tmp_path, fake_gcs):
    """A ``gs://`` URI with no catalog uri: its metadata is read at the configured endpoint."""
    import pyiceberg.io as io
    from pyiceberg.catalog.sql import SqlCatalog

    meta = tmp_path / "meta.sqlite"
    writer = SqlCatalog(
        "strata",
        uri=f"sqlite:///{meta}",
        warehouse="gs://lake/wh",
        **{io.GCS_SERVICE_HOST: fake_gcs, **_gcs_token()},
    )
    first = _two_snapshots(writer)
    config = StrataConfig(
        cache_dir=tmp_path / "cache",
        metadata_db=meta,
        gcs_anonymous=True,
        gcs_endpoint_override=fake_gcs,
        # pyiceberg's GCS FileIO has no anonymous mode, so the catalog sends a token.
        catalog_properties=_gcs_token(),
    )

    _reads_request_warehouse(config, "gs://lake/wh#taxi.trips", first)


@pytest.fixture
def azurite():
    """Azurite with a ``lake`` container."""
    from azure.storage.blob import BlobServiceClient
    from testcontainers.community.azurite import AzuriteContainer

    # :latest because older Azurite rejects the installed SDK's API version.
    container = start_container_or_skip(
        AzuriteContainer("mcr.microsoft.com/azure-storage/azurite:latest"), label="Azurite"
    )
    try:
        connection_string = container.get_connection_string()
        BlobServiceClient.from_connection_string(connection_string).create_container("lake")
        yield container
    finally:
        container.stop()


def _azure_writer(tmp_path, azurite, warehouse: str):
    """A SQL catalog in ``tmp_path`` writing ``warehouse`` through adlfs."""
    from pyiceberg.catalog.sql import SqlCatalog

    return SqlCatalog(
        "strata",
        uri=f"sqlite:///{tmp_path / 'meta.sqlite'}",
        warehouse=warehouse,
        **{"adls.connection-string": azurite.get_connection_string()},
    )


def _azure_config(tmp_path, azurite, **settings: Any) -> StrataConfig:
    blob_port = azurite.get_exposed_port(10000)
    return StrataConfig(
        cache_dir=tmp_path / "cache",
        metadata_db=tmp_path / "meta.sqlite",
        azure_account_name=azurite.account_name,
        azure_account_key=azurite.account_key,
        azure_endpoint_url=f"http://{azurite.get_container_host_ip()}:{blob_port}",
        **settings,
    )


_PYARROW_FILE_IO = {"catalog_properties": {"py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO"}}


@pytest.mark.parametrize("file_io", ["adlfs", "pyarrow"])
def test_an_azure_request_warehouse_reads_with_the_azure_settings(tmp_path, azurite, file_io):
    """An ``abfs://`` URI with no catalog uri: its metadata is read with the Azure settings.

    pyiceberg reads ``abfs://`` with adlfs (and fails without it) unless ``py-io-impl``
    names pyarrow, which takes no connection string and reads only ``abfs://<container>/``.
    """
    warehouse = (
        f"abfs://lake@{azurite.account_name}.dfs.core.windows.net/wh"
        if file_io == "adlfs"
        else "abfs://lake/wh"
    )
    first = _two_snapshots(_azure_writer(tmp_path, azurite, warehouse))
    settings: dict[str, Any] = (
        {"azure_connection_string": azurite.get_connection_string()}
        if file_io == "adlfs"
        else _PYARROW_FILE_IO
    )

    _reads_request_warehouse(
        _azure_config(tmp_path, azurite, **settings), f"{warehouse}#taxi.trips", first
    )


def test_pyarrow_cannot_read_a_table_recorded_under_the_account_host(tmp_path, azurite):
    """The locations pyiceberg reads are the ones the metadata records, not the request's.

    So no rewrite of the requested warehouse URI makes such a table readable without adlfs.
    """
    host = f"abfs://lake@{azurite.account_name}.dfs.core.windows.net/wh"
    _two_snapshots(_azure_writer(tmp_path, azurite, host))
    config = _azure_config(tmp_path, azurite, **_PYARROW_FILE_IO)

    for warehouse in (host, "abfs://lake/wh"):
        with pytest.raises(OSError, match=f"lake@{azurite.account_name}"):
            ReadPlanner(config).plan(f"{warehouse}#taxi.trips")


REST_IMAGE = "apache/iceberg-rest-fixture:1.9.2"
JDK_IMAGE = "eclipse-temurin:17-jdk"
# The REST fixture's jar holds iceberg-core, -data and Hadoop; writing Parquet
# delete files takes these too (versions matching Iceberg 1.9.2).
MAVEN_JARS = (
    "org/apache/iceberg/iceberg-parquet/1.9.2/iceberg-parquet-1.9.2.jar",
    "org/apache/parquet/parquet-hadoop-bundle/1.15.2/parquet-hadoop-bundle-1.15.2.jar",
    "org/apache/parquet/parquet-avro/1.15.2/parquet-avro-1.15.2.jar",
)


@pytest.fixture(scope="module")
def java_equality_writer(tmp_path_factory) -> tuple[Path, str]:
    """Compiled tests/java/EqualityDeletes.java (Iceberg's Java writer): directory, classpath."""
    root = tmp_path_factory.mktemp("java")
    client = docker.from_env()
    try:
        client.images.pull(JDK_IMAGE)
        client.images.pull(REST_IMAGE)
        for path in MAVEN_JARS:
            urllib.request.urlretrieve(
                f"https://repo1.maven.org/maven2/{path}", root / path.rsplit("/", 1)[1]
            )
    except Exception as exc:
        pytest.skip(f"Java writer unavailable (image or Maven Central unreachable): {exc}")
    fixture = client.containers.create(REST_IMAGE)
    try:
        archive, _ = fixture.get_archive("/usr/lib/iceberg-rest/iceberg-rest-adapter.jar")
        with tarfile.open(fileobj=io.BytesIO(b"".join(archive))) as tar:
            tar.extractall(root, filter="data")
    finally:
        fixture.remove()
    shutil.copy(Path(__file__).parent / "java" / "EqualityDeletes.java", root)
    jars = ["iceberg-rest-adapter.jar", *(path.rsplit("/", 1)[1] for path in MAVEN_JARS)]
    classpath = ":".join(f"/w/{jar}" for jar in jars)
    client.containers.run(
        JDK_IMAGE,
        ["javac", "-cp", classpath, "EqualityDeletes.java"],
        volumes={str(root): {"bind": "/w", "mode": "rw"}},
        working_dir="/w",
        remove=True,
    )
    return root, f"/w:{classpath}"


def test_rows_iceberg_java_deleted_by_key_are_not_scanned(
    tmp_path, rest_catalog, java_equality_writer
):
    """A real equality delete: Iceberg's Java writer deletes ids 2 and 4 by value."""
    catalog, properties, container, warehouse = rest_catalog
    root, classpath = java_equality_writer
    catalog.create_namespace("taxi")
    catalog.create_table("taxi.trips", schema=SCHEMA)
    catalog.load_table("taxi.trips").append(pa.table({"id": pa.array([1, 2, 3, 4], pa.int64())}))

    docker.from_env().containers.run(
        JDK_IMAGE,
        [
            "java",
            "-cp",
            classpath,
            "EqualityDeletes",
            "http://localhost:8181",
            "taxi.trips",
            "id",
            "2",
            "4",
        ],
        network_mode=f"container:{container.get_wrapped_container().id}",
        volumes={
            str(root): {"bind": "/w", "mode": "ro"},
            str(warehouse): {"bind": str(warehouse), "mode": "rw"},
        },
        user="0",
        remove=True,
    )
    config = StrataConfig(cache_dir=tmp_path / "cache", catalogs={"rest": properties})
    assert _rows(config, "rest:taxi.trips") == [1, 3]

    # A row written after the delete is not deleted by it.
    catalog.load_table("taxi.trips").append(pa.table({"id": pa.array([2], pa.int64())}))
    assert _rows(config, "rest:taxi.trips") == [1, 2, 3]
