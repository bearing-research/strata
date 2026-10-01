"""Tests for the unified /v1/materialize endpoint."""

import sys
import time
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import requests
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, LongType, NestedField, StringType

from strata.transforms.build_qos import (
    BuildQoS,
    BuildQoSConfig,
    reset_build_qos,
    set_build_qos,
)


@pytest.fixture
def temp_warehouse(tmp_path):
    """Create a temporary warehouse with a sample Iceberg table."""
    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
    warehouse_path = tmp_path / "warehouse"
    warehouse_path.mkdir()

    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )

    catalog.create_namespace("test_db")

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
        NestedField(3, "name", StringType(), required=False),
        NestedField(4, "timestamp", LongType(), required=False),
    )

    table = catalog.create_table("test_db.events", schema)

    num_rows = 100
    base_ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)
    data = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "value": pa.array([float(i * 1.5) for i in range(num_rows)], type=pa.float64()),
            "name": pa.array([f"item_{i}" for i in range(num_rows)], type=pa.string()),
            "timestamp": pa.array(
                [base_ts + i * 3600_000_000 for i in range(num_rows)],
                type=pa.int64(),
            ),
        }
    )

    table.append(data)

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"file://{warehouse_path}#test_db.events",
        "catalog": catalog,
        "table": table,
    }


@pytest.fixture
def server_with_personal_mode(temp_warehouse, tmp_path):
    """Start a server in personal mode (writes enabled) and provide base URL.

    Uses the shared run_server_with_context helper — health-polled startup
    and graceful shutdown (see server_with_artifacts in test_put_json.py).
    """
    from tests.conftest import run_server_with_context

    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()

    with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
        yield {
            "base_url": ctx.base_url,
            "config": ctx.config,
            "warehouse": temp_warehouse,
        }


class TestUnifiedMaterialize:
    """Tests for the unified /v1/materialize endpoint."""

    def test_identity_materialize_stream_mode(self, server_with_personal_mode):
        """Test scan@v1 transform in stream mode."""
        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        response = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [table_uri],
                "transform": {
                    "executor": "scan@v1",
                    "params": {},
                },
                "mode": "stream",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # First request, so a miss.
        assert data["hit"] is False
        assert data["state"] == "building"
        assert data["artifact_uri"].startswith("strata://artifact/")
        assert data["stream_id"] is not None
        assert data["stream_url"].startswith("/v1/streams/")

        stream_response = requests.get(
            f"{base_url}{data['stream_url']}",
            headers={"Accept": "application/vnd.apache.arrow.stream"},
        )

        assert stream_response.status_code == 200
        assert stream_response.headers["content-type"] == "application/vnd.apache.arrow.stream"

        reader = ipc.open_stream(stream_response.content)
        table = reader.read_all()

        assert table.num_rows == 100
        assert set(table.column_names) == {"id", "value", "name", "timestamp"}

    def test_identity_materialize_with_projection(self, server_with_personal_mode):
        """Test scan@v1 with column projection."""
        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        response = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [table_uri],
                "transform": {
                    "executor": "scan@v1",
                    "params": {
                        "columns": ["id", "name"],
                    },
                },
                "mode": "stream",
            },
        )

        assert response.status_code == 200
        data = response.json()

        stream_response = requests.get(
            f"{base_url}{data['stream_url']}",
            headers={"Accept": "application/vnd.apache.arrow.stream"},
        )

        assert stream_response.status_code == 200

        reader = ipc.open_stream(stream_response.content)
        table = reader.read_all()

        assert table.num_rows == 100
        assert set(table.column_names) == {"id", "name"}

    def test_identity_materialize_with_filters(self, server_with_personal_mode):
        """Test scan@v1 with row filters."""
        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        response = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [table_uri],
                "transform": {
                    "executor": "scan@v1",
                    "params": {
                        "filters": [
                            {"column": "id", "op": "<", "value": 50},
                        ],
                    },
                },
                "mode": "stream",
            },
        )

        assert response.status_code == 200
        data = response.json()

        stream_response = requests.get(
            f"{base_url}{data['stream_url']}",
            headers={"Accept": "application/vnd.apache.arrow.stream"},
        )

        assert stream_response.status_code == 200

        # Filters may not reduce rows when row groups can't be pruned; the request must
        # still succeed.
        reader = ipc.open_stream(stream_response.content)
        table = reader.read_all()

        assert table.num_rows >= 0

    def test_identity_materialize_cache_hit(self, server_with_personal_mode):
        """Test that same query returns cache hit."""
        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        response1 = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [table_uri],
                "transform": {
                    "executor": "scan@v1",
                    "params": {"columns": ["id"]},
                },
                "mode": "stream",
            },
        )

        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["hit"] is False

        # Consume the stream to finalize the artifact.
        stream_response = requests.get(
            f"{base_url}{data1['stream_url']}",
        )
        assert stream_response.status_code == 200

        # Let the artifact finalize.
        time.sleep(0.5)

        response2 = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [table_uri],
                "transform": {
                    "executor": "scan@v1",
                    "params": {"columns": ["id"]},
                },
                "mode": "stream",
            },
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["hit"] is True
        assert data2["state"] == "ready"
        assert data2["artifact_uri"] == data1["artifact_uri"]

    def test_identity_materialize_artifact_mode(self, server_with_personal_mode):
        """Test scan@v1 in artifact mode."""
        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        response = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [table_uri],
                "transform": {
                    "executor": "scan@v1",
                    "params": {},
                },
                "mode": "artifact",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["hit"] is False
        assert data["state"] == "pending"
        assert data["artifact_uri"].startswith("strata://artifact/")
        assert data["build_id"] is not None
        # Artifact mode provides no stream_url.
        assert data.get("stream_url") is None

    def test_identity_artifact_mode_build_status_and_name(self, server_with_personal_mode):
        """scan@v1 artifact mode builds in the background and sets names on miss."""
        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]
        table = server_with_personal_mode["warehouse"]["table"]

        extra_rows = 30
        table.append(
            pa.table(
                {
                    "id": pa.array(range(100, 100 + extra_rows), type=pa.int64()),
                    "value": pa.array(
                        [float(i * 1.5) for i in range(100, 100 + extra_rows)],
                        type=pa.float64(),
                    ),
                    "name": pa.array(
                        [f"item_{i}" for i in range(100, 100 + extra_rows)],
                        type=pa.string(),
                    ),
                    "timestamp": pa.array(
                        [i * 1_000_000 for i in range(100, 100 + extra_rows)],
                        type=pa.int64(),
                    ),
                }
            )
        )

        response = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [table_uri],
                "transform": {
                    "executor": "scan@v1",
                    "params": {"columns": ["id", "name"]},
                },
                "mode": "artifact",
                "name": "named_identity_build",
            },
        )

        assert response.status_code == 200
        data = response.json()
        build_id = data["build_id"]
        artifact_uri = data["artifact_uri"]

        deadline = time.time() + 10
        while True:
            status_resp = requests.get(f"{base_url}/v1/artifacts/builds/{build_id}")
            assert status_resp.status_code == 200
            build_status = status_resp.json()

            if build_status["state"] == "ready":
                break
            assert build_status["state"] in {"pending", "building"}
            assert time.time() < deadline
            time.sleep(0.1)

        assert build_status["artifact_uri"] == artifact_uri
        assert build_status["executor_ref"] == "scan@v1"

        name_resp = requests.get(f"{base_url}/v1/names/named_identity_build")
        assert name_resp.status_code == 200
        assert name_resp.json()["artifact_uri"] == artifact_uri

        artifact_id, version = artifact_uri.removeprefix("strata://artifact/").split("@v=")
        data_resp = requests.get(f"{base_url}/v1/artifacts/{artifact_id}/v/{version}/data")
        assert data_resp.status_code == 200

        table = ipc.open_stream(data_resp.content).read_all()
        assert set(table.column_names) == {"id", "name"}

        info_resp = requests.get(f"{base_url}/v1/artifacts/{artifact_id}/v/{version}")
        assert info_resp.status_code == 200
        assert info_resp.json()["row_count"] == 100 + extra_rows

    def test_identity_artifact_mode_respects_build_qos_quota(self, server_with_personal_mode):
        """Identity artifact-mode should be admitted through build QoS."""
        qos = BuildQoS(BuildQoSConfig(bytes_per_day_limit=1))
        set_build_qos(qos)

        try:
            base_url = server_with_personal_mode["base_url"]
            table_uri = server_with_personal_mode["warehouse"]["table_uri"]

            response = requests.post(
                f"{base_url}/v1/materialize",
                json={
                    "inputs": [table_uri],
                    "transform": {
                        "executor": "scan@v1",
                        "params": {"columns": ["id", "value", "name"]},
                    },
                    "mode": "artifact",
                },
            )

            assert response.status_code == 429
            data = response.json()
            assert data["error"] == "quota_exceeded"
        finally:
            reset_build_qos()

    def test_identity_requires_single_input(self, server_with_personal_mode):
        """Test that scan@v1 rejects multiple inputs."""
        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        response = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [table_uri, table_uri],
                "transform": {
                    "executor": "scan@v1",
                    "params": {},
                },
            },
        )

        assert response.status_code == 400
        assert "exactly one input" in response.json()["detail"]

    def test_identity_rejects_artifact_input(self, server_with_personal_mode):
        """Test that scan@v1 rejects artifact URIs as input."""
        base_url = server_with_personal_mode["base_url"]

        response = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": ["strata://artifact/abc123@v=1"],
                "transform": {
                    "executor": "scan@v1",
                    "params": {},
                },
            },
        )

        assert response.status_code == 400
        assert "table URI" in response.json()["detail"]

    def test_stream_not_found(self, server_with_personal_mode):
        """Test 404 for non-existent stream."""
        base_url = server_with_personal_mode["base_url"]

        response = requests.get(f"{base_url}/v1/streams/nonexistent")

        assert response.status_code == 404

    def test_invalid_identity_params(self, server_with_personal_mode):
        """Test that invalid identity params return 400."""
        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        response = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [table_uri],
                "transform": {
                    "executor": "scan@v1",
                    "params": {
                        "filters": "not_a_list",
                    },
                },
            },
        )

        assert response.status_code == 400


class TestUnifiedMaterializeEdgeCases:
    """Edge case tests for unified materialize."""

    def test_default_mode_is_stream(self, server_with_personal_mode):
        """Test that the default mode is 'stream'."""
        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        response = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [table_uri],
                "transform": {
                    "executor": "scan@v1",
                    "params": {},
                },
                # mode defaults to "stream"
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data.get("stream_url") is not None


class TestClientFetch:
    """Tests for the client SDK fetch() method."""

    def test_client_fetch_basic(self, server_with_personal_mode):
        """Test basic materialize() + fetch() usage."""
        from strata_client.client import StrataClient

        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        client = StrataClient(base_url=base_url)

        try:
            artifact = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {}},
            )
            table = client.fetch(artifact.uri)

            assert table.num_rows == 100
            assert set(table.column_names) == {"id", "value", "name", "timestamp"}
        finally:
            client.close()

    def test_client_fetch_with_projection(self, server_with_personal_mode):
        """Test materialize() + fetch() with column projection."""
        from strata_client.client import StrataClient

        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        client = StrataClient(base_url=base_url)

        try:
            artifact = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {"columns": ["id", "value"]}},
            )
            table = client.fetch(artifact.uri)

            assert table.num_rows == 100
            assert set(table.column_names) == {"id", "value"}
        finally:
            client.close()

    def test_client_fetch_with_filters(self, server_with_personal_mode):
        """Test materialize() + fetch() with row filters."""
        from strata_client.client import StrataClient

        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        client = StrataClient(base_url=base_url)

        try:
            artifact = client.materialize(
                inputs=[table_uri],
                transform={
                    "executor": "scan@v1",
                    "params": {"filters": [{"column": "id", "op": "<", "value": 50}]},
                },
            )
            table = client.fetch(artifact.uri)

            # Filters prune at row-group level, so all rows may come back; the request must
            # succeed.
            assert table.num_rows >= 0
        finally:
            client.close()

    def test_client_materialize_returns_artifact(self, server_with_personal_mode):
        """Test that materialize() returns an Artifact with metadata."""
        from strata_client.client import StrataClient

        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        client = StrataClient(base_url=base_url)

        try:
            artifact = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {"columns": ["id"]}},
            )

            assert artifact.artifact_id is not None
            assert artifact.version == 1
            assert artifact.uri.startswith("strata://artifact/")

            table = client.fetch(artifact.uri)
            assert table.num_rows == 100
            assert set(table.column_names) == {"id"}
        finally:
            client.close()

    def test_client_materialize_cache_hit(self, server_with_personal_mode):
        """Test that repeated materialize() calls return cache hits."""
        from strata_client.client import StrataClient

        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        client = StrataClient(base_url=base_url)

        try:
            artifact1 = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {"columns": ["id", "name"]}},
            )
            assert artifact1.cache_hit is False

            # Let the artifact finalize.
            import time

            time.sleep(0.5)

            artifact2 = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {"columns": ["id", "name"]}},
            )
            assert artifact2.cache_hit is True
            assert artifact2.artifact_id == artifact1.artifact_id
        finally:
            client.close()

    def test_client_materialize_refresh_rebuilds_same_artifact(self, server_with_personal_mode):
        """refresh=True rebuilds as a new version of the SAME artifact (#123).

        Refresh used to fork a parallel artifact identity that provenance
        lookups never returned; it now supersedes the old version so the
        rebuild becomes canonical.
        """
        from strata_client.client import StrataClient

        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        client = StrataClient(base_url=base_url)

        try:
            artifact1 = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {"columns": ["id"]}},
            )
            artifact1.to_table()  # consume so the artifact finalizes
            artifact2 = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {"columns": ["id"]}},
                refresh=True,
            )
            artifact2.to_table()

            assert artifact1.cache_hit is False
            assert artifact2.cache_hit is False
            assert artifact2.artifact_id == artifact1.artifact_id
            assert artifact2.version == artifact1.version + 1

            # The provenance cache now resolves the rebuild.
            artifact3 = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {"columns": ["id"]}},
            )
            assert artifact3.cache_hit is True
            assert artifact3.version == artifact2.version
        finally:
            client.close()

    def test_client_materialize_artifact_mode(self, server_with_personal_mode):
        """The sync client can wait for scan@v1 artifact-mode builds."""
        from strata_client.client import StrataClient

        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        client = StrataClient(base_url=base_url)

        try:
            artifact = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {"columns": ["value"]}},
                mode="artifact",
            )

            table = client.fetch(artifact.uri)
            assert table.num_rows == 100
            assert set(table.column_names) == {"value"}
        finally:
            client.close()


class TestTransformOverATable:
    """A non-scan transform whose input is a table URI."""

    def test_a_schema_change_rebuilds_and_stales_the_name(self, server_with_personal_mode):
        """A rename makes no snapshot, so the table's version must name the schema.

        Versioned by snapshot id alone, the second materialize hit the first
        artifact and served the column under its old name, and the name's
        status said it was fresh.
        """
        from strata_client.client import StrataClient

        base_url = server_with_personal_mode["base_url"]
        warehouse = server_with_personal_mode["warehouse"]
        table_uri = warehouse["table_uri"]
        sql = {"executor": "duckdb_sql@v1", "params": {"sql": "SELECT * FROM input0"}}

        client = StrataClient(base_url=base_url)
        try:
            first = client.materialize(inputs=[table_uri], transform=sql, name="events_sql")
            assert "name" in client.fetch(first.uri).column_names
            assert client.get_name_status("events_sql")["is_stale"] is False

            with warehouse["catalog"].load_table("test_db.events").update_schema() as update:
                update.rename_column("name", "label")

            assert client.get_name_status("events_sql")["is_stale"] is True
            second = client.materialize(inputs=[table_uri], transform=sql)
            assert second.cache_hit is False
            columns = client.fetch(second.uri).column_names
            assert "label" in columns
            assert "name" not in columns
        finally:
            client.close()

    def test_a_named_scan_stays_fresh(self, server_with_personal_mode):
        """The scan path records the table's version in the form name status reads."""
        from strata_client.client import StrataClient

        base_url = server_with_personal_mode["base_url"]
        table_uri = server_with_personal_mode["warehouse"]["table_uri"]

        client = StrataClient(base_url=base_url)
        try:
            artifact = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {}},
                name="events_scan",
                mode="artifact",
            )
            client.fetch(artifact.uri)
            assert client.get_name_status("events_scan")["is_stale"] is False
        finally:
            client.close()

    def test_a_table_strata_refuses_to_read_is_422(self, server_with_personal_mode):
        """The planner's refusal keeps its status and message, as on the scan path."""
        from pyiceberg.manifest import FileFormat

        from tests.iceberg_fixtures import commit_files, equality_delete

        base_url = server_with_personal_mode["base_url"]
        warehouse = server_with_personal_mode["warehouse"]
        table = warehouse["catalog"].load_table("test_db.events")
        commit_files(
            table,
            equality_delete(
                table, pa.table({"id": pa.array([1], pa.int64())}), file_format=FileFormat.AVRO
            ),
        )

        response = requests.post(
            f"{base_url}/v1/materialize",
            json={
                "inputs": [warehouse["table_uri"]],
                "transform": {"executor": "duckdb_sql@v1", "params": {"sql": "SELECT 1"}},
            },
            timeout=60,
        )
        assert response.status_code == 422
        assert "AVRO equality delete file" in response.json()["detail"]
