"""A table created and never written reads as zero rows with its schema.

There is no snapshot to key a result on, so nothing read from it dedups: the
first request after the first write sees the data.
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from fastapi.testclient import TestClient
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, LongType, NestedField, StringType


@pytest.fixture
def empty_table(tmp_path):
    """``ns.t`` (id, value, name), created and never written."""
    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
    from pyiceberg.catalog.sql import SqlCatalog

    wh = tmp_path / "warehouse"
    wh.mkdir()
    catalog = SqlCatalog("strata", uri=f"sqlite:///{wh / 'catalog.db'}", warehouse=wh.as_uri())
    catalog.create_namespace("ns")
    table = catalog.create_table(
        "ns.t",
        Schema(
            NestedField(1, "id", LongType(), required=False),
            NestedField(2, "value", DoubleType(), required=False),
            NestedField(3, "name", StringType(), required=False),
        ),
    )
    catalog.engine.dispose()
    return f"{wh.as_uri()}#ns.t", table


def _write(table, ids: list[int]) -> None:
    table.append(
        pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "value": pa.array([float(i) for i in ids], type=pa.float64()),
                "name": pa.array([f"n{i}" for i in ids], type=pa.string()),
            }
        )
    )


@pytest.fixture
def client(tmp_path):
    import strata.server as server_module
    from strata.artifact_store import reset_artifact_store
    from strata.config import StrataConfig
    from strata.server import ServerState, app

    config = StrataConfig(
        host="127.0.0.1",
        deployment_mode="personal",
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        metadata_db=tmp_path / "meta.sqlite",
    )
    reset_artifact_store()
    original = server_module._state
    server_module._state = ServerState(config)
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        server_module._state = original
        reset_artifact_store()


def _scan(table_uri: str, **params) -> dict:
    return {"inputs": [table_uri], "transform": {"executor": "scan@v1", "params": params}}


def _read(client, url: str) -> pa.Table:
    response = client.get(url)
    assert response.status_code == 200, response.text
    return ipc.open_stream(pa.BufferReader(response.content)).read_all()


class TestScan:
    def test_streams_zero_rows_with_the_tables_schema(self, client, empty_table):
        uri, _ = empty_table

        response = client.post("/v1/materialize", json=_scan(uri))

        assert response.status_code == 200, response.text
        table = _read(client, response.json()["stream_url"])
        assert table.num_rows == 0
        assert table.schema.names == ["id", "value", "name"]
        assert table.schema.field("value").type == pa.float64()

    def test_respects_the_projection(self, client, empty_table):
        uri, _ = empty_table

        response = client.post("/v1/materialize", json=_scan(uri, columns=["name", "id"]))

        assert response.status_code == 200, response.text
        table = _read(client, response.json()["stream_url"])
        assert table.num_rows == 0
        assert table.schema.names == ["name", "id"]

    def test_a_missing_column_is_still_400(self, client, empty_table):
        uri, _ = empty_table

        response = client.post("/v1/materialize", json=_scan(uri, columns=["id", "nope"]))

        assert response.status_code == 400, response.text
        assert "has no column(s) ['nope']" in response.json()["detail"]

    def test_the_first_request_after_the_first_write_sees_the_data(self, client, empty_table):
        uri, table = empty_table
        empty = client.post("/v1/materialize", json=_scan(uri)).json()
        assert _read(client, empty["stream_url"]).num_rows == 0
        again = client.post("/v1/materialize", json=_scan(uri)).json()
        assert again["hit"] is False

        _write(table, [1, 2, 3])
        response = client.post("/v1/materialize", json=_scan(uri))

        assert response.status_code == 200, response.text
        assert response.json()["hit"] is False
        assert _read(client, response.json()["stream_url"]).column("id").to_pylist() == [1, 2, 3]


def test_cache_warm_has_nothing_to_warm(client, empty_table):
    uri, _ = empty_table

    response = client.post("/v1/cache/warm", json={"tables": [uri]})

    assert response.status_code == 200, response.text
    assert response.json()["errors"] == []
    assert response.json()["tables_warmed"] == 1
    assert response.json()["row_groups_cached"] == 0


class TestThroughTheClient:
    def test_a_scan_and_a_transform_over_it_see_the_first_write(self, empty_table, tmp_path):
        from strata_client.client import StrataClient

        from tests.conftest import run_server_with_context

        uri, table = empty_table
        scan = {"executor": "scan@v1", "params": {"columns": ["id"]}}
        count = {"ref": "duckdb_sql@v1", "params": {"sql": "SELECT count(*) AS n FROM input0"}}
        with run_server_with_context(tmp_path / "cache", tmp_path / "artifacts") as ctx:
            with StrataClient(base_url=ctx.base_url) as client:
                empty = client.fetch(client.materialize(inputs=[uri], transform=scan).uri)
                assert empty.num_rows == 0
                assert empty.schema.names == ["id"]
                counted = client.materialize(inputs=[uri], transform=count)
                assert counted.to_table()["n"].to_pylist() == [0]
                assert client.materialize(inputs=[uri], transform=count).cache_hit is False

                _write(table, [1, 2, 3])

                scanned = client.fetch(client.materialize(inputs=[uri], transform=scan).uri)
                assert scanned.column("id").to_pylist() == [1, 2, 3]
                counted = client.materialize(inputs=[uri], transform=count)
                assert counted.to_table()["n"].to_pylist() == [3]
