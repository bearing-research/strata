"""A request's bad table URI is the caller's 4xx, not a 500.

A URI that names no ``namespace.table`` is a 400 carrying the parse message; a
table its catalog does not have, a local warehouse that does not exist or a
snapshot the table does not have is a 404 naming it; a projected column the
table does not have is a 400 naming it.
"""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

MALFORMED = ["events", "a.b.c", "file:///wh#a.b.c", "file:///wh#.events", "file:///wh#ns."]


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


@pytest.fixture
def sql_transform():
    """Register ``sql@v1`` so a table can be a transform input."""
    from strata.transforms.registry import (
        TransformDefinition,
        TransformRegistry,
        reset_transform_registry,
        set_transform_registry,
    )

    set_transform_registry(
        TransformRegistry(
            enabled=True,
            definitions=[TransformDefinition(ref="sql@v1", executor_url="http://executor")],
        )
    )
    yield
    reset_transform_registry()


def _transform(table_uri: str) -> dict:
    return {"inputs": [table_uri], "transform": {"executor": "sql@v1", "params": {}}}


class TestMalformedTableUri:
    @pytest.mark.parametrize("uri", MALFORMED)
    def test_materialize_is_400_with_the_message(self, client, uri):
        response = client.post("/v1/materialize", json=_scan(uri))

        assert response.status_code == 400, response.text
        assert "expected 'namespace.table' format" in response.json()["detail"]

    @pytest.mark.parametrize("route", ["/v1/cache/warm", "/v1/cache/warm/async"])
    @pytest.mark.parametrize("uri", MALFORMED)
    def test_cache_warm_is_400_with_the_message(self, client, route, uri):
        response = client.post(route, json={"tables": [uri]})

        assert response.status_code == 400, response.text
        assert "expected 'namespace.table' format" in response.json()["detail"]


@pytest.fixture
def warehouse(tmp_path):
    """A warehouse whose catalog has namespace ``ns`` and no tables."""
    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
    from pyiceberg.catalog.sql import SqlCatalog

    wh = tmp_path / "warehouse"
    wh.mkdir()
    SqlCatalog("strata", uri=f"sqlite:///{wh / 'catalog.db'}", warehouse=str(wh)).create_namespace(
        "ns"
    )
    return wh


class TestTableMissingFromItsCatalog:
    @pytest.mark.parametrize("table_id", ["ns.missing", "nons.missing"])
    def test_scan_is_404_naming_the_table(self, client, warehouse, table_id):
        uri = f"file://{warehouse}#{table_id}"

        response = client.post("/v1/materialize", json=_scan(uri))

        assert response.status_code == 404, response.text
        assert response.json()["detail"] == f"Table not found: {uri}"

    def test_a_transform_input_is_404_naming_the_table(self, client, warehouse, sql_transform):
        uri = f"file://{warehouse}#ns.missing"

        response = client.post("/v1/materialize", json=_transform(uri))

        assert response.status_code == 404, response.text
        assert response.json()["detail"] == f"Table not found: {uri}"


class TestWarehouseThatDoesNotExist:
    @pytest.mark.parametrize("scheme", ["", "file://"])
    def test_scan_is_404_naming_the_warehouse(self, client, tmp_path, scheme):
        missing = tmp_path / "nowh"

        response = client.post("/v1/materialize", json=_scan(f"{scheme}{missing}#ns.t"))

        assert response.status_code == 404, response.text
        assert response.json()["detail"] == f"Warehouse not found: {missing}"
        assert not missing.exists()

    def test_a_transform_input_is_404_naming_the_warehouse(self, client, tmp_path, sql_transform):
        missing = tmp_path / "nowh"

        response = client.post("/v1/materialize", json=_transform(f"file://{missing}#ns.t"))

        assert response.status_code == 404, response.text
        assert response.json()["detail"] == f"Warehouse not found: {missing}"


class TestScanOfWhatTheTableDoesNotHave:
    def test_unknown_snapshot_is_404_naming_it(self, client, temp_warehouse):
        body = _scan(temp_warehouse["table_uri"], snapshot_id=12345)

        response = client.post("/v1/materialize", json=body)

        assert response.status_code == 404, response.text
        assert response.json()["detail"] == "Snapshot 12345 not found in table"

    def test_missing_column_is_400_naming_it(self, client, temp_warehouse):
        body = _scan(temp_warehouse["table_uri"], columns=["id", "nope"])

        response = client.post("/v1/materialize", json=body)

        assert response.status_code == 400, response.text
        assert response.json()["detail"] == (
            "Table strata.test_db.events has no column(s) ['nope']. "
            "Available columns: ['id', 'name', 'timestamp', 'value']."
        )

    def test_the_existing_snapshot_and_columns_still_plan(self, client, temp_warehouse):
        snapshot_id = temp_warehouse["table"].current_snapshot().snapshot_id
        body = _scan(temp_warehouse["table_uri"], snapshot_id=snapshot_id, columns=["id"])

        response = client.post("/v1/materialize", json=body)

        assert response.status_code == 200, response.text


class TestWarehouseWithNoCatalogOnAService:
    def test_a_transform_input_is_400_naming_the_setting(
        self, tmp_path, monkeypatch, sql_transform
    ):
        import strata.server as server_module
        from strata.artifact_store import reset_artifact_store
        from strata.config import StrataConfig
        from strata.server import ServerState, app

        config = StrataConfig(
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            transforms_config={"enabled": True},
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            metadata_db=tmp_path / "meta.sqlite",
        )
        monkeypatch.setattr(server_module, "_state", ServerState(config))
        reset_artifact_store()
        client = TestClient(
            app,
            raise_server_exceptions=False,
            headers={"X-Strata-Proxy-Token": "test-token", "X-Strata-Principal": "user-1"},
        )
        try:
            response = client.post("/v1/materialize", json=_transform("s3://lake/wh#ns.events"))
        finally:
            reset_artifact_store()

        assert response.status_code == 400, response.text
        assert "STRATA_CATALOG_URI" in response.json()["detail"]
        assert not (tmp_path / "meta.sqlite").exists()
