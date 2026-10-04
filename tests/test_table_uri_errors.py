"""A request's bad table URI is the caller's 4xx, not a 500.

A URI that names no ``namespace.table`` is a 400 carrying the parse message; a
table its catalog does not have is a 404 naming it.
"""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

MALFORMED = ["events", "a.b.c", "file:///wh#a.b.c"]


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


def _scan(table_uri: str) -> dict:
    return {"inputs": [table_uri], "transform": {"executor": "scan@v1", "params": {}}}


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

    def test_a_transform_input_is_404_naming_the_table(self, client, warehouse):
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
        uri = f"file://{warehouse}#ns.missing"
        body = {"inputs": [uri], "transform": {"executor": "sql@v1", "params": {}}}
        try:
            response = client.post("/v1/materialize", json=body)
        finally:
            reset_transform_registry()

        assert response.status_code == 404, response.text
        assert response.json()["detail"] == f"Table not found: {uri}"
