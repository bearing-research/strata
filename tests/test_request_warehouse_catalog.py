"""A warehouse named only in a request's table URI, on a server with no catalog ``uri``.

Its catalog would be SQLite at ``metadata_db`` on this server's disk. A service
refuses it with a 400 naming ``STRATA_CATALOG_URI``; a personal server keeps it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.config import StrataConfig
from strata.iceberg import CatalogUriRequired, PyIcebergCatalog

OBJECT_STORE = ["s3://lake/wh", "gs://lake/wh", "az://lake/wh", "abfs://c@acct/wh"]


def _config(tmp_path, mode: str, **overrides) -> StrataConfig:
    return StrataConfig(
        deployment_mode=mode,
        cache_dir=tmp_path / "cache",
        metadata_db=tmp_path / "meta.sqlite",
        **overrides,
    )


class TestPersonalModeKeepsTheFallback:
    @pytest.mark.parametrize("warehouse", OBJECT_STORE)
    def test_an_object_store_warehouse_uses_metadata_db(self, tmp_path, warehouse):
        catalogs = PyIcebergCatalog(_config(tmp_path, "personal"))

        uri = catalogs._get_default_catalog_uri(warehouse)

        assert uri == f"sqlite:///{tmp_path / 'meta.sqlite'}"

    @pytest.mark.parametrize("warehouse", ["gs://lake/wh", "az://lake/wh"])
    def test_a_gcs_or_azure_warehouse_builds_its_catalog_there(self, tmp_path, warehouse):
        catalog = PyIcebergCatalog(_config(tmp_path, "personal"))._build_catalog(warehouse)
        catalog.create_namespace("ns")

        assert catalog.properties["uri"] == f"sqlite:///{tmp_path / 'meta.sqlite'}"
        assert (tmp_path / "meta.sqlite").is_file()

    def test_a_local_warehouse_keeps_its_own_catalog(self, tmp_path):
        catalogs = PyIcebergCatalog(_config(tmp_path, "personal"))

        assert catalogs._get_default_catalog_uri(str(tmp_path / "wh")) == (
            f"sqlite:///{tmp_path / 'wh' / 'catalog.db'}"
        )


class TestServiceModeRefuses:
    @pytest.mark.parametrize("warehouse", OBJECT_STORE)
    def test_an_object_store_warehouse_without_a_uri(self, tmp_path, warehouse):
        catalogs = PyIcebergCatalog(_config(tmp_path, "service"))

        with pytest.raises(CatalogUriRequired, match="STRATA_CATALOG_URI"):
            catalogs._build_catalog(warehouse)
        assert not (tmp_path / "meta.sqlite").exists()

    def test_a_configured_uri_is_used(self, tmp_path):
        uri = f"sqlite:///{tmp_path / 'shared.sqlite'}"
        catalogs = PyIcebergCatalog(_config(tmp_path, "service", catalog_properties={"uri": uri}))

        assert catalogs._get_default_catalog_uri("s3://lake/wh") == uri

    def test_a_local_warehouse_is_not_refused(self, tmp_path):
        catalogs = PyIcebergCatalog(_config(tmp_path, "service"))

        assert catalogs._get_default_catalog_uri(str(tmp_path / "wh")) == (
            f"sqlite:///{tmp_path / 'wh' / 'catalog.db'}"
        )


@pytest.fixture
def service_client(tmp_path, monkeypatch):
    import strata.server as server_module
    from strata.server import ServerState, app

    monkeypatch.setattr(server_module, "_state", ServerState(_config(tmp_path, "service")))
    return TestClient(app)


def _scan(table_uri: str) -> dict:
    return {"inputs": [table_uri], "transform": {"executor": "scan@v1", "params": {}}}


class TestServiceModeRefusalOverHttp:
    @pytest.mark.parametrize("warehouse", ["s3://lake/wh", "gs://lake/wh"])
    def test_materialize_answers_400_naming_the_setting(self, service_client, tmp_path, warehouse):
        response = service_client.post("/v1/materialize", json=_scan(f"{warehouse}#ns.events"))

        assert response.status_code == 400, response.text
        assert "STRATA_CATALOG_URI" in response.json()["detail"]
        assert "this server's disk" in response.json()["detail"]
        assert not (tmp_path / "meta.sqlite").exists()

    @pytest.mark.parametrize("route", ["/v1/cache/warm", "/v1/cache/warm/async"])
    def test_cache_warm_answers_400_naming_the_setting(self, service_client, route):
        response = service_client.post(route, json={"tables": ["s3://lake/wh#ns.events"]})

        assert response.status_code == 400, response.text
        assert "STRATA_CATALOG_URI" in response.json()["detail"]
