"""A warehouse named only in a request's table URI, on a server with no catalog ``uri``.

Its catalog would be SQLite at ``metadata_db`` on this server's disk. A service
refuses it with a 400 naming ``STRATA_CATALOG_URI``; a personal server keeps it.
"""

from __future__ import annotations

import os

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


def _store_props(catalog, prefix: str) -> dict[str, str]:
    return {k: v for k, v in catalog.properties.items() if k.startswith(prefix)}


class TestStoreSettingsReachTheCatalog:
    """The warehouse's catalog reads its metadata with the server's settings for that store."""

    def test_a_gcs_warehouse_gets_the_gcs_settings(self, tmp_path):
        config = _config(
            tmp_path,
            "personal",
            gcs_default_bucket_location="europe-west1",
            gcs_endpoint_override="http://127.0.0.1:4443",
            azure_account_name="acct",
        )

        catalog = PyIcebergCatalog(config)._build_catalog("gs://lake/wh")

        assert _store_props(catalog, "gcs.") == {
            "gcs.default-bucket-location": "europe-west1",
            "gcs.service.host": "http://127.0.0.1:4443",
        }
        assert _store_props(catalog, "adls.") == {}

    def test_a_bare_gcs_endpoint_gets_https(self, tmp_path):
        config = _config(tmp_path, "personal", gcs_endpoint_override="storage.example:443")

        catalog = PyIcebergCatalog(config)._build_catalog("gs://lake/wh")

        assert catalog.properties["gcs.service.host"] == "https://storage.example:443"

    def test_a_gcs_key_file_is_exported_for_pyarrow(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/elsewhere.json")
        key = tmp_path / "key.json"
        config = _config(tmp_path, "personal", gcs_credentials_json=str(key))

        PyIcebergCatalog(config)._build_catalog("gs://lake/wh")

        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(key)

    def test_a_gcs_warehouse_with_nothing_configured_gets_no_gcs_props(self, tmp_path):
        catalog = PyIcebergCatalog(_config(tmp_path, "personal"))._build_catalog("gs://lake/wh")

        assert _store_props(catalog, "gcs.") == {}

    @pytest.mark.parametrize(
        "warehouse", ["az://lake/wh", "abfs://lake@acct.dfs.core.windows.net/wh", "abfss://l@a/wh"]
    )
    def test_an_azure_warehouse_gets_the_azure_settings(self, tmp_path, warehouse):
        config = _config(
            tmp_path,
            "personal",
            azure_account_name="acct",
            azure_account_key="a2V5",
            azure_sas_token="sv=2024&sig=x",
            azure_connection_string="AccountName=acct;AccountKey=a2V5",
            azure_endpoint_url="http://127.0.0.1:10000",
            gcs_endpoint_override="http://127.0.0.1:4443",
        )

        catalog = PyIcebergCatalog(config)._build_catalog(warehouse)

        assert _store_props(catalog, "adls.") == {
            "adls.account-name": "acct",
            "adls.account-key": "a2V5",
            "adls.sas-token": "sv=2024&sig=x",
            "adls.connection-string": "AccountName=acct;AccountKey=a2V5",
            "adls.blob-storage-authority": "127.0.0.1:10000",
            "adls.blob-storage-scheme": "http",
        }
        assert _store_props(catalog, "gcs.") == {}

    def test_an_azure_endpoint_with_a_key_becomes_a_connection_string(self, tmp_path):
        """adlfs reads only a connection string for an emulator endpoint, so one is derived."""
        config = _config(
            tmp_path,
            "personal",
            azure_account_name="devstoreaccount1",
            azure_account_key="a2V5",
            azure_endpoint_url="http://127.0.0.1:10000",
        )

        catalog = PyIcebergCatalog(config)._build_catalog("abfs://lake@devstoreaccount1/wh")

        assert catalog.properties["adls.connection-string"] == (
            "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=a2V5;"
            "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1"
        )
        assert catalog.properties["adls.blob-storage-authority"] == "127.0.0.1:10000"

    def test_an_azure_endpoint_already_naming_the_account_is_not_doubled(self, tmp_path):
        config = _config(
            tmp_path,
            "personal",
            azure_account_name="devstoreaccount1",
            azure_account_key="a2V5",
            azure_endpoint_url="http://127.0.0.1:10000/devstoreaccount1",
        )

        catalog = PyIcebergCatalog(config)._build_catalog("abfs://lake@devstoreaccount1/wh")

        assert catalog.properties["adls.connection-string"].endswith(
            ";BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1"
        )

    def test_an_azure_endpoint_without_a_key_derives_no_connection_string(self, tmp_path):
        config = _config(
            tmp_path,
            "personal",
            azure_account_name="acct",
            azure_sas_token="sv=2024&sig=x",
            azure_endpoint_url="http://127.0.0.1:10000",
        )

        catalog = PyIcebergCatalog(config)._build_catalog("abfs://lake@acct/wh")

        assert "adls.connection-string" not in catalog.properties

    def test_an_azure_warehouse_gets_only_what_is_configured(self, tmp_path):
        config = _config(tmp_path, "personal", azure_account_name="acct")

        catalog = PyIcebergCatalog(config)._build_catalog("abfs://lake@acct/wh")

        assert _store_props(catalog, "adls.") == {"adls.account-name": "acct"}

    def test_an_azure_warehouse_with_nothing_configured_gets_no_adls_props(self, tmp_path):
        catalog = PyIcebergCatalog(_config(tmp_path, "personal"))._build_catalog("az://lake/wh")

        assert _store_props(catalog, "adls.") == {}

    def test_catalog_properties_still_win(self, tmp_path):
        config = _config(
            tmp_path,
            "personal",
            gcs_endpoint_override="http://127.0.0.1:4443",
            catalog_properties={"gcs.service.host": "http://override:1"},
        )

        catalog = PyIcebergCatalog(config)._build_catalog("gs://lake/wh")

        assert catalog.properties["gcs.service.host"] == "http://override:1"


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
