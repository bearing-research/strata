"""Named catalogs and scans across stores.

Covers resolution, the planner's identity and credential handling, and file routing with local SQL
catalogs; real REST, Glue and GCS are in ``test_lake_catalogs_integration.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog

from strata import lake_files
from strata.config import StrataConfig
from strata.fetcher import create_fetcher
from strata.iceberg import named_catalog
from strata.planner import ReadPlanner

windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="pyiceberg + pyarrow LocalFileSystem path handling broken on Windows",
)


def _config(tmp_path, catalogs):
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    return StrataConfig(cache_dir=cache, catalogs=catalogs)


class TestResolution:
    def test_a_configured_name_splits_off_its_table(self):
        config = SimpleNamespace(catalogs={"lake": {}})

        assert named_catalog("lake:taxi.trips", config) == ("lake", "taxi.trips")

    @pytest.mark.parametrize(
        "uri",
        [
            "other:taxi.trips",
            "taxi.trips",
            "C:/warehouse#taxi.trips",
            "s3://bucket/wh#taxi.trips",
            "lake://taxi.trips",
        ],
    )
    def test_anything_else_is_not_a_named_catalog(self, uri):
        config = SimpleNamespace(catalogs={"lake": {}, "C": {}})

        assert named_catalog(uri, config) == (None, uri)


def _sql_catalog(tmp_path, name: str) -> tuple[dict[str, str], SqlCatalog]:
    warehouse = tmp_path / name
    warehouse.mkdir()
    properties = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse / 'catalog.db'}",
        "warehouse": warehouse.as_uri(),
    }
    return properties, SqlCatalog(name, **{k: v for k, v in properties.items() if k != "type"})


def _rows(config, uri: str, snapshot_id: int | None = None) -> list[int]:
    plan = ReadPlanner(config).plan(uri, snapshot_id=snapshot_id)
    fetcher = create_fetcher()
    return sorted(
        value for task in plan.tasks for value in fetcher.fetch(task).column("id").to_pylist()
    )


@windows
class TestScanningByName:
    def test_two_catalogs_are_read_by_name_and_a_pin_reads_its_snapshot(self, tmp_path):
        north_props, north = _sql_catalog(tmp_path, "north")
        south_props, south = _sql_catalog(tmp_path, "south")
        schema = pa.schema([("id", pa.int64())])
        for catalog, ids in ((north, [1, 2]), (south, [10])):
            catalog.create_namespace("taxi")
            catalog.create_table("taxi.trips", schema=schema).append(pa.table({"id": ids}))
        first = north.load_table("taxi.trips").current_snapshot().snapshot_id
        north.load_table("taxi.trips").append(pa.table({"id": [3]}))
        config = _config(tmp_path, {"north": north_props, "south": south_props})

        assert _rows(config, "north:taxi.trips") == [1, 2, 3]
        assert _rows(config, "south:taxi.trips") == [10]
        assert _rows(config, "north:taxi.trips", snapshot_id=first) == [1, 2]

    def test_the_table_identity_names_the_catalog(self, tmp_path):
        props, catalog = _sql_catalog(tmp_path, "north")
        catalog.create_namespace("taxi")
        catalog.create_table("taxi.trips", schema=pa.schema([("id", pa.int64())])).append(
            pa.table({"id": [1]})
        )

        plan = ReadPlanner(_config(tmp_path, {"north": props})).plan("north:taxi.trips")

        assert plan.table_identity.catalog == "north"

    def test_a_table_annotation_resolves_through_the_named_catalog(self, tmp_path):
        from strata.notebook.models import TableSpec
        from strata.notebook.tables import resolve_table_snapshot

        props, catalog = _sql_catalog(tmp_path, "north")
        catalog.create_namespace("taxi")
        table = catalog.create_table("taxi.trips", schema=pa.schema([("id", pa.int64())]))
        table.append(pa.table({"id": [1]}))
        current = catalog.load_table("taxi.trips").current_snapshot().snapshot_id
        config = _config(tmp_path, {"north": props})

        spec = TableSpec(name="trips", uri="north:taxi.trips")
        assert resolve_table_snapshot(spec, config) == current

    def test_credentials_the_catalog_hands_the_table_reach_the_fetcher(self, tmp_path, monkeypatch):
        props, catalog = _sql_catalog(tmp_path, "north")
        catalog.create_namespace("taxi")
        catalog.create_table("taxi.trips", schema=pa.schema([("id", pa.int64())])).append(
            pa.table({"id": [1]})
        )
        seen = []
        monkeypatch.setattr(
            lake_files,
            "register_vended_credentials",
            lambda location, properties: seen.append((location, properties)),
        )

        ReadPlanner(_config(tmp_path, {"north": {**props, "s3.access-key-id": "vended"}})).plan(
            "north:taxi.trips"
        )

        ((location, properties),) = seen
        assert location.endswith("taxi/trips")
        assert properties["s3.access-key-id"] == "vended"

    def test_a_catalog_provider_can_be_injected(self, tmp_path):
        props, catalog = _sql_catalog(tmp_path, "north")
        catalog.create_namespace("taxi")
        catalog.create_table("taxi.trips", schema=pa.schema([("id", pa.int64())])).append(
            pa.table({"id": [1]})
        )
        from strata.iceberg import PyIcebergCatalog

        config = _config(tmp_path, {"north": props})
        calls = []

        class Recording(PyIcebergCatalog):
            def load_table(self, table_uri):
                calls.append(table_uri)
                return super().load_table(table_uri)

        ReadPlanner(config, catalog=Recording(config)).plan("north:taxi.trips")

        assert calls == ["north:taxi.trips"]


@windows
class TestANamedCredential:
    """A catalog names a credential, so its config carries no secret."""

    @staticmethod
    def _configured(tmp_path, credentials):
        props, catalog = _sql_catalog(tmp_path, "north")
        catalog.create_namespace("taxi")
        catalog.create_table("taxi.trips", schema=pa.schema([("id", pa.int64())])).append(
            pa.table({"id": [1, 2]})
        )
        uri = props.pop("uri")
        cache = tmp_path / "cache"
        cache.mkdir(exist_ok=True)
        config = StrataConfig(
            cache_dir=cache,
            catalogs={"north": {**props, "credential": "north-db"}},
            notebook_credentials=credentials,
        )
        return config, uri

    def test_the_credential_supplies_the_catalog_s_properties(self, tmp_path, monkeypatch):
        config, uri = self._configured(
            tmp_path, {"north-db": {"uri": "${NORTH_URI}", "s3.access-key-id": "${NORTH_KEY}"}}
        )
        monkeypatch.setenv("NORTH_URI", uri)
        monkeypatch.setenv("NORTH_KEY", "from-the-vault")
        seen = []
        monkeypatch.setattr(
            lake_files,
            "register_vended_credentials",
            lambda location, properties: seen.append(properties),
        )

        assert _rows(config, "north:taxi.trips") == [1, 2]
        assert "uri" not in config.catalogs["north"]
        assert seen[0]["s3.access-key-id"] == "from-the-vault"
        assert "credential" not in seen[0]

    def test_a_notebook_s_env_resolves_it_and_never_reaches_another_reader(
        self, tmp_path, monkeypatch
    ):
        from strata.notebook.models import TableSpec
        from strata.notebook.tables import resolve_table_snapshot

        config, server_uri = self._configured(tmp_path, {"north-db": {"uri": "${NORTH_URI}"}})
        notebook_uris, snapshots = [], []
        for name, ids in (("east", [5]), ("west", [6, 7])):
            warehouse = tmp_path / name
            warehouse.mkdir()
            uri = f"sqlite:///{warehouse / 'catalog.db'}"
            # Named north: a SQL catalog finds only rows written under its own name.
            catalog = SqlCatalog("north", uri=uri, warehouse=warehouse.as_uri())
            catalog.create_namespace("taxi")
            table = catalog.create_table("taxi.trips", schema=pa.schema([("id", pa.int64())]))
            table.append(pa.table({"id": ids}))
            notebook_uris.append(uri)
            snapshots.append(catalog.load_table("taxi.trips").current_snapshot().snapshot_id)
        spec = TableSpec(name="trips", uri="north:taxi.trips")
        monkeypatch.delenv("NORTH_URI", raising=False)

        for uri, snapshot in zip(notebook_uris, snapshots, strict=True):
            assert resolve_table_snapshot(spec, config, {"NORTH_URI": uri}) == snapshot
        # A scan (no notebook) still resolves against the server's env.
        monkeypatch.setenv("NORTH_URI", server_uri)
        assert _rows(config, "north:taxi.trips") == [1, 2]

    def test_a_missing_credential_fails_naming_it(self, tmp_path):
        from strata.notebook.models import TableSpec
        from strata.notebook.tables import resolve_table_snapshot

        config, _ = self._configured(tmp_path, {})

        with pytest.raises(ValueError, match="credential 'north-db' is not defined"):
            resolve_table_snapshot(TableSpec(name="trips", uri="north:taxi.trips"), config)


class TestFileRouting:
    def test_vended_credentials_read_files_under_their_location(self, monkeypatch):
        opened = []
        monkeypatch.setattr(
            lake_files.pq,
            "ParquetFile",
            lambda path, filesystem=None, metadata=None: opened.append((path, filesystem)),
        )
        default = object()

        assert lake_files.register_vended_credentials("s3://lake/wh/t", {"s3.region": "x"}) is False
        assert lake_files.register_vended_credentials(
            "s3://lake/wh/t",
            {"s3.access-key-id": "a", "s3.secret-access-key": "b", "s3.region": "us-east-1"},
        )
        lake_files.open_parquet("s3://lake/wh/t/data/f.parquet", default)
        lake_files.open_parquet("s3://lake/wh/other/f.parquet", default)

        assert opened[0][0] == "lake/wh/t/data/f.parquet"
        assert opened[0][1] is not default
        assert opened[1][1] is default

    def test_gcs_and_azure_files_open_through_their_filesystems(self, monkeypatch, tmp_path):
        try:
            lake_files.pafs.AzureFileSystem  # noqa: B018
        except ImportError:
            pytest.skip("this pyarrow build has no AzureFileSystem")
        made = []
        monkeypatch.setattr(
            lake_files.pafs, "GcsFileSystem", lambda **kw: made.append(("gcs", kw)) or "gcs"
        )
        monkeypatch.setattr(
            lake_files.pafs, "AzureFileSystem", lambda **kw: made.append(("az", kw)) or "az"
        )
        opened = []
        monkeypatch.setattr(
            lake_files.pq,
            "ParquetFile",
            lambda path, filesystem=None, metadata=None: opened.append((path, filesystem)),
        )
        lake_files.configure(
            SimpleNamespace(
                gcs_credentials_json=None,
                gcs_anonymous=True,
                gcs_endpoint_override="http://127.0.0.1:4443",
                azure_account_name="acct",
                azure_account_key="key",
                azure_sas_token=None,
                azure_endpoint_url="http://127.0.0.1:10000",
            )
        )

        lake_files.open_parquet("gs://lake/wh/f.parquet")
        lake_files.open_parquet("abfss://data@other.dfs.core.windows.net/wh/f.parquet")
        lake_files.open_parquet("az://data/wh/f.parquet")

        assert opened == [
            ("lake/wh/f.parquet", "gcs"),
            ("data/wh/f.parquet", "az"),
            ("data/wh/f.parquet", "az"),
        ]
        assert made[0] == (
            "gcs",
            {"anonymous": True, "endpoint_override": "127.0.0.1:4443", "scheme": "http"},
        )
        assert made[1][1]["account_name"] == "other"
        assert made[2][1] == {
            "account_name": "acct",
            "account_key": "key",
            "blob_storage_authority": "127.0.0.1:10000",
            "blob_storage_scheme": "http",
        }


class TestAclNamesTheCatalog:
    """A table in a configured catalog is authorized under that catalog's name.

    A rule for one catalog must not silently cover another's table.
    """

    @staticmethod
    def _ref(uri: str):
        from strata.config import StrataConfig
        from strata.iceberg import named_catalog, table_identity_for
        from strata.types import TableRef

        config = StrataConfig(catalogs={"lake": {"type": "rest", "uri": "http://catalog"}})
        named, _ = named_catalog(uri, config)
        return TableRef.from_table_identity(
            table_identity_for(uri, config), table_uri=uri, named_catalog_name=named
        )

    def test_a_catalogs_table_carries_its_name(self):
        assert str(self._ref("lake:taxi.trips")) == "lake:taxi.trips"

    def test_a_warehouse_uri_keeps_the_scheme_rules_are_written_for(self):
        assert str(self._ref("s3://bucket/wh#taxi.trips")) == "s3:taxi.trips"
        assert str(self._ref("/data/wh#taxi.trips")) == "file:taxi.trips"

    def test_a_rule_for_a_warehouse_does_not_reach_the_catalogs_table(self):
        from strata.auth import AclEvaluator
        from strata.config import _parse_acl_config
        from strata.types import Principal

        acl = _parse_acl_config(
            {"default": "deny", "allow_rules": [{"principal": "*", "tables": ["file:taxi.*"]}]}
        )
        principal = Principal(id="scientist", scopes=frozenset({"artifacts:write"}))

        assert AclEvaluator(acl).authorize(principal, self._ref("/data/wh#taxi.trips")) is True
        assert AclEvaluator(acl).authorize(principal, self._ref("lake:taxi.trips")) is False
