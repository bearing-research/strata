"""Deny-first ACL: one table, several ACL names.

AclEvaluator.authorize is deny-first by construction. What it cannot see
is that one physical table can be requested under several URIs, and
TableRef names the table by the *address form* (the store prefix, or a
named catalog), not by the table. Passes while the bypass exists.

    uv run pytest formal/ -v
"""

# Fixtures are imported from tests/ and then requested by name.
# ruff: noqa: F811

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import strata.server as server_module
from strata.api.dependencies import authorize_table_access
from strata.auth import set_principal
from strata.config import AclConfig, AclRule, StrataConfig
from strata.iceberg import PyIcebergCatalog, table_identity_for
from strata.types import Principal
from tests.conftest import temp_warehouse  # noqa: F401  (fixture)


@pytest.fixture
def service_config(temp_warehouse, tmp_path):
    """A service deployment whose catalog is one shared database.

    Setting ``catalog_properties["uri"]`` (Postgres in production, the
    fixture's SQLite here) is how a service deployment points Strata at a
    SQL catalog. Every warehouse URI then builds ``SqlCatalog("strata")``
    over that one database, whatever the path or scheme in front of ``#``,
    and a bare ``namespace.table`` reads the default catalog, the same
    database here.
    """
    return StrataConfig(
        deployment_mode="service",
        artifact_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "cache",
        auth_mode="trusted_proxy",
        proxy_token="t",
        catalog_name="strata",
        catalog_properties={"uri": temp_warehouse["catalog"].properties["uri"]},
        acl_config=AclConfig(
            default="allow",
            deny_rules=[AclRule(principal="*", tables=("s3:test_db.*",))],
        ),
    )


def _gate(config, monkeypatch):
    monkeypatch.setattr(server_module, "get_state", lambda: SimpleNamespace(config=config))
    set_principal(Principal(id="alice"))

    def check(uri):
        authorize_table_access(uri, table_identity_for(uri, config))

    return check


def test_a_deny_on_the_s3_name_is_bypassed_by_another_address(service_config, monkeypatch):
    """The data lives in S3 and the operator denies ``s3:test_db.*``.

    The S3 form of the URI is refused. The same table requested as a bare
    ``namespace.table``, or behind any local-looking path, is named
    ``file:test_db.events`` and allowed, and it loads the same table.
    """
    check = _gate(service_config, monkeypatch)
    denied = "s3://any-bucket/warehouse#test_db.events"
    aliases = ["test_db.events", "/not/a/real/path#test_db.events"]

    with pytest.raises(HTTPException) as refused:
        check(denied)
    assert refused.value.status_code in (403, 404)

    for uri in aliases:
        check(uri)  # no exception: allowed

    catalogs = PyIcebergCatalog(service_config)
    locations = {catalogs.load_table(uri).metadata_location for uri in [denied, *aliases]}
    assert len(locations) == 1  # one table behind all three names
