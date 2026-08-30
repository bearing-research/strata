"""Authorization under ``auth_mode="api_key"``.

Authentication and authorization are separate questions, and api_key mode
answered only the first: every gate asked ``auth_mode == "trusted_proxy"``, so
a valid key with no scopes reached admin endpoints and no tenant filter was
applied. These pin that a key is authorized the same way a proxy-asserted
principal is.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.config import StrataConfig


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A running api_key-mode service, with a key-minting helper."""
    monkeypatch.setenv("STRATA_DEPLOYMENT_MODE", "service")
    monkeypatch.setenv("STRATA_AUTH_MODE", "api_key")
    monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("STRATA_CACHE_DIR", str(tmp_path / "cache"))

    from strata.server import app

    with TestClient(app) as client:
        from strata.api_keys import get_api_key_store

        def mint(principal_id="svc", tenant=None, scopes=frozenset()):
            key, _ = get_api_key_store().create_key(
                principal_id=principal_id, tenant=tenant, scopes=scopes
            )
            return {"Authorization": f"Bearer {key}"}

        yield client, mint


class TestScopeEnforcement:
    """``require_scope`` must apply to a bearer key, not only a proxy header."""

    ADMIN_ROUTES = [("GET", "/v1/cache/entries"), ("POST", "/v1/cache/clear")]

    @pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
    def test_a_key_without_the_scope_is_refused(self, service, method, path):
        # The cache listing is deliberately cache-wide -- it returns every
        # tenant's table identities, snapshots and columns -- so an unscoped
        # key reaching it is a cross-tenant disclosure, not just an untidy 200.
        client, mint = service
        assert client.request(method, path, headers=mint(scopes=frozenset())).status_code == 403

    @pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
    def test_a_key_with_the_scope_is_allowed(self, service, method, path):
        client, mint = service
        headers = mint(scopes=frozenset({"admin:cache"}))
        assert client.request(method, path, headers=headers).status_code == 200

    @pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
    def test_admin_wildcard_still_satisfies_the_scope(self, service, method, path):
        client, mint = service
        headers = mint(scopes=frozenset({"admin:*"}))
        assert client.request(method, path, headers=headers).status_code == 200

    @pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
    def test_no_key_is_still_unauthenticated(self, service, method, path):
        # Authentication was never the broken half; this guards the boundary
        # between the two so a later change can't collapse them.
        client, _ = service
        assert client.request(method, path).status_code == 401


class TestTenantIsolation:
    def test_a_key_cannot_read_another_tenants_artifact(self, tmp_path, monkeypatch):
        # Multi-tenancy was rejected outright under api_key, which failed
        # closed -- but for the wrong reason, and it made the mode unusable
        # for the deployments that most need it.
        monkeypatch.setenv("STRATA_DEPLOYMENT_MODE", "service")
        monkeypatch.setenv("STRATA_AUTH_MODE", "api_key")
        monkeypatch.setenv("STRATA_MULTI_TENANT_ENABLED", "true")
        monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
        monkeypatch.setenv("STRATA_CACHE_DIR", str(tmp_path / "cache"))

        from strata.server import app

        with TestClient(app) as client:
            from strata.api_keys import get_api_key_store
            from strata.artifact_store import get_artifact_store

            keys = get_api_key_store()
            owner, _ = keys.create_key(principal_id="alice", tenant="team-a")
            other, _ = keys.create_key(principal_id="bob", tenant="team-b")

            store = get_artifact_store()
            version = store.create_artifact("report", provenance_hash="h", tenant="team-a")
            store.finalize_artifact("report", version, "{}", 1, 19)

            def read(key, tenant_header):
                return client.get(
                    f"/v1/artifacts/report/v/{version}",
                    headers={"Authorization": f"Bearer {key}", "X-Tenant-ID": tenant_header},
                ).status_code

            assert read(owner, "team-a") == 200
            assert read(other, "team-b") == 404
            # The filter comes from the key's principal, so asserting someone
            # else's tenant in the header buys nothing.
            assert read(other, "team-a") == 404


class TestPrincipalAuthEnabled:
    """The predicate the gates now share."""

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [("none", False), ("trusted_proxy", True), ("api_key", True)],
    )
    def test_truth_table(self, tmp_path, mode, expected):
        kwargs = {"deployment_mode": "service", "artifact_dir": tmp_path, "auth_mode": mode}
        if mode == "trusted_proxy":
            kwargs["proxy_token"] = "t"
        assert StrataConfig(**kwargs).principal_auth_enabled is expected


class TestWebSocketAuthentication:
    """The WS upgrade authenticates itself; no HTTP middleware runs for it.

    Driven through fake objects rather than ``TestClient.websocket_connect``,
    which deadlocks on Python 3.12 macOS CI in modules that share an app.
    """

    class _FakeWebSocket:
        def __init__(self, headers):
            self.headers = headers
            self.closed_with: int | None = None

        async def close(self, code=1000, reason=""):
            self.closed_with = code

    async def _authenticate(self, headers):
        from strata.notebook.ws import _authenticate_websocket

        ws = self._FakeWebSocket(headers)
        return await _authenticate_websocket(ws), ws

    @pytest.mark.asyncio
    async def test_an_upgrade_with_no_key_is_closed(self, service):
        # Left checking only for trusted_proxy, this returned True: every
        # api_key deployment's socket accepted anyone, and a socket can send
        # cell_execute -- arbitrary Python, no credential.
        ok, ws = await self._authenticate({})
        assert ok is False
        assert ws.closed_with == 1008

    @pytest.mark.asyncio
    async def test_an_upgrade_with_a_bad_key_is_closed(self, service):
        ok, ws = await self._authenticate({"authorization": "Bearer strata_dead_beef"})
        assert ok is False
        assert ws.closed_with == 1008

    @pytest.mark.asyncio
    async def test_an_upgrade_with_a_valid_key_is_accepted(self, service):
        _, mint = service
        ok, ws = await self._authenticate({k.lower(): v for k, v in mint().items()})
        assert ok is True
        assert ws.closed_with is None

    @pytest.mark.asyncio
    async def test_no_auth_deployments_are_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRATA_DEPLOYMENT_MODE", "personal")
        monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
        monkeypatch.setenv("STRATA_CACHE_DIR", str(tmp_path / "cache"))

        from strata.server import app

        with TestClient(app):
            ok, ws = await self._authenticate({})
        assert ok is True
        assert ws.closed_with is None


class TestAclUnderApiKey:
    def test_acl_rules_are_accepted_and_enforced(self, tmp_path):
        # Configuring rules under api_key was refused at startup, with a
        # message naming auth_mode='none' whatever the mode actually was.
        from strata.auth import AclEvaluator
        from strata.config import _parse_acl_config
        from strata.types import Principal, TableRef

        acl = _parse_acl_config(
            {"default": "allow", "deny": [{"principal": "*", "tables": ["file:finance.*"]}]}
        )
        config = StrataConfig(
            deployment_mode="service",
            auth_mode="api_key",
            artifact_dir=tmp_path,
            acl_config=acl,
        )
        denied = AclEvaluator(config.acl_config).authorize(
            Principal(id="analyst", tenant=None, scopes=frozenset({"tables:read"})),
            TableRef(catalog="file", namespace="finance", table="salaries"),
        )
        assert denied is False
