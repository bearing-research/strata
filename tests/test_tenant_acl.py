"""Tests for tenant-based access control in service mode.

These tests verify:
1. Tenant isolation for artifacts - different tenants can't see each other's artifacts
2. Tenant-scoped names - same name can exist for different tenants
3. Tenant-scoped provenance lookup - deduplication is tenant-isolated
4. Build status access control - builds can only be accessed by owner tenant
5. Error handling - hide forbidden as not found when configured
"""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import strata.server as server_module
from strata.artifact_store import (
    ArtifactStore,
    get_artifact_store,
    reset_artifact_store,
)
from strata.config import StrataConfig
from strata.tenant_acl import (
    TenantAccessError,
    authorize_resource_tenant,
    get_tenant_context,
    require_tenant,
    set_principal,
    tenant_scoped_lookup,
)
from strata.types import Principal


@pytest.fixture
def artifact_dir(tmp_path):
    """Create a temporary artifact directory."""
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir(parents=True)
    return artifact_path


@pytest.fixture
def artifact_store(artifact_dir):
    """Create a temporary artifact store."""
    reset_artifact_store()
    store = get_artifact_store(artifact_dir)
    yield store
    reset_artifact_store()


@pytest.fixture
def personal_config():
    """Config for personal mode (no tenant isolation)."""
    return StrataConfig.load(deployment_mode="personal")


@pytest.fixture
def service_config_no_auth():
    """Config for service mode without auth (no tenant isolation)."""
    return StrataConfig.load(deployment_mode="service", auth_mode="none")


@pytest.fixture
def service_config_with_auth():
    """Config for service mode with trusted proxy auth (tenant isolation enabled)."""
    return StrataConfig.load(
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        hide_forbidden_as_not_found=True,
    )


@pytest.fixture
def service_config_show_forbidden():
    """Config for service mode that returns 403 instead of 404."""
    return StrataConfig.load(
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        hide_forbidden_as_not_found=False,
    )


@pytest.fixture
def direct_artifact_client(artifact_dir):
    """Create a test client for direct artifact endpoints with trusted-proxy auth.

    The combination personal + trusted_proxy is rejected at config construction
    (validate_mode_coherence) because it's nonsensical in production. These
    tests still need it because delete/gc are personal-mode-only endpoints yet
    we want coverage that they respect tenant scope *when* auth happens to be
    on. We build a valid service-mode config and then mutate into personal via
    model_copy, which skips validators — fine for test-only invariant probing.
    """
    from strata.server import app

    base = StrataConfig.load(
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        artifact_dir=artifact_dir,
        hide_forbidden_as_not_found=True,
    )
    config = base.model_copy(update={"deployment_mode": "personal"})

    get_artifact_store(artifact_dir)

    mock_state = MagicMock()
    mock_state.config = config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()

    original_state = server_module._state
    server_module._state = mock_state

    yield TestClient(app)

    server_module._state = original_state


def _auth_headers(
    tenant: str,
    principal: str = "user-1",
    scopes: str | None = None,
) -> dict[str, str]:
    headers = {
        "X-Strata-Proxy-Token": "test-token",
        "X-Strata-Principal": principal,
        "X-Tenant-ID": tenant,
    }
    if scopes:
        headers["X-Strata-Scopes"] = scopes
    return headers


def _create_ready_artifact_for_tenant(
    store,
    artifact_id: str,
    tenant: str,
    blob: bytes = b"data",
) -> int:
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=f"hash-{artifact_id}",
        tenant=tenant,
    )
    store.write_blob(artifact_id, version, blob)
    store.finalize_artifact(artifact_id, version, "{}", 1, len(blob))
    return version


class TestRequireTenant:
    """Tests for require_tenant function."""

    def test_personal_mode_returns_personal(self, personal_config):
        """Personal mode should return __personal__ tenant."""
        tenant = require_tenant(personal_config)
        assert tenant == "__personal__"

    def test_service_mode_no_auth_returns_anonymous(self, service_config_no_auth):
        """Service mode without auth should return __anonymous__ tenant."""
        tenant = require_tenant(service_config_no_auth)
        assert tenant == "__anonymous__"

    def test_service_mode_with_auth_requires_principal(self, service_config_with_auth):
        """Service mode with auth should require principal."""
        set_principal(None)
        with pytest.raises(TenantAccessError) as exc_info:
            require_tenant(service_config_with_auth)
        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.message

    def test_service_mode_requires_tenant_header(self, service_config_with_auth):
        """Service mode should require tenant from principal."""
        principal = Principal(id="user-1", tenant=None)
        set_principal(principal)
        with pytest.raises(TenantAccessError) as exc_info:
            require_tenant(service_config_with_auth)
        assert exc_info.value.status_code == 400
        assert "Tenant header required" in exc_info.value.message

    def test_service_mode_returns_tenant(self, service_config_with_auth):
        """Service mode should return tenant from principal."""
        principal = Principal(id="user-1", tenant="team-a")
        set_principal(principal)
        tenant = require_tenant(service_config_with_auth)
        assert tenant == "team-a"


class TestAuthorizeResourceTenant:
    """Tests for authorize_resource_tenant function."""

    def test_personal_mode_allows_all(self, personal_config):
        """Personal mode should allow access to all resources."""
        # No principal needed
        set_principal(None)
        # Should not raise
        authorize_resource_tenant(personal_config, "any-tenant", "artifact")

    def test_service_mode_no_auth_allows_all(self, service_config_no_auth):
        """Service mode without auth should allow access to all resources."""
        set_principal(None)
        # Should not raise
        authorize_resource_tenant(service_config_no_auth, "any-tenant", "artifact")

    def test_service_mode_matching_tenant_allowed(self, service_config_with_auth):
        """Service mode should allow access when tenants match."""
        principal = Principal(id="user-1", tenant="team-a")
        set_principal(principal)
        # Should not raise
        authorize_resource_tenant(service_config_with_auth, "team-a", "artifact")

    def test_service_mode_mismatched_tenant_forbidden(self, service_config_with_auth):
        """Service mode should deny access when tenants don't match."""
        principal = Principal(id="user-1", tenant="team-a")
        set_principal(principal)
        with pytest.raises(TenantAccessError) as exc_info:
            authorize_resource_tenant(service_config_with_auth, "team-b", "artifact")
        # Should return 404 to hide resource existence
        assert exc_info.value.status_code == 404

    def test_service_mode_mismatched_tenant_shows_403(self, service_config_show_forbidden):
        """Service mode with hide_forbidden_as_not_found=False should return 403."""
        principal = Principal(id="user-1", tenant="team-a")
        set_principal(principal)
        with pytest.raises(TenantAccessError) as exc_info:
            authorize_resource_tenant(service_config_show_forbidden, "team-b", "artifact")
        assert exc_info.value.status_code == 403

    def test_admin_scope_bypasses_tenant_check(self, service_config_with_auth):
        """Admin scope should bypass tenant checks."""
        principal = Principal(id="admin-user", tenant="team-a", scopes=frozenset(["admin:*"]))
        set_principal(principal)
        # Should not raise even though tenant doesn't match
        authorize_resource_tenant(service_config_with_auth, "team-b", "artifact")

    def test_null_resource_tenant_allowed(self, service_config_with_auth):
        """Resources with no tenant (legacy) should be accessible."""
        principal = Principal(id="user-1", tenant="team-a")
        set_principal(principal)
        # Should not raise for null tenant
        authorize_resource_tenant(service_config_with_auth, None, "artifact")


class TestTenantScopedLookup:
    """Tests for tenant_scoped_lookup predicate."""

    def test_personal_mode_always_true(self, personal_config):
        """Personal mode should always return True."""
        set_principal(None)
        assert tenant_scoped_lookup(None, "any-tenant", personal_config)
        assert tenant_scoped_lookup("team-a", "team-b", personal_config)

    def test_matching_tenants(self, service_config_with_auth):
        """Matching tenants should return True."""
        principal = Principal(id="user-1", tenant="team-a")
        set_principal(principal)
        assert tenant_scoped_lookup("team-a", "team-a", service_config_with_auth)

    def test_mismatched_tenants(self, service_config_with_auth):
        """Mismatched tenants should return False."""
        principal = Principal(id="user-1", tenant="team-a")
        set_principal(principal)
        assert not tenant_scoped_lookup("team-a", "team-b", service_config_with_auth)

    def test_admin_bypasses(self, service_config_with_auth):
        """Admin scope should bypass tenant check."""
        principal = Principal(id="admin-user", tenant="team-a", scopes=frozenset(["admin:*"]))
        set_principal(principal)
        assert tenant_scoped_lookup("team-a", "team-b", service_config_with_auth)


class TestTenantContext:
    """Tests for TenantContext class."""

    def test_tenant_context_properties(self, service_config_with_auth):
        """TenantContext should expose principal properties."""
        principal = Principal(id="user-1", tenant="team-a", scopes=frozenset(["scan:create"]))
        set_principal(principal)
        ctx = get_tenant_context(service_config_with_auth)

        assert ctx.tenant_id == "team-a"
        assert ctx.principal_id == "user-1"
        assert not ctx.is_admin
        assert ctx.requires_tenant

    def test_admin_context(self, service_config_with_auth):
        """TenantContext should detect admin scope."""
        principal = Principal(id="admin-user", tenant="team-a", scopes=frozenset(["admin:*"]))
        set_principal(principal)
        ctx = get_tenant_context(service_config_with_auth)

        assert ctx.is_admin

    def test_context_authorize(self, service_config_with_auth):
        """TenantContext.authorize should work."""
        principal = Principal(id="user-1", tenant="team-a")
        set_principal(principal)
        ctx = get_tenant_context(service_config_with_auth)

        # Same tenant should not raise
        ctx.authorize("team-a")

        # Different tenant should raise
        with pytest.raises(TenantAccessError):
            ctx.authorize("team-b")

    def test_context_can_access(self, service_config_with_auth):
        """TenantContext.can_access predicate should work."""
        principal = Principal(id="user-1", tenant="team-a")
        set_principal(principal)
        ctx = get_tenant_context(service_config_with_auth)

        assert ctx.can_access("team-a")
        assert not ctx.can_access("team-b")


class TestArtifactStoreTenantIsolation:
    """Tests for artifact store tenant isolation."""

    def test_create_artifact_with_tenant(self, artifact_store):
        """Artifacts can be created with tenant."""
        version = artifact_store.create_artifact(
            artifact_id="art-1",
            provenance_hash="hash-1",
            tenant="team-a",
            principal="user-1",
        )
        assert version == 1

        artifact = artifact_store.get_artifact("art-1", 1)
        assert artifact.tenant == "team-a"
        assert artifact.principal == "user-1"

    def test_find_by_provenance_tenant_isolated(self, artifact_store):
        """Provenance lookup should be tenant-isolated when tenant is provided."""
        # Create artifacts with same provenance for different tenants
        artifact_store.create_artifact(
            artifact_id="art-a",
            provenance_hash="same-hash",
            tenant="team-a",
        )
        artifact_store.create_artifact(
            artifact_id="art-b",
            provenance_hash="same-hash",
            tenant="team-b",
        )

        # Finalize both
        artifact_store.finalize_artifact("art-a", 1, "{}", 10, 100)
        artifact_store.finalize_artifact("art-b", 1, "{}", 10, 100)

        # Tenant-scoped lookup should return correct artifact
        result_a = artifact_store.find_by_provenance("same-hash", tenant="team-a")
        assert result_a is not None
        assert result_a.id == "art-a"
        assert result_a.tenant == "team-a"

        result_b = artifact_store.find_by_provenance("same-hash", tenant="team-b")
        assert result_b is not None
        assert result_b.id == "art-b"
        assert result_b.tenant == "team-b"

        # Non-scoped lookup returns any matching artifact
        result_any = artifact_store.find_by_provenance("same-hash")
        assert result_any is not None

    def test_names_tenant_scoped(self, artifact_store):
        """Names should be scoped by tenant."""
        # Create artifacts for different tenants
        artifact_store.create_artifact(
            artifact_id="art-a",
            provenance_hash="hash-a",
            tenant="team-a",
        )
        artifact_store.finalize_artifact("art-a", 1, "{}", 10, 100)

        artifact_store.create_artifact(
            artifact_id="art-b",
            provenance_hash="hash-b",
            tenant="team-b",
        )
        artifact_store.finalize_artifact("art-b", 1, "{}", 10, 100)

        # Same name for different tenants
        artifact_store.set_name("report", "art-a", 1, tenant="team-a")
        artifact_store.set_name("report", "art-b", 1, tenant="team-b")

        # Resolve name for each tenant
        result_a = artifact_store.resolve_name("report", tenant="team-a")
        assert result_a is not None
        assert result_a.id == "art-a"

        result_b = artifact_store.resolve_name("report", tenant="team-b")
        assert result_b is not None
        assert result_b.id == "art-b"

    def test_set_name_rejects_cross_tenant_artifact(self, artifact_store):
        """A tenant-scoped name cannot point at another tenant's artifact."""
        artifact_store.create_artifact(
            artifact_id="art-a",
            provenance_hash="hash-a",
            tenant="team-a",
        )
        artifact_store.finalize_artifact("art-a", 1, "{}", 10, 100)

        with pytest.raises(ValueError, match="belongs to tenant team-a"):
            artifact_store.set_name("report", "art-a", 1, tenant="team-b")

        assert artifact_store.resolve_name("report", tenant="team-b") is None

    def test_get_name_tenant_scoped(self, artifact_store):
        """get_name should respect tenant scope."""
        artifact_store.create_artifact(
            artifact_id="art-1",
            provenance_hash="hash-1",
        )
        artifact_store.finalize_artifact("art-1", 1, "{}", 10, 100)

        # Create name with tenant
        artifact_store.set_name("myreport", "art-1", 1, tenant="team-a")

        # Get name with matching tenant
        name = artifact_store.get_name("myreport", tenant="team-a")
        assert name is not None
        assert name.tenant == "team-a"

        # Get name with wrong tenant should return None
        name = artifact_store.get_name("myreport", tenant="team-b")
        assert name is None

    def test_list_names_tenant_filtered(self, artifact_store):
        """list_names should filter by tenant."""
        artifact_store.create_artifact(
            artifact_id="art-1",
            provenance_hash="hash-1",
        )
        artifact_store.finalize_artifact("art-1", 1, "{}", 10, 100)

        # Create names for different tenants
        artifact_store.set_name("report-a", "art-1", 1, tenant="team-a")
        artifact_store.set_name("report-b", "art-1", 1, tenant="team-b")

        # List should filter by tenant
        names_a = artifact_store.list_names(tenant="team-a")
        assert len(names_a) == 1
        assert names_a[0].name == "report-a"

        names_b = artifact_store.list_names(tenant="team-b")
        assert len(names_b) == 1
        assert names_b[0].name == "report-b"

    def test_delete_name_tenant_scoped(self, artifact_store):
        """delete_name should respect tenant scope."""
        artifact_store.create_artifact(
            artifact_id="art-1",
            provenance_hash="hash-1",
        )
        artifact_store.finalize_artifact("art-1", 1, "{}", 10, 100)

        # Create name for team-a
        artifact_store.set_name("report", "art-1", 1, tenant="team-a")

        # Delete with wrong tenant should fail
        deleted = artifact_store.delete_name("report", tenant="team-b")
        assert not deleted

        # Delete with correct tenant should succeed
        deleted = artifact_store.delete_name("report", tenant="team-a")
        assert deleted


class TestMigration:
    """Tests for schema migration."""

    def test_new_database_has_tenant_columns(self, tmp_path):
        """Fresh database should have tenant columns."""
        artifact_dir = tmp_path / "fresh_artifacts"
        artifact_dir.mkdir()
        store = ArtifactStore(artifact_dir)

        # Create artifact with tenant
        version = store.create_artifact(
            artifact_id="test",
            provenance_hash="hash",
            tenant="my-tenant",
            principal="user-1",
        )

        artifact = store.get_artifact("test", version)
        assert artifact is not None
        assert artifact.tenant == "my-tenant"


class TestDirectArtifactEndpointIsolation:
    """Tests for tenant isolation on direct artifact endpoints."""

    def test_get_artifact_info_and_data_are_tenant_scoped(
        self,
        artifact_store,
        direct_artifact_client,
    ):
        version = _create_ready_artifact_for_tenant(artifact_store, "team-a-art", "team-a", b"abc")

        info_resp = direct_artifact_client.get(
            f"/v1/artifacts/team-a-art/v/{version}",
            headers=_auth_headers("team-a"),
        )
        assert info_resp.status_code == 200

        denied_info = direct_artifact_client.get(
            f"/v1/artifacts/team-a-art/v/{version}",
            headers=_auth_headers("team-b"),
        )
        assert denied_info.status_code == 404

        denied_data = direct_artifact_client.get(
            f"/v1/artifacts/team-a-art/v/{version}/data",
            headers=_auth_headers("team-b"),
        )
        assert denied_data.status_code == 404

    def test_list_artifacts_is_filtered_by_tenant(
        self,
        artifact_store,
        direct_artifact_client,
    ):
        version_a = _create_ready_artifact_for_tenant(artifact_store, "team-a-art", "team-a")
        _create_ready_artifact_for_tenant(artifact_store, "team-b-art", "team-b")

        response = direct_artifact_client.get(
            "/v1/artifacts",
            headers=_auth_headers("team-a"),
        )

        assert response.status_code == 200
        artifacts = response.json()["artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0]["artifact_uri"] == f"strata://artifact/team-a-art@v={version_a}"
        assert artifacts[0]["artifact_id"] == "team-a-art"
        assert artifacts[0]["version"] == version_a
        assert artifacts[0]["state"] == "ready"
        assert artifacts[0]["row_count"] == 1
        assert artifacts[0]["byte_size"] == 4
        assert artifacts[0]["created_at"] > 0

    def test_delete_artifact_is_tenant_scoped(
        self,
        artifact_store,
        direct_artifact_client,
    ):
        version = _create_ready_artifact_for_tenant(artifact_store, "team-a-art", "team-a")

        denied = direct_artifact_client.delete(
            f"/v1/artifacts/team-a-art/v/{version}",
            headers=_auth_headers("team-b"),
        )
        assert denied.status_code == 404
        assert artifact_store.get_artifact("team-a-art", version) is not None

        allowed = direct_artifact_client.delete(
            f"/v1/artifacts/team-a-art/v/{version}",
            headers=_auth_headers("team-a"),
        )
        assert allowed.status_code == 200
        assert artifact_store.get_artifact("team-a-art", version) is None

    def test_stats_and_usage_are_tenant_scoped(
        self,
        artifact_store,
        direct_artifact_client,
    ):
        version_a = _create_ready_artifact_for_tenant(
            artifact_store,
            "team-a-art",
            "team-a",
            b"aaaa",
        )
        version_b = _create_ready_artifact_for_tenant(
            artifact_store,
            "team-b-art",
            "team-b",
            b"bbbbbb",
        )
        artifact_store.set_name("team-a-name", "team-a-art", version_a, tenant="team-a")
        artifact_store.set_name("team-b-name", "team-b-art", version_b, tenant="team-b")

        stats_response = direct_artifact_client.get(
            "/v1/artifacts/stats",
            headers=_auth_headers("team-a"),
        )
        assert stats_response.status_code == 200
        assert stats_response.json() == {
            "total_versions": 1,
            "ready_versions": 1,
            "building_versions": 0,
            "failed_versions": 0,
            "total_bytes": 4,
            "total_rows": 1,
            "name_count": 1,
        }

        usage_response = direct_artifact_client.get(
            "/v1/artifacts/usage",
            headers=_auth_headers("team-a"),
        )
        assert usage_response.status_code == 200
        usage = usage_response.json()
        assert usage["unique_artifacts"] == 1
        assert usage["total_versions"] == 1
        assert usage["ready_versions"] == 1
        assert usage["building_versions"] == 0
        assert usage["failed_versions"] == 0
        assert usage["total_bytes"] == 4
        assert usage["total_rows"] == 1
        assert usage["name_count"] == 1
        assert usage["unreferenced_count"] == 0

    def test_garbage_collect_is_tenant_scoped(
        self,
        artifact_store,
        direct_artifact_client,
    ):
        version_a = _create_ready_artifact_for_tenant(
            artifact_store,
            "team-a-art",
            "team-a",
            b"aaaa",
        )
        version_b = _create_ready_artifact_for_tenant(
            artifact_store,
            "team-b-art",
            "team-b",
            b"bbbbbb",
        )

        response = direct_artifact_client.post(
            "/v1/artifacts/gc",
            params={"max_age_days": 0},
            headers=_auth_headers("team-a"),
        )
        assert response.status_code == 200
        assert response.json()["deleted_count"] == 1
        assert response.json()["deleted_bytes"] == 4

        assert artifact_store.get_artifact("team-a-art", version_a) is None
        assert artifact_store.get_artifact("team-b-art", version_b) is not None


def _create_scan_artifact(store, artifact_id, tenant, table_uri, blob=b"arrow-bytes"):
    """A ready scan@v1 artifact whose single input is a table URI."""
    from strata.artifact_store import TransformSpec

    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=f"hash-{artifact_id}",
        transform_spec=TransformSpec(executor="scan@v1", params={}, inputs=[table_uri]),
        tenant=tenant,
    )
    store.write_blob(artifact_id, version, blob)
    store.finalize_artifact(artifact_id, version, "{}", 1, len(blob))
    return version


@contextmanager
def _service_read_app(artifact_dir, *, auth, acl=None, hide_as_404=False):
    """A genuine service-mode TestClient (NOT mutated to personal) for read tests."""
    from strata.config import StrataConfig
    from strata.server import app

    cache_dir = artifact_dir.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    kwargs = dict(
        deployment_mode="service",
        artifact_dir=artifact_dir,
        cache_dir=cache_dir,
        hide_forbidden_as_not_found=hide_as_404,
    )
    if auth:
        kwargs.update(auth_mode="trusted_proxy", proxy_token="test-token")
    if acl is not None:
        kwargs["acl_config"] = acl
    config = StrataConfig(**kwargs)

    reset_artifact_store()
    store = get_artifact_store(artifact_dir)

    mock_state = MagicMock()
    mock_state.config = config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()

    original = server_module._state
    server_module._state = mock_state
    try:
        yield TestClient(app), store
    finally:
        server_module._state = original
        reset_artifact_store()


class TestServiceModeArtifactReads:
    """Reads work in genuine service mode (writes_enabled=False), gated by tenant
    + the table ACL of the artifact's inputs.

    Regression: ``/data`` and artifact metadata used to 403 in service mode
    (gated as writes), so a materialized result — including an identity-scan
    cache hit — was unreadable. The existing tenant-acl coverage had to mutate
    the config to personal mode to exercise these endpoints; these run in real
    service mode.
    """

    def test_data_readable_in_service_mode_no_auth(self, artifact_dir):
        with _service_read_app(artifact_dir, auth=False) as (client, store):
            v = _create_scan_artifact(store, "art-1", tenant=None, table_uri="file:///wh#db.t")
            resp = client.get(f"/v1/artifacts/art-1/v/{v}/data")
            assert resp.status_code == 200
            assert resp.content == b"arrow-bytes"

    def test_metadata_readable_in_service_mode_no_auth(self, artifact_dir):
        with _service_read_app(artifact_dir, auth=False) as (client, store):
            v = _create_scan_artifact(store, "art-1", tenant=None, table_uri="file:///wh#db.t")
            resp = client.get(f"/v1/artifacts/art-1/v/{v}")
            assert resp.status_code == 200
            assert resp.json()["state"] == "ready"

    def test_data_denied_table_rejected_with_auth(self, artifact_dir):
        from strata.config import AclConfig, AclRule

        acl = AclConfig(
            default="deny",
            deny_rules=[],
            allow_rules=[AclRule(principal="analyst", tables=("file:public.*",))],
        )
        with _service_read_app(artifact_dir, auth=True, acl=acl) as (client, store):
            v = _create_scan_artifact(
                store, "art-1", tenant="team-a", table_uri="file:///wh#secret.events"
            )
            resp = client.get(
                f"/v1/artifacts/art-1/v/{v}/data",
                headers=_auth_headers("team-a", principal="intruder"),
            )
            assert resp.status_code == 403

    def test_data_allowed_table_passes_with_auth(self, artifact_dir):
        from strata.config import AclConfig, AclRule

        acl = AclConfig(
            default="deny",
            deny_rules=[],
            allow_rules=[AclRule(principal="analyst", tables=("file:public.*",))],
        )
        with _service_read_app(artifact_dir, auth=True, acl=acl) as (client, store):
            v = _create_scan_artifact(
                store, "art-1", tenant="team-a", table_uri="file:///wh#public.events"
            )
            resp = client.get(
                f"/v1/artifacts/art-1/v/{v}/data",
                headers=_auth_headers("team-a", principal="analyst"),
            )
            assert resp.status_code == 200
            assert resp.content == b"arrow-bytes"
