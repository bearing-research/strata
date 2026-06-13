"""Tests for deployment mode configuration and personal mode safety.

These tests verify:
1. deployment_mode defaults to "personal" (the common case)
2. Invalid deployment_mode raises ValueError
3. Personal mode creates artifact_dir
4. Personal mode binding to non-loopback is blocked unless explicitly allowed
5. writes_enabled property reflects deployment_mode
6. require_writes_enabled dependency blocks writes in service mode
"""

from typing import Any, cast

import pytest

from strata.config import StrataConfig


class TestDeploymentModeConfig:
    """Tests for deployment mode configuration."""

    def test_default_is_personal(self, tmp_path):
        """Default deployment_mode is 'personal' — the common case.

        First-time ``strata-notebook`` invocations boot single-user on
        loopback. Service mode is explicit opt-in."""
        config = StrataConfig(cache_dir=tmp_path / "cache")
        assert config.deployment_mode == "personal"
        assert config.writes_enabled is True

    def test_service_mode(self, tmp_path):
        """Service mode can be explicitly set."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
        )
        assert config.deployment_mode == "service"
        assert config.writes_enabled is False

    def test_invalid_mode_raises(self, tmp_path):
        """Invalid deployment_mode raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode=cast(Any, "invalid"),
            )
        # Pydantic validation error includes 'service' or 'personal' in message
        error_str = str(exc_info.value)
        assert "'service'" in error_str or "'personal'" in error_str

    def test_personal_mode_creates_artifact_dir(self, tmp_path):
        """Personal mode creates artifact_dir if not specified."""
        # Using a custom artifact_dir to avoid touching home directory
        artifact_dir = tmp_path / "artifacts"
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            artifact_dir=artifact_dir,
        )
        assert config.artifact_dir == artifact_dir
        assert artifact_dir.exists()

    def test_service_mode_no_artifact_dir(self, tmp_path):
        """Service mode does not create artifact_dir by default."""
        config = StrataConfig(cache_dir=tmp_path / "cache", deployment_mode="service")
        assert config.artifact_dir is None


class TestPersonalModeBinding:
    """Tests for personal mode binding safety."""

    def test_loopback_binding_allowed(self, tmp_path):
        """Personal mode allows loopback binding."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            host="127.0.0.1",
            artifact_dir=tmp_path / "artifacts",
        )
        # Should not raise
        config.validate_personal_mode_binding()

    def test_localhost_binding_allowed(self, tmp_path):
        """Personal mode allows localhost binding."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            host="localhost",
            artifact_dir=tmp_path / "artifacts",
        )
        # Should not raise
        config.validate_personal_mode_binding()

    def test_ipv6_loopback_allowed(self, tmp_path):
        """Personal mode allows IPv6 loopback binding."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            host="::1",
            artifact_dir=tmp_path / "artifacts",
        )
        # Should not raise
        config.validate_personal_mode_binding()

    def test_non_loopback_blocked(self, tmp_path):
        """Personal mode blocks non-loopback binding by default."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            host="0.0.0.0",
            artifact_dir=tmp_path / "artifacts",
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate_personal_mode_binding()
        assert "Personal mode binding" in str(exc_info.value)
        assert "unsafe" in str(exc_info.value)

    def test_external_ip_blocked(self, tmp_path):
        """Personal mode blocks external IP binding."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            host="192.168.1.100",
            artifact_dir=tmp_path / "artifacts",
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate_personal_mode_binding()
        assert "Personal mode binding" in str(exc_info.value)

    def test_non_loopback_allowed_with_override(self, tmp_path):
        """Personal mode allows non-loopback binding with explicit override."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            host="0.0.0.0",
            allow_remote_clients_in_personal=True,
            artifact_dir=tmp_path / "artifacts",
        )
        # Should not raise
        config.validate_personal_mode_binding()

    def test_service_mode_allows_any_binding(self, tmp_path):
        """Service mode allows any binding (no writes anyway)."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
            host="0.0.0.0",
        )
        # Should not raise - service mode is read-only
        config.validate_personal_mode_binding()


class TestEnvOverrides:
    """Tests for environment variable overrides."""

    def test_deployment_mode_from_env(self, tmp_path, monkeypatch):
        """STRATA_DEPLOYMENT_MODE overrides default."""
        monkeypatch.setenv("STRATA_DEPLOYMENT_MODE", "personal")
        config = StrataConfig.load(
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
        )
        assert config.deployment_mode == "personal"

    def test_allow_remote_from_env(self, tmp_path, monkeypatch):
        """STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL overrides default."""
        monkeypatch.setenv("STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL", "true")
        config = StrataConfig.load(cache_dir=tmp_path / "cache")
        assert config.allow_remote_clients_in_personal is True

    def test_artifact_dir_from_env(self, tmp_path, monkeypatch):
        """STRATA_ARTIFACT_DIR overrides default."""
        artifact_dir = tmp_path / "custom_artifacts"
        monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(artifact_dir))
        monkeypatch.setenv("STRATA_DEPLOYMENT_MODE", "personal")
        config = StrataConfig.load(cache_dir=tmp_path / "cache")
        assert config.artifact_dir == artifact_dir


class TestWritesEnabled:
    """Tests for writes_enabled property."""

    def test_service_mode_writes_disabled(self, tmp_path):
        """Service mode has writes disabled."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
        )
        assert config.writes_enabled is False

    def test_personal_mode_writes_enabled(self, tmp_path):
        """Personal mode has writes enabled."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )
        assert config.writes_enabled is True


class TestModeCoherence:
    """Tests for personal/service mode coherence validation."""

    def test_personal_with_trusted_proxy_rejected(self, tmp_path):
        """Personal mode + auth_mode='trusted_proxy' is incoherent."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                auth_mode="trusted_proxy",
            )
        assert "trusted_proxy" in str(exc_info.value)

    def test_personal_with_multi_tenant_rejected(self, tmp_path):
        """Personal mode + multi_tenant_enabled=True is incoherent."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                multi_tenant_enabled=True,
            )
        assert "multi_tenant_enabled" in str(exc_info.value)

    def test_personal_with_require_tenant_header_rejected(self, tmp_path):
        """Personal mode + require_tenant_header=True is incoherent."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                require_tenant_header=True,
            )
        assert "require_tenant_header" in str(exc_info.value)

    def test_personal_with_multiple_conflicts_lists_all(self, tmp_path):
        """When several service-mode flags leak into personal, all are listed."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                auth_mode="trusted_proxy",
                multi_tenant_enabled=True,
                require_tenant_header=True,
            )
        msg = str(exc_info.value)
        assert "trusted_proxy" in msg
        assert "multi_tenant_enabled" in msg
        assert "require_tenant_header" in msg

    def test_service_with_trusted_proxy_allowed(self, tmp_path):
        """Service mode + trusted_proxy is the normal hosted configuration."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="dummy",
        )
        assert config.auth_mode == "trusted_proxy"

    def test_service_with_multi_tenant_allowed(self, tmp_path):
        """Service mode + multi_tenant + trusted-proxy auth is the normal hosted
        configuration (multi-tenancy requires auth to be a real boundary)."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
            auth_mode="trusted_proxy",
            multi_tenant_enabled=True,
            require_tenant_header=True,
        )
        assert config.multi_tenant_enabled is True
        assert config.require_tenant_header is True

    def test_personal_with_defaults_allowed(self, tmp_path):
        """Personal mode with default auth/tenant settings passes."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )
        assert config.deployment_mode == "personal"
        assert config.auth_mode == "none"
        assert config.multi_tenant_enabled is False
        assert config.require_tenant_header is False

    def test_service_with_personal_user_header_rejected(self, tmp_path):
        """personal_mode_user_header is a personal-mode shim; reject in service mode."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="service",
                personal_mode_user_header="X-Auth-User",
            )
        assert "personal_mode_user_header" in str(exc_info.value)

    def test_personal_with_user_header_allowed(self, tmp_path):
        """personal_mode_user_header is the intended shape for proxy-fronted personal deploys."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            personal_mode_user_header="Cf-Access-Authenticated-User-Email",
        )
        assert config.personal_mode_user_header == "Cf-Access-Authenticated-User-Email"

    def test_service_acl_without_auth_rejected(self, tmp_path):
        """ACL rules with auth_mode='none' are silently inert — reject them."""
        from strata.config import AclConfig

        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="service",
                acl_config=AclConfig(default="deny"),
            )
        assert "acl_config" in str(exc_info.value)

    def test_service_acl_with_trusted_proxy_allowed(self, tmp_path):
        """ACL rules are coherent once trusted-proxy auth is on."""
        from strata.config import AclConfig

        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
            auth_mode="trusted_proxy",
            acl_config=AclConfig(default="deny"),
        )
        assert config.acl_config.default == "deny"

    def test_service_default_acl_without_auth_allowed(self, tmp_path):
        """The default (allow-all, no rules) ACL is not 'configured' — allowed."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
        )
        assert config.auth_mode == "none"

    def test_service_transforms_without_artifact_dir_rejected(self, tmp_path):
        """Transform builds persist artifacts; reject if no artifact store."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="service",
                transforms_config={"enabled": True},
            )
        assert "artifact_dir" in str(exc_info.value)

    def test_service_transforms_with_artifact_dir_allowed(self, tmp_path):
        """Transforms + an artifact store is the build-service configuration."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
            artifact_dir=tmp_path / "artifacts",
            transforms_config={"enabled": True},
        )
        assert config.server_transforms_enabled is True

    def test_service_multi_tenant_without_auth_rejected(self, tmp_path):
        """Multi-tenancy is an access-control boundary: without trusted-proxy
        auth the tenant header is spoofable and reads aren't tenant-filtered, so
        multi_tenant + auth='none' is rejected at startup."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="service",
                multi_tenant_enabled=True,
            )
        assert "multi_tenant_enabled" in str(exc_info.value)
