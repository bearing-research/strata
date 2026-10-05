"""Deployment mode configuration, mode coherence and personal-mode safety."""

from typing import Any, cast

import pytest

from strata.config import StrataConfig
from strata.server import _should_warn_unset_signing_secret


class TestDeploymentModeConfig:
    def test_default_is_personal(self, tmp_path):
        """First-time ``strata-notebook`` boots single-user on loopback; service mode is opt-in."""
        config = StrataConfig(cache_dir=tmp_path / "cache")
        assert config.deployment_mode == "personal"
        assert config.writes_enabled is True

    def test_service_mode(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
        )
        assert config.deployment_mode == "service"
        assert config.writes_enabled is False

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode=cast(Any, "invalid"),
            )
        error_str = str(exc_info.value)
        assert "'service'" in error_str or "'personal'" in error_str

    def test_personal_mode_creates_artifact_dir(self, tmp_path):
        # A custom artifact_dir keeps the test out of the home directory.
        artifact_dir = tmp_path / "artifacts"
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            artifact_dir=artifact_dir,
        )
        assert config.artifact_dir == artifact_dir
        assert artifact_dir.exists()

    def test_service_mode_no_artifact_dir(self, tmp_path):
        config = StrataConfig(cache_dir=tmp_path / "cache", deployment_mode="service")
        assert config.artifact_dir is None


class TestPersonalModeBinding:
    def test_loopback_binding_allowed(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            host="127.0.0.1",
            artifact_dir=tmp_path / "artifacts",
        )
        config.validate_personal_mode_binding()

    def test_localhost_binding_allowed(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            host="localhost",
            artifact_dir=tmp_path / "artifacts",
        )
        config.validate_personal_mode_binding()

    def test_ipv6_loopback_allowed(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            host="::1",
            artifact_dir=tmp_path / "artifacts",
        )
        config.validate_personal_mode_binding()

    def test_non_loopback_blocked(self, tmp_path):
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
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            host="0.0.0.0",
            allow_remote_clients_in_personal=True,
            artifact_dir=tmp_path / "artifacts",
        )
        config.validate_personal_mode_binding()

    def test_service_mode_allows_any_binding(self, tmp_path):
        """Service mode allows any binding (no writes anyway)."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
            host="0.0.0.0",
        )
        # Service mode is read-only, so any bind is allowed.
        config.validate_personal_mode_binding()


class TestEnvOverrides:
    def test_deployment_mode_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRATA_DEPLOYMENT_MODE", "personal")
        config = StrataConfig.load(
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
        )
        assert config.deployment_mode == "personal"

    def test_allow_remote_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL", "true")
        config = StrataConfig.load(cache_dir=tmp_path / "cache")
        assert config.allow_remote_clients_in_personal is True

    def test_artifact_dir_from_env(self, tmp_path, monkeypatch):
        artifact_dir = tmp_path / "custom_artifacts"
        monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(artifact_dir))
        monkeypatch.setenv("STRATA_DEPLOYMENT_MODE", "personal")
        config = StrataConfig.load(cache_dir=tmp_path / "cache")
        assert config.artifact_dir == artifact_dir


class TestWritesEnabled:
    def test_service_mode_writes_disabled(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
        )
        assert config.writes_enabled is False

    def test_personal_mode_writes_enabled(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )
        assert config.writes_enabled is True


class TestModeCoherence:
    def test_personal_with_trusted_proxy_rejected(self, tmp_path):
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                auth_mode="trusted_proxy",
            )
        assert "trusted_proxy" in str(exc_info.value)

    def test_personal_with_multi_tenant_rejected(self, tmp_path):
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                multi_tenant_enabled=True,
            )
        assert "multi_tenant_enabled" in str(exc_info.value)

    def test_personal_with_require_tenant_header_rejected(self, tmp_path):
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                require_tenant_header=True,
            )
        assert "require_tenant_header" in str(exc_info.value)

    def test_service_with_mcp_enabled_rejected(self, tmp_path):
        """MCP has no per-request auth."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="service",
                artifact_dir=tmp_path / "artifacts",
                mcp_enabled=True,
            )
        assert "mcp_enabled" in str(exc_info.value)

    def test_personal_ignores_a_leftover_user_header_env_var(self, tmp_path, monkeypatch):
        """The retired multi-user header is an unknown setting now, so MCP is not refused."""
        monkeypatch.setenv("STRATA_PERSONAL_MODE_USER_HEADER", "X-Auth-User")
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            mcp_enabled=True,
        )
        assert config.mcp_enabled is True
        assert not hasattr(config, "personal_mode_user_header")

    def test_personal_with_mcp_enabled_allowed(self, tmp_path):
        """Personal mode with MCP is the supported combination."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            mcp_enabled=True,
        )
        assert config.mcp_enabled is True

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
        """Multi-tenancy requires auth to be a real boundary."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            multi_tenant_enabled=True,
            require_tenant_header=True,
        )
        assert config.multi_tenant_enabled is True
        assert config.require_tenant_header is True

    def test_personal_with_defaults_allowed(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )
        assert config.deployment_mode == "personal"
        assert config.auth_mode == "none"
        assert config.multi_tenant_enabled is False
        assert config.require_tenant_header is False

    def test_service_acl_without_auth_rejected(self, tmp_path):
        """ACL rules with auth_mode='none' would be silently inert."""
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
            proxy_token="test-token",
            acl_config=AclConfig(default="deny"),
        )
        assert config.acl_config.default == "deny"

    def test_service_default_acl_without_auth_allowed(self, tmp_path):
        """The default ACL (allow-all, no rules) does not count as configured."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
        )
        assert config.auth_mode == "none"

    def test_service_transforms_without_artifact_dir_rejected(self, tmp_path):
        """Transform builds persist artifacts, so they need a store."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="service",
                transforms_config={"enabled": True},
            )
        assert "artifact_dir" in str(exc_info.value)

    def test_service_transforms_with_artifact_dir_allowed(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            deployment_mode="service",
            artifact_dir=tmp_path / "artifacts",
            transforms_config={"enabled": True},
        )
        assert config.server_transforms_enabled is True

    def test_service_multi_tenant_without_auth_rejected(self, tmp_path):
        """Without trusted-proxy auth the tenant header is spoofable."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="service",
                multi_tenant_enabled=True,
            )
        assert "multi_tenant_enabled" in str(exc_info.value)

    def test_service_writes_without_auth_rejected(self, tmp_path):
        """Authenticated write-back must be attributable."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                deployment_mode="service",
                service_writes_enabled=True,
            )
        assert "service_writes_enabled" in str(exc_info.value)

    def test_service_writes_with_trusted_proxy_allowed(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="t",
            service_writes_enabled=True,
        )
        assert config.service_writes_enabled is True


class TestTeamCacheCoherence:
    """The team cache needs a store to be a cache of.

    The pairing must hold in either mode: the reader is usually a personal-mode laptop pointed at a
    service-mode store.
    """

    def test_team_cache_without_a_store_is_rejected(self, tmp_path):
        """Silently inert would recompute every cell and look like a broken cache."""
        with pytest.raises(ValueError) as exc_info:
            StrataConfig(
                cache_dir=tmp_path / "cache",
                notebook_team_cache_enabled=True,
            )
        assert "notebook_remote_store_url" in str(exc_info.value)

    def test_team_cache_with_a_store_is_allowed(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            notebook_remote_store_url="https://store.example",
            notebook_team_cache_enabled=True,
        )
        assert config.notebook_team_cache_enabled is True

    def test_a_remote_store_alone_does_not_turn_the_cache_on(self, tmp_path):
        """Publishing to a store does not imply pulling results from it."""
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            notebook_remote_store_url="https://store.example",
        )
        assert config.notebook_team_cache_enabled is False


class TestUnsetSigningSecretIsSurfaced:
    """Service mode warns when signed URLs get a throwaway signing secret.

    The fallback secret is per-process and the pull-model routes are always registered, so a
    callback landing on another replica, or after a restart, gets 403.
    """

    def test_service_mode_without_a_secret_warns(self, tmp_path):
        config = StrataConfig(
            deployment_mode="service",
            artifact_dir=tmp_path / "artifacts",
            cache_dir=tmp_path / "cache",
        )
        assert _should_warn_unset_signing_secret(config) is True

    def test_a_configured_secret_is_quiet(self, tmp_path):
        config = StrataConfig(
            deployment_mode="service",
            artifact_dir=tmp_path / "artifacts",
            cache_dir=tmp_path / "cache",
            transform_signing_secret="a-stable-secret",
        )
        assert _should_warn_unset_signing_secret(config) is False

    def test_personal_mode_is_quiet(self, tmp_path):
        """One loopback process: a per-process secret is correct there."""
        config = StrataConfig(
            deployment_mode="personal",
            cache_dir=tmp_path / "cache",
        )
        assert _should_warn_unset_signing_secret(config) is False
