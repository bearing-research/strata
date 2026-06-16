"""Tests for StrataConfig.load() precedence + normalization (config review findings).

These pin the documented precedence (defaults < pyproject < env < overrides) and
the nested-config normalization that ``load()`` performs, by monkeypatching
``_load_from_pyproject`` so the test controls the "pyproject" layer.
"""

import pytest

import strata.config as cfg
from strata.config import StrataConfig


@pytest.fixture(autouse=True)
def _clean_strata_env(monkeypatch):
    """Drop any STRATA_*/CATALOG env leaking in from the runner."""
    for name in list(__import__("os").environ):
        if name.startswith("STRATA_"):
            monkeypatch.delenv(name, raising=False)


def _pyproject(monkeypatch, data: dict):
    monkeypatch.setattr(cfg, "_load_from_pyproject", lambda: dict(data))


class TestPrecedence:
    """Finding #1: env vars must override pyproject.toml values."""

    def test_env_overrides_pyproject(self, monkeypatch, tmp_path):
        _pyproject(monkeypatch, {"host": "1.2.3.4"})
        monkeypatch.setenv("STRATA_HOST", "9.9.9.9")
        config = StrataConfig.load(cache_dir=tmp_path / "c")
        assert config.host == "9.9.9.9"

    def test_pyproject_beats_default(self, monkeypatch, tmp_path):
        _pyproject(monkeypatch, {"host": "1.2.3.4"})
        config = StrataConfig.load(cache_dir=tmp_path / "c")
        assert config.host == "1.2.3.4"

    def test_overrides_beat_env(self, monkeypatch, tmp_path):
        _pyproject(monkeypatch, {})
        monkeypatch.setenv("STRATA_HOST", "9.9.9.9")
        config = StrataConfig.load(cache_dir=tmp_path / "c", host="5.5.5.5")
        assert config.host == "5.5.5.5"

    def test_operational_override_of_pyproject_auth(self, monkeypatch, tmp_path):
        """The motivating case: STRATA_DEPLOYMENT_MODE must win over a pyproject
        deployment_mode (else operational overrides are silently ignored)."""
        _pyproject(monkeypatch, {"deployment_mode": "service"})
        monkeypatch.setenv("STRATA_DEPLOYMENT_MODE", "personal")
        config = StrataConfig.load(cache_dir=tmp_path / "c", artifact_dir=tmp_path / "a")
        assert config.deployment_mode == "personal"


class TestTrustedProxyRequiresToken:
    """Finding #2: trusted_proxy auth without a token accepts every request."""

    def test_trusted_proxy_without_token_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="proxy_token"):
            StrataConfig(
                cache_dir=tmp_path / "c",
                deployment_mode="service",
                auth_mode="trusted_proxy",
            )

    def test_trusted_proxy_with_token_ok(self, tmp_path):
        config = StrataConfig(
            cache_dir=tmp_path / "c",
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="shared-secret",
        )
        assert config.proxy_token == "shared-secret"


class TestAclNormalization:
    """Finding #3: the documented [tool.strata.acl_config] shape must parse."""

    _ACL = {
        "default": "deny",
        "deny": [{"principal": "*", "tables": ["file:secret.*"]}],
        "allow": [{"principal": "bi", "tables": ["file:pub.*"]}],
    }

    def _assert_parsed(self, config):
        assert config.acl_config.default == "deny"
        assert len(config.acl_config.deny_rules) == 1
        assert config.acl_config.deny_rules[0].tables == ("file:secret.*",)
        assert len(config.acl_config.allow_rules) == 1
        assert config.acl_config.allow_rules[0].principal == "bi"

    def test_acl_config_key_parses_rules(self, monkeypatch, tmp_path):
        _pyproject(
            monkeypatch,
            {
                "deployment_mode": "service",
                "auth_mode": "trusted_proxy",
                "proxy_token": "x",
                "acl_config": dict(self._ACL),
            },
        )
        self._assert_parsed(StrataConfig.load(cache_dir=tmp_path / "c"))

    def test_acl_key_still_parses_rules(self, monkeypatch, tmp_path):
        _pyproject(
            monkeypatch,
            {
                "deployment_mode": "service",
                "auth_mode": "trusted_proxy",
                "proxy_token": "x",
                "acl": dict(self._ACL),
            },
        )
        self._assert_parsed(StrataConfig.load(cache_dir=tmp_path / "c"))


class TestNestedConfigMerge:
    """Findings #4 and #5: env nested overrides must merge, not replace."""

    def test_transforms_env_toggle_merges_with_pyproject_block(self, monkeypatch, tmp_path):
        _pyproject(
            monkeypatch,
            {
                "deployment_mode": "service",
                "artifact_dir": str(tmp_path / "a"),
                # A transforms block with a registry but no ``enabled``.
                "transforms": {
                    "registry": [{"ref": "duckdb_sql@v1", "executor_url": "http://x:8080"}]
                },
            },
        )
        monkeypatch.setenv("STRATA_TRANSFORMS_ENABLED", "true")
        config = StrataConfig.load(cache_dir=tmp_path / "c")
        assert config.server_transforms_enabled is True  # env toggle preserved
        assert config.transforms_config["registry"][0]["ref"] == "duckdb_sql@v1"  # block kept

    def test_catalog_uri_deep_merges_into_properties(self, monkeypatch, tmp_path):
        _pyproject(monkeypatch, {"catalog_properties": {"type": "sql", "warehouse": "/wh"}})
        monkeypatch.setenv("STRATA_CATALOG_URI", "postgresql://host/db")
        config = StrataConfig.load(cache_dir=tmp_path / "c")
        assert config.catalog_properties == {
            "type": "sql",
            "warehouse": "/wh",
            "uri": "postgresql://host/db",
        }
