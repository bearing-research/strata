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

    def test_env_acl_parses_the_same_rules(self, monkeypatch, tmp_path):
        # STRATA_ACL_CONFIG reaches pydantic's env source directly and never
        # passes through load()'s normalization, so the rules were dropped and
        # only `default` survived. With `default = "allow"` that fails OPEN:
        # the operator's deny list disappears and the boot is clean.
        import json

        _pyproject(
            monkeypatch,
            {"deployment_mode": "service", "auth_mode": "trusted_proxy", "proxy_token": "x"},
        )
        monkeypatch.setenv("STRATA_ACL_CONFIG", json.dumps(self._ACL))
        self._assert_parsed(StrataConfig.load(cache_dir=tmp_path / "c"))

    def test_env_deny_rule_actually_denies(self, monkeypatch, tmp_path):
        # The parse above, carried through to the decision it exists to make.
        import json

        from strata.auth import AclEvaluator
        from strata.types import Principal, TableRef

        _pyproject(
            monkeypatch,
            {"deployment_mode": "service", "auth_mode": "trusted_proxy", "proxy_token": "x"},
        )
        monkeypatch.setenv(
            "STRATA_ACL_CONFIG",
            json.dumps({"default": "allow", "deny": [{"tables": ["file:finance.*"]}]}),
        )
        config = StrataConfig.load(cache_dir=tmp_path / "c")

        allowed = AclEvaluator(config.acl_config).authorize(
            Principal(id="analyst", tenant=None, scopes=frozenset({"tables:read"})),
            TableRef(catalog="file", namespace="finance", table="salaries"),
        )
        assert allowed is False

    @pytest.mark.parametrize("key", ["denny", "deny_rule", "rules"])
    def test_an_unknown_acl_key_is_rejected(self, monkeypatch, tmp_path, key):
        # A mistyped rule key used to be ignored, leaving an ACL that boots
        # clean and enforces nothing -- the same silent-disarm shape as above.
        _pyproject(
            monkeypatch,
            {
                "deployment_mode": "service",
                "auth_mode": "trusted_proxy",
                "proxy_token": "x",
                "acl": {"default": "allow", key: [{"tables": ["*"]}]},
            },
        )
        with pytest.raises(ValueError, match=key):
            StrataConfig.load(cache_dir=tmp_path / "c")


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


class TestGCSBucketLocationRename:
    """``STRATA_GCS_PROJECT_ID`` never set a project.

    ``GcsFileSystem`` takes no project parameter, so whatever the setting held
    went to ``default_bucket_location``. The field is named for that now, and
    the old name still reaches it so a deployment using it keeps the behaviour
    it had rather than silently losing the setting.
    """

    def test_the_new_name_populates_the_field(self, monkeypatch):
        monkeypatch.setenv("STRATA_GCS_DEFAULT_BUCKET_LOCATION", "europe-west1")

        assert StrataConfig().gcs_default_bucket_location == "europe-west1"

    def test_the_old_name_still_reaches_the_field(self, monkeypatch):
        monkeypatch.delenv("STRATA_GCS_DEFAULT_BUCKET_LOCATION", raising=False)
        monkeypatch.setenv("STRATA_GCS_PROJECT_ID", "US")

        assert StrataConfig().gcs_default_bucket_location == "US"

    def test_the_field_is_not_reachable_without_the_prefix(self, monkeypatch):
        """validation_alias replaces env_prefix rather than combining with it.

        A bare ``gcs_project_id`` in the alias list would therefore make the
        unprefixed ``GCS_PROJECT_ID`` live config — and in a GCP deployment
        that variable is ambient, so a project id would land in the location
        field exactly as before. No other setting is reachable unprefixed.
        """
        monkeypatch.delenv("STRATA_GCS_PROJECT_ID", raising=False)
        monkeypatch.delenv("STRATA_GCS_DEFAULT_BUCKET_LOCATION", raising=False)
        monkeypatch.setenv("GCS_PROJECT_ID", "my-gcp-project")
        monkeypatch.setenv("GCS_DEFAULT_BUCKET_LOCATION", "europe-west1")

        assert StrataConfig().gcs_default_bucket_location is None

    def test_env_still_beats_the_legacy_pyproject_key(self, tmp_path, monkeypatch):
        """Documented precedence is pyproject < env.

        load() drops a pyproject key that ``STRATA_{KEY}`` shadows, but this
        field answers to two env names, so the legacy spelling in a file would
        otherwise survive and — being passed as an init kwarg — outrank the
        env source.
        """
        (tmp_path / "pyproject.toml").write_text(
            '[tool.strata]\ngcs_project_id = "US"\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("STRATA_GCS_PROJECT_ID", raising=False)
        monkeypatch.setenv("STRATA_GCS_DEFAULT_BUCKET_LOCATION", "europe-west1")

        assert StrataConfig.load().gcs_default_bucket_location == "europe-west1"

    def test_the_legacy_env_name_also_beats_the_file(self, tmp_path, monkeypatch):
        """The case the generic STRATA_{KEY} rule cannot see.

        After the file key is folded onto the current name, that rule looks for
        STRATA_GCS_DEFAULT_BUCKET_LOCATION. An operator still using the old env
        name sets STRATA_GCS_PROJECT_ID, which the rule does not recognise as
        shadowing anything — so the file key survives and, as an init kwarg,
        outranks the env source.
        """
        (tmp_path / "pyproject.toml").write_text(
            '[tool.strata]\ngcs_project_id = "US"\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("STRATA_GCS_DEFAULT_BUCKET_LOCATION", raising=False)
        monkeypatch.setenv("STRATA_GCS_PROJECT_ID", "europe-west1")

        assert StrataConfig.load().gcs_default_bucket_location == "europe-west1"

    def test_the_legacy_pyproject_key_still_reaches_the_field(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.strata]\ngcs_project_id = "US"\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("STRATA_GCS_PROJECT_ID", raising=False)
        monkeypatch.delenv("STRATA_GCS_DEFAULT_BUCKET_LOCATION", raising=False)

        assert StrataConfig.load().gcs_default_bucket_location == "US"

    def test_the_new_name_wins_when_both_are_set(self, monkeypatch):
        monkeypatch.setenv("STRATA_GCS_PROJECT_ID", "US")
        monkeypatch.setenv("STRATA_GCS_DEFAULT_BUCKET_LOCATION", "europe-west1")

        assert StrataConfig().gcs_default_bucket_location == "europe-west1"
