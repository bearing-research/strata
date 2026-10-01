"""Security regression tests for trusted proxy authentication.

Invariants: requests without a valid proxy token are rejected, even with spoofed principal headers;
ACL deny beats allow; scan ownership is enforced; hide_forbidden_as_not_found gives 404 instead of
403.
"""

import pytest

from strata.auth import (
    AclEvaluator,
    get_principal,
    parse_principal,
    set_principal,
    verify_proxy_token,
)
from strata.config import AclConfig, AclRule, StrataConfig
from strata.types import Principal, TableIdentity, TableRef


class TestProxyTokenVerification:
    def test_no_token_configured_allows_all(self):
        assert verify_proxy_token(None, None) is True
        assert verify_proxy_token("any-token", None) is True

    def test_missing_request_token_rejected(self):
        assert verify_proxy_token(None, "expected-token") is False

    def test_wrong_token_rejected(self):
        assert verify_proxy_token("wrong-token", "expected-token") is False

    def test_correct_token_accepted(self):
        assert verify_proxy_token("correct-token", "correct-token") is True

    def test_timing_safe_comparison(self):
        """Token comparison is constant-time."""
        # Timing can't be tested directly; this only checks the comparison's result.

        assert verify_proxy_token("a", "b") is False
        assert verify_proxy_token("a", "a") is True


class TestPrincipalParsing:
    def test_missing_principal_raises_auth_error(self):
        from strata.auth import AuthError

        config = StrataConfig.load(
            deployment_mode="service", auth_mode="trusted_proxy", proxy_token="t"
        )
        headers = {}

        with pytest.raises(AuthError) as exc_info:
            parse_principal(headers, config)

        assert exc_info.value.status_code == 401
        assert "principal" in exc_info.value.message.lower()

    def test_principal_parsed_from_header(self):
        config = StrataConfig.load(
            deployment_mode="service", auth_mode="trusted_proxy", proxy_token="t"
        )
        headers = {
            config.principal_header: "test-user",
            config.tenant_header: "test-tenant",
            config.scopes_header: "scan:create scan:read admin:cache",
        }

        principal = parse_principal(headers, config)

        assert principal.id == "test-user"
        assert principal.tenant == "test-tenant"
        assert principal.scopes == frozenset({"scan:create", "scan:read", "admin:cache"})

    def test_principal_parsed_from_lowercase_headers(self):
        """Header parsing stays case-insensitive through ASGI request dicts."""
        config = StrataConfig.load(
            deployment_mode="service", auth_mode="trusted_proxy", proxy_token="t"
        )
        headers = {
            config.principal_header.lower(): "test-user",
            config.tenant_header.lower(): "test-tenant",
            config.scopes_header.lower(): "scan:create",
        }

        principal = parse_principal(headers, config)

        assert principal.id == "test-user"
        assert principal.tenant == "test-tenant"
        assert principal.scopes == frozenset({"scan:create"})

    def test_empty_scopes_ok(self):
        config = StrataConfig.load(
            deployment_mode="service", auth_mode="trusted_proxy", proxy_token="t"
        )
        headers = {config.principal_header: "test-user"}

        principal = parse_principal(headers, config)

        assert principal.id == "test-user"
        assert principal.scopes == frozenset()


class TestPrincipalScopes:
    def test_exact_scope_match(self):
        principal = Principal(id="user", scopes=frozenset({"scan:create"}))
        assert principal.has_scope("scan:create") is True

    def test_missing_scope_returns_false(self):
        principal = Principal(id="user", scopes=frozenset({"scan:create"}))
        assert principal.has_scope("admin:cache") is False

    def test_admin_wildcard_grants_all(self):
        principal = Principal(id="admin", scopes=frozenset({"admin:*"}))

        assert principal.has_scope("scan:create") is True
        assert principal.has_scope("admin:cache") is True
        assert principal.has_scope("anything:at:all") is True


class TestTableRef:
    def test_from_table_identity_file(self):
        identity = TableIdentity(catalog="strata", namespace="db", table="events")
        table_ref = TableRef.from_table_identity(identity, table_uri="file:///warehouse#db.events")

        assert table_ref.catalog == "file"
        assert table_ref.namespace == "db"
        assert table_ref.table == "events"
        assert str(table_ref) == "file:db.events"

    def test_from_table_identity_s3(self):
        identity = TableIdentity(catalog="strata", namespace="analytics", table="clicks")
        table_ref = TableRef.from_table_identity(
            identity, table_uri="s3://bucket/warehouse#analytics.clicks"
        )

        assert table_ref.catalog == "s3"
        assert table_ref.namespace == "analytics"
        assert table_ref.table == "clicks"
        assert str(table_ref) == "s3:analytics.clicks"


class TestAclEvaluator:
    def test_default_allow(self):
        config = AclConfig(default="allow", deny_rules=[], allow_rules=[])
        acl = AclEvaluator(config)
        principal = Principal(id="anyone")
        table_ref = TableRef(catalog="file", namespace="db", table="events")

        assert acl.authorize(principal, table_ref) is True

    def test_default_deny(self):
        config = AclConfig(default="deny", deny_rules=[], allow_rules=[])
        acl = AclEvaluator(config)
        principal = Principal(id="anyone")
        table_ref = TableRef(catalog="file", namespace="db", table="events")

        assert acl.authorize(principal, table_ref) is False

    def test_allow_rule_matches(self):
        config = AclConfig(
            default="deny",
            allow_rules=[AclRule(principal="bi-dashboard", tables=("file:db.*",))],
        )
        acl = AclEvaluator(config)
        principal = Principal(id="bi-dashboard")
        table_ref = TableRef(catalog="file", namespace="db", table="events")

        assert acl.authorize(principal, table_ref) is True

    def test_allow_rule_no_match(self):
        config = AclConfig(
            default="deny",
            allow_rules=[AclRule(principal="bi-dashboard", tables=("file:db.*",))],
        )
        acl = AclEvaluator(config)
        principal = Principal(id="other-user")
        table_ref = TableRef(catalog="file", namespace="db", table="events")

        assert acl.authorize(principal, table_ref) is False

    def test_deny_overrides_allow(self):
        """Deny rules are checked before allow rules."""
        config = AclConfig(
            default="deny",
            deny_rules=[AclRule(principal="*", tables=("file:finance.*",))],
            allow_rules=[AclRule(principal="analyst", tables=("file:*.*",))],
        )
        acl = AclEvaluator(config)
        principal = Principal(id="analyst")

        # Allowed table
        allowed_ref = TableRef(catalog="file", namespace="db", table="events")
        assert acl.authorize(principal, allowed_ref) is True

        # Denied table (deny overrides allow)
        denied_ref = TableRef(catalog="file", namespace="finance", table="salary")
        assert acl.authorize(principal, denied_ref) is False

    def test_wildcard_principal(self):
        config = AclConfig(
            default="deny",
            allow_rules=[AclRule(principal="*", tables=("file:public.*",))],
        )
        acl = AclEvaluator(config)

        table_ref = TableRef(catalog="file", namespace="public", table="data")

        # Any principal should match
        assert acl.authorize(Principal(id="user-a"), table_ref) is True
        assert acl.authorize(Principal(id="user-b"), table_ref) is True
        assert acl.authorize(Principal(id="anonymous"), table_ref) is True

    def test_tenant_match(self):
        """A rule with a tenant only matches that tenant."""
        config = AclConfig(
            default="deny",
            allow_rules=[AclRule(principal="*", tenant="data-platform", tables=("file:*.*",))],
        )
        acl = AclEvaluator(config)
        table_ref = TableRef(catalog="file", namespace="db", table="events")

        # Matching tenant
        principal_match = Principal(id="user", tenant="data-platform")
        assert acl.authorize(principal_match, table_ref) is True

        # Different tenant
        principal_no_match = Principal(id="user", tenant="other-team")
        assert acl.authorize(principal_no_match, table_ref) is False

        # No tenant
        principal_no_tenant = Principal(id="user", tenant=None)
        assert acl.authorize(principal_no_tenant, table_ref) is False

    def test_glob_pattern_matching(self):
        """Table patterns use glob matching."""
        config = AclConfig(
            default="deny",
            allow_rules=[AclRule(principal="*", tables=("file:analytics.*",))],
        )
        acl = AclEvaluator(config)
        principal = Principal(id="user")

        # Matches pattern
        assert acl.authorize(principal, TableRef("file", "analytics", "clicks")) is True
        assert acl.authorize(principal, TableRef("file", "analytics", "events")) is True

        # Does not match
        assert acl.authorize(principal, TableRef("file", "finance", "data")) is False
        assert acl.authorize(principal, TableRef("s3", "analytics", "clicks")) is False


class TestPrincipalContext:
    def test_get_set_principal(self):
        principal = Principal(id="test-user")

        set_principal(principal)
        assert get_principal() == principal

        set_principal(None)
        assert get_principal() is None

    def test_principal_context_isolation(self):
        """A principal set in one place is retrieved elsewhere in the same context."""
        principal = Principal(id="test-user", tenant="test-tenant")

        set_principal(principal)
        retrieved = get_principal()

        assert retrieved is not None
        assert retrieved.id == "test-user"
        assert retrieved.tenant == "test-tenant"

        set_principal(None)


class TestAclConfigParsing:
    def test_empty_acl_config(self):
        """Empty ACL config uses defaults."""
        config = StrataConfig.load()
        assert config.acl_config.default == "allow"
        assert config.acl_config.deny_rules == []
        assert config.acl_config.allow_rules == []

    def test_acl_config_from_dict(self):
        from strata.config import _parse_acl_config

        raw = {
            "default": "deny",
            "deny": [{"principal": "*", "tables": ["file:pii.*"]}],
            "allow": [{"principal": "admin", "tables": ["file:*.*"]}],
        }

        acl_config = _parse_acl_config(raw)

        assert acl_config.default == "deny"
        assert len(acl_config.deny_rules) == 1
        assert acl_config.deny_rules[0].principal == "*"
        assert acl_config.deny_rules[0].tables == ("file:pii.*",)
        assert len(acl_config.allow_rules) == 1
        assert acl_config.allow_rules[0].principal == "admin"


class TestTransformInputAclParity:
    """A table used as a transform input is gated by the same ACL as a direct scan.

    Both paths share ``_authorize_table_access``, so a denied table cannot be read by passing it as
    an input.
    """

    @staticmethod
    def _patch_state(monkeypatch, *, namespace, hide_as_404=False):
        from unittest.mock import MagicMock

        import strata.server as server_module

        identity = TableIdentity(catalog="file", namespace=namespace, table="events")
        plan = MagicMock()
        plan.snapshot_id = 4242
        plan.schema_id = 0
        plan.table_identity = identity

        state = MagicMock()
        state.config.auth_mode = "trusted_proxy"
        # Set explicitly: a MagicMock attribute is truthy whatever the mode is,
        # so leaving it unset would make every "no auth" case look authenticated.
        state.config.principal_auth_enabled = True
        state.config.hide_forbidden_as_not_found = hide_as_404
        state.config.acl_config = AclConfig(
            default="deny",
            deny_rules=[],
            allow_rules=[AclRule(principal="analyst", tables=("file:public.*",))],
        )
        state.planner.plan.return_value = plan
        monkeypatch.setattr(server_module, "_state", state)
        # The table branch is reached only past the artifact-store gate; stub it
        # so the test exercises the ACL, not store wiring.
        monkeypatch.setattr(server_module, "_get_artifact_store", lambda **k: MagicMock())
        return server_module

    def test_denied_table_input_is_rejected_403(self, monkeypatch):
        from fastapi import HTTPException

        server_module = self._patch_state(monkeypatch, namespace="secret")
        set_principal(Principal(id="intruder"))
        try:
            with pytest.raises(HTTPException) as exc:
                server_module._resolve_input_version("file:///wh#secret.events")
            assert exc.value.status_code == 403
        finally:
            set_principal(None)

    def test_denied_table_input_hidden_as_404(self, monkeypatch):
        from fastapi import HTTPException

        server_module = self._patch_state(monkeypatch, namespace="secret", hide_as_404=True)
        set_principal(Principal(id="intruder"))
        try:
            with pytest.raises(HTTPException) as exc:
                server_module._resolve_input_version("file:///wh#secret.events")
            assert exc.value.status_code == 404
        finally:
            set_principal(None)

    def test_allowed_table_input_resolves(self, monkeypatch):
        server_module = self._patch_state(monkeypatch, namespace="public")
        set_principal(Principal(id="analyst"))
        try:
            assert server_module._resolve_input_version("file:///wh#public.events") == "4242:0"
        finally:
            set_principal(None)

    @pytest.mark.parametrize(
        ("failure", "hide_as_404"),
        [
            ("refused", True),
            ("refused", False),
            ("unplannable", True),
            ("unplannable", False),
        ],
    )
    def test_a_denied_table_is_denied_before_it_is_planned(self, monkeypatch, failure, hide_as_404):
        """The ACL decides on the URI's identity before any manifest is read, so a denied caller
        sees neither the delete-files 422 nor the unplannable 400.
        """
        from fastapi import HTTPException

        from strata.iceberg_schema import UnsupportedTableFormatError

        server_module = self._patch_state(monkeypatch, namespace="secret", hide_as_404=hide_as_404)
        server_module._state.planner.plan.side_effect = (
            UnsupportedTableFormatError(
                "Table secret.events has an AVRO equality delete file "
                "(s3://secret-bucket/warehouse/secret/events/data/del-001.avro)."
            )
            if failure == "refused"
            else RuntimeError("catalog unreachable")
        )
        set_principal(Principal(id="intruder"))
        try:
            with pytest.raises(HTTPException) as exc:
                server_module._resolve_input_version("file:///wh#secret.events")
        finally:
            set_principal(None)

        assert exc.value.status_code == (404 if hide_as_404 else 403)
        assert "secret-bucket" not in str(exc.value.detail)
        server_module._state.planner.plan.assert_not_called()

    def test_an_allowed_table_strata_refuses_is_still_a_422(self, monkeypatch):
        from fastapi import HTTPException

        from strata.iceberg_schema import UnsupportedTableFormatError

        server_module = self._patch_state(monkeypatch, namespace="public")
        server_module._state.planner.plan.side_effect = UnsupportedTableFormatError(
            "an Avro delete"
        )
        set_principal(Principal(id="analyst"))
        try:
            with pytest.raises(HTTPException) as exc:
                server_module._resolve_input_version("file:///wh#public.events")
        finally:
            set_principal(None)

        assert exc.value.status_code == 422

    def test_a_table_uri_the_acl_cannot_name_is_denied(self, monkeypatch):
        """Deny-first: with no identity, no rule allows it; the planner could not name it either."""
        from fastapi import HTTPException

        server_module = self._patch_state(monkeypatch, namespace="public")
        monkeypatch.setattr(server_module, "_table_identity_from_uri", lambda uri: None)
        set_principal(Principal(id="analyst"))
        try:
            with pytest.raises(HTTPException) as exc:
                server_module._resolve_input_version("file:///wh#public.events")
        finally:
            set_principal(None)

        assert exc.value.status_code == 403
        server_module._state.planner.plan.assert_not_called()

    def test_scan_and_transform_share_one_gate(self, monkeypatch):
        # Both the scan path and the transform-input path call this one helper.
        from fastapi import HTTPException

        server_module = self._patch_state(monkeypatch, namespace="secret")
        identity = TableIdentity(catalog="file", namespace="secret", table="events")
        set_principal(Principal(id="intruder"))
        try:
            with pytest.raises(HTTPException) as exc:
                server_module._authorize_table_access("file:///wh#secret.events", identity)
            assert exc.value.status_code == 403
        finally:
            set_principal(None)


class TestArtifactReadAcl:
    """Reading a cached result is ACL-gated.

    The cache is shared across principals, so ``_authorize_artifact_read`` re-checks the table ACL
    of the inputs in the stored transform_spec.
    """

    @staticmethod
    def _artifact(table_uri):
        from unittest.mock import MagicMock

        from strata.artifact_store import TransformSpec

        art = MagicMock()
        art.transform_spec = TransformSpec(
            executor="scan@v1", params={}, inputs=[table_uri]
        ).to_json()
        return art

    @staticmethod
    def _patch_state(monkeypatch, *, auth, hide_as_404=False):
        from unittest.mock import MagicMock

        import strata.server as server_module

        state = MagicMock()
        state.config.auth_mode = auth
        state.config.principal_auth_enabled = auth in ("trusted_proxy", "api_key")
        state.config.hide_forbidden_as_not_found = hide_as_404
        state.config.acl_config = AclConfig(
            default="deny",
            deny_rules=[],
            allow_rules=[AclRule(principal="analyst", tables=("file:public.*",))],
        )
        monkeypatch.setattr(server_module, "_state", state)
        return server_module

    def test_denied_table_read_rejected(self, monkeypatch):
        from fastapi import HTTPException

        server_module = self._patch_state(monkeypatch, auth="trusted_proxy")
        art = self._artifact("file:///wh#secret.events")
        set_principal(Principal(id="intruder"))
        try:
            with pytest.raises(HTTPException) as exc:
                server_module._authorize_artifact_read(art)
            assert exc.value.status_code == 403
        finally:
            set_principal(None)

    def test_allowed_table_read_passes(self, monkeypatch):
        server_module = self._patch_state(monkeypatch, auth="trusted_proxy")
        art = self._artifact("file:///wh#public.events")
        set_principal(Principal(id="analyst"))
        try:
            server_module._authorize_artifact_read(art)  # no raise
        finally:
            set_principal(None)

    def test_no_auth_is_noop(self, monkeypatch):
        # Without trusted-proxy auth there is no principal or ACL, so the read is allowed
        # (tenant scoping is enforced separately by _ensure_artifact_access).
        server_module = self._patch_state(monkeypatch, auth="none")
        art = self._artifact("file:///wh#secret.events")
        server_module._authorize_artifact_read(art)  # no raise

    def test_artifact_without_table_inputs_is_noop(self, monkeypatch):
        # An artifact whose inputs are all artifacts has no table ACL to check; tenant
        # scoping is the gate.
        from unittest.mock import MagicMock

        from strata.artifact_store import TransformSpec

        server_module = self._patch_state(monkeypatch, auth="trusted_proxy")
        art = MagicMock()
        art.transform_spec = TransformSpec(
            executor="duckdb_sql@v1",
            params={},
            inputs=["strata://artifact/abc@v=1"],
        ).to_json()
        set_principal(Principal(id="intruder"))
        try:
            server_module._authorize_artifact_read(art)  # no raise
        finally:
            set_principal(None)


class TestAclRuleMatchingFailsClosed:
    """ACL rules that could never match must fail closed.

    A deny that never fires grants what it was meant to refuse, while ``validate_mode_coherence``
    still counts it as configured.
    """

    def _principal(self, principal_id: str):
        from strata.types import Principal

        return Principal(id=principal_id, tenant=None, scopes=frozenset())

    def _ref(self):
        from strata.types import TableIdentity, TableRef

        return TableRef.from_table_identity(
            TableIdentity.from_table_id("pii.events"), table_uri="s3://wh#pii.events"
        )

    def test_wildcard_principal_pattern_now_matches(self):
        """``principal`` is a pattern like tables, so ``svc-*`` must fire."""
        from strata.auth import AclEvaluator
        from strata.config import AclConfig, AclRule

        acl = AclConfig(
            default="allow",
            deny_rules=[AclRule(principal="svc-*", tables=("s3:pii.*",))],
        )
        evaluator = AclEvaluator(acl)

        assert evaluator.authorize(self._principal("svc-etl"), self._ref()) is False
        # A principal outside the pattern is unaffected.
        assert evaluator.authorize(self._principal("analyst"), self._ref()) is True

    def test_plain_star_principal_still_matches_everyone(self):
        from strata.auth import AclEvaluator
        from strata.config import AclConfig, AclRule

        evaluator = AclEvaluator(
            AclConfig(default="allow", deny_rules=[AclRule(principal="*", tables=("s3:pii.*",))])
        )
        assert evaluator.authorize(self._principal("anyone"), self._ref()) is False

    def test_exact_principal_still_matches_only_itself(self):
        from strata.auth import AclEvaluator
        from strata.config import AclConfig, AclRule

        evaluator = AclEvaluator(
            AclConfig(default="allow", deny_rules=[AclRule(principal="bob", tables=("s3:pii.*",))])
        )
        assert evaluator.authorize(self._principal("bob"), self._ref()) is False
        assert evaluator.authorize(self._principal("bobby"), self._ref()) is True

    def test_rule_without_table_patterns_is_rejected_at_config_time(self):
        """``{ principal = "bob" }`` ("deny bob everything") never matched, so it is rejected."""
        import pydantic

        from strata.config import AclRule

        with pytest.raises(pydantic.ValidationError, match="at least one table pattern"):
            AclRule(principal="bob")


class TestNonAsciiTokensAreRejectedNotCrashed:
    """A non-ASCII token must be a rejection, not a 500.

    ASGI decodes headers as latin-1 and ``hmac.compare_digest`` raises on non-ASCII ``str``, so any
    client could spike the error rate. Comparing UTF-8 bytes stays constant-time and total.
    """

    def test_non_ascii_token_returns_false(self):
        assert verify_proxy_token("tökén", "expected-token") is False

    def test_non_ascii_expected_token_returns_false(self):
        assert verify_proxy_token("presented", "expected-tökén") is False

    def test_latin1_decoded_header_byte_returns_false(self):
        # What Starlette hands us for the raw byte 0xC3.
        assert verify_proxy_token(b"\xc3".decode("latin-1"), "expected-token") is False

    def test_a_matching_non_ascii_token_still_matches(self):
        assert verify_proxy_token("tökén", "tökén") is True

    def test_ascii_behaviour_is_unchanged(self):
        assert verify_proxy_token("secret", "secret") is True
        assert verify_proxy_token("secret", "other") is False


class TestEveryStoreIsItsOwnAclSubject:
    """A rule names a table in a store; one namespace across clouds would let a rule for a local
    table grant a same-named table in a bucket.
    """

    def test_each_store_has_its_own_namespace(self):
        identity = TableIdentity.from_table_id("secret.events")
        refs = {
            uri: str(TableRef.from_table_identity(identity, table_uri=uri))
            for uri in (
                "file:///wh#secret.events",
                "s3://bucket/wh#secret.events",
                "gs://bucket/wh#secret.events",
                "abfss://c@a.dfs.core.windows.net/wh#secret.events",
            )
        }

        assert refs == {
            "file:///wh#secret.events": "file:secret.events",
            "s3://bucket/wh#secret.events": "s3:secret.events",
            "gs://bucket/wh#secret.events": "gs:secret.events",
            "abfss://c@a.dfs.core.windows.net/wh#secret.events": "az:secret.events",
        }


class TestReadingACachedScanIsGatedWhereverTheTableLives:
    """``_authorize_artifact_read`` must gate cached scans of GCS, Azure and named-catalog tables,
    not only file:// and s3://.
    """

    @staticmethod
    def _identity(uri: str):
        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.server import ServerState

        server_module._state = ServerState(
            StrataConfig(catalogs={"prod": {"type": "rest", "uri": "http://catalog"}})
        )
        return server_module._table_identity_from_uri(uri)

    def test_every_table_uri_resolves_to_an_identity_to_check(self):
        for uri in (
            "file:///wh#secret.events",
            "s3://bucket/wh#secret.events",
            "gs://bucket/wh#secret.events",
            "abfss://c@a.dfs.core.windows.net/wh#secret.events",
            "prod:secret.events",
        ):
            assert self._identity(uri) is not None, f"{uri} was not gated"

    def test_what_is_not_a_table_is_left_alone(self):
        assert self._identity("strata://artifact/rows@v=1") is None
        assert self._identity("not-a-table") is None

    def test_the_identity_is_the_one_the_planner_uses(self):
        """The pre-plan and post-plan gates must name a table the same way, or a rule matches only
        one.
        """
        from strata.config import StrataConfig
        from strata.iceberg import table_identity_for

        config = StrataConfig(catalogs={"prod": {"type": "rest", "uri": "http://catalog"}})

        assert self._identity("prod:secret.events") == table_identity_for(
            "prod:secret.events", config
        )


class TestARuleIsAsNarrowAsItReads:
    """An unrecognised key in an access rule is refused; dropping it would widen the rule."""

    def test_a_misspelled_key_is_refused(self):
        import pytest

        from strata.config import AclRule

        # The plural would leave ``tenant`` None, which matches every tenant rather than
        # the one named, while the rule still reads correctly.
        with pytest.raises(ValueError, match="tenants"):
            AclRule(principal="bob", tables=["*"], tenants="acme")

    def test_the_spelling_it_meant_still_works(self):
        from strata.config import AclRule

        assert AclRule(principal="bob", tables=["*"], tenant="acme").tenant == "acme"


class TestADenyForEveryPrefixCoversEveryAddress:
    """With a SQL catalog, a table has one ACL name per address form (``s3:`` or ``file:``), so the
    documented ``*:namespace.*`` deny pattern must cover all of them.
    """

    def test_every_address_of_the_table_is_refused(self, temp_warehouse, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from fastapi import HTTPException

        import strata.server as server_module
        from strata.api.dependencies import authorize_table_access
        from strata.iceberg import table_identity_for

        config = StrataConfig(
            deployment_mode="service",
            artifact_dir=tmp_path / "artifacts",
            cache_dir=tmp_path / "cache",
            auth_mode="trusted_proxy",
            proxy_token="t",
            catalog_name="strata",
            catalog_properties={"uri": temp_warehouse["catalog"].properties["uri"]},
            acl_config=AclConfig(
                default="allow",
                deny_rules=[AclRule(principal="*", tables=("*:test_db.*",))],
            ),
        )
        monkeypatch.setattr(server_module, "get_state", lambda: SimpleNamespace(config=config))
        set_principal(Principal(id="alice"))
        try:
            for uri in (
                "s3://any-bucket/warehouse#test_db.events",
                "test_db.events",
                "/not/a/real/path#test_db.events",
            ):
                with pytest.raises(HTTPException) as refused:
                    authorize_table_access(uri, table_identity_for(uri, config))
                assert refused.value.status_code in (403, 404), uri
        finally:
            set_principal(None)


class TestADenyOnOneAddressCoversTheTable:
    """A deny on one address form refuses the table under every name it answers to.

    With a SQL catalog, every warehouse URI (and a bare name, when the default catalog matches)
    reads one catalog, so an ``s3:`` deny must also stop ``file:`` and bare-name access.
    """

    S3 = "s3://any-bucket/warehouse#test_db.events"
    ALIASES = (
        "test_db.events",
        "/not/a/real/path#test_db.events",
        "gs://other-bucket/wh#test_db.events",
    )

    def _check(self, monkeypatch, tmp_path, acl, **config_overrides):
        from types import SimpleNamespace

        from fastapi import HTTPException

        import strata.server as server_module
        from strata.api.dependencies import authorize_table_access
        from strata.iceberg import table_identity_for

        config = StrataConfig(
            deployment_mode="service",
            artifact_dir=tmp_path / "artifacts",
            cache_dir=tmp_path / "cache",
            auth_mode="trusted_proxy",
            proxy_token="t",
            acl_config=acl,
            **config_overrides,
        )
        monkeypatch.setattr(server_module, "get_state", lambda: SimpleNamespace(config=config))
        set_principal(Principal(id="alice"))

        def allowed(uri):
            try:
                authorize_table_access(uri, table_identity_for(uri, config))
            except HTTPException as refused:
                assert refused.status_code in (403, 404), uri
                return False
            return True

        return allowed

    def _shared_catalog(self, temp_warehouse, catalog_name="strata"):
        return {
            "catalog_name": catalog_name,
            "catalog_properties": {"uri": temp_warehouse["catalog"].properties["uri"]},
        }

    def test_a_deny_on_the_s3_name_refuses_every_address(
        self, temp_warehouse, tmp_path, monkeypatch
    ):
        from strata.iceberg import PyIcebergCatalog

        overrides = self._shared_catalog(temp_warehouse)
        allowed = self._check(
            monkeypatch,
            tmp_path,
            AclConfig(
                default="allow", deny_rules=[AclRule(principal="*", tables=("s3:test_db.*",))]
            ),
            **overrides,
        )
        try:
            for uri in (self.S3, *self.ALIASES):
                assert not allowed(uri), uri
        finally:
            set_principal(None)

        # One table behind every name the deny now covers.
        catalogs = PyIcebergCatalog(StrataConfig(deployment_mode="service", **overrides))
        uris = (self.S3, *self.ALIASES)
        assert len({catalogs.load_table(uri).metadata_location for uri in uris}) == 1

    def test_an_allow_still_names_only_the_address_it_was_written_for(
        self, temp_warehouse, tmp_path, monkeypatch
    ):
        allowed = self._check(
            monkeypatch,
            tmp_path,
            AclConfig(
                default="deny", allow_rules=[AclRule(principal="*", tables=("s3:test_db.*",))]
            ),
            **self._shared_catalog(temp_warehouse),
        )
        try:
            assert allowed(self.S3)
            for uri in self.ALIASES:
                assert not allowed(uri), uri
        finally:
            set_principal(None)

    def test_a_bare_name_in_another_default_catalog_is_another_table(
        self, temp_warehouse, tmp_path, monkeypatch
    ):
        """The ``default`` catalog keeps its own tables, apart from the ``strata`` catalog warehouse
        URIs read.
        """
        allowed = self._check(
            monkeypatch,
            tmp_path,
            AclConfig(
                default="allow", deny_rules=[AclRule(principal="*", tables=("s3:test_db.*",))]
            ),
            **self._shared_catalog(temp_warehouse, catalog_name="default"),
        )
        try:
            assert allowed("test_db.events")
            assert not allowed("/not/a/real/path#test_db.events")
        finally:
            set_principal(None)

    def test_without_a_shared_catalog_the_address_is_the_table(self, tmp_path, monkeypatch):
        """Each local warehouse keeps its own catalog, so an S3 deny says nothing about a local
        namesake.
        """
        allowed = self._check(
            monkeypatch,
            tmp_path,
            AclConfig(
                default="allow", deny_rules=[AclRule(principal="*", tables=("s3:test_db.*",))]
            ),
        )
        try:
            assert not allowed(self.S3)
            assert allowed(f"{tmp_path}/warehouse#test_db.events")
        finally:
            set_principal(None)
