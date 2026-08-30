"""API key minting, verification, and the auth-mode branch."""

from __future__ import annotations

import time

import pytest

from strata.api_keys import ApiKeyStore, format_key, parse_key, reset_api_key_store
from strata.auth import AuthError, parse_api_key_principal
from strata.config import StrataConfig


@pytest.fixture
def store(tmp_path):
    return ApiKeyStore(tmp_path / "artifacts.sqlite")


class TestKeyFormat:
    def test_round_trips(self):
        assert parse_key(format_key("abc123", "s3cr3t")) == ("abc123", "s3cr3t")

    def test_secret_may_contain_underscores(self):
        # The secret is base64url and that alphabet includes '_'. An unbounded
        # split("_") reads such a key as four parts and rejects it -- which is
        # an intermittent auth failure on roughly half of every batch issued,
        # and looks like a flaky client rather than a server bug.
        assert parse_key("strata_keyid_ab_cd_ef") == ("keyid", "ab_cd_ef")

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "strata",
            "strata_only-two",
            "wrongprefix_id_secret",
            "strata__secret",  # empty key id
            "strata_id_",  # empty secret
        ],
    )
    def test_malformed_keys_are_rejected(self, bad):
        assert parse_key(bad) is None


class TestVerification:
    def test_minted_key_resolves_to_its_principal(self, store):
        presented, record = store.create_key(
            principal_id="svc-etl",
            tenant="acme",
            scopes={"artifacts:write", "admin:cache"},
        )
        principal = store.verify(presented)

        assert principal is not None
        assert principal.id == "svc-etl"
        assert principal.tenant == "acme"
        assert principal.scopes == frozenset({"artifacts:write", "admin:cache"})
        assert record.is_active

    def test_every_key_in_a_batch_verifies(self, store):
        # The regression that matters for the underscore bug: about half of
        # generated secrets contain one, so a single round-trip test passes
        # roughly half the time by luck.
        minted = [store.create_key(principal_id=f"p{i}")[0] for i in range(50)]
        assert all(store.verify(k) is not None for k in minted)

    def test_unknown_key_is_rejected(self, store):
        assert store.verify("strata_0123456789abcdef_nosuchsecret") is None

    def test_wrong_secret_for_a_real_key_id_is_rejected(self, store):
        presented, record = store.create_key(principal_id="svc")
        assert store.verify(format_key(record.key_id, "not-the-secret")) is None
        assert store.verify(presented) is not None

    def test_malformed_credential_is_rejected(self, store):
        assert store.verify("not-a-strata-key") is None

    def test_revoked_key_stops_working(self, store):
        presented, record = store.create_key(principal_id="svc")
        assert store.verify(presented) is not None

        assert store.revoke(record.key_id) is True
        assert store.verify(presented) is None

    def test_revoking_twice_reports_nothing_live(self, store):
        _, record = store.create_key(principal_id="svc")
        assert store.revoke(record.key_id) is True
        assert store.revoke(record.key_id) is False

    def test_expired_key_stops_working(self, store):
        presented, _ = store.create_key(principal_id="svc", expires_in_seconds=-1.0)
        assert store.verify(presented) is None

    def test_unexpired_key_still_works(self, store):
        presented, record = store.create_key(principal_id="svc", expires_in_seconds=3600.0)
        assert store.verify(presented) is not None
        assert record.is_active


class TestSecretsAreNotRecoverable:
    def test_the_secret_is_not_stored(self, store):
        presented, record = store.create_key(principal_id="svc")
        # Via parse_key, not a hand-rolled split: the secret is base64url and
        # can contain '_', so rsplit("_", 1) sometimes yields a one-character
        # tail that turns this into a coin flip against a hex digest.
        _, secret = parse_key(presented)

        conn = store._get_connection()
        try:
            row = conn.execute(
                "SELECT key_hash FROM api_keys WHERE key_id = ?", (record.key_id,)
            ).fetchone()
        finally:
            conn.close()

        # Only a digest is kept, so a database disclosure yields no usable
        # credential.
        assert secret not in row["key_hash"]
        assert len(row["key_hash"]) == 64  # sha256 hex

    def test_listing_never_exposes_secrets(self, store):
        presented, _ = store.create_key(principal_id="svc", description="d")
        _, secret = parse_key(presented)

        records = store.list_keys()
        assert len(records) == 1
        assert secret not in repr(records[0])


class TestListingAndUsage:
    def test_lists_newest_first_and_filters_by_principal(self, store):
        store.create_key(principal_id="a")
        time.sleep(0.01)
        store.create_key(principal_id="b")

        assert [r.principal_id for r in store.list_keys()] == ["b", "a"]
        assert [r.principal_id for r in store.list_keys(principal_id="a")] == ["a"]

    def test_touch_records_use(self, store):
        _, record = store.create_key(principal_id="svc")
        assert store.list_keys()[0].last_used_at is None

        store.touch(record.key_id)
        assert store.list_keys()[0].last_used_at is not None


class TestAuthModeBranch:
    """``parse_api_key_principal`` is what the middleware calls."""

    def _config(self):
        return StrataConfig()

    def test_bearer_key_resolves(self, store, monkeypatch):
        presented, _ = store.create_key(principal_id="svc-etl", scopes={"artifacts:write"})
        monkeypatch.setattr("strata.api_keys.get_api_key_store", lambda *a, **k: store)

        principal = parse_api_key_principal(
            {"Authorization": f"Bearer {presented}"}, self._config()
        )
        assert principal.id == "svc-etl"

    def test_header_name_is_case_insensitive(self, store, monkeypatch):
        presented, _ = store.create_key(principal_id="svc")
        monkeypatch.setattr("strata.api_keys.get_api_key_store", lambda *a, **k: store)

        assert (
            parse_api_key_principal({"authorization": f"Bearer {presented}"}, self._config()).id
            == "svc"
        )

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Authorization": ""},
            {"Authorization": "Bearer "},
            {"Authorization": "Basic abc"},
        ],
    )
    def test_missing_or_wrong_scheme_is_rejected(self, headers, store, monkeypatch):
        monkeypatch.setattr("strata.api_keys.get_api_key_store", lambda *a, **k: store)
        with pytest.raises(AuthError):
            parse_api_key_principal(headers, self._config())

    def test_bad_key_is_rejected(self, store, monkeypatch):
        monkeypatch.setattr("strata.api_keys.get_api_key_store", lambda *a, **k: store)
        with pytest.raises(AuthError):
            parse_api_key_principal({"Authorization": "Bearer strata_a_b"}, self._config())

    def test_no_key_store_fails_closed(self, monkeypatch):
        # Configured for key auth with no store: serving the request
        # unauthenticated would be the worst outcome, since the operator
        # explicitly asked for authentication.
        reset_api_key_store()
        monkeypatch.setattr("strata.api_keys.get_api_key_store", lambda *a, **k: None)
        with pytest.raises(AuthError):
            parse_api_key_principal({"Authorization": "Bearer strata_a_b"}, self._config())


class TestCliOutputStreams:
    """The secret goes to stdout alone so shell capture is exact."""

    def _args(self, tmp_path, **over):
        import argparse

        return argparse.Namespace(
            principal="svc",
            tenant=None,
            scopes=None,
            description=None,
            expires_in_days=None,
            artifact_dir=str(tmp_path),
            dsn=None,
            **over,
        )

    def test_stdout_carries_only_the_key(self, tmp_path, capsys):
        from strata.api_key_cli import cmd_create
        from strata.api_keys import ApiKeyStore

        assert cmd_create(self._args(tmp_path)) == 0
        captured = capsys.readouterr()

        # KEY=$(strata apikey create ...) must yield a usable credential with
        # no trimming. Folding the summary into stdout would make the captured
        # value a broken key, surfacing later as a confusing 401.
        key = captured.out.strip()
        assert "\n" not in key
        store = ApiKeyStore(tmp_path / "artifacts.sqlite")
        assert store.verify(key) is not None

    def test_human_summary_goes_to_stderr(self, tmp_path, capsys):
        from strata.api_key_cli import cmd_create

        cmd_create(self._args(tmp_path))
        captured = capsys.readouterr()

        assert "key id" in captured.err
        assert "cannot be shown again" in captured.err
        assert "key id" not in captured.out


class TestConfiguration:
    def _service(self, **kwargs):
        return dict(
            deployment_mode="service",
            artifact_dir="/tmp/strata-artifacts",
            **kwargs,
        )

    def test_api_key_mode_is_accepted_in_service_mode(self):
        assert StrataConfig(**self._service(auth_mode="api_key")).auth_mode == "api_key"

    def test_api_key_mode_needs_an_artifact_dir(self):
        # Keys live in the artifact store's database. Without one every request
        # would fail closed at the middleware -- safe, but useless.
        with pytest.raises(ValueError, match="artifact_dir"):
            StrataConfig(deployment_mode="service", auth_mode="api_key", artifact_dir=None)

    def test_personal_mode_rejects_api_key_auth(self):
        with pytest.raises(ValueError, match="api_key"):
            StrataConfig(deployment_mode="personal", auth_mode="api_key")
