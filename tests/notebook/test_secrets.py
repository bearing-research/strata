"""Tests for secret-manager integration (provider, session merge, route)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strata.notebook.models import NotebookState
from strata.notebook.secret_manager.infisical import InfisicalProvider
from strata.notebook.secret_manager.provider import SecretFetchResult, SecretProviderError
from strata.notebook.secret_manager.registry import _reset_for_tests, get_provider
from strata.notebook.secret_manager.session_integration import (
    MANUAL_SOURCE,
    apply_secrets_to_notebook_state,
    fetch_configured_secrets,
)


class _FakeClient:
    """Stand-in for infisicalsdk.InfisicalSDKClient in tests."""

    def __init__(
        self,
        *,
        list_secrets_return=None,
        list_secrets_exc: Exception | None = None,
        login_exc: Exception | None = None,
    ):
        self.host: str | None = None
        self.list_secrets_calls: list[dict] = []
        self.login_calls: list[tuple[str, dict]] = []
        self._list_return = list_secrets_return or SimpleNamespace(secrets=[])
        self._list_exc = list_secrets_exc
        self._login_exc = login_exc
        self.auth = SimpleNamespace(
            universal_auth=SimpleNamespace(login=self._universal_login),
            token_auth=SimpleNamespace(login=self._token_login),
        )
        self.secrets = SimpleNamespace(list_secrets=self._list_secrets)

    def _universal_login(self, client_id: str, client_secret: str):
        if self._login_exc:
            raise self._login_exc
        self.login_calls.append(
            ("universal", {"client_id": client_id, "client_secret": client_secret})
        )

    def _token_login(self, token: str):
        if self._login_exc:
            raise self._login_exc
        self.login_calls.append(("token", {"token": token}))

    def _list_secrets(self, *, project_id, environment_slug, secret_path):
        if self._list_exc:
            raise self._list_exc
        self.list_secrets_calls.append(
            {
                "project_id": project_id,
                "environment_slug": environment_slug,
                "secret_path": secret_path,
            }
        )
        return self._list_return


def _fake_secret(key: str, value: str):
    """Minimal stand-in for infisical_sdk.api_types.BaseSecret."""
    return SimpleNamespace(secretKey=key, secretValue=value)


def _install_fake_sdk_client(monkeypatch, client: _FakeClient) -> list[str]:
    """Patch the SDK client so fetch() builds our fake instead of hitting the network.

    Returns a list that captures the ``host`` passed to each constructor
    call — tests can assert on it without poking the fake client.
    """
    hosts_seen: list[str] = []

    def _factory(host: str):
        hosts_seen.append(host)
        client.host = host
        return client

    import infisical_sdk

    monkeypatch.setattr(infisical_sdk, "InfisicalSDKClient", _factory)
    return hosts_seen


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def setup_method(self) -> None:
        _reset_for_tests()

    def test_infisical_provider_resolves(self) -> None:
        provider = get_provider("infisical")
        assert provider.name == "infisical"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(SecretProviderError):
            get_provider("nope")

    def test_instances_are_cached(self) -> None:
        a = get_provider("infisical")
        b = get_provider("infisical")
        assert a is b


# ---------------------------------------------------------------------------
# Infisical provider
# ---------------------------------------------------------------------------


class TestInfisicalProvider:
    def setup_method(self) -> None:
        # Clear any ambient env so one test's setenv doesn't leak.
        for name in (
            "INFISICAL_TOKEN",
            "INFISICAL_CLIENT_ID",
            "INFISICAL_CLIENT_SECRET",
            "INFISICAL_PROJECT_ID",
            "INFISICAL_ENVIRONMENT",
            "INFISICAL_PATH",
            "INFISICAL_HOST",
        ):
            import os

            os.environ.pop(name, None)

    def test_no_credentials_returns_error(self) -> None:
        """With neither universal-auth nor token env vars set, surface a
        clear message that names both auth options."""
        result = InfisicalProvider().fetch({"project_id": "p"})
        assert result.secrets == {}
        assert result.error is not None
        assert "INFISICAL_CLIENT_ID" in result.error
        assert "INFISICAL_TOKEN" in result.error

    def test_missing_project_id_returns_error(self, monkeypatch) -> None:
        monkeypatch.setenv("INFISICAL_TOKEN", "tok")
        result = InfisicalProvider().fetch({})
        assert result.secrets == {}
        assert "project_id" in (result.error or "")

    def test_universal_auth_preferred_over_token(self, monkeypatch) -> None:
        """When both credentials are present, client-id/secret wins —
        that's the path Infisical recommends."""
        monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
        monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "cs")
        monkeypatch.setenv("INFISICAL_TOKEN", "leftover-token")
        client = _FakeClient(
            list_secrets_return=SimpleNamespace(secrets=[_fake_secret("ALPACA_API_KEY", "AK")])
        )
        _install_fake_sdk_client(monkeypatch, client)

        result = InfisicalProvider().fetch(
            {"project_id": "proj", "environment": "prod", "path": "/trading"}
        )

        assert result.error is None
        assert result.secrets == {"ALPACA_API_KEY": "AK"}
        assert client.login_calls == [("universal", {"client_id": "cid", "client_secret": "cs"})]
        assert client.list_secrets_calls == [
            {"project_id": "proj", "environment_slug": "prod", "secret_path": "/trading"}
        ]

    def test_token_auth_used_when_only_token_present(self, monkeypatch) -> None:
        monkeypatch.setenv("INFISICAL_TOKEN", "tok")
        client = _FakeClient(
            list_secrets_return=SimpleNamespace(secrets=[_fake_secret("DEBUG", "true")])
        )
        _install_fake_sdk_client(monkeypatch, client)

        result = InfisicalProvider().fetch({"project_id": "proj"})

        assert result.error is None
        assert result.secrets == {"DEBUG": "true"}
        assert client.login_calls == [("token", {"token": "tok"})]

    def test_auth_failure_surfaces_error(self, monkeypatch) -> None:
        monkeypatch.setenv("INFISICAL_TOKEN", "bad")
        client = _FakeClient(login_exc=RuntimeError("invalid token"))
        _install_fake_sdk_client(monkeypatch, client)
        result = InfisicalProvider().fetch({"project_id": "proj"})
        assert result.secrets == {}
        assert "authentication failed" in (result.error or "").lower()

    def test_list_secrets_failure_surfaces_error(self, monkeypatch) -> None:
        monkeypatch.setenv("INFISICAL_TOKEN", "tok")
        client = _FakeClient(list_secrets_exc=RuntimeError("network down"))
        _install_fake_sdk_client(monkeypatch, client)
        result = InfisicalProvider().fetch({"project_id": "proj"})
        assert result.secrets == {}
        assert "list_secrets failed" in (result.error or "")

    def test_host_routing_uses_config_then_env_then_default(self, monkeypatch) -> None:
        """config.base_url beats INFISICAL_HOST env beats the public default."""
        monkeypatch.setenv("INFISICAL_TOKEN", "tok")
        monkeypatch.setenv("INFISICAL_HOST", "https://env.example.com/")
        client = _FakeClient(list_secrets_return=SimpleNamespace(secrets=[]))
        hosts = _install_fake_sdk_client(monkeypatch, client)
        InfisicalProvider().fetch(
            {"project_id": "p", "base_url": "https://self-hosted.example.com"}
        )
        # Config value wins; trailing slash stripped.
        assert hosts == ["https://self-hosted.example.com"]


# ---------------------------------------------------------------------------
# Session merge
# ---------------------------------------------------------------------------


def _state(
    *,
    env: dict[str, str] | None = None,
    secret_manager_config: dict | None = None,
) -> NotebookState:
    return NotebookState(
        id="test",
        env=env or {},
        secret_manager_config=secret_manager_config or {},
    )


class TestApplySecretsToNotebookState:
    def test_no_secrets_block_stamps_manual_sources(self) -> None:
        state = _state(env={"DEBUG": "true", "LOG_LEVEL": "info"})
        result = apply_secrets_to_notebook_state(state)
        assert result is None
        assert state.env == {"DEBUG": "true", "LOG_LEVEL": "info"}
        assert state.env_sources == {"DEBUG": MANUAL_SOURCE, "LOG_LEVEL": MANUAL_SOURCE}
        assert state.env_fetch_error is None

    def test_fetched_secrets_fill_empty_values(self, monkeypatch) -> None:
        # Existing env has the key as a blanked sensitive placeholder —
        # typical state after reload from disk.
        state = _state(
            env={"OPENAI_API_KEY": "", "DEBUG": "true"},
            secret_manager_config={"provider": "infisical", "project_id": "p"},
        )
        _install_fake_provider(
            monkeypatch,
            secrets={"OPENAI_API_KEY": "sk-real", "NEW_KEY": "added"},
        )
        result = apply_secrets_to_notebook_state(state)
        assert result is not None and result.error is None
        assert state.env["OPENAI_API_KEY"] == "sk-real"
        assert state.env["DEBUG"] == "true"
        assert state.env["NEW_KEY"] == "added"
        assert state.env_sources["OPENAI_API_KEY"] == "infisical"
        assert state.env_sources["DEBUG"] == MANUAL_SOURCE
        assert state.env_sources["NEW_KEY"] == "infisical"

    def test_manual_override_wins_over_fetched(self, monkeypatch) -> None:
        state = _state(
            env={"OPENAI_API_KEY": "session-override"},
            secret_manager_config={"provider": "infisical", "project_id": "p"},
        )
        _install_fake_provider(monkeypatch, secrets={"OPENAI_API_KEY": "from-infisical"})
        apply_secrets_to_notebook_state(state)
        assert state.env["OPENAI_API_KEY"] == "session-override"
        assert state.env_sources["OPENAI_API_KEY"] == MANUAL_SOURCE

    def test_a_refresh_picks_up_a_rotated_secret(self, monkeypatch) -> None:
        """The first fetch leaves the old value in env, and the merge read any
        value already there as a manual override, so a rotation never took."""
        state = _state(secret_manager_config={"provider": "infisical", "project_id": "p"})
        _install_fake_provider(monkeypatch, secrets={"OPENAI_API_KEY": "sk-old"})
        apply_secrets_to_notebook_state(state)

        _install_fake_provider(monkeypatch, secrets={"OPENAI_API_KEY": "sk-rotated"})
        apply_secrets_to_notebook_state(state)

        assert state.env["OPENAI_API_KEY"] == "sk-rotated"
        assert state.env_sources["OPENAI_API_KEY"] == "infisical"

    def test_a_manual_edit_of_a_fetched_key_survives_a_refresh(self, monkeypatch) -> None:
        state = _state(secret_manager_config={"provider": "infisical", "project_id": "p"})
        _install_fake_provider(monkeypatch, secrets={"OPENAI_API_KEY": "sk-old"})
        apply_secrets_to_notebook_state(state)
        # What the Runtime panel's env edit does.
        state.env["OPENAI_API_KEY"] = "session-override"
        state.env_sources["OPENAI_API_KEY"] = MANUAL_SOURCE

        _install_fake_provider(monkeypatch, secrets={"OPENAI_API_KEY": "sk-rotated"})
        apply_secrets_to_notebook_state(state)

        assert state.env["OPENAI_API_KEY"] == "session-override"

    def test_fetch_error_surfaces_on_state(self, monkeypatch) -> None:
        state = _state(
            env={"DEBUG": "true"},
            secret_manager_config={"provider": "infisical", "project_id": "p"},
        )
        _install_fake_provider(monkeypatch, error="Infisical rejected the token")
        result = apply_secrets_to_notebook_state(state)
        assert result is not None
        assert state.env_fetch_error == "Infisical rejected the token"
        # Existing env is untouched on failure.
        assert state.env == {"DEBUG": "true"}

    def test_unknown_provider_name_surfaces_as_error(self) -> None:
        state = _state(
            env={},
            secret_manager_config={"provider": "vault"},
        )
        result = apply_secrets_to_notebook_state(state)
        assert result is not None
        assert "vault" in (result.error or "").lower() or "unknown" in (result.error or "").lower()
        assert state.env_fetch_error == result.error

    def test_missing_provider_field_is_flagged(self) -> None:
        state = _state(secret_manager_config={"project_id": "p"})
        result = fetch_configured_secrets(state)
        assert result is not None
        assert "provider" in (result.error or "").lower()


def _install_fake_provider(
    monkeypatch,
    *,
    secrets: dict[str, str] | None = None,
    error: str | None = None,
) -> None:
    """Swap the Infisical provider with a canned result."""
    from strata.notebook.secret_manager import registry

    class _Fake:
        name = "infisical"

        def fetch(self, config):
            if error is not None:
                return SecretFetchResult.failure("infisical", error)
            return SecretFetchResult(
                secrets=dict(secrets or {}),
                source="infisical",
                fetched_at="2026-04-22T00:00:00Z",
            )

    registry._cache["infisical"] = _Fake()
    monkeypatch.setattr(registry, "_cache", registry._cache)


# ---------------------------------------------------------------------------
# Route surface
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Open a notebook via the test client so we can hit /secret-manager/refresh."""
    from fastapi.testclient import TestClient

    from strata.notebook.routes import get_session_manager
    from strata.notebook.writer import add_cell_to_notebook, create_notebook

    # Fresh session manager per test.
    get_session_manager()
    try:
        nb_dir = create_notebook(tmp_path, "Secrets Route Test")
        add_cell_to_notebook(nb_dir, "c1")

        # Inject a [secret_manager] block so the refresh path has something to do.
        notebook_toml = nb_dir / "notebook.toml"
        with open(notebook_toml, "a", encoding="utf-8") as f:
            f.write('\n[secret_manager]\nprovider = "infisical"\nproject_id = "p"\n')

        from tests.notebook.e2e_fixtures import create_test_app

        app = create_test_app()
        tc = TestClient(app)
        resp = tc.post("/v1/notebooks/open", json={"path": str(nb_dir)})
        assert resp.status_code == 200, resp.text
        session_id = resp.json()["session_id"]
        yield tc, session_id, monkeypatch
    finally:
        _reset_for_tests()


class TestRefreshEndpoint:
    def test_refresh_returns_env_sources(self, client) -> None:
        tc, session_id, monkeypatch = client
        _install_fake_provider(monkeypatch, secrets={"ALPACA_API_KEY": "AKROT8"})
        resp = tc.post(f"/v1/notebooks/{session_id}/secret-manager/refresh")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["env"]["ALPACA_API_KEY"] == "AKROT8"
        assert body["env_sources"]["ALPACA_API_KEY"] == "infisical"
        assert body["env_fetch_error"] is None

    def test_refresh_surfaces_fetch_error(self, client) -> None:
        tc, session_id, monkeypatch = client
        _install_fake_provider(monkeypatch, error="Infisical down")
        resp = tc.post(f"/v1/notebooks/{session_id}/secret-manager/refresh")
        assert resp.status_code == 200
        body = resp.json()
        assert body["env_fetch_error"] == "Infisical down"

    def test_refresh_unknown_notebook_returns_404(self, client) -> None:
        tc, _session_id, _ = client
        resp = tc.post("/v1/notebooks/does-not-exist/secret-manager/refresh")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Writer: update_notebook_secret_manager
# ---------------------------------------------------------------------------


class TestUpdateNotebookSecretManager:
    def test_writes_cleaned_config_to_toml(self, tmp_path) -> None:
        import tomllib

        from strata.notebook.writer import create_notebook, update_notebook_secret_manager

        nb_dir = create_notebook(tmp_path, "Secrets Writer Test")
        update_notebook_secret_manager(
            nb_dir,
            {
                "provider": "infisical",
                "project_id": "proj",
                "environment": "prod",
                "path": "/trading",
            },
        )
        with open(nb_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["secret_manager"] == {
            "provider": "infisical",
            "project_id": "proj",
            "environment": "prod",
            "path": "/trading",
        }

    def test_strips_unknown_keys(self, tmp_path) -> None:
        """Only the whitelisted keys may make it into notebook.toml — this
        stops a malicious PUT payload from smuggling arbitrary state."""
        import tomllib

        from strata.notebook.writer import create_notebook, update_notebook_secret_manager

        nb_dir = create_notebook(tmp_path, "Secrets Filter Test")
        update_notebook_secret_manager(
            nb_dir,
            {"provider": "infisical", "project_id": "p", "secret_value": "LEAK"},
        )
        with open(nb_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert "secret_value" not in data["secret_manager"]

    def test_empty_payload_removes_block(self, tmp_path) -> None:
        import tomllib

        from strata.notebook.writer import create_notebook, update_notebook_secret_manager

        nb_dir = create_notebook(tmp_path, "Secrets Disconnect Test")
        update_notebook_secret_manager(nb_dir, {"provider": "infisical", "project_id": "p"})
        update_notebook_secret_manager(nb_dir, {})
        with open(nb_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert "secret_manager" not in data

    def test_same_config_is_no_op_no_updated_at_bump(self, tmp_path) -> None:
        """Re-saving identical values shouldn't churn updated_at — matches
        the write-if-changed pattern used by update_notebook_env etc."""
        from strata.notebook.writer import create_notebook, update_notebook_secret_manager

        nb_dir = create_notebook(tmp_path, "Secrets Idempotent Test")
        update_notebook_secret_manager(nb_dir, {"provider": "infisical", "project_id": "p"})
        before = (nb_dir / "notebook.toml").read_bytes()
        update_notebook_secret_manager(nb_dir, {"provider": "infisical", "project_id": "p"})
        after = (nb_dir / "notebook.toml").read_bytes()
        assert before == after


# ---------------------------------------------------------------------------
# Route: PUT /secret-manager/config
# ---------------------------------------------------------------------------


class TestUpdateSecretManagerConfigEndpoint:
    def test_saves_and_returns_config(self, client) -> None:
        tc, session_id, monkeypatch = client
        # Replace the fake provider's fetch so the subsequent reload's
        # apply_secrets_to_notebook_state doesn't hit real infisical.
        _install_fake_provider(monkeypatch, secrets={})
        resp = tc.put(
            f"/v1/notebooks/{session_id}/secret-manager/config",
            json={
                "provider": "infisical",
                "project_id": "new-proj",
                "environment": "prod",
                "path": "/trading",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["secret_manager_config"]["project_id"] == "new-proj"
        assert body["secret_manager_config"]["environment"] == "prod"

    def test_empty_payload_disconnects(self, client) -> None:
        tc, session_id, monkeypatch = client
        _install_fake_provider(monkeypatch, secrets={})
        resp = tc.put(f"/v1/notebooks/{session_id}/secret-manager/config", json={})
        assert resp.status_code == 200
        assert resp.json()["secret_manager_config"] == {}

    def test_unknown_notebook_returns_404(self, client) -> None:
        tc, _session_id, _ = client
        resp = tc.put(
            "/v1/notebooks/does-not-exist/secret-manager/config",
            json={"provider": "infisical"},
        )
        assert resp.status_code == 404
