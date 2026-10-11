"""Unit tests for the typed data-plane dependencies (``strata.api.dependencies``).

A read route structurally cannot open the write gate: under one service-mode config the read
dependency yields a store while the registry write path is refused.
"""

import pytest
from fastapi import HTTPException

from strata import server
from strata.api.dependencies import (
    build_transport_available,
    read_store,
    registry_decision,
    require_build_store,
    require_build_transport_store,
    runtime_build_store,
)
from strata.artifact_store import reset_artifact_store
from strata.auth import principal_context
from strata.config import StrataConfig
from strata.server import ServerState
from strata.types import Principal


def _set_state(**overrides) -> None:
    config = StrataConfig(host="127.0.0.1", port=8765, **overrides)
    reset_artifact_store()
    server._state = ServerState(config)


@pytest.fixture(autouse=True)
def _restore_state():
    saved = server._state
    try:
        yield
    finally:
        server._state = saved
        reset_artifact_store()


def test_read_dependency_cannot_reach_write_gate_in_service_mode(tmp_path):
    """Same service-mode config: ``ReadStore`` resolves, the registry write gate 403s without
    ``service_writes_enabled``.
    """
    _set_state(
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        artifact_dir=str(tmp_path / "artifacts"),
    )

    # Read gate opens.
    assert read_store() is not None

    # Write path is refused under the very same config, even for an approver.
    approver = Principal(id="admin", scopes=frozenset({"admin:*"}))
    with principal_context(approver), pytest.raises(HTTPException) as exc:
        registry_decision()
    assert exc.value.status_code == 403
    assert exc.value.detail["error"] == "writes_disabled"


def test_a_refused_artifact_route_names_only_settings_that_would_open_it(tmp_path):
    """With transforms on, a personal-only route must not still say "enable transforms"."""
    _set_state(
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        artifact_dir=str(tmp_path / "artifacts"),
        transforms_config={"enabled": True},
    )

    with pytest.raises(HTTPException) as personal_only:
        server._get_artifact_store()
    with pytest.raises(HTTPException) as write_route:
        server._get_artifact_store(allow_write=True)

    assert "transforms" not in personal_only.value.detail["message"]
    assert "deployment_mode='personal'" in personal_only.value.detail["message"]
    assert "STRATA_SERVICE_WRITES_ENABLED" in write_route.value.detail["message"]


def test_personal_mode_opens_both_gates(tmp_path):
    """Personal mode is the single operator: read and registry write both resolve."""
    _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))

    assert read_store() is not None

    decision = registry_decision()
    assert decision.store is not None
    assert decision.principal is None  # no auth in personal mode


# --- Build-store / signed-transport gate ---


def test_build_transport_gate_open_in_personal_mode(tmp_path):
    """Personal mode (``writes_enabled``) makes ``BuildTransportStore`` resolve to a real store."""
    _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))

    assert build_transport_available() is True
    assert require_build_transport_store() is not None


def test_build_transport_gate_404s_in_service_mode(tmp_path):
    """Service mode without server transforms cannot honor signed build URLs, so the dependency
    404s.
    """
    _set_state(
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert build_transport_available() is False
    with pytest.raises(HTTPException) as exc:
        require_build_transport_store()
    assert exc.value.status_code == 404


def test_require_build_store_500s_without_artifact_dir(tmp_path, monkeypatch):
    """An uninitialized build store is a 500 (misconfiguration), not a 404.

    The resolver is forced to ``None`` so the test does not depend on a mode without
    ``artifact_dir``.
    """
    _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))

    monkeypatch.setattr("strata.api.dependencies.runtime_build_store", lambda: None)
    with pytest.raises(HTTPException) as exc:
        require_build_store()
    assert exc.value.status_code == 500


def test_runtime_build_store_resolves_when_artifact_dir_set(tmp_path):
    _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))

    assert runtime_build_store() is not None


class TestArtifactInputTenantGate:
    """Artifact transform inputs clear the same tenant gate as ``GET /v1/artifacts/{id}``.

    Otherwise naming another tenant's artifact as an input reads its blob, or in pull mode yields a
    signed download URL for it.
    """

    def _seed(self, tmp_path, *, tenant):
        from strata.artifact_store import get_artifact_store

        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        _set_state(
            deployment_mode="service",
            multi_tenant_enabled=True,
            # Coherence: multi-tenant requires trusted-proxy auth.
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            artifact_dir=artifact_dir,
            # resolve_input_version reaches the store via
            # _get_artifact_store(allow_server_mode=True), which in service
            # mode needs the transforms allowlist enabled.
            transforms_config={"enabled": True, "registry": []},
        )
        store = get_artifact_store(server._state.config.artifact_dir)
        assert store is not None
        version = store.create_artifact(artifact_id="secret", provenance_hash="p1", tenant=tenant)
        return f"strata://artifact/secret@v={version}"

    def test_cross_tenant_artifact_input_is_refused(self, tmp_path):
        from strata.api.dependencies import resolve_input_version

        uri = self._seed(tmp_path, tenant="tenant-b")
        with pytest.raises(HTTPException) as exc:
            resolve_input_version(uri, tenant="tenant-a")
        assert exc.value.status_code in (403, 404)

    def test_same_tenant_artifact_input_resolves(self, tmp_path):
        from strata.api.dependencies import resolve_input_version

        uri = self._seed(tmp_path, tenant="tenant-b")
        assert resolve_input_version(uri, tenant="tenant-b") == "secret@v=1"

    def test_unknown_artifact_input_is_404(self, tmp_path):
        from strata.api.dependencies import resolve_input_version

        self._seed(tmp_path, tenant="tenant-b")
        with pytest.raises(HTTPException) as exc:
            resolve_input_version("strata://artifact/ghost@v=1", tenant="tenant-b")
        assert exc.value.status_code == 404


class TestArtifactListPaginationIsBounded:
    """SQLite treats a negative LIMIT as unbounded, so ``?limit=-1`` would load every row."""

    def test_negative_and_oversized_limits_are_rejected(self, tmp_path):
        from fastapi.testclient import TestClient

        from strata.server import app

        _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))
        client = TestClient(app)

        assert client.get("/v1/artifacts", params={"limit": -1}).status_code == 422
        assert client.get("/v1/artifacts", params={"limit": 10_000_000}).status_code == 422
        assert client.get("/v1/artifacts", params={"offset": -5}).status_code == 422
        # A sane request still works.
        assert client.get("/v1/artifacts", params={"limit": 10}).status_code == 200

    def test_superseded_is_a_state_the_listing_filters_by(self, tmp_path):
        """Callers are handed superseded versions, so they can list them."""
        from fastapi.testclient import TestClient

        from strata.artifact_store import ArtifactStore
        from strata.server import app

        _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))
        store = ArtifactStore(tmp_path / "artifacts")
        for _ in range(2):
            version = store.create_artifact("refreshed", "same-prov")
            store.finalize_artifact("refreshed", version, "{}", 1, 10)
        client = TestClient(app)

        def listed(state: str) -> list[tuple[int, str]]:
            response = client.get("/v1/artifacts", params={"state": state})
            assert response.status_code == 200
            return [(a["version"], a["state"]) for a in response.json()["artifacts"]]

        assert listed("superseded") == [(1, "superseded")]
        assert listed("ready") == [(2, "ready")]
        assert client.get("/v1/artifacts", params={"state": "bogus"}).status_code == 400

    def test_stats_and_usage_count_superseded_versions(self, tmp_path):
        """The Artifacts page shows the superseded count beside the other states."""
        from fastapi.testclient import TestClient

        from strata.artifact_store import ArtifactStore
        from strata.server import app

        _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))
        store = ArtifactStore(tmp_path / "artifacts")
        for _ in range(2):
            version = store.create_artifact("refreshed", "same-prov")
            store.finalize_artifact("refreshed", version, "{}", 1, 10)
        client = TestClient(app)

        for route in ("/v1/artifacts/stats", "/v1/artifacts/usage"):
            body = client.get(route).json()
            assert (body["ready_versions"], body["superseded_versions"]) == (1, 1)
