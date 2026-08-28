"""Unit tests for the typed data-plane dependencies (``strata.api.dependencies``).

The point of the dependencies is that a read route structurally cannot have
opened the write gate. These tests exercise that at the dependency layer,
without HTTP: under one service-mode config the read dependency yields a store
while the registry write path is refused.
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
from strata.config import StrataConfig
from strata.server import ServerState


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
    """Same service-mode config: reads open, the registry write gate is refused.

    A read route asking for ``ReadStore`` gets a usable store; the registry
    decision path (which opens the service-mode write gate) 403s without
    ``service_writes_enabled``. The read handler has no way to express the write
    gate — that is the invariant the decomposition exists to protect.
    """
    _set_state(deployment_mode="service", artifact_dir=str(tmp_path / "artifacts"))

    # Read gate opens.
    assert read_store() is not None

    # Write path is refused under the very same config.
    with pytest.raises(HTTPException) as exc:
        registry_decision()
    assert exc.value.status_code == 403
    assert exc.value.detail["error"] == "writes_disabled"


def test_personal_mode_opens_both_gates(tmp_path):
    """Personal mode is the single operator: read + registry write both resolve."""
    _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))

    assert read_store() is not None

    decision = registry_decision()
    assert decision.store is not None
    assert decision.principal is None  # no auth in personal mode


# --- Build-store / signed-transport gate (#295) -----------------------------


def test_build_transport_gate_open_in_personal_mode(tmp_path):
    """Personal mode (``writes_enabled``) makes signed build transport available.

    ``BuildTransportStore`` then resolves to a real build store rather than 404ing.
    """
    _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))

    assert build_transport_available() is True
    assert require_build_transport_store() is not None


def test_build_transport_gate_404s_in_service_mode(tmp_path):
    """Service mode without server transforms cannot issue/honor signed build URLs.

    The transport dependency 404s — the gate the manifest + finalize routes adopt.
    """
    _set_state(deployment_mode="service", artifact_dir=str(tmp_path / "artifacts"))

    assert build_transport_available() is False
    with pytest.raises(HTTPException) as exc:
        require_build_transport_store()
    assert exc.value.status_code == 404


def test_require_build_store_500s_without_artifact_dir(tmp_path, monkeypatch):
    """An uninitialized build store is a 500 (misconfiguration), not a 404.

    ``runtime_build_store`` returns ``None`` when there is nowhere to track builds;
    the ``RequiredBuildStore`` dependency surfaces that as a 500 before the body
    runs. (Simulated by forcing the resolver to ``None`` so the test does not
    depend on a mode that forbids an ``artifact_dir``.)
    """
    _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))

    monkeypatch.setattr("strata.api.dependencies.runtime_build_store", lambda: None)
    with pytest.raises(HTTPException) as exc:
        require_build_store()
    assert exc.value.status_code == 500


def test_runtime_build_store_resolves_when_artifact_dir_set(tmp_path):
    """With an ``artifact_dir``, the resolver hands back a real build store."""
    _set_state(deployment_mode="personal", artifact_dir=str(tmp_path / "artifacts"))

    assert runtime_build_store() is not None


class TestArtifactInputTenantGate:
    """Artifact transform-inputs must clear the same tenant gate a direct
    ``GET /v1/artifacts/{id}`` runs.

    Before this, ``strata://artifact/{id}@v={n}`` was parsed by regex with no
    store lookup and no tenant filter at any layer, so naming another tenant's
    artifact as a transform input read its blob — and in pull mode handed back
    a signed download URL for it.
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
