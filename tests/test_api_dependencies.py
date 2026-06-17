"""Unit tests for the typed data-plane dependencies (``strata.api.dependencies``).

The point of the dependencies is that a read route structurally cannot have
opened the write gate. These tests exercise that at the dependency layer,
without HTTP: under one service-mode config the read dependency yields a store
while the registry write path is refused.
"""

import pytest
from fastapi import HTTPException

from strata import server
from strata.api.dependencies import read_store, registry_decision
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
