"""The adaptive controller must be fed by real traffic.

Nothing called ``record_latency`` or ``record_queue_wait``, so ``get_p95()`` stayed ``None`` and
every tick was a no-op. Control-loop unit tests feed the controller by hand, so only a test of the
wiring catches this.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def booted(tmp_path, monkeypatch):
    """Boot the app through its real lifespan, returning the server state."""

    def _boot(**env: str):
        monkeypatch.setenv("STRATA_DEPLOYMENT_MODE", "personal")
        monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
        monkeypatch.setenv("STRATA_CACHE_DIR", str(tmp_path / "cache"))
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        import strata.server as server_module

        with TestClient(server_module.app):
            return server_module._state

    return _boot


def test_admission_feeds_the_controller_when_adaptive_is_on(booted):
    state = booted(STRATA_ADAPTIVE_ENABLED="true")

    assert state._adaptive_controller is not None
    # Admission holds the same controller object the lifespan started, so its two
    # signals reach the control loop.
    assert state.qos._controller is state._adaptive_controller


def test_nothing_is_attached_when_adaptive_is_off(booted):
    state = booted()

    assert state.config.adaptive_enabled is False
    assert state.qos._controller is None
