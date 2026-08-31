"""The adaptive controller must be connected to something that feeds it.

``AdaptiveConcurrencyController`` was constructed and started in the lifespan,
logged ``"Adaptive concurrency control started"``, and published a populated
``adaptive_concurrency`` block on ``/metrics`` — while nothing anywhere called
``record_latency`` or ``record_queue_wait``. Its windows stayed empty, so
``get_p95()`` returned ``None`` and every tick returned early: a five-second
timer that adjusted nothing, for eight months (#549).

The unit tests around the control loop all passed throughout, because they
feed the controller by hand. Only the wire was missing, so only a test of the
wire catches it coming loose again.
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
    # The single fact that #549 was about: admission holds the same controller
    # object the lifespan started, so its two signals reach the control loop.
    assert state.qos._controller is state._adaptive_controller


def test_nothing_is_attached_when_adaptive_is_off(booted):
    state = booted()

    assert state.config.adaptive_enabled is False
    assert state.qos._controller is None
