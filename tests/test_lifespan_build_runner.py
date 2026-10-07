"""Lifespan shutdown must stop the build runner it started, not the global one.

Adopting a runner left registered by an earlier lifespan awaits a heartbeat task on a dead event
loop ("attached to a different loop"). Shutdown also clears the registry.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.transforms.runner import get_build_runner, reset_build_runner, set_build_runner


class _ForeignRunner:
    """A runner registered by someone else, on a loop that is gone.

    ``stop()`` raises like awaiting a cross-loop task, so reaching for it fails shutdown loudly.
    """

    def __init__(self) -> None:
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1
        raise RuntimeError("got Future attached to a different loop")


@pytest.fixture
def boot(tmp_path, monkeypatch):
    def _boot(**env: str):
        monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
        monkeypatch.setenv("STRATA_CACHE_DIR", str(tmp_path / "cache"))
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        import strata.server as server_module

        with TestClient(server_module.app):
            pass

    return _boot


@pytest.fixture(autouse=True)
def _clear_runner():
    yield
    reset_build_runner()


def test_a_lifespan_that_started_no_runner_does_not_stop_someone_elses(boot):
    # Service mode without ``[tool.strata.transforms] enabled`` starts no runner.
    foreign = _ForeignRunner()
    set_build_runner(foreign)

    boot(
        STRATA_DEPLOYMENT_MODE="service",
        STRATA_AUTH_MODE="trusted_proxy",
        STRATA_PROXY_TOKEN="test-token",
    )

    assert foreign.stop_calls == 0
    # Still cleared: leaving it registered is how the next lifespan would inherit it.
    assert get_build_runner() is None


def test_a_lifespan_stops_the_runner_it_started(boot):
    # Personal mode always runs embedded transforms, so it owns a runner and must
    # shut it down.
    boot(STRATA_DEPLOYMENT_MODE="personal")

    assert get_build_runner() is None


def test_two_lifespans_in_one_process_leave_nothing_registered(boot):
    boot(STRATA_DEPLOYMENT_MODE="personal")
    boot(
        STRATA_DEPLOYMENT_MODE="service",
        STRATA_AUTH_MODE="trusted_proxy",
        STRATA_PROXY_TOKEN="test-token",
    )

    assert get_build_runner() is None
