"""Lifespan shutdown must stop the build runner it started, not the global one.

Startup assigns a local ``build_runner`` (``None`` when transforms are not
enabled for the mode). Shutdown then *reassigned* that name from
``get_build_runner()`` and awaited ``stop()`` on whatever was registered — so a
lifespan that started no runner would adopt one left behind by an earlier
lifespan in the same process and await a heartbeat task belonging to an event
loop that no longer exists:

    RuntimeError: Task ... got Future <Task cancelling
    name=... coro=<BuildRunner._heartbeat_loop()>> attached to a different loop

That is the shape of the intermittent teardown error seen on CI, on whichever
test happened to boot a lifespan after a runner was left registered on the same
xdist worker. Ownership is the fix: stop what you started, and clear the
registry either way so nothing downstream inherits a dead runner.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.transforms.runner import get_build_runner, reset_build_runner, set_build_runner


class _ForeignRunner:
    """A runner registered by someone else, on a loop that is gone.

    ``stop()`` raises the way awaiting a cross-loop task does, so a lifespan
    that reaches for this object fails its shutdown instead of quietly
    succeeding.
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

    boot(STRATA_DEPLOYMENT_MODE="service")

    assert foreign.stop_calls == 0
    # Still cleared: leaving it registered is how the next lifespan inherits it.
    assert get_build_runner() is None


def test_a_lifespan_stops_the_runner_it_started(boot):
    # Personal mode always runs embedded transforms, so this one owns a runner
    # and must shut it down.
    boot(STRATA_DEPLOYMENT_MODE="personal")

    assert get_build_runner() is None


def test_two_lifespans_in_one_process_leave_nothing_registered(boot):
    boot(STRATA_DEPLOYMENT_MODE="personal")
    boot(STRATA_DEPLOYMENT_MODE="service")

    assert get_build_runner() is None
