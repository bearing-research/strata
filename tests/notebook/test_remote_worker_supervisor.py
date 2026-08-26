"""Tests for the RemoteWorkerSupervisor (P2 of the SSH remote-worker path).

Every external effect is faked — the tunnel launcher, the health probe, the port
picker, and the SSH runner — so these assert the supervisor's orchestration
(provision → tunnel → health-check → record, plus reconcile / teardown /
shutdown) with no real ssh, socket, or HTTP.
"""

from __future__ import annotations

import pytest

from strata.notebook.remote_worker_supervisor import (
    RemoteWorkerSupervisor,
    SshWorkerError,
)
from tests.notebook.test_ssh_worker import ScriptedSshRunner, _ok

_INSTALLED = "worker=/usr/bin/strata-worker\nuv=/usr/bin/uv\nplatform=Linux x86_64\nversion=0.6.0\n"
_MISSING = "worker=\nuv=/usr/bin/uv\nplatform=Linux x86_64\nversion=\n"
_NO_UV = "worker=\nuv=\nplatform=Linux x86_64\nversion=\n"


def _runner(detect: str = _INSTALLED):
    """A scripted SSH runner for a box that provisions + launches cleanly."""
    return ScriptedSshRunner(
        [
            (lambda c: c == "true", _ok()),  # preflight
            (lambda c: "command -v strata-worker" in c, _ok(detect)),  # detect probe
            (lambda c: "uv tool install" in c, _ok()),  # install (if reached)
            (lambda c: c.startswith("cat "), _ok("")),  # is_running: no pidfile
            (lambda c: "nohup strata-worker" in c, _ok("4321\n")),  # launch → pid
            (lambda c: "kill" in c and "kill -0" not in c, _ok("stopped\n")),  # stop
        ]
    )


class FakeTunnelHandle:
    def __init__(self) -> None:
        self.alive = True
        self.terminated = 0

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.alive = False
        self.terminated += 1


class FakeTunnelLauncher:
    def __init__(self) -> None:
        self.spawns: list[tuple[str, int, int]] = []
        self.handles: list[FakeTunnelHandle] = []

    def spawn(self, ssh_target, *, local_port, remote_port):
        self.spawns.append((ssh_target, local_port, remote_port))
        handle = FakeTunnelHandle()
        self.handles.append(handle)
        return handle


def _supervisor(runner=None, *, healthy=True, launcher=None):
    launcher = launcher or FakeTunnelLauncher()
    the_runner = runner or _runner()
    sup = RemoteWorkerSupervisor(
        tunnel_launcher=launcher,
        health_probe=lambda port: healthy() if callable(healthy) else healthy,
        port_picker=lambda: 55001,
        runner_factory=lambda target: the_runner,
    )
    return sup, launcher, the_runner


# ---------------------------------------------------------------------------
# establish
# ---------------------------------------------------------------------------


def test_establish_happy_path():
    sup, launcher, _ = _supervisor()
    rec = sup.establish("gpu", "user@box")
    assert (rec.name, rec.local_port, rec.remote_port, rec.remote_pid) == ("gpu", 55001, 9000, 4321)
    assert rec.healthy is True
    assert rec.executor_url == "http://127.0.0.1:55001/v1/execute"
    # Tunnel opened local:55001 → remote:9000 on the right target.
    assert launcher.spawns == [("user@box", 55001, 9000)]
    assert sup.token_for("gpu")  # a bearer token was generated (held in memory)
    assert sup.get("gpu") == rec


def test_establish_uses_explicit_ports_and_token():
    sup, launcher, _ = _supervisor()
    rec = sup.establish("gpu", "user@box", remote_port=9100, local_port=6000, token="mytok")
    assert rec.local_port == 6000
    assert rec.remote_port == 9100
    assert launcher.spawns == [("user@box", 6000, 9100)]
    assert sup.token_for("gpu") == "mytok"


def test_establish_installs_when_missing():
    runner = _runner(detect=_MISSING)
    sup, _, _ = _supervisor(runner=runner)
    sup.establish("gpu", "user@box")
    assert any("uv tool install" in c for c in runner.calls)


def test_establish_health_failure_cleans_up():
    sup, launcher, _ = _supervisor(healthy=False)
    with pytest.raises(SshWorkerError, match="/health didn't respond"):
        sup.establish("gpu", "user@box", health_timeout=0)
    assert launcher.handles[0].terminated == 1  # tunnel torn down on failure
    assert sup.get("gpu") is None  # not recorded


def test_establish_install_false_when_missing_raises():
    sup, _, _ = _supervisor(runner=_runner(detect=_MISSING))
    with pytest.raises(SshWorkerError, match="isn't installed"):
        sup.establish("gpu", "user@box", install=False)


def test_establish_refuses_without_uv():
    sup, _, _ = _supervisor(runner=_runner(detect=_NO_UV))
    with pytest.raises(SshWorkerError, match="neither `strata-worker` nor `uv`"):
        sup.establish("gpu", "user@box")


def test_establish_is_idempotent_per_name():
    sup, launcher, _ = _supervisor()
    sup.establish("gpu", "user@box")
    sup.establish("gpu", "user@box")  # re-establish tears the first tunnel down
    assert launcher.handles[0].terminated == 1
    assert len(launcher.spawns) == 2
    assert sup.get("gpu").local_port == 55001


# ---------------------------------------------------------------------------
# reconcile / status / teardown / shutdown
# ---------------------------------------------------------------------------


def test_reconcile_respawns_a_dead_tunnel():
    sup, launcher, _ = _supervisor()
    sup.establish("gpu", "user@box")
    launcher.handles[0].alive = False  # the ssh -L died
    sup.reconcile()
    assert len(launcher.spawns) == 2  # a fresh forward on the same ports
    assert launcher.spawns[1] == ("user@box", 55001, 9000)
    assert sup.get("gpu").healthy is True


def test_reconcile_leaves_a_healthy_tunnel_alone():
    sup, launcher, _ = _supervisor()
    sup.establish("gpu", "user@box")
    sup.reconcile()
    assert len(launcher.spawns) == 1  # no respawn


def test_status_refreshes_health():
    sup, launcher, _ = _supervisor()
    sup.establish("gpu", "user@box")
    launcher.handles[0].alive = False
    assert sup.status()[0].healthy is False


def test_teardown_terminates_and_forgets():
    sup, launcher, _ = _supervisor()
    sup.establish("gpu", "user@box")
    assert sup.teardown("gpu") is True
    assert launcher.handles[0].terminated == 1
    assert sup.get("gpu") is None
    assert sup.teardown("gpu") is False  # already gone


def test_teardown_stop_remote_kills_worker():
    runner = _runner()
    sup, _, _ = _supervisor(runner=runner)
    sup.establish("gpu", "user@box")
    sup.teardown("gpu", stop_remote=True)
    assert any("kill" in c and "kill -0" not in c for c in runner.calls)


def test_shutdown_tears_down_all():
    sup, launcher, _ = _supervisor()
    sup.establish("gpu", "user@box")
    sup.establish("cpu", "user@box")
    sup.shutdown()
    assert all(h.terminated >= 1 for h in launcher.handles)
    assert sup.status() == []


def test_establish_publishes_runtime_token_teardown_clears():
    from strata.notebook.worker_secrets import (
        clear_runtime_worker_token,
        get_runtime_worker_token,
    )

    sup, _, _ = _supervisor()
    try:
        sup.establish("gpu", "user@box", token="tok123")
        # The executor can now authenticate dispatch to "gpu" by name.
        assert get_runtime_worker_token("gpu") == "tok123"
        sup.teardown("gpu")
        assert get_runtime_worker_token("gpu") is None
    finally:
        clear_runtime_worker_token("gpu")


def test_shutdown_clears_runtime_tokens():
    from strata.notebook.worker_secrets import (
        clear_runtime_worker_token,
        get_runtime_worker_token,
    )

    sup, _, _ = _supervisor()
    try:
        sup.establish("gpu", "user@box", token="tok123")
        sup.shutdown()
        assert get_runtime_worker_token("gpu") is None
    finally:
        clear_runtime_worker_token("gpu")
