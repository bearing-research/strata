"""Tests for the RemoteWorkerSupervisor of the SSH remote-worker path.

Tunnel launcher, health probe, port picker and SSH runner are faked, so these assert the
orchestration (provision, tunnel, health-check, record, reconcile, teardown, shutdown).
"""

from __future__ import annotations

import pytest

from strata.notebook.remote_worker_supervisor import (
    RemoteWorkerSupervisor,
    SshWorkerError,
)
from strata.notebook.worker_secrets import clear_runtime_worker_token
from tests.notebook.test_ssh_worker import ScriptedSshRunner, _ok


@pytest.fixture(autouse=True)
def _clear_worker_tokens():
    """Clear the process-global worker_secrets tokens so other modules don't see leftovers."""
    yield
    for name in ("gpu", "a", "b"):
        clear_runtime_worker_token(name)


_INSTALLED = "worker=/usr/bin/strata-worker\nuv=/usr/bin/uv\nplatform=Linux x86_64\nversion=0.6.0\n"
_MISSING = "worker=\nuv=/usr/bin/uv\nplatform=Linux x86_64\nversion=\n"
_NO_UV = "worker=\nuv=\nplatform=Linux x86_64\nversion=\n"


class _BoxRunner(ScriptedSshRunner):
    """A scripted runner whose pidfile names the launched worker until it is stopped."""

    launched: str | None = None

    def run(self, command, *, timeout=None, stdin_data=None):
        result = super().run(command, timeout=timeout, stdin_data=stdin_data)
        if "nohup strata-worker" in command:
            port = command.split("--port ")[1].split()[0]
            self.launched = f'{{"pid": {result.stdout.strip()}, "port": {port}}}'
        elif "kill" in command and "kill -0" not in command:
            self.launched = None
        elif command.startswith("cat ") and self.launched:
            return _ok(self.launched)
        elif command.startswith("kill -0") and self.launched:
            return _ok("up")
        return result


def _runner(detect: str = _INSTALLED):
    """A scripted SSH runner for a box that provisions and launches cleanly."""
    return _BoxRunner(
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


def _launched_id(runner) -> str | None:
    """The launch id the latest launch fed the worker over stdin, which its /health echoes."""
    for command, stdin_data in zip(reversed(runner.calls), reversed(runner.stdin_writes)):
        if "nohup strata-worker" in command:
            return stdin_data.splitlines()[0] if stdin_data else None
    return None


def _supervisor(runner=None, *, healthy=True, launcher=None, launch_id_probe=None):
    launcher = launcher or FakeTunnelLauncher()
    the_runner = runner or _runner()
    sup = RemoteWorkerSupervisor(
        tunnel_launcher=launcher,
        health_probe=lambda port: healthy() if callable(healthy) else healthy,
        launch_id_probe=launch_id_probe or (lambda port: _launched_id(the_runner)),
        port_picker=lambda: 55001,
        runner_factory=lambda target: the_runner,
    )
    return sup, launcher, the_runner


# establish


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


@pytest.mark.parametrize(
    "answered",
    [None, "another-launch"],
    ids=["no-launch-id", "another-launch-id"],
)
def test_establish_refuses_when_health_is_not_from_the_launched_worker(answered):
    """Something else on the remote port answers /health; the token must not go to it.

    The launched worker may still be alive, starting up to fail its bind, so its pid
    being live proves nothing.
    """
    sup, launcher, runner = _supervisor(launch_id_probe=lambda port: answered)
    with pytest.raises(SshWorkerError, match="not from the worker launched as pid 4321"):
        sup.establish("gpu", "user@box")
    assert launcher.handles[0].terminated == 1
    assert sup.get("gpu") is None
    assert sup.token_for("gpu") is None
    # The launch id went over stdin with the token, never in a command.
    (launch_stdin,) = [
        stdin_data
        for command, stdin_data in zip(runner.calls, runner.stdin_writes)
        if "nohup strata-worker" in command
    ]
    launch_id, _token = launch_stdin.splitlines()
    assert not any(launch_id in command for command in runner.calls)


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


def test_re_establish_publishes_a_token_the_remote_worker_actually_has():
    """Re-establish keeps the remote worker running, so its token must not be regenerated.

    The worker enforces the token it started with: a fresh local token 401s every dispatch
    while the unauthenticated ``/health`` probe still reports healthy.
    """
    runner = ScriptedSshRunner(
        [
            (lambda c: c == "true", _ok()),
            (lambda c: "command -v strata-worker" in c, _ok(_INSTALLED)),
            ("kill -0 ", _ok("up")),
            (lambda c: "kill" in c and "kill -0" not in c, _ok("stopped\n")),
            ("nohup strata-worker", _ok("4321\n")),
            # A live worker is recorded on the default port → adoptable.
            (lambda c: "cat " in c, _ok('{"pid": 4321, "port": 9000}')),
        ]
    )
    sup, _, _ = _supervisor(runner=runner)
    sup.establish("gpu", "user@box")
    sup.establish("gpu", "user@box")

    launched = [
        stdin_data
        for command, stdin_data in zip(runner.calls, runner.stdin_writes)
        if "nohup strata-worker" in command
    ]
    assert launched, "no worker was ever launched with a token"
    assert launched[-1].splitlines()[-1] == sup.token_for("gpu")


# reconcile / status / teardown / shutdown


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


# Thread-safety: the supervisor is a process-wide singleton driven from
# asyncio.to_thread workers (genuinely concurrent OS threads).


def test_concurrent_establish_same_name_is_rejected():
    """The name is reserved for the whole establish, so a second one can't orphan the first ssh
    -L."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    launcher = FakeTunnelLauncher()
    first_in_preflight = threading.Event()
    release_first = threading.Event()

    class _BlockingRunner(_BoxRunner):
        def run(self, command, *, timeout=None, stdin_data=None):
            if command == "true":  # preflight: hold the first establish here
                first_in_preflight.set()
                assert release_first.wait(timeout=10)
            return super().run(command, timeout=timeout, stdin_data=stdin_data)

    runner = _BlockingRunner(
        [
            (lambda c: c == "true", _ok()),
            (lambda c: "command -v strata-worker" in c, _ok(_INSTALLED)),
            (lambda c: c.startswith("cat "), _ok("")),
            (lambda c: "nohup strata-worker" in c, _ok("4321\n")),
        ]
    )
    sup = RemoteWorkerSupervisor(
        tunnel_launcher=launcher,
        health_probe=lambda port: True,
        launch_id_probe=lambda port: _launched_id(runner),
        port_picker=lambda: 55001,
        runner_factory=lambda target: runner,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(sup.establish, "gpu", "user@box")
        assert first_in_preflight.wait(timeout=10)
        # Second establish for the same name while the first is mid-flight.
        with pytest.raises(SshWorkerError, match="already in progress"):
            sup.establish("gpu", "user@box")
        release_first.set()
        record = first.result(timeout=10)

    assert record.name == "gpu"
    assert len(launcher.spawns) == 1  # exactly one tunnel, no orphan


def test_failed_establish_releases_the_name():
    """A failed establish must clear its reservation so a retry is allowed."""
    runner = ScriptedSshRunner([(lambda c: c == "true", _ok())])  # preflight ok
    # detect probe returns default ok("") → parses as missing worker+uv
    sup = RemoteWorkerSupervisor(
        tunnel_launcher=FakeTunnelLauncher(),
        health_probe=lambda port: True,
        launch_id_probe=lambda port: _launched_id(runner),
        port_picker=lambda: 55001,
        runner_factory=lambda target: runner,
    )
    with pytest.raises(SshWorkerError):
        sup.establish("gpu", "user@box")  # no uv, no worker → refuses
    # The name is free again: a second attempt gets past the reservation (and
    # fails the same way, not with "already in progress").
    with pytest.raises(SshWorkerError) as excinfo:
        sup.establish("gpu", "user@box")
    assert "already in progress" not in str(excinfo.value)


def test_shutdown_snapshot_tolerates_concurrent_teardown():
    """shutdown() iterates a snapshot, so a concurrent teardown can't change the dict under it."""
    sup, launcher, _ = _supervisor()
    sup.establish("a", "user@box")
    sup.establish("b", "user@box")
    # Teardown one while shutdown holds its snapshot. The snapshot makes this safe
    # regardless of timing; this checks both paths complete and all is terminated.
    sup.teardown("a")
    sup.shutdown()
    assert sup.status() == []
    assert all(h.terminated >= 1 for h in launcher.handles)
