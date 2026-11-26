"""Tests for the RemoteWorkerSupervisor of the SSH remote-worker path.

Tunnel launcher, health probe, port picker and SSH runner are faked, so these assert the
orchestration (provision, tunnel, health-check, record, reconcile, teardown, shutdown).
"""

from __future__ import annotations

import sys

import pytest

from strata.notebook.remote_worker_supervisor import (
    RemoteWorkerSupervisor,
    SshWorkerError,
)
from strata.notebook.worker_secrets import clear_runtime_worker_token
from tests.notebook.test_ssh_worker import ScriptedSshRunner, _ok


@pytest.fixture(autouse=True)
def _no_probe_spacing(monkeypatch):
    from strata.notebook import remote_worker_supervisor

    monkeypatch.setattr(remote_worker_supervisor, "_LIVE_PROBE_SPACING", 0.0)


def _until(condition) -> None:
    import time

    deadline = time.monotonic() + 30
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


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
        if "nohup " in command:
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
            (lambda c: "uv tool dir --bin" in c, _ok("/root/.local/bin\n")),  # its bin dir
            (lambda c: c.startswith("cat "), _ok("")),  # is_running: no pidfile
            (lambda c: "nohup " in c, _ok("4321\n")),  # launch → pid
            (lambda c: "kill" in c and "kill -0" not in c, _ok("stopped\n")),  # stop
        ]
    )


class FakeTunnelHandle:
    def __init__(self) -> None:
        self.alive = True
        self.terminated = 0
        self.stderr = ""

    def is_alive(self) -> bool:
        return self.alive

    def error(self) -> str:
        return self.stderr

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
        if "nohup " in command:
            return stdin_data.splitlines()[0] if stdin_data else None
    return None


def _supervisor(runner=None, *, healthy=True, launcher=None, launch_id_probe=None):
    import itertools

    launcher = launcher or FakeTunnelLauncher()
    the_runner = runner or _runner()
    ports = itertools.count(55001)
    sup = RemoteWorkerSupervisor(
        tunnel_launcher=launcher,
        health_probe=lambda port: healthy() if callable(healthy) else healthy,
        launch_id_probe=launch_id_probe or (lambda port: _launched_id(the_runner)),
        port_picker=lambda: next(ports),
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
    # The bare name is not on a non-interactive SSH PATH; the path uv reports is.
    launch = next(c for c in runner.calls if "nohup " in c)
    assert "nohup /root/.local/bin/strata-worker --host" in launch


def _server_version(monkeypatch, version: str) -> None:
    monkeypatch.setattr("strata.notebook.ssh_worker.version", lambda name: version)


def _install_spec(runner) -> str:
    return next(c for c in runner.calls if "uv tool install" in c).split("install ")[1]


@pytest.mark.parametrize(
    ("pin", "spec"),
    [
        (None, "'strata-notebook[notebook]==0.12.0'"),
        ("0.11.0", "'strata-notebook[notebook]==0.11.0'"),
        ("none", "'strata-notebook[notebook]'"),
    ],
    ids=["the-servers", "given", "none"],
)
def test_a_fresh_box_installs_the_pinned_version(monkeypatch, pin, spec):
    _server_version(monkeypatch, "0.12.0")
    runner = _runner(detect=_MISSING)
    sup, _, _ = _supervisor(runner=runner)

    rec = sup.establish("gpu", "user@box", pin=pin)

    assert _install_spec(runner) == spec
    assert rec.warnings == ()


def test_a_dev_server_installs_unpinned_and_says_so(monkeypatch):
    _server_version(monkeypatch, "0.12.1.dev3+g1234abc")
    runner = _runner(detect=_MISSING)
    sup, _, _ = _supervisor(runner=runner)

    rec = sup.establish("gpu", "user@box")

    assert _install_spec(runner) == "'strata-notebook[notebook]'"
    (warning,) = rec.warnings
    assert "0.12.1.dev3+g1234abc" in warning and "unpinned" in warning


@pytest.mark.parametrize(("found", "warns"), [("0.6.0", True), ("0.12.0", False)])
def test_a_box_whose_worker_is_another_version_warns(monkeypatch, found, warns):
    _server_version(monkeypatch, "0.12.0")
    runner = _runner(detect=_INSTALLED.replace("version=0.6.0", f"version={found}"))
    sup, _, _ = _supervisor(runner=runner)

    rec = sup.establish("gpu", "user@box")

    assert not any("uv tool install" in c for c in runner.calls)
    if warns:
        (warning,) = rec.warnings
        assert "0.6.0" in warning and "--force 'strata-notebook[notebook]==0.12.0'" in warning
    else:
        assert rec.warnings == ()


def test_establish_launches_the_detected_worker_path():
    runner = _runner()
    sup, _, _ = _supervisor(runner=runner)
    sup.establish("gpu", "user@box")
    launch = next(c for c in runner.calls if "nohup " in c)
    assert "nohup /usr/bin/strata-worker --host" in launch
    assert not any("uv tool install" in c for c in runner.calls)


def test_establish_health_failure_quotes_the_remote_log():
    runner = _runner()
    runner.rules.insert(0, (lambda c: c.startswith("tail -n"), _ok("nohup: failed to run\n")))
    sup, _, _ = _supervisor(runner=runner, healthy=False)
    with pytest.raises(SshWorkerError, match="(?s)didn't respond.*ends:\nnohup: failed to run"):
        sup.establish("gpu", "user@box", health_timeout=0)


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
        if "nohup " in command
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
    assert sup.get("gpu").local_port == launcher.spawns[1][1]


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
            ("nohup ", _ok("4321\n")),
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
        if "nohup " in command
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


def test_reconcile_republishes_the_token_once_the_reopened_tunnel_reaches_our_worker():
    from strata.notebook.worker_secrets import get_runtime_worker_token

    sup, launcher, _ = _supervisor()
    sup.establish("gpu", "user@box", token="tok")
    launcher.handles[0].alive = False

    sup.reconcile()

    assert get_runtime_worker_token("gpu") == "tok"
    assert sup.get("gpu").healthy is True


def test_reconcile_closes_a_reopened_tunnel_that_reaches_another_process(monkeypatch):
    """The worker died and something else took its port: the token must not go there."""
    from strata.notebook import remote_worker_supervisor
    from strata.notebook.worker_secrets import get_runtime_worker_token

    monkeypatch.setattr(remote_worker_supervisor, "_FOREIGN_SETTLE", 0.5)
    answering = {"id": None}
    runner = _runner()
    sup, launcher, _ = _supervisor(
        runner, launch_id_probe=lambda port: answering["id"] or _launched_id(runner)
    )
    sup.establish("gpu", "user@box", token="tok")
    launcher.handles[0].alive = False
    answering["id"] = "someone-else"

    for _ in range(3):  # each reopen answered by it: it is the box's port, not ours
        sup.reconcile()
        assert launcher.handles[-1].terminated == 1
        assert get_runtime_worker_token("gpu") is None
        assert sup.get("gpu").healthy is False
    assert sup._tunnels["gpu"].parked is True

    sup.reconcile()  # parked: not reopened only to reach it again
    assert len(launcher.spawns) == 4


@pytest.mark.parametrize("answer", ["", "someone-else"])
def test_a_live_tunnel_reaching_another_process_is_closed_and_parked(answer):
    """``""``: it answers /health with no launch id (a worker started by hand, a web page)."""
    from strata.notebook.worker_secrets import get_runtime_worker_token

    taken = {"on": False}
    runner = _runner()
    sup, launcher, _ = _supervisor(
        runner, launch_id_probe=lambda port: answer if taken["on"] else _launched_id(runner)
    )
    sup.establish("gpu", "user@box", token="tok")
    taken["on"] = True

    sup.reconcile()
    sup.reconcile()

    assert launcher.handles[0].terminated == 1 and len(launcher.spawns) == 1
    assert get_runtime_worker_token("gpu") is None
    assert sup.status()[0].healthy is False


def test_status_reports_the_last_check_without_probing():
    """A listing neither waits on a slow box nor changes the token: reconcile does."""
    runner = _runner()
    taken = {"on": False}
    probes: list[int] = []

    def probe(port: int) -> str:
        probes.append(port)
        return "" if taken["on"] else _launched_id(runner)

    sup, launcher, _ = _supervisor(runner, launch_id_probe=probe)
    sup.establish("gpu", "user@box")
    taken["on"] = True
    before = len(probes)

    assert sup.status()[0].healthy is True and len(probes) == before
    sup.reconcile()
    assert sup.status()[0].healthy is False


def test_status_reports_an_exited_forward_unhealthy():
    runner = _runner()
    sup, launcher, _ = _supervisor(runner, launch_id_probe=lambda port: _launched_id(runner))
    sup.establish("gpu", "user@box")
    launcher.handles[0].alive = False

    assert sup.status()[0].healthy is False


def test_a_reopen_answered_through_this_machines_port_is_tried_again(monkeypatch):
    """Something on the freed local port answers while ssh connects; ssh then exits, unable
    to bind it. Not parked: the next pass reopens it."""
    from strata.notebook import remote_worker_supervisor

    monkeypatch.setattr(remote_worker_supervisor, "_FOREIGN_SETTLE", 0.5)
    runner = _runner()
    answering = {"id": None}
    sup, launcher, _ = _supervisor(
        runner, launch_id_probe=lambda port: answering["id"] or _launched_id(runner)
    )
    sup.establish("gpu", "user@box", token="tok")
    launcher.handles[0].alive = False
    answering["id"] = "a-local-dev-server"

    answers = {"n": 0}

    def ssh_exits_unable_to_bind(port: int) -> str:
        # Answered while ssh still connects; then it fails to bind the port, and exits.
        answers["n"] += 1
        if answers["n"] > 1:
            launcher.handles[-1].alive = False
        return "a-local-dev-server"

    sup.launch_id_probe = ssh_exits_unable_to_bind
    sup.reconcile()
    sup.launch_id_probe = lambda port: _launched_id(runner)
    sup.reconcile()

    assert len(launcher.spawns) == 3
    assert sup.get("gpu").healthy is True


def test_a_forward_closed_here_for_a_retry_is_no_ssh_exit(monkeypatch, caplog):
    from strata.notebook import remote_worker_supervisor

    monkeypatch.setattr(remote_worker_supervisor, "_FOREIGN_SETTLE", 0.1)
    runner = _runner()
    sup, launcher, _ = _supervisor(runner)
    sup.establish("gpu", "user@box", token="tok")
    launcher.handles[0].alive = False
    launcher.handles[0].stderr = "client_loop: send disconnect: Broken pipe"
    sup.launch_id_probe = lambda port: "a-local-dev-server"
    sup.reconcile()  # reopened, answered through this machine's port: closed for a retry
    assert launcher.handles[1].terminated

    sup.launch_id_probe = lambda port: _launched_id(runner)
    sup.reconcile()

    assert sup.get("gpu").healthy is True
    exits = [r.getMessage() for r in caplog.records if "ssh exited" in r.getMessage()]
    assert len(exits) == 1 and exits[0].endswith("Broken pipe")


def test_shutdown_ends_a_reconcile_waiting_on_a_reopened_tunnel():
    import threading

    runner = _runner()
    answered = threading.Event()
    sup, launcher, _ = _supervisor(
        runner, launch_id_probe=lambda port: None if answered.is_set() else _launched_id(runner)
    )
    sup.establish("gpu", "user@box")
    answered.set()  # from now on nothing answers: the wait would last 30 s
    launcher.handles[0].alive = False
    worker = threading.Thread(target=sup.reconcile)
    worker.start()
    _until(lambda: launcher.spawns[1:])

    sup.shutdown()
    worker.join(timeout=30)

    assert not worker.is_alive()


def test_reconcile_never_closes_a_live_tunnel_whose_health_is_slow():
    """A busy box may miss a probe, with a cell streaming through the forward."""
    from strata.notebook.worker_secrets import get_runtime_worker_token

    slow = {"on": False}
    runner = _runner()
    sup, launcher, _ = _supervisor(
        runner, launch_id_probe=lambda port: None if slow["on"] else _launched_id(runner)
    )
    sup.establish("gpu", "user@box", token="tok")
    slow["on"] = True

    sup.reconcile()

    assert launcher.handles[0].terminated == 0 and len(launcher.spawns) == 1
    assert sup.get("gpu").healthy is False
    # Unconfirmed, so withheld: whatever holds the port may not be ours.
    assert get_runtime_worker_token("gpu") is None

    slow["on"] = False
    sup.reconcile()
    assert get_runtime_worker_token("gpu") == "tok" and sup.get("gpu").healthy is True


def test_a_stopped_worker_gets_one_respawn_not_one_a_cycle(monkeypatch):
    from strata.notebook import remote_worker_supervisor

    monkeypatch.setattr(remote_worker_supervisor, "_RESPAWN_HEALTH_TIMEOUT", 0.0)
    gone = {"on": False}
    runner = _runner()
    sup, launcher, _ = _supervisor(
        runner, launch_id_probe=lambda port: None if gone["on"] else _launched_id(runner)
    )
    sup.establish("gpu", "user@box")
    launcher.handles[0].alive = False
    gone["on"] = True

    for _ in range(3):
        sup.reconcile()

    assert len(launcher.spawns) == 2  # establish, then the one reopen
    assert sup.get("gpu").healthy is False


def test_reconcile_leaves_a_name_an_establish_holds():
    sup, launcher, _ = _supervisor()
    sup.establish("gpu", "user@box")
    launcher.handles[0].alive = False
    sup._pending.add("gpu")

    assert sup.reconcile() == []
    assert len(launcher.spawns) == 1


def test_an_establish_during_a_respawn_takes_over():
    """``add-ssh`` while the tunnel reconnects (what the docs advise) is not refused: the
    reconnect stops, and the establish replaces it."""
    import threading

    from strata.notebook.worker_secrets import get_runtime_worker_token

    sup, launcher, _ = _supervisor()
    sup.establish("gpu", "user@box", local_port=55001)
    launcher.handles[0].alive = False
    errors: list[Exception] = []
    spawn = launcher.spawn

    def establish() -> None:
        try:
            sup.establish("gpu", "user@box", local_port=55001, token="new")
        except Exception as exc:  # noqa: BLE001 - reported by the assertion below
            errors.append(exc)

    other = threading.Thread(target=establish)

    def spawn_while_establishing(*args, **kwargs):
        launcher.spawn = spawn  # once: the establish spawns through the launcher too
        other.start()
        _until(lambda: "gpu" in sup._pending)
        return spawn(*args, **kwargs)

    launcher.spawn = spawn_while_establishing
    sup.reconcile()
    other.join(timeout=30)

    assert not other.is_alive() and errors == []
    assert sup.get("gpu").healthy is True and get_runtime_worker_token("gpu") == "new"
    assert launcher.handles[1].terminated == 1  # the reconnect's forward, replaced


def test_one_tunnel_that_cannot_reopen_does_not_stop_the_others():
    class Flaky(FakeTunnelLauncher):
        def spawn(self, ssh_target, *, local_port, remote_port):
            if ssh_target == "user@a" and len(self.spawns) >= 2:
                self.spawns.append((ssh_target, local_port, remote_port))
                raise SshWorkerError("ssh not found")
            return super().spawn(ssh_target, local_port=local_port, remote_port=remote_port)

    launcher = Flaky()
    sup, _, _ = _supervisor(launcher=launcher)
    sup.establish("a", "user@a", local_port=55001)
    sup.establish("b", "user@b", local_port=55002)
    for handle in launcher.handles:
        handle.alive = False

    sup.reconcile()

    assert launcher.spawns[-1][0] == "user@b"
    assert sup.get("a").healthy is False and sup.get("b").healthy is True


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
            (lambda c: "nohup " in c, _ok("4321\n")),
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


async def test_the_server_reopens_a_dropped_tunnel(monkeypatch):
    """``reconcile`` runs on the server's loop, so a dropped ``ssh -L`` comes back by itself."""
    import asyncio
    import contextlib

    import strata.server as server_module
    from strata.notebook import routes

    sup, launcher, _ = _supervisor()
    sup.establish("gpu", "user@box")
    launcher.handles[0].alive = False  # the laptop slept; the forward died
    respawned = asyncio.Event()
    loop = asyncio.get_running_loop()
    reconcile = routes.reconcile_worker_supervisor

    def reconcile_and_tell() -> None:
        # Each pass finishes, token and all, before the test ends and clears tokens; none
        # starts once the reopen is seen.
        if respawned.is_set():
            return
        reconcile()
        if len(launcher.spawns) > 1:
            loop.call_soon_threadsafe(respawned.set)

    monkeypatch.setattr(routes, "reconcile_worker_supervisor", reconcile_and_tell)
    monkeypatch.setattr(routes, "_worker_supervisor", sup)
    monkeypatch.setattr(server_module, "_SSH_TUNNEL_RECONCILE_SECONDS", 0.0)
    task = asyncio.create_task(server_module._ssh_tunnel_reconcile_loop())
    try:
        await asyncio.wait_for(respawned.wait(), timeout=30)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert launcher.spawns[1] == ("user@box", 55001, 9000)


def test_a_server_with_no_ssh_worker_builds_no_supervisor(monkeypatch):
    from strata.notebook import routes

    monkeypatch.setattr(routes, "_worker_supervisor", None)

    routes.reconcile_worker_supervisor()

    assert routes._worker_supervisor is None


@pytest.mark.parametrize(
    ("status", "body", "seen"),
    [
        (200, {"launch_id": "ours"}, "ours"),
        (200, {"status": "ok"}, ""),  # answers, as no worker this server launched
        (200, {"launch_id": None}, ""),
        (500, {"launch_id": "ours"}, None),  # a failing /health: no verdict on whose
        (200, "plain text", None),
        (200, ["a list"], None),
    ],
)
def test_the_launch_id_probe_judges_only_a_healthy_answer(monkeypatch, status, body, seen):
    import httpx

    from strata.notebook.remote_worker_supervisor import _default_launch_id_probe

    def get(url, timeout):
        content = body if isinstance(body, str) else None
        if content is not None:
            return httpx.Response(status, text=content)
        return httpx.Response(status, json=body)

    monkeypatch.setattr(httpx, "get", get)

    assert _default_launch_id_probe(55001) == seen


def test_a_reopened_tunnel_that_reaches_another_of_ours_is_not_parked():
    """Its local port went to another tunnel meanwhile: a collision, not a stranger."""
    runner = _runner()
    other = {"id": None}
    sup, launcher, _ = _supervisor(
        runner, launch_id_probe=lambda port: other["id"] or _launched_id(runner)
    )
    sup.establish("a", "user@a", local_port=55001)
    sup.establish("b", "user@b", local_port=55002)
    b_launch = sup._tunnels["b"].launch_id
    launcher.handles[0].alive = False
    other["id"] = b_launch

    sup.reconcile()

    assert sup._tunnels["a"].parked is False
    assert sup.get("a").healthy is False


def test_a_probe_from_before_a_reconnect_publishes_nothing():
    """A status() probe of the old forward lands while the forward is being reopened."""
    from strata.notebook.worker_secrets import get_runtime_worker_token

    sup, _, _ = _supervisor()
    sup.establish("gpu", "user@box", token="tok")
    active = sup._tunnels["gpu"]
    clear_runtime_worker_token("gpu")  # as the reopen does first
    sup._reconnecting.add("gpu")

    sup._check("gpu", active, active.handle, active.launch_id)

    assert get_runtime_worker_token("gpu") is None


def test_an_establish_that_ends_after_shutdown_leaves_no_tunnel():
    sup, launcher, _ = _supervisor()
    spawn = launcher.spawn

    def spawn_then_shut_down(*args, **kwargs):
        handle = spawn(*args, **kwargs)
        sup.shutdown()
        return handle

    launcher.spawn = spawn_then_shut_down

    with pytest.raises(SshWorkerError, match="shutting down"):
        sup.establish("gpu", "user@box")

    assert sup.get("gpu") is None and launcher.handles[0].terminated == 1


def test_a_probe_that_lands_after_parking_revives_nothing():
    from strata.notebook.worker_secrets import get_runtime_worker_token

    sup, _, _ = _supervisor()
    sup.establish("gpu", "user@box", token="tok")
    active = sup._tunnels["gpu"]
    active.parked = True
    clear_runtime_worker_token("gpu")

    sup._check("gpu", active, active.handle, active.launch_id)

    assert get_runtime_worker_token("gpu") is None


def test_two_reconciles_reopen_a_dead_tunnel_once():
    """The second finds the first reopening it, so no forward is orphaned."""
    import threading

    runner = _runner()
    release = threading.Event()
    launcher = FakeTunnelLauncher()
    sup, _, _ = _supervisor(runner, launcher=launcher)
    sup.establish("gpu", "user@box")
    launcher.handles[0].alive = False
    spawn = launcher.spawn

    def slow_spawn(*args, **kwargs):
        release.wait(timeout=10)
        return spawn(*args, **kwargs)

    launcher.spawn = slow_spawn
    first = threading.Thread(target=sup.reconcile)
    first.start()
    _until(lambda: "gpu" in sup._reconnecting)

    sup.reconcile()
    release.set()
    first.join(timeout=30)

    assert len(launcher.spawns) == 2


class _StuckProc:
    """A wrapper that ignores SIGTERM, or one that already exited."""

    pid = 4242

    def __init__(self, *, exited: bool = False) -> None:
        self.exited = exited
        self.killed = False
        self.waits = 0

    def poll(self):
        return 0 if self.exited else None

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None):
        import subprocess

        self.waits += 1
        if timeout is not None and not self.exited:
            raise subprocess.TimeoutExpired("lifeline", timeout)
        return 0


def test_a_stuck_tunnel_is_killed_where_there_is_no_process_group(monkeypatch):
    import os

    from strata.notebook.remote_worker_supervisor import _PopenTunnelHandle

    monkeypatch.delattr(os, "killpg", raising=False)
    proc = _StuckProc()

    _PopenTunnelHandle(proc).terminate()

    assert proc.killed


def test_a_tunnel_whose_wrapper_died_has_its_group_killed(monkeypatch):
    """Its ssh may still hold the local port the reopen needs."""
    import os

    from strata.notebook.remote_worker_supervisor import _PopenTunnelHandle

    groups: list[int] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: groups.append(pgid))

    _PopenTunnelHandle(_StuckProc(exited=True)).terminate()

    assert groups == [4242]


def test_a_new_tunnel_never_takes_the_port_of_one_whose_ssh_exited():
    """It is reopened on that port, which must still be free."""
    sup, launcher, _ = _supervisor()
    sup.establish("a", "user@a")
    launcher.handles[0].alive = False
    picks = iter([55001, 55001, 55007])
    sup.port_picker = lambda: next(picks)

    assert sup.establish("b", "user@b").local_port == 55007
    with pytest.raises(SshWorkerError, match="another worker's tunnel"):
        sup.establish("c", "user@c", local_port=55001)


def test_a_tunnel_not_reaching_its_worker_is_sent_nothing():
    """What holds its local port now would be sent the cell."""
    from types import SimpleNamespace

    from strata.notebook.cell_paths.dispatch import ensure_worker_reachable

    runner = _runner()
    gone = {"on": False}
    sup, _, _ = _supervisor(
        runner, launch_id_probe=lambda port: None if gone["on"] else _launched_id(runner)
    )
    sup.establish("gpu", "user@box", token="tok")
    spec = SimpleNamespace(name="gpu")
    ensure_worker_reachable(spec)

    gone["on"] = True
    sup.reconcile()
    with pytest.raises(RuntimeError, match="does not reach the worker"):
        ensure_worker_reachable(spec)

    gone["on"] = False
    sup.reconcile()
    ensure_worker_reachable(spec)
    sup.teardown("gpu")
    ensure_worker_reachable(SimpleNamespace(name="gpu"))


def test_a_tunnel_is_terminated_once(monkeypatch):
    """A second terminate (a parked tunnel torn down) must not signal a reused pid."""
    import os

    from strata.notebook.remote_worker_supervisor import _PopenTunnelHandle

    groups: list[int] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: groups.append(pgid))
    handle = _PopenTunnelHandle(_StuckProc(exited=True))

    handle.terminate()
    handle.terminate()

    assert groups == [4242]


def test_an_ssh_that_exits_fails_establish_with_its_reason():
    """Not a wait for /health through a forward that is gone."""
    probes: list[int] = []

    class Exiting(FakeTunnelLauncher):
        def spawn(self, ssh_target, *, local_port, remote_port):
            handle = super().spawn(ssh_target, local_port=local_port, remote_port=remote_port)
            handle.alive = False
            handle.stderr = "bind [127.0.0.1]:55001: Address already in use"
            return handle

    def health(port: int) -> bool:
        probes.append(port)
        if len(probes) > 3:
            raise AssertionError("still waiting on /health through a forward that is gone")
        return False

    sup, launcher, _ = _supervisor(launcher=Exiting())
    sup.health_probe = health

    with pytest.raises(SshWorkerError, match="ssh exited") as raised:
        sup.establish("gpu", "user@box", health_timeout=600)

    assert "Address already in use" in str(raised.value)
    assert len(probes) <= 1
    assert launcher.handles[0].terminated == 1
    assert sup.get("gpu") is None


def test_a_reopen_logs_why_ssh_exited_once_per_reason_in_an_outage(caplog):
    """A box that stays unreachable is reopened every pass, and logged once a reason."""
    down = {"why": ""}

    class Box(FakeTunnelLauncher):
        def spawn(self, ssh_target, *, local_port, remote_port):
            handle = super().spawn(ssh_target, local_port=local_port, remote_port=remote_port)
            if down["why"]:
                handle.alive = False
                handle.stderr = f"ssh: connect to host box port 22: {down['why']}"
            return handle

    sup, launcher, _ = _supervisor(launcher=Box())
    sup.establish("gpu", "user@box")

    def outage(*reasons: str) -> None:
        for reason in reasons:
            down["why"] = reason
            launcher.handles[-1].alive = False
            launcher.handles[-1].stderr = f"ssh: connect to host box port 22: {reason}"
            sup.reconcile()

    outage("Connection refused", "Connection refused", "Permission denied")
    down["why"] = ""
    sup.reconcile()  # reopened, and our worker answers
    assert sup.get("gpu").healthy
    outage("Permission denied")

    exits = [r.getMessage() for r in caplog.records if "ssh exited" in r.getMessage()]
    assert [e.rsplit(": ", 1)[1] for e in exits] == [
        "Connection refused",
        "Permission denied",
        "Permission denied",
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="a shell script stands in for ssh")
def test_the_tunnel_keeps_what_ssh_wrote(tmp_path, monkeypatch):
    import os
    import time

    from strata.notebook.remote_worker_supervisor import SubprocessTunnelLauncher

    fake = tmp_path / "ssh"
    fake.write_text(
        "#!/bin/sh\n"
        "i=0; while [ $i -lt 200 ]; do echo 'debug1: noise' >&2; i=$((i+1)); done\n"
        "echo 'bind [127.0.0.1]:1: Address already in use' >&2\nexit 255\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    handle = SubprocessTunnelLauncher().spawn("box", local_port=1, remote_port=2)
    deadline = time.monotonic() + 30
    while handle.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    seen = handle.error()
    handle.terminate()

    assert seen.endswith("Address already in use")
    assert len(seen) < 2000
    assert handle.error() == seen


class _Running:
    pid = 4242

    def poll(self):
        return None


def _stderr_file(tmp_path):
    """ssh's appending end and the handle's own, as the launcher opens them."""
    import os

    path = tmp_path / "stderr"
    path.touch()
    writer = os.open(path, os.O_WRONLY | os.O_APPEND)
    return writer, open(path, "r+b")  # noqa: SIM115 - the handle closes it


def test_a_live_forwards_stderr_is_cut_back(tmp_path):
    """A line per refused connection must not grow it for as long as the forward is up."""
    import os

    from strata.notebook import remote_worker_supervisor
    from strata.notebook.remote_worker_supervisor import _PopenTunnelHandle

    writer, reader = _stderr_file(tmp_path)
    handle = _PopenTunnelHandle(_Running(), reader)
    try:
        line = b"channel 3: open failed: connect failed: Connection refused\n"
        while os.fstat(writer).st_size <= remote_worker_supervisor._STDERR_KEEP:
            os.write(writer, line)
        assert handle.is_alive()
        os.write(writer, b"bind [127.0.0.1]:1: Address already in use\n")

        assert handle.error() == "bind [127.0.0.1]:1: Address already in use"
        assert os.fstat(writer).st_size < 100
    finally:
        os.close(writer)
        reader.close()


def test_a_reason_longer_than_the_tail_is_kept(tmp_path):
    import os

    from strata.notebook import remote_worker_supervisor
    from strata.notebook.remote_worker_supervisor import _PopenTunnelHandle

    writer, reader = _stderr_file(tmp_path)
    handle = _PopenTunnelHandle(_Running(), reader)
    try:
        os.write(writer, b"debug1: noise\n")
        os.write(writer, b"Host key mismatch " + b"x" * remote_worker_supervisor._STDERR_TAIL)

        assert handle.error().endswith("x" * 100)
    finally:
        os.close(writer)
        reader.close()
