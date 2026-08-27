"""Tests for the SSH worker provisioning core (P1 of the SSH remote-worker path).

A scripted fake :class:`SshRunner` stands in for a real host, so these assert the
exact command contracts and the provisioning decisions (install-if-missing,
idempotent adoption) with no network. No real SSH runs here; a localhost-sshd
integration test is a separate, opt-in phase.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from strata.notebook.ssh_worker import (
    CommandResult,
    RemoteEnvInfo,
    RemoteWorker,
    RunningWorker,
    SshTarget,
    SshWorkerError,
    SubprocessSshRunner,
    _last_int,
    _parse_kv,
)


class ScriptedSshRunner:
    """A fake ``SshRunner``: first matching rule wins; records every command."""

    def __init__(self, rules: list[tuple[str | Callable[[str], bool], CommandResult]]) -> None:
        self.rules = rules
        self.calls: list[str] = []
        self.stdin_writes: list[str | None] = []

    def run(
        self, command: str, *, timeout: float | None = None, stdin_data: str | None = None
    ) -> CommandResult:
        self.calls.append(command)
        self.stdin_writes.append(stdin_data)
        for match, result in self.rules:
            hit = match(command) if callable(match) else match in command
            if hit:
                return result
        return CommandResult(0, "", "")


def _ok(stdout: str = "") -> CommandResult:
    return CommandResult(0, stdout, "")


def _fail(stderr: str = "boom", code: int = 1) -> CommandResult:
    return CommandResult(code, "", stderr)


# ---------------------------------------------------------------------------
# SshTarget
# ---------------------------------------------------------------------------


def test_ssh_target_argv_is_hardened():
    argv = SshTarget("user@gpu-box").ssh_argv()
    assert argv[0] == "ssh"
    assert argv[-1] == "user@gpu-box"
    assert "BatchMode=yes" in argv  # key-only, never prompt for a password
    # We never disable host-key checking — first-connect trust is the user's.
    assert "StrictHostKeyChecking=no" not in argv


@pytest.mark.parametrize("bad", ["", "   ", "-oProxyCommand=x", "has space"])
def test_ssh_target_rejects_bad(bad):
    with pytest.raises(SshWorkerError, match="invalid ssh target"):
        SshTarget(bad)


def test_ssh_target_strips_and_builds_runner():
    target = SshTarget("  host  ")
    assert target.target == "host"
    assert isinstance(target.runner(), SubprocessSshRunner)


# ---------------------------------------------------------------------------
# preflight / detect
# ---------------------------------------------------------------------------


def test_preflight_ok():
    runner = ScriptedSshRunner([("true", _ok())])
    RemoteWorker("gpu", runner).preflight()
    assert runner.calls == ["true"]


def test_preflight_failure_is_actionable():
    runner = ScriptedSshRunner([("true", _fail("Permission denied (publickey)", code=255))])
    with pytest.raises(SshWorkerError, match="key-based SSH isn't working"):
        RemoteWorker("gpu", runner).preflight()


def test_detect_parses_probe_output():
    probe = "worker=/usr/bin/strata-worker\nuv=/usr/bin/uv\nplatform=Linux x86_64\nversion=0.6.0\n"
    runner = ScriptedSshRunner([("command -v strata-worker", _ok(probe))])
    info = RemoteWorker("gpu", runner).detect()
    assert info.has_worker is True
    assert info.has_uv is True
    assert info.worker_path == "/usr/bin/strata-worker"
    assert info.platform == "Linux x86_64"
    assert info.worker_version == "0.6.0"


def test_detect_reports_missing_tools():
    probe = "worker=\nuv=\nplatform=Darwin arm64\nversion=\n"
    runner = ScriptedSshRunner([("printf", _ok(probe))])
    info = RemoteWorker("gpu", runner).detect()
    assert info.has_worker is False
    assert info.has_uv is False
    assert info.worker_version is None
    assert info.platform == "Darwin arm64"


def test_detect_failure_raises():
    runner = ScriptedSshRunner([("printf", _fail("bad host"))])
    with pytest.raises(SshWorkerError, match="remote detection failed"):
        RemoteWorker("gpu", runner).detect()


# ---------------------------------------------------------------------------
# ensure_installed
# ---------------------------------------------------------------------------


def test_ensure_installed_noop_when_present():
    runner = ScriptedSshRunner([])
    info = RemoteEnvInfo(has_worker=True, has_uv=True, worker_version="0.6.0", platform="Linux")
    RemoteWorker("gpu", runner).ensure_installed(info)
    assert runner.calls == []  # already installed → no remote command


def test_ensure_installed_uses_uv_with_extras_and_pin():
    runner = ScriptedSshRunner([("uv tool install", _ok())])
    info = RemoteEnvInfo(has_worker=False, has_uv=True, worker_version=None, platform="Linux")
    RemoteWorker("gpu", runner).ensure_installed(info, extras="notebook", pin="0.6.0")
    assert len(runner.calls) == 1
    assert "uv tool install" in runner.calls[0]
    assert "strata-notebook[notebook]==0.6.0" in runner.calls[0]


def test_ensure_installed_refuses_without_uv():
    runner = ScriptedSshRunner([])
    info = RemoteEnvInfo(has_worker=False, has_uv=False, worker_version=None, platform="Linux")
    with pytest.raises(SshWorkerError, match="neither `strata-worker` nor `uv`"):
        RemoteWorker("gpu", runner).ensure_installed(info)
    assert runner.calls == []  # no silent uv bootstrap


def test_ensure_installed_surfaces_install_failure():
    runner = ScriptedSshRunner([("uv tool install", _fail("resolve error"))])
    info = RemoteEnvInfo(has_worker=False, has_uv=True, worker_version=None, platform="Linux")
    with pytest.raises(SshWorkerError, match="uv tool install.*failed"):
        RemoteWorker("gpu", runner).ensure_installed(info)


# ---------------------------------------------------------------------------
# launch / is_running / stop
# ---------------------------------------------------------------------------


def test_launch_starts_and_parses_pid():
    # cat pidfile → empty (nothing recorded), so adoption falls through to launch.
    runner = ScriptedSshRunner(
        [
            ("cat ", _ok("")),  # is_running: no pidfile
            ("nohup strata-worker", _ok("4321\n")),  # launch echoes the pid
        ]
    )
    worker = RemoteWorker("gpu", runner).launch(port=9000, token="secret")
    assert worker == RunningWorker(pid=4321, port=9000)
    launch_cmd = next(c for c in runner.calls if "nohup strata-worker" in c)
    assert "--host 127.0.0.1 --port 9000" in launch_cmd
    # The token travels over stdin, never in the command (a command-string
    # token would sit in the local ssh argv, visible in ps, and be echoed
    # in timeout error messages that reach HTTP responses).
    assert "secret" not in launch_cmd
    assert "read -r STRATA_WORKER_TOKEN" in launch_cmd
    assert "secret\n" in runner.stdin_writes


def test_launch_never_puts_token_in_any_command():
    runner = ScriptedSshRunner(
        [
            ("cat ", _ok("")),
            ("nohup strata-worker", _ok("4321\n")),
        ]
    )
    RemoteWorker("gpu", runner).launch(port=9000, token="hunter2-bearer")
    assert all("hunter2-bearer" not in c for c in runner.calls)


def test_launch_without_token_skips_the_stdin_read():
    runner = ScriptedSshRunner(
        [
            ("cat ", _ok("")),
            ("nohup strata-worker", _ok("4321\n")),
        ]
    )
    RemoteWorker("gpu", runner).launch(port=9000, token=None)
    launch_cmd = next(c for c in runner.calls if "nohup strata-worker" in c)
    assert "read -r" not in launch_cmd
    assert runner.stdin_writes[-1] is None


def test_launch_rejects_token_with_newline():
    runner = ScriptedSshRunner([("cat ", _ok(""))])
    with pytest.raises(SshWorkerError, match="newline"):
        RemoteWorker("gpu", runner).launch(port=9000, token="bad\ntoken")


def test_timeout_error_never_contains_stdin_data(monkeypatch):
    """A launch timeout's message reaches HTTP 400 bodies and logs — the
    token (fed via stdin) must never appear in it."""
    import subprocess

    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=30)

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = SubprocessSshRunner(["ssh", "host"])
    with pytest.raises(SshWorkerError) as excinfo:
        runner.run("nohup strata-worker", timeout=30, stdin_data="sekrit-token\n")
    assert "sekrit-token" not in str(excinfo.value)


def test_launch_adopts_a_live_worker_on_same_port():
    runner = ScriptedSshRunner(
        [
            ("cat ", _ok('{"pid": 111, "port": 9000}')),
            ("kill -0 111", _ok("up")),
        ]
    )
    worker = RemoteWorker("gpu", runner).launch(port=9000, token=None)
    assert worker == RunningWorker(pid=111, port=9000)
    # Adopted → never issued the nohup launch.
    assert not any("nohup" in c for c in runner.calls)


def test_launch_does_not_adopt_a_different_port():
    runner = ScriptedSshRunner(
        [
            ("cat ", _ok('{"pid": 111, "port": 8000}')),
            ("kill -0 111", _ok("up")),
            ("nohup strata-worker", _ok("222\n")),
        ]
    )
    worker = RemoteWorker("gpu", runner).launch(port=9000, token=None)
    assert worker == RunningWorker(pid=222, port=9000)


def test_is_running_none_without_pidfile():
    runner = ScriptedSshRunner([("cat ", _ok(""))])
    assert RemoteWorker("gpu", runner).is_running() is None


def test_is_running_none_when_pid_dead():
    runner = ScriptedSshRunner(
        [
            ("cat ", _ok('{"pid": 111, "port": 9000}')),
            ("kill -0 111", _ok("")),  # not alive → no "up"
        ]
    )
    assert RemoteWorker("gpu", runner).is_running() is None


def test_stop_reports_stopped():
    runner = ScriptedSshRunner([("kill", _ok("stopped\n"))])
    assert RemoteWorker("gpu", runner).stop() is True


def test_stop_when_nothing_recorded():
    runner = ScriptedSshRunner([("if [ -f", _ok(""))])
    assert RemoteWorker("gpu", runner).stop() is False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_parse_kv_and_last_int():
    assert _parse_kv("a=1\nb=two words\nnoeq\n") == {"a": "1", "b": "two words"}
    assert _last_int("starting\npid 999") == 999
    assert _last_int("no numbers here") is None
