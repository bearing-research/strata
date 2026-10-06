"""SSH provisioning for remote notebook workers.

Brings up a ``strata-worker`` on a box reachable over SSH: preflight, detect,
install if missing, launch and stop. Commands go through :class:`SshRunner`,
so the logic is testable with a fake runner.

There is no silent uv bootstrap (install needs ``uv`` on the box), and
supervision is ``nohup`` plus a JSON pidfile under ``~/.strata``, not systemd.
The worker binds remote-localhost only and is reached over the SSH tunnel.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

# Key-only auth (``BatchMode``: never prompt), bounded connect, keepalive. No
# ``StrictHostKeyChecking=no``: first-connect host-key trust is the user's call.
DEFAULT_CONNECT_TIMEOUT = 10
_SSH_HARDENING: tuple[str, ...] = (
    "-o",
    "BatchMode=yes",
    "-o",
    f"ConnectTimeout={DEFAULT_CONNECT_TIMEOUT}",
    "-o",
    "ServerAliveInterval=15",
)

# Remote state (pidfile, log) lives here so a re-run can adopt a live worker.
_REMOTE_STATE_DIR = "~/.strata"
_INSTALL_TIMEOUT = 600.0


class SshWorkerError(Exception):
    """An SSH provisioning step failed, carrying an actionable message."""


@dataclass(frozen=True)
class CommandResult:
    """The outcome of one remote command."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class RemoteEnvInfo:
    """What a :meth:`RemoteWorker.detect` probe found on the box."""

    has_worker: bool
    has_uv: bool
    worker_version: str | None
    platform: str  # ``uname -sm`` output, e.g. "Linux x86_64"
    worker_path: str | None = None


@dataclass(frozen=True)
class RunningWorker:
    """A live remote worker recorded in the pidfile."""

    pid: int
    port: int


class SshRunner(Protocol):
    """Runs one shell command on the remote host; the seam tests replace with a fake."""

    def run(
        self, command: str, *, timeout: float | None = None, stdin_data: str | None = None
    ) -> CommandResult:
        """Execute *command* on the remote host.

        Secrets (the worker token) go in ``stdin_data``, never in *command*,
        which would show in the local ``ssh`` argv (``ps``) and in error messages.
        """
        ...


class SubprocessSshRunner:
    """:class:`SshRunner` that shells out to the system ``ssh`` for one target."""

    def __init__(self, ssh_argv: Sequence[str]) -> None:
        self._ssh_argv = list(ssh_argv)

    def run(
        self, command: str, *, timeout: float | None = None, stdin_data: str | None = None
    ) -> CommandResult:
        import subprocess

        try:
            proc = subprocess.run(
                [*self._ssh_argv, command],
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin_data,
            )
        except subprocess.TimeoutExpired as exc:
            # Never include ``stdin_data`` (secrets): this message reaches HTTP error bodies
            # and logs.
            raise SshWorkerError(f"ssh command timed out after {timeout}s: {command!r}") from exc
        except OSError as exc:
            raise SshWorkerError(f"could not run ssh: {exc}") from exc
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)


@dataclass(frozen=True)
class SshTarget:
    """A validated SSH destination (``user@host``, ``host``, or a config alias)."""

    target: str

    def __post_init__(self) -> None:
        cleaned = self.target.strip()
        if not cleaned or " " in cleaned or cleaned.startswith("-"):
            raise SshWorkerError(f"invalid ssh target {self.target!r}")
        object.__setattr__(self, "target", cleaned)

    def ssh_argv(self) -> list[str]:
        """The ``ssh`` argv prefix (hardening flags + target), sans the command."""
        return ["ssh", *_SSH_HARDENING, self.target]

    def runner(self) -> SubprocessSshRunner:
        """A real :class:`SshRunner` for this target."""
        return SubprocessSshRunner(self.ssh_argv())


class RemoteWorker:
    """Lifecycle of a ``strata-worker`` process on a remote box, over SSH.

    ``name`` namespaces the pidfile and log, so several notebooks can each run
    a worker on one box.
    """

    def __init__(self, name: str, runner: SshRunner) -> None:
        self.name = name
        self.runner = runner

    # -- connection ----------------------------------------------------------

    def preflight(self) -> None:
        """Verify non-interactive (key-based) SSH works, or raise; passwords are never handled."""
        res = self.runner.run("true", timeout=DEFAULT_CONNECT_TIMEOUT + 5)
        if not res.ok:
            raise SshWorkerError(
                "key-based SSH isn't working non-interactively "
                f"(ssh exited {res.returncode}). Check `ssh` to the host and your "
                f"keys / agent. {res.stderr.strip()}".rstrip()
            )

    def detect(self) -> RemoteEnvInfo:
        """Probe the box for ``strata-worker`` / ``uv`` / platform in one round-trip."""
        # ``|| true`` so a missing tool yields an empty value, not an aborted probe.
        script = (
            'printf "worker=%s\\n" "$(command -v strata-worker || true)"; '
            'printf "uv=%s\\n" "$(command -v uv || true)"; '
            'printf "platform=%s\\n" "$(uname -sm 2>/dev/null || true)"; '
            'printf "version=%s\\n" "$(strata-worker --version 2>/dev/null || true)"'
        )
        res = self.runner.run(script, timeout=30)
        if not res.ok:
            raise SshWorkerError(
                f"remote detection failed: {res.stderr.strip() or f'exit {res.returncode}'}"
            )
        fields = _parse_kv(res.stdout)
        worker_path = fields.get("worker") or None
        return RemoteEnvInfo(
            has_worker=bool(worker_path),
            has_uv=bool(fields.get("uv")),
            worker_version=fields.get("version") or None,
            platform=fields.get("platform") or "",
            worker_path=worker_path,
        )

    # -- install -------------------------------------------------------------

    def ensure_installed(
        self, info: RemoteEnvInfo, *, extras: str = "notebook", pin: str | None = None
    ) -> None:
        """Install ``strata-worker`` via ``uv tool install`` if it's missing.

        Raises when neither the worker nor ``uv`` is present; no uv installer is
        fetched.
        """
        if info.has_worker:
            return
        if not info.has_uv:
            raise SshWorkerError(
                "the remote host has neither `strata-worker` nor `uv` to install it. "
                "Install uv on the box (https://docs.astral.sh/uv/) or pre-install "
                "`strata-notebook` there, then retry."
            )
        spec = f"strata-notebook[{extras}]" if extras else "strata-notebook"
        if pin:
            spec = f"{spec}=={pin}"
        res = self.runner.run(f"uv tool install {shlex.quote(spec)}", timeout=_INSTALL_TIMEOUT)
        if not res.ok:
            detail = res.stderr.strip() or res.stdout.strip()
            raise SshWorkerError(f"remote `uv tool install {spec}` failed: {detail}")

    # -- process lifecycle ---------------------------------------------------

    def is_running(self) -> RunningWorker | None:
        """Return the recorded worker if its pid is alive on the box, else None."""
        res = self.runner.run(f"cat {self._pidfile()} 2>/dev/null || true", timeout=15)
        if not res.ok or not res.stdout.strip():
            return None
        try:
            data = json.loads(res.stdout.strip())
            pid, port = int(data["pid"]), int(data["port"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        alive = self.runner.run(f"kill -0 {pid} 2>/dev/null && echo up || true", timeout=15)
        if "up" not in alive.stdout:
            return None
        return RunningWorker(pid=pid, port=port)

    def launch(
        self, *, port: int, token: str | None, host: str = "127.0.0.1", adopt: bool = True
    ) -> RunningWorker:
        """Start the worker detached (``nohup``), recording a pidfile; return it.

        With ``adopt``, a live recorded worker on the same ``port`` is returned
        instead, but only when no token applies: the token a worker started with
        cannot be read back, and ``/health`` is unauthenticated, so a mismatch
        would surface only as a 401 on first dispatch. Otherwise it is replaced.
        The worker binds ``host`` (remote-localhost by default).
        """
        if adopt:
            existing = self.is_running()
            if existing is not None and existing.port == port:
                if token is None:
                    return existing
                self.stop()
        pidfile = self._pidfile()
        logfile = f"{_REMOTE_STATE_DIR}/worker-{shlex.quote(self.name)}.log"
        # The token goes over stdin: in the command it would be visible in ``ps`` and
        # echoed in timeout errors that reach HTTP responses and logs.
        if token and "\n" in token:
            raise SshWorkerError("worker token must not contain newlines")
        read_token = "IFS= read -r STRATA_WORKER_TOKEN && export STRATA_WORKER_TOKEN && "
        # Only the worker goes in the background: POSIX shells give an asynchronous
        # list /dev/null as stdin, so a backgrounded `read` would never see the token.
        cmd = (
            f"{read_token if token else ''}"
            f"mkdir -p {_REMOTE_STATE_DIR} && {{ "
            f"nohup strata-worker --host {shlex.quote(host)} --port {port} "
            f"> {logfile} 2>&1 & "
            "pid=$!; "
            f'printf \'{{"pid": %s, "port": %s}}\\n\' "$pid" {port} > {pidfile}; '
            "echo $pid; }"
        )
        res = self.runner.run(cmd, timeout=30, stdin_data=f"{token}\n" if token else None)
        if not res.ok:
            raise SshWorkerError(
                f"failed to launch remote worker: {res.stderr.strip() or f'exit {res.returncode}'}"
            )
        pid = _last_int(res.stdout)
        if pid is None:
            raise SshWorkerError(f"remote worker launch returned no pid: {res.stdout!r}")
        return RunningWorker(pid=pid, port=port)

    def stop(self) -> bool:
        """Stop the recorded worker and remove its pidfile; return whether one ran."""
        pidfile = self._pidfile()
        cmd = (
            f"if [ -f {pidfile} ]; then "
            f"pid=$(cat {pidfile} | sed -n 's/.*\"pid\":[ ]*\\([0-9]*\\).*/\\1/p'); "
            'if [ -n "$pid" ]; then kill "$pid" 2>/dev/null && echo stopped; fi; '
            f"rm -f {pidfile}; "
            "fi"
        )
        res = self.runner.run(cmd, timeout=15)
        if not res.ok:
            raise SshWorkerError(f"failed to stop remote worker: {res.stderr.strip()}")
        return "stopped" in res.stdout

    def _pidfile(self) -> str:
        return f"{_REMOTE_STATE_DIR}/worker-{shlex.quote(self.name)}.json"


def _parse_kv(text: str) -> dict[str, str]:
    """Parse ``key=value`` lines (the detect probe's output) into a dict."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def _last_int(text: str) -> int | None:
    """Return the last integer token in *text* (the pid ``echo``), or None."""
    for token in reversed(text.split()):
        if token.isdigit():
            return int(token)
    return None
