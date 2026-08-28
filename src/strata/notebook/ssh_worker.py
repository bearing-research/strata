"""SSH provisioning for remote notebook workers (P1 of the SSH remote-worker path).

Bring up a ``strata-worker`` on a box reachable over SSH: preflight the
connection, detect what's installed, install it if missing, and launch / stop it.
Everything runs through a small :class:`SshRunner` seam, so the provisioning
*logic* is unit-testable without a real host — a fake runner asserts the exact
command sequence. The local ``ssh -L`` tunnel and ``notebook.toml`` registration
build on this in later phases.

Two deliberate defaults for the open questions in the design doc
(``docs/internal/design-ssh-remote-worker.md``):

- **No silent uv bootstrap.** :meth:`RemoteWorker.ensure_installed` installs via
  ``uv tool install`` when ``uv`` is present, and otherwise raises with a clear
  message rather than fetching an installer over the network unprompted.
- **Portable supervision.** :meth:`RemoteWorker.launch` starts the worker with
  ``nohup`` and records a JSON pidfile under ``~/.strata`` for adoption / stop,
  rather than assuming ``systemd``.

The worker always binds remote-localhost (never a public port); it is reached
only through the authenticated SSH channel a later phase forwards.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

# ssh hardening applied to every invocation: key-only auth (``BatchMode`` — never
# prompt for a password), a bounded connect, and a keepalive. We intentionally do
# NOT pass ``StrictHostKeyChecking=no`` — establishing first-connect host-key
# trust is the user's to do; we surface the failure instead.
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
    """Runs one shell command on the remote host and returns its result.

    The single seam between provisioning logic and real ``ssh``; tests swap in a
    fake to assert the command sequence without a live host.
    """

    def run(
        self, command: str, *, timeout: float | None = None, stdin_data: str | None = None
    ) -> CommandResult:
        """Execute *command* on the remote host.

        ``stdin_data`` is fed to the remote command's stdin — the channel for
        secrets (the worker token), which must never appear in *command* (it
        would land in the local ``ssh`` argv, visible in ``ps``, and in error
        messages that echo the command).
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
            # ``command`` is safe to echo — secrets travel via ``stdin_data``,
            # which must never be included here (this message reaches HTTP
            # error bodies and logs).
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

    Parameters
    ----------
    name : str
        The worker's name; namespaces its pidfile / log so several notebooks can
        each run their own worker on one box.
    runner : SshRunner
        Executes commands on the box (a :class:`SubprocessSshRunner` in
        production, a fake in tests).
    """

    def __init__(self, name: str, runner: SshRunner) -> None:
        self.name = name
        self.runner = runner

    # -- connection ----------------------------------------------------------

    def preflight(self) -> None:
        """Verify non-interactive (key-based) SSH works, or raise.

        Runs a trivial remote ``true``; a nonzero exit means auth would prompt
        or the host is unreachable — we never handle SSH passwords.
        """
        res = self.runner.run("true", timeout=DEFAULT_CONNECT_TIMEOUT + 5)
        if not res.ok:
            raise SshWorkerError(
                "key-based SSH isn't working non-interactively "
                f"(ssh exited {res.returncode}). Check `ssh` to the host and your "
                f"keys / agent. {res.stderr.strip()}".rstrip()
            )

    def detect(self) -> RemoteEnvInfo:
        """Probe the box for ``strata-worker`` / ``uv`` / platform in one round-trip."""
        # ``|| true`` so a missing tool yields an empty value instead of a
        # nonzero exit that would abort the whole probe.
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

        A no-op when the worker is already present. Raises when neither the
        worker nor ``uv`` is available — we don't fetch a uv installer unprompted.
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

        With ``adopt`` (the default), a re-run that finds a live recorded worker
        on the same ``port`` returns it instead of starting a second. The worker
        binds ``host`` (remote-localhost by default) so it's reachable only over
        the SSH channel.

        A worker enforces the token it was *started* with, and that token isn't
        recorded anywhere we can read back. So a live worker is only adopted
        when no token has to apply: otherwise it is stopped and replaced, since
        adopting it would mean publishing a token the worker rejects — and
        ``/health`` is unauthenticated, so nothing would notice until the first
        cell dispatch came back 401.
        """
        if adopt:
            existing = self.is_running()
            if existing is not None and existing.port == port:
                if token is None:
                    return existing
                self.stop()
        pidfile = self._pidfile()
        logfile = f"{_REMOTE_STATE_DIR}/worker-{shlex.quote(self.name)}.log"
        # The token goes over stdin, not into the command: a token in the
        # command string would sit in the local ``ssh`` argv (visible in
        # ``ps``) and be echoed back in timeout error messages that reach
        # HTTP responses and logs. The remote shell reads it into the
        # worker's env before launching.
        if token and "\n" in token:
            raise SshWorkerError("worker token must not contain newlines")
        read_token = "IFS= read -r STRATA_WORKER_TOKEN && export STRATA_WORKER_TOKEN && "
        cmd = (
            f"{read_token if token else ''}"
            f"mkdir -p {_REMOTE_STATE_DIR} && "
            f"nohup strata-worker --host {shlex.quote(host)} --port {port} "
            f"> {logfile} 2>&1 & "
            "pid=$!; "
            f'printf \'{{"pid": %s, "port": %s}}\\n\' "$pid" {port} > {pidfile}; '
            "echo $pid"
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
