"""Server-side lifecycle for SSH-tunneled remote workers (P2 of the SSH path).

Cell dispatch runs *inside* the notebook server process, so the local end of an
``ssh -L`` tunnel must be reachable from that process — which means the tunnel
subprocess has to be owned and supervised where the server lives, not by a
transient CLI. :class:`RemoteWorkerSupervisor` is that owner: given an SSH target
it provisions a ``strata-worker`` on the box (via :mod:`strata.notebook.ssh_worker`),
opens a forward to the worker's remote-localhost port, health-checks through it,
and hands back the local ``/v1/execute`` URL a ``[[workers]]`` entry points at.

Everything external is a seam — the tunnel launcher, the health probe, the port
picker, the SSH runner — so the supervisor is unit-testable without a real host,
a real ``ssh``, or a live socket. Wiring it into the server lifespan and the REST
surface (and persisting tunnel intent for re-establish on restart) is a separate
follow-up; this module is the supervisor core.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from strata.notebook.ssh_worker import (
    _SSH_HARDENING,
    RemoteWorker,
    SshTarget,
    SshWorkerError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from strata.notebook.ssh_worker import SshRunner

# The worker's default bind port on the box. A re-establish adopts a live worker
# already listening here (see RemoteWorker.launch), so the fixed default is safe;
# callers that run several workers on one box pass an explicit remote_port.
DEFAULT_REMOTE_PORT = 9000
_HEALTH_POLL_INTERVAL = 0.25


class TunnelHandle(Protocol):
    """A running ``ssh -L`` forward."""

    def is_alive(self) -> bool:
        """Whether the tunnel process is still running."""
        ...

    def terminate(self) -> None:
        """Tear the tunnel down."""
        ...


class TunnelLauncher(Protocol):
    """Opens an ``ssh -L`` forward from ``local_port`` to the box's ``remote_port``."""

    def spawn(self, ssh_target: str, *, local_port: int, remote_port: int) -> TunnelHandle:
        """Start the forward and return a handle to it."""
        ...


@dataclass(frozen=True)
class TunnelRecord:
    """The public state of one established remote worker."""

    name: str
    ssh_target: str
    local_port: int
    remote_port: int
    remote_pid: int
    healthy: bool
    executor_url: str  # http://127.0.0.1:{local_port}/v1/execute — the [[workers]] url


class _PopenTunnelHandle:
    """:class:`TunnelHandle` backed by a ``subprocess.Popen`` running ``ssh -N -L``."""

    def __init__(self, proc: Any) -> None:
        self._proc = proc

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def terminate(self) -> None:
        import contextlib

        with contextlib.suppress(ProcessLookupError, OSError):
            self._proc.terminate()
            self._proc.wait(timeout=5)


class SubprocessTunnelLauncher:
    """:class:`TunnelLauncher` that runs a real ``ssh -N -L`` forward."""

    def spawn(self, ssh_target: str, *, local_port: int, remote_port: int) -> TunnelHandle:
        import subprocess

        argv = [
            "ssh",
            "-N",  # no remote command — just the forward
            "-o",
            "ExitOnForwardFailure=yes",
            *_SSH_HARDENING,
            "-L",
            f"{local_port}:127.0.0.1:{remote_port}",
            ssh_target,
        ]
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except OSError as exc:
            raise SshWorkerError(f"could not start ssh tunnel: {exc}") from exc
        return _PopenTunnelHandle(proc)


def _default_health_probe(local_port: int) -> bool:
    """GET ``/health`` through the tunnel; True on a 200."""
    import httpx

    try:
        resp = httpx.get(f"http://127.0.0.1:{local_port}/health", timeout=3.0)
    except httpx.HTTPError:
        return False
    return resp.status_code == 200


def _pick_free_port() -> int:
    """Ask the OS for a free local TCP port (bind :0, read it back, release)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class _ActiveTunnel:
    handle: TunnelHandle
    worker: RemoteWorker
    record: TunnelRecord
    token: str
    local_port: int
    remote_port: int


@dataclass
class RemoteWorkerSupervisor:
    """Owns the ``ssh -L`` tunnels + remote workers for a running server.

    All external effects are injected so the supervisor is testable without a
    host: ``tunnel_launcher`` opens the forward, ``health_probe`` checks the
    worker through it, ``port_picker`` chooses the local port, and
    ``runner_factory`` builds the SSH runner for provisioning.
    """

    tunnel_launcher: TunnelLauncher = field(default_factory=SubprocessTunnelLauncher)
    health_probe: Callable[[int], bool] = _default_health_probe
    port_picker: Callable[[], int] = _pick_free_port
    runner_factory: Callable[[SshTarget], SshRunner] | None = None
    _tunnels: dict[str, _ActiveTunnel] = field(default_factory=dict, init=False)

    def establish(
        self,
        name: str,
        ssh_target: str,
        *,
        remote_port: int | None = None,
        local_port: int | None = None,
        token: str | None = None,
        extras: str = "notebook",
        pin: str | None = None,
        install: bool = True,
        health_timeout: float = 10.0,
    ) -> TunnelRecord:
        """Provision + tunnel + health-check a remote worker; return its record.

        Idempotent per ``name``: re-establishing tears the previous tunnel down
        first. Raises :class:`SshWorkerError` if any step fails (the tunnel is
        cleaned up before the error propagates).
        """
        if name in self._tunnels:
            self.teardown(name)

        target = SshTarget(ssh_target)
        runner = self.runner_factory(target) if self.runner_factory else target.runner()
        worker = RemoteWorker(name, runner)
        worker.preflight()
        info = worker.detect()
        if install:
            worker.ensure_installed(info, extras=extras, pin=pin)
        elif not info.has_worker:
            raise SshWorkerError(
                f"{name}: strata-worker isn't installed on the box and install=False"
            )

        token = token or secrets.token_urlsafe(32)
        rport = remote_port or DEFAULT_REMOTE_PORT
        running = worker.launch(port=rport, token=token)
        lport = local_port or self.port_picker()
        handle = self.tunnel_launcher.spawn(
            target.target, local_port=lport, remote_port=running.port
        )
        if not self._await_health(lport, health_timeout):
            handle.terminate()
            raise SshWorkerError(
                f"{name}: tunnel opened but the worker's /health didn't respond on "
                f"127.0.0.1:{lport} within {health_timeout}s"
            )

        record = TunnelRecord(
            name=name,
            ssh_target=target.target,
            local_port=lport,
            remote_port=running.port,
            remote_pid=running.pid,
            healthy=True,
            executor_url=f"http://127.0.0.1:{lport}/v1/execute",
        )
        self._tunnels[name] = _ActiveTunnel(
            handle=handle,
            worker=worker,
            record=record,
            token=token,
            local_port=lport,
            remote_port=running.port,
        )
        return record

    def token_for(self, name: str) -> str | None:
        """Return the generated bearer token for *name* (held in memory, not on disk)."""
        active = self._tunnels.get(name)
        return active.token if active is not None else None

    def get(self, name: str) -> TunnelRecord | None:
        """Return the record for *name*, or None if not established."""
        active = self._tunnels.get(name)
        return active.record if active is not None else None

    def status(self) -> list[TunnelRecord]:
        """Return every established worker's record (with a fresh liveness flag)."""
        return [self._refresh(active) for active in self._tunnels.values()]

    def reconcile(self) -> list[TunnelRecord]:
        """Health-check every tunnel; respawn any whose forward or worker is down.

        The synchronous heartbeat: call it from the server's health loop. A dead
        ``ssh -L`` (or an unresponsive worker behind a live forward) is torn down
        and re-opened on the same ports; ``healthy`` reflects the result.
        """
        for active in list(self._tunnels.values()):
            if active.handle.is_alive() and self.health_probe(active.local_port):
                continue
            active.handle.terminate()
            new_handle = self.tunnel_launcher.spawn(
                active.record.ssh_target,
                local_port=active.local_port,
                remote_port=active.remote_port,
            )
            active.handle = new_handle
            healthy = self._await_health(active.local_port, _HEALTH_POLL_INTERVAL)
            active.record = _replace_health(active.record, healthy)
        return [active.record for active in self._tunnels.values()]

    def teardown(self, name: str, *, stop_remote: bool = False) -> bool:
        """Close *name*'s tunnel (and optionally stop the remote worker).

        Returns whether a tunnel was present. ``stop_remote`` also kills the
        ``strata-worker`` on the box; by default it's left running for reuse.
        """
        active = self._tunnels.pop(name, None)
        if active is None:
            return False
        active.handle.terminate()
        if stop_remote:
            import contextlib

            with contextlib.suppress(SshWorkerError):
                active.worker.stop()
        return True

    def shutdown(self) -> None:
        """Tear down every tunnel (leaving remote workers running for reuse).

        Wired to the server lifespan's shutdown so no ``ssh -L`` children outlive
        the server.
        """
        for active in self._tunnels.values():
            active.handle.terminate()
        self._tunnels.clear()

    def _await_health(self, local_port: int, timeout: float) -> bool:
        import time

        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if self.health_probe(local_port):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_HEALTH_POLL_INTERVAL)

    def _refresh(self, active: _ActiveTunnel) -> TunnelRecord:
        healthy = active.handle.is_alive() and self.health_probe(active.local_port)
        active.record = _replace_health(active.record, healthy)
        return active.record


def _replace_health(record: TunnelRecord, healthy: bool) -> TunnelRecord:
    from dataclasses import replace

    return replace(record, healthy=healthy)
