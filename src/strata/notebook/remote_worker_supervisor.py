"""Server-side lifecycle for SSH-tunneled remote workers.

Cells dispatch from inside the server process, so the ``ssh -L`` tunnel must be
owned there, not by a transient CLI. :class:`RemoteWorkerSupervisor` provisions
a ``strata-worker`` on the box (via :mod:`strata.notebook.ssh_worker`), forwards
to its remote-localhost port, health-checks through the tunnel, and returns the
local ``/v1/execute`` URL a ``[[workers]]`` entry points at.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from strata.notebook.ssh_worker import (
    _SSH_HARDENING,
    RemoteWorker,
    SshTarget,
    SshWorkerError,
)
from strata.notebook.worker_secrets import (
    clear_runtime_worker_token,
    set_runtime_worker_token,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from strata.notebook.ssh_worker import SshRunner

# Fixed is safe: a re-establish adopts a worker already listening here. Callers
# running several workers on one box pass an explicit remote_port.
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
    executor_url: str  # http://127.0.0.1:{local_port}/v1/execute (the [[workers]] url)


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
            "-N",  # forward only, no remote command
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


def _default_launch_id_probe(local_port: int) -> str | None:
    """The ``launch_id`` the worker answering ``/health`` through the tunnel reports."""
    import httpx

    try:
        resp = httpx.get(f"http://127.0.0.1:{local_port}/health", timeout=3.0)
        body = resp.json()
    except (httpx.HTTPError, ValueError):
        return None
    launch_id = body.get("launch_id") if isinstance(body, dict) else None
    return launch_id if isinstance(launch_id, str) else None


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
    """Owns the ``ssh -L`` tunnels and remote workers for a running server.

    All external effects (``tunnel_launcher``, ``health_probe``, ``launch_id_probe``,
    ``port_picker``, ``runner_factory``) are injected so it is testable without a host.
    """

    tunnel_launcher: TunnelLauncher = field(default_factory=SubprocessTunnelLauncher)
    health_probe: Callable[[int], bool] = _default_health_probe
    launch_id_probe: Callable[[int], str | None] = _default_launch_id_probe
    port_picker: Callable[[], int] = _pick_free_port
    runner_factory: Callable[[SshTarget], SshRunner] | None = None
    _tunnels: dict[str, _ActiveTunnel] = field(default_factory=dict, init=False)
    # Methods run on to_thread workers (real OS threads). The lock guards
    # ``_tunnels`` in short sections, never across ssh/provisioning, so
    # shutdown never waits on a slow remote install. ``_pending`` reserves a
    # name during its establish so two concurrent establishes can't both
    # pass the existence check and orphan a tunnel.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _pending: set[str] = field(default_factory=set, init=False)

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
        """Provision, tunnel and health-check a remote worker; return its record.

        Re-establishing a ``name`` tears its previous tunnel down first; a concurrent
        establish for the same ``name`` is rejected. Raises :class:`SshWorkerError` if
        any step fails, after cleaning up the tunnel.
        """
        with self._lock:
            if name in self._pending:
                raise SshWorkerError(f"{name}: an establish is already in progress")
            self._pending.add(name)
        try:
            return self._establish_locked_out(
                name,
                ssh_target,
                remote_port=remote_port,
                local_port=local_port,
                token=token,
                extras=extras,
                pin=pin,
                install=install,
                health_timeout=health_timeout,
            )
        finally:
            with self._lock:
                self._pending.discard(name)

    def _establish_locked_out(
        self,
        name: str,
        ssh_target: str,
        *,
        remote_port: int | None,
        local_port: int | None,
        token: str | None,
        extras: str,
        pin: str | None,
        install: bool,
        health_timeout: float,
    ) -> TunnelRecord:
        """The slow body of :meth:`establish`, run with ``name`` reserved in ``_pending``.

        Never holds ``_lock`` across ssh round-trips.
        """
        if self.get(name) is not None:
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
        launch_id = secrets.token_urlsafe(16)
        running = worker.launch(port=rport, token=token, launch_id=launch_id)
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
        # Another listener already on the remote port would answer /health and then
        # receive the bearer token, even while our worker is still starting up to fail
        # its bind; only the worker we launched knows this launch id.
        if self.launch_id_probe(lport) != launch_id:
            handle.terminate()
            raise SshWorkerError(
                f"{name}: /health answered on remote port {running.port}, but not from "
                f"the worker launched as pid {running.pid}; another process may hold that "
                "port, or the box's strata-worker is older than this server "
                f"(see ~/.strata/worker-{name}.log on the box)"
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
        with self._lock:
            self._tunnels[name] = _ActiveTunnel(
                handle=handle,
                worker=worker,
                record=record,
                token=token,
                local_port=lport,
                remote_port=running.port,
            )
        # Lets the executor authenticate by worker name without writing the
        # secret to notebook.toml.
        set_runtime_worker_token(name, token)
        return record

    def token_for(self, name: str) -> str | None:
        """Return the generated bearer token for *name* (held in memory, not on disk)."""
        with self._lock:
            active = self._tunnels.get(name)
        return active.token if active is not None else None

    def get(self, name: str) -> TunnelRecord | None:
        """Return the record for *name*, or None if not established."""
        with self._lock:
            active = self._tunnels.get(name)
        return active.record if active is not None else None

    def status(self) -> list[TunnelRecord]:
        """Return every established worker's record (with a fresh liveness flag)."""
        with self._lock:
            actives = list(self._tunnels.values())
        return [self._refresh(active) for active in actives]

    def reconcile(self) -> list[TunnelRecord]:
        """Health-check every tunnel; respawn any whose forward or worker is down.

        Called from the server's health loop. A dead tunnel is re-opened on the same
        ports; ``healthy`` reflects the result.
        """
        with self._lock:
            actives = list(self._tunnels.items())
        records: list[TunnelRecord] = []
        for name, active in actives:
            if active.handle.is_alive() and self.health_probe(active.local_port):
                records.append(active.record)
                continue
            active.handle.terminate()
            new_handle = self.tunnel_launcher.spawn(
                active.record.ssh_target,
                local_port=active.local_port,
                remote_port=active.remote_port,
            )
            with self._lock:
                if self._tunnels.get(name) is not active:
                    # Torn down or replaced while respawning: don't orphan
                    # the fresh forward or resurrect the entry.
                    new_handle.terminate()
                    continue
                active.handle = new_handle
            healthy = self._await_health(active.local_port, _HEALTH_POLL_INTERVAL)
            active.record = _replace_health(active.record, healthy)
            records.append(active.record)
        return records

    def teardown(self, name: str, *, stop_remote: bool = False) -> bool:
        """Close *name*'s tunnel; return whether one was present.

        ``stop_remote`` also kills the ``strata-worker`` on the box; by default it is
        left running for reuse.
        """
        with self._lock:
            active = self._tunnels.pop(name, None)
        if active is None:
            return False
        clear_runtime_worker_token(name)
        active.handle.terminate()
        if stop_remote:
            import contextlib

            with contextlib.suppress(SshWorkerError):
                active.worker.stop()
        return True

    def shutdown(self) -> None:
        """Tear down every tunnel, leaving remote workers running for reuse.

        Runs at server shutdown so no ``ssh -L`` children outlive the server.
        """
        with self._lock:
            actives = list(self._tunnels.items())
            self._tunnels.clear()
        for name, active in actives:
            clear_runtime_worker_token(name)
            active.handle.terminate()

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
