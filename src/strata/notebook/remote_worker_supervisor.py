"""Server-side lifecycle for SSH-tunneled remote workers.

Cells dispatch from inside the server process, so the ``ssh -L`` tunnel must be
owned there, not by a transient CLI. :class:`RemoteWorkerSupervisor` provisions
a ``strata-worker`` on the box (via :mod:`strata.notebook.ssh_worker`), forwards
to its remote-localhost port, health-checks through the tunnel, and returns the
local ``/v1/execute`` URL a ``[[workers]]`` entry points at.
"""

from __future__ import annotations

import logging
import secrets
import threading
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any, Protocol

from strata.notebook.ssh_worker import (
    _SSH_HARDENING,
    RemoteWorker,
    SshTarget,
    SshWorkerError,
)
from strata.notebook.worker_secrets import (
    clear_runtime_worker_token,
    set_runtime_worker_token,
    withhold_runtime_worker_token,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from strata.notebook.ssh_worker import SshRunner

# Fixed is safe: a re-establish adopts a worker already listening here. Callers
# running several workers on one box pass an explicit remote_port.
DEFAULT_REMOTE_PORT = 9000
_HEALTH_POLL_INTERVAL = 0.25
# A missed /health through a live forward is tried again before the token is withdrawn.
_LIVE_PROBE_TRIES = 3
_LIVE_PROBE_SPACING = 1.0
# A reopened forward's connect (ssh's ConnectTimeout is 10 s) and handshake, before
# /health can answer through it.
_RESPAWN_HEALTH_TIMEOUT = 30.0
# How long a reopened forward that another process answers through is watched before it is
# parked. Past ssh's ConnectTimeout: a process on this machine's freed port makes ssh exit
# when it cannot bind it, and the next pass reopens it; one on the box's port does not.
_FOREIGN_SETTLE = 12.0
# Reopens in a row another process answers through before the tunnel is parked: ssh may
# still be authenticating past the settle time while this machine's freed port answers.
_FOREIGN_REOPENS = 3
# The tail of ssh's stderr an error quotes: its reason is the last line or two.
_STDERR_TAIL = 1024
# Past this, what a live forward's ssh wrote is dropped: a line per refused connection
# would otherwise grow the file for as long as the forward stays up.
_STDERR_KEEP = 64 * 1024
_PORT_PICKS = 20

logger = logging.getLogger(__name__)


class TunnelHandle(Protocol):
    """A running ``ssh -L`` forward."""

    def is_alive(self) -> bool:
        """Whether the tunnel process is still running."""
        ...

    def terminate(self) -> None:
        """Tear the tunnel down."""
        ...

    def error(self) -> str:
        """The last of what ssh wrote to stderr, or ``""``: why it exited."""
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

    def __init__(
        self, proc: Any, stderr: IO[bytes] | None = None, stderr_path: str | None = None
    ) -> None:
        self._proc = proc
        self._terminated = False
        # What ssh writes to stderr, read through a handle of our own, whose offset ssh's
        # writes never share; the path is left to remove where it could not be at once.
        self._stderr = stderr
        self._stderr_path = stderr_path
        self._error = ""
        # A reconcile reads it while a teardown in another thread closes it.
        self._stderr_lock = threading.Lock()

    def is_alive(self) -> bool:
        alive = self._proc.poll() is None
        if alive:
            with self._stderr_lock:
                if self._stderr is not None and not self._stderr.closed:
                    if self._stderr.seek(0, 2) > _STDERR_KEEP:
                        self._stderr.truncate(0)  # ssh appends: its next line starts it
        return alive

    def error(self) -> str:
        with self._stderr_lock:
            if self._stderr is None or self._stderr.closed:
                return self._error
            size = self._stderr.seek(0, 2)
            start = max(0, size - _STDERR_TAIL)
            cut = False
            if start:
                self._stderr.seek(start - 1)
                cut = self._stderr.read(1) != b"\n"
            self._stderr.seek(start)
            lines = self._stderr.read().decode("utf-8", "replace").strip().splitlines()
            # A cut first line is a fragment, unless it is all there is.
            self._error = "\n".join(lines[1:] if cut and len(lines) > 1 else lines)
            return self._error

    def terminate(self) -> None:
        import contextlib
        import os
        import signal
        import subprocess

        if self._terminated:
            return  # its pid may be another process's by now
        self._terminated = True
        killpg = getattr(os, "killpg", None)  # not on Windows, which has no lifeline
        with contextlib.suppress(ProcessLookupError, OSError):
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # The process is the lifeline wrapper, which cannot pass a SIGKILL on: kill
                # its session's group (``start_new_session``), ssh with it.
                if killpg is None:
                    self._proc.kill()
                else:
                    killpg(self._proc.pid, signal.SIGKILL)
                self._proc.wait()
        if killpg is not None:
            # A wrapper that died on its own, or before its handler was set, leaves its ssh
            # in the group holding the port. The group's id is not reused while ssh lives.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                killpg(self._proc.pid, signal.SIGKILL)
        # Once ssh is gone, so all it wrote is there and the file can be removed.
        self.error()  # kept for whoever asks why after the file is closed
        with self._stderr_lock:
            if self._stderr is not None:
                self._stderr.close()
        if self._stderr_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(self._stderr_path)


class SubprocessTunnelLauncher:
    """:class:`TunnelLauncher` that runs a real ``ssh -N -L`` forward."""

    def spawn(self, ssh_target: str, *, local_port: int, remote_port: int) -> TunnelHandle:
        import contextlib
        import os
        import subprocess
        import tempfile

        from strata.notebook.lifeline import lifeline_command, lifeline_handoff

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
        # Under the lifeline so a killed server does not leave the forward holding its port.
        lifeline_fds, lifeline_vars = lifeline_handoff()
        # A file, not a pipe: a pipe nobody reads fills, blocks ssh and stops the forward.
        fd, path = tempfile.mkstemp(prefix="strata-ssh-tunnel-")
        reader: IO[bytes] | None = None
        started = False
        try:
            # ssh's, appending, so it writes at the end after the file is cut back.
            writer = os.open(path, os.O_WRONLY | os.O_APPEND)
            try:
                reader = open(path, "r+b")  # the handle closes it
                proc = subprocess.Popen(
                    lifeline_command(argv),
                    stdout=subprocess.DEVNULL,
                    stderr=writer,
                    env={**os.environ, **lifeline_vars},
                    pass_fds=lifeline_fds,
                    start_new_session=True,
                )
                started = True
            except OSError as exc:
                raise SshWorkerError(f"could not start ssh tunnel: {exc}") from exc
            finally:
                os.close(writer)
        finally:
            os.close(fd)
            if not started:
                if reader is not None:
                    reader.close()
                with contextlib.suppress(OSError):
                    os.unlink(path)
        try:
            os.unlink(path)  # gone with its last handle
        except OSError:
            return _PopenTunnelHandle(proc, reader, path)  # open files stay where (Windows)
        return _PopenTunnelHandle(proc, reader)


def _default_health_probe(local_port: int) -> bool:
    """GET ``/health`` through the tunnel; True on a 200."""
    import httpx

    try:
        resp = httpx.get(f"http://127.0.0.1:{local_port}/health", timeout=3.0)
    except httpx.HTTPError:
        return False
    return resp.status_code == 200


def _default_launch_id_probe(local_port: int) -> str | None:
    """The ``launch_id`` a healthy ``/health`` through the tunnel reports: ``""`` when it
    answers without one, ``None`` when nothing answers it with a 200 and a JSON object."""
    import httpx

    try:
        resp = httpx.get(f"http://127.0.0.1:{local_port}/health", timeout=3.0)
        body = resp.json()
    except (httpx.HTTPError, ValueError):
        return None
    if resp.status_code != 200 or not isinstance(body, dict):
        return None  # a failing /health says nothing about whose it is
    launch_id = body.get("launch_id")
    return launch_id if isinstance(launch_id, str) else ""


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
    # What the launched worker's /health reports; nothing else on the box knows it.
    launch_id: str
    # Another process answered through the forward: closed, until an establish.
    parked: bool = False
    # Reopens in a row that another process answered through.
    foreign_reopens: int = 0
    # Why its ssh last exited in this outage, as logged.
    last_exit: str | None = None
    # Its forward was closed here for a retry: no ssh exit to report.
    closed_for_retry: bool = False


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
    _lock: threading.Condition = field(default_factory=threading.Condition, init=False)
    _pending: set[str] = field(default_factory=set, init=False)
    # Names whose forward reconcile is reopening. An establish of one waits: the reopen
    # sees the reservation and stops.
    _reconnecting: set[str] = field(default_factory=set, init=False)
    _stopped: threading.Event = field(default_factory=threading.Event, init=False)

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
            while name in self._reconnecting:
                self._lock.wait()
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
            executable = worker.ensure_installed(info, extras=extras, pin=pin)
        elif info.worker_path:
            executable = info.worker_path
        else:
            raise SshWorkerError(
                f"{name}: strata-worker isn't installed on the box and install=False"
            )

        token = token or secrets.token_urlsafe(32)
        rport = remote_port or DEFAULT_REMOTE_PORT
        launch_id = secrets.token_urlsafe(16)
        running = worker.launch(port=rport, token=token, launch_id=launch_id, executable=executable)
        lport = self._local_port(name, local_port)
        handle = self.tunnel_launcher.spawn(
            target.target, local_port=lport, remote_port=running.port
        )
        if not self._await_health(lport, health_timeout, handle):
            exited = not handle.is_alive()
            handle.terminate()
            if exited:
                raise SshWorkerError(
                    f"{name}: ssh exited before the worker's /health answered on "
                    f"127.0.0.1:{lport}: {handle.error() or 'it wrote nothing'}"
                )
            log = worker.log_tail()
            raise SshWorkerError(
                f"{name}: tunnel opened but the worker's /health didn't respond on "
                f"127.0.0.1:{lport} within {health_timeout}s"
                + (f"; ~/.strata/worker-{name}.log on the box ends:\n{log}" if log else "")
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
            stopped = self._stopped.is_set()
            if not stopped:
                # Published with the entry, so a teardown in between cannot leave it behind.
                set_runtime_worker_token(name, token)
                self._tunnels[name] = _ActiveTunnel(
                    handle=handle,
                    worker=worker,
                    record=record,
                    token=token,
                    local_port=lport,
                    remote_port=running.port,
                    launch_id=launch_id,
                )
        if stopped:
            # The server shut down meanwhile: nothing would close this forward.
            handle.terminate()
            raise SshWorkerError(f"{name}: the server is shutting down")
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
        """Return every established worker's record: healthy as the last check found it,
        unless its forward's process has exited since."""
        with self._lock:
            actives = list(self._tunnels.items())
        records = [self._refresh(name, active) for name, active in actives]
        return [record for record in records if record is not None]

    def reconcile(self) -> list[TunnelRecord]:
        """Re-open every tunnel whose ``ssh -L`` exited, and refresh the others' health.

        The server calls it every 30 s (``_ssh_tunnel_reconcile_loop``). A live forward is
        never torn down for a missed probe: a busy box may have a cell streaming through
        it. Healthy means ``/health`` reports the worker this server launched, as at
        establish, and the worker's token is published only while it does. A live forward
        that reaches another process is closed until :meth:`establish` runs again. A
        forward whose ssh keeps exiting (an unreachable box) is reopened on every pass. A
        worker that stopped is not restarted: its forward stays open and unhealthy.
        """
        with self._lock:
            names = list(self._tunnels)
        records: list[TunnelRecord] = []
        for name in names:
            try:
                record = self._reconcile_one(name)
            except (SshWorkerError, OSError) as exc:
                # One tunnel's failure leaves the others to be checked.
                logger.warning("ssh worker %s: not reconciled: %s", name, exc)
                continue
            if record is not None:
                records.append(record)
        return records

    def _local_port(self, name: str, wanted: int | None) -> int:
        """A local port no other tunnel holds: one whose ssh exited is reopened on its own."""
        with self._lock:
            held = {t.local_port for n, t in self._tunnels.items() if n != name}
        if wanted is not None:
            if wanted in held:
                raise SshWorkerError(f"{name}: local port {wanted} is another worker's tunnel")
            return wanted
        for _ in range(_PORT_PICKS):
            port = self.port_picker()
            if port not in held:
                return port
        raise SshWorkerError(f"{name}: no free local port that no other tunnel holds")

    def _reconcile_one(self, name: str) -> TunnelRecord | None:
        with self._lock:
            active = self._tunnels.get(name)
            if active is None or name in self._pending or self._stopped.is_set():
                return None  # torn down, an establish is replacing it, or shutting down
            if active.parked or name in self._reconnecting:
                return active.record  # closed until an establish, or another pass reopens it
            handle = active.handle
            alive = handle.is_alive()
            if not alive:
                # So no establish of this name spawns on the same local port meanwhile.
                self._reconnecting.add(name)
        if alive:
            return self._check(name, active, handle, self._probe_live(handle, active))
        try:
            return self._respawn(name, active)
        finally:
            with self._lock:
                self._reconnecting.discard(name)
                self._lock.notify_all()

    def _respawn(self, name: str, active: _ActiveTunnel) -> TunnelRecord | None:
        withhold_runtime_worker_token(name)  # until the new forward reaches our worker
        active.handle.terminate()  # reaps the exited ssh
        why = active.handle.error() or "it wrote nothing"
        # Once per reason in an outage: an unreachable box is reopened on every pass. One
        # this server closed for a retry was logged as it was.
        if not active.closed_for_retry and why != active.last_exit:
            active.last_exit = why
            logger.warning("ssh worker %s: ssh exited, reopening the tunnel: %s", name, why)
        active.closed_for_retry = False
        try:
            handle = self.tunnel_launcher.spawn(
                active.record.ssh_target,
                local_port=active.local_port,
                remote_port=active.remote_port,
            )
        except SshWorkerError as exc:
            logger.warning("ssh worker %s: tunnel not reopened: %s", name, exc)
            return self._check(name, active, active.handle, None, reopening=True)
        with self._lock:
            replaced = self._tunnels.get(name) is not active
            if not replaced:
                active.handle = handle
        if replaced:
            # Torn down while respawning: don't orphan the forward or revive the entry.
            handle.terminate()
            return None
        seen = self._await_launch_id(name, active)
        return self._check(name, active, handle, seen, reopening=True)

    def _check(
        self,
        name: str,
        active: _ActiveTunnel,
        handle: TunnelHandle,
        seen: str | None,
        *,
        reopening: bool = False,
    ) -> TunnelRecord | None:
        """Record what ``/health`` reported through *handle*, *active*'s forward: *seen*,
        its launch id (``""`` none; ``None`` no healthy answer). *reopening*: the call of
        the reconnect itself."""
        ours = seen == active.launch_id
        with self._lock:
            if self._tunnels.get(name) is not active:
                return None
            if (
                active.parked
                or active.handle is not handle
                or (name in self._reconnecting and not reopening)
            ):
                # Probed through a forward that is being, or has been, replaced.
                return active.record
            # Ours only through our own live forward: one that lost its port to another of
            # these tunnels reaches that worker instead.
            others = {t.launch_id for n, t in self._tunnels.items() if n != name}
            foreign = (
                seen is not None and not ours and seen not in others and active.handle.is_alive()
            )
            # Withdrawn on anything but our worker's answer: the token must not reach a
            # process that took the box's port, answered or not.
            if ours:
                set_runtime_worker_token(name, active.token)
                active.foreign_reopens = 0
                active.last_exit = None  # the outage is over
            else:
                withhold_runtime_worker_token(name)
            retry = False
            if foreign and reopening:
                # Through a forward still connecting, the answer may be this machine's.
                active.foreign_reopens += 1
                retry = active.foreign_reopens < _FOREIGN_REOPENS
            if foreign and not retry:
                active.parked = True
            active.closed_for_retry = retry
            active.record = _replace_health(active.record, ours)
            record = active.record
        if retry:
            handle.terminate()
            logger.warning(
                "ssh worker %s: /health on 127.0.0.1:%d answered as another process while "
                "the tunnel reopened; trying again on the next check",
                name,
                active.local_port,
            )
        elif foreign:
            handle.terminate()
            logger.warning(
                "ssh worker %s: /health on 127.0.0.1:%d answered, but not as the worker "
                "this server launched (another process holds the box's port, or this "
                "machine's); tunnel closed until it is connected again",
                name,
                active.local_port,
            )
        return record

    def _probe_live(self, handle: TunnelHandle, active: _ActiveTunnel) -> str | None:
        """The launch id through a live forward, tried a few times: one slow /health on a
        busy box should not withdraw its token."""
        import time

        for attempt in range(_LIVE_PROBE_TRIES):
            seen = self.launch_id_probe(active.local_port)
            if seen is not None or not handle.is_alive() or self._stopped.is_set():
                return seen
            if attempt + 1 < _LIVE_PROBE_TRIES:
                time.sleep(_LIVE_PROBE_SPACING)
        return None

    def _await_launch_id(self, name: str, active: _ActiveTunnel) -> str | None:
        import time

        deadline = time.monotonic() + _RESPAWN_HEALTH_TIMEOUT
        settled: float | None = None  # when a foreign answer is taken as final
        while True:
            with self._lock:
                abandoned = (
                    self._stopped.is_set()
                    or name in self._pending
                    or self._tunnels.get(name) is not active
                )
            if abandoned or not active.handle.is_alive():
                return None
            seen = self.launch_id_probe(active.local_port)
            now = time.monotonic()
            if seen == active.launch_id:
                return seen
            if seen is not None:
                settled = settled if settled is not None else now + _FOREIGN_SETTLE
                if now >= settled:
                    return seen
            elif now >= deadline:
                return None
            time.sleep(_HEALTH_POLL_INTERVAL)

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
        self._stopped.set()  # ends a reconcile still waiting on a reopened forward
        with self._lock:
            actives = list(self._tunnels.items())
            self._tunnels.clear()
        for name, active in actives:
            clear_runtime_worker_token(name)
            active.handle.terminate()

    def _await_health(self, local_port: int, timeout: float, handle: TunnelHandle) -> bool:
        import time

        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if self.health_probe(local_port):
                return True
            # No forward left to answer through.
            if time.monotonic() >= deadline or not handle.is_alive():
                return False
            time.sleep(_HEALTH_POLL_INTERVAL)

    def _refresh(self, name: str, active: _ActiveTunnel) -> TunnelRecord | None:
        # The last check's verdict, and the forward's process now: no probe, so a listing
        # neither waits on a slow box nor changes the token (``reconcile`` does).
        with self._lock:
            if self._tunnels.get(name) is not active:
                return None
            record = active.record
            alive = active.handle.is_alive()
        return record if alive or not record.healthy else _replace_health(record, False)


def _replace_health(record: TunnelRecord, healthy: bool) -> TunnelRecord:
    from dataclasses import replace

    return replace(record, healthy=healthy)
