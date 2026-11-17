"""A project's scratchpad, ``<anchor>/.strata-scratch/``, and the one daemon serving it.

The first client to need it starts the daemon (a personal-mode server with the
scratchpad MCP profile, on a free loopback port, that exits when idle), opens the
scratchpad in it and writes ``server.json``; later clients find it there. A file
lock makes find-or-start one step, so two agents in one project share one daemon
and never write one directory from two sessions.

Imports stay light (no ``strata.notebook`` on the path that finds a running
daemon): the stdio bridge calls this on every tool call.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import filelock
import httpx

SCRATCH_DIR = ".strata-scratch"
IDLE_MINUTES = 30
# How long a daemon that stopped answering may take to finish dying.
_DYING_GRACE = 2.0
# Starting the server, then the scratchpad's first environment sync, can take minutes
# on a cold uv cache.
_SETUP_TIMEOUT = 600.0
# Windows: OpenProcess access, the error a live process of another user gives, and
# GetExitCodeProcess's "not exited".
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5
_STILL_ACTIVE = 259


class ScratchError(RuntimeError):
    """The scratchpad's daemon could not be found or started; the message says why."""


class NotAnswering(ScratchError):
    """The daemon's process is alive but did not answer in time: busy, not gone."""


@dataclass(frozen=True)
class Daemon:
    """A running daemon and the scratchpad session it holds open."""

    url: str
    session_id: str
    pid: int
    # The strata-notebook version that started it; another version's daemon is replaced
    # once it is idle.
    version: str = ""
    # Milliseconds per setup step, in order; empty when the daemon was already up.
    setup_ms: dict[str, int] = field(default_factory=dict)


def scratch_dir(anchor: Path) -> Path:
    return anchor / SCRATCH_DIR


def _record_path(anchor: Path) -> Path:
    return scratch_dir(anchor) / ".strata" / "server.json"


def _log_path(anchor: Path) -> Path:
    return scratch_dir(anchor) / ".strata" / "server.log"


def _sync_state(url: str, session_id: str) -> str | None:
    """The session's environment sync state, or None if nothing listens at *url* or the
    session is not open there. Raises ``httpx.HTTPError`` for a server that does not
    answer in time, or answers with an error: it may still hold the session."""
    try:
        response = httpx.get(f"{url}/v1/notebooks/{session_id}/environment", timeout=5.0)
    except httpx.ConnectError:
        return None
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()["environment"]["sync_state"]


def _environment(url: str, session_id: str) -> str | None:
    """:func:`_sync_state`, with no answer in time as None: for polling a start."""
    try:
        return _sync_state(url, session_id)
    except httpx.HTTPError:
        return None


def daemon_serving(notebook_dir: Path) -> Daemon | None:
    """The daemon ``<notebook_dir>/.strata/server.json`` names, if it answers and still
    holds the session.

    Asked, not judged by its pid: a session id is never reused, so no other server
    can answer for it.
    """
    try:
        record = json.loads((notebook_dir / ".strata" / "server.json").read_text())
    except (FileNotFoundError, NotADirectoryError):
        return None
    if scratch_dir(Path(record["anchor"])).resolve() != notebook_dir.resolve():
        # A copy of the directory, record and all: that daemon serves the original.
        return None
    try:
        state = _sync_state(record["url"], record["session_id"])
    except httpx.HTTPError as exc:
        # A process killed a moment ago still exists while its connections reset.
        if _exits_within(record["pid"], _DYING_GRACE):
            return None
        # Alive but slow, not gone: another daemon beside it would write the directory
        # from a second session.
        raise NotAnswering(
            f"the scratchpad's server (pid {record['pid']}) is not answering ({exc!r}); "
            "try again shortly"
        ) from exc
    if state is None:
        return None
    return Daemon(
        url=record["url"],
        session_id=record["session_id"],
        pid=record["pid"],
        version=record["version"],
    )


def find_daemon(anchor: Path) -> Daemon | None:
    """The live daemon serving *anchor*'s scratchpad, if any."""
    return daemon_serving(scratch_dir(anchor))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _scaffold(anchor: Path) -> None:
    """Create the scratchpad notebook, ignored by the project's git as a ``.venv`` is."""
    from strata.notebook.writer import create_notebook

    scratch = scratch_dir(anchor)
    create_notebook(
        anchor,
        "scratch",
        directory=SCRATCH_DIR,
        initialize_environment=False,
        write_gitignore_file=False,
        # Pinned, read-only: project files by `project / "path"`, never a stale cell.
        project_mount="project",
    )
    ignore = scratch / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n")


# Started by a launcher that exits at once, so the daemon is nobody's child: no client
# has to reap it, and it outlives the client that started it.
_LAUNCH = """\
import subprocess, sys
with open(sys.argv[1], "wb") as log:
    proc = subprocess.Popen(
        sys.argv[2:], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(proc.pid)
"""


def _server_command() -> list[str]:
    return [sys.executable, "-m", "strata"]


def _spawn(anchor: Path, port: int) -> int:
    """Start the daemon detached; return its pid."""
    from vestigo.file_modes import private_file

    strata_home = Path.home() / ".strata"
    env = {
        # The caches a personal server keeps, not ones beside the storage dir, which is
        # the project itself here. The environment may still say otherwise.
        "STRATA_NOTEBOOK_MOUNT_CACHE_DIR": str(strata_home / "mount_cache"),
        "STRATA_NOTEBOOK_SHARED_ENV_DIR": str(strata_home / "envs"),
        **os.environ,
        "STRATA_HOST": "127.0.0.1",
        "STRATA_PORT": str(port),
        "STRATA_DEPLOYMENT_MODE": "personal",
        "STRATA_MCP_ENABLED": "true",
        "STRATA_MCP_PROFILE": "scratchpad",
        # The server opens notebooks only inside its storage dir.
        "STRATA_NOTEBOOK_STORAGE_DIR": str(anchor),
        "STRATA_IDLE_EXIT_MINUTES": os.environ.get("STRATA_IDLE_EXIT_MINUTES", str(IDLE_MINUTES)),
        # Its own sweep, by idle days and size, replaces the count kept per value: a
        # scratchpad keeps superseded results so that reverting an edit hits again.
        "STRATA_SCRATCH_ANCHOR": str(anchor),
        "STRATA_NOTEBOOK_KEEP_SUPERSEDED_VERSIONS": "0",
    }
    try:
        log = private_file(_log_path(anchor))
        launched = subprocess.run(
            [sys.executable, "-I", "-c", _LAUNCH, str(log), *_server_command()],
            env=env,
            # Its [tool.strata] is then the scratchpad's (none), not whichever
            # pyproject.toml sits above where the first client ran.
            cwd=scratch_dir(anchor),
            capture_output=True,
            text=True,
            check=True,
        )
        return int(launched.stdout)
    except subprocess.CalledProcessError as exc:
        raise ScratchError(f"the scratchpad's server could not be launched: {exc.stderr}") from exc
    except (OSError, ValueError) as exc:
        raise ScratchError(f"the scratchpad's server could not be launched: {exc}") from exc


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) would terminate it there: ask for its exit code instead.
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Another user's process: it exists.
        return True
    return True


def _until(ready: Callable[[], bool], pid: int, what: str, deadline: float, log: Path) -> None:
    while not ready():
        if not _alive(pid):
            raise ScratchError(f"the scratchpad's server exited while {what}; see {log}")
        if time.monotonic() > deadline:
            raise ScratchError(f"the scratchpad's server timed out {what}; see {log}")
        time.sleep(0.1)


def _exits_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _kill(pid: int) -> None:
    """Kill the daemon and what it started: it leads its own session, so its group."""
    # Gone already raises ProcessLookupError, or on Windows a plain OSError.
    with contextlib.suppress(OSError):
        if sys.platform == "win32":
            os.kill(pid, signal.SIGTERM)
        else:
            os.killpg(pid, signal.SIGKILL)


def _healthy(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=2.0).is_success
    except httpx.HTTPError:
        return False


def _busy(url: str) -> bool:
    """Whether the daemon at *url* has work besides this question. One that does not say is
    taken as busy, so nothing it runs is stopped."""
    try:
        response = httpx.get(f"{url}/health/busy", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return True
    return response.json()["busy"]


def _start(
    anchor: Path, step: Callable[[str], contextlib.AbstractContextManager[None]]
) -> tuple[int, str, str]:
    """Start a daemon on *anchor* and open its scratchpad; return it, its url and session."""
    deadline = time.monotonic() + _SETUP_TIMEOUT
    if not (scratch_dir(anchor) / "notebook.toml").exists():
        with step("creating the scratchpad"):
            _scaffold(anchor)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    pid = _spawn(anchor, port)
    log = _log_path(anchor)
    try:
        with step("starting the scratchpad"):
            _until(lambda: _healthy(url), pid, "starting", deadline, log)
        with step("opening the scratchpad"):
            try:
                # Open awaits the first environment sync, which on a cold uv cache
                # takes minutes: the rest of the setup budget, not a request's.
                opened = httpx.post(
                    f"{url}/v1/notebooks/open",
                    json={"path": str(scratch_dir(anchor))},
                    timeout=max(deadline - time.monotonic(), 1.0),
                )
            except httpx.HTTPError as exc:
                raise ScratchError(
                    f"the scratchpad's server did not open it ({exc}); see {log}"
                ) from exc
            if not opened.is_success:
                raise ScratchError(
                    f"the scratchpad's server refused to open it: {opened.text[:300]}"
                )
            session_id = opened.json()["session_id"]
        with step("syncing the scratchpad's environment"):
            # Its tools refuse to run until then; failed is theirs to report, with the fix.
            _until(
                lambda: _environment(url, session_id) not in ("pending", None),
                pid,
                "syncing its environment",
                deadline,
                log,
            )
    except BaseException:
        # Not left running for nobody: no server.json names it.
        _kill(pid)
        raise
    return pid, url, session_id


def daemon_lock(anchor: Path) -> filelock.FileLock:
    """Held while a daemon is started: held too, none starts behind a writer of the files."""
    return filelock.FileLock(scratch_dir(anchor) / ".strata" / "daemon.lock")


def ensure_daemon(anchor: Path, progress: Callable[[str], None] | None = None) -> Daemon:
    """The daemon for *anchor*'s scratchpad, started (and the scratchpad created) if none runs.

    *progress* hears each setup step as it starts.
    """
    anchor = anchor.resolve()
    try:
        (scratch_dir(anchor) / ".strata").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ScratchError(f"the scratchpad cannot be created in {anchor}: {exc}") from exc
    with daemon_lock(anchor):
        running = find_daemon(anchor)
        current = version("strata-notebook")
        if running is not None and running.version == current:
            return running
        if running is not None:
            # Started by another install, so its code is not this client's; but stopping it
            # now would end what it runs for someone. It is asked again on the next call.
            if _busy(running.url):
                return running
            _stop(running)

        setup_ms: dict[str, int] = {}

        @contextmanager
        def step(name: str) -> Generator[None]:
            if progress is not None:
                progress(name)
            started = time.monotonic()
            yield
            setup_ms[name] = round((time.monotonic() - started) * 1000)

        pid, url, session_id = _start(anchor, step)
        record = {
            "url": url,
            "pid": pid,
            "session_id": session_id,
            "version": current,
            "anchor": str(anchor),
        }
        # Whole or not at all: a torn record would stop every later client.
        path = _record_path(anchor)
        partial = path.with_suffix(".json.tmp")
        partial.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(partial, path)
        return Daemon(url=url, session_id=session_id, pid=pid, version=current, setup_ms=setup_ms)


def _registry_path() -> Path:
    return Path.home() / ".strata" / "scratchpads.json"


@contextmanager
def _registry() -> Generator[dict[str, dict[str, Any]]]:
    """The registry's entries, by anchor, written back on exit."""
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(f"{path}.lock"):
        try:
            entries = json.loads(path.read_text())
        except FileNotFoundError:
            entries = {}
        before = json.dumps(entries, sort_keys=True)
        yield entries
        if json.dumps(entries, sort_keys=True) != before:
            partial = path.with_suffix(".json.partial")
            partial.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
            os.replace(partial, path)


def register(anchor: Path, *, store_bytes: int, over_by_bytes: int, used: bool) -> None:
    """Record *anchor*'s scratchpad where ``strata scratch ls`` lists every one."""
    with _registry() as entries:
        entry = entries.setdefault(str(anchor), {})
        entry.update(store_bytes=store_bytes, over_by_bytes=over_by_bytes)
        if used or "last_used" not in entry:
            entry["last_used"] = time.time()


def registered() -> dict[str, dict[str, Any]]:
    """Every registered scratchpad, by anchor; one whose directory is gone is dropped."""
    with _registry() as entries:
        for anchor in [a for a in entries if not scratch_dir(Path(a)).is_dir()]:
            del entries[anchor]
        return dict(entries)


def unregister(anchor: Path) -> None:
    with _registry() as entries:
        entries.pop(str(anchor), None)


def stop_daemon(anchor: Path, timeout: float = 30.0) -> bool:
    """Stop *anchor*'s daemon if one answers; return whether one did."""
    running = find_daemon(anchor.resolve())
    if running is None:
        return False
    _stop(running, timeout)
    return True


def _stop(running: Daemon, timeout: float = 30.0) -> None:
    # It may stop itself, idle, in between.
    with contextlib.suppress(ProcessLookupError):
        os.kill(running.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while _healthy(running.url):
        if time.monotonic() > deadline:
            raise ScratchError(f"the scratchpad's server (pid {running.pid}) did not stop")
        time.sleep(0.1)
