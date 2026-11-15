"""A project's scratchpad, ``<anchor>/.strata-scratch/``, and the one daemon serving it.

The first client to need it starts the daemon (a personal-mode server with the
scratchpad MCP profile, on a free loopback port, that exits when idle), opens the
scratchpad in it and writes ``server.json``; later clients find it there. A file
lock makes find-or-start one step, so two agents in one project share one daemon
and never write one directory from two sessions.

Imports stay light (no ``strata.notebook`` on the path that finds a running
daemon): a stdio bridge calls this on every start.
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

import filelock
import httpx

SCRATCH_DIR = ".strata-scratch"
IDLE_MINUTES = 30
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


@dataclass(frozen=True)
class Daemon:
    """A running daemon and the scratchpad session it holds open."""

    url: str
    session_id: str
    pid: int
    # Milliseconds per setup step, in order; empty when the daemon was already up.
    setup_ms: dict[str, int] = field(default_factory=dict)


def scratch_dir(anchor: Path) -> Path:
    return anchor / SCRATCH_DIR


def _record_path(anchor: Path) -> Path:
    return scratch_dir(anchor) / ".strata" / "server.json"


def _environment(url: str, session_id: str) -> str | None:
    """The session's environment sync state, or None if the session is not open there."""
    try:
        response = httpx.get(f"{url}/v1/notebooks/{session_id}/environment", timeout=5.0)
    except httpx.HTTPError:
        return None
    if not response.is_success:
        return None
    return response.json()["environment"]["sync_state"]


def find_daemon(anchor: Path) -> Daemon | None:
    """The daemon ``server.json`` names, if it answers and still holds the session.

    Asked, not judged by its pid: a session id is never reused, so no other server
    can answer for it.
    """
    try:
        record = json.loads(_record_path(anchor).read_text())
    except FileNotFoundError:
        return None
    if _environment(record["url"], record["session_id"]) is None:
        return None
    return Daemon(url=record["url"], session_id=record["session_id"], pid=record["pid"])


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
    env = {
        **os.environ,
        "STRATA_HOST": "127.0.0.1",
        "STRATA_PORT": str(port),
        "STRATA_DEPLOYMENT_MODE": "personal",
        "STRATA_MCP_ENABLED": "true",
        "STRATA_MCP_PROFILE": "scratchpad",
        # The server opens notebooks only inside its storage dir.
        "STRATA_NOTEBOOK_STORAGE_DIR": str(anchor),
        "STRATA_IDLE_EXIT_MINUTES": os.environ.get("STRATA_IDLE_EXIT_MINUTES", str(IDLE_MINUTES)),
    }
    log = scratch_dir(anchor) / ".strata" / "server.log"
    launched = subprocess.run(
        [sys.executable, "-I", "-c", _LAUNCH, str(log), *_server_command()],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(launched.stdout)


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
    return True


def _until(ready: Callable[[], bool], pid: int, what: str, deadline: float) -> None:
    while not ready():
        if not _alive(pid):
            raise ScratchError(f"the scratchpad's server exited while {what}")
        if time.monotonic() > deadline:
            raise ScratchError(f"the scratchpad's server timed out {what}")
        time.sleep(0.1)


def _healthy(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=2.0).is_success
    except httpx.HTTPError:
        return False


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
    try:
        with step("starting the scratchpad"):
            _until(lambda: _healthy(url), pid, "starting", deadline)
        with step("opening the scratchpad"):
            opened = httpx.post(
                f"{url}/v1/notebooks/open", json={"path": str(scratch_dir(anchor))}, timeout=60.0
            )
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
            )
    except BaseException:
        # Not left running for nobody: no server.json names it. Gone already raises
        # ProcessLookupError, or on Windows a plain OSError.
        with contextlib.suppress(OSError):
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        raise
    return pid, url, session_id


def ensure_daemon(anchor: Path, progress: Callable[[str], None] | None = None) -> Daemon:
    """The daemon for *anchor*'s scratchpad, started (and the scratchpad created) if none runs.

    *progress* hears each setup step as it starts.
    """
    anchor = anchor.resolve()
    (scratch_dir(anchor) / ".strata").mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(scratch_dir(anchor) / ".strata" / "daemon.lock"):
        running = find_daemon(anchor)
        if running is not None:
            return running

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
            "version": version("strata-notebook"),
            "anchor": str(anchor),
        }
        _record_path(anchor).write_text(json.dumps(record, indent=2) + "\n")
        return Daemon(url=url, session_id=session_id, pid=pid, setup_ms=setup_ms)


def stop_daemon(anchor: Path, timeout: float = 30.0) -> bool:
    """Stop *anchor*'s daemon if one answers; return whether one did."""
    running = find_daemon(anchor.resolve())
    if running is None:
        return False
    os.kill(running.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while _healthy(running.url):
        if time.monotonic() > deadline:
            raise ScratchError(f"the scratchpad's server (pid {running.pid}) did not stop")
        time.sleep(0.1)
    return True
