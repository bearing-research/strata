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
import threading
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
# WaitForSingleObject's "signaled".
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_ERROR_ACCESS_DENIED = 5
_WAIT_OBJECT_0 = 0


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
    # The interpreter it was told to run cells in (``--python``), else ``None``.
    python: str | None = None


def scratch_dir(anchor: Path) -> Path:
    return anchor / SCRATCH_DIR


def find_anchor(start: Path) -> Path:
    """The project *start* is in: the nearest directory at or above it holding a
    ``pyproject.toml``, else the git root, else *start*.

    Searched no higher than the git root, so a monorepo's ``services/api/`` with its own
    ``pyproject.toml`` is its own project and nothing above the repository is one, nor
    higher than the home directory when *start* is in it, whichever is nearer. A start in
    a scratchpad (its ``work/``, say) is in the scratchpad's project.
    """
    start = start.resolve()
    # A notebook (a scratchpad too) has a pyproject.toml of its own and is no project.
    inside = [
        d
        for d in [start, *start.parents]
        if d.name == SCRATCH_DIR or (d / "notebook.toml").is_file()
    ]
    if inside:
        start = inside[-1].parent
    try:
        home: Path | None = Path.home().resolve()
    except RuntimeError:
        # No HOME and no passwd entry, as in a container run under an arbitrary uid.
        home = None
    # Up to the nearer of the git root and home: the first met.
    for directory in [start, *start.parents]:
        if (directory / "pyproject.toml").is_file():
            return directory
        if (directory / ".git").exists():
            # A repository with no Python project, from any of its directories, is one;
            # never home, which a dotfiles repository would make every directory's.
            return directory if directory != home else start
        if directory == home:
            return start
    return start


def own_scratchpad_note(anchor: Path) -> str:
    """A note for a command that found nothing at *anchor* when the current directory has
    a scratchpad of its own (made by ``--root``, or by a version that anchored there)."""
    here = Path.cwd().resolve()
    if here != anchor and (scratch_dir(here) / "notebook.toml").is_file():
        return f"; {here} has a scratchpad of its own: --root ."
    return ""


def resolve_anchor(root: str | None) -> Path:
    """The anchor a command serves: *root* (its ``--root``), else ``STRATA_SCRATCH_ROOT``, as
    given; else the project the current directory is in (:func:`find_anchor`)."""
    # An empty --root is the current directory, as a path; an empty variable is unset.
    given = root if root is not None else os.environ.get("STRATA_SCRATCH_ROOT") or None
    if given is not None:
        return Path(given).expanduser().resolve()
    return find_anchor(Path.cwd())


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
        python=record.get("python"),
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
        scratchpad=True,
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


def _as_itself(command: list[str]) -> tuple[list[str], dict[str, str]]:
    """*command* and the environment under which the process started is this interpreter.

    A venv's ``python.exe`` on Windows is a launcher, and the interpreter its child: a kill
    or a wait on the pid started is then not one on the server, which holds its stores'
    files a moment longer. The base interpreter told which venv it is in, as multiprocessing
    starts its workers there, is the server itself.
    """
    # Not in typeshed; every CPython since 3.8 has it.
    base = getattr(sys, "_base_executable", sys.executable)
    same = os.path.normcase
    if (
        sys.platform != "win32"
        or same(command[0]) != same(sys.executable)
        or same(base) == same(sys.executable)
    ):
        return command, {}
    return [base, *command[1:]], {"__PYVENV_LAUNCHER__": sys.executable}


def _spawn(anchor: Path, port: int, python: str | None = None) -> int:
    """Start the daemon detached; return its pid."""
    from vestigo.file_modes import private_file

    strata_home = Path.home() / ".strata"
    env = {
        # The caches a personal server keeps, not ones beside the storage dir, which is
        # the project itself here. The environment may still say otherwise.
        "STRATA_NOTEBOOK_MOUNT_CACHE_DIR": str(strata_home / "mount_cache"),
        "STRATA_NOTEBOOK_SHARED_ENV_DIR": str(strata_home / "envs"),
        # uv's default off macOS. There it clones, and macOS checks each new binary on
        # its first load: ~3 s of pyarrow in the first cell, against 0.8 s for a file
        # shared with uv's cache.
        "UV_LINK_MODE": "hardlink",
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
        **({"STRATA_SCRATCH_PYTHON": python} if python is not None else {}),
    }
    command, as_venv = _as_itself(_server_command())
    env.update(as_venv)
    try:
        log = private_file(_log_path(anchor))
        launched = subprocess.run(
            [sys.executable, "-I", "-c", _LAUNCH, str(log), *command],
            env=env,
            # Its [tool.strata] is then the scratchpad's (none), not whichever
            # pyproject.toml sits above where the first client ran.
            cwd=scratch_dir(anchor),
            capture_output=True,
            text=True,
            check=True,
        )
        pid = int(launched.stdout)
        _note(pid)
        return pid
    except subprocess.CalledProcessError as exc:
        raise ScratchError(f"the scratchpad's server could not be launched: {exc.stderr}") from exc
    except (OSError, ValueError) as exc:
        raise ScratchError(f"the scratchpad's server could not be launched: {exc}") from exc


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) would terminate it there. Its handle is signaled once it has let
        # go of its files; its exit code is set some milliseconds before that.
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            return kernel32.WaitForSingleObject(handle, 0) != _WAIT_OBJECT_0
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Another user's process: it exists.
        return True
    if sys.platform == "linux":
        # A zombie has exited, though nobody reaped it (PID 1 in a container without init).
        # What /proc does not show (hidden, gone, unreadable) leaves os.kill's answer.
        with contextlib.suppress(OSError):
            # Bytes: the name before ")" is whatever the process set.
            fields = Path(f"/proc/{pid}/stat").read_bytes().rpartition(b")")[2].split()
            # "X" is a task being reaped; kernels 2.6.33 to 3.13 may show it as "x".
            if fields and fields[0] in (b"Z", b"X", b"x"):
                return False
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


# Windows gives a dead process's pid to the next one, and a kill by pid there ends
# whichever holds it. A daemon ``_spawn`` started is known by when it was created too.
_created: dict[int, int] = {}


def _created_at(kernel32: Any, handle: int) -> int | None:
    """When the process behind *handle* was created, in Windows's 100 ns ticks."""
    import ctypes
    from ctypes import wintypes

    made, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
    if not kernel32.GetProcessTimes(
        handle, ctypes.byref(made), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None
    return (made.dwHighDateTime << 32) | made.dwLowDateTime


def _note(pid: int) -> None:
    """Remember the daemon just started at *pid* as the process :func:`_kill` may end."""
    if sys.platform != "win32":
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Gone already: no process that takes the pid is it.
        _created[pid] = 0
        return
    try:
        _created[pid] = _created_at(kernel32, handle) or 0
    finally:
        kernel32.CloseHandle(handle)


def _kill(pid: int) -> None:
    """Kill the daemon and what it started: it leads its own session, so its group."""
    if sys.platform != "win32":
        # Gone already raises ProcessLookupError.
        with contextlib.suppress(OSError):
            os.killpg(pid, signal.SIGKILL)
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(
        _PROCESS_TERMINATE | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not handle:
        return  # gone already
    try:
        # One handle for the check and the kill: the pid cannot change hands between.
        noted = _created.get(pid)
        if noted is None or _created_at(kernel32, handle) == noted:
            # The exit code os.kill(pid, SIGTERM) leaves there.
            kernel32.TerminateProcess(handle, int(signal.SIGTERM))
    finally:
        kernel32.CloseHandle(handle)


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


class SetupCancel:
    """Set from another thread to stop a setup: the daemon it has started, not yet
    recorded, is killed (and the setup fails); one it has not started never starts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._set = False
        self._pid: int | None = None

    def set(self) -> None:
        with self._lock:
            self._set = True
            pid, self._pid = self._pid, None
        if pid is not None:
            _kill(pid)

    def is_set(self) -> bool:
        return self._set

    def started(self, pid: int) -> None:
        with self._lock:
            if not self._set:
                self._pid = pid
                return
        _kill(pid)

    def finished(self) -> bool:
        """Done with the daemon: whether it is still running, not killed by a cancel."""
        with self._lock:
            held, self._pid = self._pid is not None, None
            return held or not self._set


def _cancelled() -> ScratchError:
    return ScratchError("cancelled before the scratchpad's server started")


def _start(
    anchor: Path,
    step: Callable[[str], contextlib.AbstractContextManager[None]],
    python: str | None = None,
    cancel: SetupCancel | None = None,
) -> tuple[int, str, str]:
    """Start a daemon on *anchor* and open its scratchpad; return it, its url and session."""
    deadline = time.monotonic() + _SETUP_TIMEOUT
    if not (scratch_dir(anchor) / "notebook.toml").exists():
        with step("creating the scratchpad"):
            _scaffold(anchor)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    if cancel is not None and cancel.is_set():
        raise _cancelled()
    pid = _spawn(anchor, port, python)
    log = _log_path(anchor)
    try:
        if cancel is not None:
            # Killed from here on by a cancel: what the steps wait on then fails.
            cancel.started(pid)
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
    except BaseException as exc:
        # Not left running for nobody: no server.json names it.
        _kill(pid)
        if cancel is not None:
            cancelled = not cancel.finished()
            if cancelled and isinstance(exc, ScratchError):
                # Its wait failed because the cancel killed it, not as a crash.
                raise _cancelled() from exc
        raise
    return pid, url, session_id


def _interpreter(python: str | None) -> str:
    return python if python is not None else "the environment it found"


def daemon_lock(anchor: Path) -> filelock.FileLock:
    """Held while a daemon is started: held too, none starts behind a writer of the files."""
    return filelock.FileLock(scratch_dir(anchor) / ".strata" / "daemon.lock")


@contextmanager
def _held(lock: filelock.FileLock, cancel: SetupCancel | None) -> Generator[None]:
    """*lock*, polled for, so a cancel ends the wait for another client's setup."""
    while True:
        if cancel is not None and cancel.is_set():
            raise _cancelled()
        try:
            lock.acquire(timeout=0.2)
            break
        except filelock.Timeout:
            continue
    try:
        yield
    finally:
        lock.release()


def ensure_daemon(
    anchor: Path,
    progress: Callable[[str], None] | None = None,
    cancel: SetupCancel | None = None,
) -> Daemon:
    """The daemon for *anchor*'s scratchpad, started (and the scratchpad created) if none runs.

    *progress* hears each setup step as it starts; *cancel*, set, ends the wait for the
    lock and the setup, with no daemon left that no record names.
    """
    anchor = anchor.resolve()
    try:
        (scratch_dir(anchor) / ".strata").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ScratchError(f"the scratchpad cannot be created in {anchor}: {exc}") from exc
    with _held(daemon_lock(anchor), cancel):
        running = find_daemon(anchor)
        current = version("strata-notebook")
        python = os.environ.get("STRATA_SCRATCH_PYTHON") or None
        if python is None and running is not None:
            # Not asked for, the interpreter the running daemon was given stays.
            python = running.python
        if running is not None and (running.version, running.python) == (current, python):
            return running
        if running is not None:
            # Started by another install, or on another interpreter; but stopping it now
            # would end what it runs for someone. It is asked again on the next call.
            if _busy(running.url):
                if running.python != python:
                    # Its cells would run on the other interpreter, keyed as if on this one.
                    raise ScratchError(
                        f"the scratchpad's daemon runs cells on {_interpreter(running.python)}, "
                        f"not {_interpreter(python)}, and it is busy, so it was not replaced; "
                        "try again once its run finishes, or `strata scratch stop` it"
                    )
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

        pid, url, session_id = _start(anchor, step, python, cancel)
        if cancel is not None and not cancel.finished():
            # Killed by a cancel as it finished: not recorded.
            raise _cancelled()
        # Recorded next: a cancel from now on leaves it running, named.
        record = {
            "url": url,
            "pid": pid,
            "session_id": session_id,
            "version": current,
            "anchor": str(anchor),
            "python": python,
        }
        # Whole or not at all: a torn record would stop every later client.
        path = _record_path(anchor)
        partial = path.with_suffix(".json.tmp")
        partial.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(partial, path)
        return Daemon(
            url=url,
            session_id=session_id,
            pid=pid,
            version=current,
            setup_ms=setup_ms,
            python=python,
        )


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
    # Until its process is gone, not only its port: on POSIX its shutdown still closes the
    # sessions and their stores; on Windows, terminated, it holds their files until it exits.
    while _healthy(running.url) or _alive(running.pid):
        if time.monotonic() > deadline:
            raise ScratchError(f"the scratchpad's server (pid {running.pid}) did not stop")
        time.sleep(0.1)
