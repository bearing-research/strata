"""Who a cell subprocess runs as, and whether it may run on this host at all.

The env allowlist filters what a cell is given, not who it is: as the server's
own user it could read ``/proc/<server pid>/environ``, the config and every
notebook on disk. So a service-mode server refuses to run cell code locally
unless cells go to a worker on another machine or ``notebook_harness_user``
names a user to drop to (the server then runs as root). Personal mode is
unaffected. The refusal happens where cell code would start, since the worker is
chosen per cell; a cache hit starts nothing and is never refused.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import logging
import os
import signal
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

REFUSAL = (
    "This server runs in service mode, where a cell run on the server's own host "
    "could read the server's credentials. Assign the cell a server-managed worker "
    "so it runs on another machine, or set STRATA_NOTEBOOK_HARNESS_USER so cells "
    "on this host run as a separate OS user."
)


class LocalExecutionRefused(RuntimeError):
    """Cell code may not start on this host; the message says what to change."""


@dataclass(frozen=True)
class HarnessUser:
    """The OS user a cell subprocess is started as."""

    name: str
    uid: int
    gid: int
    home: str


def running_server_config() -> Any | None:
    """The live server's config, or ``None`` outside a server.

    Not ``StrataConfig.load()`` as a fallback: its default mode is service, and a
    CLI run or test has no server credentials for a cell to read.
    """
    try:
        from strata.server import get_state

        return get_state().config
    except RuntimeError:
        return None


def resolve_harness_user(config: Any | None = None) -> HarnessUser | None:
    """Who to start a cell subprocess as, ``None`` for "as this process".

    Raises:
        LocalExecutionRefused: in service mode with no harness user, or when the
            configured user cannot be switched to here.
    """
    if config is None:
        config = running_server_config()
    if config is None:
        return None

    name = str(getattr(config, "notebook_harness_user", None) or "").strip()
    if not name:
        if getattr(config, "deployment_mode", None) == "service":
            raise LocalExecutionRefused(REFUSAL)
        return None

    if os.name == "nt":
        raise LocalExecutionRefused(
            "STRATA_NOTEBOOK_HARNESS_USER is not supported on Windows; run cells on a "
            "server-managed worker instead."
        )

    import pwd

    try:
        entry = pwd.getpwnam(name)
    except KeyError:
        raise LocalExecutionRefused(
            f"STRATA_NOTEBOOK_HARNESS_USER names {name!r}, which is not a user on this host."
        ) from None

    if entry.pw_uid != os.geteuid() and os.geteuid() != 0:
        # Popen(user=) would fail with a bare PermissionError naming nothing.
        raise LocalExecutionRefused(
            f"Running cells as {name!r} (STRATA_NOTEBOOK_HARNESS_USER) needs the server "
            "to run as root, so it can switch to that user."
        )
    return HarnessUser(name=name, uid=entry.pw_uid, gid=entry.pw_gid, home=entry.pw_dir)


def spawn_kwargs(user: HarnessUser | None) -> dict[str, Any]:
    """``user=`` / ``group=`` for ``create_subprocess_exec`` or ``subprocess.run``.

    When the server is root, supplementary groups are cleared too, or the cell would
    keep root's groups. Only root can clear them, and a server switching to its own
    user has none to shed.
    """
    if user is None:
        return {}
    kwargs: dict[str, Any] = {"user": user.uid, "group": user.gid}
    if os.geteuid() == 0:
        kwargs["extra_groups"] = []
    return kwargs


def identity_env(env: dict[str, str] | None, user: HarnessUser | None) -> dict[str, str] | None:
    """The spawn environment with ``HOME`` / ``USER`` / ``LOGNAME`` naming the user.

    ``Popen(user=)`` changes only the uid, so a cell would otherwise be told its home
    is root's.
    """
    if user is None:
        return env
    merged = dict(os.environ if env is None else env)
    merged.update({"HOME": user.home, "USER": user.name, "LOGNAME": user.name})
    return merged


def leftover_processes(
    table: dict[int, tuple[int, int, int]], server_pid: int, uid: int
) -> list[int]:
    """The processes of *uid* in *table* (pid -> (parent, uid, session)) no run holds.

    A run holds what descends from the server, or from one of its ancestors (another
    worker of the same server), and what is in the session of a live harness that
    does: ``cmd &`` from a shell a cell ran is orphaned but stays in its cell's
    session. A process that left it with ``setsid`` is held by nothing.
    """
    roots: set[int] = set()
    pid = server_pid
    while pid > 1 and pid in table and pid not in roots:
        roots.add(pid)
        pid = table[pid][0]

    memo: dict[int, bool] = {}

    def held(pid: int) -> bool:
        chain: list[int] = []
        while pid not in memo:
            if pid in roots:
                memo[pid] = True
            elif pid <= 1 or pid not in table or pid in chain:
                memo[pid] = False
            else:
                chain.append(pid)
                pid = table[pid][0]
        for seen in chain:
            memo[seen] = memo[pid]
        return memo[pid]

    return [
        pid
        for pid, (_parent, owner, session) in table.items()
        if owner == uid and not held(pid) and not (session in table and held(session))
    ]


def _process_table() -> dict[int, tuple[int, int, int]]:
    """Every live process's (parent, real uid, session), from ``/proc``."""
    table: dict[int, tuple[int, int, int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/status") as status:
                fields = dict(line.split(":", 1) for line in status if ":" in line)
        except OSError:
            continue  # exited while listed
        if fields["State"].split()[0] in ("Z", "X"):
            continue  # dead, waiting to be reaped
        table[int(name)] = (
            int(fields["PPid"]),
            int(fields["Uid"].split()[0]),
            int(fields["NSsid"].split()[0]),
        )
    return table


@functools.cache
def _cannot_sweep_as_init() -> None:
    logger.warning(
        "Strata runs as PID 1, so every orphaned process is its child and a process a "
        "cell left running cannot be told from a running one. It is not stopped; run "
        "the server under an init (docker run --init)."
    )


def sweep_leftovers(user: HarnessUser | None) -> None:
    """Stop every process the harness user has left running outside a live run.

    Called before a directory is handed to a run, so a process an earlier cell left
    behind cannot watch it. Any other process of that user is stopped too, which is
    why the harness user must serve this server's cells only. Linux only.
    """
    if user is None or user.uid in (0, os.geteuid()) or sys.platform != "linux":
        return
    if os.getpid() == 1:
        _cannot_sweep_as_init()
        return
    for _ in range(5):  # what a leftover forks while it is stopped is caught next pass
        pids = leftover_processes(_process_table(), os.getpid(), user.uid)
        if not pids:
            return
        logger.info("Stopping %d process(es) left running as %s", len(pids), user.name)
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    logger.warning("Processes left running as %s keep forking; stopped what was there", user.name)


def hand_over(path: Path, user: HarnessUser | None) -> None:
    """Give the harness user a per-run directory it has to write into.

    ``mkdtemp`` / ``TemporaryDirectory`` are private to their creator; the server
    keeps access either way because it is root.
    """
    if user is None or user.uid == os.geteuid():
        return
    sweep_leftovers(user)
    # By directory fd, never by path: the user can swap a directory it already owns for
    # a link between a path walk's link check and its descent. *path* goes last.
    for _root, dirs, files, root_fd in os.fwalk(path, follow_symlinks=False):
        for name in (*dirs, *files):
            os.chown(name, user.uid, user.gid, dir_fd=root_fd, follow_symlinks=False)
    os.chown(path, user.uid, user.gid, follow_symlinks=False)


def prepare_work_dir(notebook_dir: Path, user: HarnessUser | None) -> Path:
    """The notebook's ``work/``, made if missing and owned by the user cells run as.

    One left by an earlier, unisolated run may be a link or a cell's own directory:
    a link is refused, and only the directory changes owner, never what is in it.
    The notebook's directory stays the server's, so a cell cannot swap ``work/``.
    """
    from strata.notebook.layout import WORK_DIR

    work = notebook_dir / WORK_DIR
    with contextlib.suppress(FileExistsError):
        work.mkdir()
    mode = os.lstat(work).st_mode
    if not stat.S_ISDIR(mode):
        raise UnsafeRunFile(f"{work} is not a directory, so cells cannot run in it")
    if user is not None and user.uid != os.geteuid():
        sweep_leftovers(user)
        os.chown(work, user.uid, user.gid, follow_symlinks=False)
    return work


def let_through(path: Path, user: HarnessUser | None) -> None:
    """Let the harness user reach entries of *path* by name without owning it.

    Owning it would let a process an earlier cell left running plant a link where the
    server is about to write.
    """
    if user is None or user.uid == os.geteuid():
        return
    os.chown(path, -1, user.gid)
    os.chmod(path, 0o710)


class UnsafeRunFile(RuntimeError):
    """A run-directory entry is not a plain file, so the server will not touch it."""


# ELOOP is a symlink under O_NOFOLLOW (EMLINK on FreeBSD); ENOTDIR a non-directory.
_LINK_ERRNOS = frozenset({errno.ELOOP, errno.EMLINK, errno.ENOTDIR})


def _refuse(name: str) -> UnsafeRunFile:
    return UnsafeRunFile(
        f"Refusing cell output {name!r}: it is a symlink or not a regular file in the "
        "cell's run directory."
    )


def _check_name(name: str) -> None:
    if not name or name in (".", "..") or any(c in name for c in "/\\\x00"):
        raise UnsafeRunFile(f"Refusing cell output {name!r}: not a plain file name.")


def _open_dir(directory: Path, name: str) -> int:
    try:
        return os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno in _LINK_ERRNOS:
            raise _refuse(name) from exc
        raise


def _lstat_checked(path: Path, name: str, want: int) -> None:
    """The no-O_NOFOLLOW (Windows) fallback: refuse a link or the wrong kind of entry."""
    st = os.lstat(path)
    if stat.S_IFMT(st.st_mode) != want:
        raise _refuse(name)


def open_run_file(directory: Path, name: str) -> BinaryIO:
    """Open *name* in a handed-over run directory for reading, following no link.

    The harness user owns that directory, so it can swap a file (or the directory)
    for a symlink or hard link to something only the server may read.
    """
    _check_name(name)
    if not hasattr(os, "O_NOFOLLOW"):
        _lstat_checked(directory, name, stat.S_IFDIR)
        _lstat_checked(directory / name, name, stat.S_IFREG)
        return open(directory / name, "rb")
    dir_fd = _open_dir(directory, name)
    try:
        return _open_at(dir_fd, name, name)
    finally:
        os.close(dir_fd)


def open_tree_file(root: Path, rel: str) -> BinaryIO:
    """Open the file at relative POSIX path *rel* under *root*, following no link on the way.

    For a tree the harness user writes, such as a read-write mount's staging copy: it can
    swap any entry for a link to something only the server may read.
    """
    parts = rel.split("/")
    for part in parts:
        _check_name(part)
    if not hasattr(os, "O_NOFOLLOW"):
        current = root
        _lstat_checked(current, rel, stat.S_IFDIR)
        for part in parts[:-1]:
            current = current / part
            _lstat_checked(current, rel, stat.S_IFDIR)
        _lstat_checked(current / parts[-1], rel, stat.S_IFREG)
        return open(current / parts[-1], "rb")
    dir_fd = _open_dir(root, rel)
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
            except OSError as exc:
                if exc.errno in _LINK_ERRNOS:
                    raise _refuse(rel) from exc
                raise
            os.close(dir_fd)
            dir_fd = next_fd
        return _open_at(dir_fd, parts[-1], rel)
    finally:
        os.close(dir_fd)


def _open_at(dir_fd: int, name: str, shown: str) -> BinaryIO:
    """Open regular file *name* in the open directory *dir_fd*, no link, one hard link."""
    try:
        # O_NONBLOCK so a planted FIFO cannot hang the open.
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in _LINK_ERRNOS:
            raise _refuse(shown) from exc
        raise
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        os.close(fd)
        raise _refuse(shown)
    return os.fdopen(fd, "rb")


def read_run_file(directory: Path, name: str) -> bytes:
    """The bytes of *name* in a handed-over run directory (see ``open_run_file``)."""
    with open_run_file(directory, name) as f:
        return f.read()


def write_run_file(directory: Path, name: str, data: bytes) -> None:
    """Write *name* into a run directory the harness user can also write to.

    Whatever is there is unlinked and the file created exclusively, so a planted
    link is replaced, never written through.
    """
    _check_name(name)
    if not hasattr(os, "O_NOFOLLOW"):
        _lstat_checked(directory, name, stat.S_IFDIR)
        (directory / name).unlink(missing_ok=True)
        with open(directory / name, "xb") as f:
            f.write(data)
        return
    dir_fd = _open_dir(directory, name)
    try:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name, dir_fd=dir_fd)
        fd = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o666, dir_fd=dir_fd
        )
    finally:
        os.close(dir_fd)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
