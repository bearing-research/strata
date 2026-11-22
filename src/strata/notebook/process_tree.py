"""Process-tree-aware subprocess termination.

``Process.kill`` signals only the direct child, so its descendants (DataLoader
workers, fork-server workers) are reparented to PID 1 and keep running. Spawn
with :py:func:`subprocess_kwargs_for_new_group` and stop with
:py:func:`terminate_subprocess_tree` to signal the whole group instead.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Line limit for the newline-delimited JSON the notebook reads from its subprocesses.
# asyncio's 64 KiB default raises on longer lines, and harness frames embed full stdout
# and base64 display payloads. 256 MiB is a ceiling, not an allocation.
SUBPROCESS_LINE_LIMIT = 256 * 1024 * 1024


def subprocess_kwargs_for_new_group() -> dict[str, Any]:
    """Spawn kwargs that put the child into its own process group.

    POSIX uses ``start_new_session=True``; Windows uses
    ``CREATE_NEW_PROCESS_GROUP`` so ``CTRL_BREAK_EVENT`` reaches the whole group.
    """
    if sys.platform == "win32":
        import subprocess as _subprocess

        return {"creationflags": _subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


# Held for the process's life: when it closes, at exit however that comes, Windows ends
# every process left in the job (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE).
_job: int | None = None


def end_children_with_this_process() -> None:
    """On Windows, end every process this one starts, and theirs, when it exits.

    POSIX has the lifeline and the process group for that; Windows has neither, so a
    server terminated (``strata scratch stop``, a crash) left a running cell's harness
    and what it started running on. The server joins a job of its own that ends them as
    it closes: its handle is the only one, and the server's exit closes it. A process
    started with ``CREATE_BREAKAWAY_FROM_JOB`` leaves the job, as one a cell detaches on
    POSIX (``setsid``) leaves the group; an application a cell opens (``webbrowser``,
    ``os.startfile``) that was not already running does not, and ends with the server.
    A no-op elsewhere, and once done.
    """
    global _job
    if sys.platform != "win32" or _job is not None:
        return
    import ctypes
    from ctypes import wintypes

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64)
            for name in ("Read", "Write", "Other", "ReadBytes", "WriteBytes", "OtherBytes")
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        logger.warning("No job for this process's children: error %d", ctypes.get_last_error())
        return
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE | _BREAKAWAY_OK
    if not kernel32.SetInformationJobObject(
        wintypes.HANDLE(job),
        _EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ) or not kernel32.AssignProcessToJobObject(
        wintypes.HANDLE(job), wintypes.HANDLE(kernel32.GetCurrentProcess())
    ):
        # In a job that forbids it (Windows 7 nests none): its children outlive it, as before.
        logger.warning(
            "This process's children will outlive it: its job refused: error %d",
            ctypes.get_last_error(),
        )
        kernel32.CloseHandle(wintypes.HANDLE(job))
        return
    _job = job


_KILL_ON_JOB_CLOSE = 0x2000
_BREAKAWAY_OK = 0x800
_EXTENDED_LIMIT_INFORMATION = 9


# How often Windows is sent the Ctrl-Break again while the process lives.
_CTRL_BREAK_EVERY = 0.25


async def _exit(proc: asyncio.subprocess.Process) -> None:
    """Wait for *proc* to exit; on Windows, send its group Ctrl-Break again meanwhile.

    Once is not enough there: a venv's ``python.exe`` is a launcher that ignores the
    event for the interpreter it starts, and one sent before that interpreter is up
    reaches nothing.
    """
    if sys.platform != "win32":
        await proc.wait()
        return
    while True:
        try:
            await asyncio.wait_for(proc.wait(), timeout=_CTRL_BREAK_EVERY)
            return
        except TimeoutError:
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except OSError:
                # Exited as the wait gave up, or no console to send through: the wait says.
                continue


async def terminate_subprocess_tree(
    proc: asyncio.subprocess.Process,
    *,
    grace_seconds: float = 2.0,
) -> None:
    """SIGTERM, grace period, then SIGKILL the subprocess and its descendants.

    ``proc`` must have been spawned with :py:func:`subprocess_kwargs_for_new_group`,
    or only the direct child is signalled. Returns once the process is reaped and
    our ends of its pipes are closed; a process that exits mid-way
    (``ProcessLookupError``) counts as success.
    """
    if proc.returncode is not None:
        return  # already exited cleanly
    try:
        await _reap(proc, grace_seconds)
    finally:
        if proc.returncode is not None:
            await _close_pipes(proc)


async def _close_pipes(proc: asyncio.subprocess.Process) -> None:
    """Close our ends of a reaped process's pipes.

    asyncio closes them once the other ends close, and a descendant that outlives the
    process holds those open: on Windows a venv launcher's interpreter, for a moment. A
    loop closed in that moment finds the transport unclosed.
    """
    # asyncio gives no public way to the transport.
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        transport.close()
        # One turn of the loop: the pipe transports close in callbacks.
        await asyncio.sleep(0)


async def _reap(proc: asyncio.subprocess.Process, grace_seconds: float) -> None:
    pid = proc.pid
    if pid is None:
        try:
            await proc.wait()
        except Exception:
            pass
        return

    # Stage 1: graceful termination.
    try:
        if sys.platform == "win32":
            # Reaches the new process group only if CREATE_NEW_PROCESS_GROUP was set on spawn.
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        # Raced with natural exit.
        return
    except OSError as exc:
        logger.warning(
            "SIGTERM to subprocess group pid=%s failed: %s; falling through to SIGKILL",
            pid,
            exc,
        )

    try:
        await asyncio.wait_for(_exit(proc), timeout=grace_seconds)
        return  # graceful shutdown succeeded
    except TimeoutError:
        logger.info(
            "Subprocess pid=%s did not exit within %.1fs of SIGTERM; sending SIGKILL",
            pid,
            grace_seconds,
        )
    except Exception:
        # Any other wait failure falls through to force-kill.
        logger.exception("Unexpected error waiting for subprocess pid=%s", pid)

    # Stage 2: force-kill the group.
    try:
        if sys.platform == "win32":
            # No process-group SIGKILL on Windows: kill only the direct child. Descendants may
            # leak; full tree termination would need a Win32 Job Object.
            proc.kill()
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError as exc:
        logger.warning("SIGKILL to subprocess group pid=%s failed: %s", pid, exc)
        return

    try:
        await proc.wait()
    except Exception:
        logger.exception("Failed to reap subprocess pid=%s after SIGKILL", pid)


def kill_subprocess_tree_nowait(proc: asyncio.subprocess.Process) -> None:
    """Synchronous best-effort SIGKILL of the process group, for paths that can't await.

    No grace period and no reap; a process that is already gone is ignored.
    """
    if proc.returncode is not None:
        return
    pid = proc.pid
    if pid is None:
        return
    try:
        if sys.platform == "win32":
            proc.kill()
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        return
