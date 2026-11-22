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
