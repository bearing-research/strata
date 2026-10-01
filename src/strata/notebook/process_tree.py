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


async def terminate_subprocess_tree(
    proc: asyncio.subprocess.Process,
    *,
    grace_seconds: float = 2.0,
) -> None:
    """SIGTERM, grace period, then SIGKILL the subprocess and its descendants.

    ``proc`` must have been spawned with :py:func:`subprocess_kwargs_for_new_group`,
    or only the direct child is signalled. Returns once the process is reaped;
    a process that exits mid-way (``ProcessLookupError``) counts as success.
    """
    if proc.returncode is not None:
        return  # already exited cleanly

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
        await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
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
