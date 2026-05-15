"""Process-tree-aware subprocess termination.

The default :py:meth:`asyncio.subprocess.Process.kill` sends SIGKILL to
the direct child only. When that child has spawned its own descendants
(PyTorch DataLoader workers via ``multiprocessing``, GPU streams,
fork-server workers, …), those descendants get reparented to PID 1 and
keep running. We've seen this leak ~150 MB of orphaned DataLoader
workers on cancelled training cells.

This module wraps the two operations correctly:

- :py:func:`subprocess_kwargs_for_new_group` returns the kwargs that
  put the child in its own process group on the current platform.
  Callers thread these into ``asyncio.create_subprocess_exec``.
- :py:func:`terminate_subprocess_tree` sends SIGTERM to the whole
  group, waits a brief grace period, then SIGKILL the group if any
  descendant survived. Standard SIGTERM-then-SIGKILL escalation that
  Docker, k8s, and systemd use.

The harness's own ``finally`` block runs during the grace period if
the user code is at an await point; if it's blocked in C-level code
(numpy / torch loops), SIGKILL after the grace period still wins.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Any

logger = logging.getLogger(__name__)


def subprocess_kwargs_for_new_group() -> dict[str, Any]:
    """Spawn kwargs that put the child into its own process group.

    POSIX: ``start_new_session=True`` makes the child a session leader
    (and therefore a process-group leader). Equivalent to calling
    ``os.setsid`` in a preexec hook.

    Windows: ``CREATE_NEW_PROCESS_GROUP`` does the equivalent for
    Win32 console processes, so ``CTRL_BREAK_EVENT`` can target the
    whole group rather than just the direct child.
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
    """SIGTERM → grace period → SIGKILL the subprocess and its descendants.

    Requires ``proc`` to have been spawned with the kwargs from
    :py:func:`subprocess_kwargs_for_new_group`. Without that, signals
    only reach the direct child and descendants leak (which is exactly
    the bug this module exists to fix).

    Returns once the process is reaped. Tolerates races: if the process
    exits between any of our calls, ``ProcessLookupError`` is caught
    and treated as success.
    """
    if proc.returncode is not None:
        return  # already exited cleanly

    pid = proc.pid
    if pid is None:
        # asyncio.Process should always have a pid after spawn; if not,
        # there's nothing meaningful to do.
        try:
            await proc.wait()
        except Exception:
            pass
        return

    # ---- Stage 1: graceful termination ----
    try:
        if sys.platform == "win32":
            # CTRL_BREAK_EVENT reaches the new process group (only works
            # when CREATE_NEW_PROCESS_GROUP was set on spawn).
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        # Raced with natural exit; nothing to wait on.
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
        # Any other failure waiting falls through to force-kill.
        logger.exception("Unexpected error waiting for subprocess pid=%s", pid)

    # ---- Stage 2: force-kill the group ----
    try:
        if sys.platform == "win32":
            # No process-group SIGKILL on Windows; the best we can do is
            # terminate the direct child. Descendants spawned via subprocess
            # without job-object containment may still leak. This is the
            # standard trade-off; full Windows process-tree termination
            # would need an explicit Win32 Job Object.
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
    """Synchronous best-effort SIGKILL of the process group.

    Used by shutdown_nowait paths that can't ``await``. No grace
    period, no reap: the caller is on the way down anyway. Tolerates
    races (process already gone) silently.
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
