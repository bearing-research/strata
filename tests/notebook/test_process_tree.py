"""Tests for process-tree-aware subprocess termination.

SIGTERM returns promptly, SIGKILL wins over a process that ignores it, and the
whole tree dies, not just the direct child. Windows (plain ``proc.kill``) is
not tested here.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

import pytest

from strata.notebook.process_tree import (
    kill_subprocess_tree_nowait,
    subprocess_kwargs_for_new_group,
    terminate_subprocess_tree,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-specific signal semantics; Windows path is best-effort fallback",
)


def _is_alive(pid: int) -> bool:
    """True if pid still exists (signal 0)."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


@pytest.mark.asyncio
async def test_graceful_termination_via_sigterm():
    """A SIGTERM-respecting process exits within the grace period, with no SIGKILL."""
    # Python with a signal handler that exits cleanly on SIGTERM.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **subprocess_kwargs_for_new_group(),
    )
    pid = proc.pid

    # Let the child install its handler.
    await asyncio.sleep(0.2)

    await terminate_subprocess_tree(proc, grace_seconds=2.0)

    # Exited via its SIGTERM handler (sys.exit(0)): the graceful path, not the
    # SIGKILL-after-grace fallback (which would leave returncode == -SIGKILL).
    assert proc.returncode == 0
    assert not _is_alive(pid)


@pytest.mark.asyncio
async def test_sigkill_fallback_when_sigterm_ignored():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **subprocess_kwargs_for_new_group(),
    )
    pid = proc.pid
    await asyncio.sleep(0.2)

    await terminate_subprocess_tree(proc, grace_seconds=0.5)

    # SIGTERM was ignored, so only the SIGKILL fallback could have killed it;
    # proven by the return code, not by timing the grace period.
    assert proc.returncode == -signal.SIGKILL
    assert not _is_alive(pid)


@pytest.mark.asyncio
async def test_terminates_child_processes_not_just_parent():
    """Children die with the parent, as DataLoader workers must with the harness.

    Without the new process group and ``killpg`` they survive as orphans of PID 1.
    """
    # Parent prints its own pid and the pids of two children, then
    # all three sleep. We read the pids off stdout, terminate the
    # parent's tree, and verify all three pids are gone.
    script = (
        "import os, sys, time\n"
        "child_pids = []\n"
        "for _ in range(2):\n"
        "    pid = os.fork()\n"
        "    if pid == 0:\n"
        "        time.sleep(60)\n"  # child: just wait
        "        sys.exit(0)\n"
        "    child_pids.append(pid)\n"
        "print(os.getpid(), child_pids[0], child_pids[1], flush=True)\n"
        "time.sleep(60)\n"  # parent: also wait
    )

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **subprocess_kwargs_for_new_group(),
    )
    assert proc.stdout is not None

    # First line carries the three pids.
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
    parent_pid, child_a, child_b = (int(p) for p in line.decode().split())

    # Sanity: all three are alive before we terminate.
    assert _is_alive(parent_pid)
    assert _is_alive(child_a)
    assert _is_alive(child_b)

    await terminate_subprocess_tree(proc, grace_seconds=2.0)

    # Wait briefly for the kernel to actually reap the descendants
    # (killpg returns before exit is processed by the kernel).
    for _ in range(20):
        if not (_is_alive(parent_pid) or _is_alive(child_a) or _is_alive(child_b)):
            break
        await asyncio.sleep(0.1)

    assert not _is_alive(parent_pid), "parent still running"
    assert not _is_alive(child_a), "child A leaked — process-group kill didn't reach it"
    assert not _is_alive(child_b), "child B leaked — process-group kill didn't reach it"


@pytest.mark.asyncio
async def test_returns_immediately_for_exited_process():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "pass",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **subprocess_kwargs_for_new_group(),
    )
    await proc.wait()
    assert proc.returncode == 0

    await terminate_subprocess_tree(proc, grace_seconds=2.0)
    # No-op for an already-exited process: its clean exit code is untouched
    # (and the per-test timeout guards against a regression that blocks here).
    assert proc.returncode == 0


@pytest.mark.asyncio
async def test_kill_subprocess_tree_nowait_kills_tree():
    """The sync variant used by ``shutdown_nowait`` also reaches descendants."""
    script = (
        "import os, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(60)\n"
        "    sys.exit(0)\n"
        "print(os.getpid(), pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **subprocess_kwargs_for_new_group(),
    )
    assert proc.stdout is not None
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
    parent_pid, child_pid = (int(p) for p in line.decode().split())

    kill_subprocess_tree_nowait(proc)
    # Reap the parent so the wait state isn't ambiguous.
    await proc.wait()

    # The sync variant uses SIGKILL; children should be gone within the kernel's
    # signal-delivery window.
    for _ in range(20):
        if not (_is_alive(parent_pid) or _is_alive(child_pid)):
            break
        await asyncio.sleep(0.1)

    assert not _is_alive(parent_pid)
    assert not _is_alive(child_pid), "child leaked under sync kill"
