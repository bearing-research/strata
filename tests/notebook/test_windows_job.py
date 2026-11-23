"""On Windows a server's children end with it, as the lifeline ends them on POSIX."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Starts a grandchild that sleeps, says its pid, and waits to be killed.
_PARENT = textwrap.dedent(
    """
    import subprocess, sys, time
    from strata.notebook.process_tree import end_children_with_this_process
    if sys.argv[1] == "bound":
        end_children_with_this_process()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    print(child.pid, flush=True)
    time.sleep(120)
    """
)


def test_off_windows_it_does_nothing(monkeypatch):
    from strata.notebook import process_tree

    # Another test's server may have made the job already: this one starts from none.
    monkeypatch.setattr(process_tree, "_job", None)
    monkeypatch.setattr(process_tree.sys, "platform", "linux")
    process_tree.end_children_with_this_process()

    assert process_tree._job is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
@pytest.mark.parametrize("bound", [True, False], ids=["bound", "unbound"])
def test_a_terminated_parent_takes_its_children_with_it(bound):
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    synchronize, terminate, wait_timeout = 0x00100000, 0x0001, 0x102

    with subprocess.Popen(
        [sys.executable, "-c", _PARENT, "bound" if bound else "unbound"],
        stdout=subprocess.PIPE,
        text=True,
    ) as parent:
        try:
            assert parent.stdout is not None
            pid = int(parent.stdout.readline())
            # Held from before the kill: the pid is not another process's when waited on.
            child = kernel32.OpenProcess(synchronize | terminate, False, pid)
            assert child, ctypes.get_last_error()
        finally:
            parent.kill()  # TerminateProcess, as os.kill(pid, SIGTERM) is here
    try:
        waited = kernel32.WaitForSingleObject(wintypes.HANDLE(child), 10_000 if bound else 1_000)
        if bound:
            assert waited == 0  # WAIT_OBJECT_0: ended with its parent
        else:
            assert waited == wait_timeout  # without the job it runs on: what this fixes
    finally:
        kernel32.TerminateProcess(wintypes.HANDLE(child), 1)
        kernel32.CloseHandle(wintypes.HANDLE(child))
