"""``scratch.SetupCancel``: a cancelled setup leaves no daemon that no record names."""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from strata import scratch

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the daemon leads its own group")


def _daemon() -> int:
    """A stand-in for the daemon, launched as ``_spawn`` launches it: the leader of its own
    session, and no child of this process, so a killed one is reaped, never a zombie."""
    launcher = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],"
        " start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print(child.pid)\n"
    )
    done = subprocess.run([sys.executable, "-c", launcher], capture_output=True, text=True)
    return int(done.stdout)


def _ended(pid: int) -> bool:
    deadline = time.monotonic() + 30
    while scratch._alive(pid):
        if time.monotonic() > deadline:
            scratch._kill(pid)
            return False
        time.sleep(0.05)
    return True


def test_a_cancel_kills_a_daemon_started_and_not_yet_recorded():
    cancel = scratch.SetupCancel()
    pid = _daemon()
    cancel.started(pid)

    cancel.set()

    assert _ended(pid)


def test_a_daemon_started_after_the_cancel_is_killed_at_once():
    cancel = scratch.SetupCancel()
    cancel.set()
    pid = _daemon()

    cancel.started(pid)

    assert _ended(pid)


def test_a_recorded_daemon_outlives_a_cancel(monkeypatch):
    killed: list[int] = []
    monkeypatch.setattr(scratch, "_kill", killed.append)
    cancel = scratch.SetupCancel()
    cancel.started(4242)
    cancel.finished()

    cancel.set()

    assert killed == []


def test_a_cancel_during_a_setup_step_fails_the_setup_and_ends_its_daemon(tmp_path, monkeypatch):
    started: list[int] = []
    cancel = scratch.SetupCancel()

    def spawn(anchor, port, python=None):
        started.append(_daemon())
        return started[-1]

    def healthy(url):
        # The daemon never answers; the setup waits on it until the cancel.
        if not cancel.is_set():
            threading.Thread(target=cancel.set).start()
        return False

    monkeypatch.setattr(scratch, "_spawn", spawn)
    monkeypatch.setattr(scratch, "_healthy", healthy)

    with pytest.raises(scratch.ScratchError, match="exited while starting"):
        scratch.ensure_daemon(tmp_path, cancel=cancel)

    assert _ended(started[0])
    assert scratch.find_daemon(tmp_path) is None


def test_a_cancelled_setup_starts_nothing(tmp_path, monkeypatch):
    cancel = scratch.SetupCancel()
    cancel.set()
    monkeypatch.setattr(scratch, "_spawn", lambda *args: pytest.fail("a daemon was started"))

    with pytest.raises(scratch.ScratchError, match="cancelled"):
        scratch.ensure_daemon(tmp_path, cancel=cancel)


def test_a_cancel_after_the_lock_starts_nothing(tmp_path, monkeypatch):
    cancel = scratch.SetupCancel()
    monkeypatch.setattr(scratch, "_spawn", lambda *args: pytest.fail("a daemon was started"))

    def progress(step):
        # While the scratchpad is created, before the daemon is started.
        cancel.set()

    with pytest.raises(scratch.ScratchError, match="cancelled"):
        scratch.ensure_daemon(tmp_path, progress, cancel)
