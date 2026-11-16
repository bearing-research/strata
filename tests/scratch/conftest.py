"""Fixtures for the scratchpad's daemon and its stdio bridge."""

from __future__ import annotations

import contextlib
import os
import signal
from pathlib import Path

import pytest

from strata import scratch


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An empty project; the daemons it starts keep their stores under *tmp_path*."""
    for key in [k for k in os.environ if k.startswith("STRATA_")]:
        monkeypatch.delenv(key)
    # The test's own home (conftest patches Path.home), so its daemons share it.
    monkeypatch.setenv("HOME", str(Path.home()))
    # Where Windows looks instead: its Path.home() ignores HOME.
    monkeypatch.setenv("USERPROFILE", str(Path.home()))
    monkeypatch.setenv("STRATA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("STRATA_NOTEBOOK_WARM_POOL_SIZE", "0")
    # A backstop: a daemon a failed test leaves behind stops itself.
    monkeypatch.setenv("STRATA_IDLE_EXIT_MINUTES", "1")
    started: list[int] = []
    spawn = scratch._spawn

    def recorded(anchor, port):
        pid = spawn(anchor, port)
        started.append(pid)
        return pid

    monkeypatch.setattr(scratch, "_spawn", recorded)
    root = tmp_path / "project"
    root.mkdir()
    yield root
    # Every one this test started, not only the one server.json names.
    for pid in started:
        # Gone already: ProcessLookupError, or on Windows a plain OSError.
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
