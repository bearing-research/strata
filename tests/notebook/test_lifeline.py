"""A cell subprocess dies with the process that spawned it.

Each test starts a throwaway parent that spawns a real harness through a production
spawn path, SIGKILLs that parent, and checks the harness goes too. SIGKILL runs no
cleanup, so only the lifeline pipe can tell the harness.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import strata.notebook
from strata.notebook.dependencies import resolve_uv

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="no lifeline on Windows")

_HARNESS = Path(strata.notebook.__file__).parent / "harness.py"

# Bounds only stop a hang; neither is what the test measures.
_STARTUP_BOUND = 60.0
_EXIT_BOUND = 60.0

_PRELUDE = """
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace

work = Path(sys.argv[1])
harness = Path(sys.argv[2])
source = (work / "cell.py").read_text()
run = work / "run"
run.mkdir()
manifest = run / "manifest.json"
manifest.write_text(json.dumps({"source": source, "output_dir": str(run)}))
"""

_PARENTS = {
    # A cold cell under ``uv run``: the harness's parent is uv, not the killed process.
    "cold_harness_under_uv": """
from strata.notebook.executor import CellExecutor

uv = sys.argv[3]
CellExecutor._harness_command = lambda self, m, v, u: [
    uv, "run", "--no-project", "--python", sys.executable, "python", str(self.harness_path), str(m)
]
executor = CellExecutor.__new__(CellExecutor)
executor.harness_path = harness
executor.session = SimpleNamespace(path=work)
asyncio.run(executor._run_harness(manifest, Path(sys.executable), 600.0))
""",
    "batch_harness": """
from strata.notebook.executor import CellExecutor

(work / ".strata").mkdir()
executor = CellExecutor.__new__(CellExecutor)
executor.harness_path = harness
executor.session = SimpleNamespace(
    path=work,
    venv_python=Path(sys.executable),
    notebook_state=SimpleNamespace(get_cell=lambda _cell_id: None),
)
spec = {"cell_id": "c1", "source": source, "env": {}, "mount_manifest": {}}
asyncio.run(executor._run_batch([spec], use_cache=False, batch_timeout_seconds=600.0))
""",
    "warm_pool_worker": """
from strata.notebook.pool import WarmProcessPool

async def main():
    pool = WarmProcessPool(work, pool_size=1, python_executable=sys.executable)
    await pool._spawn_warm_process()
    warm = pool._available.get_nowait()
    warm.process.stdin.write(f"{manifest}\\n".encode())
    await warm.process.stdin.drain()
    await warm.process.wait()

asyncio.run(main())
""",
    "strata_worker_harness": """
from strata.notebook.remote_executor import _run_harness

asyncio.run(_run_harness(harness, manifest, 600.0))
""",
}


def _gone(pid: int) -> bool:
    """True once *pid* has exited; a zombie awaiting its reaper counts as exited."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return state.stdout.strip().startswith("Z") or not state.stdout.strip()


@pytest.mark.parametrize("path", sorted(_PARENTS))
def test_the_harness_dies_with_its_spawner(tmp_path: Path, path: str) -> None:
    extra: list[str] = []
    if path == "cold_harness_under_uv":
        uv = resolve_uv()
        if uv is None:
            pytest.skip("uv is not installed")
        extra = [uv]

    pid_file = tmp_path / "harness.pid"
    (tmp_path / "cell.py").write_text(
        textwrap.dedent(f"""
        import os, time
        with open({str(pid_file) + ".tmp"!r}, "w") as f:
            f.write(str(os.getpid()))
        os.replace({str(pid_file) + ".tmp"!r}, {str(pid_file)!r})
        time.sleep(600)
        """)
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", _PRELUDE + _PARENTS[path], str(tmp_path), str(_HARNESS), *extra],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    harness_group: int | None = None
    try:
        deadline = time.monotonic() + _STARTUP_BOUND
        while not pid_file.exists():
            if parent.poll() is not None:
                pytest.fail(f"the parent exited first: {parent.stderr.read().decode()}")
            if time.monotonic() > deadline:
                pytest.fail("the cell never started")
            time.sleep(0.05)
        harness_pid = int(pid_file.read_text())
        harness_group = os.getpgid(harness_pid)
        assert harness_group != os.getpgrp(), "the harness must run in its own process group"

        parent.kill()
        parent.wait()

        deadline = time.monotonic() + _EXIT_BOUND
        while not _gone(harness_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _gone(harness_pid), "the harness outlived the process that spawned it"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if harness_group is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(harness_group, signal.SIGKILL)
