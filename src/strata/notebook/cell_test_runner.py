"""Run a cell's pytest tests in an isolated run dir.

Executor-agnostic: given the cell source, the user's test source, the resolved
upstream inputs, and the notebook venv's python, stage a temp run dir and shell
to ``pytest`` using the generated ``cell_test_conftest`` plugin, then return the
structured results the plugin writes to ``results.json``.

This is deliberately *not* the keystroke path — only ``executor.run_cell_tests``
calls it. The plugin (``cell_test_conftest.py``) is copied into the run dir as
``conftest.py`` so pytest auto-loads it; staging the user's file under a
``test_*.py`` name is what gets native collection AND assertion rewriting (see
the conftest module docstring).
"""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

_CONFTEST_TEMPLATE = Path(__file__).parent / "cell_test_conftest.py"

# Wall-clock ceiling for a single cell's test run. Cell tests are meant to be
# quick unit checks; a runaway test shouldn't hang the WS connection forever.
_DEFAULT_TIMEOUT_SECONDS = 120.0


class PytestUnavailableError(RuntimeError):
    """Raised when ``pytest`` is not importable in the notebook venv.

    The executor turns this into a ``pytest_unavailable`` result so the UI can
    surface an actionable "add pytest to this notebook's environment" message
    rather than a raw ``ModuleNotFoundError``.
    """


def _pytest_available(venv_python: Path) -> bool:
    """Probe the venv for an importable ``pytest`` before staging a run."""
    try:
        probe = subprocess.run(
            [str(venv_python), "-c", "import pytest"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def run_cell_tests_in_dir(
    *,
    rundir: Path,
    venv_python: Path,
    cell_source: str,
    test_source: str,
    inputs: dict[str, Any],
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Stage *rundir* and run pytest; return the parsed ``results.json`` dict.

    The result dict has totals (``passed``/``failed``/``errored``/``skipped``)
    plus a ``tests`` list of ``{name, nodeid, outcome, message}``.

    Raises:
        PytestUnavailableError: ``pytest`` is not importable in *venv_python*.
    """
    if not _pytest_available(venv_python):
        raise PytestUnavailableError("pytest is not installed in this notebook's environment")

    rundir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_CONFTEST_TEMPLATE, rundir / "conftest.py")
    (rundir / "cell_source.py").write_text(cell_source, encoding="utf-8")
    (rundir / "inputs.pkl").write_bytes(pickle.dumps(inputs))
    test_file = rundir / "test_cell.py"
    test_file.write_text(test_source, encoding="utf-8")

    proc = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pytest",
            str(test_file),
            f"--confcutdir={rundir}",
            f"--rootdir={rundir}",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=rundir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )

    results_path = rundir / "results.json"
    if not results_path.exists():
        # pytest exited before ``pytest_sessionfinish`` wrote results — a
        # collection error in the test file itself (syntax error, bad import)
        # is the common cause. Surface the captured output as one error so the
        # user sees *why* nothing ran instead of an empty pass.
        detail = (proc.stdout + proc.stderr).strip() or "pytest produced no results"
        return {
            "passed": 0,
            "failed": 0,
            "errored": 1,
            "skipped": 0,
            "tests": [
                {
                    "name": "<collection>",
                    "nodeid": "",
                    "outcome": "error",
                    "message": detail,
                }
            ],
        }

    return json.loads(results_path.read_text(encoding="utf-8"))
