"""Run a cell's pytest tests in an isolated run dir.

Stages the cell source, the user's test file and the resolved inputs in a temp
dir, then runs ``pytest`` in the notebook venv with the ``cell_test_conftest``
plugin copied in as ``conftest.py``. Staging the tests under a ``test_*.py``
name gets native collection and assertion rewriting. Not on the keystroke path.
"""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

from strata.notebook.harness_user import HarnessUser, hand_over, spawn_kwargs

_CONFTEST_TEMPLATE = Path(__file__).parent / "cell_test_conftest.py"

# Cell tests are quick unit checks; a runaway test must not hang the WS connection.
_DEFAULT_TIMEOUT_SECONDS = 120.0


class PytestUnavailableError(RuntimeError):
    """Raised when ``pytest`` is not importable in the notebook venv.

    The executor maps this to a ``pytest_unavailable`` result so the UI can say
    "add pytest to this notebook's environment".
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
    env: dict[str, str] | None = None,
    run_as: HarnessUser | None = None,
) -> dict[str, Any]:
    """Stage *rundir* and run pytest; return the parsed ``results.json`` dict.

    The dict has totals (``passed``/``failed``/``errored``/``skipped``) plus a
    ``tests`` list of ``{name, nodeid, outcome, message}``. ``env`` and ``run_as``
    are the cell harness's: a test run imports the cell's source, so it is cell code.

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
    hand_over(rundir, run_as)

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
        env=env,
        **spawn_kwargs(run_as),
    )

    results_path = rundir / "results.json"
    if not results_path.exists():
        # pytest exited before ``pytest_sessionfinish`` wrote results, usually a
        # collection error (syntax error, bad import). Surface the output so the
        # user sees why nothing ran instead of an empty pass.
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
