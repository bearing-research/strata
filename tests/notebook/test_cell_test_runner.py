"""Tests for the cell-test runner wrapper (cell_test_runner.py).

The conftest plugin's own behaviour (assert rewriting, outcome counting) is
covered by ``test_cell_test_conftest.py``. Here we exercise the thin wrapper:
staging, the pytest-missing probe, and the collection-error fallback when
pytest exits before writing ``results.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from strata.notebook.cell_test_runner import (
    PytestUnavailableError,
    run_cell_tests_in_dir,
)

_PY = Path(sys.executable)


def test_pass_and_fail_totals(tmp_path):
    res = run_cell_tests_in_dir(
        rundir=tmp_path / "run",
        venv_python=_PY,
        cell_source="def add(a, b):\n    return a + b\n",
        test_source=(
            "def test_pass(cell):\n    assert cell.add(1, 2) == 3\n"
            "def test_fail(cell):\n    assert cell.add(1, 2) == 5\n"
        ),
        inputs={},
    )
    assert res["passed"] == 1
    assert res["failed"] == 1
    assert res["errored"] == 0


def test_inputs_are_injected(tmp_path):
    res = run_cell_tests_in_dir(
        rundir=tmp_path / "run",
        venv_python=_PY,
        cell_source="def scale(x):\n    return x * factor\n",
        test_source="def test_scale(cell):\n    assert cell.scale(3) == 30\n",
        inputs={"factor": 10},  # `factor` is an upstream input, not cell-defined
    )
    assert res["passed"] == 1


def test_collection_error_becomes_one_error(tmp_path):
    # A syntax error in the test file means pytest never reaches
    # sessionfinish, so results.json is absent — the wrapper synthesizes a
    # single error carrying the captured output.
    res = run_cell_tests_in_dir(
        rundir=tmp_path / "run",
        venv_python=_PY,
        cell_source="x = 1\n",
        test_source="def test_broken(cell):\n    assert (\n",  # unbalanced paren
        inputs={},
    )
    assert res["errored"] == 1
    assert res["passed"] == 0
    assert res["tests"][0]["outcome"] == "error"


def test_pytest_unavailable_raises(tmp_path):
    # A nonexistent interpreter can't import pytest → the actionable error.
    with pytest.raises(PytestUnavailableError):
        run_cell_tests_in_dir(
            rundir=tmp_path / "run",
            venv_python=tmp_path / "no-such-python",
            cell_source="x = 1\n",
            test_source="def test_x(cell):\n    assert True\n",
            inputs={},
        )
