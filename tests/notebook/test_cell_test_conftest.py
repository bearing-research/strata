"""Tests for the per-cell-test pytest plugin (cell_test_conftest.py).

These run the plugin the way the cell-test runner will: stage an isolated run
dir (conftest + inputs.pkl + cell_source.py + a test_*.py), invoke pytest as a
subprocess with ``--confcutdir``, and read back ``results.json``. The headline
assertion is that **assertion rewriting fires** — a failed ``assert`` carries
the introspected diff, not a bare ``AssertionError`` — since that is the whole
reason for the conftest-plugin approach.
"""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

_CONFTEST_SRC = Path(__file__).resolve().parents[2] / "src/strata/notebook/cell_test_conftest.py"


def _run(tmp_path: Path, cell_source: str, inputs: dict, test_source: str) -> dict:
    """Stage a run dir and execute the plugin exactly as the runner will."""
    rundir = tmp_path / "run"
    rundir.mkdir()
    shutil.copyfile(_CONFTEST_SRC, rundir / "conftest.py")
    (rundir / "cell_source.py").write_text(cell_source)
    (rundir / "inputs.pkl").write_bytes(pickle.dumps(inputs))
    test_file = rundir / "test_cell.py"
    test_file.write_text(test_source)

    subprocess.run(
        [
            sys.executable,
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
    )
    return json.loads((rundir / "results.json").read_text())


def test_pass_fail_and_input_exposure(tmp_path):
    cell_source = "def add(a, b):\n    return a + b\n"
    test_source = (
        "def test_pass(cell):\n"
        "    assert cell.add(1, 2) == 3\n"
        "def test_fail(cell):\n"
        "    assert cell.add(1, 2) == 5\n"
        "def test_input_visible(cell):\n"
        "    assert cell.base == 10\n"  # `base` came from inputs, not the cell
    )
    res = _run(tmp_path, cell_source, {"base": 10}, test_source)

    assert res["passed"] == 2
    assert res["failed"] == 1
    assert res["errored"] == 0
    by_name = {t["name"]: t for t in res["tests"]}
    assert by_name["test_pass"]["outcome"] == "passed"
    assert by_name["test_input_visible"]["outcome"] == "passed"
    assert by_name["test_fail"]["outcome"] == "failed"


def test_assertion_rewriting_fires(tmp_path):
    """The load-bearing property: a failed assert shows the introspected diff."""
    res = _run(
        tmp_path,
        "def add(a, b):\n    return a + b\n",
        {},
        "def test_fail(cell):\n    assert cell.add(1, 2) == 5\n",
    )
    msg = next(t["message"] for t in res["tests"] if t["name"] == "test_fail")
    # Rewritten assert renders the operands; a bare AssertionError would not.
    assert "assert 3 == 5" in msg


def test_failure_message_includes_captured_stdout(tmp_path):
    """A print() before the failing assert is captured in the message, not lost.

    ``report.longrepr`` carries only the traceback + assert diff; the test's own
    stdout/stderr is appended from ``report.capstdout`` so debugging output a test
    emitted is visible in the UI.
    """
    res = _run(
        tmp_path,
        "x = 1\n",
        {},
        "def test_fail(cell):\n    print('debug: x is', cell.x)\n    assert cell.x == 2\n",
    )
    msg = next(t["message"] for t in res["tests"] if t["name"] == "test_fail")
    assert "assert 1 == 2" in msg  # the assert diff is still there
    assert "Captured stdout" in msg
    assert "debug: x is 1" in msg  # the test's own print survived


def test_passing_test_has_no_captured_output_noise(tmp_path):
    """Captured output is only appended on failure — passing tests stay clean."""
    res = _run(
        tmp_path,
        "x = 1\n",
        {},
        "def test_ok(cell):\n    print('chatty')\n    assert cell.x == 1\n",
    )
    msg = next(t["message"] for t in res["tests"] if t["name"] == "test_ok")
    assert msg == ""  # passing → empty message, no captured-stdout block


def test_cell_source_error_is_an_error_not_a_fail(tmp_path):
    res = _run(
        tmp_path,
        "raise RuntimeError('boom')\n",  # the cell itself blows up
        {},
        "def test_anything(cell):\n    assert True\n",
    )
    assert res["errored"] == 1
    assert res["passed"] == 0
    assert "Cell did not execute" in res["tests"][0]["message"]


def test_collection_failure_is_an_error(tmp_path):
    """A syntax error in the test file is an error, not a silent zero.

    The module fails to import, so no runtest report fires; the
    ``pytest_collectreport`` hook is what keeps ``results.json`` from reading
    as an all-pass "no tests".
    """
    res = _run(
        tmp_path,
        "x = 1\n",
        {},
        "def test_broken(cell):\n    assert (\n",  # unbalanced paren
    )
    assert res["errored"] == 1
    assert res["passed"] == 0


def test_skip_counted_separately(tmp_path):
    res = _run(
        tmp_path,
        "x = 1\n",
        {},
        "import pytest\n"
        "@pytest.mark.skip(reason='nope')\n"
        "def test_skipped(cell):\n"
        "    assert False\n",
    )
    assert res["skipped"] == 1
    assert res["failed"] == 0


def test_parametrize_and_fixtures_work(tmp_path):
    """Real pytest features (the reason for requiring pytest) are available."""
    res = _run(
        tmp_path,
        "def square(n):\n    return n * n\n",
        {},
        "import pytest\n"
        "@pytest.mark.parametrize('n,expected', [(2, 4), (3, 9), (4, 17)])\n"
        "def test_square(cell, n, expected):\n"
        "    assert cell.square(n) == expected\n",
    )
    assert res["passed"] == 2  # (2,4) and (3,9)
    assert res["failed"] == 1  # (4,17) is wrong
