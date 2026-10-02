"""Tests for the per-cell-test pytest plugin (cell_test_conftest.py).

Each test stages a run dir as the runner does, runs pytest with ``--confcutdir`` and reads
``results.json``. The headline: assertion rewriting fires, so a failed ``assert`` carries
the introspected diff, not a bare ``AssertionError``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from strata.notebook.serializer import serialize_value

_NOTEBOOK_SRC = Path(__file__).resolve().parents[2] / "src/strata/notebook"
_CONFTEST_SRC = _NOTEBOOK_SRC / "cell_test_conftest.py"


def _run(tmp_path: Path, cell_source: str, inputs: dict, test_source: str) -> dict:
    """Stage a run dir and execute the plugin exactly as the runner will."""
    rundir = tmp_path / "run"
    rundir.mkdir()
    shutil.copyfile(_CONFTEST_SRC, rundir / "conftest.py")
    (rundir / "cell_source.py").write_text(cell_source)
    input_dir = tmp_path / "inputs"
    specs = {}
    for name, value in inputs.items():
        payload = serialize_value(value, input_dir, name)
        specs[name] = {"content_type": payload["content_type"], "file": payload["file"]}
    manifest = {
        "serializer": str(_NOTEBOOK_SRC / "serializer.py"),
        "input_dir": str(input_dir),
        "inputs": specs,
    }
    (rundir / "inputs.json").write_text(json.dumps(manifest))
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
    """A print() before the failing assert is appended from ``report.capstdout``.

    ``report.longrepr`` alone has only the traceback and assert diff.
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
    """Captured output is appended only on failure."""
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

    No runtest report fires, so the ``pytest_collectreport`` hook keeps ``results.json`` from
    reading as an all-pass "no tests".
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
