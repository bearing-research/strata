"""Template pytest plugin for per-cell unit tests.

Named ``cell_test_conftest`` so the project's own pytest run does not load it;
the cell-test runner copies it to ``<rundir>/conftest.py``. The run dir also
holds ``inputs.json`` (the serializer's path, the input dir and a
``{var_name: {content_type, file}}`` map), ``cell_source.py``, the
user's tests staged as ``test_<cell>.py`` (so pytest rewrites their asserts),
and the ``results.json`` this plugin writes on session finish.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_RUNDIR = Path(__file__).resolve().parent


def _load_inputs() -> dict[str, object]:
    """Deserialize the upstream inputs here, in the notebook venv, never in the server."""
    manifest = json.loads((_RUNDIR / "inputs.json").read_text())
    spec = importlib.util.spec_from_file_location("_nb_serializer", manifest["serializer"])
    assert spec is not None and spec.loader is not None
    serializer = importlib.util.module_from_spec(spec)
    sys.modules["_nb_serializer"] = serializer
    spec.loader.exec_module(serializer)

    input_dir = Path(manifest["input_dir"])
    inputs: dict[str, object] = {}
    for name, item in manifest["inputs"].items():
        try:
            inputs[name] = serializer.deserialize_value(
                item["content_type"], input_dir / item["file"]
            )
        except Exception as exc:  # noqa: BLE001 - e.g. R-only; a test reading it fails on the name
            print(f"Input {name} was not loaded: {type(exc).__name__}: {exc}", file=sys.stderr)
    return inputs


@pytest.fixture(scope="session")
def cell():
    """The cell's executed namespace, attribute-accessible.

    The cell body runs once per test run. A failure in the cell source surfaces
    as a setup error on every test that requests ``cell``, not a collection error.
    """
    namespace: dict[str, object] = {}
    namespace.update(_load_inputs())

    cell_source = (_RUNDIR / "cell_source.py").read_text()
    try:
        exec(compile(cell_source, "cell_source.py", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 - any cell error → a readable setup error
        pytest.fail(f"Cell did not execute (cannot test it): {type(exc).__name__}: {exc}")

    # Hide exec machinery (__builtins__, dunders) from the cell namespace.
    public = {k: v for k, v in namespace.items() if not k.startswith("__")}
    return types.SimpleNamespace(**public)


# Per-test outcome accumulation: nodeid -> {phase: (outcome, longrepr_text)}.
_phase_reports: dict[str, dict[str, tuple[str, str]]] = {}


def pytest_runtest_logreport(report) -> None:  # noqa: ANN001 - pytest Report
    text = str(report.longrepr) if report.longrepr is not None else ""
    # ``report.longrepr`` is only the traceback; captured stdout/stderr live in
    # ``capstdout``/``capstderr``, so append them or a failing test's prints are lost.
    if report.failed:
        if report.capstdout:
            text = f"{text}\n\n--- Captured stdout ---\n{report.capstdout.rstrip()}"
        if report.capstderr:
            text = f"{text}\n\n--- Captured stderr ---\n{report.capstderr.rstrip()}"
    _phase_reports.setdefault(report.nodeid, {})[report.when] = (report.outcome, text)


def pytest_collectreport(report) -> None:  # noqa: ANN001 - pytest CollectReport
    """Record collection failures (e.g. a syntax error in the test file).

    A module that fails to import never produces a runtest report, so without
    this ``results.json`` would read as "no tests". Filed as a failed ``setup``
    phase so ``_final_outcome`` reports ``error``.
    """
    if report.failed:
        nodeid = report.nodeid or "<collection>"
        text = str(report.longrepr) if report.longrepr is not None else ""
        _phase_reports.setdefault(nodeid, {}).setdefault("setup", ("failed", text))


def _final_outcome(phases: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """Collapse a test's setup/call/teardown phases into one outcome and message.

    Setup failure is error, a skip is skipped, call failure is failed, and a
    teardown failure on an otherwise-passing test is error.
    """
    setup_outcome, setup_text = phases.get("setup", ("passed", ""))
    call = phases.get("call")
    teardown_outcome, teardown_text = phases.get("teardown", ("passed", ""))

    if setup_outcome == "failed":
        return "error", setup_text
    if setup_outcome == "skipped" or (call is not None and call[0] == "skipped"):
        return "skipped", (call[1] if call else setup_text)
    if call is None:
        return "error", "test did not run"
    if call[0] == "failed":
        return "failed", call[1]
    if teardown_outcome == "failed":
        return "error", teardown_text
    return "passed", ""


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001, ARG001
    tests = []
    totals = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for nodeid, phases in _phase_reports.items():
        outcome, message = _final_outcome(phases)
        totals[outcome] = totals.get(outcome, 0) + 1
        tests.append(
            {
                "name": nodeid.split("::", 1)[-1],
                "nodeid": nodeid,
                "outcome": outcome,
                "message": message,
            }
        )

    (_RUNDIR / "results.json").write_text(
        json.dumps(
            {
                "passed": totals["passed"],
                "failed": totals["failed"],
                "errored": totals["error"],
                "skipped": totals["skipped"],
                "tests": tests,
            }
        )
    )
