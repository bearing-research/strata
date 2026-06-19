"""Generated pytest plugin for per-cell unit tests (design: design-cell-unit-tests.md).

This module is the TEMPLATE for the ``conftest.py`` Strata writes into an
isolated run dir when running a cell's tests. It is named ``cell_test_conftest``
(not ``conftest``) precisely so the project's own pytest run does NOT auto-load
it; the cell-test runner copies it to ``<rundir>/conftest.py`` at run time.

At run time the run dir contains, as siblings of this conftest:
- ``inputs.pkl``     — the cell's upstream inputs (``{var_name: value}``), pickled.
- ``cell_source.py`` — a copy of the cell's source.
- ``test_<cell>.py`` — the user's test file, staged under a ``test_*.py`` name so
  pytest collects it natively AND rewrites its assertions (rewriting only fires
  for modules matching ``python_files``).
- ``results.json``   — written here on session finish.

The plugin does three Strata-specific jobs: (1) build the cell's namespace once
(deserialize inputs + exec the cell source), (2) expose it via the ``cell``
fixture, (3) report structured per-test outcomes to ``results.json`` (no junit).
Hooks/fixtures are defined inline (no ``pytest_plugins`` — deprecated in
non-root conftests).
"""

from __future__ import annotations

import json
import pickle
import types
from pathlib import Path

import pytest

_RUNDIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def cell():
    """The cell's executed namespace, attribute-accessible.

    ``cell.X`` is whatever ``X`` is after the cell ran — a function/class it
    defines, an upstream input, or a module-level constant. The cell body runs
    once here (test runs re-execute the cell). A failure in the cell source
    surfaces as a clear setup error on every test that requests ``cell`` rather
    than an opaque collection error.
    """
    namespace: dict[str, object] = {}
    inputs = pickle.loads((_RUNDIR / "inputs.pkl").read_bytes())
    namespace.update(inputs)

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
    _phase_reports.setdefault(report.nodeid, {})[report.when] = (report.outcome, text)


def pytest_collectreport(report) -> None:  # noqa: ANN001 - pytest CollectReport
    """Record collection failures (e.g. a syntax error in the test file).

    Without this, a module that fails to import never produces a runtest
    report, so ``pytest_sessionfinish`` would write an all-zero ``results.json``
    and the failure would read as "no tests" instead of an error. We file it as
    a failed ``setup`` phase so ``_final_outcome`` collapses it to ``error``.
    """
    if report.failed:
        nodeid = report.nodeid or "<collection>"
        text = str(report.longrepr) if report.longrepr is not None else ""
        _phase_reports.setdefault(nodeid, {}).setdefault("setup", ("failed", text))


def _final_outcome(phases: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """Collapse a test's setup/call/teardown phases into one outcome + message.

    setup failure ⇒ error; a skip ⇒ skipped; call failure ⇒ failed (the
    assertion); teardown failure on an otherwise-passing test ⇒ error.
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
