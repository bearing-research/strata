"""Integration tests for cell unit tests: executor + WebSocket handler.

These spawn a real pytest subprocess (via ``executor.run_cell_tests``) using the
test interpreter as the notebook venv, so they exercise the full slice:
materialize upstreams → inject inputs → run pytest → persist + broadcast.

WS handlers are driven directly with a fake WebSocket — never ``TestClient``'s
``websocket_connect`` (the py3.12/macOS portal hang, see project memory).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import cast

import pytest
from fastapi import WebSocket

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.runtime_state import load_runtime_state
from strata.notebook.session import NotebookSession
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)
from tests.notebook.e2e_fixtures import FakeNotebookWebSocket
from tests.notebook.e2e_fixtures import _reset_ws_globals as _reset_e2e_ws_globals


@pytest.fixture(autouse=True)
def _reset_ws_globals():
    _reset_e2e_ws_globals()
    yield
    _reset_e2e_ws_globals()


def _session_with(cells: list[tuple[str, str, str | None]]) -> NotebookSession:
    """Build a session whose notebook venv is the test interpreter.

    ``cells`` is a list of ``(cell_id, source, after_cell_id)``.
    """
    tmp = Path(tempfile.mkdtemp())
    nb = create_notebook(tmp, "test_notebook")
    for cell_id, source, after in cells:
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
    session = NotebookSession(parse_notebook(nb), nb)
    # Run the harness + pytest under the interpreter that has pytest + the
    # notebook extra, instead of bare "python" off PATH.
    session.venv_python = Path(sys.executable)
    return session


@pytest.mark.asyncio
async def test_run_cell_tests_pass_and_fail():
    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    executor = CellExecutor(session)

    result = await executor.run_cell_tests(
        "cell1",
        "def test_pass(cell):\n    assert cell.add(1, 2) == 3\n"
        "def test_fail(cell):\n    assert cell.add(1, 2) == 5\n",
    )

    assert result.passed == 1
    assert result.failed == 1
    assert result.errored == 0
    assert result.cell_source_hash
    assert result.test_source_hash
    assert result.ran_at > 0
    # The failing assertion carries the rewritten diff.
    fail = next(t for t in result.tests if t.outcome == "failed")
    assert "assert 3 == 5" in fail.message


@pytest.mark.asyncio
async def test_run_cell_tests_persists_and_writes_test_file():
    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    executor = CellExecutor(session)

    await executor.run_cell_tests("cell1", "def test_ok(cell):\n    assert cell.add(2, 2) == 4\n")

    # Persisted to runtime state...
    reloaded = load_runtime_state(session.path)
    assert reloaded.cells["cell1"].test_result["passed"] == 1
    # ...and surfaced on the in-memory cell.
    cell = session.notebook_state.get_cell("cell1")
    assert cell is not None and cell.test_result is not None
    assert cell.test_result.passed == 1


@pytest.mark.asyncio
async def test_run_cell_tests_injects_upstream_inputs():
    # cell_b references `factor` defined in cell_a — run_cell_tests must
    # materialize cell_a and inject its `factor` artifact as a test input.
    session = _session_with(
        [
            ("cell_a", "factor = 10\n", None),
            ("cell_b", "def scale(x):\n    return x * factor\n", "cell_a"),
        ]
    )
    executor = CellExecutor(session)

    result = await executor.run_cell_tests(
        "cell_b",
        "def test_scale(cell):\n    assert cell.scale(3) == 30\n",
    )

    assert result.passed == 1
    assert result.failed == 0
    assert result.input_fingerprint  # an upstream input was fingerprinted


@pytest.mark.asyncio
async def test_run_cell_tests_auto_provisions_pytest_then_retries(monkeypatch):
    """A missing pytest auto-installs into the dev group and the run retries once."""
    from strata.notebook import cell_test_runner, dependencies
    from strata.notebook.dependencies import DependencyChangeResult

    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    executor = CellExecutor(session)

    calls = {"n": 0}
    passed_run = {
        "passed": 1,
        "failed": 0,
        "errored": 0,
        "skipped": 0,
        "tests": [
            {
                "name": "test_ok",
                "nodeid": "test_cell.py::test_ok",
                "outcome": "passed",
                "message": "",
            }
        ],
    }

    def fake_run(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise cell_test_runner.PytestUnavailableError("pytest missing")
        return passed_run

    installs = {"n": 0}

    def fake_ensure(notebook_dir, tool, *, timeout=120):
        installs["n"] += 1
        assert tool == "pytest"
        return DependencyChangeResult(success=True, package=tool, action="add")

    monkeypatch.setattr(cell_test_runner, "run_cell_tests_in_dir", fake_run)
    monkeypatch.setattr(dependencies, "ensure_dev_tool", fake_ensure)

    result = await executor.run_cell_tests("cell1", "def test_ok(cell):\n    assert True\n")

    assert installs["n"] == 1  # provisioned exactly once
    assert calls["n"] == 2  # retried after the install
    assert result.auto_installed == ["pytest"]
    assert result.pytest_unavailable is False
    assert result.passed == 1


@pytest.mark.asyncio
async def test_run_cell_tests_auto_provision_failure_surfaces_unavailable(monkeypatch):
    """If the auto-install fails, fall back to the actionable pytest_unavailable flag."""
    from strata.notebook import cell_test_runner, dependencies
    from strata.notebook.dependencies import DependencyChangeResult

    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    executor = CellExecutor(session)

    calls = {"n": 0}

    def fake_run(**_kwargs):
        calls["n"] += 1
        raise cell_test_runner.PytestUnavailableError("pytest missing")

    def fake_ensure(notebook_dir, tool, *, timeout=120):
        return DependencyChangeResult(success=False, package=tool, action="add", error="boom")

    monkeypatch.setattr(cell_test_runner, "run_cell_tests_in_dir", fake_run)
    monkeypatch.setattr(dependencies, "ensure_dev_tool", fake_ensure)

    result = await executor.run_cell_tests("cell1", "def test_ok(cell):\n    assert True\n")

    assert calls["n"] == 1  # no retry when the install fails
    assert result.pytest_unavailable is True
    assert result.auto_installed == []


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------


def _register_fake_ws(session: NotebookSession):
    from strata.notebook.ws import _ensure_execution_state, _notebook_connections

    fake = FakeNotebookWebSocket()
    _notebook_connections.setdefault(session.id, []).append(cast(WebSocket, fake))
    return fake, _ensure_execution_state(session.id)


@pytest.mark.asyncio
async def test_handle_cell_run_tests_broadcasts_status_and_results():
    from strata.notebook.ws import _handle_cell_run_tests

    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    fake, execution_state = _register_fake_ws(session)

    await _handle_cell_run_tests(
        websocket=cast(WebSocket, fake),
        session=session,
        payload={
            "cell_id": "cell1",
            "test_source": "def test_pass(cell):\n    assert cell.add(1, 2) == 3\n",
        },
        execution_state=execution_state,
        notebook_id=session.id,
    )

    statuses = [f["payload"]["status"] for f in fake.frames_of("cell_test_status")]
    assert "running" in statuses
    assert statuses[-1] == "ready"

    results = fake.frames_of("cell_test_results")
    assert results
    payload = results[-1]["payload"]
    assert payload["cell_id"] == "cell1"
    assert payload["passed"] == 1
    assert payload["stale"] is False

    # Test source was committed to the sibling file.
    assert (session.path / "cells" / "cell1.test.py").exists()


@pytest.mark.asyncio
async def test_handle_cell_run_tests_failure_reports_error_status():
    from strata.notebook.ws import _handle_cell_run_tests

    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    fake, execution_state = _register_fake_ws(session)

    await _handle_cell_run_tests(
        websocket=cast(WebSocket, fake),
        session=session,
        payload={
            "cell_id": "cell1",
            "test_source": "def test_fail(cell):\n    assert cell.add(1, 2) == 99\n",
        },
        execution_state=execution_state,
        notebook_id=session.id,
    )

    statuses = [f["payload"]["status"] for f in fake.frames_of("cell_test_status")]
    assert statuses[-1] == "error"
    assert fake.frames_of("cell_test_results")[-1]["payload"]["failed"] == 1


@pytest.mark.asyncio
async def test_handle_cell_run_tests_rejects_non_python_cell():
    from strata.notebook.ws import _handle_cell_run_tests

    tmp = Path(tempfile.mkdtemp())
    nb = create_notebook(tmp, "test_notebook")
    add_cell_to_notebook(nb, "md1", None, language="markdown")
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    fake, execution_state = _register_fake_ws(session)

    await _handle_cell_run_tests(
        websocket=cast(WebSocket, fake),
        session=session,
        payload={"cell_id": "md1", "test_source": "def test_x(cell): assert True\n"},
        execution_state=execution_state,
        notebook_id=session.id,
    )

    errors = fake.frames_of("error")
    assert errors
    assert "Python" in errors[-1]["payload"]["error"]
    assert not fake.frames_of("cell_test_results")
