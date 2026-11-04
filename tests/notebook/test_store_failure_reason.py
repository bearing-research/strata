"""A run whose value can't be kept says which value and why, on a single run and on Run All."""

from __future__ import annotations

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.session import SessionManager
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from strata.notebook.ws import NotebookExecutionState, _execute_run_all

pytestmark = pytest.mark.integration

CASES = [
    pytest.param(
        "import threading\nv = threading.Lock()",
        "its output v (lock) could not be saved for the cells that read it: "
        "cannot pickle '_thread.lock' object",
        id="does-not-serialize",
    ),
    pytest.param(
        "if False:\n    v = 1",
        "v is not set when it finishes, and another cell reads it.",
        id="never-set",
    ),
]


def _session(tmp_path, source):
    notebook_dir = create_notebook(tmp_path, "store_failure")
    add_cell_to_notebook(notebook_dir, "a")
    write_cell(notebook_dir, "a", source)
    add_cell_to_notebook(notebook_dir, "b", after_cell_id="a")
    write_cell(notebook_dir, "b", "y = v")
    session = SessionManager().open_notebook(notebook_dir)
    session.ensure_venv_synced()
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize(("source", "reason"), CASES)
async def test_a_single_run_says_why(tmp_path, source, reason):
    session = _session(tmp_path, source)

    result = await CellExecutor(session).execute_cell("a", source)

    assert not result.success
    assert reason in result.error
    assert "Check server logs" not in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize(("source", "reason"), CASES)
async def test_run_all_says_why(tmp_path, monkeypatch, source, reason):
    from strata.notebook import ws

    class _Manager:
        def get_session(self, _session_id):
            return session

    errors: list[str] = []

    async def record(_session_id, message):
        if message["type"] == "cell_error":
            errors.append(message["payload"]["error"])

    session = _session(tmp_path, source)
    monkeypatch.setattr(ws, "_broadcast_message", record)
    monkeypatch.setattr(ws, "_get_session_manager", lambda: _Manager())

    await _execute_run_all(None, session, ["a", "b"], NotebookExecutionState(), session.id)

    assert any(reason in error for error in errors), errors
    assert session.notebook_state.get_cell("a").status == "error"
