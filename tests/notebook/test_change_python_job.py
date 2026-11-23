"""Changing a notebook's Python: the venv is rebuilt, and a failed rebuild is rolled back."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook.parser import parse_notebook
from strata.notebook.python_versions import read_requested_python_minor
from strata.notebook.session import NotebookSession
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    update_requires_python,
    write_cell,
)
from tests.shim_helpers import write_interpreter_shim


class _Uv:
    """Stands in for ``uv sync``: records what each sync saw, answers from a script."""

    def __init__(self, notebook_dir: Path, outcomes: list[str]) -> None:
        self.notebook_dir = notebook_dir
        self.outcomes = outcomes
        self.saw: list[tuple[str | None, bool]] = []

    async def sync_streaming(self, *, python_version, timeout, on_update):
        venv_python = self.notebook_dir / ".venv" / "bin" / "python"
        self.saw.append((read_requested_python_minor(self.notebook_dir), venv_python.exists()))
        outcome = self.outcomes.pop(0)
        if outcome == "edit, then fail":
            # Someone writes the file while uv runs.
            pyproject = self.notebook_dir / "pyproject.toml"
            pyproject.write_text(pyproject.read_text() + "\n[tool.mine]\nkept = true\n")
            return SimpleNamespace(success=False, error="no wheel for 3.99", operation_log=None)
        if outcome == "raise":
            raise OSError("uv vanished")
        if outcome == "ok":
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            write_interpreter_shim(venv_python)
            return SimpleNamespace(success=True, error=None, operation_log=None)
        return SimpleNamespace(success=False, error=outcome, operation_log=None)


@pytest.fixture
def session(tmp_path):
    notebook_dir = create_notebook(tmp_path, "Py", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", "x = 1\n")
    update_requires_python(notebook_dir, "3.12")
    venv_python = notebook_dir / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    write_interpreter_shim(venv_python)
    return NotebookSession(parse_notebook(notebook_dir), notebook_dir)


async def _change(session: NotebookSession, uv: _Uv, minor: str):
    session.backend.sync_streaming = uv.sync_streaming  # type: ignore[method-assign]
    job = await session.environment_jobs.submit(action="change_python", python_version=minor)
    await session.environment_jobs.wait()
    return job


async def test_the_venv_is_rebuilt_on_the_new_minor(session):
    uv = _Uv(session.path, ["ok"])

    job = await _change(session, uv, "3.13")

    assert (job.status, job.error) == ("completed", None)
    # The old venv is gone before the sync, or uv refuses the interpreter mismatch.
    assert uv.saw == [("3.13", False)]
    assert read_requested_python_minor(session.path) == "3.13"


async def test_a_failed_rebuild_restores_the_old_minor_and_syncs_it(session):
    before = (session.path / "pyproject.toml").read_text()
    uv = _Uv(session.path, ["no interpreter for 3.99", "ok"])

    job = await _change(session, uv, "3.99")

    assert (job.status, job.error) == ("failed", "no interpreter for 3.99")
    assert uv.saw == [("3.99", False), ("3.12", False)]
    assert (session.path / "pyproject.toml").read_text() == before
    assert (session.path / ".venv" / "bin" / "python").exists()


async def test_a_rollback_that_fails_reports_the_first_failure(session):
    uv = _Uv(session.path, ["no interpreter for 3.99", "raise"])

    job = await _change(session, uv, "3.99")

    # The old pyproject is back, and the error is what went wrong, not the rollback.
    assert (job.status, job.error) == ("failed", "no interpreter for 3.99")
    assert read_requested_python_minor(session.path) == "3.12"


async def test_a_hand_written_spec_is_restored_as_it_was(session):
    """``>=3.12`` names no single minor, so the rollback restores the file, not a minor."""
    pyproject = session.path / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('"==3.12.*"', '">=3.12"'))
    before = pyproject.read_text()
    assert read_requested_python_minor(session.path) is None
    uv = _Uv(session.path, ["no interpreter for 3.99", "ok"])

    job = await _change(session, uv, "3.99")

    assert (job.status, job.error) == ("failed", "no interpreter for 3.99")
    assert pyproject.read_text() == before
    assert len(uv.saw) == 2, "the restored environment is synced again"


async def test_an_edit_made_during_the_sync_survives_the_rollback(session):
    uv = _Uv(session.path, ["edit, then fail", "ok"])

    job = await _change(session, uv, "3.99")

    assert (job.status, job.error) == ("failed", "no wheel for 3.99")
    text = (session.path / "pyproject.toml").read_text()
    assert "[tool.mine]\nkept = true" in text
    assert read_requested_python_minor(session.path) == "3.12"
