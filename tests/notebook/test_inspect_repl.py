"""Focused tests for the live inspect REPL implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.inspect_repl import InspectSession
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


class _FakeStdout:
    def __init__(self):
        self._lines = [b'{"ok": true, "result": "ready"}\n']

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeStdin:
    def __init__(self):
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None


class _FakeProcess:
    def __init__(self):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout()
        self.stderr = _FakeStdout()
        self.returncode = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


@pytest.mark.asyncio
async def test_inspect_repl_uses_session_python(monkeypatch, tmp_path):
    """Inspect subprocess should use the notebook session interpreter."""
    nb_dir = create_notebook(tmp_path, "inspect_repl")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1")

    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.venv_python = Path("/custom/notebook/python")

    async def fake_materialize_upstreams(self, cell_id):
        return None

    def fake_load_input_blobs(self, cell_id, output_dir):
        return {}

    spawned: list[str] = []

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        spawned[:] = [str(part) for part in cmd]
        return _FakeProcess()

    monkeypatch.setattr(
        CellExecutor,
        "_materialize_upstreams",
        fake_materialize_upstreams,
    )
    monkeypatch.setattr(
        CellExecutor,
        "_load_input_blobs",
        fake_load_input_blobs,
    )
    monkeypatch.setattr(
        "strata.notebook.inspect_repl.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    inspect = InspectSession("c1")
    status = await inspect.start(session)

    assert status == "ready"
    assert spawned[0] == "/custom/notebook/python"
    assert spawned[1].endswith("inspect_harness.py")

    await inspect.close()
