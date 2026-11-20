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
    nb_dir = create_notebook(tmp_path, "inspect_repl")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1")

    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.venv_python = Path("/custom/notebook/python")

    async def fake_materialize_upstreams(self, cell_id):
        return None

    def fake_load_input_blobs(self, cell_id, output_dir, *, fanout_variant=None):
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


@pytest.mark.asyncio
async def test_inspect_repl_does_not_start_without_an_interpreter(monkeypatch, tmp_path):
    """Inspect input is cell code, so without the notebook's interpreter the REPL does not
    start, rather than fall back to ``python`` on PATH.
    """
    nb_dir = create_notebook(tmp_path, "inspect_repl")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1")
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    assert session.venv_python is None

    spawned: list[str] = []

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        spawned[:] = [str(part) for part in cmd]
        return _FakeProcess()

    monkeypatch.setattr(
        "strata.notebook.inspect_repl.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    inspect = InspectSession("c1")
    status = await inspect.start(session)

    assert "not ready" in status
    assert not inspect.ready
    assert spawned == []


def test_exit_at_the_inspect_prompt_fails_the_line_not_the_process(tmp_path):
    """The builtin ``exit()`` closes ``sys.stdin`` before it raises: the commands keep
    their own stream, so the next one is still answered."""
    import json
    import subprocess
    import sys

    from strata.notebook import inspect_harness

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"inputs": {}, "output_dir": str(tmp_path)}))
    commands = "".join(
        json.dumps(c) + "\n"
        for c in (
            {"expr": "exit()"},
            {"expr": "input()"},
            # fd 0 itself, which a process the prompt starts inherits: not the pipe.
            {"expr": "__import__('stat').S_ISFIFO(__import__('os').fstat(0).st_mode)"},
            {"expr": "1 + 1"},
            {"cmd": "close"},
        )
    )
    run = subprocess.run(
        [sys.executable, inspect_harness.__file__, str(manifest)],
        input=commands,
        capture_output=True,
        text=True,
        timeout=60,
    )
    replies = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]

    assert run.returncode == 0, run.stderr
    ready, exited, read, fd_zero, two = replies
    assert ready["result"] == "ready"
    assert exited["ok"] is False and "SystemExit" in exited["error"]
    # Code at the prompt reads nothing of the commands.
    assert read["ok"] is False, read
    assert fd_zero["result"] == "False", fd_zero
    assert two["result"] == "2"
