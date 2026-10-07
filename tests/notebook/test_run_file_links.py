"""The server reads and writes a cell's run directory without following links.

A harness user owns the run directory it is handed, so it can leave a symlink (or a
hard link) where an output should be. The server reads those outputs as its own user,
so following the link would hand the cell a file only the server may read, such as
``/proc/<server pid>/environ`` or ``artifacts.sqlite``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import threading
from pathlib import Path

import pytest

from strata.notebook.harness_user import UnsafeRunFile, read_run_file, write_run_file

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")

SECRET = b"TOP-SECRET-OF-THE-SERVER\n"


@pytest.fixture
def secret(tmp_path: Path) -> Path:
    path = tmp_path / "server_secret.txt"
    path.write_bytes(SECRET)
    path.chmod(0o600)
    return path


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


class TestReadRunFile:
    def test_a_plain_file_is_read(self, run_dir):
        (run_dir / "x.json").write_bytes(b"1")

        assert read_run_file(run_dir, "x.json") == b"1"

    @posix_only
    def test_a_symlinked_file_is_refused(self, run_dir, secret):
        (run_dir / "x.json").symlink_to(secret)

        with pytest.raises(UnsafeRunFile, match="x.json"):
            read_run_file(run_dir, "x.json")

    @posix_only
    def test_a_symlinked_directory_is_refused(self, tmp_path, secret):
        """The harness user owns the run directory itself, so it can swap that too."""
        (tmp_path / "run").symlink_to(tmp_path, target_is_directory=True)

        with pytest.raises(UnsafeRunFile):
            read_run_file(tmp_path / "run", secret.name)

    @posix_only
    def test_a_hard_link_is_refused(self, run_dir, secret):
        os.link(secret, run_dir / "x.json")

        with pytest.raises(UnsafeRunFile):
            read_run_file(run_dir, "x.json")

    @posix_only
    def test_a_fifo_is_refused_without_blocking(self, run_dir):
        os.mkfifo(run_dir / "x.json")
        outcome: list[BaseException] = []

        def _read() -> None:
            try:
                read_run_file(run_dir, "x.json")
            except UnsafeRunFile as exc:
                outcome.append(exc)

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(timeout=30)

        assert len(outcome) == 1

    def test_a_directory_is_refused(self, run_dir):
        (run_dir / "x.json").mkdir()

        with pytest.raises(UnsafeRunFile):
            read_run_file(run_dir, "x.json")

    @pytest.mark.parametrize("name", ["../server_secret.txt", "/etc/hosts", "sub/x.json", "..", ""])
    def test_a_name_that_leaves_the_directory_is_refused(self, run_dir, secret, name):
        """Display and variable file names come from the harness's own result."""
        with pytest.raises(UnsafeRunFile, match="not a plain file name"):
            read_run_file(run_dir, name)


@posix_only
class TestWriteRunFile:
    def test_a_planted_symlink_is_replaced_not_written_through(self, run_dir, secret):
        (run_dir / "f.cell_module.json").symlink_to(secret)

        write_run_file(run_dir, "f.cell_module.json", b"{}")

        assert secret.read_bytes() == SECRET
        assert not (run_dir / "f.cell_module.json").is_symlink()
        assert (run_dir / "f.cell_module.json").read_bytes() == b"{}"

    def test_a_planted_hard_link_is_replaced_not_written_through(self, run_dir, secret):
        os.link(secret, run_dir / "x.json")

        write_run_file(run_dir, "x.json", b"1")

        assert secret.read_bytes() == SECRET
        assert (run_dir / "x.json").read_bytes() == b"1"

    def test_a_symlinked_directory_is_refused(self, tmp_path, secret):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (tmp_path / "run").symlink_to(elsewhere, target_is_directory=True)

        with pytest.raises(UnsafeRunFile):
            write_run_file(tmp_path / "run", "x.json", b"1")

        assert list(elsewhere.iterdir()) == []


def _session(tmp_path: Path, cells: list[tuple[str, str]]):
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    notebook_dir = create_notebook(tmp_path / "nb", "Links")
    after = None
    for cell_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after)
        write_cell(notebook_dir, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def _swap_at_exit(prefix: str, target: Path) -> str:
    """Cell code that, as the harness exits, swaps its ``prefix*`` outputs for links."""
    return f"""
import atexit, json, os, sys
_out = json.load(open(sys.argv[1]))["output_dir"]
def _swap():
    for n in os.listdir(_out):
        if n.startswith({prefix!r}):
            os.remove(os.path.join(_out, n))
            os.symlink({str(target)!r}, os.path.join(_out, n))
atexit.register(_swap)
"""


def _stored_bytes(session) -> list[bytes]:
    root = session.path / ".strata" / "artifacts"
    return [p.read_bytes() for p in root.rglob("*") if p.is_file() and not p.is_symlink()]


@posix_only
class TestACellCannotLinkItsOutputs:
    def test_a_linked_output_fails_the_cell_and_is_never_stored(self, tmp_path, secret):
        source = _swap_at_exit("x.", secret) + "x = 1\n"
        session = _session(tmp_path, [("a", source), ("b", "y = x\n")])
        from strata.notebook.executor import CellExecutor

        result = asyncio.run(CellExecutor(session).execute_cell("a", source))

        assert result.success is False
        assert result.error is not None and "Refusing cell output 'x." in result.error
        assert not any(SECRET.strip() in blob for blob in _stored_bytes(session))

    def test_a_linked_display_is_never_stored(self, tmp_path, secret):
        source = _swap_at_exit("__display__", secret) + "'# shown'\n"
        session = _session(tmp_path, [("a", source)])
        from strata.notebook.executor import CellExecutor

        result = asyncio.run(CellExecutor(session).execute_cell("a", source))

        assert result.success is False
        assert result.error is not None and "Refusing cell output '__display__" in result.error
        assert not any(SECRET.strip() in blob for blob in _stored_bytes(session))

    def test_the_module_export_is_not_written_through_a_planted_link(self, tmp_path, secret):
        """The server writes ``f.cell_module.json`` into the run directory after the harness."""
        plant = f"""
import atexit, json, os, sys
_out = json.load(open(sys.argv[1]))["output_dir"]
atexit.register(lambda: os.symlink({str(secret)!r}, os.path.join(_out, "f.cell_module.json")))
"""
        source = plant + "def f():\n    return 1\n"
        session = _session(tmp_path, [("a", source), ("b", "y = f()\n")])
        from strata.notebook.executor import CellExecutor

        result = asyncio.run(CellExecutor(session).execute_cell("a", source))

        assert result.success, result.error
        assert secret.read_bytes() == SECRET

    def test_an_embedded_worker_does_not_pack_a_linked_output(self, tmp_path, secret, monkeypatch):
        """``embedded://`` packs the run directory into a bundle; a link must not be packed."""
        from types import SimpleNamespace

        from strata.notebook.executor import CellExecutor

        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(
                    deployment_mode="personal",
                    notebook_harness_user=None,
                    transforms_config={
                        "notebook_workers": [
                            {
                                "name": "embedded",
                                "backend": "executor",
                                "config": {"url": "embedded://local"},
                            }
                        ]
                    },
                )
            ),
        )
        source = _swap_at_exit("x.", secret) + "x = 1\n"
        session = _session(tmp_path, [("a", source), ("b", "y = x\n")])
        session.notebook_state.worker = "embedded"

        result = asyncio.run(CellExecutor(session).execute_cell("a", source))

        assert result.success is False
        assert result.error is not None and "Refusing cell output 'x." in result.error
        assert not any(SECRET.strip() in blob for blob in _stored_bytes(session))

    def test_an_embedded_worker_writes_nothing_through_a_swapped_run_directory(
        self, tmp_path, monkeypatch
    ):
        """A process the cell leaves running still owns the run directory after the harness.

        It can move the directory away and leave a link to anywhere in its place, so the
        bundle the embedded worker packs and unpacks must not be written by path under it.
        """
        from types import SimpleNamespace

        from strata.notebook.executor import CellExecutor

        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(
                    deployment_mode="personal",
                    notebook_harness_user=None,
                    transforms_config={
                        "notebook_workers": [
                            {
                                "name": "embedded",
                                "backend": "executor",
                                "config": {"url": "embedded://local"},
                            }
                        ]
                    },
                )
            ),
        )
        victim = tmp_path / "victim"
        victim.mkdir()
        moved = tmp_path / "moved"
        source = f"""
import atexit, json, os, sys
_out = json.load(open(sys.argv[1]))["output_dir"]
def _swap():
    os.rename(_out, {str(moved)!r})
    os.symlink({str(victim)!r}, _out)
atexit.register(_swap)
x = 1
"""
        session = _session(tmp_path, [("a", source), ("b", "y = x\n")])
        session.notebook_state.worker = "embedded"

        with contextlib.suppress(OSError):
            # Removing the run directory meets the link the cell left; not what is tested.
            asyncio.run(CellExecutor(session).execute_cell("a", source))

        assert list(victim.iterdir()) == []
