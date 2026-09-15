"""One renv library per renv.lock, shared by the notebooks that have it. Item 47.

Most of this runs without R: the restore and the Rscript that installs are
stood in for, and what is under test is where the library is, who shares it,
and that changing one notebook's packages leaves the others' alone. The last
test restores a real lockfile twice and runs an R cell against the link.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook import dependencies, shared_env, writer
from strata.notebook.env import compute_execution_env_hash
from strata.notebook.shared_env import COMPLETE_MARKER, collect
from tests.notebook.conftest import skip_if_no_r

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the link is a symlink")

_FIXTURE = Path(__file__).parent / "fixtures" / "renv_jsonlite"
_LOCK = '{"R": {"Version": "4.4.0"}, "Packages": {"jsonlite": {"Version": "2.0.0"}}}\n'


@pytest.fixture
def shared(tmp_path, monkeypatch):
    root = tmp_path / "envs"
    config = SimpleNamespace(
        notebook_env_backend="shared",
        notebook_shared_env_dir=root,
        notebook_storage_dir=tmp_path / "notebooks",
    )
    monkeypatch.setattr("strata.server._state", SimpleNamespace(config=config))
    return root


@pytest.fixture
def fake_r(monkeypatch):
    """R without R: a fixed build, and a restore that installs the lock's packages."""
    monkeypatch.setattr(shared_env, "r_build", lambda: "R version 4.4.0 aarch64-apple-darwin20")
    restores: list[tuple[Path, dict[str, str] | None]] = []

    def restore(notebook_dir: Path, *, timeout: int, env: dict[str, str] | None = None) -> bool:
        restores.append((notebook_dir, env))
        for package in ("jsonlite",) + (("ggplot2",) if "ggplot2" in _lock(notebook_dir) else ()):
            (notebook_dir / "renv" / "library" / "R-4.4" / package).mkdir(parents=True)
        return True

    monkeypatch.setattr(writer, "_renv_restore_locked", restore)
    return restores


def _lock(notebook_dir: Path) -> str:
    return (notebook_dir / "renv.lock").read_text()


def _notebook(parent: Path, name: str, lock: str = _LOCK) -> Path:
    notebook = parent / name
    (notebook / "renv").mkdir(parents=True)
    (notebook / "renv.lock").write_text(lock)
    return notebook


def _library(notebook: Path) -> Path:
    library = notebook / "renv" / "library"
    assert library.is_symlink(), f"{library} is not a link"
    return Path(os.readlink(library))


def test_two_notebooks_with_one_lock_share_one_library_restored_once(tmp_path, shared, fake_r):
    first = _notebook(tmp_path, "first")
    second = _notebook(tmp_path, "second")
    # A per-notebook library from before the switch gives way to the link.
    (second / "renv" / "library" / "R-4.4" / "stale").mkdir(parents=True)
    hashes = {nb: compute_execution_env_hash(nb) for nb in (first, second)}

    assert writer._renv_sync(first) is True
    assert writer._renv_sync(second) is True

    assert _library(first) == _library(second)
    assert _library(first).parent == shared.resolve() / "r"
    assert [nb for nb, _ in fake_r] == [first]
    assert fake_r[0][1] == {"RENV_PATHS_CACHE": str(shared.resolve() / "r" / "cache")}
    assert (second / "renv" / "library" / "R-4.4" / "jsonlite").is_dir()
    assert not (second / "renv" / "library" / "R-4.4" / "stale").exists()
    assert {nb: compute_execution_env_hash(nb) for nb in (first, second)} == hashes


def test_a_different_lock_gets_its_own_library(tmp_path, shared, fake_r):
    first = _notebook(tmp_path, "first")
    second = _notebook(tmp_path, "second", lock=_LOCK.replace("2.0.0", "1.8.9"))

    assert writer._renv_sync(first) and writer._renv_sync(second)

    assert _library(first) != _library(second)
    assert len(fake_r) == 2


def _install(snippets: list[str], *, succeed: bool):
    """An Rscript that installs ggplot2 and snapshots, or fails part way."""

    async def run(notebook_dir, snippet, *, timeout, display_name, on_update=None, env=None):
        snippets.append(snippet)
        library = notebook_dir / "renv" / "library"
        # Whatever it installs goes into the notebook's own library.
        assert not library.is_symlink()
        (library / "R-4.4" / "ggplot2").mkdir(parents=True)
        if succeed:
            (notebook_dir / "renv.lock").write_text(_LOCK.replace("}}}", '}, "ggplot2": {}}}'))
        return dependencies._RscriptCommandResult(
            success=succeed,
            error=None if succeed else "renv::install failed (exit 1)",
            operation_log=dependencies.EnvironmentOperationLog(command="Rscript"),
        )

    return run


@pytest.mark.asyncio
async def test_installing_a_package_changes_only_that_notebooks_library(
    tmp_path, shared, fake_r, monkeypatch
):
    first = _notebook(tmp_path, "first")
    second = _notebook(tmp_path, "second")
    assert writer._renv_sync(first) and writer._renv_sync(second)
    shared_library = _library(second)
    snippets: list[str] = []
    monkeypatch.setattr(
        dependencies, "run_rscript_command_streaming", _install(snippets, succeed=True)
    )

    result = await dependencies.renv_add(first, "ggplot2")

    assert result.success and result.lockfile_changed
    assert snippets[0].startswith("renv::restore(prompt = FALSE)\n")
    assert _library(second) == shared_library
    assert not (shared_library / "R-4.4" / "ggplot2").exists()
    assert _library(first) != shared_library
    assert (_library(first) / "R-4.4" / "ggplot2").is_dir()
    assert (_library(first) / COMPLETE_MARKER).exists()

    # A third notebook that takes the new lock links to that library without
    # restoring anything.
    third = _notebook(tmp_path, "third", lock=_lock(first))
    assert writer._renv_sync(third)
    assert _library(third) == _library(first)
    assert [nb for nb, _ in fake_r] == [first]


@pytest.mark.asyncio
async def test_a_failed_install_links_back_to_the_library_for_the_lock(
    tmp_path, shared, fake_r, monkeypatch
):
    notebook = _notebook(tmp_path, "nb")
    assert writer._renv_sync(notebook)
    before = _library(notebook)
    monkeypatch.setattr(dependencies, "run_rscript_command_streaming", _install([], succeed=False))

    result = await dependencies.renv_add(notebook, "ggplot2")

    assert result.success is False
    assert _library(notebook) == before
    assert not (before / "R-4.4" / "ggplot2").exists()


def test_collect_removes_an_unlinked_library_and_keeps_a_linked_one(tmp_path, shared, fake_r):
    kept = _notebook(tmp_path, "kept")
    gone = _notebook(tmp_path, "gone", lock=_LOCK.replace("2.0.0", "1.8.9"))
    assert writer._renv_sync(kept) and writer._renv_sync(gone)
    unlinked = _library(gone)
    shutil.rmtree(gone)
    (shared / "r" / "cache" / "jsonlite").mkdir(parents=True)

    result = collect(shared, ttl_days=1, now=time.time() + 2 * 86400)

    assert result.removed == [f"r/{unlinked.name}"]
    assert result.referenced == [f"r/{_library(kept).name}"]
    assert not unlinked.exists()
    assert (shared / "r" / "cache" / "jsonlite").is_dir()


@skip_if_no_r
@pytest.mark.asyncio
async def test_a_real_lock_restores_once_and_an_r_cell_runs_on_the_link(
    tmp_path, shared, monkeypatch
):
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import CellLanguage
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    source = "library(jsonlite)\nout <- as.character(toJSON(list(ok = TRUE), auto_unbox = TRUE))\n"
    notebooks = []
    for name in ("first", "second"):
        notebook = writer.create_notebook(tmp_path / name, name, initialize_environment=False)
        writer.add_cell_to_notebook(notebook, "c1", None, language="r")
        writer.write_cell(notebook, "c1", source)
        shutil.copy(_FIXTURE / "renv.lock", notebook / "renv.lock")
        shutil.copy(_FIXTURE / ".Rprofile", notebook / ".Rprofile")
        shutil.copytree(_FIXTURE / "renv", notebook / "renv")
        notebooks.append(notebook)
    restores: list[Path] = []
    real_restore = writer._renv_restore_locked

    def counted(notebook_dir: Path, *, timeout: int, env: dict[str, str] | None = None) -> bool:
        restores.append(notebook_dir)
        return real_restore(notebook_dir, timeout=timeout, env=env)

    monkeypatch.setattr(writer, "_renv_restore_locked", counted)

    assert writer._renv_sync(notebooks[0]) is True
    assert writer._renv_sync(notebooks[1]) is True

    assert restores == [notebooks[0]]
    assert _library(notebooks[0]) == _library(notebooks[1])
    assert list(_library(notebooks[1]).rglob("jsonlite"))

    state = parse_notebook(notebooks[1])
    for cell in state.cells:
        cell.language = CellLanguage.R
    session = NotebookSession(state, notebooks[1])
    result = await CellExecutor(session).execute_cell("c1", source)
    assert result.success is True, result.error
    assert result.outputs["out"]["preview"] == '{"ok":true}'
