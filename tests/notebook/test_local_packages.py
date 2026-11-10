"""A local path package's content keys every cell: the lock records it by path alone."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import pytest

from strata.notebook.env import (
    compute_execution_env_hash,
    compute_lockfile_hash,
    local_packages,
)
from strata.notebook.executor import CellExecutor
from strata.notebook.session import SessionManager
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

LIB_SOURCE = 'def hello() -> str:\n    return "{}"\n'


def _library(root: Path, name: str = "mylib") -> Path:
    lib = root / name
    (lib / "src" / name).mkdir(parents=True)
    (lib / "src" / name / "__init__.py").write_text(LIB_SOURCE.format("one"))
    (lib / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0.1.0"\n')
    (lib / "README.md").write_text("docs\n")
    return lib


def _lock(nb_dir: Path, **sources: tuple[str, Path]) -> None:
    """A ``uv.lock`` whose root depends on each package, from ``(kind, folder)``."""
    deps = ", ".join(f'{{ name = "{name}" }}' for name in sources)
    lines = [
        "version = 1",
        'requires-python = ">=3.12"',
        "",
        "[[package]]",
        'name = "nb"',
        'version = "0.1.0"',
        'source = { virtual = "." }',
        f"dependencies = [{deps}]",
    ]
    for name, (kind, folder) in sources.items():
        rel = os.path.relpath(folder, nb_dir)
        lines += ["", "[[package]]", f'name = "{name}"', 'version = "0.1.0"']
        lines.append(f'source = {{ {kind} = "{rel}" }}')
    (nb_dir / "uv.lock").write_text("\n".join(lines) + "\n")


def _install(nb_dir: Path, name: str, record: str, files: dict[str, str]) -> Path:
    """A dist installed in the notebook's venv: its ``RECORD`` and site-packages *files*."""
    site = nb_dir / ".venv" / "lib" / "python3.12" / "site-packages"
    dist_info = site / f"{name}-0.1.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "RECORD").write_text(record)
    for rel, content in files.items():
        (site / rel).write_text(content)
    return site


def _editable(tmp_path: Path) -> tuple[Path, Path]:
    lib = _library(tmp_path)
    nb_dir = tmp_path / "nb"
    nb_dir.mkdir()
    _lock(nb_dir, mylib=("editable", lib))
    _install(
        nb_dir,
        "mylib",
        "mylib.pth,,\nmylib-0.1.0.dist-info/RECORD,,\n",
        {"mylib.pth": f"{lib / 'src'}\n"},
    )
    return nb_dir, lib


def test_an_edit_under_an_editable_package_import_root_changes_the_key(tmp_path):
    nb_dir, lib = _editable(tmp_path)
    before = local_packages(nb_dir).key
    assert before is not None

    (lib / "src" / "mylib" / "__init__.py").write_text(LIB_SOURCE.format("two"))

    assert local_packages(nb_dir).key != before


def test_files_outside_the_import_roots_and_caches_do_not_change_the_key(tmp_path):
    nb_dir, lib = _editable(tmp_path)
    before = local_packages(nb_dir).key

    (lib / "README.md").write_text("other docs\n")  # outside src, which the .pth adds
    cache = lib / "src" / "mylib" / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"\0compiled")
    (lib / "src" / ".git").mkdir()
    (lib / "src" / ".git" / "HEAD").write_text("ref\n")
    (lib / "src" / "env").mkdir()
    (lib / "src" / "env" / "pyvenv.cfg").write_text("home = x\n")

    assert local_packages(nb_dir).key == before


def test_without_an_install_the_package_folder_is_keyed(tmp_path):
    lib = _library(tmp_path)
    nb_dir = tmp_path / "nb"
    nb_dir.mkdir()
    _lock(nb_dir, mylib=("editable", lib))
    before = local_packages(nb_dir).key

    (lib / "README.md").write_text("other docs\n")

    assert local_packages(nb_dir).key != before


def test_a_notebook_inside_its_package_folder_is_not_keyed(tmp_path):
    lib = _library(tmp_path)
    nb_dir = lib / "notebooks" / "nb"
    nb_dir.mkdir(parents=True)
    _lock(nb_dir, mylib=("editable", lib))
    before = local_packages(nb_dir).key

    (nb_dir / "cells").mkdir()
    (nb_dir / "cells" / "c1.py").write_text("x = 1\n")

    assert local_packages(nb_dir).key == before
    (lib / "src" / "mylib" / "__init__.py").write_text(LIB_SOURCE.format("two"))
    assert local_packages(nb_dir).key != before


def test_a_directory_package_is_keyed_by_what_was_installed(tmp_path):
    lib = _library(tmp_path)
    nb_dir = tmp_path / "nb"
    nb_dir.mkdir()
    _lock(nb_dir, mylib=("directory", lib))
    _install(nb_dir, "mylib", "mylib/__init__.py,sha256=aaa,10\n", {})
    before = local_packages(nb_dir).key

    # The venv holds a copy: an edit not yet installed is not what runs.
    (lib / "src" / "mylib" / "__init__.py").write_text(LIB_SOURCE.format("two"))
    assert local_packages(nb_dir).key == before

    _install(nb_dir, "mylib", "mylib/__init__.py,sha256=bbb,10\n", {})
    assert local_packages(nb_dir).key != before


def test_a_lock_without_local_packages_keys_as_before(tmp_path):
    nb_dir = tmp_path / "nb"
    nb_dir.mkdir()
    (nb_dir / "uv.lock").write_text(
        'version = 1\nrequires-python = ">=3.12"\n\n[[package]]\nname = "nb"\n'
        'version = "0.1.0"\nsource = { editable = "." }\n'
    )

    assert local_packages(nb_dir).key is None
    assert compute_execution_env_hash(
        nb_dir, local_packages=local_packages(nb_dir)
    ) == compute_lockfile_hash(nb_dir)


def test_the_lockfile_hash_leaves_local_packages_out(tmp_path):
    # It says whether the venv was synced from the lock: an edit needs no sync.
    nb_dir, lib = _editable(tmp_path)
    lock_hash, env_hash = (
        compute_lockfile_hash(nb_dir),
        compute_execution_env_hash(nb_dir, local_packages=local_packages(nb_dir)),
    )

    (lib / "src" / "mylib" / "__init__.py").write_text(LIB_SOURCE.format("two"))

    assert compute_lockfile_hash(nb_dir) == lock_hash
    assert compute_execution_env_hash(nb_dir, local_packages=local_packages(nb_dir)) != env_hash


def test_a_package_folder_that_cannot_be_read_does_not_fail_the_key(tmp_path):
    nb_dir, lib = _editable(tmp_path)
    locked = tmp_path / "locked"
    (locked / "src").mkdir(parents=True)
    _install(nb_dir, "mylib", "mylib.pth,,\n", {"mylib.pth": f"{locked / 'src'}\n"})
    locked.chmod(0)
    try:
        assert local_packages(nb_dir).key is not None
    finally:
        locked.chmod(0o755)


def _notebook_over(tmp_path: Path, lib: Path, source: str):
    """A session over a notebook whose cell c1 runs *source* with *lib* installed editable."""
    nb_dir = create_notebook(tmp_path, "nb")
    _lock(nb_dir, mylib=("editable", lib))
    _install(nb_dir, "mylib", "mylib.pth,,\n", {"mylib.pth": f"{lib / 'src'}\n"})
    # The interpreter imports the package from its source, as an editable install does. It
    # is the suite's own shim, not a real interpreter a write would truncate.
    venv_python = nb_dir / ".venv" / "bin" / "python"
    assert not venv_python.is_symlink() and venv_python.read_text().startswith("#!/bin/sh")
    venv_python.write_text(
        f"#!/bin/sh\nPYTHONPATH={shlex.quote(str(lib / 'src'))} "
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", source)
    add_cell_to_notebook(nb_dir, "c2", after_cell_id="c1")
    write_cell(nb_dir, "c2", "shown = greeting\n")
    return SessionManager().open_notebook(nb_dir)


@pytest.mark.asyncio
async def test_a_cell_reruns_after_its_editable_package_changes(tmp_path):
    lib = _library(tmp_path)
    source = "import mylib\ngreeting = mylib.hello()\n"
    session = _notebook_over(tmp_path, lib, source)
    executor = CellExecutor(session)

    first = await executor.execute_cell("c1", source)
    assert first.success, first.error
    assert first.outputs["greeting"]["preview"] == "one"

    # Another length: a .pyc is checked by source mtime in seconds and size.
    (lib / "src" / "mylib" / "__init__.py").write_text(LIB_SOURCE.format("two again"))

    assert session.compute_staleness()["c1"].status.value == "stale"
    again = await executor.execute_cell("c1", source)
    assert again.success, again.error
    assert not again.cache_hit
    assert again.outputs["greeting"]["preview"] == "two again"


@pytest.mark.asyncio
async def test_a_package_edited_while_the_cell_ran_keeps_its_outputs_for_that_run(tmp_path):
    lib = _library(tmp_path)
    module = lib / "src" / "mylib" / "__init__.py"
    # The cell edits the library as it runs, as an editor saving meanwhile would.
    source = (
        "import mylib\ngreeting = mylib.hello()\n"
        f"open({str(module)!r}, 'w').write({LIB_SOURCE.format('edited meanwhile')!r})\n"
    )
    session = _notebook_over(tmp_path, lib, source)
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", source)
    assert result.success, result.error
    assert "local package mylib changed after the run began" in result.stderr

    # Its outputs are not what the key before the edit describes, nor the one after it.
    module.write_text(LIB_SOURCE.format("one"))
    assert session.compute_staleness()["c1"].status.value == "stale"
    again = await executor.execute_cell("c1", source)
    assert not again.cache_hit


def test_a_staleness_walk_reads_the_packages_once(tmp_path, monkeypatch):
    lib = _library(tmp_path)
    session = _notebook_over(tmp_path, lib, "import mylib\ngreeting = mylib.hello()\n")
    reads: list[Path] = []

    def counted(notebook_dir: Path):
        reads.append(notebook_dir)
        return local_packages(notebook_dir)

    monkeypatch.setattr("strata.notebook.session.local_packages", counted)
    session.compute_staleness()

    assert reads == [session.path]
