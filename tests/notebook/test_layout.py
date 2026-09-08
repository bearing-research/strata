"""What a notebook directory puts in git, and what it keeps out.

The split was documented and nowhere else: no function returned the committed
set, and nothing wrote a ``.gitignore``. So a new notebook directory had none,
and ``git add -A`` swept in the virtualenv, the SQLite artifact store, and
every blob it held.
"""

from __future__ import annotations

import subprocess

import pytest

from strata.notebook.layout import committed_paths, gitignore_contents, write_gitignore
from strata.notebook.writer import create_notebook


@pytest.fixture
def notebook(tmp_path):
    create_notebook(tmp_path, "demo", initialize_environment=False)
    return tmp_path / "demo"


class TestGitignore:
    def test_a_new_notebook_gets_one(self, notebook):
        assert (notebook / ".gitignore").exists()

    def test_it_covers_the_runtime_state(self, notebook):
        contents = (notebook / ".gitignore").read_text()

        for pattern in (".strata/", ".venv/", "renv/library/"):
            assert pattern in contents

    def test_an_existing_one_is_never_replaced(self, tmp_path):
        """A notebook may live inside a project with its own rules."""
        target = tmp_path / ".gitignore"
        target.write_text("mine\n")

        assert write_gitignore(tmp_path) is False
        assert target.read_text() == "mine\n"

    def test_it_can_be_declined(self, tmp_path):
        create_notebook(tmp_path, "nogit", initialize_environment=False, write_gitignore_file=False)

        assert not (tmp_path / "nogit" / ".gitignore").exists()


class TestCommittedPaths:
    def test_it_names_the_committed_set(self, notebook):
        (notebook / "cells" / "a1b2c3d4.py").write_text("x = 1\n")

        paths = {str(p) for p in committed_paths(notebook)}

        assert "notebook.toml" in paths
        assert "pyproject.toml" in paths
        assert "cells/a1b2c3d4.py" in paths

    def test_runtime_state_is_not_in_it(self, notebook):
        (notebook / ".strata").mkdir()
        (notebook / ".strata" / "runtime.json").write_text("{}")
        (notebook / ".venv").mkdir()
        (notebook / ".venv" / "pyvenv.cfg").write_text("")

        paths = {str(p) for p in committed_paths(notebook)}

        assert not any(p.startswith(".strata") for p in paths)
        assert not any(p.startswith(".venv") for p in paths)

    def test_cell_tests_are_committed(self, notebook):
        """A cell's tests are as much the notebook as the cell is."""
        tests_dir = notebook / "cells" / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_a1b2c3d4.py").write_text("def test_x(): pass\n")

        assert "cells/tests/test_a1b2c3d4.py" in {str(p) for p in committed_paths(notebook)}

    def test_pycache_is_excluded(self, notebook):
        cache = notebook / "cells" / "__pycache__"
        cache.mkdir()
        (cache / "a.cpython-313.pyc").write_bytes(b"\x00")

        assert not any("__pycache__" in str(p) for p in committed_paths(notebook))


class TestAgainstRealGit:
    """The item's own 'done when', run against git rather than reasoned about."""

    def test_git_add_all_stages_exactly_the_committed_set(self, notebook):
        def git(*args):
            return subprocess.run(
                ["git", *args],
                cwd=notebook,
                capture_output=True,
                text=True,
                check=True,
            ).stdout

        # The state a notebook is actually in when someone first commits it:
        # a cell written, an environment installed, outputs produced.
        (notebook / "cells" / "a1b2c3d4.py").write_text("x = 1\n")
        (notebook / "uv.lock").write_text("# lock\n")
        (notebook / ".venv").mkdir()
        (notebook / ".venv" / "pyvenv.cfg").write_text("home = /usr\n")
        strata_dir = notebook / ".strata" / "artifacts"
        strata_dir.mkdir(parents=True)
        (strata_dir / "artifacts.sqlite").write_bytes(b"SQLite format 3\x00")
        (notebook / ".strata" / "runtime.json").write_text("{}")

        git("init", "-q")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        git("add", "-A")
        staged = sorted(p for p in git("diff", "--cached", "--name-only").split("\n") if p)

        assert staged == [str(p) for p in committed_paths(notebook)], (
            "git add -A must stage the committed set and nothing else"
        )
        assert not any(p.startswith((".strata", ".venv")) for p in staged)


def test_gitignore_contents_is_stable_and_commented():
    """It lands in someone's repository, so it explains itself."""
    contents = gitignore_contents()

    assert contents.startswith("#")
    assert contents.endswith("\n")
