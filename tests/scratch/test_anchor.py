"""Which project a scratchpad command serves: its anchor."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from strata import scratch


def _dirs(root: Path, *paths: str) -> None:
    for path in paths:
        (root / path).mkdir(parents=True, exist_ok=True)


def _projects(root: Path, *paths: str) -> None:
    for path in paths:
        (root / path).mkdir(parents=True, exist_ok=True)
        (root / path / "pyproject.toml").write_text("[project]\nname = 'p'\n")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path.resolve() / "repo"
    _dirs(root, ".git", "docs", "services/api/src/api")
    _projects(root, ".", "services/api")
    return root


def test_the_nearest_project_at_or_above_the_start_is_the_anchor(repo):
    assert scratch.find_anchor(repo / "services/api/src/api") == repo / "services/api"
    assert scratch.find_anchor(repo / "services/api") == repo / "services/api"
    assert scratch.find_anchor(repo / "docs") == repo


def test_nothing_above_the_git_root_is_a_project(tmp_path):
    tmp_path = tmp_path.resolve()
    _projects(tmp_path, "outer")
    _dirs(tmp_path, "outer/repo/.git", "outer/repo/sub")

    # With no project in it, the repository is one, from any of its directories.
    assert scratch.find_anchor(tmp_path / "outer/repo/sub") == tmp_path / "outer/repo"


def test_a_worktrees_git_file_bounds_it_too(tmp_path):
    tmp_path = tmp_path.resolve()
    _projects(tmp_path, "outer")
    _dirs(tmp_path, "outer/wt/sub")
    (tmp_path / "outer/wt/.git").write_text("gitdir: elsewhere\n")

    assert scratch.find_anchor(tmp_path / "outer/wt/sub") == tmp_path / "outer/wt"


def test_outside_a_repository_nothing_above_home_is_a_project(tmp_path, monkeypatch):
    tmp_path = tmp_path.resolve()
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    _projects(tmp_path, ".", "home/proj")
    _dirs(home, "proj/a/b", "loose/c")

    assert scratch.find_anchor(home / "proj/a/b") == home / "proj"
    assert scratch.find_anchor(home / "loose/c") == home / "loose/c"


def test_home_bounds_it_below_a_repository_above_home(tmp_path, monkeypatch):
    # A checkout above home (``/``, in some containers) is not the search's bound.
    tmp_path = tmp_path.resolve()
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    _projects(tmp_path, ".")
    _dirs(tmp_path, ".git", "home/loose/c")

    assert scratch.find_anchor(home / "loose/c") == home / "loose/c"


def test_with_no_home_directory_the_repository_bounds_it(repo, monkeypatch):
    def no_home(cls):
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", classmethod(no_home))
    assert scratch.find_anchor(repo / "services/api/src") == repo / "services/api"


def test_a_start_inside_a_scratchpad_is_in_its_project(repo):
    _projects(repo, ".strata-scratch")
    _dirs(repo, ".strata-scratch/work/out")

    assert scratch.find_anchor(repo / ".strata-scratch/work/out") == repo
    assert scratch.find_anchor(repo / ".strata-scratch") == repo


def test_outside_home_and_any_repository_the_search_goes_up(tmp_path, monkeypatch):
    tmp_path = tmp_path.resolve()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "elsewhere"))
    _projects(tmp_path, "proj")
    _dirs(tmp_path, "proj/a")

    assert scratch.find_anchor(tmp_path / "proj/a") == tmp_path / "proj"


def test_a_given_root_is_the_anchor_as_given(repo, monkeypatch):
    monkeypatch.chdir(repo / "services/api/src")
    monkeypatch.delenv("STRATA_SCRATCH_ROOT", raising=False)
    assert scratch.resolve_anchor(None) == repo / "services/api"

    monkeypatch.setenv("STRATA_SCRATCH_ROOT", str(repo / "docs"))
    assert scratch.resolve_anchor(None) == repo / "docs"
    assert scratch.resolve_anchor(str(repo / "services")) == repo / "services"
    # An empty --root is the current directory, never a search.
    assert scratch.resolve_anchor("") == repo / "services/api/src"
    monkeypatch.setenv("STRATA_SCRATCH_ROOT", "")
    assert scratch.resolve_anchor(None) == repo / "services/api"


def test_the_cli_serves_the_project_it_is_run_in(repo):
    """``strata scratch rm`` from a subdirectory acts on the project's scratchpad."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env["HOME"] = str(Path.home())
    removed = subprocess.run(
        [sys.executable, "-m", "strata.cli", "scratch", "rm"],
        cwd=repo / "services/api/src",
        env=env,
        capture_output=True,
        text=True,
    )

    assert f"{repo / 'services/api'} has no scratchpad" in removed.stderr
