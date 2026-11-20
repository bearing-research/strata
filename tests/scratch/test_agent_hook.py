"""The agent hook (``strata/agent_hook.py``) and the digest it reads (``scratch_digest``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from strata import agent_hook, scratch
from strata.cli import main
from strata.notebook import code_reads

HOOK = Path(agent_hook.__file__)


def _hook(cwd: Path, mode: str = "session-start", prompt: str = "") -> str:
    """The hook as Claude Code runs it: by path, isolated, the payload on stdin."""
    done = subprocess.run(
        [sys.executable, "-I", str(HOOK), mode],
        input=json.dumps({"cwd": str(cwd), "prompt": prompt}),
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stderr == ""
    return done.stdout


def _run(project: Path, source: str, capsys) -> dict:
    assert main(["scratch", "run", "-c", source, "--root", str(project)]) == 0
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def held(project, capsys):
    """A scratchpad whose cell read ``data.txt``, its daemon stopped since."""
    data = project / "data.txt"
    data.write_text("41")
    _run(project, f"value = int(open({str(data)!r}).read())", capsys)
    assert scratch.stop_daemon(project)
    return project, data


def test_a_cell_whose_reads_hold_is_cached(held):
    project, _ = held

    block = _hook(project)

    assert block.startswith('<strata_scratchpad note="')
    (line,) = [line for line in block.splitlines() if line.startswith("- ")]
    assert line.startswith("- value (cell ")
    assert line.endswith(" to build, cached")


def test_a_file_edited_with_the_daemon_stopped_is_stale_never_cached(held):
    project, data = held
    data.write_text("42")

    block = _hook(project)

    assert "value (cell " in block
    assert "stale, data.txt changed" in block
    assert "cached" not in block


def test_downstream_of_a_stale_cell_is_stale(project, capsys):
    data = project / "data.txt"
    data.write_text("41")
    _run(project, f"value = int(open({str(data)!r}).read())", capsys)
    _run(project, "doubled = value * 2", capsys)
    assert scratch.stop_daemon(project)
    data.write_text("42")

    lines = _hook(project).splitlines()

    (doubled,) = [line for line in lines if line.startswith("- doubled ")]
    assert doubled.endswith("stale, upstream")


def test_the_prompt_hook_says_only_the_cells_it_names(held):
    project, _ = held

    assert "- value " in _hook(project, "prompt", "plot value against time")
    assert _hook(project, "prompt", "rename the module") == ""


def test_a_directory_with_no_scratchpad_says_nothing(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")

    assert _hook(tmp_path) == ""


def test_a_digest_the_hook_cannot_read_says_nothing(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    digest = tmp_path / agent_hook.DIGEST
    digest.parent.mkdir(parents=True)
    digest.write_text('{"version": 1, "cel')

    assert _hook(tmp_path) == ""


def test_past_the_budget_a_cell_is_as_of_never_cached(held, monkeypatch):
    project, _ = held
    digest = json.loads((project / agent_hook.DIGEST).read_text())
    monkeypatch.setattr(agent_hook, "MAX_PATHS", 0)

    found = agent_hook.verdicts(digest, str(project))

    assert set(found.values()) == {
        "as of " + time.strftime("%Y-%m-%d %H:%M", time.localtime(digest["written_at"] / 1000))
    }


def test_a_base_path_still_absent_costs_no_budget(held, monkeypatch):
    """Most of the base is names nothing is at; only the cell's own reads count."""
    project, _ = held
    digest = json.loads((project / agent_hook.DIGEST).read_text())
    absent = {str(project / f"nothing{i}.py"): agent_hook.ABSENT for i in range(50)}
    # The folder's listing moved, so every base path is looked at.
    digest["environment"] = {
        "observed": {},
        "base": absent,
        "anchor": str(project),
        "anchor_listing": "list:before",
    }
    monkeypatch.setattr(agent_hook, "MAX_PATHS", 5)

    assert set(agent_hook.verdicts(digest, str(project)).values()) == {"cached"}


def test_an_environment_path_that_moved_makes_every_cell_stale(held):
    project, _ = held
    digest = json.loads((project / agent_hook.DIGEST).read_text())
    moved = project / "uv.lock"
    digest["environment"] = {"observed": {str(moved): [False, "sha256:before"]}, "base": {}}

    found = agent_hook.verdicts(digest, str(project))

    assert set(found.values()) == {"stale, environment changed (uv.lock)"}


@pytest.mark.parametrize("shadow", ["json.py", "json/__init__.py"])
def test_a_module_that_would_shadow_the_base_makes_every_cell_stale(held, shadow):
    """As the daemon's E check sees it: a file named like a preloaded module, in the
    project folder, is what an import there would find first."""
    project, _ = held
    digest = json.loads((project / agent_hook.DIGEST).read_text())
    assert any(p.startswith(str(project / "json")) for p in digest["environment"]["base"])
    (project / shadow).parent.mkdir(exist_ok=True)
    (project / shadow).write_text("")

    found = agent_hook.verdicts(digest, str(project))

    assert {v.partition(" (")[0] for v in found.values()} == {"stale, environment changed"}


def test_the_hook_is_quick(held, capsys):
    """Not asserted (no timing thresholds here); said, for the design's 100 ms."""
    project, _ = held
    started = time.perf_counter()
    for _ in range(5):
        _hook(project)
    with capsys.disabled():
        print(f"\nagent hook: {(time.perf_counter() - started) / 5 * 1000:.0f} ms a run")


# --- the copies the hook carries, pinned to their originals --------------------------


@pytest.fixture
def paths(tmp_path):
    file = tmp_path / "a.py"
    file.write_text("x = 1\n")
    folder = tmp_path / "pkg"
    folder.mkdir()
    (folder / "b.py").write_text("")
    (folder / ".hidden").write_text("")
    return [str(file), str(folder), str(tmp_path / "missing")]


def test_observe_is_code_reads_observe(paths):
    for path in paths:
        for listing in (False, True):
            ours = agent_hook.observe(path, listing=listing)
            theirs = code_reads.observe(path, listing=listing)
            assert ours[0] == theirs[0]
            assert (ours[1] is None) == (theirs[1] is None)
            if ours[1] is not None and theirs[1] is not None:
                assert ours[1][:5] == theirs[1][:5]


def test_watch_state_fingerprint_and_racy_are_code_reads(paths):
    taken = time.time_ns()
    for path in paths:
        for mode in (None, False, True):
            assert agent_hook.watch_state(path, mode) == code_reads.watch_state(path, mode)
        assert agent_hook.fingerprint(path, taken) == code_reads.fingerprint(path, taken)
    for print_ in ([1, 2, 3, 10, 10, 10 + agent_hook.RACY_NS], [1, 2, 3, 0, 0, 5], [0] * 6):
        assert agent_hook.racy(print_) == code_reads.racy(print_)
    assert (agent_hook.ABSENT, agent_hook.LARGE_BYTES, agent_hook.RACY_NS) == (
        code_reads.ABSENT,
        code_reads.LARGE_BYTES,
        code_reads.RACY_NS,
    )


@pytest.mark.parametrize(
    "layout",
    [
        ["pyproject.toml", "src/pkg/"],
        [".git/", "services/api/pyproject.toml", "services/api/src/"],
        [".git/", "docs/"],
        ["pyproject.toml", ".strata-scratch/work/"],
        ["notes/"],
    ],
)
def test_find_anchor_is_scratch_find_anchor(tmp_path, monkeypatch, layout):
    monkeypatch.setenv("HOME", str(Path.home()))
    root = Path.home() / "repo"
    for entry in layout:
        target = root / entry
        if entry.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("")
    deepest = max((root / e for e in layout if e.endswith("/")), key=lambda p: len(p.parts))

    for start in [root, deepest]:
        assert agent_hook.find_anchor(str(start)) == str(scratch.find_anchor(start))


def test_the_hook_imports_only_the_standard_library():
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1]); "
            "print(sorted(m for m in sys.modules if m.startswith('strata')))",
            str(HOOK),
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert done.stdout.strip() == "[]"
