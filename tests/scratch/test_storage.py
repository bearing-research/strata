"""A scratchpad's storage: swept by idle days then size, overage shown, every one listed."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
import tomllib

import pytest

from strata import scratch
from strata.cli import main
from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.notebook.scratch_storage import SAFE_TO_DELETE, read_storage, sweep
from strata.notebook.writer import create_notebook

DAY = 86400.0


@pytest.fixture
def store(tmp_path):
    """A scratchpad notebook with live cells ``a`` and ``b``, and its store."""
    notebook = create_notebook(tmp_path, "scratch", initialize_environment=False)
    toml = notebook / "notebook.toml"
    cells = '[[cells]]\nid = "a"\nfile = "a.py"\n\n[[cells]]\nid = "b"\nfile = "b.py"\n'
    toml.write_text(toml.read_text().replace("cells = []\n", "") + cells)
    notebook_id = tomllib.loads(toml.read_text())["notebook_id"]
    manager = NotebookArtifactManager(notebook_id, artifact_dir=notebook / ".strata" / "artifacts")
    return notebook, manager


def _values(manager, cell: str, sizes: list[int], used_days_ago: list[float]) -> str:
    """Versions 1..n of *cell*'s output, each *sizes* bytes and last used that many days ago."""
    store = manager.artifact_store
    artifact_id = manager.cell_artifact_id(cell, "x")
    for version, (size, ago) in enumerate(zip(sizes, used_days_ago, strict=True), start=1):
        store.create_artifact(artifact_id, f"prov-{cell}-{version}")
        store.write_blob(artifact_id, version, b"x")
        store.finalize_artifact(artifact_id, version, "{}", 1, size)
        conn = sqlite3.connect(store.db_path)
        try:
            when = time.time() - ago * DAY
            conn.execute(
                "UPDATE artifact_versions SET created_at = ?, last_used_at = ? "
                "WHERE id = ? AND version = ?",
                (when, when, artifact_id, version),
            )
            conn.commit()
        finally:
            conn.close()
    return artifact_id


def _kept(manager, artifact_id: str, versions: int) -> list[int]:
    return [
        v for v in range(1, versions + 1) if manager.artifact_store.get_artifact(artifact_id, v)
    ]


def _sweep(notebook, limit: int):
    return sweep(notebook, soft_limit_bytes=limit, max_idle_days=14, min_idle_seconds=0)


def test_under_the_limit_only_what_sat_unused_past_the_window_goes(store):
    notebook, manager = store
    # A superseded result used yesterday is the one a reverted edit hits.
    x = _values(manager, "a", [10, 10, 10], used_days_ago=[20, 1, 30])

    left = _sweep(notebook, limit=10**9)

    assert _kept(manager, x, 3) == [2, 3]
    assert left.over_by_bytes == 0


def test_over_the_limit_the_least_recently_used_go_first(store):
    notebook, manager = store
    x = _values(manager, "a", [100, 100, 100, 100], used_days_ago=[3, 1, 2, 5])

    left = _sweep(notebook, limit=250)

    # Down to 80% of the limit, least recent use first; the current value stays.
    assert _kept(manager, x, 4) == [2, 4]
    assert left.store_bytes == 200
    assert left.over_by_bytes == 0


def test_live_values_over_the_limit_are_kept_and_the_overage_recorded(store):
    notebook, manager = store
    a = _values(manager, "a", [300], used_days_ago=[60])
    b = _values(manager, "b", [100], used_days_ago=[60])

    left = _sweep(notebook, limit=200)

    assert _kept(manager, a, 1) == [1] and _kept(manager, b, 1) == [1]
    assert left.over_by_bytes == 200
    assert left.largest == [{"cell": "a", "bytes": 300}, {"cell": "b", "bytes": 100}]
    assert read_storage(notebook) == left


def test_the_registry_lists_each_scratchpad_until_its_directory_goes(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    here, there = tmp_path / "here", tmp_path / "there"
    for anchor in (here, there):
        scratch.scratch_dir(anchor).mkdir(parents=True)
        scratch.register(anchor, store_bytes=1, over_by_bytes=0, used=True)
    used = scratch.registered()[str(here)]["last_used"]

    # A sweep is not a use.
    scratch.register(here, store_bytes=5, over_by_bytes=2, used=False)
    assert scratch.registered()[str(here)] == {
        "store_bytes": 5,
        "over_by_bytes": 2,
        "last_used": used,
    }

    scratch.scratch_dir(there).rmdir()
    assert list(scratch.registered()) == [str(here)]


def test_the_daemon_keeps_every_result_for_its_own_sweep(tmp_path, monkeypatch):
    launched = {}

    def run(command, env, **kwargs):
        launched.update(env)
        return subprocess.CompletedProcess(command, 0, stdout="1\n")

    scratch.scratch_dir(tmp_path).joinpath(".strata").mkdir(parents=True)
    monkeypatch.setattr(scratch.subprocess, "run", run)
    scratch._spawn(tmp_path, 1)

    # Not the count kept per value on open: reverting an edit must still hit.
    assert launched["STRATA_NOTEBOOK_KEEP_SUPERSEDED_VERSIONS"] == "0"
    assert launched["STRATA_SCRATCH_ANCHOR"] == str(tmp_path)


def test_a_started_daemon_sweeps_and_lists_its_scratchpad(project):
    scratch.ensure_daemon(project)
    deadline = time.monotonic() + 120
    while str(project.resolve()) not in scratch.registered():
        assert time.monotonic() < deadline, "the daemon never swept its scratchpad"
        time.sleep(0.2)
    assert read_storage(scratch.scratch_dir(project.resolve())) is not None


def test_the_cli_sweeps_lists_and_removes_a_scratchpad(project, capsys):
    scratch._scaffold(project)
    anchor = str(project.resolve())

    assert main(["scratch", "gc", "--root", str(project)]) == 0
    swept = json.loads(capsys.readouterr().out)
    assert swept[anchor]["over_by_bytes"] == 0

    assert main(["scratch", "ls", "--format", "json"]) == 0
    (listed,) = json.loads(capsys.readouterr().out)
    assert (listed["anchor"], listed["running"]) == (anchor, False)
    assert main(["scratch", "ls"]) == 0
    assert SAFE_TO_DELETE in capsys.readouterr().out

    assert main(["scratch", "rm", "--root", str(project)]) == 0
    capsys.readouterr()
    assert not scratch.scratch_dir(project).exists()
    assert main(["scratch", "ls"]) == 0
    assert capsys.readouterr().out.strip() == "no scratchpads"


@pytest.mark.asyncio
async def test_a_retirement_that_fails_does_not_stop_the_sweep(tmp_path, monkeypatch):
    import asyncio

    from strata import server
    from strata.config import StrataConfig
    from strata.notebook import scratch_storage

    swept = asyncio.Event()

    async def failing(*args, **kwargs):
        raise OSError("notebook.toml went away")

    monkeypatch.setattr(scratch_storage, "retire_in_session", failing)
    monkeypatch.setattr(scratch_storage, "upkeep", lambda *a, **k: swept.set())
    monkeypatch.setattr(scratch_storage, "session_holding", lambda *a: ("s", object()))
    config = StrataConfig(cache_dir=tmp_path / "cache", artifact_dir=tmp_path / "a")
    task = asyncio.create_task(server._scratch_upkeep_loop(tmp_path, config))
    try:
        await asyncio.wait_for(swept.wait(), timeout=30)
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_the_first_pass_waits_for_the_scratchpads_session(tmp_path, monkeypatch):
    """The client opens it just after the start, and an idle daemon stops within the hour."""
    import asyncio

    from strata import server
    from strata.config import StrataConfig
    from strata.notebook import scratch_storage

    opened = asyncio.Event()
    asked = []

    def holding(*args):
        asked.append(opened.is_set())
        return ("s", object()) if opened.is_set() else None

    retired = asyncio.Event()

    async def retire(*args, **kwargs):
        assert opened.is_set()
        retired.set()
        return []

    monkeypatch.setattr(scratch_storage, "session_holding", holding)
    monkeypatch.setattr(scratch_storage, "retire_in_session", retire)
    monkeypatch.setattr(scratch_storage, "upkeep", lambda *a, **k: None)
    config = StrataConfig(cache_dir=tmp_path / "cache", artifact_dir=tmp_path / "a")
    task = asyncio.create_task(server._scratch_upkeep_loop(tmp_path, config))
    try:
        async with asyncio.timeout(30):
            while len(asked) < 2:
                await asyncio.sleep(0.05)
        assert not retired.is_set()
        opened.set()
        await asyncio.wait_for(retired.wait(), timeout=30)
    finally:
        task.cancel()


def test_a_zero_limit_is_no_limit(store):
    notebook, manager = store
    held = _values(manager, "a", [100, 100], used_days_ago=[1, 0])
    storage = _sweep(notebook, 0)
    assert storage.over_by_bytes == 0
    assert _kept(manager, held, 2) == [1, 2]


def test_the_largest_are_live_cells_only():
    from types import SimpleNamespace

    from strata.notebook.scratch_storage import _largest_cells

    latest = [SimpleNamespace(id="live", byte_size=10), SimpleNamespace(id="gone", byte_size=99)]
    manager = SimpleNamespace(
        notebook_id="nb",
        artifact_store=SimpleNamespace(list_latest_by_id_prefix=lambda prefix: latest),
        cell_output_of=lambda artifact_id: (artifact_id, "v"),
    )
    # A retired cell's value something still holds is not the agent's to report.
    assert _largest_cells(manager, ["live"]) == [{"cell": "live", "bytes": 10}]
