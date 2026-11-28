"""What the notebook lists of a cell's checkpoints, and a branch started from one over the socket.

Each cell runs in the real harness, as in Gate 1 (``test_local_runs``), whose notebook this
borrows.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient
from vestigo.provenance import TransformSpec

from strata.notebook.checkpoints import cell_checkpoints
from strata.notebook.local_runs import LocalRun, LocalRuns
from tests.notebook.e2e_fixtures import _reset_ws_globals
from tests.notebook.test_local_runs import TRAIN, TRAIN_FILES, _ledger, _notebook, _only_build, _run
from tests.notebook.test_ws import _drain_execution, _make_fake_ws

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="no event pipe on Windows")

# TRAIN without its one crash.
WHOLE = TRAIN.replace("epoch == 1", "epoch == 99")


def _listed(session, cell_id: str = "train"):
    return cell_checkpoints(session.get_artifact_manager(), cell_id)


def test_a_runs_checkpoints_are_listed_newest_first_with_what_each_holds(
    tmp_path: Path, monkeypatch
):
    session, executor = _notebook(tmp_path, WHOLE)
    source = session.notebook_state.get_cell("train").source
    assert _run(executor, "train", source).success is True
    _, _, releases = _ledger(session)
    (build,) = _only_build(session, "train")
    by_sequence = {r.sequence: r for r in releases.releases_of(build.build_id, "checkpoint")}

    looked_up = []
    get = releases.get
    monkeypatch.setattr(releases, "get", lambda rid: looked_up.append(rid) or get(rid))

    (run,) = _listed(session)

    # One run's releases are read together, not once per checkpoint.
    assert len(looked_up) == 1
    assert run["build_id"] == build.build_id
    assert run["status"] == "finished"
    assert run["variant"] is None and run["init"] is None
    assert run["started_at"] == build.created_at
    assert [c["sequence"] for c in run["checkpoints"]] == [4, 3, 2, 1]
    assert [c["release_id"] for c in run["checkpoints"]] == [
        by_sequence[n].release_id for n in (4, 3, 2, 1)
    ]
    newest = run["checkpoints"][0]
    assert newest["parts"] == ["metrics", "model", "optimizer_state"]
    assert newest["files"] == 0 and newest["step"] is None
    outputs = releases.bundle(newest["release_id"]).outputs
    assert newest["byte_size"] == sum(v.byte_size for v in outputs) > 0
    assert newest["created_at"] == by_sequence[4].settled_at
    # Epoch 3 had the best loss, so the head ended on its checkpoint.
    assert [c["heads"] for c in run["checkpoints"]] == [["best-validation"], [], [], []]
    # A cell that never checkpointed has nothing to list.
    assert _listed(session, "features") == []


def test_a_branch_is_listed_as_its_own_run_with_the_release_it_started_from(tmp_path: Path):
    session, executor = _notebook(tmp_path, WHOLE)
    source = session.notebook_state.get_cell("train").source
    assert _run(executor, "train", source).success is True
    (trunk,) = _listed(session)
    second = next(c for c in trunk["checkpoints"] if c["sequence"] == 2)

    assert _run(executor, "train", source, init=second["release_id"]).success is True

    branch, trunk_again = _listed(session)
    assert branch["init"] == second["release_id"] and trunk_again["init"] is None
    assert trunk_again["build_id"] == trunk["build_id"]
    # The same ids at new versions: the branch's two checkpoints are its own, not the trunk's.
    assert [c["sequence"] for c in branch["checkpoints"]] == [2, 1]
    assert [c["sequence"] for c in trunk_again["checkpoints"]] == [4, 3, 2, 1]
    assert {c["release_id"] for c in branch["checkpoints"]}.isdisjoint(
        c["release_id"] for c in trunk_again["checkpoints"]
    )
    # One head per cell: the branch's best epoch took it from the trunk's.
    assert [c["heads"] for c in branch["checkpoints"]] == [["best-validation"], []]
    assert all(c["heads"] == [] for c in trunk_again["checkpoints"])


def test_a_run_that_died_is_listed_as_interrupted_until_it_is_resumed(tmp_path: Path):
    session, executor = _notebook(tmp_path)
    source = session.notebook_state.get_cell("train").source
    assert _run(executor, "train", source).success is False

    (run,) = _listed(session)
    assert run["status"] == "interrupted"
    assert [c["sequence"] for c in run["checkpoints"]] == [2, 1]

    assert _run(executor, "train", source).success is True
    (resumed,) = _listed(session)
    assert resumed["build_id"] == run["build_id"] and resumed["status"] == "finished"
    assert [c["sequence"] for c in resumed["checkpoints"]] == [4, 3, 2, 1]


def test_a_rerun_fails_the_run_it_replaces_and_both_are_listed(tmp_path: Path):
    session, executor = _notebook(tmp_path)
    source = session.notebook_state.get_cell("train").source
    assert _run(executor, "train", source).success is False

    assert asyncio.run(executor.execute_cell_rerun("train", source)).success is True

    fresh, died = _listed(session)
    assert (fresh["status"], died["status"]) == ("finished", "failed")
    assert [c["sequence"] for c in fresh["checkpoints"]] == [4, 3, 2, 1]
    assert [c["sequence"] for c in died["checkpoints"]] == [2, 1]


def _open_run(session, **kwargs) -> tuple[LocalRuns, LocalRun]:
    """A run of ``train`` opened on the ledger alone, its lease live, with no harness."""
    runs = LocalRuns(session.get_artifact_manager())
    run = runs.open(
        cell_id="train",
        computation_key="a-key",
        input_versions={},
        timeout_seconds=60.0,
        resume=False,
        **{"variant": None, **kwargs},
    )
    return runs, run


def _checkpoint(runs: LocalRuns, run: LocalRun, run_dir: Path, sequence: int, **parts) -> None:
    spec = {}
    for name, value in parts.items():
        (run_dir / f"{name}.json").write_text(json.dumps(value))
        spec[name] = {"file": f"{name}.json", "content_type": "json/object"}
    runs.handle(run, {"event": "checkpoint", "sequence": sequence, "parts": spec}, run_dir)


def test_a_run_still_going_is_listed_as_running_under_its_variant(tmp_path: Path):
    session, _ = _notebook(tmp_path)
    runs, run = _open_run(session, variant="lr=0.1")
    _checkpoint(runs, run, tmp_path, 1, model={"w": 0})
    runs.handle(run, {"event": "head", "name": "best", "sequence": 1}, tmp_path)

    (listed,) = _listed(session)

    assert (listed["build_id"], listed["status"], listed["variant"]) == (
        run.build_id,
        "running",
        "lr=0.1",
    )
    (checkpoint,) = listed["checkpoints"]
    assert (checkpoint["parts"], checkpoint["heads"]) == (["model"], ["best"])


def test_a_run_is_running_by_the_ledgers_clock_not_this_hosts(tmp_path: Path):
    session, _ = _notebook(tmp_path)
    _, builds, releases = _ledger(session)
    for store in (builds, releases):
        store._clock = lambda: time.time() - 3600.0
    runs, run = _open_run(session)
    _checkpoint(runs, run, tmp_path, 1, model={"w": 0})

    (listed,) = _listed(session)

    assert listed["status"] == "running"


def test_an_output_made_ready_outside_its_release_is_not_a_checkpoint(tmp_path: Path):
    session, _ = _notebook(tmp_path)
    runs, run = _open_run(session)
    _checkpoint(runs, run, tmp_path, 1, model={"w": 0})
    manager = session.get_artifact_manager()
    store = manager.artifact_store
    # A release prepared and never accepted, though its output was finalized: ``init``
    # takes an accepted release only.
    pending = manager.release_store.prepare(run.build_id, run.attempt, "checkpoint", sequence=2)
    artifact_id = manager.cell_artifact_id("train", "model") + "@ckpt=2"
    version = store.create_artifact(
        artifact_id,
        "a-staged-output",
        TransformSpec(executor="notebook/cell@v1", params={"variable_name": "model"}, inputs=[]),
        release_id=pending.release_id,
    )
    store.blob_store.write_blob(artifact_id, version, b"{}")
    assert store.finalize_artifact(artifact_id, version, "", None, 2).state == "ready"

    (listed,) = _listed(session)

    assert [c["sequence"] for c in listed["checkpoints"]] == [1]


def test_a_checkpoint_accepted_while_the_listing_reads_waits_for_the_next_one(
    tmp_path: Path, monkeypatch
):
    session, _ = _notebook(tmp_path)
    runs, run = _open_run(session)
    _checkpoint(runs, run, tmp_path, 1, model={"w": 0})
    _checkpoint(runs, run, tmp_path, 2, model={"w": 1}, optimizer=[1])
    store = session.get_artifact_manager().artifact_store
    scan = store.list_versions_by_id_prefix

    def mid_acceptance(prefix: str):
        # The versions as a read before checkpoint 2's acceptance saw them: one part
        # staged, the other not yet there. The release reads that follow see it accepted.
        return [
            replace(v, state="building") if "@ckpt=2" in v.id else v
            for v in scan(prefix)
            if "optimizer" not in v.id
        ]

    monkeypatch.setattr(store, "list_versions_by_id_prefix", mid_acceptance)
    (listed,) = _listed(session)
    assert [c["sequence"] for c in listed["checkpoints"]] == [1]

    monkeypatch.undo()
    (listed,) = _listed(session)
    assert [(c["sequence"], c["parts"]) for c in listed["checkpoints"]] == [
        (2, ["model", "optimizer"]),
        (1, ["model"]),
    ]


def test_a_checkpoint_prune_collected_is_no_longer_listed(tmp_path: Path):
    session, executor = _notebook(tmp_path, WHOLE)
    assert _run(executor, "train", session.notebook_state.get_cell("train").source).success

    session.get_artifact_manager().prune(keep_superseded=0, min_idle_seconds=0)

    (run,) = _listed(session)
    assert [(c["sequence"], c["heads"]) for c in run["checkpoints"]] == [(4, ["best-validation"])]


def test_a_checkpoint_of_files_lists_its_step_and_how_many_files(tmp_path: Path):
    session, executor = _notebook(tmp_path, TRAIN_FILES.replace("epoch == 1", "epoch == 99"))
    assert _run(executor, "train", session.notebook_state.get_cell("train").source).success

    (run,) = _listed(session)

    assert [(c["sequence"], c["step"], c["files"]) for c in run["checkpoints"]] == [
        (4, 3, 2),
        (3, 2, 2),
        (2, 1, 2),
        (1, 0, 2),
    ]
    assert all(c["parts"] == [] for c in run["checkpoints"])
    # epoch (one digit) and model/w.bin (four bytes), and the manifest that lists them.
    assert all(c["byte_size"] > 5 for c in run["checkpoints"])


@pytest.fixture
def served(tmp_path: Path, monkeypatch):
    """A Gate 1 notebook that has run, open on the app's session manager."""
    from strata.notebook import routes
    from strata.notebook.session import SessionManager

    session, executor = _notebook(tmp_path, WHOLE)
    assert _run(executor, "train", session.notebook_state.get_cell("train").source).success
    manager = SessionManager()
    manager._sessions[session.id] = session
    monkeypatch.setattr(routes, "_session_manager", manager)
    _reset_ws_globals()
    yield session
    _reset_ws_globals()


def test_the_route_lists_a_cells_checkpoints(served):
    from strata.server import app

    with TestClient(app) as client:
        listed = client.get(f"/v1/notebooks/{served.id}/cells/train/checkpoints")
        none = client.get(f"/v1/notebooks/{served.id}/cells/features/checkpoints")
        missing = client.get(f"/v1/notebooks/{served.id}/cells/ghost/checkpoints")

    assert listed.status_code == 200
    assert listed.json() == {"cell_id": "train", "runs": _listed(served)}
    assert [c["sequence"] for c in listed.json()["runs"][0]["checkpoints"]] == [4, 3, 2, 1]
    assert none.json() == {"cell_id": "features", "runs": []}
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_a_cell_execute_frame_with_init_branches_from_that_release(served):
    from strata.notebook.ws import _handle_cell_execute

    (trunk,) = _listed(served)
    second = next(c for c in trunk["checkpoints"] if c["sequence"] == 2)
    fake, execution_state = _make_fake_ws(served)

    await _handle_cell_execute(
        cast(WebSocket, fake),
        served,
        {"cell_id": "train", "init": second["release_id"]},
        execution_state,
        served.id,
    )
    await _drain_execution(execution_state)

    assert not fake.frames_of("cell_error"), fake.frames_of("cell_error")
    branch, _ = _listed(served)
    assert branch["init"] == second["release_id"] and branch["status"] == "finished"
    # From epoch 1's state: epochs 2 and 3, not all four again.
    assert [c["sequence"] for c in branch["checkpoints"]] == [2, 1]


@pytest.mark.asyncio
async def test_a_branch_with_an_upstream_to_run_runs_it_without_a_cascade_prompt(served):
    from strata.notebook.models import CellStatus
    from strata.notebook.ws import _handle_cell_execute

    (trunk,) = _listed(served)
    second = next(c for c in trunk["checkpoints"] if c["sequence"] == 2)
    served.notebook_state.get_cell("features").status = CellStatus.IDLE
    fake, execution_state = _make_fake_ws(served)

    # An ordinary Run would ask first, and the yes would come back with no init.
    await _handle_cell_execute(
        cast(WebSocket, fake),
        served,
        {"cell_id": "train", "init": second["release_id"]},
        execution_state,
        served.id,
    )
    await _drain_execution(execution_state)

    assert fake.frames_of("cascade_prompt") == []
    assert [run["init"] for run in _listed(served)] == [second["release_id"], None]


@pytest.mark.asyncio
async def test_a_cell_execute_frame_whose_init_is_not_a_release_id_runs_nothing(served):
    from strata.notebook.ws import _handle_cell_execute

    fake, execution_state = _make_fake_ws(served)

    for not_an_id in (2, ""):
        await _handle_cell_execute(
            cast(WebSocket, fake),
            served,
            {"cell_id": "train", "init": not_an_id},
            execution_state,
            served.id,
        )
        await _drain_execution(execution_state)

    first, second = fake.frames_of("error")
    assert "init" in first["payload"]["error"] and "init" in second["payload"]["error"]
    assert execution_state.execution_task is None and execution_state.requested_cell is None
    # Neither started a run: the one build is the fixture's.
    assert len(_only_build(served, "train")) == 1


@pytest.mark.asyncio
async def test_a_branch_of_a_ready_cell_asks_before_a_large_read_and_stays_a_branch(
    tmp_path: Path, monkeypatch
):
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import CellStatus
    from strata.notebook.ws import _handle_cell_execute, _handle_read_confirm
    from tests.notebook.test_table_index import _session, _warehouse

    config = StrataConfig(cache_dir=tmp_path / "cache", notebook_confirm_read_bytes=0)
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
    _, uri = _warehouse(tmp_path)
    session = _session(tmp_path, {"c1": f"# @table events {uri}\nsnap = events_snapshot\n"})
    # Run headless, unasked: an ordinary Run of the ready cell is a cache hit and asks nothing.
    await CellExecutor(session).execute_cell("c1", session.notebook_state.cells[0].source)
    session.notebook_state.get_cell("c1").status = CellStatus.READY
    _reset_ws_globals()
    fake, execution_state = _make_fake_ws(session)
    try:
        await _handle_cell_execute(
            cast(WebSocket, fake),
            session,
            {"cell_id": "c1", "init": "no-such-release"},
            execution_state,
            session.id,
        )
        (prompt,) = fake.frames_of("read_confirm_prompt")
        assert execution_state.execution_task is None
        await _handle_read_confirm(
            cast(WebSocket, fake),
            session,
            {"prompt_id": prompt["payload"]["prompt_id"]},
            execution_state,
            session.id,
        )
        await _drain_execution(execution_state)
    finally:
        _reset_ws_globals()

    assert [cell["cell_id"] for cell in prompt["payload"]["cells"]] == ["c1"]
    # The run the yes started is still the branch: it looked for the release.
    (error,) = fake.frames_of("cell_error")
    assert "no-such-release" in error["payload"]["error"]
