"""Gate 1 (ADR 002 §8): a local cell run is an execution with checkpoints, resume and branch.

Each cell runs in the real harness; the ledger is the notebook's own store. The
training loop is a toy so the gate is about the lifecycle, not the model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from pathlib import Path

import pytest
from vestigo.provenance import checkpoint_output_key, derive_subkey, run_output_key
from vestigo.releases import ReleaseRejected

from strata.notebook.executor import CellExecutor
from strata.notebook.local_runs import LocalRun, LocalRuns, RunFenced, RunRefused
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="no event pipe on Windows")

# Four epochs, a checkpoint each; the first run dies after its second checkpoint, once.
TRAIN = """
import pathlib
state = strata.restore()
start = state["metrics"]["epoch"] + 1 if state else 0
best = state["metrics"].get("best", 9.0) if state else 9.0
for epoch in range(start, 4):
    loss = [0.9, 0.3, 0.5, 0.2][epoch]
    if loss < best:
        best = loss
    seq = strata.checkpoint(
        model={"w": epoch}, optimizer_state=[epoch], metrics={"epoch": epoch, "best": best}
    )
    if loss == best:
        strata.head("best-validation", seq)
    if epoch == 1 and not pathlib.Path("crashed").exists():
        pathlib.Path("crashed").write_text("1")
        raise RuntimeError("killed mid-run")
model = {"w": 3, "snapshot": features["snapshot"]}
metrics = {"epochs": 4, "best": best}
"""

# The feature cell reads a lake table: the run's structural node gets a ``strata/table`` leaf.
FEATURES = """
# @table events {uri}
features = {{"snapshot": events_snapshot}}
"""

EVALUATE = """
score = model["w"] * metrics["epochs"]
"""


def _dialect():
    """The ledger's SQL dialect; the Postgres run of this gate replaces it."""
    return None


def _notebook(tmp_path: Path, train: str = TRAIN) -> tuple[NotebookSession, CellExecutor]:
    from tests.notebook.test_table_index import _warehouse

    _, table_uri = _warehouse(tmp_path)
    notebook_dir = create_notebook(tmp_path, "Gate 1", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "features")
    write_cell(notebook_dir, "features", FEATURES.format(uri=table_uri))
    add_cell_to_notebook(notebook_dir, "train", "features")
    write_cell(notebook_dir, "train", train)
    add_cell_to_notebook(notebook_dir, "evaluate", "train")
    write_cell(notebook_dir, "evaluate", EVALUATE)
    # Only a consumed variable is stored, so a reader of ``score`` follows.
    add_cell_to_notebook(notebook_dir, "report", "evaluate")
    write_cell(notebook_dir, "report", "report = score\n")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir, ledger_dialect=_dialect())
    session.refresh_environment_runtime()
    return session, CellExecutor(session)


def _ledger(session: NotebookSession):
    manager = session.get_artifact_manager()
    return manager.artifact_store, manager.build_store, manager.release_store


def _only_build(session: NotebookSession, cell_id: str):
    store, builds, _ = _ledger(session)
    conn = builds._get_connection()
    try:
        rows = conn.execute(
            "SELECT build_id FROM artifact_builds WHERE artifact_id LIKE ? ORDER BY created_at",
            (f"%_cell_{cell_id}_%",),
        ).fetchall()
    finally:
        conn.close()
    assert rows, f"no build for cell {cell_id}"
    return [builds.get_build(row["build_id"]) for row in rows]


def _run(executor: CellExecutor, cell_id: str, source: str, **kwargs):
    return asyncio.run(executor.execute_cell(cell_id, source, **kwargs))


class TestGate1:
    def test_a_training_cell_publishes_checkpoints_while_it_runs(self, tmp_path: Path):
        session, executor = _notebook(tmp_path, TRAIN.replace("epoch == 1", "epoch == 99"))
        result = _run(executor, "train", session.notebook_state.get_cell("train").source)
        assert result.success is True, result.error

        store, builds, releases = _ledger(session)
        (build,) = _only_build(session, "train")
        checkpoints = [r for r in releases.releases_of(build.build_id, "checkpoint")]
        assert [r.sequence for r in checkpoints] == [1, 2, 3, 4]
        assert all(r.state == "accepted" for r in checkpoints)
        for release in checkpoints:
            bundle = releases.bundle(release.release_id)
            names = sorted(v.id.split("_var_")[1].split("@")[0] for v in bundle.outputs)
            assert names == ["metrics", "model", "optimizer_state"]
            assert {v.release_id for v in bundle.outputs} == {release.release_id}
            assert all(v.state == "ready" for v in bundle.outputs)
            assert all(v.id.endswith(f"@ckpt={release.sequence}") for v in bundle.outputs)
            # The build was building while each checkpoint was accepted; it is ready now.
            assert release.created_at <= build.completed_at
        assert build.state == "ready"

        # The final release holds the cell's outputs, under the cached keys, with the
        # build's computation key on every row of the run.
        (final,) = releases.releases_of(build.build_id, "final")
        outputs = releases.bundle(final.release_id).outputs
        assert sorted(v.id.split("_var_")[1] for v in outputs) == ["metrics", "model"]
        parent = build.computation_key
        assert parent and all(v.computation_key == parent for v in outputs)
        assert {v.provenance_hash for v in outputs} == {
            derive_subkey(parent, "model"),
            derive_subkey(parent, "metrics"),
        }
        first = releases.bundle(checkpoints[0].release_id).outputs
        assert all(v.computation_key == parent for v in first)
        assert checkpoint_output_key(parent, build.build_id, 1, "model") in {
            v.provenance_hash for v in first
        }
        # The placeholder the build targeted first is a failed row, not a current value.
        placeholder = store.get_artifact(f"nb_{session.notebook_state.id}_cell_train_run", 1)
        assert placeholder is not None and placeholder.state == "failed"
        assert (build.artifact_id, build.version) == (outputs[0].id, outputs[0].version)

    def test_a_run_killed_after_a_checkpoint_resumes_in_a_new_attempt(self, tmp_path: Path):
        session, executor = _notebook(tmp_path)
        source = session.notebook_state.get_cell("train").source
        crashed = _run(executor, "train", source)
        assert crashed.success is False and "killed mid-run" in (crashed.error or "")

        store, builds, releases = _ledger(session)
        (build,) = _only_build(session, "train")
        assert build.state == "building", "a run with a checkpoint stays resumable"
        old_attempt = build.attempt
        assert [r.sequence for r in releases.releases_of(build.build_id, "checkpoint")] == [1, 2]

        # The same computation, run normally again: no cache hit, the execution continues.
        resumed = _run(executor, "train", source)
        assert resumed.success is True, resumed.error
        assert resumed.cache_hit is False

        (same_build,) = _only_build(session, "train")
        assert same_build.build_id == build.build_id and same_build.state == "ready"
        assert same_build.attempt != old_attempt
        attempt = builds.get_attempt(build.build_id, same_build.attempt)
        second = next(
            r for r in releases.releases_of(build.build_id, "checkpoint") if r.sequence == 2
        )
        assert attempt.resumed_from == second.release_id, "bound to an explicit release"
        assert [r.sequence for r in releases.releases_of(build.build_id, "checkpoint")] == [
            1,
            2,
            3,
            4,
        ]
        assert releases.bundle(second.release_id).outputs[0].state == "ready"

        # The old attempt can no longer publish: the ledger refuses it, and the executor
        # ends a run whose line arrives late.
        with pytest.raises(ReleaseRejected):
            releases.prepare(build.build_id, old_attempt, "checkpoint")
        stale = LocalRun(
            build_id=build.build_id,
            attempt=old_attempt,
            cell_id="train",
            computation_key=build.computation_key,
            variant=None,
            input_versions={},
        )
        with pytest.raises(RunFenced):
            LocalRuns(session.get_artifact_manager()).handle(
                stale, {"event": "checkpoint", "sequence": 9, "parts": {}}, tmp_path
            )

    def test_a_rerun_starts_a_new_execution(self, tmp_path: Path):
        session, executor = _notebook(tmp_path)
        source = session.notebook_state.get_cell("train").source
        _run(executor, "train", source)
        (interrupted,) = _only_build(session, "train")

        result = asyncio.run(executor.execute_cell_rerun("train", source))
        assert result.success is True, result.error
        store, builds, releases = _ledger(session)
        old, new = _only_build(session, "train")
        assert old.build_id == interrupted.build_id and old.state == "failed"
        assert new.state == "ready"
        # A fresh execution restored nothing and ran every epoch.
        assert [r.sequence for r in releases.releases_of(new.build_id, "checkpoint")] == [
            1,
            2,
            3,
            4,
        ]

    def test_a_branch_is_a_new_execution_from_a_release_with_the_same_definition(
        self, tmp_path: Path
    ):
        session, executor = _notebook(tmp_path, TRAIN.replace("epoch == 1", "epoch == 99"))
        source = session.notebook_state.get_cell("train").source
        assert _run(executor, "train", source).success is True
        store, builds, releases = _ledger(session)
        (trunk,) = _only_build(session, "train")
        second = next(
            r for r in releases.releases_of(trunk.build_id, "checkpoint") if r.sequence == 2
        )
        trunk_final = releases.bundle(releases.releases_of(trunk.build_id, "final")[0].release_id)

        branched = _run(executor, "train", source, init=second.release_id)
        assert branched.success is True, branched.error
        assert branched.cache_hit is False

        trunk_again, branch = _only_build(session, "train")
        assert trunk_again.build_id == trunk.build_id
        assert branch.build_id != trunk.build_id and branch.state == "ready"
        assert branch.params == {"init": second.release_id}
        assert branch.computation_key == derive_subkey(
            trunk.computation_key, f"init={second.release_id}"
        )
        # Started from epoch 1's state: its own sequence, two checkpoints for epochs 2 and 3.
        assert [r.sequence for r in releases.releases_of(branch.build_id, "checkpoint")] == [1, 2]
        branch_final = releases.bundle(releases.releases_of(branch.build_id, "final")[0].release_id)
        assert {v.provenance_hash for v in branch_final.outputs}.isdisjoint(
            {v.provenance_hash for v in trunk_final.outputs}
        )
        definition = {
            store.definition_of(v.id, v.version) for v in trunk_final.outputs + branch_final.outputs
        }
        assert len(definition) == 1 and None not in definition

    def test_the_evaluation_cell_reads_model_and_metrics_from_one_release(self, tmp_path: Path):
        session, executor = _notebook(tmp_path, TRAIN.replace("epoch == 1", "epoch == 99"))
        source = session.notebook_state.get_cell("train").source
        assert _run(executor, "train", source).success is True
        evaluated = _run(executor, "evaluate", EVALUATE)
        assert evaluated.success is True, evaluated.error

        store, builds, releases = _ledger(session)
        score = store.get_latest_version(f"nb_{session.notebook_state.id}_cell_evaluate_var_score")
        inputs = json.loads(score.input_versions or "{}")
        read = {
            uri.split("_var_")[1].split("@v=")[0]: store.get_artifact(
                uri.removeprefix("strata://artifact/").split("@v=")[0],
                int(uri.split("@v=")[1]),
            )
            for uri in inputs
        }
        assert set(read) == {"model", "metrics"}
        assert read["model"].release_id == read["metrics"].release_id is not None

        # ADR 002 §8 condition 6: from the evaluation cell's node the structure reaches the
        # training definition, the feature cell and the table leaf; from the leaf, every
        # lineage. The notebook's DAG is never consulted: the walk is the ledger's.
        evaluate_node = store.definition_of(score.id, score.version)
        train_node = store.definition_of(read["model"].id, read["model"].version)
        features = store.get_latest_version(
            f"nb_{session.notebook_state.id}_cell_features_var_features"
        )
        features_node = store.definition_of(features.id, features.version)
        ancestors = store.graph.ancestors([evaluate_node])
        assert train_node in ancestors and features_node in ancestors
        leaves = [n for n in ancestors if store.graph.get(n).domain == "strata/table"]
        assert len(leaves) == 1
        (table_leaf,) = leaves
        uri, location = store.graph.get(table_leaf).payload.decode().split("\n")
        assert uri.endswith("#db.events")
        assert location.endswith("/warehouse/db/events"), "where the catalog found the table"
        # The read is an input the record names, at the snapshot it resolved.
        recorded = json.loads(features.input_versions or "{}")
        assert recorded[uri].isdigit(), recorded
        dependents = store.graph.dependents(table_leaf)
        assert {features_node, train_node, evaluate_node} <= set(dependents)

    def test_prune_keeps_the_head_and_the_final_checkpoint_and_collects_the_rest(
        self, tmp_path: Path
    ):
        session, executor = _notebook(tmp_path, TRAIN.replace("epoch == 1", "epoch == 99"))
        source = session.notebook_state.get_cell("train").source
        assert _run(executor, "train", source).success is True
        store, builds, releases = _ledger(session)
        (build,) = _only_build(session, "train")
        checkpoints = releases.releases_of(build.build_id, "checkpoint")
        best = next(r for r in checkpoints if r.sequence == 4)  # epoch 3, loss 0.2
        runs = LocalRuns(session.get_artifact_manager())
        head = runs.head_name(
            LocalRun(build.build_id, build.attempt, "train", build.computation_key, None, {}),
            "best-validation",
        )
        assert releases.reference(head)[0] == best.release_id

        session.get_artifact_manager().prune(keep_superseded=0, min_idle_seconds=0)

        kept = {r.sequence for r in checkpoints if releases.bundle(r.release_id).outputs}
        assert kept == {4}
        (final,) = releases.releases_of(build.build_id, "final")
        assert all(v.state == "ready" for v in releases.bundle(final.release_id).outputs)

    def test_a_computation_already_running_is_refused(self, tmp_path: Path):
        session, _ = _notebook(tmp_path)
        runs = LocalRuns(session.get_artifact_manager())
        first = runs.open(
            cell_id="train",
            computation_key="parent",
            variant=None,
            input_versions={},
            timeout_seconds=30,
            resume=True,
        )
        with pytest.raises(RunRefused, match="already running"):
            runs.open(
                cell_id="train",
                computation_key="parent",
                variant=None,
                input_versions={},
                timeout_seconds=30,
                resume=True,
            )
        assert runs.builds.get_build(first.build_id).state == "building"


STOCHASTIC = """
# @stochastic
import random
sample = random.random()
"""


def test_a_child_the_cell_leaves_behind_does_not_hold_the_run_open(tmp_path: Path):
    """The harness keeps the event pipe from the cell's children, and the executor stops
    reading soon after the harness exits either way, so a lingering child cannot stall the
    run (CI lost xdist workers to this: pytest-timeout exits the process)."""
    session, executor = _notebook(tmp_path)
    pid_file = tmp_path / "child.pid"
    source = (
        "import subprocess\n"
        "child = subprocess.Popen(['sleep', '60'], close_fds=False, stdin=subprocess.DEVNULL, "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "model, metrics = {'w': 1}, {'epochs': 1}\n"
    )
    try:
        outcome = asyncio.run(asyncio.wait_for(executor.execute_cell("train", source), timeout=45))
    finally:
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
    assert outcome.success is True, outcome.error


def test_a_stochastic_cell_persists_every_run_and_never_serves_the_cache(tmp_path: Path):
    notebook_dir = create_notebook(tmp_path, "Dice", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "roll")
    write_cell(notebook_dir, "roll", STOCHASTIC)
    add_cell_to_notebook(notebook_dir, "use", "roll")
    write_cell(notebook_dir, "use", "doubled = sample * 2")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)

    first = asyncio.run(executor.execute_cell("roll", STOCHASTIC))
    second = asyncio.run(executor.execute_cell("roll", STOCHASTIC))
    assert first.success and second.success
    assert second.cache_hit is False and second.execution_method != "cached"

    store, builds, releases = _ledger(session)
    one, two = _only_build(session, "roll")
    assert {one.state, two.state} == {"ready"} and one.computation_key == two.computation_key
    sample_id = f"nb_{session.notebook_state.id}_cell_roll_var_sample"
    rows = [store.get_artifact(sample_id, v) for v in (1, 2)]
    assert [r.state for r in rows] == ["ready", "ready"], "both runs persist"
    assert {r.provenance_hash for r in rows} == {
        run_output_key(one.computation_key, one.build_id, "sample"),
        run_output_key(two.computation_key, two.build_id, "sample"),
    }
    assert store.find_by_provenance(derive_subkey(one.computation_key, "sample")) is None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.warm_pool
async def test_a_warm_worker_publishes_checkpoints_on_its_protocol_stream(tmp_path, warm_pool):
    session, cold = _notebook(tmp_path, TRAIN.replace("epoch == 1", "epoch == 99"))
    # The feature cell takes the pool's one warm process otherwise; run it first.
    features = session.notebook_state.get_cell("features").source
    assert (await cold.execute_cell("features", features)).success is True
    pool = await warm_pool(session)
    executor = CellExecutor(session, pool)
    source = session.notebook_state.get_cell("train").source

    result = await executor.execute_cell("train", source)

    assert result.success is True, result.error
    assert result.execution_method == "warm"
    store, builds, releases = _ledger(session)
    (build,) = _only_build(session, "train")
    assert build.state == "ready"
    assert [r.sequence for r in releases.releases_of(build.build_id, "checkpoint")] == [1, 2, 3, 4]
