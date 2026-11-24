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
import threading
import time
from pathlib import Path

import pytest
from vestigo import checkpoint_files
from vestigo.provenance import (
    TransformSpec,
    checkpoint_output_key,
    derive_subkey,
    run_output_key,
)
from vestigo.releases import ReleaseRejected

from strata.notebook.executor import CellExecutor
from strata.notebook.local_runs import LocalRun, LocalRuns, RunFenced, RunRefused, lease_owner
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

# TRAIN as a checkpoint of files, what vestigo.torch.HostedRun saves through.
TRAIN_FILES = """
import pathlib
folder = strata.restore_files()
# A checkpoint of files has no parts.
assert strata.restore() == ({} if folder else None)
start = int((folder / "epoch").read_text()) + 1 if folder else 0
for epoch in range(start, 4):
    def write(into, epoch=epoch):
        (into / "model").mkdir()
        (into / "model" / "w.bin").write_bytes(bytes([epoch]) * 4)
        (into / "epoch").write_text(str(epoch))
    seq = strata.checkpoint_files(epoch, write)
    if epoch == 3:
        strata.head("best-validation", seq)
    if epoch == 1 and not pathlib.Path("crashed").exists():
        pathlib.Path("crashed").write_text("1")
        raise RuntimeError("killed mid-run")
model = {"w": 3, "snapshot": features["snapshot"]}
metrics = {"epochs": 4, "best": 0.2}
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
        # A build targets its placeholder until the run releases outputs.
        rows = conn.execute(
            "SELECT build_id FROM artifact_builds WHERE artifact_id LIKE ? OR artifact_id = ?"
            " ORDER BY created_at",
            (f"%_cell_{cell_id}_%", session.get_artifact_manager().run_placeholder_id(cell_id)),
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
        bundles = releases.bundles_of(build.build_id, "checkpoint")
        assert [b.release.sequence for b in bundles] == [1, 2, 3, 4]
        assert all(b.release.state == "accepted" for b in bundles)
        for bundle in bundles:
            release = bundle.release
            names = sorted(v.id.split("_var_")[1].split("@")[0] for v in bundle.outputs)
            assert names == ["metrics", "model", "optimizer_state"]
            assert {v.release_id for v in bundle.outputs} == {release.release_id}
            assert all(v.state == "ready" for v in bundle.outputs)
            assert all(v.id.endswith(f"@ckpt={release.sequence}") for v in bundle.outputs)
            # The build was building while each checkpoint was accepted; it is ready now.
            assert release.created_at <= build.completed_at
        assert build.state == "ready"

        # The final release holds the cell's outputs and console, under the cached keys,
        # with the build's computation key on every row of the run.
        (final,) = releases.releases_of(build.build_id, "final")
        outputs = releases.bundle(final.release_id).outputs
        assert sorted(v.id.split("_var_")[1] for v in outputs) == [
            "__console__",
            "metrics",
            "model",
        ]
        parent = build.computation_key
        assert parent and all(v.computation_key == parent for v in outputs)
        assert {v.provenance_hash for v in outputs} == {
            derive_subkey(parent, "model"),
            derive_subkey(parent, "metrics"),
            derive_subkey(parent, "__console__"),
        }
        first = bundles[0].outputs
        assert all(v.computation_key == parent for v in first)
        assert checkpoint_output_key(parent, build.build_id, 1, "model") in {
            v.provenance_hash for v in first
        }
        # The placeholder the build targeted first is a failed row, not a current value.
        placeholder = store.get_artifact(f"nb_{session.notebook_state.id}_run_train", 1)
        assert placeholder is not None and placeholder.state == "failed"
        # It targets one of the run's values now, not its console.
        assert (build.artifact_id, build.version) in {
            (v.id, v.version) for v in outputs if not v.id.endswith("__console__")
        }

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
        assert attempt.principal == same_build.principal_id, "the resume is this caller's"
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

    def test_a_running_computation_refuses_whoever_asks(self, tmp_path: Path):
        """A scratchpad cell asks before its cache check: its outputs are not under the key
        its run is a build of."""
        session, _ = _notebook(tmp_path)
        runs = LocalRuns(session.get_artifact_manager())
        runs.refuse_if_running("train", "parent")
        runs.open(
            cell_id="train",
            computation_key="parent",
            variant=None,
            input_versions={},
            timeout_seconds=30,
            resume=True,
        )

        with pytest.raises(RunRefused, match="already running"):
            runs.refuse_if_running("train", "parent")
        runs.refuse_if_running("train", "other")

    def test_a_run_overtaken_by_a_newer_value_stores_nothing(self, tmp_path: Path, monkeypatch):
        """Another writer makes a newer ``x`` current first: the run's release is refused."""
        notebook_dir = create_notebook(tmp_path, "Overtaken", initialize_environment=False)
        add_cell_to_notebook(notebook_dir, "a")
        write_cell(notebook_dir, "a", "x = 1\n")
        add_cell_to_notebook(notebook_dir, "b", "a")
        write_cell(notebook_dir, "b", "y = x\n")
        session = NotebookSession(
            parse_notebook(notebook_dir), notebook_dir, ledger_dialect=_dialect()
        )
        session.refresh_environment_runtime()
        store, _, _ = _ledger(session)
        artifact_id = session.get_artifact_manager().cell_artifact_id("a", "x")
        real = LocalRuns.finalize

        def overtaken(self, run, staged):
            spec = TransformSpec(executor="elsewhere@v1", params={}, inputs=[])
            newer = store.create_artifact(artifact_id, "elsewhere", spec)
            store.write_blob(artifact_id, newer, b"newer")
            store.finalize_artifact(artifact_id, newer, "{}", row_count=None, byte_size=5)
            return real(self, run, staged)

        monkeypatch.setattr(LocalRuns, "finalize", overtaken)
        result = _run(CellExecutor(session), "a", "x = 1\n")

        latest = store.get_latest_version(artifact_id)
        assert latest is not None and latest.provenance_hash == "elsewhere", result
        assert [v.state for v in store.list_versions_by_id_prefix(artifact_id)] == [
            "failed",
            "ready",
        ]

    @staticmethod
    def _cells(tmp_path: Path, **cells: str) -> NotebookSession:
        notebook_dir = create_notebook(tmp_path, "Records", initialize_environment=False)
        after = None
        for cell_id, source in cells.items():
            add_cell_to_notebook(notebook_dir, cell_id, after)
            write_cell(notebook_dir, cell_id, source)
            after = cell_id
        session = NotebookSession(
            parse_notebook(notebook_dir), notebook_dir, ledger_dialect=_dialect()
        )
        session.refresh_environment_runtime()
        return session

    def test_a_runs_displays_and_console_are_in_its_final_release(self, tmp_path: Path):
        """What a reopen or a hit shows of the run is released with its variables."""
        session = self._cells(tmp_path, a="x = 1\nprint('hi')\nx\n", b="y = x\n")
        result = _run(CellExecutor(session), "a", "x = 1\nprint('hi')\nx\n")
        assert result.success is True, result.error

        _, _, releases = _ledger(session)
        (build,) = _only_build(session, "a")
        (final,) = releases.releases_of(build.build_id, "final")
        outputs = releases.bundle(final.release_id).outputs
        assert sorted(v.id.split("_var_")[1] for v in outputs) == [
            "__console__",
            "__display__0",
            "x",
        ]
        assert all(v.computation_key == build.computation_key for v in outputs)
        (display,) = [v for v in outputs if v.id.endswith("__display__0")]
        assert result.display_outputs[0]["artifact_uri"] == (
            f"strata://artifact/{display.id}@v={display.version}"
        )
        assert result.display_outputs[0]["computed_at"] == display.created_at

    def test_a_leaf_whose_release_fails_keeps_no_display_or_console(
        self, tmp_path: Path, monkeypatch
    ):
        """Nothing of a run is current unless its release is: the run fails instead."""
        session = self._cells(tmp_path, a="print('hi')\n1 + 1\n")

        def refused(self, run, staged):
            raise ReleaseRejected("refused")

        monkeypatch.setattr(LocalRuns, "finalize", refused)
        result = _run(CellExecutor(session), "a", "print('hi')\n1 + 1\n")

        assert result.success is False
        manager = session.get_artifact_manager()
        store = manager.artifact_store
        for name in ("__console__", "__display__0"):
            artifact_id = manager.cell_artifact_id("a", name)
            assert store.get_latest_version(artifact_id) is None
            assert [v.state for v in store.list_versions_by_id_prefix(artifact_id)] == ["failed"]

    ONE = "x = 1\nprint('one')\nx\n"
    TWO = "x = 2\nprint('two')\nx\n"

    def _edit(self, session: NotebookSession, executor: CellExecutor, source: str):
        session.notebook_state.get_cell("a").source = source
        write_cell(session.path, "a", source)
        session.re_analyze_cell("a")
        result = _run(executor, "a", source)
        assert result.success is True, result.error
        return result

    def test_a_revert_is_one_release_of_its_copies(self, tmp_path: Path):
        """K1, K2, then K1 again: the first run's value, display and console come back together."""
        session = self._cells(tmp_path, a=self.ONE, b="y = x\n")
        executor = CellExecutor(session)
        for source in (self.ONE, self.TWO):
            self._edit(session, executor, source)
        assert self._edit(session, executor, self.ONE).cache_hit is True

        store, _, releases = _ledger(session)
        first, _, revert = _only_build(session, "a")
        assert revert.computation_key == first.computation_key
        assert revert.state == "ready"
        (final,) = releases.releases_of(revert.build_id, "final")
        outputs = releases.bundle(final.release_id).outputs
        assert sorted(v.id.split("_var_")[1] for v in outputs) == [
            "__console__",
            "__display__0",
            "x",
        ]
        for copy in outputs:
            name = copy.id.split("_var_")[1]
            assert copy.version == 3
            assert copy.provenance_hash == derive_subkey(first.computation_key, name)
            latest = store.get_latest_version(copy.id)
            assert latest is not None and (latest.version, latest.release_id) == (
                3,
                final.release_id,
            )

    def test_a_revert_whose_source_is_gone_makes_nothing_current(self, tmp_path: Path):
        session = self._cells(tmp_path, a=self.ONE, b="y = x\n")
        executor = CellExecutor(session)
        for source in (self.ONE, self.TWO):
            self._edit(session, executor, source)
        manager = session.get_artifact_manager()
        store = manager.artifact_store
        value, console = (manager.cell_artifact_id("a", name) for name in ("x", "__console__"))
        store.blob_store.delete_blob(console, 1)
        (first, _) = _only_build(session, "a")

        promoted = LocalRuns(manager).promote(
            cell_id="a",
            computation_key=first.computation_key,
            variant=None,
            sources=[(value, 1), (console, 1)],
        )

        assert promoted is None
        latest = store.get_latest_version(value)
        assert latest is not None and latest.version == 2
        assert store.get_artifact(value, 3).state == "failed"
        *_, revert = _only_build(session, "a")
        assert revert.state == "failed"
        assert store.get_artifact(revert.artifact_id, revert.version).state == "failed"

    def test_a_revert_waits_for_a_run_of_its_computation(self, tmp_path: Path):
        """A run of the key holds its build's lease: its own release decides what is current."""
        session = self._cells(tmp_path, a=self.ONE, b="y = x\n")
        executor = CellExecutor(session)
        for source in (self.ONE, self.TWO):
            self._edit(session, executor, source)
        manager = session.get_artifact_manager()
        runs = LocalRuns(manager)
        first, second = _only_build(session, "a")
        runs.open(
            cell_id="a",
            computation_key=first.computation_key,
            variant=None,
            input_versions={},
            timeout_seconds=30,
            resume=False,
        )

        promoted = runs.promote(
            cell_id="a",
            computation_key=first.computation_key,
            variant=None,
            sources=[(manager.cell_artifact_id("a", "x"), 1)],
        )

        assert promoted is None
        assert len(_only_build(session, "a")) == 3  # the run's; no revert's
        latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("a", "x"))
        assert latest is not None and latest.version == 2


LAPSE_WORK = """
import time
strata.checkpoint(step={"n": 1})
time.sleep(1.0)
strata.checkpoint(step={"n": 2})
result = 42
"""


class TestAResumeAfterAnUnfinishedCheckpoint:
    def test_the_resumed_run_releases_a_checkpoint_for_every_epoch(
        self, tmp_path: Path, monkeypatch
    ):
        """A checkpoint the dead attempt began holds its sequence; the resume numbers past it.

        It numbered from the restored release, so its first checkpoint collided with the
        abandoned row and was dropped, and a run dying there again never got past it (LR-4).
        """
        session, executor = _notebook(tmp_path)
        source = session.notebook_state.get_cell("train").source
        _, _, releases = _ledger(session)
        accept = releases.accept

        def second_fails_once(release_id, staged):
            if releases.get(release_id).sequence == 2 and not (tmp_path / "failed").exists():
                (tmp_path / "failed").write_text("1")
                raise OSError("disk full")
            return accept(release_id, staged)

        monkeypatch.setattr(releases, "accept", second_fails_once)
        crashed = _run(executor, "train", source)
        assert crashed.success is False
        (build,) = _only_build(session, "train")
        assert [
            (r.sequence, r.state) for r in releases.releases_of(build.build_id, "checkpoint")
        ] == [
            (1, "accepted"),
            (2, "abandoned"),
        ]

        resumed = _run(executor, "train", source)

        assert resumed.success is True, resumed.error
        accepted = [
            releases.bundle(r.release_id)
            for r in releases.releases_of(build.build_id, "checkpoint")
            if r.state == "accepted" and r.attempt != build.attempt
        ]
        store = session.get_artifact_manager().artifact_store
        epochs = [
            json.loads(store.read_blob(v.id, v.version))["epoch"]
            for bundle in accepted
            for v in bundle.outputs
            if "_var_metrics@ckpt=" in v.id
        ]
        assert epochs == [1, 2, 3]


class TestALapsedLeaseWithNoRival:
    """A lease lapses with nothing wrong (a laptop asleep, a blocked loop): the run goes on.

    The ledger's clock jumps two minutes mid-run, past the 60 s lease, as the wall
    clock does when a laptop wakes. Finding 13 (formal/tla/LocalRun.tla).
    """

    def _session(self, tmp_path: Path, work: str):
        notebook_dir = create_notebook(tmp_path, "Lapse", initialize_environment=False)
        add_cell_to_notebook(notebook_dir, "work")
        write_cell(notebook_dir, "work", work)
        add_cell_to_notebook(notebook_dir, "reader", "work")
        write_cell(notebook_dir, "reader", "seen = result\n")
        session = NotebookSession(
            parse_notebook(notebook_dir), notebook_dir, ledger_dialect=_dialect()
        )
        session.refresh_environment_runtime()
        offset = [0.0]
        _, builds, releases = _ledger(session)
        for store in (builds, releases):
            store._clock = lambda: time.time() + offset[0]
        return session, CellExecutor(session), offset

    def _stored(self, session: NotebookSession):
        manager = session.get_artifact_manager()
        return manager.artifact_store.get_latest_version(manager.cell_artifact_id("work", "result"))

    def test_a_checkpoint_after_the_lapse_is_released(self, tmp_path: Path, monkeypatch):
        session, executor, offset = self._session(tmp_path, LAPSE_WORK)
        handle = LocalRuns.handle

        def handle_then_sleep(self, run, payload, run_dir):
            handle(self, run, payload, run_dir)
            if payload.get("sequence") == 1:
                offset[0] = 120.0

        monkeypatch.setattr(LocalRuns, "handle", handle_then_sleep)
        result = _run(executor, "work", LAPSE_WORK)

        assert result.success is True, result.error
        assert self._stored(session) is not None
        (build,) = _only_build(session, "work")
        _, builds, releases = _ledger(session)
        assert build.state == "ready"
        assert [r.sequence for r in releases.releases_of(build.build_id, "checkpoint")] == [1, 2]
        assert len(builds.attempts_of(build.build_id)) == 2, "the run went on in a new attempt"

    def test_a_run_that_finished_across_the_lapse_keeps_its_outputs(self, tmp_path: Path):
        work = "import time\ntime.sleep(1.5)\nresult = 42\n"
        session, executor, offset = self._session(tmp_path, work)

        async def run():
            task = asyncio.create_task(executor.execute_cell("work", work))
            await asyncio.sleep(0.8)
            offset[0] = 120.0
            return await task

        result = asyncio.run(run())

        assert result.success is True, result.error
        assert self._stored(session) is not None

    def test_a_build_another_process_took_still_fences_the_run(self, tmp_path: Path):
        session, _, offset = self._session(tmp_path, LAPSE_WORK)
        runs = LocalRuns(session.get_artifact_manager())
        run = runs.open(
            cell_id="work",
            computation_key="k",
            variant=None,
            input_versions={},
            timeout_seconds=60.0,
            resume=True,
        )
        offset[0] = 120.0
        _, builds, _ = _ledger(session)
        assert builds.open_attempt(
            run.build_id, "rival", writable_until=time.time() + 600, lease_owner="other:1"
        )

        assert runs._hold(run) is False
        with pytest.raises(RunFenced):
            runs.handle(run, {"event": "checkpoint", "sequence": 1, "parts": {}}, tmp_path)


class TestAnotherProcessBetweenTheReadAndTheWrite:
    """A second process on the store acts on the lapsed build between open's read and write.

    Once the second process has resumed the build, this one is refused, as on one process,
    and that build is left running. SYSTEMS LR-2.
    """

    def _lapsed(self, tmp_path: Path):
        session, _ = _notebook(tmp_path)
        runs = LocalRuns(session.get_artifact_manager())
        first = runs.open(
            cell_id="train",
            computation_key="k",
            variant=None,
            input_versions={},
            timeout_seconds=60.0,
            resume=True,
        )
        runs.handle(first, {"event": "checkpoint", "sequence": 1, "parts": {}}, tmp_path)
        runs.builds.renew_lease(first.build_id, lease_owner(first.attempt), 0.0)
        # A lease is live through its deadline: the ledger's writes come a second after it.
        for store in (runs.builds, runs.releases):
            store._clock = lambda: time.time() + 1.0
        return runs, first.build_id

    @staticmethod
    def _then(runs: LocalRuns, monkeypatch, act) -> None:
        """Run *act* once, right after open reads the build's latest checkpoint."""
        latest = runs.releases.latest_accepted

        def read_then_act(build_id):
            found = latest(build_id)
            monkeypatch.setattr(runs.releases, "latest_accepted", latest)
            act(build_id, found)
            return found

        monkeypatch.setattr(runs.releases, "latest_accepted", read_then_act)

    @staticmethod
    def _resume_elsewhere(runs: LocalRuns):
        def act(build_id, found):
            runs.releases.resume(
                build_id,
                "elsewhere",
                writable_until=time.time() + 600,
                lease_owner="other-host:1",
                lease_duration_seconds=60.0,
                from_release=found.release_id,
            )

        return act

    def _open(self, runs: LocalRuns, *, resume: bool):
        return runs.open(
            cell_id="train",
            computation_key="k",
            variant=None,
            input_versions={},
            timeout_seconds=60.0,
            resume=resume,
        )

    @pytest.mark.parametrize("resume", [False, True], ids=["rerun", "normal run"])
    def test_a_build_resumed_elsewhere_refuses_this_run(self, tmp_path: Path, monkeypatch, resume):
        runs, build_id = self._lapsed(tmp_path)
        self._then(runs, monkeypatch, self._resume_elsewhere(runs))

        with pytest.raises(RunRefused):
            self._open(runs, resume=resume)

        build = runs.builds.get_build(build_id)
        assert build.state == "building", "the other process's run goes on"
        assert build.lease_owner == "other-host:1"
        assert runs.builds.find_open_build("k").build_id == build_id, "and no second one opened"

    def test_a_build_failed_elsewhere_opens_a_new_one(self, tmp_path: Path, monkeypatch):
        runs, build_id = self._lapsed(tmp_path)
        self._then(runs, monkeypatch, lambda b, _: runs.builds.fail_build(b, "elsewhere"))

        run = self._open(runs, resume=False)

        assert run.build_id != build_id
        assert runs.builds.get_build(build_id).error_message == "elsewhere"

    def test_a_miss_with_the_claim_unchanged_raises(self, tmp_path: Path, monkeypatch):
        runs, _ = self._lapsed(tmp_path)

        def refused(*args, **kwargs):
            raise ValueError("refused for a reason that stays")

        monkeypatch.setattr(runs.releases, "resume", refused)
        monkeypatch.setattr(runs.builds, "fail_build", lambda *a, **k: False)

        with pytest.raises(ValueError, match="stays"):
            self._open(runs, resume=True)
        with pytest.raises(RuntimeError, match="could not be failed"):
            self._open(runs, resume=False)


class TestTwoProcessesThatBothFindNoOpenBuild:
    """Both processes read no open build of the computation; B opens one before A writes.

    A gets B's build back from the open and is refused, as if it had read it, and opens
    no build of its own: one run of the computation, not two. SYSTEMS LR-2.
    """

    @staticmethod
    def _open(runs: LocalRuns):
        return runs.open(
            cell_id="train",
            computation_key="k",
            variant=None,
            input_versions={},
            timeout_seconds=60.0,
            resume=True,
        )

    def _race(self, tmp_path: Path, monkeypatch):
        session, _ = _notebook(tmp_path)
        a_runs = LocalRuns(session.get_artifact_manager())
        b_runs = LocalRuns(session.get_artifact_manager())
        find = a_runs.builds.find_open_build
        opened = []

        def read_then_b_opens(*args, **kwargs):
            found = find(*args, **kwargs)
            # The store is shared: undo the patch first, so B reads as itself.
            monkeypatch.setattr(a_runs.builds, "find_open_build", find)
            opened.append(self._open(b_runs))
            return found

        monkeypatch.setattr(a_runs.builds, "find_open_build", read_then_b_opens)
        return a_runs, opened

    def test_the_second_open_is_refused_and_opens_no_build(self, tmp_path: Path, monkeypatch):
        a_runs, opened = self._race(tmp_path, monkeypatch)

        with pytest.raises(RunRefused):
            self._open(a_runs)

        (b,) = opened
        assert a_runs.builds.find_open_build("k").build_id == b.build_id, "B's run goes on"
        building = [
            v
            for v in a_runs.store.list_versions_by_id_prefix(
                a_runs._manager.run_placeholder_id("train")
            )
            if v.state == "building"
        ]
        assert len(building) == 1, "A's placeholder is failed, and only B's targets a build"

    def test_a_revert_that_meets_a_run_just_opened_waits_for_it(self, tmp_path: Path, monkeypatch):
        a_runs, opened = self._race(tmp_path, monkeypatch)

        def stage(release_id, staged):
            raise AssertionError("staged beside a live run")

        made = a_runs.release(
            cell_id="train", computation_key="k", variant=None, what="revert", stage=stage
        )

        assert made is None
        (b,) = opened
        assert a_runs.builds.find_open_build("k").build_id == b.build_id

    def test_an_open_that_raises_fails_its_placeholder(self, tmp_path: Path, monkeypatch):
        session, _ = _notebook(tmp_path)
        runs = LocalRuns(session.get_artifact_manager())

        def lost(*args, **kwargs):
            raise RuntimeError("the database went away")

        monkeypatch.setattr(runs.builds, "open_build", lost)
        with pytest.raises(RuntimeError, match="went away"):
            self._open(runs)

        versions = runs.store.list_versions_by_id_prefix(runs._manager.run_placeholder_id("train"))
        assert versions and all(v.state == "failed" for v in versions)


class TestTwoProcessesWithOneHostAndPid:
    """Two processes on one store whose hostname and pid agree, as two containers given one
    hostname both run as pid 1. Two LocalRuns in this process stand for them.

    A's run lapses after a checkpoint and B resumes the build. A's run then ends without a
    final release, and its close must leave B's lease alone (SYSTEMS LR-3,
    ``LocalRun_SharedOwner``).
    """

    def test_a_lapsed_runs_close_leaves_the_run_that_resumed_its_build(self, tmp_path: Path):
        session, _ = _notebook(tmp_path)
        a_runs = LocalRuns(session.get_artifact_manager())
        b_runs = LocalRuns(session.get_artifact_manager())

        def open_run(runs):
            return runs.open(
                cell_id="train",
                computation_key="k",
                variant=None,
                input_versions={},
                timeout_seconds=60.0,
                resume=True,
            )

        a = open_run(a_runs)
        a_runs.handle(a, {"event": "checkpoint", "sequence": 1, "parts": {}}, tmp_path)
        a_runs.builds.renew_lease(a.build_id, lease_owner(a.attempt), 0.0)
        # A lease is live through its deadline: the ledger's writes come a second after it.
        for store in (a_runs.builds, a_runs.releases):
            store._clock = lambda: time.time() + 1.0
        b = open_run(b_runs)
        assert (b.build_id, b.attempt != a.attempt) == (a.build_id, True), "B resumed it"
        for store in (a_runs.builds, a_runs.releases):
            store._clock = time.time

        assert a_runs._hold(a) is False, "A's renewal does not renew B's lease"
        a_runs._end_without_final(a)

        build = b_runs.builds.get_build(b.build_id)
        assert build.state == "building"
        assert build.lease_expires_at > time.time(), "B's lease is still live"
        with pytest.raises(RunRefused):
            open_run(a_runs)
        b_runs.handle(b, {"event": "checkpoint", "sequence": 2, "parts": {}}, tmp_path)

    def test_a_close_during_a_hold_names_the_attempt_the_hold_opened(
        self, tmp_path: Path, monkeypatch
    ):
        session, _ = _notebook(tmp_path)
        runs = LocalRuns(session.get_artifact_manager())
        run = runs.open(
            cell_id="train",
            computation_key="k",
            variant=None,
            input_versions={},
            timeout_seconds=60.0,
            resume=True,
        )
        runs.handle(run, {"event": "checkpoint", "sequence": 1, "parts": {}}, tmp_path)
        runs.builds.renew_lease(run.build_id, lease_owner(run.attempt), 0.0)
        for store in (runs.builds, runs.releases):
            store._clock = lambda: time.time() + 1.0
        opened, go_on = threading.Event(), threading.Event()
        open_attempt = runs.builds.open_attempt

        def open_then_wait(*args, **kwargs):
            took = open_attempt(*args, **kwargs)
            opened.set()
            go_on.wait(5)
            return took

        monkeypatch.setattr(runs.builds, "open_attempt", open_then_wait)
        hold = threading.Thread(target=runs._hold, args=(run,))
        hold.start()
        assert opened.wait(5)
        # The run ends while the hold has opened its new attempt and not yet named it.
        close = threading.Thread(target=runs._end_without_final, args=(run,))
        close.start()
        close.join(0.2)
        go_on.set()
        hold.join(5)
        close.join(5)

        build = runs.builds.get_build(run.build_id)
        assert build.attempt == run.attempt
        assert build.lease_expires_at <= time.time() + 1.0, "the close dropped that lease"


class TestAbandonedRuns:
    """An interrupted run stays resumable for a week; then prune ends it and its checkpoints go."""

    def _crashed(self, tmp_path: Path):
        session, executor = _notebook(tmp_path, TRAIN_FILES)
        crashed = _run(executor, "train", session.notebook_state.get_cell("train").source)
        assert crashed.success is False and "killed mid-run" in (crashed.error or "")
        (build,) = _only_build(session, "train")
        assert build.state == "building"
        return session, executor, build

    def test_a_run_not_resumed_within_the_window_is_ended(self, tmp_path: Path, monkeypatch):
        from strata.notebook import artifact_integration

        session, _, build = self._crashed(tmp_path)
        _, builds, releases = _ledger(session)
        monkeypatch.setattr(artifact_integration, "_ABANDONED_RUN_SECONDS", 0.0)

        result = session.get_artifact_manager().prune(keep_superseded=0, min_idle_seconds=0)

        assert result["abandoned_runs"] == 1
        assert builds.get_build(build.build_id).state == "failed"
        # Its newest checkpoint was kept for a resume; now nothing is.
        bundles = releases.bundles_of(build.build_id, "checkpoint")
        assert [b.release.sequence for b in bundles] == [1, 2]
        assert not any(releases.bundle(b.release.release_id).outputs for b in bundles)

    def test_a_run_within_the_window_still_resumes(self, tmp_path: Path):
        session, executor, build = self._crashed(tmp_path)
        _, builds, releases = _ledger(session)

        result = session.get_artifact_manager().prune(keep_superseded=0, min_idle_seconds=0)
        assert result["abandoned_runs"] == 0
        assert builds.get_build(build.build_id).state == "building"

        resumed = _run(executor, "train", session.notebook_state.get_cell("train").source)
        assert resumed.success is True, resumed.error
        (same,) = _only_build(session, "train")
        attempt = builds.get_attempt(same.build_id, same.attempt)
        assert (
            attempt.resumed_from
            == releases.bundles_of(build.build_id, "checkpoint")[1].release.release_id
        )

    def test_a_run_resumed_after_it_was_listed_is_left_alone(self, tmp_path: Path, monkeypatch):
        from strata.notebook import artifact_integration

        session, _, build = self._crashed(tmp_path)
        manager = session.get_artifact_manager()
        monkeypatch.setattr(artifact_integration, "_ABANDONED_RUN_SECONDS", 0.0)
        listed = manager.build_store.list_expired_leases(dispatch="external")
        assert [b.build_id for b in listed] == [build.build_id]

        LocalRuns(manager).open(
            cell_id="train",
            computation_key=build.computation_key,
            variant=None,
            input_versions={},
            timeout_seconds=30,
            resume=True,
        )
        monkeypatch.setattr(manager.build_store, "list_expired_leases", lambda **_: listed)

        assert manager.fail_abandoned_runs() == 0
        assert manager.build_store.get_build(build.build_id).state == "building"


STOCHASTIC = """
# @stochastic
import random
sample = random.random()
"""


class TestCheckpointHolds:
    def test_a_checkpoint_the_idle_floor_keeps_keeps_what_it_was_built_from(self, tmp_path: Path):
        """A collectable checkpoint the sweep keeps still holds the value it read."""
        import sqlite3
        import time

        session, executor = _notebook(tmp_path, TRAIN.replace("epoch == 1", "epoch == 99"))
        train = session.notebook_state.get_cell("train")
        assert _run(executor, "train", train.source).success is True
        store, _, releases = _ledger(session)
        (first,) = _only_build(session, "train")
        features_id = f"nb_{session.notebook_state.id}_cell_features_var_features"
        read = store.get_latest_version(features_id)
        # New features and a shorter second run: the first run's last two checkpoints
        # stay their ids' current values, collectable, and alone read the old features.
        features = session.notebook_state.get_cell("features")
        features.source = features.source.replace("events_snapshot}", 'events_snapshot, "v": 2}')
        train.source = train.source.replace("range(start, 4)", "range(start, 2)")
        assert _run(executor, "train", train.source).success is True
        assert len(_only_build(session, "train")) == 2
        tail = [r for r in releases.releases_of(first.build_id, "checkpoint") if r.sequence > 2]
        assert len(tail) == 2
        # Every value but the checkpoints idle past the floor.
        conn = sqlite3.connect(store.db_path)
        try:
            conn.execute(
                "UPDATE artifact_versions SET created_at = 0, last_used_at = ? "
                "WHERE id NOT LIKE '%@ckpt=%'",
                (time.time() - 31 * 86400,),
            )
            conn.commit()
        finally:
            conn.close()

        session.get_artifact_manager().prune(keep_superseded=0, min_idle_seconds=7 * 86400)

        assert all(releases.bundle(r.release_id).outputs for r in tail)
        assert store.get_artifact(read.id, read.version) is not None


def _files(store, bundle) -> dict[str, bytes]:
    return {
        path: store.read_blob(v.id, v.version)
        for path, v in checkpoint_files.files_of(bundle.outputs).items()
    }


class TestCheckpointFiles:
    def test_a_checkpoint_of_files_is_one_release_in_the_layout_vestigo_reads(self, tmp_path: Path):
        session, executor = _notebook(tmp_path, TRAIN_FILES.replace("epoch == 1", "epoch == 99"))
        result = _run(executor, "train", session.notebook_state.get_cell("train").source)
        assert result.success is True, result.error

        store, builds, releases = _ledger(session)
        (build,) = _only_build(session, "train")
        bundles = releases.bundles_of(build.build_id, "checkpoint")
        assert [b.release.sequence for b in bundles] == [1, 2, 3, 4]
        for epoch, bundle in enumerate(bundles):
            assert bundle.release.state == "accepted"
            assert checkpoint_files.step_of(bundle.outputs) == epoch
            assert _files(store, bundle) == {
                "epoch": str(epoch).encode(),
                "model/w.bin": bytes([epoch]) * 4,
            }
            assert {v.release_id for v in bundle.outputs} == {bundle.release.release_id}
            assert all(v.computation_key == build.computation_key for v in bundle.outputs)
            (manifest,) = [v for v in bundle.outputs if v.params.get("manifest")]
            listed = json.loads(store.read_blob(manifest.id, manifest.version))["files"]
            assert [(e["file"], e["byte_size"]) for e in listed] == [
                ("epoch", 1),
                ("model/w.bin", 4),
            ]
        assert build.state == "ready"

        # Retention keeps the head's checkpoint whole and takes the rest.
        session.get_artifact_manager().prune(keep_superseded=0, min_idle_seconds=0)
        kept = [
            b.release.sequence for b in bundles if releases.bundle(b.release.release_id).outputs
        ]
        assert kept == [4]

    def test_a_run_resumes_and_branches_from_a_checkpoint_of_files(self, tmp_path: Path):
        session, executor = _notebook(tmp_path, TRAIN_FILES)
        source = session.notebook_state.get_cell("train").source
        crashed = _run(executor, "train", source)
        assert crashed.success is False and "killed mid-run" in (crashed.error or "")

        resumed = _run(executor, "train", source)
        assert resumed.success is True, resumed.error
        store, builds, releases = _ledger(session)
        (build,) = _only_build(session, "train")
        bundles = releases.bundles_of(build.build_id, "checkpoint")
        # The resumed attempt read epoch 1 back from the folder and went on from 2.
        assert [checkpoint_files.step_of(b.outputs) for b in bundles] == [0, 1, 2, 3]
        attempt = builds.get_attempt(build.build_id, build.attempt)
        assert attempt.resumed_from == bundles[1].release.release_id

        branched = _run(executor, "train", source, init=bundles[0].release.release_id)
        assert branched.success is True, branched.error
        _, branch = _only_build(session, "train")
        steps = [
            checkpoint_files.step_of(b.outputs)
            for b in releases.bundles_of(branch.build_id, "checkpoint")
        ]
        assert steps == [1, 2, 3]

    def test_a_resumed_runs_files_are_copied_not_held_whole(self, tmp_path: Path, monkeypatch):
        session, executor = _notebook(tmp_path, TRAIN_FILES)
        source = session.notebook_state.get_cell("train").source
        _run(executor, "train", source)
        _, _, releases = _ledger(session)
        (build,) = _only_build(session, "train")
        last = releases.bundles_of(build.build_id, "checkpoint")[-1]
        files = {v.id for v in checkpoint_files.files_of(last.outputs).values()}
        assert files
        store = session.get_artifact_manager().artifact_store
        read = store.read_blob
        whole: list[str] = []

        def recording(artifact_id, *args, **kwargs):
            whole.append(artifact_id)
            return read(artifact_id, *args, **kwargs)

        monkeypatch.setattr(store, "read_blob", recording)

        resumed = _run(executor, "train", source)

        assert resumed.success is True, resumed.error
        assert not files & set(whole)

    def test_a_file_outside_its_checkpoint_folder_is_not_released(self, tmp_path: Path):
        session, _ = _notebook(tmp_path)
        runs = LocalRuns(session.get_artifact_manager())
        run = runs.open(
            cell_id="train",
            computation_key="parent",
            variant=None,
            input_versions={},
            timeout_seconds=60,
            resume=True,
        )
        run_dir = tmp_path / "run"
        for sequence in (1, 2, 3, 4):
            (run_dir / "checkpoints" / str(sequence)).mkdir(parents=True)
        (run_dir / "secret").write_text("server file")
        (run_dir / "checkpoints" / "2" / "link").symlink_to(run_dir / "secret")
        (run_dir / "checkpoints" / "4" / "ok").write_text("1")
        # The last opens, but a restore could not write it out under that path.
        cases = ((1, "../../secret"), (2, "link"), (3, "/etc/hosts"), (4, "./ok"))
        for sequence, path in cases:
            event = {"event": "checkpoint", "sequence": sequence, "step": 1, "files": [path]}
            runs.handle(run, event, run_dir)
        store, builds, releases = _ledger(session)
        assert {r.state for r in releases.releases_of(run.build_id, "checkpoint")} == {"abandoned"}
        assert releases.latest_accepted(run.build_id) is None

    def test_a_checkpoint_that_cannot_begin_is_skipped(self, tmp_path: Path, monkeypatch):
        """As one that cannot be staged: a checkpoint is a convenience, the cell goes on."""
        session, _ = _notebook(tmp_path)
        runs = LocalRuns(session.get_artifact_manager())
        run = runs.open(
            cell_id="train",
            computation_key="parent",
            variant=None,
            input_versions={},
            timeout_seconds=60,
            resume=True,
        )

        def locked(*args, **kwargs):
            raise OSError("database is locked")

        monkeypatch.setattr(runs.releases, "prepare", locked)

        runs.handle(run, {"event": "checkpoint", "sequence": 1, "parts": {}}, tmp_path)

        assert runs.releases.latest_accepted(run.build_id) is None


def test_a_head_the_os_timed_out_is_the_run_s_error_not_its_timeout(tmp_path: Path, monkeypatch):
    """The OS's ETIMEDOUT (a store on a network filesystem) is raised as TimeoutError: it
    is neither the cell's time running out nor a harness that left no result."""
    import errno

    session, executor = _notebook(tmp_path, TRAIN.replace("epoch == 1", "epoch == 99"))

    def timed_out(self, run, payload):
        raise OSError(errno.ETIMEDOUT, "Operation timed out")

    monkeypatch.setattr(LocalRuns, "_head", timed_out)
    _run(executor, "features", session.notebook_state.get_cell("features").source)

    result = _run(executor, "train", session.notebook_state.get_cell("train").source)

    assert result.success is False
    assert result.error.startswith("Execution failed: "), result.error
    assert "Operation timed out" in result.error


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
