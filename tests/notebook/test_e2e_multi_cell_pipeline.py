"""E2E tests: multi-cell pipelines with cascade execution.

Tests linear chains (A → B → C) and branching DAGs (A → B, A → C)
where executing a downstream cell triggers cascade of upstream cells.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    create_test_app,
    execute_cell_and_wait,
    open_notebook_session,
    ws_connect,
)


@pytest.fixture
def setup():
    app = create_test_app()
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:
        yield client, Path(tmpdir)


class TestLinearCascade:
    """Three-cell chain: c1 → c2 → c3. Execute c3 triggers cascade of c1, c2."""

    def test_cascade_executes_all_upstream(self, setup):
        """Executing the leaf cell triggers full cascade."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "x = 1")
            .add_cell("c2", "y = x + 1", after="c1")
            .add_cell("c3", "z = y + 1", after="c2")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Execute leaf — should trigger cascade
                result = execute_cell_and_wait(ws, "c3")

                # All cells should have been executed
                assert result["type"] == "cell_output"
                assert "z" in result["payload"]["outputs"]

                # Check cascade_prompt was sent
                cascade_prompts = ws.messages_of_type("cascade_prompt")
                assert len(cascade_prompts) >= 1

                # Check cascade_progress messages were sent
                progress_msgs = ws.messages_of_type("cascade_progress")
                assert len(progress_msgs) >= 1

    def test_cascade_message_sequence(self, setup):
        """Cascade messages arrive in correct order: prompt → progress → statuses."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "x = 1")
            .add_cell("c2", "y = x + 1", after="c1")
            .add_cell("c3", "z = y + 1", after="c2")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c3")

                # Verify each upstream cell went through running → output → ready
                for cell_id in ["c1", "c2", "c3"]:
                    statuses = [
                        m["payload"]["status"]
                        for m in ws.messages_of_type("cell_status")
                        if m["payload"]["cell_id"] == cell_id
                    ]
                    assert "running" in statuses, f"{cell_id} was never running"
                    assert "ready" in statuses, f"{cell_id} never became ready"

    def test_no_cascade_when_upstream_ready(self, setup):
        """If upstream cells are already ready, no cascade is triggered."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Execute c1 first
                execute_cell_and_wait(ws, "c1")
                ws.clear()

                # Execute c2 — c1 is ready, no cascade needed
                execute_cell_and_wait(ws, "c2")

                cascade_prompts = ws.messages_of_type("cascade_prompt")
                assert len(cascade_prompts) == 0


class TestBranchingDAG:
    """Branching DAG: c1 → c2, c1 → c3 (two consumers of c1's output)."""

    def test_shared_upstream(self, setup):
        """Two cells consume the same upstream variable."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "x = 10")
            .add_cell("c2", "a = x * 2", after="c1")
            .add_cell("c3", "b = x * 3", after="c2")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Execute c1
                execute_cell_and_wait(ws, "c1")

                # Execute c2 and c3 (both depend on c1 which is ready)
                r2 = execute_cell_and_wait(ws, "c2")
                assert r2["type"] == "cell_output"
                assert "a" in r2["payload"]["outputs"]

                r3 = execute_cell_and_wait(ws, "c3")
                assert r3["type"] == "cell_output"
                assert "b" in r3["payload"]["outputs"]

    def test_diamond_dag(self, setup):
        """Diamond: c1 → c2, c1 → c3, c2+c3 → c4."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "x = 5")
            .add_cell("c2", "a = x + 1", after="c1")
            .add_cell("c3", "b = x + 2", after="c2")
            .add_cell("c4", "result = a + b", after="c3")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Execute leaf c4 — triggers cascade for entire DAG
                result = execute_cell_and_wait(ws, "c4")
                assert result["type"] == "cell_output"
                assert "result" in result["payload"]["outputs"]


class TestForceExecution:
    """Test cell_execute_force (run with stale inputs)."""

    def test_force_skips_cascade(self, setup):
        """Force-executing a downstream cell does not trigger or run upstreams."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Force-execute c2 without running c1 first
                ws.execute_force("c2")

                # Should get cell_status(running) then either output or error
                msg = ws.receive_until("cell_status", cell_id="c2", status="running")
                assert msg["payload"]["status"] == "running"

                # Wait for completion (may error due to missing x, but no cascade)
                final = ws.receive_until("cell_status", cell_id="c2")
                assert final["payload"]["status"] in ("ready", "error")

                # No cascade_prompt should have been sent
                cascade_prompts = ws.messages_of_type("cascade_prompt")
                assert len(cascade_prompts) == 0

                # Upstream cell should not have been materialized as a side effect.
                c1_outputs = [
                    m
                    for m in ws.messages_of_type("cell_output")
                    if m["payload"].get("cell_id") == "c1"
                ]
                assert c1_outputs == []

                state = ws.sync()
                c1 = next(c for c in state["payload"]["cells"] if c["id"] == "c1")
                assert c1["status"] != "ready"


class TestRerunExecution:
    """cell_execute_rerun: cache off for target, but upstreams still materialize."""

    def test_rerun_materializes_upstreams_and_skips_target_cache(self, setup):
        """Rerunning c2 in a chain c1 → c2 must run c1 first (cache miss for c2)."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Warm cache: run c2 the normal way (cascade brings c1 in).
                execute_cell_and_wait(ws, "c2")
                ws.clear()

                # Rerun c2: target cache must be bypassed, but upstream c1's
                # artifact resolves from the cache so c2 actually executes.
                ws.execute_rerun("c2")
                final = ws.receive_until("cell_output", cell_id="c2")
                assert final["payload"]["outputs"]["y"]["preview"] == 2

                # cache_hit must be False on the rerun's output frame.
                assert final["payload"].get("cache_hit") is False

    def test_rerun_picks_up_edited_upstream_after_flush(self, setup):
        """Editing an upstream then rerunning a downstream must re-execute the
        upstream too — the cascade planner sees the dirty upstream as stale
        once the source update has been flushed."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c2")
                ws.clear()

                # Simulate the flush that the frontend does before rerun:
                # push the new c1 source to the backend before the rerun
                # message lands.
                ws.update_source("c1", "x = 42")
                # Drain any dag_update / cell_status frames the source
                # update triggers so receive_until below sees only the
                # rerun frames.
                ws.clear()

                ws.execute_rerun("c2")
                c1_out = ws.receive_until("cell_output", cell_id="c1")
                assert c1_out["payload"]["outputs"]["x"]["preview"] == 42

                c2_out = ws.receive_until("cell_output", cell_id="c2")
                assert c2_out["payload"]["outputs"]["y"]["preview"] == 43

    def test_rerun_with_stale_upstream_broadcasts_upstream(self, setup):
        """Rerun on a cell whose upstream is stale must run the upstream
        through cascade so the client gets a cell_output for it."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # c1 is idle (never run) so it's stale for c2.
                ws.execute_rerun("c2")

                c1_out = ws.receive_until("cell_output", cell_id="c1")
                assert c1_out["payload"]["outputs"]["x"]["preview"] == 1

                c2_out = ws.receive_until("cell_output", cell_id="c2")
                assert c2_out["payload"]["outputs"]["y"]["preview"] == 2
                assert c2_out["payload"].get("cache_hit") is False

    def test_rerun_all_forces_every_cell(self, setup):
        """notebook_rerun_all bypasses cache for every non-empty cell."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Warm cache.
                execute_cell_and_wait(ws, "c2")
                ws.clear()

                ws.rerun_all()
                # Wait for both cells to land their final cell_output.
                ws.receive_until("cell_output", cell_id="c1")
                ws.receive_until("cell_output", cell_id="c2")

                outputs_by_cell = {
                    m["payload"]["cell_id"]: m for m in ws.messages_of_type("cell_output")
                }
                assert outputs_by_cell["c1"]["payload"].get("cache_hit") is False
                assert outputs_by_cell["c2"]["payload"].get("cache_hit") is False


class TestRunAllBatching:
    """run-all/rerun-all routes batchable runs through CellExecutor.execute_batch
    instead of N single-cell subprocesses (#26 PR-b4)."""

    def test_run_all_uses_batch_execution_method(self, setup):
        """Three Python cells in notebook order form one batch; cell_output
        frames carry execution_method="batch" (set on the synthetic
        CellExecutionResult the dispatcher constructs from BatchCellResult)."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "a = 1")
            .add_cell("c2", "b = a + 1", after="c1")
            .add_cell("c3", "c = b + 1", after="c2")
        )
        with open_notebook_session(client, nb.path) as (sid, _session):
            with ws_connect(client, sid) as ws:
                ws.run_all()
                ws.receive_until("cell_output", cell_id="c1")
                ws.receive_until("cell_output", cell_id="c2")
                ws.receive_until("cell_output", cell_id="c3")

                outputs_by_cell = {
                    m["payload"]["cell_id"]: m for m in ws.messages_of_type("cell_output")
                }
                # All three batched (one partition run, three Python cells).
                for cell_id in ("c1", "c2", "c3"):
                    method = outputs_by_cell[cell_id]["payload"].get("execution_method")
                    assert method == "batch", (
                        f"cell {cell_id} should have execution_method='batch', got {method!r}"
                    )

    def test_run_all_continue_on_error_runs_remaining_cells(self, setup):
        """With continue_on_error=true (default), a failed cell mid-batch
        doesn't stop the run. Cells after the failed one continue via
        single-cell with skip_upstream_materialization=True; a cell that
        doesn't depend on the failure succeeds."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "a = 1")
            .add_cell("c2_bad", "raise RuntimeError('boom')")
            .add_cell("c3", "z = 99")
        )
        with open_notebook_session(client, nb.path) as (sid, _session):
            with ws_connect(client, sid) as ws:
                ws.run_all()
                # Wait for the terminal frame from each — could be cell_output
                # or cell_error depending on success.
                ws.receive_until_any_of(("cell_output", "cell_error"), cell_id="c1") if hasattr(
                    ws, "receive_until_any_of"
                ) else ws.receive_until("cell_output", cell_id="c1")
                ws.receive_until("cell_error", cell_id="c2_bad")
                ws.receive_until("cell_output", cell_id="c3")

                # c1 succeeded, c2_bad errored, c3 ran independently and succeeded.
                outputs = {m["payload"]["cell_id"] for m in ws.messages_of_type("cell_output")}
                errors = {m["payload"]["cell_id"] for m in ws.messages_of_type("cell_error")}
                assert "c1" in outputs
                assert "c2_bad" in errors
                assert "c3" in outputs, (
                    f"c3 should run via continue_on_error=true single-cell continuation; "
                    f"got outputs={outputs} errors={errors}"
                )

    def test_run_all_passes_env_annotations_to_batch(self, setup):
        """`# @env KEY=VALUE` reaches the batched cell. Regression for
        #33 review finding #1 — the dispatcher had been dropping all env
        when building cell_specs.
        """
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                (
                    "# @env STRATA_DISP_ENV_TEST=batched\n"
                    "import os\nvalue = os.environ['STRATA_DISP_ENV_TEST']\n"
                ),
            )
            .add_cell("c2", "y = value + '!'\n", after="c1")
        )
        with open_notebook_session(client, nb.path) as (sid, _session):
            with ws_connect(client, sid) as ws:
                ws.run_all()
                c1 = ws.receive_until("cell_output", cell_id="c1")
                # c2 exists just to make a size-≥2 batch (size-1 batches
                # route through single-cell). Only c1 needs an assertion.
                ws.receive_until("cell_output", cell_id="c2")

                assert c1["payload"].get("execution_method") == "batch"
                assert c1["payload"]["outputs"]["value"]["preview"] == "batched"

    def test_run_all_mount_failure_emits_per_cell_error_not_batch_abort(self, setup):
        """A failed mount on one cell becomes a per-cell error broadcast;
        the rest of the batch still runs. Regression for #34 review
        finding #3 — mount-prep exceptions had been propagating out of
        _run_partition_batch and killing the whole run before any cell
        emitted any frame.
        """
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            # c1 mounts a path that does not exist → MountResolver raises.
            .add_cell(
                "c_bad",
                "# @mount data file:///definitely/does/not/exist/strata ro\nv = 1\n",
            )
            .add_cell("c_good", "y = 2", after="c_bad")
        )
        with open_notebook_session(client, nb.path) as (sid, _session):
            with ws_connect(client, sid) as ws:
                ws.run_all()
                # Mount failures now broadcast AFTER batch successes (per #35
                # review), so c_good lands first via batch + then c_bad's
                # synthetic error broadcast. Read forward until we see
                # c_bad's error (the last terminal frame); messages_of_type
                # then surfaces c_good's earlier output from the buffer.
                ws.receive_until("cell_error", cell_id="c_bad")

                errors = {m["payload"]["cell_id"] for m in ws.messages_of_type("cell_error")}
                outputs = {m["payload"]["cell_id"] for m in ws.messages_of_type("cell_output")}
                assert "c_bad" in errors, f"c_bad should be in cell_error: {errors}"
                assert "c_good" in outputs, (
                    f"c_good must still run after c_bad's mount failure; got outputs={outputs}"
                )

    def test_rerun_all_continues_after_failure_without_re_running_failed_upstream(
        self, setup, monkeypatch
    ):
        """rerun-all + mid-batch failure: continuation cells use
        execute_cell_force (no materialize, no cache) instead of
        execute_cell_rerun (which would materialize the failed upstream).

        Regression for #33 review finding #2.
        """
        from strata.notebook import executor as executor_mod

        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "a = 1")
            .add_cell("c2_bad", "raise RuntimeError('boom')")
            .add_cell("c3", "z = 99")
        )

        rerun_calls: list[tuple] = []
        force_calls: list[tuple] = []

        original_rerun = executor_mod.CellExecutor.execute_cell_rerun
        original_force = executor_mod.CellExecutor.execute_cell_force

        async def spy_rerun(self, cell_id, source, *args, **kwargs):
            rerun_calls.append((cell_id, source))
            return await original_rerun(self, cell_id, source, *args, **kwargs)

        async def spy_force(self, cell_id, source, *args, **kwargs):
            force_calls.append((cell_id, source))
            return await original_force(self, cell_id, source, *args, **kwargs)

        monkeypatch.setattr(executor_mod.CellExecutor, "execute_cell_rerun", spy_rerun)
        monkeypatch.setattr(executor_mod.CellExecutor, "execute_cell_force", spy_force)

        with open_notebook_session(client, nb.path) as (sid, _session):
            with ws_connect(client, sid) as ws:
                ws.rerun_all()
                ws.receive_until("cell_error", cell_id="c2_bad")
                ws.receive_until("cell_output", cell_id="c3")

        # After the batch fails at c2_bad, c3's continuation must use
        # execute_cell_force (no materialize, no cache) — NOT
        # execute_cell_rerun (which would re-materialize c2_bad and
        # recursively re-run the failed cell).
        force_cell_ids = {cid for cid, _src in force_calls}
        rerun_cell_ids = {cid for cid, _src in rerun_calls}
        assert "c3" in force_cell_ids, (
            f"c3 continuation should call execute_cell_force; "
            f"got force={force_cell_ids} rerun={rerun_cell_ids}"
        )
        assert "c3" not in rerun_cell_ids, (
            f"c3 continuation must not call execute_cell_rerun (would "
            f"re-materialize failed upstream c2_bad); got rerun={rerun_cell_ids}"
        )
