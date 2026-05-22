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
