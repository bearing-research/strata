"""Test: 3-cell cascade after source edit.

Scenario:
  c1: x = 1
  c2: y = x + 1
  c3: print(y)

After running all three, edit c1 to x=2, then run c3.
Expected: cascade re-runs c1 → c2 → c3, prints "3".
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


class TestCascadeAfterEdit:
    """Run c3 after editing c1: the cascade goes through all three cells."""

    @pytest.fixture
    def setup(self):
        app = create_test_app()
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmpdir:
            nb = (
                NotebookBuilder(Path(tmpdir))
                .add_cell("c1", "x = 1")
                .add_cell("c2", "y = x + 1", after="c1")
                .add_cell("c3", "print(y)", after="c2")
            )
            yield client, nb

    def test_cascade_reruns_all_after_edit(self, setup):
        client, nb = setup

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                ws.clear()
                execute_cell_and_wait(ws, "c2")
                ws.clear()
                result3 = execute_cell_and_wait(ws, "c3")

                # c3 prints y = x + 1 = 2.
                assert result3["type"] == "cell_output" or result3["type"] == "cell_status"
                c3_outputs = [
                    m
                    for m in ws.messages
                    if m["type"] == "cell_output" and m["payload"].get("cell_id") == "c3"
                ]
                if c3_outputs:
                    assert "2" in c3_outputs[-1]["payload"].get("stdout", "")

                ws.clear()

                ws.update_source("c1", "x = 2")
                ws.receive_until("dag_update")
                ws.clear()

                # Running c3 must cascade c1 → c2 → c3.
                execute_cell_and_wait(ws, "c3")

                cascade_msgs = ws.messages_of_type("cascade_prompt")
                assert len(cascade_msgs) > 0, (
                    "Expected cascade_prompt but got none. "
                    f"Message types: {[m['type'] for m in ws.messages]}"
                )

                # x=2, y=x+1=3
                c3_outputs = [
                    m
                    for m in ws.messages
                    if m["type"] == "cell_output" and m["payload"].get("cell_id") == "c3"
                ]
                assert len(c3_outputs) > 0, (
                    f"Expected cell_output for c3 but got none. "
                    f"Message types: {[m['type'] for m in ws.messages]}"
                )
                stdout = c3_outputs[-1]["payload"].get("stdout", "")
                assert "3" in stdout, f"Expected '3' in stdout but got: {stdout!r}"

    def test_cascade_works_when_cells_added_incrementally(self, setup):
        """Run c1 and c2, then add c3, edit c1 and run c3, as users do in the UI.

        When c2 first ran, c3 did not exist, so ``consumed_variables[c2]`` was empty and ``y``
        was never stored.
        """
        client, nb = setup

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # c3 exists in the fixture already; this replays c2 running before c3
                # referenced y, so y was never stored.

                execute_cell_and_wait(ws, "c1")
                ws.clear()
                execute_cell_and_wait(ws, "c2")
                ws.clear()

                # c3's first run resolves y.
                execute_cell_and_wait(ws, "c3")
                c3_out = [
                    m
                    for m in ws.messages
                    if m["type"] == "cell_output" and m["payload"].get("cell_id") == "c3"
                ]
                assert c3_out, "Expected cell_output for c3"
                assert "2" in c3_out[-1]["payload"].get("stdout", ""), (
                    f"Expected '2' but got {c3_out[-1]['payload'].get('stdout', '')!r}"
                )
                ws.clear()

                ws.update_source("c1", "x = 2")
                ws.receive_until("dag_update")
                ws.clear()

                # c3 must cascade c1→c2→c3 and print 3.
                execute_cell_and_wait(ws, "c3")

                cascade_msgs = ws.messages_of_type("cascade_prompt")
                assert len(cascade_msgs) > 0, (
                    f"Expected cascade. Types: {[m['type'] for m in ws.messages]}"
                )

                c3_out = [
                    m
                    for m in ws.messages
                    if m["type"] == "cell_output" and m["payload"].get("cell_id") == "c3"
                ]
                assert c3_out, (
                    f"Expected cell_output for c3. Types: {[m['type'] for m in ws.messages]}"
                )
                stdout = c3_out[-1]["payload"].get("stdout", "")
                assert "3" in stdout, f"Expected '3' in stdout but got: {stdout!r}"

                errors = [m for m in ws.messages if m["type"] == "cell_error"]
                assert not errors, f"Unexpected errors during cascade: {errors}"

    def test_cascade_from_cold_start(self, setup):
        """Cold start: every cell is idle with no artifact_uris, so the cascade runs c1→c2→c3 from
        scratch.
        """
        client, nb = setup

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # No prior runs, as on a fresh open.
                ws.update_source("c1", "x = 2")
                ws.receive_until("dag_update")
                ws.clear()

                execute_cell_and_wait(ws, "c3")

                cascade_msgs = ws.messages_of_type("cascade_prompt")
                assert len(cascade_msgs) > 0, (
                    f"Expected cascade. Types: {[m['type'] for m in ws.messages]}"
                )

                c3_out = [
                    m
                    for m in ws.messages
                    if m["type"] == "cell_output" and m["payload"].get("cell_id") == "c3"
                ]
                assert c3_out, (
                    f"Expected cell_output for c3. Types: {[m['type'] for m in ws.messages]}"
                )
                stdout = c3_out[-1]["payload"].get("stdout", "")
                assert "3" in stdout, f"Expected '3' in stdout but got: {stdout!r}"

                errors = [m for m in ws.messages if m["type"] == "cell_error"]
                assert not errors, f"Unexpected errors: {errors}"

    def test_staleness_propagates_to_downstream(self, setup):
        """After editing c1, c2 and c3 should not be 'ready'."""
        client, nb = setup

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                ws.clear()
                execute_cell_and_wait(ws, "c2")
                ws.clear()
                execute_cell_and_wait(ws, "c3")
                ws.clear()

                for cell in session.notebook_state.cells:
                    assert cell.status == "ready", (
                        f"Cell {cell.id} should be ready, got {cell.status}"
                    )

                ws.update_source("c1", "x = 2")
                ws.receive_until("dag_update")

                # The edit invalidates c1 and, through it, c2 and c3.
                c1 = next(c for c in session.notebook_state.cells if c.id == "c1")
                c2 = next(c for c in session.notebook_state.cells if c.id == "c2")
                c3 = next(c for c in session.notebook_state.cells if c.id == "c3")

                assert c1.status != "ready", f"c1 should be stale, got {c1.status}"
                assert c2.status != "ready", (
                    f"c2 should be stale (upstream c1 changed), got {c2.status}"
                )
                assert c3.status != "ready", (
                    f"c3 should be stale (upstream chain changed), got {c3.status}"
                )

    def test_artifact_store_state_after_cascade(self, setup):
        """The cascade stores c1's artifact; without it c2 fails with 'name x is not defined'."""
        client, nb = setup

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                ws.clear()
                execute_cell_and_wait(ws, "c2")
                ws.clear()
                execute_cell_and_wait(ws, "c3")
                ws.clear()

                artifact_mgr = session.get_artifact_manager()
                notebook_id = session.notebook_state.id
                c1_x_id = f"nb_{notebook_id}_cell_c1_var_x"
                c2_y_id = f"nb_{notebook_id}_cell_c2_var_y"

                art_x_v1 = artifact_mgr.artifact_store.get_latest_version(c1_x_id)
                assert art_x_v1 is not None, (
                    f"Expected artifact for c1:x after initial run. artifact_id={c1_x_id}"
                )
                assert art_x_v1.state == "ready", (
                    f"Expected c1:x artifact to be ready, got {art_x_v1.state}"
                )

                art_y_v1 = artifact_mgr.artifact_store.get_latest_version(c2_y_id)
                assert art_y_v1 is not None, (
                    f"Expected artifact for c2:y after initial run. artifact_id={c2_y_id}"
                )

                ws.update_source("c1", "x = 2")
                ws.receive_until("dag_update")
                ws.clear()

                execute_cell_and_wait(ws, "c3")

                errors = [m for m in ws.messages if m["type"] == "cell_error"]
                assert not errors, f"Unexpected errors during cascade: {errors}"

                art_x_v2 = artifact_mgr.artifact_store.get_latest_version(c1_x_id)
                assert art_x_v2 is not None, (
                    f"Expected artifact for c1:x after cascade. artifact_id={c1_x_id}"
                )
                assert art_x_v2.state == "ready", (
                    f"Expected c1:x artifact to be ready after cascade, got {art_x_v2.state}"
                )
                # A new version (different provenance from v1).
                assert art_x_v2.version >= art_x_v1.version, (
                    f"Expected new version for c1:x, "
                    f"got v{art_x_v2.version} (was v{art_x_v1.version})"
                )

                art_y_v2 = artifact_mgr.artifact_store.get_latest_version(c2_y_id)
                assert art_y_v2 is not None, (
                    f"Expected artifact for c2:y after cascade. artifact_id={c2_y_id}"
                )

                c1 = next(c for c in session.notebook_state.cells if c.id == "c1")
                c2 = next(c for c in session.notebook_state.cells if c.id == "c2")
                assert c1.artifact_uri is not None, "c1.artifact_uri should be set after cascade"
                assert c2.artifact_uri is not None, "c2.artifact_uri should be set after cascade"

                c3_outputs = [
                    m
                    for m in ws.messages
                    if m["type"] == "cell_output" and m["payload"].get("cell_id") == "c3"
                ]
                assert c3_outputs, "Expected cell_output for c3"
                stdout = c3_outputs[-1]["payload"].get("stdout", "")
                assert "3" in stdout, f"Expected '3' in stdout but got: {stdout!r}"

    def test_upstream_rerun_on_missing_artifact(self, setup):
        """If an upstream artifact is missing, the upstream cell is re-run.

        c1 succeeds but its artifact is gone; resolving ``x`` for c2 detects the gap and re-runs
        c1.
        """
        client, nb = setup

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                ws.clear()

                artifact_mgr = session.get_artifact_manager()
                notebook_id = session.notebook_state.id
                c1_x_id = f"nb_{notebook_id}_cell_c1_var_x"
                art = artifact_mgr.artifact_store.get_latest_version(c1_x_id)
                assert art is not None, "Precondition: c1:x artifact must exist"

                # Delete the artifact row so it looks missing.
                conn = artifact_mgr.artifact_store._get_connection()
                try:
                    conn.execute(
                        "DELETE FROM artifact_versions WHERE id = ?",
                        (c1_x_id,),
                    )
                    conn.commit()
                finally:
                    conn.close()

                assert artifact_mgr.artifact_store.get_latest_version(c1_x_id) is None

                # c2 must detect the missing artifact, re-run c1, and succeed.
                execute_cell_and_wait(ws, "c2")

                c2 = next(c for c in session.notebook_state.cells if c.id == "c2")
                assert c2.status == "ready", (
                    f"c2 should be ready after auto-rerun of c1, got {c2.status}"
                )

                # Re-created by the retry.
                art_after = artifact_mgr.artifact_store.get_latest_version(c1_x_id)
                assert art_after is not None, "c1:x artifact should exist after auto-rerun"

    def test_provenance_dedup_does_not_break_cascade(self, setup):
        """Provenance dedup must not poison the canonical artifact ID.

        A foreign artifact with c1's provenance under another ID (an old cell layout or a copied
        notebook) gives a cache hit, but the canonical ``nb_..._cell_c1_var_x`` must still end up
        ready, or downstream cells fail with "name 'x' is not defined".
        """
        client, nb = setup

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                artifact_mgr = session.get_artifact_manager()
                notebook_id = session.notebook_state.id

                # Poison the store: a foreign artifact with the provenance c1 will compute,
                # under a different artifact ID (as an old cell layout or a copied notebook
                # leaves behind).

                import hashlib

                from strata.notebook.env import compute_lockfile_hash
                from strata.notebook.provenance import (
                    compute_provenance_hash,
                    compute_source_hash,
                )

                source_hash = compute_source_hash("x = 1")
                env_hash = compute_lockfile_hash(session.path)
                cell_prov = compute_provenance_hash([], source_hash, env_hash)
                var_prov = hashlib.sha256(f"{cell_prov}:x".encode()).hexdigest()

                from strata.artifact_store import TransformSpec

                foreign_id = f"nb_{notebook_id}_cell_GHOST_var_x"
                fv = artifact_mgr.artifact_store.create_artifact(
                    artifact_id=foreign_id,
                    provenance_hash=var_prov,
                    transform_spec=TransformSpec(
                        executor="notebook/cell@v1",
                        params={"content_type": "json/object"},
                        inputs=[],
                    ),
                )
                artifact_mgr.artifact_store.blob_store.write_blob(
                    foreign_id,
                    fv,
                    b"1",
                )
                artifact_mgr.artifact_store.finalize_artifact(
                    foreign_id,
                    fv,
                    "",
                    0,
                    1,
                )

                assert artifact_mgr.find_cached(var_prov) is not None
                canonical_id = f"nb_{notebook_id}_cell_c1_var_x"
                assert artifact_mgr.artifact_store.get_latest_version(canonical_id) is None

                # c3 cascades c1→c2→c3.
                execute_cell_and_wait(ws, "c3")

                cascade_msgs = ws.messages_of_type("cascade_prompt")
                assert len(cascade_msgs) > 0, (
                    f"Expected cascade. Types: {[m['type'] for m in ws.messages]}"
                )

                # x=1, y=x+1=2
                c3_out = [
                    m
                    for m in ws.messages
                    if m["type"] == "cell_output" and m["payload"].get("cell_id") == "c3"
                ]
                assert c3_out, (
                    f"Expected cell_output for c3. Types: {[m['type'] for m in ws.messages]}"
                )
                stdout = c3_out[-1]["payload"].get("stdout", "")
                assert "2" in stdout, f"Expected '2' in stdout but got: {stdout!r}"

                art = artifact_mgr.artifact_store.get_latest_version(
                    canonical_id,
                )
                assert art is not None, (
                    f"Canonical artifact {canonical_id} must be ready after cascade execution."
                )
                assert art.state == "ready"

                errors = [m for m in ws.messages if m["type"] == "cell_error"]
                assert not errors, f"Unexpected errors: {errors}"

    def test_rest_edit_then_ws_run_triggers_cascade(self, setup):
        """Edit via REST PUT, then run via WebSocket: the cascade must trigger.

        The REST endpoint must recompute staleness, or every cell stays "ready" and the planner
        sees no cascade.
        """
        client, nb = setup

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                ws.clear()
                execute_cell_and_wait(ws, "c2")
                ws.clear()
                execute_cell_and_wait(ws, "c3")
                ws.clear()

                for cell in session.notebook_state.cells:
                    assert cell.status == "ready", (
                        f"Cell {cell.id} should be ready, got {cell.status}"
                    )

                # Edit c1 via REST, not the WebSocket.
                resp = client.put(
                    f"/v1/notebooks/{sid}/cells/c1",
                    json={"source": "x = 2"},
                )
                assert resp.status_code == 200
                rest_data = resp.json()

                assert "cells" in rest_data, "REST response should include 'cells' with statuses"

                c1 = next(c for c in session.notebook_state.cells if c.id == "c1")
                assert c1.status != "ready", f"c1 should be stale after REST edit, got {c1.status}"

                execute_cell_and_wait(ws, "c3")

                cascade_msgs = ws.messages_of_type("cascade_prompt")
                assert len(cascade_msgs) > 0, (
                    f"Expected cascade after REST edit. Types: {[m['type'] for m in ws.messages]}"
                )

                c3_out = [
                    m
                    for m in ws.messages
                    if m["type"] == "cell_output" and m["payload"].get("cell_id") == "c3"
                ]
                assert c3_out, "Expected cell_output for c3"
                stdout = c3_out[-1]["payload"].get("stdout", "")
                assert "3" in stdout, f"Expected '3' in stdout but got: {stdout!r}"
