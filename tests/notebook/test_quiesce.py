"""Holding a notebook still for a consistent copy. Item 55.

A notebook is a directory Strata writes as cells finish. A nightly copy or a
project move taken mid-write carries a runtime.json and artifacts from
different moments. Quiescing waits out running work, then refuses runs and
edits until released — or until ``max_hold_seconds``, so a caller that dies
cannot freeze a notebook.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from strata.notebook import quiesce
from strata.notebook.quiesce import NotebookQuiesced
from strata.notebook.routes import projects_router, router
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from tests.notebook.test_routes import open_session_id


@pytest.fixture(autouse=True)
def _no_holds_leak(monkeypatch):
    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)
    quiesce.reset()
    yield
    quiesce.reset()


@pytest.fixture
def client():
    from strata.server import _notebook_quiesced

    app = FastAPI()
    app.include_router(router)
    app.include_router(projects_router)
    # The server's own handler, so a writer's refusal is tested as it answers.
    app.add_exception_handler(NotebookQuiesced, _notebook_quiesced)
    return TestClient(app)


def _notebook(parent: Path, name: str = "Held") -> Path:
    notebook_dir = create_notebook(parent, name)
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", "x = 41 + 1")
    add_cell_to_notebook(notebook_dir, "c2", "c1")
    write_cell(notebook_dir, "c2", "y = x")
    return notebook_dir


class TestAHeldNotebook:
    def test_a_run_during_the_hold_is_refused_with_409_naming_why(self, client, tmp_path):
        session_id = open_session_id(client, _notebook(tmp_path))

        held = client.post(f"/v1/notebooks/{session_id}/quiesce", json={"timeout_seconds": 1})
        refused = client.post(f"/v1/notebooks/{session_id}/cells/c1/execute")

        assert held.status_code == 200, held.text
        assert refused.status_code == 409
        assert "held still for a copy" in refused.json()["detail"]["message"]

    def test_an_edit_during_the_hold_is_refused(self, client, tmp_path):
        """Refused by the writer itself, so a route that edits does not have to
        remember to check."""
        notebook_dir = _notebook(tmp_path)
        session_id = open_session_id(client, notebook_dir)
        client.post(f"/v1/notebooks/{session_id}/quiesce", json={"timeout_seconds": 1})

        response = client.put(f"/v1/notebooks/{session_id}/cells/c1", json={"source": "x = 0"})

        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "NOTEBOOK_QUIESCED"
        assert (notebook_dir / "cells" / "c1.py").read_text().strip() == "x = 41 + 1"

    def test_release_ends_it(self, client, tmp_path):
        session_id = open_session_id(client, _notebook(tmp_path))
        client.post(f"/v1/notebooks/{session_id}/quiesce", json={"timeout_seconds": 1})

        released = client.post(f"/v1/notebooks/{session_id}/release")
        ran = client.post(f"/v1/notebooks/{session_id}/cells/c1/execute")

        assert released.json()["released"] is True
        assert ran.status_code == 200, ran.text

    def test_a_caller_that_never_releases_cannot_freeze_it(self, client, tmp_path):
        session_id = open_session_id(client, _notebook(tmp_path))
        client.post(
            f"/v1/notebooks/{session_id}/quiesce",
            json={"timeout_seconds": 0, "max_hold_seconds": 0.2},
        )
        time.sleep(0.3)

        ran = client.post(f"/v1/notebooks/{session_id}/cells/c1/execute")

        assert ran.status_code == 200, ran.text

    def test_a_second_quiesce_of_a_held_notebook_is_a_conflict(self, client, tmp_path):
        session_id = open_session_id(client, _notebook(tmp_path))
        client.post(f"/v1/notebooks/{session_id}/quiesce", json={"timeout_seconds": 0})

        again = client.post(f"/v1/notebooks/{session_id}/quiesce", json={"timeout_seconds": 0})

        assert again.status_code == 409


class TestDraining:
    def test_running_work_is_waited_for_and_can_still_write(self, client, tmp_path, monkeypatch):
        """While draining, a cell finishing has to be able to write the result
        it was allowed to finish; only after does the notebook hold still."""
        from strata.notebook.session import NotebookSession

        notebook_dir = _notebook(tmp_path)
        session_id = open_session_id(client, notebook_dir)
        observed = []
        polls = iter([True, True, False])

        def _running(self):
            busy = next(polls, False)
            # A write during the drain, as a finishing cell would make.
            observed.append(_writable(notebook_dir))
            return busy

        monkeypatch.setattr(NotebookSession, "_has_active_execution", _running)

        held = client.post(f"/v1/notebooks/{session_id}/quiesce", json={"timeout_seconds": 5})

        assert held.json()["cancelled_cells"] == {}
        assert observed and all(observed)
        assert not _writable(notebook_dir)

    def test_work_outliving_the_timeout_is_cancelled_and_named(self, client, tmp_path, monkeypatch):
        from strata.notebook.session import NotebookSession

        notebook_dir = _notebook(tmp_path)
        session_id = open_session_id(client, notebook_dir)
        monkeypatch.setattr(NotebookSession, "_has_active_execution", lambda self: True)
        cancelled = []

        async def _cancel(notebook_id):
            cancelled.append(notebook_id)
            return ["c2"]

        monkeypatch.setattr("strata.notebook.ws.cancel_notebook_execution", _cancel)

        held = client.post(f"/v1/notebooks/{session_id}/quiesce", json={"timeout_seconds": 0.1})

        assert cancelled == [session_id]
        assert held.json()["cancelled_cells"] == {str(notebook_dir.resolve()): ["c2"]}


def _writable(notebook_dir: Path) -> bool:
    try:
        quiesce.assert_writable(notebook_dir)
    except NotebookQuiesced:
        return False
    return True


def test_a_copy_taken_during_the_hold_verifies_against_its_own_digests(client, tmp_path):
    """The point of the whole thing: every artifact in the copy hashes to the
    digest its own store recorded, and runtime.json names only rows that are
    there."""
    import json

    notebook_dir = _notebook(tmp_path)
    session_id = open_session_id(client, notebook_dir)
    assert client.post(f"/v1/notebooks/{session_id}/cells/c1/execute").status_code == 200

    client.post(f"/v1/notebooks/{session_id}/quiesce", json={"timeout_seconds": 5})
    copy = tmp_path / "copy"
    shutil.copytree(notebook_dir, copy)
    client.post(f"/v1/notebooks/{session_id}/release")

    artifacts = copy / ".strata" / "artifacts"
    conn = sqlite3.connect(artifacts / "artifacts.sqlite")
    rows = conn.execute(
        "SELECT id, version, content_sha256 FROM artifact_versions WHERE state = 'ready'"
    ).fetchall()
    conn.close()
    assert rows, "the cell's run should have stored something to verify"
    from strata.artifact_store import ArtifactStore

    store = ArtifactStore(artifacts)
    for artifact_id, version, digest in rows:
        blob = store.blob_store.read_blob(artifact_id, version)
        assert blob is not None
        assert hashlib.sha256(blob).hexdigest() == digest
    runtime = json.loads((copy / ".strata" / "runtime.json").read_text())
    assert runtime["cells"]["c1"]["last_provenance_hash"]


class TestProjects:
    def test_a_project_hold_covers_every_notebook_under_it(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "strata.notebook.routes._get_notebook_storage_root", lambda: tmp_path.resolve()
        )
        project = tmp_path / "project"
        project.mkdir()
        first = open_session_id(client, _notebook(project, "One"))
        second_dir = _notebook(project, "Two")  # not open

        held = client.post("/v1/projects/project/quiesce", json={"timeout_seconds": 0})

        assert held.status_code == 200, held.text
        assert client.post(f"/v1/notebooks/{first}/cells/c1/execute").status_code == 409
        assert not _writable(second_dir)
        # And a notebook inside it cannot be held separately on top.
        assert client.post(f"/v1/notebooks/{first}/quiesce", json={}).status_code == 409

        client.post("/v1/projects/project/release")
        assert _writable(second_dir)


def test_service_mode_needs_the_admin_scope(client, tmp_path, monkeypatch):
    session_id = open_session_id(client, _notebook(tmp_path))
    monkeypatch.setattr(
        "strata.server._state",
        SimpleNamespace(config=SimpleNamespace(principal_auth_enabled=True)),
    )
    monkeypatch.setattr(
        "strata.auth.get_principal",
        lambda: SimpleNamespace(has_scope=lambda scope: scope == "notebook:execute"),
    )

    response = client.post(f"/v1/notebooks/{session_id}/quiesce", json={})

    assert response.status_code == 403
    assert "admin:notebooks" in response.json()["detail"]


async def test_an_edit_over_the_websocket_is_refused_and_not_written(tmp_path):
    """The browser's edits arrive as cell_source_update frames, not REST."""
    from typing import cast

    from fastapi import WebSocket

    from strata.notebook.ws import _handle_cell_source_update
    from tests.notebook.test_ws import _make_fake_ws, open_session

    notebook_dir = _notebook(tmp_path)
    session = open_session(notebook_dir)
    hold = quiesce.begin(notebook_dir, 60)
    quiesce.settle(hold)
    fake, execution_state = _make_fake_ws(session)

    await _handle_cell_source_update(
        cast(WebSocket, fake),
        session,
        {"cell_id": "c1", "source": "x = 0"},
        execution_state,
        session.id,
    )

    errors = fake.frames_of("error")
    assert errors and "held still for a copy" in str(errors[-1]["payload"])
    assert (notebook_dir / "cells" / "c1.py").read_text().strip() == "x = 41 + 1"
    assert session.notebook_state.get_cell("c1").source.strip() == "x = 41 + 1"


class TestWritersThatBypassRoutes:
    """Backstops for paths that reach a notebook's files without an edit route:
    a result landing, and runtime state being saved."""

    def test_runtime_state_is_not_saved_into_a_held_notebook(self, tmp_path):
        from strata.notebook.runtime_state import load_runtime_state, save_runtime_state

        notebook_dir = _notebook(tmp_path)
        state = load_runtime_state(notebook_dir)
        quiesce.settle(quiesce.begin(notebook_dir, 60))

        with pytest.raises(NotebookQuiesced):
            save_runtime_state(notebook_dir, state)

    def test_an_artifact_does_not_land_in_a_held_notebook(self, tmp_path):
        from strata.notebook.artifact_integration import NotebookArtifactManager

        notebook_dir = _notebook(tmp_path)
        manager = NotebookArtifactManager("nb", artifact_dir=notebook_dir / ".strata" / "artifacts")
        quiesce.settle(quiesce.begin(notebook_dir, 60))

        with pytest.raises(NotebookQuiesced):
            manager.store_cell_output(
                cell_id="c1",
                variable_name="x",
                blob_data=b"42",
                content_type="json/object",
                provenance_hash="ab" * 32,
            )


async def test_cancelling_a_notebook_names_the_cell_it_stopped(tmp_path):
    import asyncio

    from strata.notebook.ws import _ensure_execution_state, cancel_notebook_execution

    execution_state = _ensure_execution_state("nb-cancel")
    started = asyncio.Event()

    async def _long_cell():
        started.set()
        await asyncio.sleep(60)

    execution_state.execution_task = asyncio.create_task(_long_cell())
    execution_state.running_cell = "c7"
    await started.wait()

    stopped = await cancel_notebook_execution("nb-cancel")

    assert stopped == ["c7"]
    assert execution_state.execution_task.cancelled()
