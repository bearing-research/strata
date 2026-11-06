"""The Artifacts page, opened from a notebook, lists that notebook's own store."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.notebook.executor import CellExecutor
from strata.notebook.session import SessionManager
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_the_store_lists_what_the_cells_produced(tmp_path, monkeypatch):
    from strata.notebook import routes
    from strata.server import app

    notebook_dir = create_notebook(tmp_path, "stored")
    add_cell_to_notebook(notebook_dir, "load")
    write_cell(notebook_dir, "load", "# @name loader\nrows = [1, 2, 3]")
    add_cell_to_notebook(notebook_dir, "sum", after_cell_id="load")
    write_cell(notebook_dir, "sum", "total = sum(rows)")

    manager = SessionManager()
    session = manager.open_notebook(notebook_dir)
    session.ensure_venv_synced()
    source = session.notebook_state.get_cell("load").source
    assert (await CellExecutor(session).execute_cell("load", source)).success

    monkeypatch.setattr(routes, "_session_manager", manager)
    with TestClient(app) as client:
        stats = client.get(f"/v1/notebooks/{session.id}/store/stats").json()
        listed = client.get(f"/v1/notebooks/{session.id}/store/artifacts").json()
        server = client.get("/v1/artifacts").json()
        bad = client.get(f"/v1/notebooks/{session.id}/store/artifacts?sort=name")

    assert stats["ready_versions"] >= 1
    # The run's placeholder is in the store, failed, and neither listed nor counted.
    store = session.get_artifact_manager().artifact_store
    placeholders = store.list_versions_by_id_prefix(f"nb_{session.notebook_state.id}_run_")
    assert [v.state for v in placeholders] == ["failed"]
    assert stats["failed_versions"] == 0
    assert stats["total_versions"] == len(listed["artifacts"])
    assert not any(r["artifact_id"] == placeholders[0].id for r in listed["artifacts"])
    assert listed["notebook_name"] == "stored"
    row = next(r for r in listed["artifacts"] if r["output"] == "rows")
    assert row["cell_id"] == "load"
    assert row["cell_name"] == "loader"
    assert row["state"] == "ready"
    assert row["artifact_uri"].startswith(f"strata://artifact/{row['artifact_id']}@v=")
    # The server's store doesn't hold it, which is why the page has to look here.
    assert all(a["artifact_id"] != row["artifact_id"] for a in server["artifacts"])
    assert bad.status_code == 422
