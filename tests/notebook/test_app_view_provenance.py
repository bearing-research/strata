"""When an app-view output was computed and how it was got, live and in an exported snapshot."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from strata.notebook.executor import CellExecutor
from strata.notebook.export import ExportFormat, ExportOptions, export_notebook
from strata.notebook.runtime_state import how_last_got, load_runtime_state
from strata.notebook.session import SessionManager
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

pytestmark = pytest.mark.integration

LOAD = "rows = [1, 2, 3]"
SHOW = "# @name Total\ntotal = sum(rows)\ntotal"


def _opened(tmp_path):
    notebook_dir = create_notebook(tmp_path, "app")
    add_cell_to_notebook(notebook_dir, "load")
    write_cell(notebook_dir, "load", LOAD)
    add_cell_to_notebook(notebook_dir, "show", after_cell_id="load")
    write_cell(notebook_dir, "show", SHOW)
    manager = SessionManager()
    session = manager.open_notebook(notebook_dir)
    session.ensure_venv_synced()
    return manager, session, notebook_dir


def _app_export(notebook_dir) -> str:
    return export_notebook(
        notebook_dir, ExportOptions(output_format=ExportFormat.HTML, app_view=True)
    )


@pytest.mark.asyncio
async def test_a_run_and_then_a_hit_read_right_with_no_server(tmp_path):
    _manager, session, notebook_dir = _opened(tmp_path)
    executor = CellExecutor(session)
    assert (await executor.execute_cell("load", LOAD)).success
    ran = await executor.execute_cell("show", SHOW)
    assert ran.success and not ran.cache_hit
    session.record_execution("show", ran.duration_ms, cache_hit=False)

    computed_at = load_runtime_state(notebook_dir).cells["show"].display_outputs[-1]["computed_at"]
    assert computed_at > 0
    # Rendered from the files alone: no session, no server.
    assert re.search(r"Computed \d{4}-\d\d-\d\d \d\d:\d\d UTC · took ", _app_export(notebook_dir))
    payload = session.serialize_cell(session.notebook_state.get_cell("show"))
    assert payload["last_got"] == {"cache_hit": False, "ms": int(ran.duration_ms)}
    assert payload["last_duration_ms"] == int(ran.duration_ms)

    hit = await executor.execute_cell("show", SHOW)
    assert hit.cache_hit
    session.record_execution("show", 1.0, cache_hit=True)
    # A hit reports when the result was computed, not when it was loaded.
    assert hit.display_outputs[-1]["computed_at"] == computed_at
    assert " · from cache, saved " in _app_export(notebook_dir)
    payload = session.serialize_cell(session.notebook_state.get_cell("show"))
    assert payload["last_got"] == {"cache_hit": True, "ms": int(ran.duration_ms)}
    assert payload["last_duration_ms"] == 1


@pytest.mark.asyncio
async def test_lineage_of_an_output_reads_the_notebooks_store(tmp_path, monkeypatch):
    from strata.notebook import routes
    from strata.server import app

    manager, session, _ = _opened(tmp_path)
    executor = CellExecutor(session)
    assert (await executor.execute_cell("load", LOAD)).success
    result = await executor.execute_cell("show", SHOW)
    uri = result.display_outputs[-1]["artifact_uri"]
    artifact_id, version = uri.removeprefix("strata://artifact/").split("@v=")

    monkeypatch.setattr(routes, "_session_manager", manager)
    with TestClient(app) as client:
        here = f"/v1/notebooks/{session.id}/artifacts/{artifact_id}/v/{version}/lineage"
        graph = client.get(here).json()
        server = client.get(f"/v1/artifacts/{artifact_id}/v/{version}/lineage")
        missing = client.get(f"/v1/notebooks/{session.id}/artifacts/nope/v/1/lineage")

    assert graph["artifact_uri"] == uri
    assert any("_cell_load_var_rows" in node["uri"] for node in graph["nodes"])
    # Which is why this route exists.
    assert server.status_code == 404
    assert missing.status_code == 404


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ([], None),
        ([{"duration_ms": None, "cache_hit": False}], None),
        ([{"duration_ms": 2100.0, "cache_hit": False}], (False, 2100)),
        (
            [{"duration_ms": 2100.0, "cache_hit": False}, {"duration_ms": 9.0, "cache_hit": True}],
            (True, 2100),
        ),
        # A hit's own record of what its run took beats the last local run.
        (
            [
                {"duration_ms": 2100.0, "cache_hit": False},
                {"duration_ms": 9.0, "cache_hit": True, "hit_run_ms": 500},
            ],
            (True, 500),
        ),
        ([{"duration_ms": 9.0, "cache_hit": True, "team_saved_ms": 700}], (True, 700)),
        # Nothing known to price it against.
        ([{"duration_ms": 9.0, "cache_hit": True}], (True, 0)),
        # A load that took longer than the run saved nothing.
        (
            [{"duration_ms": 5.0, "cache_hit": False}, {"duration_ms": 9.0, "cache_hit": True}],
            (True, 0),
        ),
    ],
)
def test_how_a_result_was_got(samples, expected):
    assert how_last_got(samples) == expected
