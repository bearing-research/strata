"""E2E: the app-view snapshot export over the live REST API.

Drives the real ``GET /v1/notebooks/{id}/export?app_view=1`` path end to
end — open a notebook session, seed the cached outputs + widget values a
run leaves on disk, then download the export and assert it is the app-view
snapshot: display outputs and current widget values, no cell sources.

The regular (non-``app_view``) export is exercised alongside as a contrast
so a future change that blurs the two profiles fails loudly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from strata.notebook.runtime_state import persist_cell_widget_values
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    update_cell_display_outputs,
    write_cell,
)
from tests.notebook.e2e_fixtures import create_test_app, open_notebook_session


@pytest.fixture
def setup():
    """App + client + a temp dir, matching the other notebook e2e suites."""
    client = TestClient(create_test_app())
    with tempfile.TemporaryDirectory() as tmpdir:
        yield client, Path(tmpdir)


def _seed_app_notebook(parent: Path) -> Path:
    """A notebook with a widget cell (persisted value) and a compute cell
    (cached table output), the on-disk state a run would leave behind."""
    nb = create_notebook(parent, "snapshot_e2e")

    add_cell_to_notebook(nb, "controls", language="widget")
    write_cell(nb, "controls", "alpha = slider(0, 1, default=0.5)\n")
    persist_cell_widget_values(nb, "controls", {"alpha": 0.7})

    add_cell_to_notebook(nb, "compute")
    write_cell(nb, "compute", "# @name Scores\nsecret_src_marker = 1\n")
    update_cell_display_outputs(
        nb,
        "compute",
        [
            {
                "content_type": "arrow/ipc",
                "rows": 1,
                "columns": ["name", "score"],
                "preview": [{"name": "Alice", "score": 95}],
                "bytes": 0,
            }
        ],
    )
    return nb


def test_app_view_snapshot_downloads_outputs_without_source(setup):
    client, tmpdir = setup
    nb = _seed_app_notebook(tmpdir)

    with open_notebook_session(client, nb) as (session_id, _session):
        resp = client.get(f"/v1/notebooks/{session_id}/export?fmt=html&app_view=1")

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    disposition = resp.headers["content-disposition"]
    assert "attachment" in disposition
    assert "-app.html" in disposition  # snapshot filename is distinguished

    body = resp.text
    assert "Alice" in body and "95" in body  # the compute cell's output is shown
    assert "secret_src_marker" not in body  # ...but not its source
    assert "alpha" in body and "0.7" in body  # widget rendered at its current value


def test_app_view_snapshot_markdown_variant(setup):
    client, tmpdir = setup
    nb = _seed_app_notebook(tmpdir)

    with open_notebook_session(client, nb) as (session_id, _session):
        resp = client.get(f"/v1/notebooks/{session_id}/export?fmt=markdown&app_view=1")

    assert resp.status_code == 200
    assert "-app.md" in resp.headers["content-disposition"]
    assert "| Alice | 95 |" in resp.text
    assert "secret_src_marker" not in resp.text


def test_regular_export_still_includes_source(setup):
    # Contrast: the default document export keeps cell sources.
    client, tmpdir = setup
    nb = _seed_app_notebook(tmpdir)

    with open_notebook_session(client, nb) as (session_id, _session):
        resp = client.get(f"/v1/notebooks/{session_id}/export?fmt=markdown")

    assert resp.status_code == 200
    assert "secret_src_marker" in resp.text  # source present without app_view
