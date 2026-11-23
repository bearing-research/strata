"""How the notebook config routes report a failure, one route at a time.

Each maps the same exceptions inline (no central mapper), so each is checked: a
route that dropped or reordered a clause would report a missing notebook, a
refused write or a held notebook as a server fault. These are the routes that
reload the session inside their try block; others inject a failure elsewhere.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.notebook.quiesce import NotebookQuiesced
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from tests.notebook.e2e_fixtures import create_test_app, open_notebook_session

# Each reloads the session inside its try block, after the write it was asked for.
ROUTES = {
    "reorder": ("put", "cells/reorder", {"cell_ids": ["b", "a"]}),
    "mounts": ("put", "mounts", {"mounts": []}),
    "connections": ("put", "connections", {"connections": []}),
    "workers": ("put", "workers", {"workers": []}),
    "worker": ("put", "worker", {"worker": None}),
    "timeout": ("put", "timeout", {"timeout": 30}),
    "env": ("put", "env", {"env": {}}),
    "secret_manager": ("put", "secret-manager/config", {}),
    "add_cell": ("post", "cells", {}),
    "rename": ("put", "name", {"name": "Renamed"}),
}

FAILURES = {
    "missing": (FileNotFoundError("the notebook is gone"), 404),
    "refused": (PermissionError("read-only"), 403),
    "invalid": (ValueError("not like that"), 400),
    "held": (NotebookQuiesced("held for a copy"), 409),
    "unexpected": (RuntimeError("a bug"), 500),
}


@pytest.fixture
def opened(tmp_path):
    notebook_dir = create_notebook(tmp_path, "Errors", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "a")
    write_cell(notebook_dir, "a", "x = 1\n")
    add_cell_to_notebook(notebook_dir, "b", after_cell_id="a")
    write_cell(notebook_dir, "b", "y = x + 1\n")
    client = TestClient(create_test_app())
    with open_notebook_session(client, notebook_dir) as (session_id, session):
        yield client, f"/v1/notebooks/{session_id}", session


@pytest.mark.parametrize("failure", sorted(FAILURES))
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_failure_maps_to_its_status(opened, monkeypatch, route, failure):
    client, base, session = opened
    method, path, body = ROUTES[route]
    exc, status = FAILURES[failure]

    def _fail() -> None:
        raise exc

    monkeypatch.setattr(session, "reload", _fail)

    response = getattr(client, method)(f"{base}/{path}", json=body)

    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    if failure == "held":
        assert detail == {"message": "held for a copy", "code": "NOTEBOOK_QUIESCED"}
    elif failure == "unexpected":
        # The bug's text stays in the server log, not the response.
        assert detail == "Internal server error"
    else:
        assert detail == str(exc)
