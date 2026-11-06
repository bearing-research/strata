"""``notebook_state`` goes out exactly as the session builds it, validated on the way."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from strata.notebook.models import CellOutput, CellTestResult
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from strata.notebook.ws import _json_encode
from strata.notebook.ws_payloads import environment_job_finished_payload, notebook_state_payload


@pytest.fixture
def session(tmp_path) -> NotebookSession:
    nb = create_notebook(tmp_path, "State", initialize_environment=False)
    for cell_id, after, language, source in (
        ("controls", None, "widget", "alpha = slider(0, 1, default=0.5)\n"),
        ("helpers", "controls", "python", "# @name Helpers\ndef double(x):\n    return 2 * x\n"),
        ("use", "helpers", "python", "x = 1\nx = double(alpha)\n"),
        ("query", "use", "sql", "SELECT 1 AS one\n"),
        ("notes", "query", "markdown", "# Notes\n"),
        ("again", "notes", "python", "x = 3\n"),
    ):
        add_cell_to_notebook(nb, cell_id, after, language=language)
        write_cell(nb, cell_id, source)
    session = NotebookSession(parse_notebook(nb), nb)
    use = session.notebook_state.get_cell("use")
    assert use is not None
    use.display_outputs = [
        CellOutput(content_type="json/object", preview={"x": 1}, computed_at=1.5),
        CellOutput(content_type="arrow/ipc", rows=1, columns=["a"], preview=[[1]]),
    ]
    use.display_output = use.display_outputs[-1]
    use.test_result = CellTestResult(passed=1, ran_at=2)
    use.error = "boom"
    return session


def _wire(payload: dict) -> object:
    """What a client parses from the frame."""
    return json.loads(_json_encode(payload))


def _state(session: NotebookSession) -> dict:
    state = session.serialize_notebook_state()
    dag = session.dag
    assert dag is not None
    state["dag"] = {
        "edges": dag.serialize_edges(),
        "roots": list(dag.roots),
        "leaves": list(dag.leaves),
        "topological_order": dag.topological_order,
        "variant_groups": [],
    }
    return state


def test_the_state_goes_out_as_the_session_built_it(session):
    state = _state(session)
    assert _wire(notebook_state_payload(state)) == _wire(state)


def test_every_overlay_is_modelled(session):
    cells = {cell["id"]: cell for cell in _state(session)["cells"]}
    # Each would be refused if the cell view did not declare it.
    assert cells["controls"]["widget"]["descriptors"][0]["kind"] == "slider"
    assert cells["helpers"]["module_exports"] == [{"name": "double", "kind": "function"}]
    assert cells["use"]["test_result"]["stale"] is True
    assert cells["use"]["display_outputs"][0]["computed_at"] == 1.5
    assert cells["again"]["shadow_warnings"]


def test_a_field_the_cell_view_does_not_have_is_refused(session):
    state = _state(session)
    state["cells"][0]["bogus"] = 1
    with pytest.raises(ValidationError):
        notebook_state_payload(state)


def test_a_finished_environment_job_goes_out_as_built(session):
    job = {
        "id": "j1",
        "action": "add",
        "command": "uv add rich",
        "status": "completed",
        "started_at": 1,
        "finished_at": 2,
        "duration_ms": 1,
    }
    payload = {
        "environment_job": job,
        "environment_job_history": [job],
        "cells": session.serialize_cells(),
        "lockfile_changed": True,
        "stale_cell_count": 0,
        "stale_cell_ids": [],
        "environment": session.serialize_environment_state(),
        "r_environment": session.serialize_r_environment_state(),
        "dependencies": [{"name": "rich", "version": None, "specifier": ">=13"}],
    }
    assert _wire(environment_job_finished_payload(payload)) == _wire(payload)
