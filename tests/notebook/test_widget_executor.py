"""Execution tests for widget cells (P2).

A widget cell materializes one ``json/object`` value artifact per control with
no subprocess; a downstream Python cell consumes those values like any upstream
output. Values come from ``runtime.json`` (P3 writes them), falling back to the
declared defaults.
"""

from __future__ import annotations

import json

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellLanguage
from strata.notebook.parser import parse_notebook
from strata.notebook.runtime_state import load_runtime_state, save_runtime_state
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook

_WIDGET_SRC = "alpha = slider(0, 1, default=0.5)\nmode = dropdown(['a', 'b'], default='b')"


@pytest.fixture
def widget_session(tmp_path):
    notebook_dir = create_notebook(tmp_path, "Widget Test")
    add_cell_to_notebook(notebook_dir, "controls", None)
    add_cell_to_notebook(notebook_dir, "consume", "controls")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)

    controls = session.notebook_state.get_cell("controls")
    controls.language = CellLanguage.WIDGET
    controls.source = _WIDGET_SRC
    consume = session.notebook_state.get_cell("consume")
    consume.source = "beta = alpha * 2\nchosen = mode\nbeta"
    session._analyze_and_build_dag()
    return session


def _stored_value(session, cell_id, name):
    mgr = session.get_artifact_manager()
    art_id = f"nb_{session.notebook_state.id}_cell_{cell_id}_var_{name}"
    art = mgr.artifact_store.get_latest_version(art_id)
    assert art is not None, f"no artifact for {name}"
    return json.loads(mgr.load_artifact_data(art.id, art.version))


@pytest.mark.asyncio
async def test_widget_cell_stores_default_values(widget_session):
    executor = CellExecutor(widget_session)
    result = await executor.execute_cell("controls", _WIDGET_SRC)

    assert result.success is True
    assert result.execution_method == "widget"
    assert result.outputs["alpha"]["preview"] == 0.5
    assert result.outputs["mode"]["preview"] == "b"
    assert _stored_value(widget_session, "controls", "alpha") == 0.5
    assert _stored_value(widget_session, "controls", "mode") == "b"


@pytest.mark.asyncio
async def test_runtime_value_overrides_default(widget_session):
    executor = CellExecutor(widget_session)
    await executor.execute_cell("controls", _WIDGET_SRC)

    # Simulate a P3 widget_update: the user dragged alpha to 0.25.
    state = load_runtime_state(widget_session.path)
    state.get_or_create_cell("controls").widget_values = {"alpha": 0.25}
    save_runtime_state(widget_session.path, state)

    result = await executor.execute_cell("controls", _WIDGET_SRC)
    assert result.outputs["alpha"]["preview"] == 0.25
    assert result.cache_hit is False  # value changed → re-stored
    assert _stored_value(widget_session, "controls", "alpha") == 0.25


@pytest.mark.asyncio
async def test_unchanged_value_is_a_cache_hit(widget_session):
    executor = CellExecutor(widget_session)
    await executor.execute_cell("controls", _WIDGET_SRC)

    again = await executor.execute_cell("controls", _WIDGET_SRC)
    assert again.cache_hit is True  # same values → no re-store


@pytest.mark.asyncio
async def test_downstream_python_cell_consumes_widget_values(widget_session):
    executor = CellExecutor(widget_session)

    # Running the consumer materializes the upstream widget cell automatically,
    # so the harness receives alpha=0.5 and mode="b" from the widget artifacts.
    result = await executor.execute_cell("consume", "beta = alpha * 2\nchosen = mode\nbeta")
    assert result.success is True, result.error

    # beta = alpha * 2 = 1.0; chosen = mode = "b".
    assert result.outputs["beta"]["preview"] == 1.0
    assert result.outputs["chosen"]["preview"] == "b"
