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


def test_persist_cell_widget_values_merges(tmp_path):
    from strata.notebook.runtime_state import load_runtime_state, persist_cell_widget_values

    persist_cell_widget_values(tmp_path, "c1", {"alpha": 0.5, "mode": "a"})
    merged = persist_cell_widget_values(tmp_path, "c1", {"alpha": 0.25})  # partial update

    assert merged == {"alpha": 0.25, "mode": "a"}
    assert load_runtime_state(tmp_path).cells["c1"].widget_values == {"alpha": 0.25, "mode": "a"}


def test_serialize_cell_emits_widget_block(widget_session):
    """serialize_cell attaches descriptors + current values for widget cells."""
    controls = widget_session.notebook_state.get_cell("controls")
    controls.widget_values = {"alpha": 0.25}

    data = widget_session.serialize_cell(controls)

    assert "widget" in data
    by_name = {d["name"]: d for d in data["widget"]["descriptors"]}
    assert set(by_name) == {"alpha", "mode"}
    assert by_name["alpha"]["kind"] == "slider"
    assert by_name["mode"]["params"]["options"] == ["a", "b"]
    assert data["widget"]["values"] == {"alpha": 0.25}


def test_serialize_cell_no_widget_block_for_python(widget_session):
    data = widget_session.serialize_cell(widget_session.notebook_state.get_cell("consume"))
    assert "widget" not in data


@pytest.mark.asyncio
async def test_widget_publishes_artifact_uris_for_downstream_provenance(widget_session):
    """Regression: a widget value change must reach downstream provenance.

    The widget cell must publish each control's value artifact onto
    ``cell.artifact_uris`` (like a Python cell's multi-output vars). Without it,
    a downstream consumer's ``_collect_input_hashes`` finds no upstream
    artifact, its provenance is blind to the control value, and it cache-hits
    the stale output — i.e. dragging a slider never updates downstream cells,
    which is exactly what live mode / the interactive app view depend on.
    """
    from strata.notebook.runtime_state import persist_cell_widget_values

    executor = CellExecutor(widget_session)
    await executor.execute_cell("controls", _WIDGET_SRC)

    controls = widget_session.notebook_state.get_cell("controls")
    assert "alpha" in controls.artifact_uris
    assert "mode" in controls.artifact_uris

    # The downstream consumer resolves the widget's value artifacts as inputs.
    hashes_before = widget_session._collect_input_hashes("consume")
    assert hashes_before, "downstream saw no upstream widget artifacts"

    # Change the control value and re-materialize the widget cell.
    persist_cell_widget_values(widget_session.path, "controls", {"alpha": 0.9})
    await executor.execute_cell("controls", _WIDGET_SRC)

    hashes_after = widget_session._collect_input_hashes("consume")
    assert hashes_after and hashes_after != hashes_before, (
        "downstream input hashes did not change when the widget value changed — "
        "its provenance would cache-hit the stale output"
    )
