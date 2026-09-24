"""Property test: random edit/run sequences against a real notebook.

A diamond a -> (b, c) -> d of small integer cells, plus a sink e that
reads d (only variables some cell reads are stored as artifacts). Hypothesis picks a
sequence of source edits and runs; each step goes through the real
notebook WebSocket with real cell execution. Two oracles, both
independent of Strata's own staleness code:

1. A run's value equals evaluating the current sources from scratch.
2. A cell reported READY holds a stored value equal to that evaluation.

Needs Hypothesis, which is not a project dependency. Slow (real cells):

    uv run --with hypothesis pytest formal/test_staleness_properties.py
"""

# Fixtures are imported from tests/ and then requested by name (or autouse).
# ruff: noqa: F811

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import _reset_process_globals  # noqa: F401  (autouse)
from tests.notebook.conftest import fast_notebook_env  # noqa: F401  (autouse)
from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    _reset_ws_globals,
    create_test_app,
    execute_cell_and_wait,
    open_notebook_session,
    ws_connect,
)

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

CELLS = ["a", "b", "c", "d"]  # the cells with a value to check
SINK = ("e", "sink = w")
VAR = {"a": "x", "b": "y", "c": "z", "d": "w"}


def source(cell: str, k: int) -> str:
    return {
        "a": f"x = {k}",
        "b": f"y = x + {k}",
        "c": f"z = x * {k}",
        "d": f"w = y + z + {k}",
    }[cell]


def evaluate(ks: dict[str, int]) -> dict[str, int]:
    """Reference semantics: the current sources, run top to bottom."""
    x = ks["a"]
    y = x + ks["b"]
    z = x * ks["c"]
    return {"x": x, "y": y, "z": z, "w": y + z + ks["d"]}


steps = st.lists(
    st.one_of(
        st.tuples(st.just("edit"), st.sampled_from(CELLS), st.integers(0, 3)),
        st.tuples(st.just("run"), st.sampled_from([*CELLS, SINK[0]]), st.just(0)),
    ),
    min_size=1,
    max_size=6,
)


def stored_value(session, cell_id: str):
    cell = session.notebook_state.get_cell(cell_id)
    uri = cell.artifact_uris.get(VAR[cell_id]) if cell.artifact_uris else None
    if not uri:
        return None
    artifact_id, _, version = uri.removeprefix("strata://artifact/").partition("@v=")
    blob = session.get_artifact_manager().artifact_store.blob_store.read_blob(
        artifact_id, int(version)
    )
    return json.loads(blob)


@settings(
    max_examples=int(os.environ.get("STALENESS_EXAMPLES", "15")),
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
@given(steps)
def test_reported_ready_values_match_a_fresh_evaluation(plan):
    _reset_ws_globals()
    client = TestClient(create_test_app())
    ks = {cell: 1 for cell in CELLS}
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = NotebookBuilder(Path(tmpdir))
        previous = None
        for cell in CELLS:
            nb.add_cell(cell, source(cell, ks[cell]), previous)
            previous = cell
        nb.add_cell(*SINK, previous)
        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                for action, cell, k in plan:
                    if action == "edit":
                        ks[cell] = k
                        ws.update_source(cell, source(cell, k))
                        ws.receive_until("dag_update")
                    else:
                        execute_cell_and_wait(ws, cell)
                        if cell in VAR:
                            expected = evaluate(ks)[VAR[cell]]
                            got = stored_value(session, cell)
                            assert got == expected, f"run {cell}: stored {got}, expected {expected}"

                    truth = evaluate(ks)
                    for other in CELLS:
                        state = session.notebook_state.get_cell(other)
                        if state.status.value == "ready":
                            got = stored_value(session, other)
                            assert got == truth[VAR[other]], (
                                f"after {action} {cell}: {other} is READY holding "
                                f"{got}, current sources give {truth[VAR[other]]}"
                            )
