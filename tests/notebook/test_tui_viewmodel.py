"""Unit tests for the notebook TUI view model — the pure dispatch core.

No Textual, no sockets: feed the view model a fake-frame stream (the same shapes
the WS broadcasts) and assert the folded per-cell state. This is where the TUI's
logic lives; the app is a thin renderer over it.
"""

from __future__ import annotations

from strata.notebook.tui.viewmodel import NotebookViewModel


def _state(*cells: dict) -> dict:
    return {"name": "My Notebook", "cells": list(cells)}


def test_apply_notebook_state_seeds_cells_in_order():
    vm = NotebookViewModel()
    vm.apply_notebook_state(
        _state(
            {"id": "a", "source": "x = 1", "status": "ready", "language": "python", "name": "f"},
            {"id": "b", "source": "y = x + 1", "status": "idle"},
        )
    )
    assert vm.notebook_name == "My Notebook"
    assert vm.cell_order == ["a", "b"]
    assert vm.cells["a"].source == "x = 1"
    assert vm.cells["a"].status == "ready"
    assert vm.cells["a"].name == "f"
    assert vm.cells["b"].language == "python"  # defaulted


def test_display_outputs_normalized_from_snapshot():
    vm = NotebookViewModel()
    vm.apply_notebook_state(
        _state(
            {
                "id": "a",
                "display_outputs": [{"content_type": "text/markdown", "markdown_text": "# hi"}],
            },
            {"id": "b", "display_output": {"content_type": "image/png", "inline_data_url": "x"}},
        )
    )
    assert vm.cells["a"].display_outputs[0]["markdown_text"] == "# hi"
    # The singular display_output is normalized to a one-element list.
    assert vm.cells["b"].display_outputs[0]["content_type"] == "image/png"


def test_cell_status_frame_updates_status():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a", "status": "idle"}))
    assert vm.apply_frame("cell_status", {"cell_id": "a", "status": "running"}) == {"a"}
    assert vm.cells["a"].status == "running"


def test_cell_console_accumulates():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame("cell_console", {"cell_id": "a", "stream": "stdout", "text": "line1\n"})
    vm.apply_frame("cell_console", {"cell_id": "a", "stream": "stderr", "text": "line2\n"})
    assert vm.cells["a"].console == "line1\nline2\n"


def test_cell_output_sets_outputs_and_clears_error():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame("cell_error", {"cell_id": "a", "error": "boom"})
    assert vm.cells["a"].error == "boom"
    vm.apply_frame("cell_output", {"cell_id": "a", "outputs": [{"name": "x", "preview": 1}]})
    assert vm.cells["a"].outputs == [{"name": "x", "preview": 1}]
    assert vm.cells["a"].error is None


def test_cell_output_delta_streams_and_retry_clears():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame("cell_output_delta", {"cell_id": "a", "kind": "delta", "text": "hel"})
    vm.apply_frame("cell_output_delta", {"cell_id": "a", "kind": "delta", "text": "lo"})
    assert vm.cells["a"].stream_text == "hello"
    # A retry frame (schema validation failed) clears the buffer first.
    vm.apply_frame("cell_output_delta", {"cell_id": "a", "kind": "retry", "text": "again"})
    assert vm.cells["a"].stream_text == "again"


def test_unknown_and_unaddressed_frames_are_noops():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    # Not-yet-handled frame type (M2+/M3) → no-op.
    assert vm.apply_frame("cascade_progress", {"plan_id": "p", "completed": 1}) == set()
    # Frame for an unknown cell → no-op (no crash).
    assert vm.apply_frame("cell_status", {"cell_id": "ghost", "status": "running"}) == set()
    # Frame with no cell_id → no-op.
    assert vm.apply_frame("agent_progress", {"step": 1}) == set()


def test_resync_preserves_live_console_and_outputs():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a", "status": "idle"}))
    vm.apply_frame("cell_console", {"cell_id": "a", "text": "kept\n"})
    vm.apply_frame("cell_output", {"cell_id": "a", "outputs": [{"name": "x"}]})
    # A fresh snapshot (e.g. manual resync) must not wipe what we already saw.
    vm.apply_notebook_state(_state({"id": "a", "status": "ready"}))
    assert vm.cells["a"].status == "ready"  # snapshot wins for status
    assert vm.cells["a"].console == "kept\n"  # live console preserved
    assert vm.cells["a"].outputs == [{"name": "x"}]  # live outputs preserved


def test_edges_parsed_from_notebook_state_dag():
    vm = NotebookViewModel()
    payload = {
        "name": "NB",
        "cells": [{"id": "a"}, {"id": "b"}],
        "dag": {"edges": [{"from_cell_id": "a", "to_cell_id": "b", "variable": "x"}]},
    }
    vm.apply_notebook_state(payload)
    assert vm.edges == [("a", "b")]


def test_dag_update_frame_refreshes_edges():
    vm = NotebookViewModel()
    vm.apply_notebook_state({"name": "NB", "cells": [{"id": "a"}, {"id": "b"}, {"id": "c"}]})
    assert vm.edges == []
    changed = vm.apply_frame(
        "dag_update",
        {
            "edges": [
                {"from_cell_id": "a", "to_cell_id": "b"},
                {"from_cell_id": "b", "to_cell_id": "c"},
            ]
        },
    )
    assert vm.edges == [("a", "b"), ("b", "c")]
    assert changed == {"a", "b", "c"}  # whole-graph change


def test_malformed_dag_edges_ignored():
    vm = NotebookViewModel()
    vm.apply_notebook_state(
        {"name": "NB", "cells": [{"id": "a"}], "dag": {"edges": ["bad", {"from_cell_id": "a"}]}}
    )
    assert vm.edges == []  # entries missing a string from/to are skipped


def test_dropped_cell_is_removed_on_resync():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}, {"id": "b"}))
    vm.apply_notebook_state(_state({"id": "a"}))  # b removed
    assert vm.cell_order == ["a"]
    assert "b" not in vm.cells
