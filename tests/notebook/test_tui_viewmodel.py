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
    # Not-yet-handled frame type (e.g. impact_preview) → no-op.
    assert vm.apply_frame("impact_preview", {"target_cell_id": "a"}) == set()
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


def test_cascade_progress_sets_banner():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    changed = vm.apply_frame(
        "cascade_progress",
        {"plan_id": "p", "current_cell_id": "a", "completed": 2, "total": 4},
    )
    assert changed == set()  # notebook-level, not a specific cell
    assert "2/4" in vm.banner and "a" in vm.banner


def test_cascade_prompt_sets_banner():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame("cascade_prompt", {"cell_id": "a", "plan_id": "p", "cells_to_run": ["x", "y"]})
    assert "2 upstream" in vm.banner


def test_environment_job_sets_banner():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame(
        "environment_job_progress",
        {"environment_job": {"action": "add", "package": "pytest", "status": "running"}},
    )
    assert "pytest" in vm.banner and "running" in vm.banner


def test_cell_iteration_progress_sets_iteration():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    changed = vm.apply_frame(
        "cell_iteration_progress", {"cell_id": "a", "iteration": 3, "max_iter": 10}
    )
    assert changed == {"a"}
    assert vm.cells["a"].iteration == "iter 3/10"


def test_agent_text_delta_streams_into_one_block():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame("agent_text_delta", {"job_id": "j", "text": "Let me "})
    vm.apply_frame("agent_text_delta", {"job_id": "j", "text": "look at the data."})
    assert vm.agent_feed == ["Let me look at the data."]  # merged into one entry
    assert vm.agent_status == "thinking"
    assert "agent" in vm.banner


def test_agent_progress_then_text_are_separate_entries():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame("agent_text_delta", {"text": "thinking"})
    vm.apply_frame("agent_progress", {"event": "tool_call", "detail": "edit cell a"})
    vm.apply_frame("agent_text_delta", {"text": "done editing"})
    assert vm.agent_feed == ["thinking", "• tool_call: edit cell a", "done editing"]


def test_agent_confirm_request_shows_awaiting_driver():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame("agent_confirm_request", {"job_id": "j", "description": "delete cell b"})
    assert vm.agent_status == "awaiting confirm"
    assert "awaiting driver" in vm.agent_feed[-1]
    assert "delete cell b" in vm.agent_feed[-1]


def test_agent_done_summarizes():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame(
        "agent_done",
        {"job_id": "j", "model": "claude", "tokens": {"input": 100, "output": 50}},
    )
    assert vm.agent_status == "done"
    assert "agent done" in vm.agent_feed[-1]
    assert "claude" in vm.agent_feed[-1] and "100+50" in vm.agent_feed[-1]


def test_agent_frames_are_notebook_level():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    # No cell id returned — agent activity isn't tied to one cell row.
    assert vm.apply_frame("agent_text_delta", {"text": "hi"}) == set()


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
