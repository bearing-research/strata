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


def test_name_annotation_overrides_persisted_name():
    vm = NotebookViewModel()
    vm.apply_notebook_state(
        _state(
            # @name annotation wins over the persisted name field.
            {"id": "a", "source": "# @name load\nimport pyarrow", "name": "persisted"},
            # No annotation → fall back to the persisted name.
            {"id": "b", "source": "y = 1", "name": "kept"},
            # Neither → empty (the app shows the id prefix).
            {"id": "c", "source": "z = 1"},
        )
    )
    assert vm.cells["a"].name == "load"
    assert vm.cells["b"].name == "kept"
    assert vm.cells["c"].name == ""


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
    vm.apply_frame(
        "cell_output",
        {"cell_id": "a", "outputs": [{"name": "x", "preview": 1}], "duration_ms": 250},
    )
    assert vm.cells["a"].outputs == [{"name": "x", "preview": 1}]
    assert vm.cells["a"].error is None
    assert vm.cells["a"].duration_ms == 250  # timing captured from the frame


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


def test_resync_preserves_test_results():
    # Test results arrive only via cell_test_* frames; the periodic resync (a
    # fresh notebook_state) must not blank the badge + per-test cases.
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a", "status": "idle"}))
    vm.apply_frame(
        "cell_test_results",
        {
            "cell_id": "a",
            "passed": 2,
            "failed": 1,
            "tests": [{"name": "t", "outcome": "failed", "message": "boom"}],
        },
    )
    assert vm.cells["a"].test_summary == "✗ 2/3"
    vm.apply_notebook_state(_state({"id": "a", "status": "ready"}))
    assert vm.cells["a"].test_summary == "✗ 2/3"  # badge survives resync
    assert vm.cells["a"].test_cases[0]["name"] == "t"  # per-test cases survive


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


def test_cell_test_frames_set_badge_and_banner():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a", "name": "featurize"}))

    # running → pending badge + banner.
    assert vm.apply_frame("cell_test_status", {"cell_id": "a", "status": "running"}) == {"a"}
    assert vm.cells["a"].test_summary == "tests…"
    assert "running tests" in vm.banner

    # results → outcome badge + banner; the trailing ready status keeps the badge.
    changed = vm.apply_frame(
        "cell_test_results",
        {"cell_id": "a", "passed": 4, "failed": 0, "errored": 0, "skipped": 0, "stale": False},
    )
    assert changed == {"a"}
    assert vm.cells["a"].test_summary == "✓ 4/4"
    assert "✓ 4/4" in vm.banner
    vm.apply_frame("cell_test_status", {"cell_id": "a", "status": "ready"})
    assert vm.cells["a"].test_summary == "✓ 4/4"  # not clobbered


def test_cell_test_results_capture_per_test_cases():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame(
        "cell_test_results",
        {
            "cell_id": "a",
            "passed": 1,
            "failed": 1,
            "tests": [
                {"name": "test_ok", "outcome": "passed", "message": ""},
                {"name": "test_bad", "outcome": "failed", "message": "assert 1 == 2"},
            ],
        },
    )
    cases = vm.cells["a"].test_cases
    assert [c["name"] for c in cases] == ["test_ok", "test_bad"]
    assert cases[1]["outcome"] == "failed" and "assert 1 == 2" in cases[1]["message"]
    assert vm.cells["a"].test_unavailable is False


def test_cell_test_results_pytest_unavailable_flag():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    vm.apply_frame("cell_test_results", {"cell_id": "a", "pytest_unavailable": True})
    assert vm.cells["a"].test_unavailable is True
    assert vm.cells["a"].test_cases == []


def test_cell_test_results_failure_and_unavailable_badges():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))

    vm.apply_frame(
        "cell_test_results",
        {"cell_id": "a", "passed": 2, "failed": 1, "errored": 0, "skipped": 1, "stale": True},
    )
    assert vm.cells["a"].test_summary == "✗ 2/4 ·stale"

    vm.apply_frame("cell_test_results", {"cell_id": "a", "pytest_unavailable": True})
    assert vm.cells["a"].test_summary == "⚠ pytest n/a"


def test_point_to_point_frames_stay_noops():
    # impact_preview / profiling_summary / inspect_result are sent only to the
    # requesting client, so a read-only spectator never receives them.
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    assert vm.apply_frame("impact_preview", {"target_cell_id": "a"}) == set()
    assert vm.apply_frame("profiling_summary", {"total_execution_ms": 1}) == set()
    assert vm.apply_frame("inspect_result", {"cell_id": "a", "action": "open"}) == set()
    assert vm.banner == ""  # nothing surfaced


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


def test_agent_note_renders_mcp_action_and_explicit_note():
    vm = NotebookViewModel()
    vm.apply_notebook_state(_state({"id": "a"}))
    # An auto-narrated MCP tool action (source="mcp") → "↹" glyph.
    vm.apply_frame("agent_note", {"source": "mcp", "text": "ran cell a → ok"})
    assert vm.agent_feed[-1] == "↹ ran cell a → ok"
    # An explicit note the external agent pushed (source="agent") → "✎" glyph.
    vm.apply_frame("agent_note", {"source": "agent", "text": "refactoring featurize"})
    assert vm.agent_feed[-1] == "✎ refactoring featurize"
    assert vm.agent_status == "agent"


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
