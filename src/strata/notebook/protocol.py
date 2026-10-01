"""Notebook client wire protocol.

The names and shapes a non-Vue client (TUI, scripts, integrations) needs
to talk to the notebook backend, with no dependency on FastAPI / Starlette
internals. Lives in its own module so importing the enum doesn't drag the
WS handler tree along, and so ``session.py`` / ``routes.py`` can reference
the same symbols without circular-import gymnastics.
"""

from __future__ import annotations

from enum import StrEnum


class MessageType(StrEnum):
    """Notebook WebSocket protocol message types.

    StrEnum so the dispatch keys and emit-site type fields stay in sync;
    a typo at any send site becomes an import-time error instead of a
    silent protocol drift the frontend would have to discover at
    runtime. StrEnum values remain plain ``str``, so existing tests and
    JSON serialization continue to interop.
    """

    # Client → Server
    CELL_EXECUTE = "cell_execute"
    CELL_EXECUTE_CASCADE = "cell_execute_cascade"
    CELL_EXECUTE_FORCE = "cell_execute_force"
    CELL_EXECUTE_RERUN = "cell_execute_rerun"
    CELL_CANCEL = "cell_cancel"
    NOTEBOOK_RUN_ALL = "notebook_run_all"
    NOTEBOOK_RERUN_ALL = "notebook_rerun_all"
    CELL_SOURCE_UPDATE = "cell_source_update"
    CELL_RUN_TESTS = "cell_run_tests"
    NOTEBOOK_SYNC = "notebook_sync"
    IMPACT_PREVIEW_REQUEST = "impact_preview_request"
    PROFILING_REQUEST = "profiling_request"
    INSPECT_OPEN = "inspect_open"
    INSPECT_EVAL = "inspect_eval"
    INSPECT_CLOSE = "inspect_close"
    DEPENDENCY_ADD = "dependency_add"
    DEPENDENCY_REMOVE = "dependency_remove"
    VARIANT_SET_ACTIVE = "variant_set_active"
    VARIANT_ADD = "variant_add"
    # Widget control value(s) changed. Payload: ``{cell_id, values: {name: value}}``
    # (partial allowed). Server persists the value, re-materializes the widget's
    # artifacts and marks downstream stale (the user then runs them).
    WIDGET_UPDATE = "widget_update"
    # The cell this client is on, or null. Payload: ``{cell_id, author?}``.
    CELL_FOCUS = "cell_focus"

    # Server → Client
    ERROR = "error"
    CELL_STATUS = "cell_status"
    CELL_OUTPUT = "cell_output"
    # Incremental output while a cell runs (today only prompt-cell LLM streaming).
    # Payload: ``{cell_id, attempt, kind, text}``; ``kind`` is ``"delta"`` (append) or
    # ``"retry"`` (validation failed: clear the buffer). Ephemeral: never persisted or
    # replayed; the final CELL_OUTPUT is canonical.
    CELL_OUTPUT_DELTA = "cell_output_delta"
    CELL_CONSOLE = "cell_console"
    CELL_ERROR = "cell_error"
    CELL_ITERATION_PROGRESS = "cell_iteration_progress"
    # One completed variant of a ``# @per_variant`` fan-out cell. Payload:
    # ``{cell_id, variant, index, total, success, duration_ms, error?}``.
    CELL_VARIANT_PROGRESS = "cell_variant_progress"
    # Cell unit-test lifecycle. CELL_TEST_STATUS mirrors CELL_STATUS ({cell_id, status});
    # CELL_TEST_RESULTS carries per-test outcomes, totals and a staleness flag.
    CELL_TEST_STATUS = "cell_test_status"
    CELL_TEST_RESULTS = "cell_test_results"
    DAG_UPDATE = "dag_update"
    NOTEBOOK_STATE = "notebook_state"
    CASCADE_PROMPT = "cascade_prompt"
    CASCADE_PROGRESS = "cascade_progress"
    IMPACT_PREVIEW = "impact_preview"
    INSPECT_RESULT = "inspect_result"
    PROFILING_SUMMARY = "profiling_summary"
    # Who is on the session and which cell each is on; sent on join, leave and focus
    # change. Payload: ``{principals: [{principal, focused_cell_id, since}], you}``.
    PRESENCE = "presence"

    # Server → Client (environment job lifecycle)
    ENVIRONMENT_JOB_STARTED = "environment_job_started"
    ENVIRONMENT_JOB_PROGRESS = "environment_job_progress"
    ENVIRONMENT_JOB_FINISHED = "environment_job_finished"
    # Legacy alias sent alongside ENVIRONMENT_JOB_FINISHED for add/remove so existing
    # dependency_changed listeners keep working.
    DEPENDENCY_CHANGED = "dependency_changed"

    # Server → Client (external agent)
    # An MCP/CLI agent's tool actions (source="mcp") or explicit narration
    # (source="agent"), shown as notes in the terminal viewer's Agent tab.
    AGENT_NOTE = "agent_note"


__all__ = ["MessageType"]
