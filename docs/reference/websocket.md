# WebSocket Protocol

The notebook UI communicates with the backend via a WebSocket connection for real-time updates.

For a client-author orientation that walks the bootstrap flow and load-bearing rules (path-parameter gotcha, owner gating, cold-start payload, grace window), start at the [Notebook Client Protocol](notebook-protocol.md) page; this page is the message-level reference.

Every frame type below corresponds to a member of `strata.notebook.protocol.MessageType` — that enum is the canonical source. If the tables here and the enum diverge, the enum wins.

## Connection

```
ws://localhost:8765/v1/notebooks/ws/{session_id}
```

The `{session_id}` is the one returned by `POST /v1/notebooks/open` or `/create`. A session is single-process: opening the same notebook from a second tab returns a different session ID and runs an isolated execution context.

In service mode (proxy auth), the same headers required for REST endpoints — `X-Strata-Principal`, `X-Strata-Proxy-Token`, and `X-Tenant-ID` if multi-tenant — must be present on the WebSocket upgrade. A missing or invalid token closes the connection with `1008 Policy Violation`.

## Envelope

All messages are JSON with this shape:

```json
{
  "type": "message_type",
  "seq": 1,
  "ts": "2026-01-01T00:00:00Z",
  "payload": { ... }
}
```

`seq` and `ts` are present on server → client messages; the server doesn't require them on client → server. See [Sequence numbers](#sequence-numbers) below.

## Client → Server Messages

### Cell Execution

| Type                   | Payload                                  | Description                                                 |
| ---------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| `cell_execute`         | `{ "cell_id": "..." }`                   | Run cell (triggers cascade check)                           |
| `cell_execute_cascade` | `{ "cell_id": "...", "plan_id": "..." }` | Confirm cascade execution                                   |
| `cell_execute_force`   | `{ "cell_id": "..." }`                   | Run cell ignoring staleness (no upstream materialization)   |
| `cell_execute_rerun`   | `{ "cell_id": "..." }`                   | Force re-execute target cell while cascading upstream rebuilds |
| `cell_cancel`          | `{ "cell_id": "..." }`                   | Cancel running cell                                         |
| `notebook_run_all`     | `{ "continue_on_error": true }`          | Run all cells in topological order (default continues on error) |
| `notebook_rerun_all`   | `{ "continue_on_error": true }`          | Re-execute every cell with cache off                        |

### Cell Editing

| Type                 | Payload                                 | Description    |
| -------------------- | --------------------------------------- | -------------- |
| `cell_source_update` | `{ "cell_id": "...", "source": "..." }` | Source changed |

### Cell Tests

| Type             | Payload                                      | Description                                                                              |
| ---------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `cell_run_tests` | `{ "cell_id": "...", "test_source": "..." }` | Persist the cell's unit-test source (`cells/{id}.test.py`) and run it. Python cells only. |

### State

| Type                     | Payload                | Description                           |
| ------------------------ | ---------------------- | ------------------------------------- |
| `notebook_sync`          | `{}`                   | Request full state (for reconnection) |
| `impact_preview_request` | `{ "cell_id": "..." }` | Get upstream/downstream effects       |
| `profiling_request`      | `{}`                   | Get execution metrics                 |

### Inspect REPL

| Type            | Payload                               | Description         |
| --------------- | ------------------------------------- | ------------------- |
| `inspect_open`  | `{ "cell_id": "..." }`                | Open REPL for cell  |
| `inspect_eval`  | `{ "cell_id": "...", "expr": "..." }` | Evaluate expression |
| `inspect_close` | `{ "cell_id": "..." }`                | Close REPL          |

### Dependencies

| Type                | Payload                | Description                                                     |
| ------------------- | ---------------------- | --------------------------------------------------------------- |
| `dependency_add`    | `{ "package": "..." }` | Compatibility shorthand for starting an `add` environment job   |
| `dependency_remove` | `{ "package": "..." }` | Compatibility shorthand for starting a `remove` environment job |

### Variants

| Type                 | Payload                              | Description                                |
| -------------------- | ------------------------------------ | ------------------------------------------ |
| `variant_set_active` | `{ "group": "...", "name": "..." }`  | Switch the active variant in a group       |
| `variant_add`        | `{ "group": "..." }`                 | Add a new variant cell, cloning the active |

### AI Agent (client → server)

| Type                      | Payload                                                | Description                                       |
| ------------------------- | ------------------------------------------------------ | ------------------------------------------------- |
| `agent_cancel`            | `{}`                                                   | Cancel a running AI agent                         |
| `agent_confirm_response`  | `{ "job_id": "...", "approved": true, ... }`           | Reply to an `agent_confirm_request` from the server |

## Server → Client Messages

### Cell Status

| Type                      | Payload                                                                                                                  | Description                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| `cell_status`             | `{ "cell_id": "...", "status": "running" }`                                                                              | Status changed                                   |
| `cell_output`             | `{ "cell_id": "...", "outputs": {...}, "display": {...}, "displays": [...], "cache_hit": false }`                        | Execution result, including rich visible outputs |
| `cell_output_delta`       | `{ "cell_id": "...", "attempt": 1, "kind": "delta", "text": "..." }`                                                     | Streamed partial output while the cell runs (today: prompt cells). `kind: "delta"` appends `text` to a per-cell buffer; `kind: "retry"` means schema validation failed — clear the buffer, `attempt` is the new attempt number, `text` is the first validator error. Ephemeral: never persisted or replayed; the final `cell_output` is canonical. Cache hits emit no deltas. |
| `cell_console`            | `{ "cell_id": "...", "stream": "stdout", "text": "..." }`                                                                | Incremental output                               |
| `cell_error`              | `{ "cell_id": "...", "error": "..." }`                                                                                   | Execution error                                  |
| `cell_iteration_progress` | `{ "cell_id": "...", "iteration": 3, "max_iter": 50, "artifact_uri": "...", "content_type": "...", "duration_ms": 128 }` | Per-iteration update from a `@loop` cell         |
| `cell_test_status`        | `{ "cell_id": "...", "status": "running" }`                                                                              | Test run lifecycle: `running` → `ready` / `error` (mirrors `cell_status`) |
| `cell_test_results`       | `{ "cell_id": "...", "passed": 2, "failed": 1, "errored": 0, "skipped": 0, "tests": [{ "name": "...", "nodeid": "...", "outcome": "passed", "message": "..." }], "stale": false, "pytest_unavailable": false, "ran_at": 1718000000000 }` | Per-test outcomes + totals from a `cell_run_tests`. `outcome` ∈ `passed`/`failed`/`error`/`skipped`; `message` carries the rewritten-assert diff for failures. `stale` flags the result against a since-changed cell/test/input. |

### Cascade

| Type               | Payload                                                                      | Description                   |
| ------------------ | ---------------------------------------------------------------------------- | ----------------------------- |
| `cascade_prompt`   | `{ "cell_id": "...", "plan_id": "...", "cells_to_run": [...], "estimated_duration_ms": 0 }` | Upstream cells need execution |
| `cascade_progress` | `{ "plan_id": "...", "current_cell_id": "...", "completed": 1, "total": 3 }` | Cascade progress              |

### DAG

| Type         | Payload                                               | Description                 |
| ------------ | ----------------------------------------------------- | --------------------------- |
| `dag_update` | `{ "edges": [...], "roots": [...], "leaves": [...] }` | DAG changed after cell edit |

### State

| Type                | Payload                                                               | Description                              |
| ------------------- | --------------------------------------------------------------------- | ---------------------------------------- |
| `notebook_state`    | `{ "id": "...", "cells": [...], "dag": {...} }`                       | Full state (response to `notebook_sync`) |
| `impact_preview`    | `{ "target_cell_id": "...", "upstream": [...], "downstream": [...] }` | Impact analysis result                   |
| `profiling_summary` | `{ "total_execution_ms": ..., "cell_profiles": [...] }`               | Profiling metrics                        |

### Inspect

| Type             | Payload                                                           | Description |
| ---------------- | ----------------------------------------------------------------- | ----------- |
| `inspect_result` | `{ "action": "eval", "ok": true, "result": "42", "type": "int" }` | REPL result |

### Dependencies

| Type                       | Payload                                                   | Description                                      |
| -------------------------- | --------------------------------------------------------- | ------------------------------------------------ |
| `environment_job_started`  | `{ "environment_job": {...} }`                            | Background environment job accepted              |
| `environment_job_progress` | `{ "environment_job": {...} }`                            | Background environment job phase/log update      |
| `environment_job_finished` | `{ "environment_job": {...}, "environment": {...}, ... }` | Background environment job completed or failed   |
| `dependency_changed`       | `{ "package": "...", "action": "add", "success": true }`  | Legacy compatibility event after add/remove jobs |

### AI Agent

| Type                    | Payload                                                          | Description                                                  |
| ----------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------ |
| `agent_text_delta`      | `{ "job_id": "...", "text": "..." }`                             | Streaming token delta from the agent's assistant message     |
| `agent_confirm_request` | `{ "job_id": "...", "tool": "...", "args": {...}, ... }`         | Agent is asking the client to approve a destructive tool use |
| `agent_progress`        | `{ "job_id": "...", "event": "...", "detail": "...", ... }`      | Incremental agent-loop status (tool start/end, iteration)    |
| `agent_done`            | `{ "job_id": "...", "content": "...", "model": "...", ... }`     | Agent finished, failed, or was cancelled                     |

### Errors

| Type    | Payload              | Description    |
| ------- | -------------------- | -------------- |
| `error` | `{ "error": "..." }` | Protocol error |

## Sequence numbers

Every server → client message carries a `seq` from a single counter scoped to the **notebook session** (not the WebSocket connection). The counter increments on every outbound message; it persists across reconnects to the same session and only resets when the session itself is closed (via `DELETE /v1/notebooks/{session_id}` or a server restart).

What the client uses `seq` for:

- **Ordering.** Messages arrive in `seq` order under normal conditions. If your client coalesces state updates, key dedupe on `seq` rather than `type`.
- **Gap detection across reconnects.** After reconnecting, the first message you receive may have a `seq` far higher than the last one you saw — events emitted while you were disconnected are not buffered. Treat any gap (or any reconnect) as a reason to send `notebook_sync` and replace local state.
- **One-way ack.** The client doesn't echo `seq` back; the server tracks no per-connection ack state.

## Reconnection semantics

Disconnects happen — proxy timeouts, network drops, server restarts, tab sleep. The recovery protocol:

1. **Client reconnects** to `ws://.../v1/notebooks/ws/{session_id}` with the same session ID. The session itself is in-memory on the server and survives reconnects; it's cleaned up only when closed explicitly via `DELETE /v1/notebooks/{session_id}` or when the server restarts.
2. **Server accepts the reconnect** and resumes emitting messages from the session's existing `seq` counter (continuing, not resetting). If the previous client disconnected within the **60-second cancel grace window** and a cell is still running, the execution survives the disconnect — the client picks up where it left off.
3. **Client sends `notebook_sync`** as its first message after reconnecting. The server responds with `notebook_state` containing the full current state (cells, DAG, cell statuses, latest display outputs).
4. **Client replaces local state** with the synced payload and resumes listening.

There is **no replay** of missed messages — events emitted while the client was disconnected are lost. State persisted to the artifact store (`cell_output`, finished cell statuses) is recovered via `notebook_sync`; transient progress events (`cell_console` mid-stream, `cell_output_delta` for a streaming prompt cell, `cell_iteration_progress` for a `@loop` cell, `cascade_progress`) are not.

### Cancel-on-disconnect grace window

When the **last** WebSocket for a notebook drops, the handler schedules a teardown task instead of running it immediately. Any incoming upgrade for the same `session_id` within ~60 seconds cancels the pending task and resumes against the preserved execution and inspect state. Past the window, the running execution is cancelled and inspect REPLs are closed.

This is the trade-off Vue's close-tab-to-cancel semantics make with TUI-style transients: closing a tab still cancels (just after a ~60s delay), but a tmux detach or a network blip won't kill a long-running cell. The grace constant lives at `_GRACE_CANCEL_SECONDS` in `src/strata/notebook/ws.py` if you need to tune it for your deployment.

## Close codes

| Code | Meaning |
| --- | --- |
| `1000` | Normal closure (client or server initiated) |
| `1008` | Policy violation — session not found, ownership mismatch in per-user personal mode, or service-mode auth failure on the upgrade |

If the session has been closed server-side (notebook deleted, server restart), the WebSocket upgrade is refused with `1008`. The client should call `POST /v1/notebooks/open` to start a new session.

The server does not send protocol-level pings; the WebSocket library's default frame keepalive is what holds idle connections open. If your client sees no traffic for an extended period and you can't tell whether the connection is live, the safest probe is to send `notebook_sync` and watch for the response.
