# WebSocket Protocol

The notebook UI communicates with the backend via a WebSocket connection for real-time updates.

For a client-author orientation that walks the bootstrap flow and load-bearing rules (path-parameter gotcha, tenant gating, cold-start payload, grace window), start at the [Notebook Client Protocol](notebook-protocol.md) page; this page is the message-level reference.

Every frame type below corresponds to a member of `strata.notebook.protocol.MessageType` - that enum is the canonical source. If the tables here and the enum diverge, the enum wins.

## Connection

```
ws://localhost:8765/v1/notebooks/ws/{session_id}
```

The `{session_id}` is the one returned by `POST /v1/notebooks/open` or `/create`. A session is single-process. On a personal server, opening the same notebook again returns the session already open on that path, so a second tab, the terminal viewer and an agent all drive one execution context and see each other's runs. In service mode each open creates its own session.

Under principal auth, the upgrade carries the same credentials as a REST call: `X-Strata-Principal`, `X-Strata-Proxy-Token`, and `X-Tenant-ID` if multi-tenant under `trusted_proxy`, or `Authorization: Bearer <key>` under `api_key`. A missing or invalid credential closes the connection with `1008 Policy Violation`. Each client → server frame is then checked against the notebook scope it needs (`notebook:read`, `notebook:write` or `notebook:execute`); a frame the principal's scopes do not cover is answered with an `error` frame, `code: "insufficient_scope"`.

A read-only app view connects with `?role=viewer`. Such a connection may send only `widget_update` and `notebook_sync`; anything else is answered with `code: "read_only"`.

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
| `cell_cancel`          | `{ "cell_id": "..." }`                   | Cancel running cell (see [Cancelling a SQL cell](#cancelling-a-sql-cell)) |
| `notebook_run_all`     | `{ "continue_on_error": true }`          | Run all cells in topological order (default continues on error) |
| `notebook_rerun_all`   | `{ "continue_on_error": true }`          | Re-execute every cell with cache off                        |

### Cell Editing

| Type                 | Payload                                                  | Description    |
| -------------------- | -------------------------------------------------------- | -------------- |
| `cell_source_update` | `{ "cell_id": "...", "source": "...", "force": false, "author": "..." }` | Source changed. Refused with `cell_locked` when someone else changed the cell moments ago, unless `force` is true. `author` is optional and used only without principal auth |
| `cell_focus`         | `{ "cell_id": "...", "author": "..." }`                 | The cell this client is on, or `null`. Updates `presence` for everyone |

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
| `variant_add`        | `{ "group": "...", "author": "..." }` | Add a new variant cell, cloning the active. `author` is optional |
| `widget_update`      | `{ "cell_id": "...", "values": { "<name>": <value> } }` | Set widget control value(s); re-materializes + stales downstream |

## Server → Client Messages

### Cell Status

| Type                      | Payload                                                                                                                  | Description                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| `cell_status`             | `{ "cell_id": "...", "status": "running", "remote_worker": "...", "remote_transport": "...", "remote_build_state": "...", "staleness_reasons": [...], "causality": {...} }` | Status changed. Only `cell_id` and `status` are always present: the `remote_*` fields come with `running` for a remote cell (`remote_build_state` is `starting` while a worker provisions the job), `staleness_reasons` and `causality` with a staleness update |
| `cell_output`             | `{ "cell_id": "...", "outputs": {...}, "display": {...}, "displays": [...], "cache_hit": false, "duration_ms": 128, "artifact_uri": "...", "artifact_uris": {...}, "stdout": "...", "stderr": "...", "execution_method": "...", "mutation_warnings": [...] }` | Execution result, including rich visible outputs. `artifact_uris` maps each stored variable to its artifact. A remote run adds `remote_worker`, `remote_transport`, `remote_build_id` and `remote_build_state` |
| `cell_output_delta`       | `{ "cell_id": "...", "attempt": 1, "kind": "delta", "text": "..." }`                                                     | Streamed partial output while the cell runs (today: prompt cells). `kind: "delta"` appends `text` to a per-cell buffer; `kind: "retry"` means schema validation failed - clear the buffer, `attempt` is the new attempt number, `text` is the first validator error. `kind: "notice"` is a provider-degradation message to show beside the stream, not part of its content. Ephemeral: never persisted or replayed; the final `cell_output` is canonical. Cache hits emit no deltas. |
| `cell_console`            | `{ "cell_id": "...", "stream": "stdout", "text": "...", "chunk_seq": 0 }`                                                | Incremental output. Append `text` to the cell's console for `stream`. A remote cell's output arrives while it runs, numbered per stream by `chunk_seq` from 0: chunk 0 starts that stream's console for the run, so replace what the last run left. `chunk_seq` is `null` on console sent when the cell finishes |
| `cell_error`              | `{ "cell_id": "...", "error": "...", "suggest_install": "...", "suggest_install_language": "python" }`                    | Execution error. `error_code`, when present, is a stable name to branch on (`fetch_pin_mismatch`: a `# @fetch ... sha256=` pin no longer matches the URL's bytes). `suggest_install` names a missing package to offer installing, with `python` or `r` for which installer. A remote run adds the `remote_*` fields of `cell_output` plus `remote_error_code` |
| `cell_iteration_progress` | `{ "cell_id": "...", "iteration": 3, "max_iter": 50, "artifact_uri": "...", "content_type": "...", "until_reached": false, "duration_ms": 128 }` | Per-iteration update from a `@loop` cell. `until_reached` is true on the iteration where `@loop_until` held |
| `cell_variant_progress`   | `{ "cell_id": "...", "variant": "rf", "index": 1, "total": 3, "success": true, "duration_ms": 128, "error": null }`        | Per-variant update from a `# @per_variant` fan-out cell |
| `cell_test_status`        | `{ "cell_id": "...", "status": "running" }`                                                                              | Test run lifecycle: `running` → `ready` / `error` (mirrors `cell_status`) |
| `cell_test_results`       | `{ "cell_id": "...", "passed": 2, "failed": 1, "errored": 0, "skipped": 0, "tests": [{ "name": "...", "nodeid": "...", "outcome": "passed", "message": "..." }], "stale": false, "pytest_unavailable": false, "ran_at": 1718000000000, "auto_installed": [] }` | Per-test outcomes + totals from a `cell_run_tests`. `outcome` ∈ `passed`/`failed`/`error`/`skipped`; `message` carries the rewritten-assert diff for failures. `stale` flags the result against a since-changed cell/test/input. `auto_installed` lists packages (pytest) installed into the notebook environment for this run. |

### Cascade

| Type               | Payload                                                                      | Description                   |
| ------------------ | ---------------------------------------------------------------------------- | ----------------------------- |
| `cascade_prompt`   | `{ "cell_id": "...", "plan_id": "...", "cells_to_run": [...], "estimated_duration_ms": 0 }` | Upstream cells need execution |
| `cascade_progress` | `{ "plan_id": "...", "current_cell_id": "...", "completed": 1, "total": 3 }` | Cascade progress              |

### DAG

| Type         | Payload                                               | Description                 |
| ------------ | ----------------------------------------------------- | --------------------------- |
| `dag_update` | `{ "edges": [...], "roots": [...], "leaves": [...], "topological_order": [...], "cells": [...], "variant_groups": [...] }` | DAG changed after cell edit. Each `cells` entry carries `id`, `defines`, `references`, `upstream_ids`, `downstream_ids`, `is_leaf`, `annotation_diagnostics`, `variant_group` / `variant_name` / `variant_active`, `is_module_cell` / `module_exports`, and `created_by` / `updated_by` |

### State

| Type                | Payload                                                               | Description                              |
| ------------------- | --------------------------------------------------------------------- | ---------------------------------------- |
| `notebook_state`    | `{ "id": "...", "cells": [...], "dag": {...}, "environment": {...}, "environment_job": {...}, "environment_job_history": [...], "r_environment": {...}, ... }` | Full state (response to `notebook_sync`). Carries the rest of the notebook's state as well (name, mounts, workers, env, variant groups) |
| `impact_preview`    | `{ "target_cell_id": "...", "upstream": [...], "downstream": [...], "estimated_ms": 0 }` | Impact analysis result. `upstream` entries carry `cell_id`, `cell_name`, `reason`, `skip`, `estimated_ms`; `downstream` entries carry `cell_id`, `cell_name`, `current_status`, `new_status` |
| `profiling_summary` | `{ "total_execution_ms": ..., "cache_hits": ..., "cache_misses": ..., "cache_savings_ms": ..., "team_cache_savings_ms": ..., "team_cache_hits": ..., "team_contributors": [...], "team_promotions": [...], "total_artifact_bytes": ..., "cell_profiles": [...] }` | Profiling metrics. `cell_profiles` entries carry `cell_id`, `cell_name`, `status`, `duration_ms`, `cache_hit`, `artifact_uri`, `execution_count` |

### Inspect

| Type             | Payload                                                           | Description |
| ---------------- | ----------------------------------------------------------------- | ----------- |
| `inspect_result` | `{ "cell_id": "...", "action": "eval", "expr": "...", "ok": true, "result": "42", "type": "int" }` | REPL result. `action` is `open`, `eval` or `close`; a failed eval carries `error` |

### Dependencies

| Type                       | Payload                                                   | Description                                      |
| -------------------------- | --------------------------------------------------------- | ------------------------------------------------ |
| `environment_job_started`  | `{ "environment_job": {...} }`                            | Background environment job accepted              |
| `environment_job_progress` | `{ "environment_job": {...} }`                            | Background environment job phase/log update      |
| `environment_job_finished` | `{ "environment_job": {...}, "environment_job_history": [...], "cells": [...], "lockfile_changed": false, "stale_cell_count": 0, "stale_cell_ids": [...], "environment": {...}, "r_environment": {...}, "dependencies": [...], "resolved_dependencies": [...] }` | Background environment job completed or failed. The environment and dependency fields are present only on success; an import job adds `warnings` and `imported_count` |
| `dependency_changed`       | `{ "package": "...", "action": "add", "success": true, "error": null, "lockfile_changed": true, "stale_cell_count": 0, "cells": [...] }` | Legacy compatibility event after add/remove jobs. On success it also carries `environment`, `dependencies` and `resolved_dependencies` |

### Agent

| Type         | Payload                                         | Description |
| ------------ | ----------------------------------------------- | ----------- |
| `agent_note` | `{ "source": "mcp" \| "agent", "text": "..." }` | An outside agent driving this notebook over MCP narrating what it did (`mcp`) or a note it pushed itself (`agent`) |

### Presence

| Type       | Payload                                                                                          | Description |
| ---------- | ------------------------------------------------------------------------------------------------ | ----------- |
| `presence` | `{ "principals": [{ "principal": "alice", "focused_cell_id": "c1", "since": 1789455008.4 }], "you": "bob" }` | Who is on the session and which cell each is on. Sent on connect, disconnect and focus change, and after an edit over REST |

### Session

| Type             | Payload                                    | Description |
| ---------------- | ------------------------------------------ | ----------- |
| `session_closed` | `{ "reason": "idle", "message": "..." }` | The server closed this session; the socket closes next with `1000`. `reason` is `idle`, `session_limit`, `memory`, `closed` (the close route) or `deleted`. Reopen the notebook by path; see [Session lifetime](#session-lifetime) |

### Errors

| Type    | Payload              | Description    |
| ------- | -------------------- | -------------- |
| `error` | `{ "error": "...", "code": "...", "cell_id": "...", "held_by": "..." }` | A request could not be served. Only `error` is always present; see [the `error` frame](notebook-protocol.md#the-error-frame) for the codes |

## Sequence numbers

Every server → client message carries a `seq` from a single counter scoped to the **notebook session** (not the WebSocket connection). The counter increments on every outbound message; it persists across reconnects to the same session and only resets when the session itself is closed (see [Session lifetime](#session-lifetime)).

What the client uses `seq` for:

- **Ordering.** Messages arrive in `seq` order under normal conditions. Every frame carries its own number, including the console and result of one execution and each cell of a batch, so if your client coalesces state updates, key dedupe on `seq` rather than `type`.
- **Gap detection across reconnects.** After reconnecting, the first message you receive may have a `seq` far higher than the last one you saw - events emitted while you were disconnected are not buffered. Treat any gap (or any reconnect) as a reason to send `notebook_sync` and replace local state.
- **One-way ack.** The client doesn't echo `seq` back; the server tracks no per-connection ack state.

## Reconnection semantics

Disconnects happen - proxy timeouts, network drops, server restarts, tab sleep. The recovery protocol:

1. **Client reconnects** to `ws://.../v1/notebooks/ws/{session_id}` with the same session ID. The session itself is in-memory on the server and survives reconnects until it is closed (see [Session lifetime](#session-lifetime)).
2. **Server accepts the reconnect** and resumes emitting messages from the session's existing `seq` counter (continuing, not resetting). If the previous client disconnected within the **60-second cancel grace window** and a cell is still running, the execution survives the disconnect - the client picks up where it left off.
3. **Client sends `notebook_sync`** as its first message after reconnecting. The server responds with `notebook_state` containing the full current state (cells, DAG, cell statuses, latest display outputs).
4. **Client replaces local state** with the synced payload and resumes listening.

There is **no replay** of missed messages - events emitted while the client was disconnected are lost. State persisted to the artifact store (`cell_output`, finished cell statuses) is recovered via `notebook_sync`; transient progress events (`cell_console` mid-stream, `cell_output_delta` for a streaming prompt cell, `cell_iteration_progress` for a `@loop` cell, `cell_variant_progress` for a `# @per_variant` fan-out cell, `cascade_progress`) are not. The one exception is a remote cell's live console: while it runs, its `console_stdout` / `console_stderr` in `notebook_state` hold the last 64 KiB it has streamed of each stream, so a client that joins mid-run shows the run so far and appends the `cell_console` frames that follow.

### Session lifetime

A session ends when:

- the notebook is deleted (`DELETE /v1/notebooks/{session_id}`, or `POST /v1/notebooks/delete-by-path`, which closes any session open on that directory),
- a client closes it (`POST /v1/notebooks/{session_id}/close`), which leaves the notebook on disk,
- nobody has edited, run or focused anything in it for `STRATA_NOTEBOOK_SESSION_TTL_SECONDS` (default 4 hours), with a tab connected or not,
- more than `STRATA_NOTEBOOK_MAX_SESSIONS` (default 50) are open: the least recently used is closed,
- `STRATA_NOTEBOOK_SESSION_MIN_AVAILABLE_MB` is set and the host's available memory is below it: the least recently used are closed until it is above,
- or the server restarts.

`notebook_sync`, previews, profiling requests, unknown frames and pings are not activity; a running cell is. The checks run every minute and when a notebook is opened, and never close a session with a running cell, an environment job, a held soft lock or a quiesce hold. Connected clients get `session_closed` with the reason before the socket closes. A reconnect to a closed session is refused with `1008`; call `POST /v1/notebooks/open` again for a new session ID. See [Session lifetime and `session_closed`](notebook-protocol.md#session-lifetime-and-session_closed).

### Cancelling a SQL cell

A cancel stops the notebook waiting for the cell: the run is abandoned, no
result is published, the cell leaves `running`, and the next cell can start.
The query itself keeps going in the database until it finishes, because ADBC
exposes cancellation per driver and several drivers (SQLite among them) answer
`NOT_IMPLEMENTED`. So a cancelled long query still costs the database what it
was going to cost; what it no longer costs is the notebook.

### Cancel-on-disconnect grace window

When the **last** WebSocket for a notebook drops, the handler schedules a teardown task instead of running it immediately. Any incoming upgrade for the same `session_id` within ~60 seconds cancels the pending task and resumes against the preserved execution and inspect state. Past the window, the execution that was running when the last socket dropped is cancelled and inspect REPLs are closed; a run another surface (REST, CLI, MCP) started during the window is left alone.

This is the trade-off Vue's close-tab-to-cancel semantics make with TUI-style transients: closing a tab still cancels (just after a ~60s delay), but a tmux detach or a network blip won't kill a long-running cell. The grace constant lives at `_GRACE_CANCEL_SECONDS` in `src/strata/notebook/ws.py` if you need to tune it for your deployment.

## Close codes

| Code | Meaning |
| --- | --- |
| `1000` | Normal closure (client or server initiated) |
| `1008` | Policy violation - session not found or another tenant's, or an auth failure on the upgrade |
| `1011` | Internal error while handling a frame |

If the session has been closed server-side (notebook deleted, evicted, server restart), the WebSocket upgrade is refused with `1008`. A session closed while you are connected sends `session_closed` and then closes with `1000`. The client should call `POST /v1/notebooks/open` to start a new session.

The server does not send protocol-level pings; the WebSocket library's default frame keepalive is what holds idle connections open. If your client sees no traffic for an extended period and you can't tell whether the connection is live, the safest probe is to send `notebook_sync` and watch for the response.
