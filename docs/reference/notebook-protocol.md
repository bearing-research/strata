# Notebook Client Protocol

A single reference for writing a non-Vue client (TUI, scripting, third-party
integration) against the notebook backend. The deeper per-endpoint and
per-frame details live in [REST API Reference](rest-api.md) and
[WebSocket Protocol](websocket.md); this page is the orientation map plus the
load-bearing rules that aren't obvious from either of those individually.

If you want exhaustive request/response shapes, the live OpenAPI document at
`GET /openapi.json` is authoritative - Swagger UI is at `GET /docs`.

## What the backend is

The notebook backend is a FastAPI service that exposes:

| Surface | Purpose |
| --- | --- |
| `POST /v1/notebooks/...` (REST) | Lifecycle (open / create / import / delete), discovery, and every structural edit (cells, mounts, env, deps, workers) |
| `WS /v1/notebooks/ws/{session_id}` | Live execution: cell status, output streams, DAG updates, cascade prompts, inspect REPL, agent notes |

The Vue frontend is a thin consumer of both. Anything Vue can do, a second
client can do - there's no internal API.

## Bootstrap flow

The minimum sequence to render a notebook view:

1. **Open the notebook.** `POST /v1/notebooks/open` with the notebook
   directory path. The response carries everything you need to render the UI
   cold - see [Cold-start payload](#cold-start-payload) below. The
   `session_id` in the response is the route parameter for every subsequent
   call. The environment sync (`uv sync`, `renv` restore) runs as an
   environment job that the open waits for, so cells can run once it
   returns; sockets already connected to a reopened session see its
   `environment_job_*` frames.

   Alternatively, if you already have a `session_id` from a previous open
   (page refresh case), `GET /v1/notebooks/sessions/{session_id}` returns
   the same payload shape.
2. **Connect the WebSocket.** `ws://.../v1/notebooks/ws/{session_id}`. The
   handler verifies the session exists and is visible to the caller's tenant -
   refuses with close code `1008 Notebook not found` otherwise. A
   browser upgrade whose `Origin` is neither the server's own nor listed in
   `STRATA_CORS_ALLOW_ORIGINS` closes first with `1008 Origin not allowed`;
   clients that send no `Origin` (the TUI, scripts) are unaffected. The
   only frame sent on accept is `presence` (see
   [Presence and soft locks](#presence-and-soft-locks)).
3. **Send `notebook_sync`** as the first client → server message. The server
   answers with a `notebook_state` frame containing the same fields as the
   open response. This is your sole resync primitive on reconnects - there's
   no `resume_after_seq`.
4. **Listen.** Execution events (cell status, output, console, errors,
   cascade prompts, DAG updates, environment-job lifecycle, agent notes) all
   arrive over the WS. Subsequent structural edits - adding / removing /
   reordering cells, updating env / mounts / workers, dependency mutations
   - go via REST; the backend re-broadcasts the affected state through the
   WS automatically.

That is the entire bootstrap. The remaining sections of this page explain the
gotchas in that flow.

## Path parameter gotcha: session_id vs notebook id

The route parameter `{notebook_id}` (in both REST and WS) is **the
`session_id` returned from `POST /open`** - not the `notebook_id` field
inside `notebook.toml`. The TOML id is the on-disk stable identifier; the
session id is the runtime handle the server uses to look you up.

A non-Vue client that passes the TOML id will get clean 404s from every
endpoint. The Vue client doesn't have this confusion because it always
stores the session id from the open response.

## Auth

### Personal mode (default)

Single-user trust: a personal server has one user, and every caller can hit
every endpoint. This is the local-dev default.

### Service mode

- `auth_mode = "trusted_proxy"` is the usual deployment shape; `api_key`
  is the other authenticated mode.
- Under `trusted_proxy`, every `/v1/*` request needs `X-Strata-Principal`,
  `X-Strata-Proxy-Token`, and `X-Tenant-ID` (if multi-tenant); under
  `api_key`, `Authorization: Bearer <key>`. The WS upgrade carries the same
  credentials; a missing or invalid one closes with `1008`. Under `api_key`
  the key is checked again on every frame that needs `notebook:write` or
  `notebook:execute`, and the socket closes with `1008 Unauthorized` once
  the key is revoked or expired.
- `/open`, `/create` and `/discover` work in service mode. What is
  personal-mode-only is narrower: the two delete routes (`DELETE
  /{session_id}` and the path-keyed `POST /delete-by-path`) and the two
  `/sessions` routes, which return `403 Forbidden` elsewhere.
- `/open` returns the session the same principal already has open on that
  path, and never another principal's: service mode has no per-session owner
  check, so a session id is what keeps members of one tenant out of each
  other's live sessions. Without a principal, every open starts a new session.
- A session records the tenant of whoever opened, created or imported it.
  Another tenant's session looks missing: every `/{session_id}` REST route
  answers `404` and the WS upgrade closes with `1008 Notebook not found`, as
  for an unknown id (MCP applies the same rule). `admin:*` reaches every
  session, and a session opened without a tenant is open to all. With
  `multi_tenant_enabled`, `/discover`, `/open`, `/create` and the imports
  are confined to the tenant's own `<notebook_storage_dir>/<tenant>/`
  (`admin:*` keeps the whole root).
- Every route on the `/v1/notebooks` and `/v1/projects` routers is scope
  gated under principal auth, by the same `notebook:read` /
  `notebook:write` / `notebook:execute` table that checks the frames below
  - so a principal that cannot run a cell over this socket cannot run it
  over REST either. A route nobody classified requires `notebook:execute`.
  See the [REST API page](rest-api.md#authentication) for the full list.

## Cold-start payload

`POST /v1/notebooks/open` and `GET /v1/notebooks/sessions/{session_id}`
both return the **complete state needed to render the notebook view**. No
further calls are required before showing a useful UI. The shape is
`session.serialize_notebook_state()` plus four open-only fields:

| Field | Where it comes from |
| --- | --- |
| `session_id` | The route parameter for every subsequent call. |
| `path` | Absolute notebook directory path. |
| `dag` | Formatted upstream/downstream/staleness map. |
| Runtime config | `deployment_mode`, `default_parent_path`, `available_python_versions`, `default_python_version`, `python_selection_fixed`, `registry_enabled`, `team_store_configured`. |
| `id`, `name`, `worker`, `timeout`, `env`, `ai` | `notebook.toml`, plus fetched secrets in `env`; secret values are masked (see [Update Notebook Default Env](rest-api.md#update-notebook-default-env)) |
| `env_sources`, `env_fetch_error`, `env_fetched_at` | Secret-manager fetch status |
| `workers`, `mounts`, `connections`, `malformed_connections`, `secret_manager_config`, `variant_groups` | `notebook.toml` |
| `cells` (full) | Source, status, display outputs, console stdout/stderr, provenance hashes, causality chains, DAG shadow warnings, per-cell overrides. |
| `environment` | Live: Python version, lockfile hash, package counts, last-synced timestamp, sync status. |
| `environment_job` / `environment_job_history` | Currently-running env mutation + recent past jobs. |
| `r_environment` | The notebook's `renv` state, for R cells. |

### What is *not* in the cold-start payload

Some Vue panels lazy-fetch additional data only when the user opens them.
A non-Vue client can ignore these until it needs to render the
corresponding panel:

| Lazy fetch | Triggered by | Why deferred |
| --- | --- | --- |
| `GET /{sid}/workers` | NotebookPage `onMounted` (worker badge in header) | Auto-detected backends (Docker, local) are runtime state, change between requests. Vue auto-fetches once on mount; a TUI can skip it until the user opens a worker panel. |
| `GET /{sid}/dependencies` | Environment panel open | Resolved deps from `uv.lock`; expensive on large lockfiles. The snapshot already has `environment.resolved_package_count`. |
| `GET /{sid}/environment` | Environment panel re-fetch | Refreshes after a mutation; snapshot has the version current at open. |
| `GET /{sid}/connections/{name}/schema` | Connection detail open | Adapter call per connection. |
| WS `profiling_request` (answered with `profiling_summary`) | Profiling panel open | Computed on demand. |

## Reconnection and the cancel-on-disconnect grace window

When the last WS for a notebook drops, the handler does *not* cancel the
in-flight execution and drop inspect / execution state immediately.
Instead it schedules a teardown task that fires after a
**60-second reconnect grace window**; any incoming upgrade for the same
notebook within that window cancels the pending teardown and resumes
against the preserved execution state.

What this means for a client:

- **A tmux detach / VPN blip / browser refresh does not kill a running
  cell** as long as you reconnect within ~60 seconds.
- **Vue's "close tab to cancel"** still works - the user just waits past the
  window. The grace constant (`_GRACE_CANCEL_SECONDS` in `ws.py`) is a
  module-level number you can override at startup if you want a different
  default.
- **Only the run that was going when the last tab left is cancelled.** A run
  another surface (REST, CLI, MCP) starts during the window is left alone.
- **Missed deltas are not replayed.** Per-cell deltas emitted while you were
  disconnected (`cell_console` mid-stream, `cell_output_delta`,
  `cell_iteration_progress`, `cascade_progress`) are dropped. On reconnect, send `notebook_sync` and
  rebuild from the fresh `notebook_state`. Persisted state (finished
  `cell_status`, latest `cell_output`) survives - it's recovered through
  the snapshot. So does a running remote cell's console: the snapshot's
  `console_stdout` / `console_stderr` hold the last 64 KiB of what it has
  streamed so far, and the `cell_console` frames after it append.
- **Sequence numbers continue across reconnects.** Every server-to-client
  message carries a `seq` from a per-notebook counter. The counter doesn't
  reset on reconnect; if you see a large gap, that's expected - treat it as
  a hint to drop local in-flight state and replace from `notebook_state`.

See [WebSocket Protocol → Reconnection semantics](websocket.md#reconnection-semantics)
for the message-level detail.

## Session lifetime and `session_closed`

An open session holds the notebook's warm processes
(`STRATA_NOTEBOOK_WARM_POOL_SIZE` per pool). Closing one loses nothing else:
`runtime.json` and the notebook's artifact store hold what it computed, so
reopening it restores every result. The server closes a session when:

- **Nobody used it for `STRATA_NOTEBOOK_SESSION_TTL_SECONDS`** (default four
  hours), whether or not a tab is connected. Use is an edit, a run or a focus:
  a frame that needs `notebook:write` or `notebook:execute`, `cell_focus`, a
  REST call above `notebook:read`, or an MCP tool call. `notebook_sync`,
  previews, profiling requests, unknown frames, WebSocket pings and REST reads
  do not count, so a tab left open does not keep its session forever. A
  running cell does: idleness starts when the run ends.
- **More than `STRATA_NOTEBOOK_MAX_SESSIONS` are open.** The least recently
  used goes first.
- **Available memory is below `STRATA_NOTEBOOK_SESSION_MIN_AVAILABLE_MB`**
  (off by default; Linux only). The least recently used goes first, one at a
  time, until memory is above the floor or nothing idle is left. Checked
  before each open and every minute.
- **A client asks:** `POST /v1/notebooks/{session_id}/close` closes it without
  touching the notebook (`409` while a cell runs or the environment is
  changing). `DELETE /v1/notebooks/{session_id}` closes it as it deletes.

A server pass every minute applies the first three. None of them closes a
session with a running cell, an environment job, a cell soft lock still held,
or a [quiesce](rest-api.md) hold. Before the socket closes (code `1000`,
`Session closed`), each client gets a `session_closed` frame:

```json
{"type": "session_closed", "seq": 41, "payload": {
  "reason": "idle",
  "message": "This notebook was closed after a period without activity."
}}
```

`reason` is `idle`, `session_limit`, `memory`, `closed` or `deleted`. Do not
reconnect to the old `session_id` (the upgrade closes with `1008 Notebook not
found`); reopen the notebook by path with `POST /v1/notebooks/open`, which
starts a new session. The browser shows the message with a Reopen button;
the [terminal viewer](../notebook/tui.md#when-the-server-closes-the-session)
shows it in a notification and reopens on `r`.

## Message types

The full list of WS frame types lives on the [WebSocket Protocol](websocket.md)
page. **Every type corresponds to a member of
`strata.notebook.protocol.MessageType`** - that enum is the canonical
source. A non-Vue client can enumerate the full set by iterating it:

```python
from strata.notebook.protocol import MessageType

for member in MessageType:
    print(member.name, member.value)
```

The enum is the single source of truth; the docs are organized for
human readability but the values match exactly. If you see a frame whose
`type` doesn't match an enum value, treat that as a bug to report.

### Payload types

`MessageType` names the frames; it does not describe what rides in `payload`.
For the frames that have a payload model, TypeScript declarations are generated
from those models into `frontend/src/types/ws-payloads.generated.ts`:

```ts
import { isTypedFrame } from '@/types/notebook'

if (isTypedFrame(msg, 'error')) {
  // msg.payload.code narrows to the union of known codes, without a cast
}
```

The file is committed, so the frontend build needs no Python step. Regenerate it
after changing a payload model:

```bash
uv run python scripts/generate_ws_types.py
```

A test fails if the committed file and the models disagree, so the two cannot
drift apart quietly. Frames without a model keep `payload: unknown` and have to
be read from the emit site; `WsServerPayloadMap` lists the frames that do have
one.

### The `error` frame

Any request can be answered with `error` instead of the frame you expected,
so a client has to handle it on every message it sends:

| Field     | Always present | Meaning                                             |
| --------- | -------------- | --------------------------------------------------- |
| `error`   | yes            | Human-readable message. Not a stable identifier      |
| `code`    | no             | Machine-readable class, when the error has one       |
| `cell_id` | no             | The cell concerned, on `cell_busy` and `cell_locked` |
| `held_by` | no             | Who changed the cell, on `cell_locked`               |

Branch on `code`, never on the text of `error`. The codes:

| Code                 | Meaning                                                       |
| -------------------- | ------------------------------------------------------------- |
| `ENVIRONMENT_BUSY`   | An environment job holds the notebook; retry when it finishes  |
| `cell_busy`          | The cell is executing and its source cannot be edited yet      |
| `cell_locked`        | `held_by` changed the cell moments ago; resend with `force` to take it over |
| `read_only`          | The message is not allowed in app view                         |
| `insufficient_scope` | The connection's scopes do not cover the request               |

An error carrying no `code` is a plain failure with no machine-readable class;
the key is absent rather than null, so test for presence. The shape is defined
by `ErrorPayload` in `strata.notebook.ws_payloads`, which every emit site
builds through - a field that is not on that model cannot reach a client.

## Presence and soft locks

Several clients can be on one session: people in browser tabs, a TUI, an agent.
The server tells each of them who else is there.

**Identity.** Under `trusted_proxy` and `api_key` it is the principal. With no
authentication it is the `author` a client declares on its frames (`strata
agent` and MCP clients name themselves), and `local` when it declares none.
Several connections with one identity are one entry.

**`presence`** lists `{principal, focused_cell_id, since}` per identity, plus
`you`, the receiving connection's own identity, so a client can leave itself
out. It arrives on connect, whenever a connection joins, leaves or focuses a
cell, and after a cell edit over REST. An identity editing over REST has no
connection, so it is shown on the cell it edited for a minute after its last
edit.

**`cell_focus`** `{cell_id}` (or `null`) tells the server which cell this
connection is on. Editing a cell also focuses it.

**Soft locks.** The identity that last changed a cell holds it for
`STRATA_NOTEBOOK_CELL_LOCK_SECONDS` (default 5). An edit by a different
identity inside that window is not applied:

- over the WebSocket, `cell_source_update` is answered with an `error` frame,
  `code: "cell_locked"`, `cell_id` and `held_by`;
- over REST, `PUT .../cells/{cell_id}` returns `409` with the same `code`,
  `cell_id` and `held_by` in `detail`.

Resend with `force: true` to take the cell over. An identity never contends
with itself, so one person in several tabs, or a single-user personal session,
edits exactly as before. `0` turns the lock off.

## Where to go next

- [REST API Reference](rest-api.md) - every endpoint with request /
  response shapes.
- [WebSocket Protocol](websocket.md) - every C→S and S→C frame with
  payload shapes.
- [notebook.toml Schema](notebook-toml.md) - what the on-disk config
  looks like.
- [Configuration](configuration.md) - the server-side knobs
  (deployment mode, auth mode, storage root, …).
