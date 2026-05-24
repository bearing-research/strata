# Notebook Client Protocol

A single reference for writing a non-Vue client (TUI, scripting, third-party
integration) against the notebook backend. The deeper per-endpoint and
per-frame details live in [REST API Reference](rest-api.md) and
[WebSocket Protocol](websocket.md); this page is the orientation map plus the
load-bearing rules that aren't obvious from either of those individually.

If you want exhaustive request/response shapes, the live OpenAPI document at
`GET /openapi.json` is authoritative — Swagger UI is at `GET /docs`.

## What the backend is

The notebook backend is a FastAPI service that exposes:

| Surface | Purpose |
| --- | --- |
| `POST /v1/notebooks/...` (REST) | Lifecycle (open / create / import / delete), discovery, and every structural edit (cells, mounts, env, deps, workers, AI config) |
| `WS /v1/notebooks/ws/{session_id}` | Live execution: cell status, output streams, DAG updates, cascade prompts, inspect REPL, agent loop |

The Vue frontend is a thin consumer of both. Anything Vue can do, a second
client can do — there's no internal API.

## Bootstrap flow

The minimum sequence to render a notebook view:

1. **Open the notebook.** `POST /v1/notebooks/open` with the notebook
   directory path. The response carries everything you need to render the UI
   cold — see [Cold-start payload](#cold-start-payload) below. The
   `session_id` in the response is the route parameter for every subsequent
   call.

   Alternatively, if you already have a `session_id` from a previous open
   (page refresh case), `GET /v1/notebooks/sessions/{session_id}` returns
   the same payload shape.
2. **Connect the WebSocket.** `ws://.../v1/notebooks/ws/{session_id}`. The
   handler verifies the session exists and (if owned) that the caller is the
   owner — refuses with close code `1008 Notebook not found` otherwise. No
   initial frame is sent on accept.
3. **Send `notebook_sync`** as the first client → server message. The server
   answers with a `notebook_state` frame containing the same fields as the
   open response. This is your sole resync primitive on reconnects — there's
   no `resume_after_seq`.
4. **Listen.** Execution events (cell status, output, console, errors,
   cascade prompts, DAG updates, environment-job lifecycle, agent loop) all
   arrive over the WS. Subsequent structural edits — adding / removing /
   reordering cells, updating env / mounts / workers, dependency mutations
   — go via REST; the backend re-broadcasts the affected state through the
   WS automatically.

That is the entire bootstrap. The remaining sections of this page explain the
gotchas in that flow.

## Path parameter gotcha: session_id vs notebook id

The route parameter `{notebook_id}` (in both REST and WS) is **the
`session_id` returned from `POST /open`** — not the `notebook_id` field
inside `notebook.toml`. The TOML id is the on-disk stable identifier; the
session id is the runtime handle the server uses to look you up.

A non-Vue client that passes the TOML id will get clean 404s from every
endpoint. The Vue client doesn't have this confusion because it always
stores the session id from the open response.

## Auth and ownership

### Personal mode (default)

- **No header configured.** Single-user trust. Every caller can hit every
  endpoint. This is the local-dev default.
- **`STRATA_PERSONAL_MODE_USER_HEADER` set.** Personal mode behind an
  authenticating proxy (Cloudflare Access, Pomerium, …). The proxy injects
  the configured header (e.g. `X-Authenticated-User`); the backend reads it
  via `_caller_identity` and uses it for:
  - **Storage scoping** — each user gets a private subdir under the storage
    root (`/discover`, `/create`, path-keyed deletes).
  - **Owner stamping** — `notebook.toml` records `owner = "<header value>"`
    on create.
  - **Owner enforcement** — every `WS /{session_id}` upgrade and every
    REST `/{session_id}/...` route checks the caller's header against the
    notebook's owner. Mismatch returns close `1008` or HTTP 404 (same
    generic "Notebook not found" body, so probes can't enumerate owners).
  - **Unowned notebooks pass through** — legacy notebooks without an
    `owner` field (and notebooks created by services that don't send the
    header) accept any caller.

The owner gate was previously asymmetric — only WS upgrades enforced it; a
leaked `session_id` was a bearer capability for the REST surface. As of
PR #47 every `SessionDep` route inherits the check by routing through
`get_notebook_session`. There are no opt-outs.

### Service mode

- `auth_mode = "trusted_proxy"` is the deployment shape.
- Every `/v1/*` request needs `X-Strata-Principal`, `X-Strata-Proxy-Token`,
  and `X-Tenant-ID` (if multi-tenant). The WS upgrade carries the same
  headers; missing or invalid token closes with `1008`.
- Notebook-session lifecycle endpoints (`/open`, `/create`, session
  reconnect, `/discover`, path-keyed deletes) are personal-mode-only and
  return `400 Bad Request` in service mode — write surface in service mode
  routes through the artifact build pipeline instead. See the
  [REST API page](rest-api.md#authentication) for the full list.

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
| Runtime config | `default_parent_path`, configured Python versions, deployment mode, user-header status. |
| `id`, `name`, `owner`, `worker`, `timeout`, `env`, `ai` | `notebook.toml` |
| `env_sources`, `env_fetch_error`, `env_fetched_at` | Secret-manager fetch status |
| `workers`, `mounts`, `connections`, `malformed_connections`, `secret_manager_config`, `variant_groups` | `notebook.toml` |
| `cells` (full) | Source, status, display outputs, console stdout/stderr, provenance hashes, causality chains, DAG shadow warnings, per-cell overrides. |
| `environment` | Live: Python version, lockfile hash, package counts, last-synced timestamp, sync status. |
| `environment_job` / `environment_job_history` | Currently-running env mutation + recent past jobs. |

### What is *not* in the cold-start payload

Some Vue panels lazy-fetch additional data only when the user opens them.
A non-Vue client can ignore these until it actually needs to render the
corresponding panel:

| Lazy fetch | Triggered by | Why deferred |
| --- | --- | --- |
| `GET /{sid}/workers` | NotebookPage `onMounted` (worker badge in header) | Auto-detected backends (Docker, local) are runtime state, change between requests. Vue auto-fetches once on mount; a TUI can skip it until the user opens a worker panel. |
| `GET /{sid}/dependencies` | Environment panel open | Resolved deps from `uv.lock`; expensive on large lockfiles. The snapshot already has `environment.resolved_package_count`. |
| `GET /{sid}/environment` | Environment panel re-fetch | Refreshes after a mutation; snapshot has the version current at open. |
| `GET /{sid}/llm/models`, `GET /{sid}/llm/status` | LLM picker / panel open | Provider API call. |
| `GET /{sid}/connections/{name}/schema` | Connection detail open | Adapter call per connection. |
| `GET /{sid}/profiling-summary` | Profiling panel open | Computed on demand. |

## Reconnection and the cancel-on-disconnect grace window

The WS handler used to cancel any in-flight execution and drop inspect /
execution state the instant the last connection dropped. PR #48 introduced a
**60-second reconnect grace window**: when the last client disconnects, the
teardown is scheduled instead of run inline. Any incoming upgrade for the
same notebook within the window cancels the pending teardown and resumes
against the preserved execution state.

What this means for a client:

- **A tmux detach / VPN blip / browser refresh does not kill a running
  cell** as long as you reconnect within ~60 seconds.
- **Vue's "close tab to cancel"** still works — the user just waits past the
  window. The grace constant (`_GRACE_CANCEL_SECONDS` in `ws.py`) is a
  module-level number you can override at startup if you want a different
  default.
- **Missed deltas are not replayed.** Per-cell deltas emitted while you were
  disconnected (`cell_console` mid-stream, `cell_iteration_progress`,
  `cascade_progress`) are dropped. On reconnect, send `notebook_sync` and
  rebuild from the fresh `notebook_state`. Persisted state (finished
  `cell_status`, latest `cell_output`) survives — it's recovered through
  the snapshot.
- **Sequence numbers continue across reconnects.** Every server-to-client
  message carries a `seq` from a per-notebook counter. The counter doesn't
  reset on reconnect; if you see a large gap, that's expected — treat it as
  a hint to drop local in-flight state and replace from `notebook_state`.

See [WebSocket Protocol → Reconnection semantics](websocket.md#reconnection-semantics)
for the message-level detail.

## Message types

The full list of WS frame types lives on the [WebSocket Protocol](websocket.md)
page; the catch is that **every type now corresponds to a member of
`strata.notebook.protocol.MessageType`** (PR #46). Before that PR, seven
S→C frame types (`environment_job_*`, `dependency_changed`, `agent_*`) were
raw string literals emitted ad-hoc. A non-Vue client can now enumerate the
canonical set by iterating the enum:

```python
from strata.notebook.protocol import MessageType

for member in MessageType:
    print(member.name, member.value)
```

The enum is the single source of truth; the docs are organized for
human readability but the values match exactly. If you see a frame whose
`type` doesn't match an enum value, treat that as a bug to report.

## Where to go next

- [REST API Reference](rest-api.md) — every endpoint with request /
  response shapes.
- [WebSocket Protocol](websocket.md) — every C→S and S→C frame with
  payload shapes.
- [notebook.toml Schema](notebook-toml.md) — what the on-disk config
  looks like.
- [Configuration](configuration.md) — the server-side knobs
  (`personal_mode_user_header`, deployment mode, storage root, …).
