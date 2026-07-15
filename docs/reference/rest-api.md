# REST API Reference

This page documents the **notebook** REST surface, mounted under `/v1/notebooks`. Strata Core also exposes a `POST /v1/materialize` endpoint for direct artifact materialization; see the [Library Quickstart](../getting-started/core.md) for that surface.

The remote-worker contract — `/v1/execute`, `/v1/notebook-execute`, `/v1/execute-manifest`, `/health` — is documented separately on the [Executor Protocol](executor-protocol.md) page. Those endpoints live on a different process (`strata-worker`), not the main server.

!!! note "Session ID vs Notebook ID"
Route parameters use the **session ID** (a UUID generated when the notebook is opened), not the persistent `notebook_id` from `notebook.toml`. The session ID is returned by the `open` and `create` endpoints.

## Conventions

### Canonical machine-readable spec

The server exposes a live OpenAPI document at runtime:

| Path | What it serves |
| --- | --- |
| `GET /openapi.json` | Full OpenAPI 3.1 schema — request and response models for every endpoint, generated from the FastAPI route definitions. |
| `GET /docs` | Swagger UI for interactive try-it-out. |
| `GET /redoc` | ReDoc browser for the same schema. |

When this page and the live spec disagree, the live spec wins — it's generated from the source of truth. This page exists for orientation and walks through the high-traffic flows.

### Authentication

Authentication depends on `deployment_mode`:

| Mode | Default auth | Required headers |
| --- | --- | --- |
| `personal` | None (single user) | None |
| `personal` + `STRATA_PERSONAL_MODE_USER_HEADER` | Caller identity from the named header (set by an authenticating proxy) | The header you configured (e.g. `X-Authenticated-User`) |
| `service` + `auth_mode="trusted_proxy"` | Proxy-injected identity | `X-Strata-Principal`, `X-Strata-Scopes`, `X-Strata-Proxy-Token`; `X-Tenant-ID` if multi-tenant |

In service mode, **every** `/v1/*` endpoint requires `X-Strata-Principal` (the proxy-asserted user identity) and the `X-Strata-Proxy-Token` shared secret. Two further restrictions narrow what each endpoint accepts:

**Personal-mode-only endpoints.** These notebook-session-lifecycle operations return `400 Bad Request` outside personal mode (long-lived sessions and filesystem notebook management don't fit a multi-tenant service deployment). Artifact and registry *writes*, by contrast, are available in service mode when `service_writes_enabled` is on, with the `artifacts:write` scope — see [Service Mode](../deployment/service-mode.md#authenticated-write-back-the-shared-research-store).

| Endpoint | Why personal-mode-only |
| --- | --- |
| `DELETE /v1/notebooks/{session_id}` | Filesystem delete of a notebook directory |
| `POST /v1/notebooks/delete-by-path` | Same, addressed by path |
| `GET /v1/notebooks/discover` | Walks the storage root for any notebook |
| Notebook session lifecycle (`/open`, `/create`, session reconnect) | Long-lived sessions land on one server process; service-mode multi-tenant deploys use the artifact API instead |

**Scope-gated endpoints.** These require a scope token in `X-Strata-Scopes` beyond authenticated principal:

| Endpoint | Required scope |
| --- | --- |
| `POST /v1/cache/clear` | `admin:cache` |

All other endpoints documented below need only `X-Strata-Principal` + `X-Strata-Proxy-Token` in service mode, and no auth in personal mode. Endpoints below carry a `Personal mode only` callout where applicable; otherwise treat them as available in both modes.

Personal mode with no header configured is effectively trust-on-first-call — anyone reaching the server can use it. Deploying personal mode to a public URL without an auth proxy is a [trust-model decision](../deployment/modes.md); see [Fly.io deployment](../deployment/fly.md#trust-model) for the load-bearing details.

### Error shape

All `4xx` and `5xx` responses use FastAPI's standard JSON shape:

```json
{"detail": "<human-readable error message>"}
```

Validation errors (`422`) come from Pydantic and contain structured field info:

```json
{
  "detail": [
    {
      "loc": ["body", "python_version"],
      "msg": "String should have at most 16 characters",
      "type": "string_too_long"
    }
  ]
}
```

### Status codes you'll see

| Status | Common cause |
| --- | --- |
| `200` | Success |
| `204` | Success, no body (e.g. `DELETE` operations) |
| `400` | Malformed request (invalid path, bad enum value, ACL block) |
| `401` | Service mode auth header missing or proxy-token mismatch |
| `403` | Authenticated, but missing the required scope (e.g. `admin:cache`) — returned as `404` if `STRATA_HIDE_FORBIDDEN_AS_NOT_FOUND=true` (the default) |
| `404` | Notebook session not found, or hidden 403 (see above) |
| `409` | Conflict — concurrent environment job, conflicting cell edit, or attempt to use a destructive endpoint outside personal mode |
| `413` | Request body or scan response exceeded the configured byte cap |
| `422` | Pydantic validation error on the request body |
| `429` | Rate limit exceeded — global, per-client, or per-tenant |
| `500` | Server bug — captured to logs with the request ID |

### Request IDs

Every request gets an `X-Request-ID` response header (and the same value is echoed if the client sent one in). Log lines and traces include this ID — copy it when filing bugs.

### Tenancy

In multi-tenant deployments, the `X-Tenant-ID` header (1–64 alphanumeric, `_`, `-`) scopes every endpoint to that tenant's namespace. The tenant ID is hashed into cache keys, artifact paths, and QoS limiter pools. ACL evaluation is **deny-first**: explicit denies cannot be overridden by allows. See [Service Mode](../deployment/service-mode.md) for the full contract.

## Notebook Lifecycle

### Create Notebook

```
POST /v1/notebooks/create
```

```json
{
  "parent_path": "/path/to/directory",
  "name": "My Notebook",
  "python_version": "3.13",
  "starter_cell": true
}
```

Returns notebook state with `session_id`.

### Open Notebook

```
POST /v1/notebooks/open
```

```json
{
  "path": "/path/to/notebook"
}
```

Returns notebook state with `session_id` and `dag`.

### Import Jupyter Notebook

```
POST /v1/notebooks/import
Content-Type: multipart/form-data
```

Form fields:

| Field         | Type            | Description                                                                  |
| ------------- | --------------- | ---------------------------------------------------------------------------- |
| `file`        | file (required) | The `.ipynb` upload. Hard cap of 50 MB.                                      |
| `name`        | string          | Override the notebook name (defaults to the upload's filename stem).         |
| `parent_path` | string          | Override the storage location (must lie inside the configured storage root). |

Converts the upload through `strata import` and opens a session on
the result. Returns the same notebook state shape as `POST
/v1/notebooks/create` plus an `import_report` field with the
converter's per-cell findings (translated magics, captured deps,
warnings, full report markdown). See
[Import from Jupyter](../notebook/import.md) for the magic
translation table and limitations.

### Delete Notebook

```
DELETE /v1/notebooks/{session_id}
```

Deletes the notebook directory and closes the session.

### Discover Notebooks

```
GET /v1/notebooks/discover
```

Lists notebook directories under the configured storage root. Returns
`{ "root", "notebooks": [{ "path", "name", "notebook_id", "updated_at" }] }`
sorted newest-first. Used by the "Open existing" UI so users pick from a list
instead of typing a filesystem path. **Personal mode only.**

### Delete Notebook By Path

```
POST /v1/notebooks/delete-by-path
```

```json
{
  "path": "/path/to/notebook"
}
```

Deletes a notebook directory by filesystem path. This is primarily a personal-mode
management endpoint.

### Rename Notebook

```
PUT /v1/notebooks/{session_id}/name
```

```json
{
  "name": "New Name"
}
```

## Sessions

### List Sessions

```
GET /v1/notebooks/sessions
```

Returns `{ "sessions": [{ "session_id", "name", "path", ... }] }`.

### Get Session

```
GET /v1/notebooks/sessions/{session_id}
```

Returns full notebook state (same shape as `open`). Used for page refresh reconnection.

## Cells

### List Cells

```
GET /v1/notebooks/{session_id}/cells
```

### Add Cell

```
POST /v1/notebooks/{session_id}/cells
```

```json
{
  "after_cell_id": "optional-cell-id",
  "language": "python"
}
```

`language` may be `python`, `prompt`, `markdown`, or `sql`. Defaults to `python`.

### Update Cell Source

```
PUT /v1/notebooks/{session_id}/cells/{cell_id}
```

```json
{
  "source": "x = 1"
}
```

Returns updated cell, DAG, and all cells (with refreshed staleness).

### Delete Cell

```
DELETE /v1/notebooks/{session_id}/cells/{cell_id}
```

### Reorder Cells

```
PUT /v1/notebooks/{session_id}/cells/reorder
```

```json
{
  "cell_ids": ["cell-1", "cell-3", "cell-2"]
}
```

### Execute Cell (REST)

```
POST /v1/notebooks/{session_id}/cells/{cell_id}/execute?mode=normal
```

The optional `mode` query parameter selects the run mode: `normal` (default —
use the cache and materialize stale upstreams), `rerun` (bypass the target
cell's cache, still materialize upstreams), or `force` (run against whatever
upstream artifacts already exist). An unrecognized mode returns `400`.

!!! tip
For interactive use, prefer the WebSocket `cell_execute` message. The REST endpoint is for programmatic access.

### Run Cell Tests (REST)

```
POST /v1/notebooks/{session_id}/cells/{cell_id}/tests
```

Runs the committed `cells/{cell_id}.test.py` via pytest against a re-executed
copy of the cell and returns per-test outcomes: `{ "cell_id", "passed",
"failed", "errored", "skipped", "pytest_unavailable", "ran_at", "tests": [{
"name", "nodeid", "outcome", "message" }] }`. Python cells only — a non-Python
cell or one with no test source returns `400`. The REST twin of the WebSocket
`cell_run_tests` message, for clients (the CLI, agents) that don't drive a
socket.

### List Loop Cell Iterations

```
GET /v1/notebooks/{session_id}/cells/{cell_id}/iterations?variable=<name>
```

Lists stored iteration artifacts for a `@loop` cell. The `variable` query
parameter defaults to the loop's `carry` variable if omitted. Non-loop cells
and loops with no completed iterations return an empty list, safe to poll
from the inspect panel.

Returns `{ "cell_id", "variable", "iterations": [{ "iteration", "artifact_uri",
"artifact_id", "version", "content_type", "byte_size", "row_count",
"created_at" }] }`.

## DAG

### Get DAG

```
GET /v1/notebooks/{session_id}/dag
```

Returns edges, roots, leaves, and topological order.

## Environment

### List Dependencies

```
GET /v1/notebooks/{session_id}/dependencies
```

### Add Dependency

```
POST /v1/notebooks/{session_id}/dependencies
```

```json
{
  "package": "pandas>=2.0"
}
```

### Remove Dependency

```
DELETE /v1/notebooks/{session_id}/dependencies/{package_name}
```

### Get Environment State

```
GET /v1/notebooks/{session_id}/environment
```

### Sync Environment

```
POST /v1/notebooks/{session_id}/environment/sync
```

Runs `uv sync` synchronously and invalidates any stale cell runtimes. Returns
the full environment payload plus `lockfile_changed`, `operation_log`
(command, duration, stdout/stderr), and the per-cell staleness map.

For long syncs prefer the background `POST /environment/jobs` path, this
endpoint blocks the request until the sync finishes.

### Get Current Environment Job

```
GET /v1/notebooks/{session_id}/environment/jobs/current
```

### Start Environment Job

```
POST /v1/notebooks/{session_id}/environment/jobs
```

```json
{
  "action": "add",
  "package": "scikit-learn"
}
```

Actions: `add`, `remove`, `sync`, `import`.

For `import`, send exactly one of `requirements` or `environment_yaml`.

### Export Requirements

```
GET /v1/notebooks/{session_id}/environment/requirements.txt
```

### Import Requirements

```
POST /v1/notebooks/{session_id}/environment/requirements.txt
```

### Preview Requirements Import

```
POST /v1/notebooks/{session_id}/environment/requirements.txt/preview
```

### Import environment.yaml

```
POST /v1/notebooks/{session_id}/environment/environment.yaml
```

### Preview environment.yaml Import

```
POST /v1/notebooks/{session_id}/environment/environment.yaml/preview
```

## Workers

### List Workers

```
GET /v1/notebooks/{session_id}/workers
```

### Update Notebook Worker

```
PUT /v1/notebooks/{session_id}/worker
```

```json
{
  "worker": "my-worker-name"
}
```

### Update Worker Catalog

```
PUT /v1/notebooks/{session_id}/workers
```

## Mounts

### Update Notebook Mounts

```
PUT /v1/notebooks/{session_id}/mounts
```

## Connections

### List Notebook Connections

```
GET /v1/notebooks/{session_id}/connections
```

Returns:

```json
{
  "connections": [
    {
      "name": "warehouse",
      "driver": "sqlite",
      "path": "analytics.db"
    }
  ]
}
```

### Replace Notebook Connections

```
PUT /v1/notebooks/{session_id}/connections
```

```json
{
  "connections": [
    {
      "name": "warehouse",
      "driver": "sqlite",
      "path": "analytics.db"
    }
  ]
}
```

The list is canonical: sending an empty list deletes the entire `[connections]`
block. Literal auth values are blanked on disk during the write round-trip, but
kept in-memory until the session reloads.

Returns:

```json
{
  "connections": [...],
  "malformed_connections": [...],
  "cells": [...]
}
```

### Enumerate Connection Schema

```
GET /v1/notebooks/{session_id}/connections/{name}/schema
```

Enumerates the tables and columns visible through the named connection. Used by
the schema sidebar. Opens the connection on the read path and returns backend
errors directly as `4xx` so auth / driver / connectivity failures are visible
to the UI.

## Export

### Export Notebook

```
GET /v1/notebooks/{session_id}/export?fmt={zip,markdown,html}
```

One endpoint, three output formats:

| `fmt`        | Returns                                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| `zip` *(default)* | Reproducible bundle, `notebook.toml`, `pyproject.toml`, `uv.lock`, cells, `provenance.json`.        |
| `markdown`   | Single-file rendering for sharing / docs ingestion. Same engine as `strata export`.                      |
| `html`       | Standalone HTML with embedded CSS + Pygments syntax highlighting.                                        |

Markdown and HTML renderings additionally accept `include_inactive_variants=true` to stack all variants of every group. Prompt-cell responses are intentionally excluded from rendered formats (see [Export](../notebook/export.md)).

## AI

### Get AI Status

```
GET /v1/notebooks/{session_id}/ai/status
```

### List Provider Models

```
GET /v1/notebooks/{session_id}/ai/models
```

### Update Notebook AI Model

```
PUT /v1/notebooks/{session_id}/ai/model
```

```json
{
  "model": "gpt-5.4"
}
```

### Chat Completion

```
POST /v1/notebooks/{session_id}/ai/complete
```

### Streaming Chat

```
POST /v1/notebooks/{session_id}/ai/stream
```

Server-Sent Events stream with `delta`, `done`, and `error` events.

### Agent Run

```
POST /v1/notebooks/{session_id}/ai/agent
```

### Reset Agent Session

```
POST /v1/notebooks/{session_id}/ai/agent/reset
```

Clears the assistant's in-memory conversation / tool session for that notebook.

## Runtime

### Get Server Runtime Config

```
GET /v1/notebooks/config
```

Returns deployment mode, available Python versions, and default paths for the
server as a whole. Not notebook-scoped.

### Update Notebook Default Timeout

```
PUT /v1/notebooks/{session_id}/timeout
```

```json
{
  "timeout": 60
}
```

`timeout` is seconds (0 < t ≤ 86400) or `null` to clear back to the system
default. Returns the new timeout and the refreshed cell list.

### Update Notebook Default Env

```
PUT /v1/notebooks/{session_id}/env
```

```json
{
  "env": {
    "OPENAI_API_KEY": "sk-...",
    "LOG_LEVEL": "info"
  }
}
```

Replaces the `[env]` block in `notebook.toml`. Sensitive values (keys matching
`KEY`/`SECRET`/`TOKEN`/`PASSWORD`/`CREDENTIAL`) are blanked on disk but kept
in-memory for the session so key-dependent cells keep working. Returns the
merged env, per-key sources, and refreshed cell list.

### Update Secret Manager Config

```
PUT /v1/notebooks/{session_id}/secret-manager/config
```

```json
{
  "provider": "infisical",
  "project_id": "your-project-id",
  "environment": "dev",
  "path": "/",
  "base_url": null
}
```

Persists the `[secret_manager]` block to `notebook.toml` and immediately
refetches. An empty payload (all fields null) removes the block:
"disconnect from secret manager". Credentials are never part of this payload;
they must be exported in the server's shell environment.

### Refresh Secret Manager

```
POST /v1/notebooks/{session_id}/secret-manager/refresh
```

Re-fetches secrets from the configured manager and merges them into env.
Never returns 500 on fetch failure, the error surfaces in
`env_fetch_error` so the UI can display it next to the Refresh button.

## Observability

### Server logs

```
GET /v1/logs
GET /v1/logs/stream
```

`GET /v1/logs` returns a snapshot of the in-memory log ring buffer (most recent
records first); `GET /v1/logs/stream` is a Server-Sent Events stream that tails
new records live. Both accept optional `level` / `logger` filters. Powers the
web UI **Logs** page.

### Artifacts listing

```
GET /v1/artifacts
GET /v1/artifacts/stats
```

`GET /v1/artifacts` lists stored artifacts with optional `since` (ISO timestamp),
`sort`, and `order` query parameters; `GET /v1/artifacts/stats` returns summary
counts and byte totals. Powers the web UI **Artifacts** page.

## Core API

### Materialize

```
POST /v1/materialize
```

```json
{
  "inputs": ["file:///warehouse#db.events"],
  "transform": {
    "executor": "scan@v1",
    "params": { "columns": ["id", "value"] }
  },
  "mode": "stream",
  "name": "my_result"
}
```

### Get Stream

```
GET /v1/streams/{stream_id}
```

Returns Arrow IPC stream.

### Health

```
GET /health
```

### Metrics

```
GET /metrics
GET /metrics/prometheus
```
