# REST API Reference

This page documents the **notebook** REST surface, mounted under `/v1/notebooks`. Strata Core also exposes a `POST /v1/materialize` endpoint for direct artifact materialization — see the [Core Quickstart](../getting-started/core.md) for that surface.

!!! note "Session ID vs Notebook ID"
Route parameters use the **session ID** (a UUID generated when the notebook is opened), not the persistent `notebook_id` from `notebook.toml`. The session ID is returned by the `open` and `create` endpoints.

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
POST /v1/notebooks/{session_id}/cells/{cell_id}/execute
```

!!! tip
For interactive use, prefer the WebSocket `cell_execute` message. The REST endpoint is for programmatic access.

### List Loop Cell Iterations

```
GET /v1/notebooks/{session_id}/cells/{cell_id}/iterations?variable=<name>
```

Lists stored iteration artifacts for a `@loop` cell. The `variable` query
parameter defaults to the loop's `carry` variable if omitted. Non-loop cells
and loops with no completed iterations return an empty list — safe to poll
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

For long syncs prefer the background `POST /environment/jobs` path — this
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

### Export Notebook Bundle

```
GET /v1/notebooks/{session_id}/export
```

Returns a zip bundle containing `notebook.toml`, cells, and environment files.

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
refetches. An empty payload (all fields null) removes the block —
"disconnect from secret manager". Credentials are never part of this payload;
they must be exported in the server's shell environment.

### Refresh Secret Manager

```
POST /v1/notebooks/{session_id}/secret-manager/refresh
```

Re-fetches secrets from the configured manager and merges them into env.
Never returns 500 on fetch failure — the error surfaces in
`env_fetch_error` so the UI can display it next to the Refresh button.

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
