# notebook.toml Schema

`notebook.toml` is the **committed** configuration for a notebook. It declares the cells, their metadata, the per-notebook environment, mounts, workers, database connections, and any AI/secret-manager wiring. The file is human-editable and git-diffable; the matching backend writer round-trips it so external edits survive UI saves.

Runtime state — display outputs, per-cell provenance hashes, console snapshots, `uv sync` timestamps — lives in `.strata/runtime.json`, **not** here. `notebook.toml` only changes on structural edits (add/remove/reorder a cell, change a worker/timeout/env/mounts/AI settings).

The schema is defined in `src/strata/notebook/models.py::NotebookToml` and the round-tripping rules in `src/strata/notebook/writer.py`.

## Top-level keys

```toml
notebook_id = "01HZJV4Y9G..."       # required; UUID-like, backend-generated on create
name = "Iris classifier"            # human-readable display name
owner = "alice@example.com"         # optional; stamped when STRATA_PERSONAL_MODE_USER_HEADER is set
created_at = 2026-04-12T10:31:00Z
updated_at = 2026-05-18T18:04:22Z

worker = "fly-cpu"                  # notebook-level default; overridden by @worker annotations
timeout = 300                       # notebook-level default in seconds; overridden by @timeout
```

| Key | Type | Description |
| --- | --- | --- |
| `notebook_id` | string (required) | Stable opaque ID generated on create. Never edit by hand. |
| `name` | string | Display name. Default: `"Untitled Notebook"`. |
| `owner` | string \| absent | Stamped on create when `STRATA_PERSONAL_MODE_USER_HEADER` is set and the request carries that header. Unowned notebooks (no key) are visible/deletable by any caller. |
| `created_at` | datetime (UTC) | Set on create; never updated. |
| `updated_at` | datetime (UTC) | Bumped on structural edits only. Runtime writers don't touch it. |
| `worker` | string \| absent | Notebook-level default worker name. Overridden by cell-level `worker` (below) or `# @worker` annotations. |
| `timeout` | float \| absent | Notebook-level default cell timeout in seconds. Same precedence as `worker`. |

## `[env]` — Notebook environment variables

```toml
[env]
LOG_LEVEL = "info"
DATA_BUCKET = "s3://my-bucket"
ANTHROPIC_API_KEY = ""              # blanked: keys matching KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL
```

**What gets blanked.** Values for keys whose name contains `KEY`, `SECRET`, `TOKEN`, `PASSWORD`, or `CREDENTIAL` are written as empty strings to disk. The actual secret is read from the runtime environment at execution time. The blanked entry is still committed so users can see which env vars a notebook *expects* without leaking the value into the repo.

**Whole-block elision.** If every entry is either empty or a blanked sensitive key, the writer omits the `[env]` block entirely on save. Typing an API key into the Runtime panel doesn't churn the committed file.

## `[ai]` — AI assistant configuration

```toml
[ai]
model = "claude-sonnet-4-6"
```

| Key | Type | Description |
| --- | --- | --- |
| `model` | string | Notebook-level default LLM model. Overridden by `# @model <id>` in prompt cells. Cleared by the AI panel when the user picks "use server default". |
| `approval_timeout_seconds` | float | How long an agent destructive-tool confirm prompt waits before being treated as a decline. Default 120. |

Advanced provider fields (`base_url`, `timeout_seconds`, token ceilings, …) are documented in [AI Integration](../notebook/ai.md#custom-provider-configuration).

The API key and base URL come from the runtime environment (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` / `STRATA_AI_API_KEY` / `STRATA_AI_BASE_URL`) — they never live in this file.

## `[secret_manager]` — External secret-manager wiring

```toml
[secret_manager]
provider = "infisical"
project_id = "62a9b0f1c4..."
environment = "prod"
path = "/notebook-secrets"
base_url = "https://app.infisical.com"
```

Routing-only config for an external secret manager (Infisical, Doppler, AWS SM, etc.). The authentication token for the manager itself lives in the strata-notebook's process environment, **not** here. See [Secret Manager](../notebook/secrets.md).

Allowed keys: `provider`, `project_id`, `environment`, `path`, `base_url`. Unknown keys are dropped on save.

## `[[mounts]]` — Filesystem mounts

```toml
[[mounts]]
name = "taxi_zones"
uri = "s3://nyc-tlc/misc"
mode = "ro"

[mounts.options]
anon = true
endpoint_url = "https://s3.us-east-1.amazonaws.com"
```

Each mount becomes a `pathlib.Path` variable in the cell namespace. Cells access remote data via standard Path operations; the executor materializes the prefix on first read.

| Key | Type | Description |
| --- | --- | --- |
| `name` | string (required) | Identifier — injected as a `Path` variable in the cell. Must match `[a-zA-Z_][a-zA-Z0-9_]*`. |
| `uri` | string (required) | `file:///path`, `s3://bucket/prefix`, `gs://bucket/prefix`, `az://container/prefix`. |
| `mode` | `"ro"` \| `"rw"` | Default `"ro"`. |
| `pin` | string \| absent | Pinned version/etag — disables auto-fingerprinting. |
| `options` | table | Backend storage options passed through to fsspec. Common keys: `anon`, `endpoint_url`, `profile`. |

Cell-level mounts (under `[[cells.mounts]]`) supplement notebook-level ones.

## `[[workers]]` — Remote worker registry

```toml
[[workers]]
name = "fly-cpu"
backend = "executor"
runtime_id = "fly-cpu-v1"

[workers.config]
url = "https://my-strata-worker.fly.dev/v1/execute"
transport = "http"
token_env = "STRATA_FLY_WORKER_TOKEN"
```

| Key | Type | Description |
| --- | --- | --- |
| `name` | string (required) | Worker name referenced in `# @worker <name>` annotations. Must match `[a-zA-Z0-9][a-zA-Z0-9._-]*`. |
| `backend` | `"local"` \| `"executor"` | Default `"local"`. `"executor"` for HTTP workers; `"local"` for in-process. |
| `runtime_id` | string \| absent | Stable fingerprint hashed into cell provenance. Bump to invalidate the cache for cells using this worker. |
| `config` | table | Backend-specific. For `"executor"`: `url`, `transport` (`"http"` or `"signed"`), `token` (literal — dev only), `token_env` (env var name — preferred). See [Distributed Workers](../notebook/workers.md). |

## `[connections.<name>]` — Named database connections

```toml
[connections.warehouse]
driver = "postgresql"
host = "warehouse.internal"
database = "analytics"

[connections.warehouse.auth]
user = "${WAREHOUSE_USER}"
password = "${WAREHOUSE_PASSWORD}"

[connections.warehouse.options]
application_name = "strata"
connect_timeout = 5
```

SQL cells reference these by name via `# @sql connection=<name>`.

| Key | Type | Description |
| --- | --- | --- |
| `<name>` | section header | Connection name — referenced by `# @sql connection=<name>`. Must match `[a-zA-Z_][a-zA-Z0-9_]*`. |
| `driver` | string (required) | One of the shipped adapters: `duckdb`, `sqlite`, `postgresql`, `snowflake`, `bigquery`. MotherDuck and MySQL are planned but not yet implemented. |
| `auth` | table | `${VAR}` indirections only. Resolved from the process environment at execute time; never hashed into provenance. |
| `options` | table | Driver-specific runtime tunables that don't change which objects the connection sees (`application_name`, `connect_timeout`, etc.). |
| (driver-specific top-level keys) | varies | `uri`, `host`, `account`, `database`, `role`, `path`, ... — interpreted by the driver adapter. |

**Malformed connection preservation.** If a `[connections.<name>]` block fails validation (bad name, missing `driver`, etc.), it's preserved verbatim under `[[malformed_connection]]` on save so a typo doesn't get silently erased by an unrelated edit. The annotation-validation layer surfaces a user-visible diagnostic.

## `[[variant_group]]` — Active-variant pointers

```toml
[[variant_group]]
group = "model_choice"
active = "logistic"
```

Cells declare group membership via `# @variant <group> <name>` in their source. This block records which member is currently active — only the active cell participates in the DAG.

| Key | Type | Description |
| --- | --- | --- |
| `group` | string (required) | Variant group identifier. Must match the group name used in `# @variant`. |
| `active` | string (required) | Currently active variant name within the group. |

## `[[cells]]` — Cell registry

```toml
[[cells]]
id = "a1b2c3d4"
file = "a1b2c3d4_load_data.py"
language = "python"
order = 100

[[cells]]
id = "e5f6g7h8"
file = "e5f6g7h8_train.py"
language = "python"
order = 200
worker = "fly-cpu"            # cell-level override of notebook default
timeout = 600                 # cell-level override

  [cells.env]                 # cell-level env additions (same blanking rules)
  CUDA_VISIBLE_DEVICES = "0"

  [[cells.mounts]]            # cell-level mount additions
  name = "model_weights"
  uri = "s3://my-models/bge-large"
  mode = "ro"
```

| Key | Type | Description |
| --- | --- | --- |
| `id` | string (required) | Stable cell identifier. Backend generates an 8-character UUID prefix when cells are created via UI / REST; hand-edits can use any unique string (e.g. `seed`, `top-orders`). This is what `@after` and `@loop start_from=` resolve against — **not** `@name`. See [Cell IDs](../notebook/annotations.md#cell-ids). |
| `file` | string (required) | Path to the cell source under `cells/`. |
| `language` | `"python"` \| `"prompt"` \| `"sql"` \| `"markdown"` | Default `"python"`. |
| `order` | float | Display order. Float so cells can be inserted between existing ones without renumbering. Default `0`. |
| `worker` | string \| absent | Cell-level worker override. Beaten by `# @worker` in the cell source. |
| `timeout` | float \| absent | Cell-level timeout override. Beaten by `# @timeout`. |
| `env` | table | Cell-level env additions / overrides. Same sensitive-key blanking as the notebook-level `[env]`. |
| `mounts` | array of `MountSpec` | Cell-level mounts. Supplement notebook-level mounts. |

## Annotation precedence

When the same piece of metadata is declared in multiple places, the most specific wins:

1. `# @worker X` / `# @timeout T` / `# @env K=V` annotations in the cell source (highest)
2. Cell-level `worker` / `timeout` / `env` in `[[cells]]`
3. Notebook-level `worker` / `timeout` / `env` in the top-level table

Annotations are the canonical per-cell configuration surface; there's no UI editor for per-cell overrides. See [Cell Annotations](../notebook/annotations.md).

## What lives elsewhere

| Belongs in `.strata/` (runtime state, gitignored) | Belongs in `notebook.toml` |
| --- | --- |
| Cached display outputs (DataFrames, plots, scalars) | Cell IDs, source filenames, order |
| Per-cell provenance hashes | Cell-level worker/timeout/env overrides |
| Per-cell console (stdout/stderr) snapshots | Mounts, workers, connections, variant pointers |
| `uv sync` timestamps | AI default model, secret-manager routing |
| Artifact-store SQLite + blobs | Notebook owner, name, created/updated timestamps |

Runtime writers never touch `notebook.toml`; structural-edit writers never touch `.strata/`. The invariant is enforced at the writer layer.

## Round-trip safety

The writer preserves unknown top-level keys verbatim. If you hand-edit the file with a key the parser doesn't know about, it survives saves — useful for experimental settings or external tooling. Known-but-malformed blocks (a `[connections.<name>]` with no `driver`, a typoed worker name) are also preserved under `[[malformed_connection]]` / `[[malformed_worker]]` so you don't lose the data while you debug.
