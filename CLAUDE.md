# CLAUDE.md

Guidance for Claude Code working in this repo.

## Working Guidelines

Behavioral guidelines to reduce common LLM coding mistakes (adapted from
[andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md)).
These bias toward caution over speed; for trivial tasks, use judgment.

1. **Think before coding.** Don't assume, don't hide confusion, surface
   tradeoffs. State assumptions explicitly and ask when uncertain; if
   multiple interpretations exist, present them rather than picking
   silently; if a simpler approach exists, say so and push back when
   warranted.
2. **Simplicity first.** Minimum code that solves the problem, nothing
   speculative — no unrequested features, abstractions for single-use
   code, configurability, or error handling for impossible scenarios. If
   200 lines could be 50, rewrite it.
3. **Surgical changes.** Touch only what you must; match existing style
   even if you'd do it differently; don't refactor what isn't broken.
   Remove only the imports/variables your own changes orphaned — mention
   pre-existing dead code, don't delete it. Every changed line should
   trace directly to the request.
4. **Goal-driven execution.** Turn tasks into verifiable goals ("fix the
   bug" → "write a test that reproduces it, then make it pass"); state a
   brief plan with a verify step per item; loop until verified.

## What is Strata?

Strata is a **materialization and persistence layer** for long-running, iterative,
expensive computations. It exposes a single primitive:

```
materialize(inputs, transform, environment) → artifact
```

Results are immutable + versioned, identical computations are deduplicated via a
provenance hash, and lineage is explicit. Strata sits **below orchestration** and
**outside execution** — it is not a workflow engine, scheduler, or query engine.

```
Orchestration → decides what to run
Executors     → decide how to compute
Strata        → decides whether it already exists, and persists it
```

Strata also provides snapshot-aware Iceberg scanning. Parquet row groups are
cached as Arrow IPC keyed by immutable snapshot IDs (no cache invalidation).
Streaming is bounded-memory; two-tier QoS prevents bulk queries from starving
dashboards. A small Rust extension (`rust/`) accelerates two hot paths only:
mmap'd cache reads and byte-level Arrow IPC concatenation. Storage backends:
local, S3, GCS, Azure.

## Build & Development

```bash
uv sync --all-extras                      # install + build Rust ext (matches CI)
uv run pytest                             # all tests
uv run pytest tests/test_smoke.py -v      # one file
uv run pre-commit run --all-files         # format + lint
uv run ty check src/                      # type check (Astral)
uv run python -m strata                   # start server
```

Full inventory of installed binaries (`strata-notebook`, `strata`,
`strata-worker`, `python -m strata`, package vs CLI name) lives at
`docs/getting-started/installation.md#commands-reference` — the
canonical list other pages should link to instead of re-introducing
each command from scratch.

`--all-extras` is what CI runs (`.github/workflows/ci.yml`). Plain
`uv sync` works for users who only want to run the server, but the
test suite's harness fixtures point the per-notebook venv at the
dev interpreter, so the dev env needs the `[notebook]` extra
(orjson, pyarrow, cloudpickle, …) for cell-execution tests to pass.

## Architecture

### Request flow (unified materialize)

1. Client `POST /v1/materialize` with inputs (table URIs) + transform (e.g. `scan@v1`).
2. Server checks artifact cache by provenance hash → return on hit.
3. `planner.py` resolves Iceberg snapshot → `ReadPlan` with row-group `Task`s.
4. `cache.py` checks disk cache, reads Parquet on miss, writes Arrow IPC.
5. Server streams Arrow IPC via `GET /v1/streams/{stream_id}` while persisting.
6. Artifact finalized for future cache hits.

### Cache key & pruning

Cache key: `hash(tenant | table_identity | snapshot_id | file_path | row_group_id | projection_fingerprint)`.
`TableIdentity` is canonical (`catalog.namespace.table`) to avoid URI-variation duplicates.

Filters apply at two levels: Iceberg manifest stats (skip files) → Parquet
row-group min/max (skip row groups). Pruning is conservative: when safety
cannot be proved, read more data rather than risk dropping rows.

### Deployment modes

`deployment_mode` (env `STRATA_DEPLOYMENT_MODE`, default `service`). Coherence
enforced at startup by `validate_mode_coherence` in `config.py`.

| Flag                    | personal              | service                        |
| ----------------------- | --------------------- | ------------------------------ |
| `writes_enabled`        | always `True`         | always `False`                 |
| Transform build runner  | always on (embedded `duckdb_sql@v1`) | requires `[tool.strata.transforms] enabled` |
| `auth_mode`             | must be `none`        | typically `trusted_proxy`      |
| `multi_tenant_enabled`  | must be `False`       | `True` or `False`              |
| `artifact_dir` default  | `~/.strata/artifacts` | none (must be explicit)        |
| Non-loopback bind       | only with `allow_remote_clients_in_personal=True` | unrestricted |

**Personal + per-user scoping**: setting `STRATA_PERSONAL_MODE_USER_HEADER`
turns on a thin per-user filter for proxy-fronted personal deployments. The
caller's identity (from that header) is stamped as `notebook.toml` `owner` on
create; `discover` and `delete` filter by owner. Unowned (legacy) notebooks
remain global. Not multi-tenancy — for true isolation use service mode.

### Multi-tenancy & auth

Multi-tenant: `X-Tenant-ID` header (validated 1–64 alphanumeric+`_-`); tenant
hashed into cache keys + dirs; per-tenant QoS limiters and metrics; tenant
registry is LRU-bounded. See `tenant.py`, `tenant_registry.py`.

Auth: trusted-proxy model — Strata does not authenticate, only the proxy can
reach it (network-layer enforced). Proxy injects `X-Strata-Principal`,
`X-Strata-Tenant`, `X-Strata-Scopes`, `X-Strata-Proxy-Token`. ACL evaluation
is **deny-first** (deny rules → allow rules → default). Enforcement points:
`POST /v1/materialize` (table/artifact), `GET /v1/streams/{id}` (stream
ownership), `POST /v1/cache/clear` (`admin:cache` scope). See `auth.py`,
`tenant_acl.py`. **Cache stays shared across principals** — ACL gates request
admission and result retrieval, not cache contents.

### Artifact store & transforms

Artifacts deduplicate by provenance hash, chain into DAG pipelines, and carry
human-readable name pointers. SQLite metadata + pluggable blob backends
(`LocalBlobStore`, `S3BlobStore`, `GCSBlobStore`, `AzureBlobStore`). Configure
via `STRATA_ARTIFACT_BLOB_BACKEND` plus backend-specific env vars (see
`blob_store.py`, `config.py`).

Two executor protocols:
- **v1 push**: Strata POSTs multipart (`metadata` + `inputN`) to executor;
  executor returns Arrow IPC.
- **v2 pull** (`pull_model_enabled=True`): Strata sends a `BuildManifest` of
  signed URLs; executor fetches inputs, uploads result, POSTs to `finalize_url`.
  Avoids bandwidth bottleneck at Strata. See `transforms/signed_urls.py`.

### Where to look

HTTP layer: `server.py` (app + lifespan/middleware, plus the
materialize/streams handlers); per-domain routers in `api/routers/`
(`cache`, `debug`, `registry`, `metrics_health`, `admin`, `artifacts`,
`names`, `builds`); typed mode/auth/tenant gates in `api/dependencies.py`;
pure request-shaping logic in `services/` (`artifact`, `registry`, `build`).
Data plane: `types.py`, `planner.py`, `cache.py`, `fetcher.py`, `metadata_*.py`,
`fast_io.py` (+ `rust/src/lib.rs`).
Artifact / build: `artifact_store.py`, `blob_store.py`, `transforms/`.
Auth / tenancy: `auth.py`, `tenant.py`, `tenant_acl.py`, `tenant_registry.py`.
Observability: `tracing.py`, `logging.py`, `health.py`, `circuit_breaker.py`,
`rate_limiter.py`, `cache_metrics.py`, `cache_stats.py`, `pool_metrics.py`.

## Configuration

`StrataConfig` loads from `pyproject.toml` `[tool.strata]` or `STRATA_*` env
vars. See `config.py` for the full surface — host/port, cache dir+size,
S3/GCS/Azure credentials, QoS slots/limits, tracing, logging, timeouts, rate
limiting, multi-tenancy, auth.

## Important invariants

1. **Immutability ⇒ correctness**: Iceberg snapshots and Parquet row groups are
   immutable, so cached results are valid forever for a given key.
2. **Conservative pruning**: when pruning safety is unprovable, read more
   rather than risk dropping rows (see `_should_prune_row_group`).
3. **Bounded streaming**: response memory is O(row group), not O(query result).
4. **Pre-flight 413**: oversized scans rejected before streaming, computed
   from Parquet metadata.
5. **Two-tier QoS**: interactive vs bulk semaphores prevent starvation; in
   multi-tenant mode each tenant gets its own limiter pools.
6. **S3 path normalization**: paths normalized for consistent cache keys
   (`_normalize_s3_path` in `planner.py`).
7. **ACL is deny-first**: explicit denies cannot be bypassed by allows.
8. **Cache is shared across principals**: ACL gates request access, not cache
   contents — preserves the main perf win.

## Testing

Iceberg test warehouses are built via the `temp_warehouse` fixture, which
wires up a `SqlCatalog` + sample table. The canonical fixture table is
`test_db.events` (`id`, `value`, `name`, `timestamp`). Benchmarks live in
`benchmarks/`.

---

## Strata Notebook

Strata Notebook is a content-addressed compute graph over Python with an
interactive UX. Every cell output is a Strata artifact; every cell execution
is `materialize(inputs, transform, environment) → artifact`. The notebook is an
**orchestration layer** over Strata (cascade planning, staleness tracking);
the cell harness is the **executor**.

### File format

```
notebook_dir/
├── notebook.toml          # committed config (id, name, cells, workers, mounts, env, ai)
├── pyproject.toml         # uv config
├── uv.lock
├── cells/{cell_id}.py     # cell source (8-char UUID prefix)
└── .strata/               # gitignored runtime state
    ├── runtime.json       # display outputs, provenance hashes, env metadata
    ├── console/           # per-cell stdout/stderr ({cell_id}.json)
    └── artifacts/         # SQLite + blobs (nb_…@v=N.arrow)
```

`notebook.toml` is **committed config**, `.strata/` is **runtime state**.
Display outputs, console snapshots, per-cell provenance, and `uv sync`
timestamps live in `runtime.json`. `notebook.toml.updated_at` only bumps on
structural edits (add/remove/reorder cell, change worker/timeout/env/mounts/ai).
Runtime writers never touch `notebook.toml`.

Legacy notebooks are auto-migrated on first open
(`runtime_state.migrate_from_legacy_notebook_toml`).

Sensitive env values (`KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL`) are blanked
before persisting; `update_notebook_env` skips the `[env]` block entirely if
all entries are empty/blanked, so typing an API key doesn't churn committed
notebooks.

### Cell execution

Python cells:
1. Provenance: `sha256(sorted_input_hashes + source_hash + env_hash)`
2. `artifact_store.find_by_provenance(hash)` → return on hit
3. Resolve inputs from artifact store → temp dir
4. Spawn subprocess running `harness.py` in notebook venv
5. Harness exec's source, serializes new variables
6. Store each consumed variable as `nb_{notebook_id}_cell_{cell_id}_var_{var}`
7. Broadcast `cell_status` / `cell_output` / `cell_console` over WebSocket

Prompt cells (`prompt_executor.py`): no subprocess; render `{{ var }}` template,
dispatch via `llm.chat_completion` (Anthropic native tool-use when
`@output_schema` is set, otherwise OpenAI-compat), validate against schema with
retry, store response as artifact. Provenance includes a schema fingerprint.

Loop cells (`@loop max_iter=N carry=var`): one harness subprocess per iteration,
`carry` threaded between iterations, `…@iter=k` artifacts per step, final state
becomes the cell's canonical artifact. WebSocket emits `cell_iteration_progress`.

There are three single-cell "run" modes (`executor.py::execute_cell{,_force,_rerun}`):

| Mode    | Target cache | Materialize upstreams | UI                                 |
| ------- | ------------ | --------------------- | ---------------------------------- |
| normal  | on           | on (cascade if stale) | `▶` button, Shift+Enter            |
| force   | off          | **off** (stale ok)    | "Run this only" — no surfaced UI   |
| rerun   | off          | on (cascade if stale) | `↻` button, Cmd+Shift+Enter        |

`notebook_run_all` runs every cell in `execute_cell` (normal) mode;
`notebook_rerun_all` runs every cell in `execute_cell_rerun` mode.
When rerun on a single cell finds stale upstreams, it dispatches through
`_execute_cascade(target_force=True)` so per-step status/output frames still
broadcast — silent in-executor upstream rebuilds would skip those frames.

### Materialize: Core SDK vs notebook

The word "materialize" names *two* distinct pipelines in this codebase. They
deliberately do not share the entry point — only the artifact-store substrate.

| Axis              | Core SDK `client.materialize`          | Notebook `CellExecutor._materialize_cell` |
| ----------------- | -------------------------------------- | ----------------------------------------- |
| Entry point       | HTTP `POST /v1/materialize`            | In-process method call                    |
| Unit of work      | One *transform* (e.g. `scan@v1`)       | One *cell* (ad-hoc Python source)         |
| Provenance key    | `(table_identity, snapshot, columns, filters)` for scan; `transform_spec.to_json() + sorted input hashes` for others — and `transform_spec` itself carries the inputs **in order**, so input order is significant (positional `input0`/`input1` transforms don't dedup under reordering) | `(sorted_inputs, source_hash, env_hash, mount_fingerprints)` |
| Inputs            | List of table / artifact URIs          | Upstream cell variables (resolved via DAG) |
| Outputs           | Single artifact (Arrow IPC stream)     | Multi-output fan-out — one artifact per consumed variable via `derive_subkey` |
| Execution         | Server dispatches to registered HTTP executors (v1-push / v2-pull) or built-in scan planner | Local subprocess harness in notebook venv, or HTTP executor |
| Shared substrate  | `artifact_store.find_by_provenance` / `put` / `set_name`; `notebook.provenance.derive_subkey` (used by both for sub-artifact keys) |

Trying to force one through the other warps either core's transform contract
(to encode source + env + multi-output) or imposes HTTP latency on every
keystroke-sensitive cell run. The split is intentional; the sharing happens
one level down at the artifact store.

### Serialization

Content type is selected by value type and stored in
`transform_spec.params.content_type`:

- `arrow/ipc` — Arrow-representable values (pyarrow Table/RecordBatch, pandas,
  numpy ndarrays + scalars, datetime/Decimal/UUID/bytes/complex). Shape encoded
  in schema metadata `strata.arrow.shape` ∈ `table|tensor|scalar`; reader
  reconstructs the exact Python type.
- `json/object` — dicts, lists, primitive scalars
- `pickle/object` — everything else (cloudpickle by default;
  `STRATA_NOTEBOOK_OBJECT_CODEC` to override; falls back to stdlib pickle)
- `image/png`, `text/markdown` — display-only
- `module/import|cell|cell-instance` — module objects and cell-defined classes

All preview / TOML writes go through `serializer.to_serialization_safe`
(coerces None / datetime / Decimal / numpy scalars to JSON+TOML-safe primitives).

### DAG & variable analysis

Each cell yields `defines` (top-level assignments) and `references` (free
variables anywhere — module scope, decorators, defaults, class bases, type
annotations except under `from __future__ import annotations`, function bodies
via `symtable`). The DAG builder connects references to last-definer producers,
computes a topological order (Kahn), detects cycles, and tracks
`consumed_variables[cell_id]` — the variables this cell produces that
downstream cells reference. Only consumed variables are stored as artifacts.
DAG rebuilds on every source change.

### Source annotations

`#` comments at the top of a cell, parsed by `annotations.py`. **Annotations
always win over persisted config** — they are the single per-cell
configuration surface.

Python cells: `name`, `worker`, `timeout`, `env KEY=VALUE`,
`mount <name> <uri> [ro|rw]`, `loop max_iter= carry= [start_from=]`,
`loop_until <expr>`.

Prompt cells: `name`, `model`, `temperature`, `max_tokens`, `system`,
`output json`, `output_schema {…}` (JSON Schema; OpenAI strict json_schema,
Anthropic tool-use, others json_object fallback), `validate_retries N`.

Mounts inject `pathlib.Path` variables (schemes: `file|s3|gs|az`); options
carry fsspec storage settings. See `mounts.py::MountResolver`.

Validation (`annotation_validation.py`) runs on open / worker-catalog reload /
WS source flush — **never on keystrokes**. Diagnostics surface as a header
pill but never block execution.

### Cascade execution

When a cell's upstream cells aren't ready (idle / stale / error),
`CascadePlanner` BFS-walks backwards, returns cells in topological order with
reasons, and the WebSocket sends `cascade_prompt` → frontend auto-accepts →
sequential execution. Cell status is tracked on `session.notebook_state.cells`
(checked by the planner to avoid spurious cascades).

### API surface

REST `/v1/notebooks`: `POST /create`, `POST /open`, `GET /{id}/cells`,
`PUT|POST|DELETE /{id}/cells[...]`, `POST /{id}/cells/{cell_id}/execute`,
`GET /{id}/dag`. **`{id}` in routes is the session ID, not the
`notebook.toml` id.**

WebSocket `/v1/notebooks/ws/{notebook_id}`:
- C→S: `cell_execute`, `cell_execute_cascade`, `cell_execute_force`,
  `cell_execute_rerun`, `cell_source_update`, `notebook_sync`,
  `notebook_run_all`, `notebook_rerun_all`, `impact_preview_request`,
  `inspect_open|eval|close`
- S→C: `cell_status`, `cell_output`, `cell_output_delta`, `cell_error`,
  `cell_console`, `cell_iteration_progress`, `cascade_prompt`,
  `cascade_progress`, `dag_update`, `impact_preview`, `notebook_state`

Frame-type strings are owned by the `MessageType` StrEnum in
`src/strata/notebook/protocol.py`. The full client-author reference
(bootstrap, auth model, reconnect grace, cold-start payload, every
message type) is `docs/reference/notebook-protocol.md` — keep that doc
in sync when adding routes / frames.

### Frontend (`frontend/`)

Vue 3 + TS + Vite, talks to `http://localhost:8765` (override with
`VITE_STRATA_URL`).

Source updates are **local-only on keystroke** — `updateSource()` updates the
buffer and marks the cell dirty. Dirty cells flush via WS `cell_source_update`
after 2s idle, on editor blur, or immediately before Shift+Enter execution.
The backend re-analyzes async and broadcasts `dag_update` + `cell_status`.
Other mutations (add/remove/reorder) still go via REST.

### Running

```bash
uv run uvicorn strata.server:app --host 0.0.0.0 --port 8765
cd frontend && npm run dev   # hot reload
```

### Notebook invariants

1. **Artifact store is the sole source of truth** for inter-cell variables
   in **single-cell** execution — no in-memory cache, every read goes through
   the store. The run-all batching path (`CellExecutor.execute_batch`,
   issue #26) is the deliberate exception: cells in a batch share a live
   Python namespace within one harness subprocess, and the store is the
   *spill target* for later out-of-order single-cell re-runs. Single-cell
   execution is unchanged.
2. **Cell IDs are backend-generated** (8-char UUID prefix); the frontend
   never invents IDs.
3. **DAG is authoritative on the backend**. Source updates are debounced and
   fire-and-forget; the backend broadcasts authoritative defines / references /
   upstream / downstream / staleness asynchronously. Frontend never blocks on
   a round-trip during typing.
4. **Cell status lives on the session**. After execution
   `session.notebook_state.cells[i].status` becomes `ready` or `error`; the
   cascade planner checks this to decide whether upstream cells need re-running.
5. **Annotations beat persisted config** (no UI editor for per-cell overrides).
6. **`notebook.toml` = committed, `.strata/` = runtime**. Runtime writers
   never touch `notebook.toml`.
7. **Prompt-cell console is a separate `CellState` field**, not folded into
   the output, so `@output_schema` cells keep a clean structured display value.
