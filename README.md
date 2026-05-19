# Strata

[![CI](https://github.com/bearing-research/strata/actions/workflows/ci.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/ci.yml)
[![Pre-commit](https://github.com/bearing-research/strata/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/pre-commit.yml)
[![Docker](https://github.com/bearing-research/strata/actions/workflows/docker.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/docker.yml)

**Strata is a content-addressed computation graph with an interactive notebook UI.**

Every cell output is a versioned artifact keyed by its provenance: source,
inputs, and environment. Strata reads each cell's AST to build the
dependency graph automatically, so re-running a notebook is mostly a series
of cache hits. Touch one cell and the cascade re-executes only the cells
that depend on it. Identical inputs produce the same artifact whether the
second run comes a minute later or a year later, on the same machine or a
different one.

Prompt cells make AI calls first-class DAG nodes, cached by template,
inputs, and model config. `# @worker gpu-fly` dispatches a cell to a remote
GPU. `# @mount data s3://bucket/prefix ro` makes an S3 prefix available as a
local `pathlib.Path` inside the cell. The whole notebook is plain `.py`
files plus a manifest, so commits are git-diffable and there are no JSON
blobs or execution metadata bleeding into the history.

**Docs:** [bearing-research.github.io/strata](https://bearing-research.github.io/strata/)

## Quick Start

Both paths below run in **personal mode**: single-user, writes enabled, no
proxy auth. For multi-tenant or hosted deployments, see
[Deployment Modes](https://bearing-research.github.io/strata/deployment/modes/).

```bash
# Docker (recommended). docker-compose.yml sets personal mode for you.
docker compose up -d --build
# Then open http://localhost:8765

# Or from source — requires uv (see Requirements below).
uv sync
cd frontend && npm ci && npm run build && cd ..
STRATA_DEPLOYMENT_MODE=personal uv run strata-server
# Then open http://localhost:8765
```

### Requirements

Runtime (Docker or `uv run strata-server`):

- [uv](https://docs.astral.sh/uv/) ≥ 0.8 — Strata refuses to start
  outside a uv-managed environment. The startup check looks for the
  `uv = <version>` marker that uv writes to `pyvenv.cfg`; `uv run` and
  `uvx` produce envs with this marker, hand-rolled `python -m venv`
  venvs do not. Conda and pip-venv users need to install uv and
  re-launch Strata from a uv-managed env — existing data and other
  environments are untouched, but Strata's own runtime has to be
  uv-managed.

Source build (only if you're building from this repo, not using
Docker or `uv add strata-notebook`):

- A Rust toolchain (rustup), for `maturin` to compile the native
  extension.
- Node 25+ / npm, for the frontend `npm ci && npm run build` step.
- Python 3.12+ is handled automatically by `uv sync`.

Why uv at runtime: the notebook subsystem shells out to `uv` to
manage per-notebook `.venv/` directories, and the project's dev
workflow assumes uv as the install path. Failing fast at startup with
a clear message beats a confusing subprocess error later.

## Notebook Features

- **Content-addressed caching.** Same code plus same inputs equals an instant cache hit, zero recomputation.
- **Automatic dependency tracking.** DAG built from variable analysis, no manual wiring.
- **Cascade execution.** Change upstream code, downstream cells auto-invalidate.
- **Distributed workers.** Annotate `@worker gpu-fly` and the cell dispatches to a remote GPU.
- **Prompt cells.** LLM-powered cells with `{{ variable }}` template injection.
- **SQL cells.** First-class SQL cells with `# @sql connection=<name>`, named-bind parameters, and DuckDB / Postgres / SQLite drivers.
- **AI assistant.** Streaming chat with conversation memory, agent mode for autonomous notebook building.
- **Environment management.** Per-notebook Python venvs via uv, isolated from each other.
- **Rich outputs.** DataFrames, matplotlib plots, markdown, images.
- **Cell operations.** Reorder, duplicate, fold, keyboard shortcuts.
- **Headless runner.** `strata run ./my-notebook` for CI and scheduled execution.

## The Cache Advantage

Every notebook platform re-executes from scratch when you change one cell.
Strata doesn't. The artifact store deduplicates by provenance hash. If
the code and inputs haven't changed, the result is served instantly.

```
First run:     load data (10s) → clean (3s) → train (20s) → evaluate (1s)  = 34s
Change model:  load data (✓)   → clean (✓)  → train (20s) → evaluate (1s)  = 21s
Re-run:        load data (✓)   → clean (✓)  → train (✓)   → evaluate (✓)   = <1s
```

This isn't a feature bolted on. It's the architecture. Every cell
execution is a `materialize(inputs, transform) → artifact` operation,
and the cache is correct by construction because it's keyed on content,
not time.

## Distributed Execution

Each cell can declare which worker it runs on via a single annotation:

```python
# @worker my-gpu
embeddings = model.encode(abstracts, batch_size=256)
```

You define workers in `notebook.toml`. Each one points at an HTTP
endpoint that implements the Strata executor protocol. A worker can be
a GPU box on RunPod, a DataFusion cluster on Fly, a beefy EC2 instance,
or anything else that speaks HTTP. The notebook routes the cell to the
declared worker at execution time, and the UI shows a live
"dispatching to my-gpu" badge while it runs.

No deployment code, no infrastructure glue. Bring your own compute,
one annotation per cell.

## Source Annotations

Every piece of per-cell metadata is a comment directive in the cell's
source. The source is the single canonical place for cell config:
annotations always win over any stored defaults.

```python
# @name Extract embeddings
# @worker gpu-fly
# @timeout 600
# @env MODEL_PATH=/models/bge-large
# @mount dataset s3://corpus/2024-q4 ro
embeddings = model.encode(dataset / "abstracts.jsonl")
```

Diagnostics fire on open, reload, and after an edit settles:
`worker_unknown`, `mount_uri_unsupported`, `mount_shadows_notebook`,
`timeout_not_numeric`, `env_malformed`. They surface as a pill in the
cell header and log structured warnings for headless runs.

## Mounts

Mounts bind a remote URI to a local path inside the cell. Supported
schemes: `file://`, `s3://`, `gs://`, `az://`. Credentials flow through
fsspec options: set `anon = true` for public buckets, or drop it to
use the standard credential chain.

```toml
[[mounts]]
name = "taxi_zones"
uri = "s3://nyc-tlc/misc"
mode = "ro"
options = { anon = true }
```

Inside the cell, `taxi_zones` is a `pathlib.Path`. Strata materializes
it on first read and caches the bytes locally for the session.

## Examples

| Example                                             | What it shows                                                                       |
| --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [pandas_basics](examples/pandas_basics)             | Linear DataFrame chain, caching, staleness propagation                              |
| [iris_classification](examples/iris_classification) | End-to-end ML, DAG branching, mixed output types                                    |
| [titanic_ml](examples/titanic_ml)                   | Feature engineering + model comparison                                              |
| [s3_mount](examples/s3_mount)                       | Reading a public S3 bucket via a mount                                              |
| [arxiv_classifier](examples/arxiv_classifier)       | Distributed execution via `@worker` + Modal GPU + Fly cluster                       |
| [markdown_showcase](examples/markdown_showcase)     | Markdown cells, dynamic `Markdown(...)` outputs, security cases                     |
| [library_cells](examples/library_cells)             | Cross-cell library code: pure module cells, mixed runtime+library cells, the limits |
| [news_alpha_trader](examples/news_alpha_trader)     | Multi-stage trading pipeline with prompt cells and structured LLM outputs           |

## Known rough edges

Strata is at 0.1 and a few surfaces are explicitly exploratory. The core
(materialization, artifact store, DAG, caching, headless run) is stable
in the alpha sense; these are the bits where the API or coverage is
still moving:

- **Prompt-cell API.** Streaming, conversation memory, and structured-output
  validation are not yet finalized — expect breaking changes in 0.x.
- **SQL cell cloud drivers.** DuckDB / Postgres / SQLite are exercised in
  CI. MotherDuck, MySQL, BigQuery, and Snowflake adapters exist but lack
  integration test coverage; pin a Strata version in production until that
  lands.
- **Wire / on-disk formats.** `notebook.toml`, `runtime.json`, and the
  artifact cache layout may change between minor versions during 0.x.
  Rely on the Python API surface, not the file shapes.

---

## Library usage

Strata's HTTP API exposes the materialization layer directly,
driveable from Python via `StrataClient`. Useful for direct table
scans, custom transforms, and headless workflows; the notebook
executor is a separate pipeline that writes to the same artifact
store. The client talks to a running Strata server, so this workflow
has two steps: start the server, then call it from your code.

```bash
# 1. Install + start the server (in a uv-managed env).
uv add "strata-notebook[notebook]"
uv run strata-server

# 2. From another process, point the client at it:
```

```python
from strata import StrataClient

client = StrataClient(base_url="http://localhost:8765")
artifact = client.materialize(
    inputs=["file:///warehouse#db.events"],
    transform={"executor": "scan@v1", "params": {"columns": ["id", "value"]}},
)
table = client.fetch(artifact.uri)  # Arrow table, cached by provenance
```

The server provides: provenance-based deduplication, immutable
versioned artifacts, lineage tracking, Iceberg table scanning with
row-group caching, pluggable blob storage (local/S3/GCS/Azure),
multi-tenancy, trusted-proxy auth, and an executor protocol for
external compute.

**[Library docs →](https://bearing-research.github.io/strata/getting-started/core/)**

---

## Architecture

```
┌─────────────────────────────────────────────┐
│ Notebook UI (Vue.js + WebSocket)            │
│ cells, DAG view, AI assistant, workers      │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ Notebook Backend (FastAPI)                  │
│ session, cascade, executor, prompt cells    │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ Strata Core                                 │
│ materialize, artifacts, lineage, dedupe     │
└─────────────────────────────────────────────┘
```

The notebook is an orchestration layer over Core. It decides what to
run next (cascade planning, staleness tracking). The cell harness is an
executor. Core decides whether results already exist and persists them.

## Development

```bash
uv sync                                # Install deps + build Rust extension
uv run pytest                          # Run all tests
uv run pre-commit run --all-files      # Lint + format
cd frontend && npm run dev             # Frontend dev server (hot reload)
```

## License

Apache 2.0
