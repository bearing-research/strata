# Strata

[![CI](https://github.com/bearing-research/strata/actions/workflows/ci.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/ci.yml)
[![Pre-commit](https://github.com/bearing-research/strata/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/pre-commit.yml)
[![Docker](https://github.com/bearing-research/strata/actions/workflows/docker.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/docker.yml)

**Strata is a content-addressed computation graph with an interactive notebook UI.**

Built for ML, AI, and data work that runs longer than a keystroke. Every
cell output becomes a content-addressed artifact, so re-runs are cache hits
when nothing's changed and the DAG cascade re-executes only what did.
Provenance is automatic — Strata reads each cell's AST, links references
back to their producers, and invalidates downstream cells when an upstream
source changes.

Prompt cells make AI calls first-class DAG nodes — cached by template,
inputs, and model config. `# @worker gpu-fly` dispatches a cell to a remote
GPU. `# @mount data s3://bucket/prefix ro` makes an S3 prefix available as a
local `pathlib.Path` inside the cell. The whole notebook is plain `.py`
files plus a manifest — git-diffable, no JSON blobs, no execution metadata
bleeding into commits.

**Docs:** [bearing-research.github.io/strata](https://bearing-research.github.io/strata/)

## Quick Start

Both paths below run in **personal mode** — single-user, writes enabled, no
proxy auth. For multi-tenant or hosted deployments, see
[Deployment Modes](https://bearing-research.github.io/strata/deployment/modes/).

```bash
# Docker (recommended) — docker-compose.yml sets personal mode for you
docker compose up -d --build
# Then open http://localhost:8765

# Or from source — set personal mode explicitly
uv sync
cd frontend && npm ci && npm run build && cd ..
STRATA_DEPLOYMENT_MODE=personal uv run strata-server
# Then open http://localhost:8765
```

### Install as a dependency

```bash
# Strata core (materialization, artifact store, Iceberg scanning):
pip install strata-notebook

# Add the notebook display + serialization stack
# (DataFrame/Series/ndarray, cloudpickle, matplotlib):
pip install "strata-notebook[notebook]"

# Or with uv:
uv add "strata-notebook[notebook]"

# Install from Git (e.g., to track main between releases):
pip install "strata-notebook @ git+https://github.com/bearing-research/strata.git"
pip install "strata-notebook @ git+https://github.com/bearing-research/strata.git@<sha>"
```

The Python module is still imported as `strata`:

```python
from strata.client import StrataClient
```

## Notebook Features

- **Content-addressed caching** — same code + same inputs = instant cache hit, zero recomputation
- **Automatic dependency tracking** — DAG built from variable analysis, no manual wiring
- **Cascade execution** — change upstream code, downstream cells auto-invalidate
- **Distributed workers** — annotate `@worker gpu-fly` and the cell dispatches to a remote GPU
- **Prompt cells** — LLM-powered cells with `{{ variable }}` template injection
- **AI assistant** — streaming chat with conversation memory, agent mode for autonomous notebook building
- **Environment management** — per-notebook Python venvs via uv, isolated from each other
- **Rich outputs** — DataFrames, matplotlib plots, markdown, images
- **Cell operations** — reorder, duplicate, fold, keyboard shortcuts
- **Headless runner** — `strata run ./my-notebook` for CI and scheduled execution

## The Cache Advantage

Every notebook platform re-executes from scratch when you change one cell.
Strata doesn't. The artifact store deduplicates by provenance hash —
if the code and inputs haven't changed, the result is served instantly.

```
First run:     load data (10s) → clean (3s) → train (20s) → evaluate (1s)  = 34s
Change model:  load data (✓)   → clean (✓)  → train (20s) → evaluate (1s)  = 21s
Re-run:        load data (✓)   → clean (✓)  → train (✓)   → evaluate (✓)   = <1s
```

This is not a feature bolted on — it's the architecture. Every cell
execution is a `materialize(inputs, transform) → artifact` operation.
The cache is correct by construction because it's keyed on content, not
time.

## Distributed Execution

Each cell can declare which worker it runs on via a single annotation:

```python
# @worker my-gpu
embeddings = model.encode(abstracts, batch_size=256)
```

You define workers in `notebook.toml` — each one points at an HTTP
endpoint that implements the Strata executor protocol. A worker can be
a GPU box on RunPod, a DataFusion cluster on Fly, a beefy EC2 instance,
or anything else that speaks HTTP. The notebook routes the cell to the
declared worker at execution time, and the UI shows a live
"dispatching → my-gpu" badge while it runs.

No deployment code, no infrastructure glue. Bring your own compute,
one annotation per cell.

## Source Annotations

Every piece of per-cell metadata is a comment directive in the cell's
source. The source is the single canonical place for cell config —
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
fsspec options — set `anon = true` for public buckets, or drop it to
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
| [pandas_basics](examples/pandas_basics)             | Linear DataFrame chain — caching, staleness propagation                             |
| [iris_classification](examples/iris_classification) | End-to-end ML, DAG branching, mixed output types                                    |
| [titanic_ml](examples/titanic_ml)                   | Feature engineering + model comparison                                              |
| [s3_mount](examples/s3_mount)                       | Reading a public S3 bucket via a mount                                              |
| [arxiv_classifier](examples/arxiv_classifier)       | Distributed execution via `@worker` + Modal GPU + Fly cluster                       |
| [markdown_showcase](examples/markdown_showcase)     | Markdown cells, dynamic `Markdown(...)` outputs, security cases                     |
| [library_cells](examples/library_cells)             | Cross-cell library code: pure module cells, mixed runtime+library cells, the limits |
| [news_alpha_trader](examples/news_alpha_trader)     | Multi-stage trading pipeline with prompt cells and structured LLM outputs           |

---

## Strata Core

The notebook is built on **Strata Core**, a standalone materialization
and artifact layer that can be used independently as a Python library
and REST API:

```python
from strata import StrataClient

client = StrataClient()
artifact = client.materialize(
    inputs=["file:///warehouse#db.events"],
    transform={"executor": "scan@v1", "params": {"columns": ["id", "value"]}},
)
table = client.fetch(artifact.uri)  # Arrow table, cached by provenance
```

Core provides: provenance-based deduplication, immutable versioned
artifacts, lineage tracking, Iceberg table scanning with row-group
caching, pluggable blob storage (local/S3/GCS/Azure), multi-tenancy,
trusted proxy auth, and an executor protocol for external compute.

**[Core documentation →](https://bearing-research.github.io/strata/getting-started/core/)**

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
uv sync                          # Install deps + build Rust extension
uv run pytest                    # Run all tests
pre-commit run --all-files       # Lint + format
cd frontend && npm run dev       # Frontend dev server (hot reload)
```

## License

Apache 2.0
