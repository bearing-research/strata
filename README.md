# Strata

[![PyPI](https://img.shields.io/pypi/v/strata-notebook.svg)](https://pypi.org/project/strata-notebook/)
[![Python versions](https://img.shields.io/pypi/pyversions/strata-notebook.svg)](https://pypi.org/project/strata-notebook/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bearing-research/strata/blob/main/LICENSE)
[![CI](https://github.com/bearing-research/strata/actions/workflows/ci.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/ci.yml)
[![Pre-commit](https://github.com/bearing-research/strata/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/pre-commit.yml)
[![Docker](https://github.com/bearing-research/strata/actions/workflows/docker.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/docker.yml)
[![Docs](https://github.com/bearing-research/strata/actions/workflows/docs.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/bearing-research/strata/branch/main/graph/badge.svg?token=GBAX34U2PO)](https://codecov.io/gh/bearing-research/strata)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/bearing-research/strata/badge)](https://securityscorecards.dev/viewer/?uri=github.com/bearing-research/strata)

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

## Highlights

- **content-addressed:** every cell output is keyed by source + inputs + environment — identical work hits the cache forever
- **reactive:** edit a cell, the cascade re-runs only the downstream cells that depend on it
- **dag-from-ast:** Strata reads each cell's AST to wire upstream/downstream — no decorators, no manual edges
- **dag-view:** the dependency graph renders alongside the cells — double-click any node to jump to its source
- **ambient client (0.3.0):** every cell gets a ready `strata` client in its namespace — publish and consume artifacts across cells with no boilerplate
- **registry in the UI (0.3.0):** promote and approve named artifacts from a notebook dashboard — pending-approval queue, alias chips, and `model ← features ← scan ← table` lineage
- **git-friendly:** notebooks are plain `.py` files plus a TOML manifest — readable diffs, no JSON blobs
- **prompt cells:** LLM calls are first-class DAG nodes, `{{ variable }}` interpolation from upstream cells, cached by template + inputs + model config
- **SQL cells:** named connections, bind-parameter templating, drivers for DuckDB / SQLite / Postgres / Snowflake / BigQuery
- **R cells (0.2.0):** Python and R cells share a DAG; cross-language Arrow exchange means a `pandas.DataFrame` is a `data.frame` for the next cell. First-class in the UI — Add-R-cell menu, an R environment panel with one-click renv bootstrap + package install, automatic `renv::restore()` on open, and inline plots (ggplot2 / base graphics render to PNG). Runs headlessly too — `strata run` executes R cells for CI
- **loop cells:** `# @loop max_iter=N carry=state` iterates a cell with explicit carry between steps — each iteration is its own artifact
- **distributed:** `# @worker gpu-fly` dispatches a single cell to a remote box — bring your own compute
- **mounts:** `# @mount data s3://bucket/prefix ro` makes any S3 / GCS / Azure prefix a local `pathlib.Path`
- **isolated envs:** every notebook gets its own uv-managed `.venv/`, locked and reproducible
- **auto-install:** missing import in a cell? one click adds the package via uv and re-runs
- **headless:** `strata run ./my-notebook` for CI and scheduled execution — same DAG, same cache
- **also a library:** the materialization layer is exposed via HTTP + a `StrataClient`, usable from any Python process
- **slim client package (0.3.0):** `pip install strata-client` pulls just httpx + pyarrow — use the store from any pipeline or service, no server install
- **production-ready:** Iceberg-aware scans, trusted-proxy auth, multi-tenancy, S3 / GCS / Azure / local blob backends

## Quick Start

Both paths below run in **personal mode**: single-user, writes enabled, no
proxy auth. For multi-tenant or hosted deployments, see
[Deployment Modes](https://bearing-research.github.io/strata/deployment/modes/).

```bash
# Docker. docker-compose.yml sets personal mode for you.
docker compose up -d --build
# Then open http://localhost:8765

# Or install via uv (recommended). Fetches the wheel from PyPI into a
# uv-managed tool env at ~/.local/share/uv/tools/strata-notebook with
# the CLI on PATH. Plain `pip install` is not supported — Strata refuses
# to start outside a uv-managed env (see Requirements below).
uv tool install strata-notebook
strata-notebook
# Then open http://localhost:8765
```

For the full inventory of installed commands (`strata-notebook`, `strata`,
`strata-worker`, `python -m strata`), see the
[Commands reference](https://bearing-research.github.io/strata/getting-started/installation/#commands-reference).

Source builds — `git clone + uv sync` — work too and are documented in
[Installation](https://bearing-research.github.io/strata/getting-started/installation/);
needed only if you're modifying Strata itself.

### Requirements

- **[uv](https://docs.astral.sh/uv/) ≥ 0.8** — install via the
  [uv installer](https://docs.astral.sh/uv/getting-started/installation/)
  (`curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux;
  PowerShell installer on Windows). Strata refuses to start outside
  a uv-managed environment: the startup check looks for the
  `uv = <version>` marker that uv writes to `pyvenv.cfg`. `uv tool
  install`, `uv add`, and `uv run` all produce envs with this
  marker; plain `pip install` into a hand-rolled `python -m venv`
  does not, and Strata will refuse to start there. Conda and
  pip-venv users need to install uv and re-launch from a uv-managed
  env — existing data and other environments are untouched. uv
  fetches a matching Python for you, so you don't need Python
  pre-installed.

Source build (only if you're building Strata itself from a git clone,
not using PyPI or Docker):

- **[Rust toolchain](https://rustup.rs/)** (rustup) — for `maturin`
  to compile the native extension. PyPI wheels skip this step.
- **[Node 24+ / npm](https://nodejs.org/)** — for the frontend
  `npm ci && npm run build` step. PyPI wheels bundle the prebuilt SPA.
- Python 3.12+ is handled automatically by `uv sync`.

Windows: `uv tool install strata-notebook` works directly. Source builds
work via WSL2 (smoother) or native Windows (uv + rustup + Node have
Windows installers).

Why uv at runtime: the notebook subsystem shells out to `uv` to
manage per-notebook `.venv/` directories, and the project's dev
workflow assumes uv as the install path. Failing fast at startup with
a clear message beats a confusing subprocess error later.

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
execution is a `materialize(inputs, transform, environment) → artifact` operation,
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
| [pandas_basics](https://bearing-research.github.io/strata/examples/pandas_basics/)             | Linear DataFrame chain, caching, staleness propagation                              |
| [iris_classification](https://bearing-research.github.io/strata/examples/iris_classification/) | End-to-end ML, DAG branching, mixed output types                                    |
| [titanic_ml](https://bearing-research.github.io/strata/examples/titanic_ml/)                   | Feature engineering + model comparison                                              |
| [s3_mount](https://bearing-research.github.io/strata/examples/s3_mount/)                       | Reading a public S3 bucket via a mount                                              |
| [arxiv_classifier](https://bearing-research.github.io/strata/examples/arxiv_classifier/)       | Distributed execution via `@worker` + Modal GPU + Fly cluster                       |
| [markdown_showcase](https://bearing-research.github.io/strata/examples/markdown_showcase/)     | Markdown cells, dynamic `Markdown(...)` outputs, security cases                     |
| [library_cells](https://bearing-research.github.io/strata/examples/library_cells/)             | Cross-cell library code: pure module cells, mixed runtime+library cells, the limits |
| [news_alpha_trader](https://bearing-research.github.io/strata/examples/news_alpha_trader/)     | Multi-stage trading pipeline with prompt cells and structured LLM outputs           |

## Known rough edges

Strata is young and a few surfaces are explicitly exploratory. The core
(materialization, artifact store, DAG, caching, headless run) is stable
in the alpha sense; these are the bits where the API or coverage is
still moving:

- **Prompt-cell API.** Streaming, conversation memory, and structured-output
  validation are not yet finalized — expect breaking changes in 0.x.
- **SQL cell cloud drivers.** DuckDB / SQLite / PostgreSQL are exercised
  in CI. BigQuery and Snowflake adapters ship but lack integration test
  coverage; pin a Strata version in production until that lands.
  MotherDuck and MySQL are planned but not yet implemented.
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
uv tool install strata-notebook
strata-notebook

# 2. In your own project, install the slim client — a separate package
#    (httpx + pyarrow only, no server deps, plain pip is fine) — and
#    point it at the running server:
pip install strata-client
```

```python
from strata_client import StrataClient

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
