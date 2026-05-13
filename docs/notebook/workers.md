# Distributed Workers

Strata Notebook can dispatch individual cells to remote machines via the **executor protocol**. A worker is any HTTP endpoint that accepts cell source code and inputs, runs them, and returns the outputs. You bring the compute, Strata handles the routing, serialization, and caching.

## How It Works

```
┌─────────────────────┐    multipart POST     ┌──────────────────────┐
│  Strata Notebook     │ ──────────────────►  │  Worker (HTTP)        │
│  (orchestrator)      │                       │  remote_executor.py   │
│                      │  ◄──────────────────  │                       │
│  routes cell to      │    gzipped bundle     │  runs harness.py      │
│  @worker annotation  │    (outputs + blobs)  │  returns results      │
└─────────────────────┘                       └──────────────────────┘
```

1. You annotate a cell with `# @worker my-gpu`
2. Strata looks up `my-gpu` in the notebook's `[[workers]]` config
3. The cell source + serialized input variables are sent as a multipart HTTP POST
4. The worker runs the cell in a subprocess and returns outputs as a gzipped bundle
5. Strata stores the outputs as artifacts, cache hits work identically to local cells

## Registering Workers

Add workers from the **Workers panel** in the sidebar. Give each one a name (used in `@worker` annotations), the executor endpoint URL, and, optionally, a `runtime_id` that pins provenance so cache hits are stable across worker upgrades.

The result lands in `notebook.toml` as `[[workers]]` entries:

```toml
[[workers]]
name = "my-gpu"
backend = "executor"
runtime_id = "my-gpu-a10g"

[workers.config]
url = "https://my-worker.example.com/v1/execute"
transport = "http"
```

| Field | Description |
|-------|------------|
| `name` | The name used in `@worker` annotations |
| `backend` | Always `"executor"` for HTTP workers |
| `runtime_id` | Optional stable identifier for provenance (changing this invalidates cache) |
| `config.url` | The HTTP endpoint for the executor protocol |
| `config.transport` | `"http"` for direct push, `"signed"` for pull-model with signed URLs |

Register as many workers as you need, each cell picks its target independently. An example notebook with two workers ends up looking like:

```toml
[[workers]]
name = "df-cluster"
backend = "executor"
runtime_id = "df-cluster"
[workers.config]
url = "https://my-datafusion.fly.dev/v1/execute"
transport = "http"

[[workers]]
name = "gpu"
backend = "executor"
runtime_id = "gpu-a10g"
[workers.config]
url = "https://my-gpu-worker.modal.run/v1/execute"
transport = "http"
```

## Running a Worker

Strata ships a reference executor as the `strata-worker` console script:

```bash
strata-worker --host 0.0.0.0 --port 9000
```

This starts a FastAPI server that:

- Accepts multipart cell execution requests at `/v1/execute`
- Runs cells via the same `harness.py` subprocess as local execution
- Returns outputs as a gzipped bundle
- Exposes `/health` for monitoring

The worker runs cells using **its own Python environment**: whatever packages are installed in the worker's interpreter are available to cells. This is how you provide GPU libraries (torch, sentence-transformers) or data engines (datafusion) without installing them locally. Install your workload dependencies before launching `strata-worker`.

### Deploying to Fly.io (CPU worker)

```dockerfile
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git ca-certificates && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install --no-cache-dir \
    "strata-notebook @ git+https://github.com/bearing-research/strata.git@main" \
    "datafusion>=42" \
    "pandas>=2" \
    "pyarrow>=18"

FROM python:3.12-slim
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

EXPOSE 8080
CMD ["strata-worker", "--host", "0.0.0.0", "--port", "8080"]
```

Deploy with `fly deploy`. Register the Fly URL as a worker via the Workers panel.

### Deploying to Modal (GPU worker)

```python
import modal

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "git", "ca-certificates")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install(
        "maturin>=1.5",
        "fastapi>=0.115", "uvicorn>=0.30", "httpx>=0.27",
        "python-multipart>=0.0.9",
        "pyarrow>=18.0.0", "pandas>=2.0.0", "numpy>=1.26.0",
        "torch>=2.3", "sentence-transformers>=3.0", "scikit-learn>=1.5",
        "strata-notebook @ git+https://github.com/bearing-research/strata.git@main",
    )
)

app = modal.App("my-gpu-worker", image=gpu_image)

@app.function(gpu="A10G", scaledown_window=60)
@modal.asgi_app()
def gpu_executor():
    from strata.notebook.remote_executor import create_notebook_executor_app
    return create_notebook_executor_app()
```

Deploy with `modal deploy worker.py`. The printed URL becomes the worker's `config.url`.

## Using Workers in Cells

Annotate any cell with `# @worker <name>`:

```python
# @name Embed Abstracts
# @worker gpu
# @timeout 300
embeddings = model.encode(abstracts, batch_size=256)
```

The worker annotation is the **only** change needed, the cell code itself is identical to what you'd write for local execution. If the worker has the right packages installed, it just works.

### Precedence

If multiple levels define a worker, the most specific wins:

1. `# @worker X` annotation in the cell source (highest)
2. Cell-level worker override (from the cell's config)
3. Notebook-level worker default (from the Workers panel)

### Caching

Remote execution results are cached identically to local cells. The provenance hash includes the worker's `runtime_id`, so:

- Same code + same inputs + same worker = cache hit (instant, no remote call)
- Changing the worker (e.g., switching from `gpu-a10g` to `gpu-h100`) invalidates the cache for that cell

### Local Development

For local testing without cloud deployment, run multiple workers on different ports:

```bash
# Terminal 1: DataFusion worker
uv run --with datafusion strata-worker --port 9000

# Terminal 2: GPU worker (with ML packages)
uv run --with torch --with sentence-transformers strata-worker --port 9001
```

Register both in the Workers panel pointing at `http://127.0.0.1:9000/v1/execute` and `http://127.0.0.1:9001/v1/execute` during development. Swap to cloud URLs when deploying, cells using `@worker gpu` move over without any code change.

## Health Checks

Every worker exposes `GET /health`:

```bash
curl https://my-worker.example.com/health
```

```json
{
  "status": "healthy",
  "capabilities": {
    "protocol_versions": ["v1"],
    "transform_refs": ["notebook_cell@v1"],
    "features": {
      "notebook_protocol_version": "notebook-cell-v1",
      "output_format": "notebook-output-bundle@v1"
    }
  },
  "uptime_seconds": 42.5,
  "active_executions": 0
}
```

The notebook UI shows worker health status as a badge next to cells that use that worker.

## Live Status

When a cell dispatches to a remote worker, the UI shows a pulsing **"dispatching → my-gpu"** badge during execution. After completion, the worker name and transport type appear in the cell metadata.
