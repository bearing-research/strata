# Distributed Workers

Strata Notebook can dispatch individual cells to remote machines via the **executor protocol**. A worker is any HTTP endpoint that accepts cell source code and inputs, runs them, and returns the outputs. You bring the compute; Strata handles the routing, serialization, and caching.

## How it works

```
┌─────────────────────┐    multipart POST     ┌──────────────────────┐
│  Strata Notebook    │ ──────────────────►  │  Worker (HTTP)        │
│  (orchestrator)     │                       │  remote_executor.py   │
│                     │  ◄──────────────────  │                       │
│  routes cell to     │    gzipped bundle     │  runs harness.py      │
│  @worker annotation │    (outputs + blobs)  │  returns results      │
└─────────────────────┘                       └──────────────────────┘
```

1. You annotate a cell with `# @worker my-gpu`.
2. Strata looks up `my-gpu` in the notebook's `[[workers]]` config.
3. Cell source + serialized input variables are sent as a multipart `POST /v1/execute`.
4. The worker runs the cell in a subprocess and returns outputs as a gzipped bundle.
5. Strata stores the outputs as artifacts; cache hits work identically to local cells.

Cells run in **the worker's Python environment**, so install your workload dependencies (torch, datafusion, sentence-transformers, etc.) into the worker image before launching it. Unlike Strata's own server, the worker process does **not** require a uv-managed env — it can be pip-installed into a plain Docker image.

For the wire-level contract — request envelopes, response bundle format, error codes, the pull-model with signed URLs — see the [Executor Protocol](../reference/executor-protocol.md) reference. This page covers deployment and registration; that one covers the bytes on the wire and is what you'd implement against to write a custom worker that doesn't use `strata-worker`.

## Quick start: run a worker locally

Start by getting a worker running on your own machine. This verifies your install before you spend time on a cloud deploy, and the same `# @worker name` annotation works against both local and cloud workers.

**1. Start the worker:**

```bash
uv run strata-worker --port 9000
```

You should see uvicorn start up:

```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:9000
```

**2. Verify it's healthy:**

```bash
curl http://localhost:9000/health
```

Expected response:

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
  "uptime_seconds": 5.2,
  "active_executions": 0
}
```

**3. Register it in your notebook.** Either through the **Workers panel** in the sidebar, or by editing `notebook.toml`:

```toml
[[workers]]
name = "local"
backend = "executor"
runtime_id = "local-dev"

[workers.config]
url = "http://127.0.0.1:9000/v1/execute"
transport = "http"
```

**4. Use it in a cell:**

```python
# @worker local
import platform
hostname = platform.node()
```

When the cell runs, the UI shows a pulsing **"dispatching → local"** badge during execution. The `hostname` artifact is what the worker process saw, not your laptop — confirming the cell really ran remotely.

Once this works locally, the cloud deploys below just change `config.url` from `http://127.0.0.1:9000` to the worker's public URL.

## Deploy to the cloud

Strata ships a reference executor as the `strata-worker` console script. Any platform that can run an HTTP service on a Python image will work. Two walkthroughs follow:

| Platform | Best for | Cost model |
| -------- | -------- | ---------- |
| **Fly.io** | CPU workloads (DataFusion, pandas-heavy pipelines) that need always-on or fast cold starts | Per-second VM billing; can scale to zero |
| **Modal** | GPU workloads (torch, embeddings, fine-tuning) that benefit from scale-to-zero | Per-second VM billing; cold-start ~10–30 s for GPU |

You can register many workers per notebook; each cell picks its target independently. Mixing Fly (cheap CPU) and Modal (on-demand GPU) is a common setup.

### Fly.io (CPU worker)

**Prerequisites:**

```bash
# Install the Fly CLI (macOS; see https://fly.io/docs/flyctl/install/ for others)
brew install flyctl

# Log in (opens a browser)
fly auth login
```

**1. Create a project directory with three files:**

```
my-strata-worker/
├── Dockerfile
├── fly.toml
└── .dockerignore
```

**`Dockerfile`** — installs `strata-notebook` and your workload deps from a uv-managed venv. Unlike `strata-server`, the worker entry (`strata-worker`) is not gated by Strata's runtime guard, so a plain `pip install` would also work — but the uv-python base image keeps tooling consistent across server + worker and drops a few hundred MB of build stage versus a source install.

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV VIRTUAL_ENV=/opt/strata-venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install strata-notebook + your workload deps into a uv-managed
# venv. Pin to an exact strata-notebook version in production so
# workers can't drift relative to the notebook server's expected
# protocol version.
RUN uv venv $VIRTUAL_ENV && \
    uv pip install \
      strata-notebook \
      "datafusion>=42" \
      "pandas>=2" \
      "pyarrow>=18"

EXPOSE 8080
CMD ["strata-worker", "--host", "0.0.0.0", "--port", "8080"]
```

**`fly.toml`**:

```toml
app = "my-strata-worker"
primary_region = "iad"  # pick a region close to your Strata server

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0  # set to 1 for always-on; 0 for scale-to-zero

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory = "1gb"  # bump for pandas/duckdb workloads
```

**`.dockerignore`** (keeps the build context small):

```
.git
*.pyc
__pycache__
.venv
```

**2. Deploy:**

```bash
fly launch --no-deploy   # first time only — creates the app, accepts fly.toml
fly deploy
```

The first build takes ~30 seconds (wheel download + layer assembly). Subsequent deploys reuse the layer cache and finish in seconds.

**3. Verify the deployed worker:**

```bash
curl https://my-strata-worker.fly.dev/health
```

Expect the same JSON as the local-worker step. If you see a 404 or timeout, jump to [Troubleshooting](#troubleshooting).

**4. Register in `notebook.toml`:**

```toml
[[workers]]
name = "fly-cpu"
backend = "executor"
runtime_id = "fly-cpu-v1"

[workers.config]
url = "https://my-strata-worker.fly.dev/v1/execute"
transport = "http"
```

### Modal (GPU worker)

**Prerequisites:**

```bash
pip install modal
modal token new   # one-time browser auth
```

**1. Create `worker.py`:**

```python
import modal

# Modal's pip_install pulls wheels from PyPI. strata-notebook ships
# pre-built abi3-py312 wheels so no Rust toolchain is needed, and the
# worker entry isn't gated by the runtime guard (only strata-server is)
# so Modal's standard image stack works without going through uv.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow>=18.0.0", "pandas>=2.0.0", "numpy>=1.26.0",
        # Your workload dependencies:
        "torch>=2.3",
        "sentence-transformers>=3.0",
        # Pin to an exact version in production so the worker
        # protocol can't drift relative to the notebook server.
        "strata-notebook",
    )
)

app = modal.App("my-gpu-worker", image=gpu_image)


@app.function(gpu="A10G", scaledown_window=60)
@modal.asgi_app()
def gpu_executor():
    from strata.notebook.remote_executor import create_notebook_executor_app
    return create_notebook_executor_app()
```

**2. Deploy:**

```bash
modal deploy worker.py
```

Modal prints the deployed URL after the build finishes — something like `https://your-username--my-gpu-worker-gpu-executor.modal.run`. The first build with torch + sentence-transformers takes ~5 minutes; redeploys with no changes hit the layer cache and finish in seconds.

**3. Verify:**

```bash
curl https://your-username--my-gpu-worker-gpu-executor.modal.run/health
```

The first request after a scale-down cold-starts the container (~20–30 s for GPU). Health-check requests do **not** start the GPU function on most Modal plans — if `/health` returns immediately, the function is warm; if it doesn't respond, send a real cell from the notebook to wake it.

**4. Register in `notebook.toml`:**

```toml
[[workers]]
name = "modal-gpu"
backend = "executor"
runtime_id = "modal-a10g-v1"

[workers.config]
url = "https://your-username--my-gpu-worker-gpu-executor.modal.run/v1/execute"
transport = "http"
```

## Registering workers

Workers live in `notebook.toml` under `[[workers]]`. You can add them through the **Workers panel** sidebar (which writes the same TOML) or edit the file directly:

| Field | Description |
| --- | --- |
| `name` | Used in `@worker <name>` annotations and the dropdown UI |
| `backend` | Always `"executor"` for HTTP workers |
| `runtime_id` | Stable identifier hashed into cell provenance — see [Caching](#caching-and-provenance) below |
| `config.url` | The HTTP endpoint for the executor protocol |
| `config.transport` | `"http"` for direct push, `"signed"` for pull-model with signed URLs |
| `config.token` | Literal bearer token (dev only) — see [Authentication](#authentication) |
| `config.token_env` | Env var name holding the bearer token (preferred for prod) |

A typical multi-worker notebook ends up with:

```toml
[[workers]]
name = "fly-cpu"
backend = "executor"
runtime_id = "fly-cpu-v1"
[workers.config]
url = "https://my-strata-worker.fly.dev/v1/execute"
transport = "http"
token_env = "STRATA_FLY_WORKER_TOKEN"

[[workers]]
name = "modal-gpu"
backend = "executor"
runtime_id = "modal-a10g-v1"
[workers.config]
url = "https://...--my-gpu-worker-gpu-executor.modal.run/v1/execute"
transport = "http"
token_env = "STRATA_MODAL_WORKER_TOKEN"
```

## Authentication

By default the worker accepts any caller that can reach its URL. For any worker deployed to a public endpoint, set a bearer token so only your notebook server can dispatch cells.

**1. Generate a token** (any opaque string; 32+ random bytes is plenty):

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**2. Set `STRATA_WORKER_TOKEN` on the worker.**

For Fly.io, store it as a secret (encrypted, injected at runtime, not visible in fly.toml):

```bash
fly secrets set STRATA_WORKER_TOKEN=<paste-token-here>
```

For Modal, attach a secret to the function:

```python
@app.function(
    gpu="A10G",
    scaledown_window=60,
    secrets=[modal.Secret.from_name("strata-worker-token")],
)
@modal.asgi_app()
def gpu_executor():
    ...
```

…then create the Modal secret once: `modal secret create strata-worker-token STRATA_WORKER_TOKEN=<paste-token-here>`.

**3. Tell the notebook server about the token.** Export it as an environment variable wherever you run `strata-server`:

```bash
export STRATA_FLY_WORKER_TOKEN=<paste-token-here>
uv run strata-server
```

…and reference that env var in `notebook.toml`:

```toml
[workers.config]
url = "https://my-strata-worker.fly.dev/v1/execute"
transport = "http"
token_env = "STRATA_FLY_WORKER_TOKEN"
```

`token_env` is preferred over `token` because the literal-token form gets committed to your notebook repo. Use `token = "..."` only for one-off local experiments.

A worker with `STRATA_WORKER_TOKEN` set rejects unauthenticated requests with `401 Unauthorized`. `/health` stays open so platform health probes work without the secret.

## Using workers in cells

Annotate any cell with `# @worker <name>`:

```python
# @name Embed Abstracts
# @worker modal-gpu
# @timeout 300
embeddings = model.encode(abstracts, batch_size=256)
```

The worker annotation is the **only** change needed; the cell code itself is identical to local execution. If the worker has the right packages installed, it just works.

### Precedence

When multiple levels define a worker, the most specific wins:

1. `# @worker X` annotation in the cell source (highest)
2. Cell-level worker override (from the cell's stored config)
3. Notebook-level worker default (from the Workers panel)

## Caching and provenance

Remote execution results are cached identically to local cells. The provenance hash includes the worker's `runtime_id`, so:

- Same code + same inputs + same `runtime_id` = cache hit, no remote call.
- Changing `runtime_id` (e.g., switching from `gpu-a10g` to `gpu-h100`) invalidates the cache for cells using that worker.

**When to bump `runtime_id`:**

- You upgraded the worker's Python dependencies (new torch version, new model weights baked into the image) and want downstream cells to re-run.
- You moved a worker to different hardware (CPU type, GPU SKU) and the numerical output may differ.
- You explicitly want to bust the cache for a debugging session.

**When to leave `runtime_id` alone:**

- Redeploying the same image (no dep changes). The cache is correct by construction; re-running is wasted compute.
- Scaling the number of worker instances. Output is deterministic given the same inputs.

If you don't set `runtime_id`, Strata uses the worker `name` as a fallback. That's fine for solo notebooks but risks cache surprises if two notebooks both have a worker named `gpu` pointing at different deployments — set `runtime_id` explicitly in shared notebooks.

## Health checks

Every worker exposes `GET /health`. The notebook UI polls this and shows a green/red badge next to cells that use the worker; cells refuse to dispatch to an unhealthy worker.

```bash
curl https://my-worker.example.com/health
```

The `/health` endpoint is **not** gated by `STRATA_WORKER_TOKEN` — platform health probes (Fly, k8s liveness, Cloudflare) don't need the secret.

## Troubleshooting

**`401 Unauthorized` when running a cell.**
`STRATA_WORKER_TOKEN` is set on the worker but the notebook isn't sending it. Confirm `token_env` (or `token`) in `notebook.toml` matches an env var that's actually exported in the strata-server's shell. Restart `strata-server` after exporting; it reads env at startup.

**`Connection refused` or `Could not resolve host`.**
`config.url` doesn't match where the worker is actually listening. From the strata-server host, run `curl <config.url base>/health` — it should respond. For Fly, `fly status` shows the public hostname; for Modal, `modal app list` shows deployed URLs.

**Worker `/health` works but cells fail with `ModuleNotFoundError: <package>`.**
The worker image is missing the dependency the cell needs. Add it to the Dockerfile's `pip install` (Fly) or the `.pip_install(...)` chain (Modal) and redeploy. The worker uses **its own** Python env; nothing from the notebook server's env transfers.

**Cells dispatched to a Modal worker hang for 30+ seconds before output.**
Cold start. Modal scales the function to zero after `scaledown_window` seconds idle; the first request after a scale-down has to provision a fresh container. Either bump `scaledown_window`, set `min_containers=1` on the `@app.function`, or just expect the latency on the first cell after idle.

**`413 Payload Too Large` from the worker.**
A cell input is larger than the worker's max-input limit. Default is 256 MB; override with `STRATA_WORKER_MAX_INPUT_BYTES=<bytes>` on the worker. Better: shrink the input by selecting columns / filtering rows in an upstream cell.

**Fly build fails with `error: failed to fetch wheel` from a workload dep.**
Some Python deps (torch, sentence-transformers) don't ship abi3 wheels and fall back to building from source. If your worker needs one, add `build-essential` (plus the dep-specific toolchain) to the Dockerfile via `RUN apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*` before the `uv pip install` step.

**Modal redeploy hangs at "Building image".**
You changed `.pip_install(...)` — Modal is rebuilding the image layer. With torch + sentence-transformers this takes ~5 minutes the first time on a new image hash. Subsequent deploys with no dep changes hit the layer cache and finish in seconds.

## Live status

When a cell dispatches to a remote worker, the UI shows a pulsing **"dispatching → <name>"** badge during execution. After completion, the worker name and transport type appear in the cell metadata.
