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

Cells run in **the notebook's locked environment** on a `strata-worker` of this version or later: Strata sends the notebook's `uv.lock` with the cell, and the worker builds that environment once per lock (with `uv`, which the image needs) and reuses it for every later cell with the same lock. The worker advertises this as `locked_environments` in `/health`, and says `false` when `uv` is not on its `PATH`. A worker that answers without it (an older `strata-worker`, one without `uv`, or a custom worker; a `404` on `/health` counts as an answer) runs cells in its own Python environment, so install your workload dependencies (torch, datafusion, sentence-transformers, etc.) into that image before launching it. A worker that cannot be asked at all (unreachable, timed out, or a `5xx` on `/health`) is refused for a notebook that has a `uv.lock`, rather than running the cell in an environment its provenance would not describe. Either way the worker process does **not** require a uv-managed env - it can be pip-installed into a plain Docker image. See [the `environment` block](../reference/executor-protocol.md#the-environment-block) for `STRATA_WORKER_ENV_ROOT` and a prebuilt-environment registry.

R cells work the same way with the notebook's `renv.lock`: the worker restores it with `renv` into one library per lock and R build, reuses it for every later R cell with that lock, and runs the cell with that library first on R's library path. It advertises this as `locked_r_environments`, `true` when its R can load `renv`; a notebook with an `renv.lock` is refused on a worker that cannot be asked, as above. See [the R `environment` block](../reference/executor-protocol.md#the-r-environment-block).

For the wire-level contract - request envelopes, response bundle format, error codes, the pull-model with signed URLs - see the [Executor Protocol](../reference/executor-protocol.md) reference. This page covers deployment and registration; that one covers the bytes on the wire and is what you'd implement against to write a custom worker that doesn't use `strata-worker`.

## Quick start: run a worker locally

Start by getting a worker running on your own machine. This verifies your install before you spend time on a cloud deploy, and the same `# @worker name` annotation works against both local and cloud workers.

**1. Start the worker:**

```bash
strata-worker --host 127.0.0.1 --port 9000
```

A worker runs whatever code it is sent, so this one listens on loopback only.
Without `STRATA_WORKER_TOKEN` that is also the default bind; with the token set
the default is `0.0.0.0`. Bind wider (`--host 0.0.0.0`) only with a token, as
in [Authentication](#authentication): a worker on `0.0.0.0` with
no token lets anyone who reaches the port run code on the machine, and logs a
warning saying so.

Run the installed `strata-worker` (for example `.venv/bin/strata-worker`), not
`uv run strata-worker`: `uv run` stays alive as the worker's parent with
`STRATA_WORKER_TOKEN` still in its environment, where a cell can read it from
`/proc/<pid>/environ`. On Linux the worker logs a warning at startup when its
parent still holds the token.

You should see uvicorn start up:

```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://127.0.0.1:9000
```

**2. Verify it's healthy:**

```bash
curl http://localhost:9000/health
```

Expected response (`locked_environments` is `false` if `uv` is not on the worker's `PATH`, `languages` adds `"r"` when `Rscript` is installed, `locked_r_environments` is `true` when that R can load `renv`, `launch_id` is set only on a worker Strata [launched over SSH](#run-cells-on-a-machine-you-can-ssh-to), and `hardware` lists what the machine reports):

```json
{
  "status": "healthy",
  "capabilities": {
    "protocol_versions": ["v1"],
    "transform_refs": ["notebook_cell@v1"],
    "features": {
      "notebook_protocol_version": "notebook-cell-v1",
      "output_format": "notebook-output-bundle@v1",
      "pull_model": true,
      "cancel": true,
      "locked_environments": true,
      "locked_r_environments": false,
      "languages": ["python"]
    }
  },
  "version": "1.0.0",
  "uptime_seconds": 5.2,
  "launch_id": null,
  "active_executions": 0,
  "max_concurrent": null,
  "gpu_slots": null,
  "free_gpu_slots": null,
  "hardware": {"cpus": 10, "memory_mb": 32768}
}
```

**3. Register it in your notebook.** Either through the **Workers panel** in the sidebar, or by editing `notebook.toml`:

```toml
[[workers]]
name = "local-dev"
backend = "executor"
runtime_id = "local-dev"

[workers.config]
url = "http://127.0.0.1:9000/v1/execute"
transport = "direct"
```

Any name except `local`. That one is reserved for the built-in in-process
worker: `resolve_worker_spec` returns it before it ever looks at the
notebook's own definitions, so a worker you name `local` is never dispatched
to, and the cell runs in-process while appearing to be configured.

**4. Use it in a cell:**

```python
# @worker local-dev
import platform

hostname = platform.node()
```

When the cell runs, the UI shows a pulsing **"dispatching → local-dev"** badge during execution. The `hostname` artifact is what the worker process saw. On one machine that is the same name your laptop reports, so it confirms the round trip rather than the location; point the worker at another host and the name changes.

### Sharing one machine: concurrency and GPUs

A worker runs as many cells at once as it is sent. On a machine several people's cells reach, cap it, and let the worker hand out GPUs itself:

```bash
strata-worker --port 9000 --max-concurrent 2 --gpu-slots 2
```

- **`--max-concurrent N`**: the worker refuses execution number N+1 with `503` and a `Retry-After` header, before downloading any inputs. Unset, it is unlimited.
- **`--gpu-slots N`**: each execution gets a free GPU index from `0..N-1`, set as `CUDA_VISIBLE_DEVICES` for that cell and released when it finishes. The worker's choice overrides any `CUDA_VISIBLE_DEVICES` the cell asked for, so two concurrent cells never share a GPU on a caller's say-so. With every slot taken, the request is refused like any full worker.
- **`--log-level`**: `debug`, `info` (the default), `warning` or `error`.

Both limits are enforced by the worker, not trusted to whatever dispatches to it, and both are advertised in `/health`. For `uvicorn --factory` deployments that cannot pass flags, set `STRATA_WORKER_MAX_CONCURRENT` and `STRATA_WORKER_GPU_SLOTS`.

Once this works locally, the cloud deploys below just change `config.url` from `http://127.0.0.1:9000` to the worker's public URL.

### A worker behind NAT: `--connect`

A machine the server cannot reach (behind NAT, or a firewall that allows only outbound traffic) can still be a worker if it can reach a relay. With `--connect`, the worker binds no port: it opens one outbound WebSocket to the relay, and the relay presents it at a URL the server dispatches to like any other.

```bash
export STRATA_WORKER_TOKEN=<worker-token>          # checked on every request, as always
export STRATA_WORKER_CONNECT_TOKEN=<relay-token>   # presented to the relay
strata-worker --connect wss://relay.example.com/connect
```

Register the worker at the URL the relay gives it, for example `config.url = "https://relay.example.com/w/abc/v1/execute"`, with `token_env` naming the worker token as usual. The relay token is read only from the environment, never from a flag, because the cells the worker runs could read its command line; it is removed from the environment at startup like the worker token. The worker reconnects with backoff when the connection drops, and exits with status 1 if the relay refuses the token.

Strata does not ship a relay. A relay is a small service that accepts the worker's WebSocket and forwards HTTP requests over it; the [Worker Relay Protocol](../reference/worker-connect.md) specifies it. A `signed` worker still fetches its inputs and posts its console to the server's signed URLs directly, so it needs outbound reach to them; a `direct` worker needs only the relay.

## Run cells on a machine you can SSH to

If you have a box you reach over SSH (a GPU machine, a bigger instance, one closer to the data), Strata can turn it into a worker for you in one step, without you deploying anything or hand-editing `notebook.toml`. Strata SSHes in, installs `strata-worker` if it's missing, launches it bound to the box's localhost, opens an `ssh -L` tunnel back to the notebook server, and registers it as a `direct`-transport worker. Cells then run on that box, cached by provenance like any other worker.

**Driving a coding agent (the common case).** When you're driving the notebook with a coding agent ([Drive with a Coding Agent](agent.md)), tell it the target and it calls the `connect_ssh_worker` MCP tool:

```
You: use ssh user@gpu-box for the training cells
Agent → connect_ssh_worker(session_id, "user@gpu-box")
```

The agent's working agreement already tells it to reach for this when you hand it an SSH target, and to keep light cells local with `# @worker local`. To wire the worker up as the session opens instead, pass it to the on-ramp:

```bash
strata agent ./my-notebook --worker-ssh user@gpu-box
```

**From the CLI.** Against a running server (the tunnel is owned by the server, since that's where cells dispatch from), with the session id the server assigned:

```bash
strata worker add-ssh user@gpu-box --server http://localhost:8765 --session <session-id>
# tear it down (add --stop-remote to also stop the worker process on the box):
strata worker rm-ssh gpu-box --server http://localhost:8765 --session <session-id>
```

`add-ssh` makes the new worker the notebook default (pass `--no-default` to skip); `--name` overrides the name derived from the host, and `--no-install` fails fast rather than installing `strata-worker` on the box.

**What to know:**

- **Key-based SSH only.** Strata runs `ssh` in batch mode and never handles passwords, so the target must authenticate non-interactively (an agent/key that works when you run `ssh user@gpu-box` yourself). A `user@host`, a bare `host`, or an `~/.ssh/config` alias all work.
- **The first connect can take a minute** while it installs `strata-worker` on the box (via `uv tool install`); a reconnect skips the install, but it starts a fresh worker rather than adopting one already running.
- **Security.** The worker binds the box's `127.0.0.1` (never a public port) and is reachable only through the authenticated SSH tunnel. A per-worker bearer token is generated for defense-in-depth; it's held in the notebook server's memory and **never written to `notebook.toml`**. Each launch also hands the worker a one-off `launch_id` (over stdin, like the token), and Strata connects only when `/health` through the tunnel reports it, so another process already listening on the box's port never receives the token.
- **A remote cell runs on the box's filesystem.** Absolute paths in the cell resolve there, not on your machine, and a `file://` mount is refused on any remote worker, and cloud mounts use the box's own credentials. Results are cached under the remote environment's identity, so they don't collide with local runs.

## Deploy to the cloud

Strata ships a reference executor as the `strata-worker` console script. **There is no supported-platform list.** A worker is a plain HTTP service, and Strata holds no code for any particular host - `# @worker` resolves to one of two backends, `local` or `executor`, and `executor` is simply a URL. Kubernetes, EC2, Cloud Run, a box under a desk and the two platforms below are all the same thing to it.

The two walkthroughs that follow are worked examples chosen to bracket the tradeoff, not the options:

| Worked example | Best for | Cost model |
| -------- | -------- | ---------- |
| **Fly.io** | CPU workloads (DataFusion, pandas-heavy pipelines) that need always-on or fast cold starts | Per-second VM billing; can scale to zero |
| **Modal** | GPU workloads (torch, embeddings, fine-tuning) that benefit from scale-to-zero | Per-second VM billing; cold-start ~10–30 s for GPU |

Two other routes are documented elsewhere on this page and are often the
shorter path:

- **A machine you can SSH to** - [one command](#run-cells-on-a-machine-you-can-ssh-to), no deploy, no account. Usually the fastest way to reach a GPU you already have.
- **Machines started per job** - [worker pools](#worker-pools-a-different-layer), whose Docker, Fly and RunPod backends boot hardware on demand. A different layer, dispatched by a proxy rather than by the notebook.

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

**`Dockerfile`** - installs `strata-notebook` with its `notebook` extra (the harness needs `orjson` and `cloudpickle` to hand back what a cell produces) and your workload deps into a uv-managed venv. Unlike `strata-notebook`, the worker entry (`strata-worker`) is not gated by Strata's runtime guard, so a plain `pip install` would also work - but the uv-python base image puts `uv` on `PATH`, which the worker needs to run cells in the notebook's locked environment, and keeps tooling consistent across server + worker.

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
      "strata-notebook[notebook]" \
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
fly launch --no-deploy   # first time only - creates the app, accepts fly.toml
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
transport = "direct"
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
# worker entry isn't gated by the runtime guard (strata and strata-notebook
# are) so Modal's standard image stack works. uv is installed alongside so the
# worker can run cells in the notebook's locked environment.
gpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pyarrow>=18.0.0",
    "pandas>=2.0.0",
    "numpy>=1.26.0",
    # Your workload dependencies:
    "torch>=2.3",
    "sentence-transformers>=3.0",
    # Pin to an exact version in production so the worker
    # protocol can't drift relative to the notebook server.
    "strata-notebook[notebook]",
    "uv",
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

Modal prints the deployed URL after the build finishes - something like `https://your-username--my-gpu-worker-gpu-executor.modal.run`. The first build with torch + sentence-transformers takes ~5 minutes; redeploys with no changes hit the layer cache and finish in seconds.

**3. Verify:**

```bash
curl https://your-username--my-gpu-worker-gpu-executor.modal.run/health
```

The first request after a scale-down cold-starts the container (~20–30 s for GPU). Health-check requests do **not** start the GPU function on most Modal plans - if `/health` returns immediately, the function is warm; if it doesn't respond, send a real cell from the notebook to wake it.

**4. Register in `notebook.toml`:**

```toml
[[workers]]
name = "modal-gpu"
backend = "executor"
runtime_id = "modal-a10g-v1"

[workers.config]
url = "https://your-username--my-gpu-worker-gpu-executor.modal.run/v1/execute"
transport = "direct"
```

## Worker pools: a different layer

Everything above is a worker **you** run and **the notebook** dispatches to.
[`strata-pool`](worker-pool.md) is a separate package that hands out machines
on demand instead - starting one when no warm machine of that type is free,
holding it for a tenant, and stopping it when the work finishes.

The distinction that matters is *who calls whom*:

| | Registered worker | Pool-run machine |
| --- | --- | --- |
| Who starts it | You, and it stays up | The pool, on demand; stopped once idle |
| Who dispatches to it | The notebook, from `# @worker` | A caller **above** the notebook |
| Entry point | `POST /v1/execute` (multipart push) or `/v1/execute-manifest` (signed-URL pull), by transport | `POST /execute`, always a manifest |
| Default port | 9000 | 8080 (`DockerBackend`'s `worker_port`) |
| Lifetime | Long-lived, shared across cells | Belongs to one tenant, then destroyed |

Both rows are the **same `strata-worker` binary**, which serves every one of
those paths - so one image covers both uses, and what differs is who dispatches
and which way the input bytes move. Under the push model Strata sends the
inputs; under a manifest the worker fetches its own inputs and uploads its own
result, which is why job bytes never pass through the pool.

**The notebook has no pool client.** Nothing in `strata` imports `strata_pool`,
and `# @worker` cannot name a pool machine: the annotation resolves against the
effective worker policy - notebook-scoped `[[workers]]` in personal mode, the
server-managed registry in service mode - and a pool machine appears in
neither. Composing the two is the job of a proxy above both, which checks the
artifact store first and submits to the pool only on a miss. Getting that order
wrong boots machines to recompute results that already exist.

So the choice is not which platform, but which shape:

- **Hardware you keep** - register it as a worker and point cells at it.
- **Hardware you rent per job**, with metering and tenant isolation, driven by
  a service you are building around Strata - that is the pool.

```bash
pip install strata-pool            # library
pip install "strata-pool[server]"  # plus the HTTP service
```

It ships Docker, Fly and RunPod backends, per-worker credentials, tenant-scoped
machines, a fleet cap, and usage metering. The backends talk to Docker, Fly
and RunPod over `httpx`, so the base install pulls only `httpx`; the extras
are `server` (the HTTP service) and `postgres` (a Postgres store). Build the
image the pool drives with:

```bash
docker build -f worker.Dockerfile -t strata-worker:latest .
```

For R cells, build it with R, the packages `harness.R` needs, and `renv` to restore a notebook's `renv.lock`:

```bash
docker build -f worker.Dockerfile --build-arg WITH_R=true -t strata-worker:r .
```

The image installs a released `strata-notebook` from PyPI, pinned by
`ARG STRATA_VERSION`. The worker must be the same release as the server
(workers and servers upgrade together), so pass
`--build-arg STRATA_VERSION=<server version>` when the two differ. The pin
moves to each new release right after it is published; until then a fresh
image installs the previous release. Building it inside a checkout does not
pick up local worker changes; build a wheel for that. The container refuses to start
without `STRATA_WORKER_TOKEN`, since it binds `0.0.0.0`; the pool mints one per
machine, and you set it yourself to run the image by hand.

See [Worker Pool](worker-pool.md) for machine types, dispatch, the HTTP
service, and the isolation model.

## Registering workers

Workers live in `notebook.toml` under `[[workers]]`. You can add them through the **Workers panel** sidebar (which writes the same TOML) or edit the file directly:

| Field | Description |
| --- | --- |
| `name` | Used in `@worker <name>` annotations and the dropdown UI |
| `backend` | Always `"executor"` for HTTP workers |
| `runtime_id` | Stable identifier hashed into cell provenance - see [Caching](#caching-and-provenance) below |
| `config.url` | The HTTP endpoint for the executor protocol |
| `config.transport` | `"direct"` for direct push (the default), `"signed"` for the pull-model with signed URLs |
| `config.token` | Literal bearer token (dev only) - see [Authentication](#authentication) |
| `config.token_env` | Env var name holding the bearer token (preferred for prod) |

A typical multi-worker notebook ends up with:

```toml
[[workers]]
name = "fly-cpu"
backend = "executor"
runtime_id = "fly-cpu-v1"
[workers.config]
url = "https://my-strata-worker.fly.dev/v1/execute"
transport = "direct"
token_env = "STRATA_FLY_WORKER_TOKEN"

[[workers]]
name = "modal-gpu"
backend = "executor"
runtime_id = "modal-a10g-v1"
[workers.config]
url = "https://...--my-gpu-worker-gpu-executor.modal.run/v1/execute"
transport = "direct"
token_env = "STRATA_MODAL_WORKER_TOKEN"
```

### From the CLI

`strata worker` edits the same `[[workers]]` block from a shell, without a running server. A running server picks the change up on its next session reload.

```bash
strata worker ls ./my-notebook
strata worker add ./my-notebook fly-cpu \
  --url https://my-strata-worker.fly.dev/v1/execute \
  --runtime-id fly-cpu-v1 --token-env STRATA_FLY_WORKER_TOKEN --default
strata worker default ./my-notebook fly-cpu   # omit the name, or pass local, to clear
strata worker rm ./my-notebook fly-cpu
```

`add` replaces a worker of the same name; `--transport` defaults to `direct` and must be `direct` or `signed` (`manifest` and `build` are aliases of `signed`; the MCP tool and the REST workers route refuse anything else too), and `--default` also makes it the notebook default. `rm` refuses the built-in `local` worker and clears the default if it named the removed one. Each prints the resulting worker list, as JSON by default (`--format human` for a table). In service mode the server owns worker definitions, so these commands are refused. For a box you reach over SSH, use [`add-ssh` / `rm-ssh`](#run-cells-on-a-machine-you-can-ssh-to) instead.

### Signed workers and private hosts

A `signed` (pull) worker fetches its inputs from, and uploads its result to, the URLs in the manifest, so it refuses any manifest URL whose host resolves to a private, loopback, link-local or other non-public address (`100.64.0.0/10` included). A Strata server on your own network or tailnet trips this. Name its host in `STRATA_WORKER_ALLOWED_HOSTS` on the worker, or set `STRATA_WORKER_ALLOW_LOCAL_HOSTS=1` to turn the check off (local dev). Those connections also ignore `HTTPS_PROXY`, so a worker that reaches the server only through a proxy needs `STRATA_WORKER_ALLOW_LOCAL_HOSTS`. See [Worker configuration](../reference/configuration.md#worker).

### Server-managed workers (service mode)

A service-mode server keeps its own registry, managed through
`/v1/admin/notebook-workers*` rather than any notebook's `[[workers]]`. Two
things are worth knowing about where it lives:

**It is persisted in the artifact metadata store.** Changes made through the
admin routes are written to the same database as the artifact metadata
(Postgres with `STRATA_ARTIFACT_METADATA_DSN`, otherwise `artifacts.sqlite` in
the artifact directory), so they survive a restart, and every node sharing
that database sees them on its next request. Without an artifact store (a
scan-only server) the registry is kept in memory only.

**The stored registry wins over `[tool.strata.transforms] notebook_workers`.**
The configured table is the bootstrap; once anything has been changed through
the admin routes, the stored registry is in force and editing the config table
has no effect. An *empty* registry is a decision, not an absence, so removing
every worker through the API does not fall back.

**Upgrading from a release that kept `notebook_workers.json`.** On its first
start the server imports that file from the artifact directory into the
metadata store, renames it `notebook_workers.json.migrated` (`.migrated.1` and
so on when that name is taken, so an earlier copy is never replaced), and logs
it. The import is skipped (the file is still renamed) when the store already
holds a registry, so a second node starting with an old copy does not overwrite
it. A name the file lists twice is imported once, from its last entry, which is
the one an earlier release dispatched to.

**A `signed` worker needs transforms enabled.** It runs as a build on the
server, so in service mode set `STRATA_TRANSFORMS_ENABLED=true`
(`[tool.strata.transforms] enabled = true`), which also needs
`STRATA_ARTIFACT_DIR`. Without it the cell fails with
"Signed notebook executor transport requires personal-mode writes or
server-mode transforms to be enabled". A `direct` worker needs neither.

`POST /v1/admin/notebook-workers/reload` refreshes every worker's health and
drops cached health for workers no longer listed. The registry itself needs no
reload, since it is read from the store on every request; a fleet manager
changes it through the admin routes.

An entry has the fields of a notebook's `[[workers]]` plus `enabled`, in the
same shape whether it comes from the admin routes or from
`[tool.strata.transforms] notebook_workers`. `POST /v1/admin/notebook-workers`
adds one, `PUT /v1/admin/notebook-workers/{name}` replaces it, `PATCH` takes
`{"enabled": false}`, and `PUT /v1/admin/notebook-workers` replaces the whole
registry with `{"workers": [...]}`. The routes answer only in service mode
(`409` otherwise) and, under principal auth, need the `admin:notebook-workers`
scope.

```bash
curl -X POST https://strata.example.com/v1/admin/notebook-workers \
  -H 'Content-Type: application/json' \
  -d '{"name": "gpu-a100", "backend": "executor", "runtime_id": "gpu-a100-v1",
       "config": {"url": "https://gpu.internal/v1/execute", "transport": "direct",
                  "token_env": "STRATA_GPU_WORKER_TOKEN"}}'
```

```toml
[tool.strata.transforms]
notebook_workers = [
  { name = "gpu-a100", backend = "executor", runtime_id = "gpu-a100-v1", config = { url = "https://gpu.internal/v1/execute", transport = "direct", token_env = "STRATA_GPU_WORKER_TOKEN" } },
]
```

`token_env` names a variable in the server's environment.

**Personal-mode servers get the registry too.** A personal server started with
a registry offers those machine types to every notebook it opens, with no
`[[workers]]` block in `notebook.toml`, which is the point, since writing one
into every notebook puts the catalogue in git diffs and drifts as soon as it
changes. They appear in the Workers panel with `source: server`.

A notebook's own `[[workers]]` still win: define `gpu-a100` in your
`notebook.toml` and that is what `@worker gpu-a100` resolves to, the same
precedence annotations have over persisted config. Your notebook's worker
definitions stay editable either way.

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
def gpu_executor(): ...
```

…then create the Modal secret once: `modal secret create strata-worker-token STRATA_WORKER_TOKEN=<paste-token-here>`.

**3. Tell the notebook server about the token.** Export it as an environment variable wherever you run `strata-notebook`:

```bash
export STRATA_FLY_WORKER_TOKEN=<paste-token-here>
uv run strata-notebook
```

…and reference that env var in `notebook.toml`:

```toml
[workers.config]
url = "https://my-strata-worker.fly.dev/v1/execute"
transport = "direct"
token_env = "STRATA_FLY_WORKER_TOKEN"
```

`token_env` is preferred over `token` because the literal-token form gets committed to your notebook repo. Use `token = "..."` only for one-off local experiments.

A worker with `STRATA_WORKER_TOKEN` set rejects unauthenticated requests with `401 Unauthorized`. `/health` stays open so platform health probes work without the secret. The worker app takes the token out of its process environment when it is created and holds it in memory, zeroing the value in the environment block the process started with, so a cell it runs cannot read it from `os.environ` or from `/proc/<ppid>/environ`. The same goes for the credential variables (`STRATA_NOTEBOOK_CREDENTIALS`, `STRATA_NOTEBOOK_MOUNT_CREDENTIALS`). This holds for `strata-worker` and for an app built directly with `create_notebook_executor_app()`, as in the Modal example above.

## Using workers in cells

Annotate any cell with `# @worker <name>`:

```python
# @name Embed Abstracts
# @worker modal-gpu
# @timeout 300
embeddings = model.encode(abstracts, batch_size=256)
```

The worker annotation is the **only** change needed; the cell code itself is identical to local execution. On a worker that runs locked environments the cell gets the notebook's own packages; on one that does not, the worker's image needs them installed.

### Precedence

When multiple levels define a worker, the most specific wins:

1. `# @worker X` annotation in the cell source (highest)
2. Cell-level worker override (from the cell's stored config)
3. Notebook-level worker default (from the Workers panel)

### Device and tensor placement (PyTorch / GPU)

Each cell runs in an isolated process and exchanges **values** (pickled
artifacts), not a shared Python kernel. That has a consequence specific to
device-bound objects like `torch.device` and CUDA tensors when a DAG mixes a
**local CPU** machine with a **remote GPU** `@worker`:

- **Don't share a resolved `device` across the boundary.** A cell that does
  `device = "cuda" if torch.cuda.is_available() else "cpu"` and exports `device`
  resolves it *where that cell ran*. If it runs locally (no CUDA), the value is
  `"cpu"`, and a GPU `@worker` cell consuming it will train on **CPU on the GPU
  box**. Resolve the device **at point of use, inside each cell**, rather than
  passing one `device` artifact downstream.
- **Save/load model weights with explicit `map_location`.** A CUDA tensor or a
  CUDA-resident model pickled on the worker fails to deserialize on a CPU-only
  host (`torch.cuda.is_available()` is `False` there). Move tensors to CPU
  before they become a cross-boundary artifact (`state = model.cpu().state_dict()`),
  or load with `torch.load(path, map_location="cpu")`.
- **Prefer passing a state_dict / checkpoint path** between a training cell and
  downstream analysis cells, rather than a live CUDA model object.

This is partly notebook design - Strata can't know your placement intent - but
the isolation boundary makes it a footgun worth designing around explicitly.

## Caching and provenance

Remote execution results are cached identically to local cells. The provenance hash includes the worker's `runtime_id`, so:

- Same code + same inputs + same `runtime_id` = cache hit, no remote call.
- Changing `runtime_id` (e.g., switching from `gpu-a10g` to `gpu-h100`) invalidates the cache for cells using that worker.

`runtime_id` names the hardware and drivers a worker runs on. The Python packages a cell imports come from the notebook's lock on a worker that runs locked environments, and the lock is already part of the provenance hash.

**When to bump `runtime_id`:**

- You upgraded what the image provides outside the lock (CUDA, drivers, model weights baked into the image), or, on a worker that runs cells in its own environment, its Python dependencies, and want downstream cells to re-run.
- You moved a worker to different hardware (CPU type, GPU SKU) and the numerical output may differ.
- You explicitly want to bust the cache for a debugging session.

**When to leave `runtime_id` alone:**

- Redeploying the same image (no dep changes). The cache is correct by construction; re-running is wasted compute.
- Scaling the number of worker instances. Output is deterministic given the same inputs.

If you don't set `runtime_id`, the identity is the backend, the name and a hash of the worker's config, so two notebooks with a worker named `gpu` pointing at different URLs already get different identities. Set `runtime_id` when you want the opposite: two differently configured workers treated as one runtime, so a result computed on either is a cache hit for the other.

## Health checks

Every worker exposes `GET /health`. The notebook UI polls this and shows a green/red badge next to cells that use the worker. The badge is advisory: execution does not consult it, so a cell dispatched to a worker that has gone down fails when the connection does, not before.

```bash
curl https://my-worker.example.com/health
```

The `/health` endpoint is **not** gated by `STRATA_WORKER_TOKEN` - platform health probes (Fly, k8s liveness, Cloudflare) don't need the secret.

## Troubleshooting

**`401 Unauthorized` when running a cell.**
`STRATA_WORKER_TOKEN` is set on the worker but the notebook isn't sending it. Confirm `token_env` (or `token`) in `notebook.toml` matches an env var that's exported in the strata-notebook's shell. Restart `strata-notebook` after exporting; it reads env at startup.

**`Connection refused` or `Could not resolve host`.**
`config.url` doesn't match where the worker is actually listening. From the strata-notebook host, run `curl <config.url base>/health` - it should respond. For Fly, `fly status` shows the public hostname; for Modal, `modal app list` shows deployed URLs.

**Worker `/health` works but cells fail with `ModuleNotFoundError: <package>`.**
The cell ran in the worker's own Python env, which is missing the dependency. That happens when the worker does not report `locked_environments: true` in `/health` (usually because `uv` is not on its `PATH`), or when the notebook has no `uv.lock`. Install `uv` in the image so the notebook's lock is used, or add the package to the Dockerfile's `pip install` (Fly) or the `.pip_install(...)` chain (Modal) and redeploy.

**Cells dispatched to a Modal worker hang for 30+ seconds before output.**
Cold start. Modal scales the function to zero after `scaledown_window` seconds idle; the first request after a scale-down has to provision a fresh container. Either bump `scaledown_window`, set `min_containers=1` on the `@app.function`, or just expect the latency on the first cell after idle.

**`413 Payload Too Large` from the worker.**
A cell input is larger than the worker's max-input limit. Default is 2 GiB; override with `STRATA_WORKER_MAX_INPUT_BYTES=<bytes>` on the worker. Inputs are written to disk as they arrive, so the limit to size against is the worker's disk, not its memory. Better: shrink the input by selecting columns / filtering rows in an upstream cell.

**Fly build fails with `error: failed to fetch wheel` from a workload dep.**
Some Python deps (torch, sentence-transformers) don't ship abi3 wheels and fall back to building from source. If your worker needs one, add `build-essential` (plus the dep-specific toolchain) to the Dockerfile via `RUN apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*` before the `uv pip install` step.

**Modal redeploy hangs at "Building image".**
You changed `.pip_install(...)` - Modal is rebuilding the image layer. With torch + sentence-transformers this takes ~5 minutes the first time on a new image hash. Subsequent deploys with no dep changes hit the layer cache and finish in seconds.

## Live status

When a cell dispatches to a remote worker, the UI shows a pulsing **"dispatching → <name>"** badge during execution, and the cell's console output streams back from the worker as it is produced, replacing what the cell's last run printed. Someone who opens the notebook mid-run, or reloads the page, sees the last 64 KiB the cell has printed so far and the rest as it arrives. After completion, the worker name and transport type appear in the cell metadata.
