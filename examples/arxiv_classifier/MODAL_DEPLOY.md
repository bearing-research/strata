# Deploying the gpu-fly Modal worker

The arxiv_classifier demo dispatches its expensive cells (`embed`, `train`)
to a `gpu-fly` worker. In production this is a Modal app running on an
A10G GPU. This file documents how to deploy and update it.

## One-time setup

```bash
# Install the Modal CLI (if you don't have it)
uv tool install modal

# Authenticate against your Modal account
modal token new
# This opens a browser; click approve, you're done.

# Verify
modal profile current
```

## Deploying

```bash
# From the repo root
modal deploy examples/arxiv_classifier/modal_gpu_worker.py
```

The first deploy is slow — Modal builds the image from scratch, which
includes installing the Rust toolchain and compiling Strata's native
extension. Expect 5-10 minutes. Subsequent deploys reuse the cached image
and finish in seconds unless dependencies changed.

When deploy finishes, Modal prints something like:

```
✓ Created objects.
├── 🔨 Created mount...
├── 🔨 Created function gpu_executor.
└── 🔨 Created web function gpu_executor =>
    https://<your-workspace>--strata-gpu-worker-gpu-executor.modal.run
```

Copy that URL — you'll wire it into `notebook.toml` next.

## Wiring the URL into notebook.toml

Edit `examples/arxiv_classifier/notebook.toml` and update the `gpu-fly`
worker's `config.url`. Modal exposes the FastAPI app at the root, so the
executor protocol's `/v1/execute` endpoint is at `<modal-url>/v1/execute`.

```toml
[[workers]]
name = "gpu-fly"
backend = "executor"
runtime_id = "gpu-fly"
[workers.config]
url = "https://<your-workspace>--strata-gpu-worker-gpu-executor.modal.run/v1/execute"
transport = "http"
```

## Verifying it works

```bash
# Hit the health endpoint
curl https://<your-workspace>--strata-gpu-worker-gpu-executor.modal.run/health
```

You should see a JSON response with `"status": "healthy"` and
capability information. The first call after a deploy or after the
auto-stop window pays a cold-start penalty (~10-20s).

## Cost notes

- A10G is roughly $1.10/hr while running
- `scaledown_window=60` makes the container sleep 60 seconds after the
  last request; you only pay while it's awake
- A 5-minute demo session costs ~$0.10
- For YC review, increase `scaledown_window` to 600 or 1800 to avoid
  cold starts on partner clicks; revert after review

## Iterating during development

Use `modal serve` instead of `modal deploy` for fast iteration:

```bash
modal serve examples/arxiv_classifier/modal_gpu_worker.py
```

This streams logs, hot-reloads on file changes, and uses a temporary
URL that disappears when you Ctrl-C. No `modal deploy` needed.

## Updating after Strata source changes

The Modal image installs Strata from `git+https://github.com/bearing-research/strata.git@main`.
After pushing a new commit to `main`, redeploy to pick it up:

```bash
modal deploy examples/arxiv_classifier/modal_gpu_worker.py
```

To pin to a specific tag for stability before YC review, edit
`modal_gpu_worker.py` and change the dependency to
`strata @ git+https://github.com/bearing-research/strata.git@v0.X.Y`.
