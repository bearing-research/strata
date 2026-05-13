"""Template: deploy a Strata worker on Modal with GPU.

The Strata worker is the `strata-worker` console script — it runs an
HTTP endpoint that accepts cell execution requests. Cells run in this
Modal container's Python environment, so install whatever your cells
need (torch, sentence-transformers, datafusion, ...).

Adapt to your workload:
1. Update `PROJECT_GIT_URL` to point at your fork or branch
2. Update `WORKLOAD_DEPS` with the packages your cells import
3. Adjust GPU class, `scaledown_window`, and `max_containers` to taste

Deploy:
    modal deploy examples/arxiv_classifier/modal_gpu_worker.py

Modal prints the URL — paste it into notebook.toml as a `[[workers]]`
entry with `transport = "http"` and `url = "<modal-url>/v1/execute"`.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Configure your deployment here
# ---------------------------------------------------------------------------

# Install Strata itself. Pin to a commit or tag for reproducible builds.
# PyPI users will just write "strata" here.
PROJECT_GIT_URL = "strata-notebook[notebook] @ git+https://github.com/bearing-research/strata.git@7d29f01"

# Dependencies needed by cells that run on this worker. Keep this list
# tight — the image rebuilds when it changes.
WORKLOAD_DEPS: list[str] = [
    "torch>=2.3",
    "sentence-transformers>=3.0",
    "scikit-learn>=1.5",
    "pandas>=2.0.0",
    "numpy>=1.26.0",
    "pyarrow>=18.0.0",
]

APP_NAME = "strata-gpu-worker"
GPU = "A10G"
SCALEDOWN_WINDOW = 60  # seconds idle before stopping
MAX_CONTAINERS = 1  # cell execution is exclusive; don't pipeline


# ---------------------------------------------------------------------------
# Boilerplate — usually no need to edit below
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "git", "ca-certificates")
    # Rust toolchain — required to build Strata's native extension
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("maturin>=1.5")
    .pip_install(PROJECT_GIT_URL, *WORKLOAD_DEPS)
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=MAX_CONTAINERS,
    timeout=600,
)
@modal.asgi_app()
def worker():
    """Expose the Strata worker as a Modal ASGI endpoint."""
    from strata.notebook.remote_executor import create_notebook_executor_app

    return create_notebook_executor_app()
