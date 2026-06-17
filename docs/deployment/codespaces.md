# GitHub Codespaces

Click **"Open in Codespaces"** on the repo to get a ready-to-use Strata environment with the server running automatically.

## What's included

The `.devcontainer/` configuration provides:

- Python 3.13 (devcontainer feature, on an Ubuntu 22.04 base)
- VS Code extensions: Python, Ruff
- Strata installed from the published PyPI wheel — no Rust or Node build

## Setup flow

1. **`postCreateCommand`** (`setup.sh`) runs once on container creation:
    - Installs `uv`
    - `uv tool install strata-notebook` — the prebuilt wheel, so it finishes in under a minute (no Rust compilation)

2. **`postStartCommand`** (`start.sh`) runs on every container start:
    - Starts the Strata server in the background and waits for the health check to pass

## Port forwarding

Port **8765** is forwarded automatically with `onAutoForward: "openBrowser"`, so your browser opens the notebook UI as soon as the server is ready.

## Contributing to Strata

The Codespace installs the published wheel, which is all you need to *use*
Strata. To **develop** Strata — building the Rust extension and frontend
from source — set up a full dev environment instead:

```bash
curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y   # Rust toolchain
git clone https://github.com/bearing-research/strata.git
cd strata && uv sync --all-extras
```
