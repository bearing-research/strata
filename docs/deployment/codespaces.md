# GitHub Codespaces

Click **"Open in Codespaces"** on the repo to get a full development environment with the server running automatically.

## What's included

The `.devcontainer/` configuration provides:

- Python 3.13 via devcontainer features
- Rust toolchain for the native extension
- Node.js 25 for frontend development
- VS Code extensions: Python, Ruff, Volar, rust-analyzer

## Setup flow

1. **`postCreateCommand`** (`setup.sh`) runs once on container creation:
    - Installs `uv`
    - Runs `uv sync` (builds Rust extension)
    - Builds the frontend (`npm ci && npm run build`)

2. **`postStartCommand`** (`start.sh`) runs on every container start:
    - Creates a scratch notebook directory if none exists
    - Starts the Strata server in the background
    - Waits for the health check to pass

## Port forwarding

Port **8765** is forwarded automatically with `onAutoForward: "openBrowser"`, so your browser opens the notebook UI as soon as the server is ready.

## First-time startup

The initial `postCreateCommand` takes 3-5 minutes (Rust compilation). Subsequent starts are fast since the build is cached in the Codespace volume.

PyPI wheels now ship, so the Codespace setup is `uv tool install
strata-notebook` and finishes in under a minute. If you're using a
Codespace to **contribute to Strata** (not just try it), install the
Rust toolchain and clone the repo manually:

```bash
curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y
git clone https://github.com/bearing-research/strata.git
cd strata && uv sync --all-extras
```
