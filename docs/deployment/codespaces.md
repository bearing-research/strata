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

!!! info "Will simplify once PyPI wheels ship"
    The Rust toolchain install is only required because
    `strata-notebook` isn't on PyPI yet. Once we publish pre-built
    wheels (tracked for 0.1.0), the devcontainer can drop the Rust
    step and rely on `uv add strata-notebook` pulling a wheel
    directly. Initial setup will drop to well under a minute.
