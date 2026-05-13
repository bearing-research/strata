# Installation

## Prerequisites

- **Python 3.12+**
- **Rust toolchain** (for the native Arrow IPC extension)
- **Node.js 25+** (only if building the frontend from source)

## GitHub Codespaces (zero setup)

Click **"Open in Codespaces"** on the [repo](https://github.com/bearing-research/strata)
and the server is running by the time the browser tab opens. No local
toolchain needed. See [Codespaces](../deployment/codespaces.md) for what
the devcontainer provisions.

## Docker (Easiest)

No local toolchain required:

```bash
docker compose up -d --build
```

Open [http://localhost:8765](http://localhost:8765) in your browser.

## From Source

### 1. Install dependencies and build the Rust extension

```bash
uv sync
```

This installs all Python dependencies and compiles the Rust extension via maturin.

### 2. Build the frontend (optional)

If you want the notebook UI served by the backend:

```bash
cd frontend
npm ci
npm run build
cd ..
```

The server auto-detects `frontend/dist/` and serves it.

### 3. Start the server

```bash
uv run strata-server
```

Or equivalently:

```bash
uv run python -m strata
```

The server starts on port 8765 by default. Open [http://localhost:8765](http://localhost:8765).

## Verify

```bash
curl http://localhost:8765/health
# {"status":"ok"}
```

## Development Commands

```bash
# Run all tests
uv run pytest

# Format and lint
pre-commit run --all-files

# Type check
uv run ty check src/

# Start frontend dev server (hot reload, proxies to backend)
cd frontend && npm run dev
```
