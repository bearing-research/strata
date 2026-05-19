# Installation

## Prerequisites

- **[uv](https://docs.astral.sh/uv/) ≥ 0.8** — install via the
  [uv installer](https://docs.astral.sh/uv/getting-started/installation/)
  (`curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux;
  PowerShell installer on Windows). uv fetches a compatible Python
  automatically, so you don't need Python 3.12+ pre-installed. Strata
  refuses to start outside a uv-managed env.
- **[Rust toolchain](https://rustup.rs/)** (`rustup`) — only for
  source builds (not Docker or `uv add`). Needed by `maturin` to
  compile the native Arrow IPC extension; `cargo` and `rustc` must
  be on `PATH` when you run `uv sync`.
- **[Node.js 25+](https://nodejs.org/)** — only if building the
  frontend from source.

Windows: source builds work via WSL2 (smoother) or native Windows
(uv + rustup + Node have Windows installers). Day-to-day dev is on
macOS/Linux; WSL2 is the better-trodden path.

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

## Commands reference

The PyPI package is `strata-notebook`; the installed Python module
and CLI binary are both named `strata`.

| Command | What it does |
| --- | --- |
| `uv run strata-server` | Start the HTTP server (notebook UI + REST API). Same as `uv run python -m strata`. |
| `uv run strata run <notebook-dir>` | Headless notebook execution for CI / scheduled runs. See [Headless Runner](../notebook/cli.md). |
| `uv run strata export <notebook-dir>` | Render a notebook to markdown or HTML. See [Export](../notebook/export.md). |
| `uv run strata import <ipynb-file>` | Convert a Jupyter `.ipynb` into a Strata notebook directory. See [Import from Jupyter](../notebook/import.md). |
| `uv run strata-worker --port 9000` | Start a remote worker for `# @worker` cells. See [Distributed Workers](../notebook/workers.md). |

The `uv run` prefix ensures the command resolves to the binary
inside the uv-managed venv. If you've activated the venv
(`source .venv/bin/activate`), you can drop the prefix and call
`strata-server` / `strata` / `strata-worker` directly.

## Development Commands

```bash
# Sync with all optional extras — matches CI. Required for the test
# suite because the harness fixtures point the per-notebook venv at
# the dev interpreter, so the dev env needs the [notebook] extra
# (orjson, pyarrow, cloudpickle) for cell-execution tests.
uv sync --all-extras

# Run all tests
uv run pytest

# Format and lint
uv run pre-commit run --all-files

# Type check
uv run ty check src/

# Start frontend dev server (hot reload, proxies to backend)
cd frontend && npm run dev
```

### Integration test dependencies

A subset of integration tests needs a real PostgreSQL instance (the
Iceberg SQL catalog tests). A throwaway Postgres container is shipped
as `docker-compose.test.yml`:

```bash
# Start the test Postgres (port 5432)
docker compose -f docker-compose.test.yml up -d

# Run the integration suite
uv run pytest tests/test_*_integration.py

# Tear it down
docker compose -f docker-compose.test.yml down
```

The compose file is dev-only: it has no health-bound dependency on
the main `docker-compose.yml`, and the credentials are intentionally
fixed (`strata`/`strata`) for predictable local connection strings.
