---
description: Install Strata, and the commands it gives you.
---

# Install

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** for the recommended install - via the
  [uv installer](https://docs.astral.sh/uv/getting-started/installation/)
  (`curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux;
  PowerShell installer on Windows). uv fetches a compatible Python
  automatically, so you don't need Python 3.12+ pre-installed. The
  other installs below need Python 3.12+ instead.
- **[Node.js 26+](https://nodejs.org/)** - only if building the
  frontend from source.

Windows: source builds work via WSL2 (smoother) or native Windows
(uv and Node have Windows installers). Day-to-day dev is on
macOS/Linux; WSL2 is the better-trodden path.

## From PyPI (recommended)

```bash
uv tool install strata-notebook
strata-notebook
```

This puts `strata-notebook`, `strata`, `strata-notebook-tui`,
`strata-worker` and `strata-agent-hook` on `PATH` in a tool environment of
their own. The wheel
is pure Python and ships the frontend prebuilt.

### pipx, pip or conda

Any other installer works too. Strata depends on uv (0.12 or 0.13), so every
install brings a `uv` binary, and Strata runs that uv for every notebook
environment whatever is on `PATH`. You don't need uv installed beforehand,
only Python 3.12 or newer.

```bash
# pipx: an environment of its own, like uv tool
pipx install strata-notebook

# pip into a virtual environment
python -m venv ~/strata && ~/strata/bin/pip install strata-notebook

# pip into a conda environment
conda create -n strata python=3.13 && conda activate strata
pip install strata-notebook
```

After a pip install, run the commands from that environment: activate it, or
call `~/strata/bin/strata-notebook`. Strata isn't on conda-forge yet; pip into
the conda environment as above.

The `uv` binary lands in the environment beside `strata`, so an activated
environment puts that uv first on `PATH`. To run another uv, set `STRATA_UV` to
its path ([configuration](../reference/configuration.md)). A
[project the scratchpad attaches to](../notebook/mcp.md#a-projects-own-daemon)
runs the uv on your `PATH` first, since that uv wrote the project's lock.

### Upgrade and remove

| Installed with | Upgrade | Remove |
| --- | --- | --- |
| uv tool | `uv tool upgrade strata-notebook` | `uv tool uninstall strata-notebook` |
| pipx | `pipx upgrade strata-notebook` | `pipx uninstall strata-notebook` |
| pip, in a venv or conda environment | `pip install -U strata-notebook` | Delete the environment |

`uv tool upgrade` also moves uv to the newest version Strata allows. pip and pipx
leave it where it is unless the new Strata needs a newer one; add `uv` to the pip
command (`pip install -U strata-notebook uv`) to take it as well.

### Extras

Add extras in brackets with any installer, for example
`uv tool install "strata-notebook[mcp,tui]"` or
`pipx install "strata-notebook[mcp,tui]"` for `strata agent`:

| Extra | Enables |
| --- | --- |
| `mcp` | The `/mcp` endpoint a coding agent connects to. It is off until the server starts with `STRATA_MCP_ENABLED=true`, which `strata agent` sets for the server it starts. `strata agent` needs it. |
| `tui` | The terminal viewer: `strata watch`, `strata-notebook-tui`, and `strata agent` unless you pass `--no-tui`. |
| `sql` | SQL cells. Add a driver extra for each database you connect to: `sql-duckdb`, `sql-sqlite`, `sql-postgres`, `sql-snowflake`, `sql-bigquery`. |
| `postgres` | A Postgres metadata database for the artifact store (`STRATA_ARTIFACT_METADATA_DSN`), for several nodes sharing one store. |
| `azure` | The Azure Blob Storage artifact backend, notebook `az://` mounts, and reading Azure lake tables (`adlfs`). |
| `s3` | The S3 artifact backend (`STRATA_ARTIFACT_BLOB_BACKEND=s3`, through boto3, presigned build-manifest URLs included) and notebook `s3://` mounts (`s3fs`). |
| `gcs` | The GCS artifact backend (`STRATA_ARTIFACT_BLOB_BACKEND=gcs`, through google-cloud-storage, presigned build-manifest URLs included) and notebook `gs://` mounts (`gcsfs`). |
| `otel` | OpenTelemetry tracing. |
| `infisical` | The [Infisical secret provider](../notebook/secrets.md). |
| `notebook` | The cell runtime libraries (pandas, numpy, orjson, cloudpickle, matplotlib, Pillow) in Strata's own environment. A remote worker image needs it; a local server does not, since each notebook gets its own venv. |

## GitHub Codespaces (zero setup)

On the [repo](https://github.com/bearing-research/strata), click
**Code → Codespaces → Create codespace on main**, and the server is running by the time the browser tab opens. No local
toolchain needed. See [Codespaces](../deployment/codespaces.md) for what
the devcontainer provisions.

## Docker (easiest)

No local toolchain required beyond Docker. From a clone of the
[repo](https://github.com/bearing-research/strata), which holds the compose file:

```bash
docker compose up -d --build
```

Open [http://localhost:8765](http://localhost:8765) in your browser.

Releases after 0.9.0 also publish the image as `ghcr.io/bearing-research/strata:<version>`;
see [Docker → Prebuilt image](../deployment/docker.md#prebuilt-image).

## From source

### 1. Install dependencies

```bash
uv sync
```

This installs all Python dependencies and Strata itself, editable.

### 2. Build the frontend (optional)

If you want the notebook UI served by the backend:

```bash
cd frontend
npm ci
npm run build
cd ..
```

The server serves `src/strata/_frontend/` if it exists, else
`frontend/dist/`. `_frontend/` is the bundle a release copies in, so after a
fresh build either copy `frontend/dist/` into `src/strata/_frontend/` or
remove `src/strata/_frontend/`.

### 3. Start the server

```bash
uv run strata-notebook
```

Or equivalently:

```bash
uv run python -m strata
```

The server starts on port 8765 by default. Open [http://localhost:8765](http://localhost:8765).

## Verify

```bash
strata --version
# strata-notebook <version>
curl http://localhost:8765/health
# {"status":"ok","pid":12345}
```

## Commands reference

The PyPI package is `strata-notebook`; the installed Python module
and CLI binary are both named `strata`.

The table assumes a source checkout, where `uv run` resolves each command
inside the project venv. After an install from PyPI (uv tool, pipx, or pip into
an environment you have activated), drop the `uv run` prefix and call
`strata-notebook` / `strata` / `strata-worker` directly.

| Command | What it does |
| --- | --- |
| `uv run strata-notebook` | Start the HTTP server (notebook UI + REST API). Same as `uv run python -m strata`. |
| `uv run strata run <notebook-dir>` | Headless notebook execution for CI / scheduled runs. See [CLI](../notebook/cli.md). |
| `uv run strata validate <notebook-dir>` | Static checks (schema, annotations, DAG) without executing. See [CLI](../notebook/cli.md#strata-validate). |
| `uv run strata tutorial [name]` | Copy a bundled sample into the notebook storage directory under a fresh name, open it and print its URL, starting a server if none is running. With no name, list the samples. See [Quickstart](notebook.md#1-take-the-tutorial). |
| `uv run strata new <name>` | Scaffold a notebook directory. See [CLI](../notebook/cli.md#strata-new). |
| `uv run strata cell <action>`, `strata dag`, `strata status`, `strata dep <action>` | Agent-facing notebook tooling: inspect (`cell list/show`, `dag`, `status`), save a display output such as a plot to a file (`cell output`), execute (`cell run/test`), cancel a server run (`cell cancel`, on a server only), and author (`cell add/edit/rm/mv/annotate`, `cell pin-fetch`, `dep add/rm`) - offline against a directory, `--server/--session` against a running server, or through the daemon serving a directory when one runs. See [Notebook CLI](../notebook/cli.md#inspecting-a-notebook-cell-dag-status). |
| `uv run strata worker <action>` | Manage a notebook's workers: `ls`, `add`/`rm`/`default` (local, edits `notebook.toml`), and `add-ssh`/`rm-ssh` (provision + tunnel a worker over SSH, against a running server). See [Workers](../notebook/workers.md#run-cells-on-a-machine-you-can-ssh-to). |
| `uv run strata agent <notebook-dir>` | One-command on-ramp for driving a notebook with a coding agent: create-or-open the notebook, start (or reuse) a server with the MCP endpoint enabled, open a session, write the `.mcp.json` + `CLAUDE.md` the agent needs, and attach the TUI (skip it with `--no-tui` and watch in the web UI instead). Then run `claude` in the notebook directory to drive it live. Needs the `[mcp]` extra, plus `[tui]` unless you pass `--no-tui`. See [Use a coding agent](../notebook/agent.md). |
| `uv run strata scratch start\|stop\|status [--root DIR]` | Start, stop or show a project's scratchpad daemon: a server with the [scratchpad MCP profile](../notebook/mcp.md#a-projects-own-daemon) serving `<DIR>/.strata-scratch/`, which it creates on first start. `DIR` defaults to the project the command runs in: the nearest directory at or above it holding a `pyproject.toml`, within its git repository, else the repository ([which project](../notebook/mcp.md#a-projects-own-daemon)). One per project, shared by every client there; it stops itself after 30 idle minutes. |
| `uv run strata scratch ls\|gc\|rm\|retire` | List every project's scratchpad (size, last use, overage, daemon up), retire its idle cells and sweep it now (`gc [--root DIR \| --all]`), stop its daemon and delete it (`rm [--root DIR]`, which costs reruns only), or retire one cell by hand (`retire <cell> [--root DIR]`). See [A project's own daemon](../notebook/mcp.md#a-projects-own-daemon). |
| `uv run strata scratch run\|show [--root DIR]` | Run Python in the project's scratchpad (`run -c CODE`, or `--file`, `--cell ID` to replace one, `--fresh`) or print what it holds (`show [NAME]`), as the MCP `run` and `scratchpad` tools do, starting its daemon if none runs; prints JSON. For an agent with only a shell. Needs the `[mcp]` extra. See [From a shell](../notebook/mcp.md#from-a-shell-strata-scratch-run-and-show). |
| `strata agent install\|uninstall` | From a `uv tool` or `pipx` install of Strata, since the config it writes runs that `strata` in every project: wire the project scratchpad into Claude Code's or Codex's own config (`--target claude,codex`, default every one on PATH; `--yes` skips the question): the MCP entry, its permission, and a block in `CLAUDE.md` / `AGENTS.md`. `uninstall` removes exactly that. See [Install it into the agent](../notebook/mcp.md#install-it-into-the-agent-strata-agent-install). |
| `uv run strata mcp [--root DIR]` | Serve a project's scratchpad to a coding agent as a stdio MCP server, for the agent's MCP config: it lists the tools at once and starts the project's scratchpad daemon as the session opens when the project has a scratchpad, else on the first tool call. Needs the `[mcp]` extra. See [MCP server](../notebook/mcp.md#from-an-agents-config-strata-mcp). |
| `uv run strata export <notebook-dir>` | Render a notebook to markdown or HTML (add `--app-view` for a frozen dashboard snapshot), or write a portable snapshot zip with `--to snapshot`. See [Export and import](../notebook/export.md). |
| `uv run strata import <file>` | Convert a Jupyter `.ipynb` into a Strata notebook directory, or unpack a snapshot zip back into one. See [Coming from Jupyter](../notebook/import.md). |
| `uv run strata artifact <cmd> [--artifact-dir DIR]` | Inspect the server's configured artifact store (its metadata DSN and blob backend included) without a running server, or one local SQLite store with `--artifact-dir`: `list`, `show <ref>`, `lineage <ref>` (renders model ← features ← scan ← table @ snapshot), `pull <ref> --to FILE`, `audit [name]` (registry history), `pending` (approval queue), `verify`, `gc` (collect what nothing holds, least recently used first; `--dry-run` to preview, `--max-bytes`, `--max-idle-days`, `--min-idle-seconds`). Also publishes and moves results: `publish` / `unpublish` (a link anyone can open), `promote` (copy an artifact and its chain to the team store and name it there), `export` (write a tabular artifact into an Iceberg table), `archive` (a self-contained bundle: page, bytes, manifest, README). `<ref>` is a name, `id@v=N`, or bare id. |
| `uv run strata migrate --to-dsn <dsn>` | Copy artifact-store metadata from SQLite to Postgres. Setting `STRATA_ARTIFACT_METADATA_DSN` alone starts an empty store; this carries an existing one across. Metadata only: blobs are configured separately. |
| `uv run strata apikey <cmd>` | Mint, list and revoke API keys for a server running with `auth_mode="api_key"`: `create`, `list` (never shows secrets), `revoke <id>`. |
| `uv run strata env gc` | Remove shared environments no notebook links to. Only relevant with the shared environment backend. See [Environments](../notebook/environment.md). |
| `uv run strata watch [dir\|--session]` | Attach the read-only terminal viewer to a live notebook session without writing any config (unlike `strata agent`). Needs the `[tui]` extra. See [Terminal viewer](../notebook/tui.md). |
| `uv run strata-notebook-tui` | Read-only terminal viewer that attaches to a running notebook session. Needs the `[tui]` extra. See [Terminal viewer](../notebook/tui.md). |
| `strata-worker --port 9000` | Start a remote worker for `# @worker` cells. It binds `127.0.0.1` unless `STRATA_WORKER_TOKEN` is set (then `0.0.0.0`) or `--host` says otherwise. Run the installed binary (for example `.venv/bin/strata-worker`), not through `uv run`, whose process would keep `STRATA_WORKER_TOKEN` readable to cells. With `--connect wss://...` it binds no port and dials out to a relay instead. See [Workers](../notebook/workers.md). |

In a checkout with the venv activated (`source .venv/bin/activate`),
you can drop the prefix too.

## Development commands

```bash
# Sync with all optional extras - matches CI. Required for the test
# suite because the harness fixtures point the per-notebook venv at
# the dev interpreter, so the dev env needs the [notebook] extra
# (orjson, pyarrow, cloudpickle) for cell-execution tests.
uv sync --all-extras

# Run all tests
uv run pytest

# Format and lint
uv run pre-commit run --all-files

# Type check (scope is set in pyproject.toml, as in CI)
uv run ty check

# Frontend dev server with hot reload. Vite does not proxy to the
# backend: point the UI at the server, and let the server accept
# Vite's origin. The server in one terminal, Vite in another:
STRATA_CORS_ALLOW_ORIGINS='["http://localhost:5173"]' uv run strata-notebook
cd frontend && npm ci && VITE_STRATA_URL=http://localhost:8765 npm run dev
```

### Integration test dependencies

The end-to-end integration script needs a real PostgreSQL instance for
its Iceberg SQL catalog. A throwaway Postgres container is shipped
as `docker-compose.test.yml`:

```bash
# Start the test Postgres (port 5432)
docker compose -f docker-compose.test.yml up -d

# Run the integration script against it (starts its own server)
STRATA_CATALOG_URI=postgresql://strata:strata@localhost:5432/iceberg_catalog \
  uv run python scripts/integration_test.py --start-server

# Tear it down
docker compose -f docker-compose.test.yml down
```

The compose file is dev-only: it has no health-bound dependency on
the main `docker-compose.yml`, and the credentials are intentionally
fixed (`strata`/`strata`) for predictable local connection strings.
