# Headless Runner

`strata run` executes every cell in a notebook directory without starting the
server or opening the UI. Useful for CI, scheduled jobs, and sanity-checking
that a notebook still works after a dependency bump.

For rendering a notebook to a shareable markdown or HTML file (without
executing anything), see [Export](export.md) and its `strata export`
sibling command. For converting an existing Jupyter `.ipynb` into a
Strata notebook directory, see [Import from Jupyter](import.md) and
its `strata import` sibling.

It reuses the same `NotebookSession` and `CellExecutor` the UI uses, so the
execution path is identical, artifact cache hits, cascade ordering, worker
dispatch, and mount resolution all behave the same way.

## Usage

```bash
strata run <notebook_dir> [options]
```

`<notebook_dir>` must be a path to a directory containing `notebook.toml`
(plus `cells/`, `pyproject.toml`, `uv.lock`). It's the same on-disk layout the
UI works with; you can pass any notebook directory from `STRATA_NOTEBOOK_STORAGE_DIR`
or anywhere else on the filesystem. `strata run` is **local-only** — it does
not talk to a running `strata-notebook` or a service-mode deployment. Each
invocation opens its own `NotebookSession`, runs the cells, and exits.

### Options

| Flag          | Description                                                      |
| ------------- | ---------------------------------------------------------------- |
| `--force`     | Ignore the artifact cache and re-execute every cell from scratch |
| `--no-sync`   | Skip `uv sync`; require `.venv/` to already exist                |
| `--format`    | `human` (default) or `json` — both write to stdout               |
| `--quiet`     | Suppress per-cell status lines (human format only)               |

`--format json` writes a single JSON object to **stdout** when the run
finishes: `{"notebook", "success", "duration_ms", "cells": [...]}` with one
entry per cell (`id`, `status` ∈ `ok|error|skipped`, `duration_ms`,
`cache_hit`, plus `stdout` / `stderr` — truncated at 10k chars — and
`error` / `reason` where applicable). Cache hits replay the stored
artifact without re-emitting console output, so `stdout` can be absent on
warm runs (`--force` re-executes). Errors and pre-flight diagnostics go to
stderr regardless of format. Pipe stdout to `jq`:
`strata run ... --format json | jq '.cells[] | select(.status == "error")'`.

### Exit Codes

| Code | Meaning                                                        |
| ---- | -------------------------------------------------------------- |
| `0`  | All cells succeeded                                            |
| `1`  | One or more cells failed                                       |
| `2`  | Invocation error (bad path, env sync failure, malformed TOML)  |

CI scripts can branch on the exit code; the structured output stays parseable
even when some cells failed (each failure produces a JSON object with
`status: "error"` and the traceback).

## Example: GitHub Actions

```yaml
jobs:
  notebooks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@v8.1.0
      - name: Install Strata
        # uv tool install creates a uv-managed env at
        # ~/.local/share/uv/tools/strata-notebook with the strata
        # CLI on PATH — satisfies the runtime guard.
        run: uv tool install strata-notebook
      - name: Run notebook
        run: strata run ./notebooks/daily_report --format json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

`--format json` emits a single structured result object, so downstream steps
can parse per-cell status without screen-scraping.

## What Gets Run

Every cell in the notebook executes in topological order, exactly as if you
had clicked **Run All** in the UI. Cached artifacts are reused, the first run
populates the cache; subsequent runs on unchanged source + inputs return
instantly.

Passing `--force` invalidates the cache and forces a full rebuild, which is
what you usually want in CI if you're testing that the code *still produces*
the expected artifacts.

## What Does Not Happen

- **No server starts.** No ports are bound; no UI is served.
- **No WebSocket broadcasts.** Progress is written to stdout only.
- **No interactive prompts.** A cascade that would pop a confirmation in the
  UI just runs, the CLI treats every cell as "confirmed."
- **No AI assistant.** `strata run` only executes declarative cells.

## Environment & Secrets

`strata run` reads the notebook's `[env]` and `[secret_manager]` blocks the
same way the server does. Secret-manager credentials (e.g.
`INFISICAL_CLIENT_ID` / `INFISICAL_CLIENT_SECRET`) must be present in the
shell that invokes the command, they are never stored in the notebook.

For notebooks that require env vars set only via the Runtime panel (never
committed), export them before invoking `strata run`.

## `strata validate`

Static checks without executing anything — no environment sync, no
subprocesses, no LLM calls:

```bash
strata validate <notebook_dir> [--format human|json]
```

- `notebook.toml` parses and the cell files load
- the DAG builds without cycles
- per-cell annotation diagnostics — the **same validation the server runs
  on open / reload** (`worker_unknown`, `loop_missing_carry`,
  `sql_missing_connection`, malformed `@output_schema`, …)

Exit codes mirror `strata run`: `0` valid (warnings allowed), `1` invalid
(parse failure, DAG cycle, or any error-severity diagnostic), `2`
invocation error. `--format json` emits
`{"notebook", "valid", "errors", "cells", "summary"}` where each cell
carries its `defines` / `references` (so you can check the DAG wiring you
intended) and its `diagnostics` with `severity` / `code` / `message` /
`line`.

The intended loop for scripts and coding agents: **write files → validate →
fix → run**. Validation is cheap enough to call after every edit; `strata
run` is the expensive step. See
[Authoring Notebooks Programmatically](agent-authoring.md).

## `strata new`

Scaffold a notebook directory without the server:

```bash
strata new "My Analysis" [--parent DIR] [--python 3.12] [--no-env] [--format human|json]
```

Creates `<parent>/my_analysis/` with `notebook.toml`, `pyproject.toml`
(pre-seeded with the notebook runtime packages), and an empty `cells/`
directory, then syncs the venv (skip with `--no-env`; `strata run` syncs it
later). Idempotent on an existing notebook directory: the `notebook_id` and
any existing cells are preserved, so re-running it never orphans artifacts.

## Inspecting a notebook (`cell`, `dag`, `status`)

For agents (and humans) that need to read a notebook's state without executing
it, three read-only commands print structured JSON (default) or a compact human
view. They open the notebook locally — no server, no env sync:

```bash
strata cell list <notebook_dir>            # every cell: id, name, status, source
strata cell show <notebook_dir> <cell_id>  # one cell: source, status, outputs, console, staleness
strata dag       <notebook_dir>            # dependency edges + topological order
strata status    <notebook_dir>            # per-cell status + staleness summary
```

Each takes `--format human|json` (JSON is the default — these are agent-first).
The JSON shapes match the server's REST API (`GET /{id}/cells`, `GET /{id}/dag`),
so a script written against the local CLI keeps working against a running
server later. Exit codes follow the same contract as `run` / `validate`: `0`
success, `1` operation failure (e.g. unknown cell — a structured `{"error": …}`
on stdout), `2` invocation error (bad path) on stderr.

```bash
strata cell list my_analysis | jq '.cells[] | select(.status == "error") | .id'
```

### Reading a live session (`--server` / `--session`)

The four read commands also inspect a session on a **running** `strata-notebook`
— the same session a human is watching in the TUI or web UI — instead of a local
directory. Pass `--server <url> --session <id>` in place of the notebook path:

```bash
strata cell list --server http://localhost:8765 --session $SID
strata cell show --server http://localhost:8765 --session $SID <cell_id>
strata dag       --server http://localhost:8765 --session $SID
strata status    --server http://localhost:8765 --session $SID
```

`<id>` is the session id (the route `{id}`, not the `notebook.toml` id). The
output is identical to the local backend — both project the same wire shape — so
a script written against a local notebook works unchanged against a live session.
`cell run` / `cell test` and the authoring commands accept the same selector
(see below), so the entire surface works against a live session. Remote
operations target a personal-mode server (the use case is driving the session
you're watching locally).

## Running a cell or its tests (`cell run`, `cell test`)

Beyond inspection, an agent can execute one cell at a time (not the whole
notebook) and run a cell's unit tests, both with structured output:

```bash
strata cell run  <notebook_dir> <cell_id> [--rerun | --force] [--no-sync]
strata cell test <notebook_dir> <cell_id> [--no-sync]
```

`cell run` materializes the cell (using the cache and re-running stale upstreams
by default; `--rerun` bypasses the target's cache, `--force` runs against
whatever upstream artifacts already exist). `cell test` runs the cell's
`cells/{cell_id}.test.py` via pytest and reports per-test outcomes. Both **sync
the venv first** (like `strata run`) unless you pass `--no-sync`.

Both also accept `--server <url> --session <id>` to run on a live session instead
of a local directory (`strata cell run --server http://localhost:8765 --session
$SID <cell_id> --rerun`). The server owns its venv, so the remote path never syncs
— `--no-sync` is a local-only flag — and `--rerun` / `--force` map to the
server's run modes, so remote execution has the same three modes as local.

`--format json` (default) writes a single clean JSON object to **stdout** — the
executor's logs go to stderr, so the stdout stream stays parseable:

```bash
strata cell run nb featurize --format json | jq '{status, cache_hit, error}'
strata cell test nb featurize --format json | jq '.cases[] | select(.outcome != "passed")'
```

Exit codes: `cell run` → `0` ran ok, `1` the cell errored (or unknown cell), `2`
setup error (no venv under `--no-sync`, sync failure). `cell test` → `0` all
passed, `1` a test failed/errored, `2` pytest unavailable in the venv.

## Authoring cells and dependencies (`cell add/edit/rm/mv`, `dep`)

An agent can also build and edit a notebook through commands (instead of writing
`notebook.toml` + `cells/*.py` by hand):

```bash
strata cell add  <notebook_dir> --file body.py [--after <id>] [--language python|markdown|sql|r|prompt]
strata cell edit <notebook_dir> <cell_id> --file body.py     # replace a cell's source
strata cell rm   <notebook_dir> <cell_id>                     # delete a cell
strata cell mv   <notebook_dir> <cell_id> --to <index>        # reorder (0-based)
strata dep add   <notebook_dir> <package>                     # uv add
strata dep rm    <notebook_dir> <package>                     # uv remove
```

`--file -` reads cell source from stdin. `cell add` mints a backend-style 8-char
id (the same scheme the server uses) and prints the new cell; `mv` prints the new
order. Dependency commands run `uv add` / `uv remove` and report whether the
lockfile changed. All take `--format json|human` and use the shared exit codes
(`0` ok, `1` operation failure such as an unknown cell or a failed `uv` resolve,
`2` invocation error).

```bash
echo 'total = sum(nums)' | strata cell add nb --file - --after load --format json | jq .id
```

These also accept `--server <url> --session <id>` to author into a live session
instead of a local directory — edits land in the running notebook a human is
watching in the TUI. (`cell add` makes two calls: it mints the cell on the
server, then sets its source.)

Together with inspect (`cell list/show`, `dag`, `status`) and execution (`cell
run/test`), this is the full agent surface — and **all of it** works either
offline against a directory or `--server/--session` against a running server. An
MCP server wrapping the same operations lands in a later release.
