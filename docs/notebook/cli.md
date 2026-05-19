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
not talk to a running `strata-server` or a service-mode deployment. Each
invocation opens its own `NotebookSession`, runs the cells, and exits.

### Options

| Flag          | Description                                                      |
| ------------- | ---------------------------------------------------------------- |
| `--force`     | Ignore the artifact cache and re-execute every cell from scratch |
| `--no-sync`   | Skip `uv sync`; require `.venv/` to already exist                |
| `--format`    | `human` (default) or `json` — both write to stdout               |
| `--quiet`     | Suppress per-cell status lines (human format only)               |

`--format json` writes one JSON object per cell as it finishes, followed by a
final summary record, all to **stdout**. Errors and pre-flight diagnostics go
to stderr regardless of format. Pipe stdout to `jq` for filtering, or capture
to a file (`strata run ... --format json > run.jsonl`) for later parsing.

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
        # Strata refuses to start outside a uv-managed env, so install
        # as a uv tool (creates a uv-marked venv at ~/.local/share/uv/
        # tools/strata-notebook and adds the CLI to PATH).
        run: uv tool install strata-notebook
      - name: Run notebook
        run: strata run ./notebooks/daily_report --format json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

`--format json` emits one JSON object per cell plus a summary record at the
end, so downstream steps can grep/parse per-cell status without screen-scraping.

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
