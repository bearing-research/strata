# Headless Runner

`strata run` executes every cell in a notebook directory without starting the
server or opening the UI. Useful for CI, scheduled jobs, and sanity-checking
that a notebook still works after a dependency bump.

For rendering a notebook to a shareable markdown or HTML file (without
executing anything), see [Export](export.md) and its `strata export`
sibling command.

It reuses the same `NotebookSession` and `CellExecutor` the UI uses, so the
execution path is identical — artifact cache hits, cascade ordering, worker
dispatch, and mount resolution all behave the same way.

## Usage

```bash
strata run <notebook_dir> [options]
```

### Options

| Flag          | Description                                                      |
| ------------- | ---------------------------------------------------------------- |
| `--force`     | Ignore the artifact cache and re-execute every cell from scratch |
| `--no-sync`   | Skip `uv sync`; require `.venv/` to already exist                |
| `--format`    | `human` (default) or `json`                                      |
| `--quiet`     | Suppress per-cell status lines (human format only)               |

### Exit Codes

| Code | Meaning                                                        |
| ---- | -------------------------------------------------------------- |
| `0`  | All cells succeeded                                            |
| `1`  | One or more cells failed                                       |
| `2`  | Invocation error (bad path, env sync failure, malformed TOML)  |

## Example: GitHub Actions

```yaml
jobs:
  notebooks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@v8.1.0
      - name: Install Strata
        run: uv pip install --system strata-notebook
      - name: Run notebook
        run: strata run ./notebooks/daily_report --format json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

`--format json` emits one JSON object per cell plus a summary record at the
end, so downstream steps can grep/parse per-cell status without screen-scraping.

## What Gets Run

Every cell in the notebook executes in topological order, exactly as if you
had clicked **Run All** in the UI. Cached artifacts are reused — the first run
populates the cache; subsequent runs on unchanged source + inputs return
instantly.

Passing `--force` invalidates the cache and forces a full rebuild, which is
what you usually want in CI if you're testing that the code *still produces*
the expected artifacts.

## What Does Not Happen

- **No server starts.** No ports are bound; no UI is served.
- **No WebSocket broadcasts.** Progress is written to stdout only.
- **No interactive prompts.** A cascade that would pop a confirmation in the
  UI just runs — the CLI treats every cell as "confirmed."
- **No AI assistant.** `strata run` only executes declarative cells.

## Environment & Secrets

`strata run` reads the notebook's `[env]` and `[secret_manager]` blocks the
same way the server does. Secret-manager credentials (e.g.
`INFISICAL_CLIENT_ID` / `INFISICAL_CLIENT_SECRET`) must be present in the
shell that invokes the command — they are never stored in the notebook.

For notebooks that require env vars set only via the Runtime panel (never
committed), export them before invoking `strata run`.
