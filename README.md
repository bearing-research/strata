# Strata

[![PyPI](https://img.shields.io/pypi/v/strata-notebook.svg)](https://pypi.org/project/strata-notebook/)
[![Python versions](https://img.shields.io/pypi/pyversions/strata-notebook.svg)](https://pypi.org/project/strata-notebook/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bearing-research/strata/blob/main/LICENSE)
[![CI](https://github.com/bearing-research/strata/actions/workflows/ci.yml/badge.svg)](https://github.com/bearing-research/strata/actions/workflows/ci.yml)

**Strata is a notebook your coding agent can drive, and everything it computes stays.**

Coding agents explore by writing throwaway scripts: `python -c`, files in
`/tmp`, gone by the next session. Point one at Strata and the work lands in a
notebook instead. Every result is cached by what produced it, every cell
records who wrote it, and you can watch the whole thing in your browser while
the agent works.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/web-agent-live-dark.gif">
  <img alt="A coding agent's session beside a browser on the same notebook, asked to plot CO₂ at Mauna Loa and how fast it is rising: it writes and runs a cell that fetches NOAA's monthly record; adds a growth cell that fails with a KeyError, red in the browser too; fixes the one line and reruns only that cell, which prints the rise per decade; adds the plot; and checks the status and sums up, with the fetch served from the cache. The agent's words are scripted; its commands, their output and the browser are real." src="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/web-agent-live-light.gif">
</picture>

*An agent adds and runs cells through the `strata` CLI while the browser on the
same session updates. Captured from a real run, not a mock-up.*

## Try it

You need [uv](https://docs.astral.sh/uv/) (or pipx, pip or conda; see
[Install](https://bearingresearch.com/strata/latest/getting-started/installation/)) and
[Claude Code](https://claude.com/claude-code) (or another MCP client).

```bash
uv tool install "strata-notebook[mcp,tui]"

strata agent ./my-notebook      # terminal 1: server, session and a live viewer
cd ./my-notebook && claude      # terminal 2: the agent drives the notebook
```

Ask the agent for an analysis and watch the cells arrive. Open
<http://localhost:8765> to follow along in the browser, or to take over and edit.

## Two ways an agent uses it

**The notebook is the deliverable.** `strata agent` scaffolds the notebook,
starts the server, and writes the `.mcp.json` and `CLAUDE.md` the agent needs.
You keep the result.
[Use a coding agent →](https://bearingresearch.com/strata/latest/notebook/agent/)

**The notebook is a scratchpad.** An agent working on something else runs its
throwaway Python in a cached cell instead of `python -c`, so the value from
ten turns ago is still there, in your project's environment. Wire it into
Claude Code or Codex once:

```bash
uv tool install "strata-notebook[mcp]"
strata agent install
```

Measured on a project whose loader takes 20 s, four questions in a row over its
data took Claude Code 0.42 of the median time they took without it on the tasks
the scratchpad was tuned on, and 0.67 on iteration tasks held out from tuning.
The data was loaded once in 21 of 21 tuned iteration and plot trials, against 3
of 21 without it (held out: 9 of 9, against 2 of 9).
[What it saves →](https://bearingresearch.com/strata/latest/notebook/scratchpad/#what-it-saves)

Other agents connect to the MCP endpoint at `/mcp` (on a server started with
`STRATA_MCP_ENABLED=true`, as `strata agent` does), or script the `strata` CLI,
which returns JSON and stable exit codes.

## Why the work holds up

**Nothing runs twice.** A cell's identity is its source, its inputs and its
environment. The step an agent ran ten turns ago is a cache hit now, and an
edit re-runs only the cells below it.

```
First run:       load (2.5s) → summarize (0.4s) → plot (0.9s)  = 3.7s
Edit summarize:  load (✓)    → summarize (0.4s) → plot (0.6s)  = 1.0s
Run again:       load (✓)    → summarize (✓)    → plot (✓)     = <0.1s
```

That is a real run of the tutorial below, whose load step waits two seconds on purpose.

**You can review what it did.** Every cell records whether a person or an
agent wrote it. Watch live in the browser or with `strata watch` in a terminal,
or read the directory afterwards.

**It reads like code in git.** A notebook is one plain-text file per cell
(`.py`, `.sql`, `.md`, ...) plus a TOML manifest. Outputs and caches stay in a gitignored `.strata/` folder.

```
my-notebook/
├── notebook.toml      # cells, order, workers, mounts
├── pyproject.toml     # this notebook's own uv environment
├── uv.lock
└── cells/
    ├── 3f9a1c2e.py    # one file per cell
    └── 8b07d4aa.py
```

**Heavy cells go where the compute is.** `# @worker gpu-box` sends one cell to
another machine in the notebook's locked environment. Hand an agent an SSH
target and it provisions the worker itself.

**Whether agents use it is measured.** `evals/agent_notebook/` scores 9 recorded
agent sessions in CI, plus a group that checks whether an un-primed session
reaches for the notebook at all.

## For people, too

Run `strata-notebook` and open <http://localhost:8765> for the full editor:
Python, SQL, R, prompt and widget cells in one dependency graph, with an
interactive data viewer and a read-only app view.
[Notebook quickstart →](https://bearingresearch.com/strata/latest/getting-started/notebook/)

New here? `strata tutorial caching` opens a two-minute tour of what the cache saves.
`strata tutorial` lists the samples that ship with Strata, and **Open a sample** on the
home page does the same.

Coming from Jupyter? `strata import notebook.ipynb` converts a notebook.
Like marimo and Pluto, Strata builds the graph from variable references; unlike
them, every cell is cached by default and a single cell can run on another
machine. [How Strata compares →](https://bearingresearch.com/strata/latest/notebook/comparison/)

## Examples

<table>
<tr>
<td width="33%" valign="top">
<a href="https://bearingresearch.com/strata/latest/examples/caching/"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/caching-dark.webp">
<img alt="A bar chart of revenue by region." src="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/caching-light.webp">
</picture></a>
<br><b><a href="https://bearingresearch.com/strata/latest/examples/caching/">caching</a></b><br>The two-minute tutorial: what the cache saves, and what an edit runs again
</td>
<td width="33%" valign="top">
<a href="https://bearingresearch.com/strata/latest/examples/keeling_curve/"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/keeling_curve-dark.webp">
<img alt="CO₂ at Mauna Loa rising from about 315 to 427 ppm since 1958, beside the average year's cycle." src="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/keeling_curve-light.webp">
</picture></a>
<br><b><a href="https://bearingresearch.com/strata/latest/examples/keeling_curve/">keeling_curve</a></b><br>NOAA's CO₂ record, and a published figure that shows its sources
</td>
<td width="33%" valign="top">
<a href="https://bearingresearch.com/strata/latest/examples/model_variants/"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/model_variants-dark.webp">
<img alt="A confusion matrix for the active logistic-regression variant." src="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/model_variants-light.webp">
</picture></a>
<br><b><a href="https://bearingresearch.com/strata/latest/examples/model_variants/">model_variants</a></b><br>Three classifiers in one DAG slot; switch and only downstream reruns
</td>
</tr>
<tr>
<td width="33%" valign="top">
<a href="https://bearingresearch.com/strata/latest/examples/review_triage/"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/review_triage-dark.webp">
<img alt="A structured JSON verdict from a prompt cell." src="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/review_triage-light.webp">
</picture></a>
<br><b><a href="https://bearingresearch.com/strata/latest/examples/review_triage/">review_triage</a></b><br>A prompt cell with schema-validated LLM output
</td>
<td width="33%" valign="top">
<a href="https://bearingresearch.com/strata/latest/examples/sql_orders_report/"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/sql_orders_report-dark.webp">
<img alt="The top five orders above $50 and revenue by product category." src="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/sql_orders_report-light.webp">
</picture></a>
<br><b><a href="https://bearingresearch.com/strata/latest/examples/sql_orders_report/">sql_orders_report</a></b><br>SQL, Python and markdown cells in one graph
</td>
<td width="33%" valign="top">
<a href="https://bearingresearch.com/strata/latest/examples/widget_playground/"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/widget_playground-dark.webp">
<img alt="The plot a slider and dropdown drive, titled with their values." src="https://raw.githubusercontent.com/bearing-research/strata/main/docs/assets/examples/widget_playground-light.webp">
</picture></a>
<br><b><a href="https://bearingresearch.com/strata/latest/examples/widget_playground/">widget_playground</a></b><br>Controls that drive downstream cells, live
</td>
</tr>
</table>

[All examples →](https://bearingresearch.com/strata/latest/notebook/examples/)

## What is still moving

Materialization, the artifact store, the DAG, caching and headless runs are
settled and covered by tests. Three surfaces are not:

- **Prompt-cell API:** streaming and structured-output validation will change.
- **SQL cloud drivers:** DuckDB, SQLite and PostgreSQL run in CI. BigQuery and
  Snowflake ship without integration coverage.
- **On-disk formats:** `notebook.toml`, `.strata/runtime.json` and the cache
  layout may change between minor versions.

## Learn more

- [Documentation](https://bearingresearch.com/strata/latest/), including the full feature list
- [Installation](https://bearingresearch.com/strata/latest/getting-started/installation/): uv, pipx, pip and conda, Docker, and source builds
- [Use Strata as a library](https://bearingresearch.com/strata/latest/getting-started/core/) with `strata-client`
- [Deployment](https://bearingresearch.com/strata/latest/deployment/modes/): service mode, multi-tenancy, workers
- [Issues](https://github.com/bearing-research/strata/issues) for bugs and questions

## Development

```bash
uv sync --all-extras                   # install dependencies (matches CI)
uv run pytest                          # run the tests
uv run pre-commit run --all-files      # lint and format
```

For the frontend with hot reload, run the server and Vite side by side; Vite
does not proxy to the backend:

```bash
STRATA_CORS_ALLOW_ORIGINS='["http://localhost:5173"]' uv run strata-notebook
cd frontend && npm ci && VITE_STRATA_URL=http://localhost:8765 npm run dev
```

## License

Apache 2.0
