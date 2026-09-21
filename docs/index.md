# Strata

**Strata is a notebook your coding agent can drive, and everything it computes stays.**

![The Strata notebook UI: a Python cell with its source and an interactive table
of results, the sidebar panels for mounts, workers and environment, and a bottom
drawer showing the cell DAG and per-cell timings.](assets/notebook-anatomy-light.png#only-light)
![The Strata notebook UI: a Python cell with its source and an interactive table
of results, the sidebar panels for mounts, workers and environment, and a bottom
drawer showing the cell DAG and per-cell timings.](assets/notebook-anatomy-dark.png#only-dark)

Coding agents explore by writing throwaway scripts that nobody sees and
nothing remembers. Point one at Strata and the work lands in a notebook
instead: every result cached by what produced it, every cell recording who
wrote it, and the whole thing open in your browser while the agent works.

That holds because of what a cell is. Each one is keyed by its source, its
inputs and its environment, so the expensive step an agent ran ten turns ago
is a cache hit now, and the same inputs give the same result a year later on
another machine. Change one cell and only the cells below it run again.

It is a good notebook for a person, too. Ask a model a question in a prompt
cell and the answer is cached like any other result. Send the slow cell to a
GPU with one line. The notebook itself is plain `.py` files plus a manifest,
so a commit is a readable diff rather than a wall of JSON.

---

## Strata Notebook

The interactive notebook surface: Python, prompt, SQL, and loop cells, each
producing artifacts that flow through an auto-built DAG.

**Highlights:**

- **content-addressed:** every cell output is keyed by source + inputs + environment - identical work hits the cache forever
- **reactive:** edit a cell, the cascade re-runs only the downstream cells that depend on it
- **dag-from-ast:** Strata reads each cell's AST to wire upstream/downstream - no decorators, no manual edges
- **dag-view:** the dependency graph renders alongside the cells - double-click any node to jump to its source
- **git-friendly:** notebooks are plain `.py` files plus a TOML manifest - readable diffs, no JSON blobs
- **prompt cells:** LLM calls are first-class DAG nodes, `{{ variable }}` interpolation from upstream cells, cached by template + inputs + model config
- **SQL cells:** named connections, bind-parameter templating, drivers for DuckDB / SQLite / Postgres / Snowflake / BigQuery
- **loop cells:** `# @loop max_iter=N carry=state` iterates a cell with explicit carry between steps - each iteration is its own artifact
- **interactive widgets + app view:** a `widget` cell is a control panel (slider / dropdown / …) whose values downstream cells consume; with **⚡ Live** on, dragging a control recomputes the dependents. Open a notebook as a read-only **app**, **embed** it as an `<iframe>`, or export a frozen **snapshot**
- **distributed:** `# @worker gpu-fly` dispatches a single cell to a remote box - bring your own compute
- **mounts:** `# @mount data s3://bucket/prefix ro` makes any S3 / GCS / Azure prefix a local `pathlib.Path`
- **isolated envs:** every notebook gets its own uv-managed `.venv/`, locked and reproducible
- **auto-install:** missing import in a cell? one click adds the package via uv and re-runs
- **headless:** `strata run ./my-notebook` for CI and scheduled execution - same DAG, same cache
- **every input recorded:** `# @fetch` makes bytes from a URL an input and `# @dataset` makes a registry name one, both content-addressed like everything else
- **lake-aware SQL:** read a named catalog and the notebook's mounts in one query, pinned to the snapshot the cell's provenance records
- **the same environment, elsewhere:** a remote cell runs in the notebook's own locked environment rather than whatever the worker image happens to hold
- **a figure gets a URL:** publish an artifact and anyone with the link sees the plot, the code behind it, and the environment of every step - no account, no install
- **promote to the team:** copy a result and its whole chain into a shared store, from the notebook, the CLI or inside a cell - and a teammate's earlier run can serve your cell
- **a notebook travels:** export the whole state as one bundle, artifacts included, and import it back elsewhere
- **share a session:** presence, cell focus and soft locks, with every cell recording who wrote it
- **worker pool** (a separate package, `pip install strata-pool`)**:** machines start on demand - Fly, Docker or RunPod - are held for a tenant, hand out GPUs per cell, and stop when the work does

### Three ways to drive it

The same notebook - the same cells, the same cache, the same artifacts - has
several front doors. They operate on the same notebook directory, and because
opening a notebook twice reuses the same session rather than forking it, they
can all be pointed at it at once: a browser and a terminal viewer watching the
same cell run.

| Surface | For | Start here |
| --- | --- | --- |
| **Web UI** | Writing and running cells yourself, with rendered tables, plots, and the DAG view. | [Quickstart - Web UI](getting-started/notebook.md) |
| **Terminal (TUI)** | Watching a notebook live from a terminal - a second pane, an SSH session, or beside your editor. Read-only. | [Quickstart - Terminal](notebook/tui.md) |
| **Coding agent** | Handing the notebook to Claude Code and watching it build, through MCP tools. The notebook is what you wanted. | [Quickstart - Coding agent](notebook/agent.md) |

![The terminal viewer over five moments: an empty notebook, a cell arriving,
that cell finishing in 1.0s with its output, a second cell arriving that reads
the first one's variable, and both cells green.](assets/tui-agent-live.gif)

Two of those front doors at once: a coding agent adds and runs cells through
the CLI while `strata-notebook-tui` mirrors it live in another terminal.
Captured from a real run rather than assembled from canned frames.

There is also a use that is not a front door at all: an agent working on
something else entirely, using a notebook as a **cached scratchpad** for
throwaway Python instead of temp scripts. Nobody watches that one, which is
rather the point - see [Agent scratchpad](notebook/scratchpad.md).

[:octicons-arrow-right-24: Notebook Quickstart](getting-started/notebook.md){ .md-button .md-button--primary }

---

## Use Strata as a library

Strata's HTTP API exposes the materialization layer directly,
driveable from Python via `StrataClient`. Useful for direct table
scans, custom transforms, and headless workflows; the notebook
executor is a separate pipeline that writes to the same artifact
store. The client talks to a running Strata server.

[:octicons-arrow-right-24: Library Quickstart](getting-started/core.md){ .md-button }

---

## Quick Start

=== "Docker"

    ```bash
    docker compose up -d --build
    ```

    Then open [http://localhost:8765](http://localhost:8765).

=== "From source"

    ```bash
    uv sync
    cd frontend && npm ci && npm run build && cd ..
    uv run strata-notebook
    ```

    Then open [http://localhost:8765](http://localhost:8765).

See [Installation](getting-started/installation.md) for full details.

## Status

Both surfaces (Notebook and Core) are functional and shipped from
PyPI. Strata is still pre-1.0, so the API may change between 0.x
minors; pin to a minor if you need stability.
