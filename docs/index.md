# Strata

**Strata is a notebook your coding agent can drive, and everything it computes stays.**

![The Strata notebook UI: two named Python cells, load and summarize, each with
its source and an interactive table of results, the sidebar panels for mounts,
workers and environment, and a bottom drawer showing the DAG of load, summarize
and plot with per-cell timings.](assets/notebook-anatomy-light.png#only-light)
![The Strata notebook UI: two named Python cells, load and summarize, each with
its source and an interactive table of results, the sidebar panels for mounts,
workers and environment, and a bottom drawer showing the DAG of load, summarize
and plot with per-cell timings.](assets/notebook-anatomy-dark.png#only-dark)

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

The interactive notebook surface: Python, R, SQL, prompt, widget and loop cells, each
producing artifacts that flow through an auto-built DAG.

**Highlights:**

**For an agent**

- **a scratchpad that lasts:** a Claude Code plugin points an agent at a cached notebook cell instead of a `/tmp` script, so the snippet it ran ten turns ago is still there
- **any MCP client:** a warm notebook session is exposed at `/mcp`, and you can watch the agent work in the browser or the terminal while it does
- **built to be scripted:** `strata` inspects, runs and authors a notebook with JSON output and stable exit codes, offline or against a live session
- **one command to start:** `strata agent ./nb` stands up the server, the session, the config and the viewer together
- **you can tell who did what:** every cell records whether a person or an agent wrote it

**Why nothing runs twice**

- **results are keyed by what made them:** a cell's source, its inputs and its environment decide its identity, so identical work is a cache hit and no result ever needs invalidating
- **only what changed re-runs:** edit one cell and Strata re-runs the cells below it and nothing else
- **no wiring:** Strata reads each cell to find what it uses and what it defines, so there are no decorators and no edges to maintain by hand
- **readable in git:** a notebook is plain `.py` files plus a TOML manifest, so a diff shows the code that changed
- **the graph is visible:** the dependency graph renders beside the cells, and double-clicking a node jumps to its source

**Cell types**

- **prompt cells:** LLM calls as DAG nodes, `{{ variable }}` interpolation from upstream, cached by template + inputs + model config
- **SQL cells:** named connections and bind-parameter templating, with drivers for DuckDB, SQLite, Postgres, Snowflake and BigQuery
- **R cells:** Python and R share one DAG, exchanging Arrow, so a `pandas.DataFrame` is a `data.frame` for the next cell
- **loop cells:** `# @loop max_iter=N carry=state` iterates with an explicit carry, each step its own artifact
- **variant sweeps:** run every variant of a group and hand the downstream cell a `{name: value}` dict, or fan out with `# @per_variant` so adding a variant only runs the new one
- **widget cells:** declarative controls that downstream cells consume as inputs; with Live on, dragging one recomputes what depends on it
- **cell unit tests:** a Tests panel runs real pytest against a cell's defs and upstream inputs, doubling as a health badge

**Compute and data**

- **distributed:** `# @worker gpu-fly` dispatches a cell to a remote box; bring your own compute
- **worker pool:** a separate package (`pip install strata-pool`): machines (Fly, Docker or RunPod) start on demand, are held for a tenant, hand out GPUs per cell, and stop when the work does
- **the same environment, elsewhere:** a remote cell runs in the notebook's own locked environment rather than whatever the worker image holds
- **the machine starts before the run:** focusing a cell bound to a pooled worker, or starting Run All, asks the pool to bring up a machine while you are still reading the code
- **remote cells over SSH:** hand an agent an SSH target and it provisions a worker there, tunnels to it, and routes heavy cells over, cached like everything else
- **workers behind NAT:** `strata-worker --connect` dials out to a relay, so a machine with no open port serves cells
- **lake-aware SQL:** read a named catalog and the notebook's mounts in one query, pinned to the snapshot the cell's provenance records
- **Iceberg as other engines write it:** scans apply merge-on-read deletes and follow schema changes by field id, so tables Spark, Flink and DuckDB write read correctly, from SQL, REST and Glue catalogs, with warehouses on S3, GCS or Azure
- **every input recorded:** `# @fetch` makes bytes from a URL an input, `# @dataset` makes a registry name one, both content-addressed
- **folders as outputs:** `# @output_dir ckpt` gives a cell a folder whose files are stored as an output, kept even when the run fails, and handed to the next cell as a `Path`
- **write back to the lake:** `# @write_table readings lake:lab.readings append` writes a variable into an Iceberg table when the run succeeds, and each snapshot names the result that made it
- **an append keeps the cache:** cached row groups are keyed by data file, not snapshot, so after an append a scan reads only the new files
- **mounts:** `# @mount data s3://bucket/prefix ro` makes any S3, GCS or Azure prefix a local `pathlib.Path`
- **one mount cache per machine:** every notebook shares one bounded copy of each remote file, so a prefix with one new file downloads only that file
- **isolated envs:** every notebook gets its own uv-managed `.venv/`, locked and reproducible
- **headless:** `strata run ./my-notebook` for CI and scheduled execution, same DAG and same cache

**Teams**

- **promote to the team:** `strata artifact promote` copies a result and the chain behind it into the team's store, from the notebook, the CLI or inside a cell
- **a colleague's run counts:** a cell can be served from a teammate's earlier result: the expensive middle of a pipeline, not only what someone chose to name
- **a figure gets a URL:** publish an artifact and anyone with the link sees the plot, the code, and the environment of every step behind it, with no account and no install
- **a notebook travels:** export the whole state as one bundle and import it back, artifacts included
- **share a session:** presence, cell focus and soft locks, with every cell recording who wrote it
- **safe to share in service mode:** cells run as their own OS user once you name one, notebook scopes are enforced on every notebook route under principal auth, and an MCP tool call runs as its caller
- **host it for many people:** in service mode every notebook route, the WebSocket and MCP stay within the caller's tenant, the server runs under a path behind your proxy, and the editor can be framed inside your own product
- **a store that stays bounded:** results nobody has used for a while are collected least recently used first, never anything named, pinned, published or still needed

**Surfaces**

- **terminal viewer:** `strata watch` renders a running notebook live in the terminal, read-only, over SSH or beside your editor
- **stop a run from anywhere:** cancel a running cell from the browser, `strata cell cancel`, REST or MCP, and on Linux and macOS a cell stops when the server that started it does
- **app view:** open a notebook as a read-only app, embed it as an `<iframe>`, or export a self-contained snapshot
- **data viewer:** DataFrame outputs render in a grid you can page, sort, filter and search, backed by the full artifact rather than a preview
- **also a library:** `pip install strata-client` talks to the store from any pipeline or service, with no server install
- **production:** Iceberg-aware scans, trusted-proxy auth, multi-tenancy, and S3, GCS, Azure or local blob backends

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

![A terminal running strata CLI commands beside a browser on the same notebook, over seven moments: the empty notebook, a load cell arriving, the cell running, finishing after about two seconds, a summarize cell arriving that reads its variable, both cells green, and the load cell run again and served from the cache in milliseconds.](assets/web-agent-live-light.gif#only-light)
![A terminal running strata CLI commands beside a browser on the same notebook, over seven moments: the empty notebook, a load cell arriving, the cell running, finishing after about two seconds, a summarize cell arriving that reads its variable, both cells green, and the load cell run again and served from the cache in milliseconds.](assets/web-agent-live-dark.gif#only-dark)

Two of those front doors at once: a coding agent adds and runs cells through
the `strata` CLI while the web UI on the same session updates live. Captured
from a real run rather than assembled from canned frames.

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

=== "uv (recommended)"

    ```bash
    uv tool install "strata-notebook[mcp,tui]"
    strata-notebook
    ```

    Then open [http://localhost:8765](http://localhost:8765). The `mcp`
    and `tui` extras are what `strata agent` and `strata watch` need.
    Plain `pip install` is not supported.

=== "Docker"

    ```bash
    docker compose up -d --build
    ```

    Then open [http://localhost:8765](http://localhost:8765).

    Or run the prebuilt image (linux/amd64), published with each release:

    ```bash
    docker run --rm -p 127.0.0.1:8765:8765 -v strata_state:/home/strata/.strata \
      ghcr.io/bearing-research/strata:0.10.0
    ```

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
