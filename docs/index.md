---
hide:
  - navigation
  - toc
---

# Strata { .landing-title }

**Strata is a notebook your coding agent can drive, and everything it computes stays.**

![A terminal running strata CLI commands beside a browser on the same notebook, over seven moments: the empty notebook, a load cell arriving, the cell running, finishing after about two seconds, a summarize cell arriving that reads its variable, both cells green, and the load cell run again and served from the cache in milliseconds.](assets/web-agent-live-light.gif#only-light){ .hero }
![A terminal running strata CLI commands beside a browser on the same notebook, over seven moments: the empty notebook, a load cell arriving, the cell running, finishing after about two seconds, a summarize cell arriving that reads its variable, both cells green, and the load cell run again and served from the cache in milliseconds.](assets/web-agent-live-dark.gif#only-dark){ .hero }

*An agent adds and runs cells through the `strata` CLI while the browser on the same
session updates. Captured from a real run, not a mock-up.*

<div class="grid cards" markdown>

-   **Coding agent**

    ---

    Hand a notebook to Claude Code and watch it build through MCP tools.

    [Use a coding agent →](notebook/agent.md)

-   **Web UI**

    ---

    Write and run cells yourself, with rendered tables, plots and the DAG.

    [Your first notebook →](getting-started/notebook.md)

-   **Terminal**

    ---

    Watch a notebook live from a second pane or an SSH session. Read-only.

    [Terminal viewer →](notebook/tui.md)

</div>

## Install

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
      ghcr.io/bearing-research/strata:0.11.0
    ```

=== "From source"

    ```bash
    uv sync
    cd frontend && npm ci && npm run build && cd ..
    uv run strata-notebook
    ```

    Then open [http://localhost:8765](http://localhost:8765).

See [Install](getting-started/installation.md) for full details.

## Why the work holds up

**[Nothing runs twice.](concepts/caching.md)** A cell's identity is its source, its
inputs and its environment. The step an agent ran ten turns ago is a cache hit now, and
an edit re-runs only the cells below it.

![A browser on the tutorial notebook over five moments: nothing has run; Run All computes every cell; Run All again and every code cell is a cache hit showing the time it saved; one cell is edited, so it and the chart below read stale while the load cell stays cached; Run All again, and only those two run while the load cell still reads cached, saved 2.1s.](assets/cache-payoff-light.gif#only-light)
![A browser on the tutorial notebook over five moments: nothing has run; Run All computes every cell; Run All again and every code cell is a cache hit showing the time it saved; one cell is edited, so it and the chart below read stale while the load cell stays cached; Run All again, and only those two run while the load cell still reads cached, saved 2.1s.](assets/cache-payoff-dark.gif#only-dark)

*Edit one cell in `strata tutorial`: only the cells that read it run again, and the
cell above keeps what it saved. Captured from a real run, not a mock-up.*

**[No wiring to maintain.](concepts/dag-and-staleness.md)** Strata reads each cell to
find what it uses and what it defines, and shows which results an edit has left stale.

**[Every result knows where it came from.](concepts/artifacts-and-lineage.md)** Every
output is a versioned artifact, so you can trace it back through what produced it.

**[It reads like code in git.](concepts/file-format-and-git.md)** A notebook is plain
`.py` files plus a TOML manifest. Outputs and caches stay in a gitignored folder.

There is more on the [Features](concepts/features.md) page, and Strata also works
[as a library](getting-started/core.md).

Strata is pre-1.0, so the API may change between 0.x minors; pin to a minor if you need
stability.
