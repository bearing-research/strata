# Strata

**Strata is a content-addressed computation graph with an interactive notebook UI.**

Every cell output is a versioned artifact keyed by its provenance: source,
inputs, and environment. Strata reads each cell's AST to build the
dependency graph automatically, so re-running a notebook is mostly a series
of cache hits. Prompt cells make AI calls first-class DAG nodes, cached by
template, inputs, and model config. The `# @worker gpu-fly` annotation
dispatches a cell to a remote GPU. The whole notebook is plain `.py` files
plus a manifest, so commits are git-diffable and there are no JSON blobs
or execution metadata bleeding into the history.

---

## Strata Notebook

The interactive notebook surface: Python, prompt, SQL, and loop cells, each
producing artifacts that flow through an auto-built DAG.

**Key features:**

- Content-addressed caching (same code + inputs = cache hit)
- Automatic DAG from variable analysis
- Git-friendly format: cells are plain `.py` files, outputs and runtime
  state live outside the committed tree (no JSON blobs, no diffs on every run)
- Distributed workers (`@worker gpu-fly` dispatches to remote GPU)
- Prompt cells with `{{ variable }}` injection into AI calls
- AI assistant with streaming chat and agent mode
- Per-notebook Python environments via uv
- Headless runner (`strata run`) for CI

[:octicons-arrow-right-24: Notebook Quickstart](getting-started/notebook.md){ .md-button .md-button--primary }

---

## Use Strata as a library

Strata's HTTP API exposes the materialization layer directly,
driveable from Python via `StrataClient`. Useful for direct table
scans, custom transforms, and headless workflows; the notebook
executor is a separate pipeline that writes to the same artifact
store. The client talks to a running Strata server, so this workflow
has two steps: start the server, then call it from your code.

```python
# Prereq: `uv run strata-server` running in another terminal.
from strata import StrataClient

client = StrataClient(base_url="http://localhost:8765")
artifact = client.materialize(
    inputs=["file:///warehouse#db.events"],
    transform={"executor": "scan@v1", "params": {}},
)
table = client.fetch(artifact.uri)
```

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
    STRATA_DEPLOYMENT_MODE=personal uv run strata-server
    ```

    Then open [http://localhost:8765](http://localhost:8765).

See [Installation](getting-started/installation.md) for full details.

## Status

Strata is currently in **alpha**. Both surfaces (Notebook and Core) are
functional but the API may change before 1.0.
