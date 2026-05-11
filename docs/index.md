# Strata

**Notebooks for long-running ML, AI, and data work — cached, distributed, and git-friendly.**

Strata is designed for work that runs longer than a single keystroke. Every
cell output becomes a content-addressed artifact, so re-runs are cache hits
when nothing's changed and the DAG cascade re-executes only what did. Prompt
cells make AI calls first-class participants; `# @worker gpu-fly` dispatches
a cell to a remote GPU; and the whole notebook is plain `.py` files plus a
manifest — git-diffable, no JSON blobs.

---

## Strata Notebook

The interactive notebook surface: Python, prompt, SQL, and loop cells, each
producing artifacts that flow through an auto-built DAG.

**Key features:**

- Content-addressed caching (same code + inputs = cache hit)
- Automatic DAG from variable analysis
- Git-friendly format — cells are plain `.py` files, outputs and runtime
  state live outside the committed tree (no JSON blobs, no diffs on every run)
- Distributed workers (`@worker gpu-fly` dispatches to remote GPU)
- Prompt cells with `{{ variable }}` injection into AI calls
- AI assistant with streaming chat and agent mode
- Per-notebook Python environments via uv
- Headless runner (`strata run`) for CI

[:octicons-arrow-right-24: Notebook Quickstart](getting-started/notebook.md){ .md-button .md-button--primary }

---

## Strata Core

The notebook is built on Strata Core — a standalone materialization
and artifact layer. Core can also be used independently as a Python
client library and REST API for any workflow that needs provenance-based
caching, lineage tracking, or Iceberg table scanning.

```python
from strata import StrataClient

client = StrataClient()
artifact = client.materialize(
    inputs=["file:///warehouse#db.events"],
    transform={"executor": "scan@v1", "params": {}},
)
table = client.fetch(artifact.uri)
```

[:octicons-arrow-right-24: Core API Quickstart](getting-started/core.md){ .md-button }

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
