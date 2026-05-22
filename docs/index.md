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

**Highlights:**

- **content-addressed:** every cell output is keyed by source + inputs + environment — identical work hits the cache forever
- **reactive:** edit a cell, the cascade re-runs only the downstream cells that depend on it
- **dag-from-ast:** Strata reads each cell's AST to wire upstream/downstream — no decorators, no manual edges
- **git-friendly:** notebooks are plain `.py` files plus a TOML manifest — readable diffs, no JSON blobs
- **prompt cells:** LLM calls are first-class DAG nodes, cached by template + inputs + model config
- **SQL cells:** named connections, bind-parameter templating, drivers for DuckDB / SQLite / Postgres / Snowflake / BigQuery
- **distributed:** `# @worker gpu-fly` dispatches a single cell to a remote box — bring your own compute
- **mounts:** `# @mount data s3://bucket/prefix ro` makes any S3 / GCS / Azure prefix a local `pathlib.Path`
- **isolated envs:** every notebook gets its own uv-managed `.venv/`, locked and reproducible
- **headless:** `strata run ./my-notebook` for CI and scheduled execution — same DAG, same cache

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

Strata is at **0.1.0** — the first stable release. Both surfaces
(Notebook and Core) are functional and shipped from PyPI. The API
may still change between 0.x minors before 1.0; pin to a minor if
you need stability.
