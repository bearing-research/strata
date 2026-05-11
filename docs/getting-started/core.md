# Core API Quickstart

Strata Core is the programmatic materialization and artifact layer. Use this when you want the `materialize()` API, artifact caching, lineage tracking, or snapshot-aware Iceberg table scanning — without the notebook UI on top.

## The primitive

```
materialize(inputs, transform) → artifact
```

This gives you:

- Immutable, versioned artifacts
- Provenance-based deduplication (same inputs + transform = cache hit)
- Explicit lineage
- Safe reuse across runs and processes

Reading from an Iceberg table is itself a `materialize` call with the built-in `scan@v1` transform: inputs are the table URIs, params hold optional projections and filters. The cache key includes the table's snapshot ID, so once you've scanned a snapshot the result is reusable forever — there's no invalidation problem.

## 1. Start the server

```bash
uv run strata-server
```

## 2. Run the demo

```bash
uv run python examples/hello_world.py
```

This creates a local Iceberg table with 100K rows and times three reads against it:

```
Cold run     (no cache)             ~500ms   — read Parquet, cache as Arrow IPC
Warm run     (in-memory cache hit)  ~50ms    — serve from process memory
Restart run  (disk cache hit)       ~60ms    — serve from on-disk Arrow IPC
```

Same inputs, same transform, three different cache states — and the third is still ~10× faster than the first because the disk cache survives restarts.

## 3. Materialize a result

```python
from strata.client import StrataClient

client = StrataClient()

artifact = client.materialize(
    inputs=["file:///warehouse#db.events"],
    transform={
        "executor": "scan@v1",
        "params": {
            "columns": ["id", "value"],
            "filters": [{"column": "value", "op": ">", "value": 100}],
        },
    },
)

print(f"URI: {artifact.uri}")
print(f"Cache hit: {artifact.cache_hit}")
```

## 4. Fetch the result

```python
table = client.fetch(artifact.uri)
df = table.to_pandas()
```

## 5. Integration with data libraries

=== "Pandas"

    ```python
    from strata.integration.pandas import fetch_to_pandas
    df = fetch_to_pandas("file:///warehouse#db.events")
    ```

=== "Polars"

    ```python
    from strata.integration.polars import fetch_to_polars
    df = fetch_to_polars("file:///warehouse#db.events")
    ```

=== "DuckDB"

    ```python
    from strata.integration.duckdb import StrataScanner
    with StrataScanner() as scanner:
        scanner.register("events", "file:///warehouse#db.events")
        result = scanner.query("SELECT category, COUNT(*) FROM events GROUP BY category")
        print(result.to_pandas())
    ```

## Core behaviors

- Same inputs + transform → existing artifact, no recomputation
- Artifacts are immutable and versioned
- Names are mutable pointers to specific artifact versions
- Provenance hash is derived from pinned inputs and transform identity

## What's next

- [Configuration](../reference/configuration.md) — all environment variables (cache, fetcher, S3 / GCS / Azure, auth, timeouts)
- [Deployment Modes](../deployment/modes.md) — `personal` vs `service` mode and the auth boundary
- [REST API](../reference/rest-api.md) — notebook protocol surface (separate from the `/v1/materialize` endpoint this page calls into)
