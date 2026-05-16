# Strata Examples

Two kinds of examples live here:

- **Notebooks** — directories containing a `notebook.toml` + Python/SQL/prompt
  cells. Open these with the Strata Notebook UI (`strata notebook` server)
  or run them headlessly with `strata run`.
- **SDK scripts** — standalone `*.py` files that talk to a running Strata
  server through `StrataClient`. Start the server first, then run them
  with `uv run` or `python`.

If you're not sure where to start, **the notebooks are the main UX surface
of Strata**. The SDK scripts are for users who want to query Iceberg tables
from their own Python programs.

---

## Notebooks

Open with the Notebook UI:

```bash
uv run python -m strata               # starts the server at http://127.0.0.1:8765
# then in the UI: File → Open → pick a directory below
```

Or run headlessly:

```bash
strata run examples/iris_classification
```

### Where to start

| Notebook | What it shows |
|---|---|
| [`pandas_basics/`](pandas_basics/) | Linear Pandas pipeline — load, select, group, summarize. Smallest end-to-end notebook. |
| [`iris_classification/`](iris_classification/) | sklearn classifier on the Iris dataset. Shows how artifacts flow between cells. |
| [`titanic_ml/`](titanic_ml/) | Survival prediction end-to-end: load → feature-engineer → train → score. |

### Feature showcases

| Notebook | What it shows |
|---|---|
| [`markdown_showcase/`](markdown_showcase/) | Markdown cells, prose-and-code interleaving. |
| [`library_cells/`](library_cells/) | A cell exports `def`s/`class`es as a shared library across the notebook. |
| [`model_variants/`](model_variants/) | `# @variant` annotation — A/B comparison of two model configurations. |
| [`loop_hill_climb/`](loop_hill_climb/) | `# @loop` annotation — iterative refinement with carried state. |
| [`s3_mount/`](s3_mount/) | `# @mount` annotation — read data from an S3 bucket as a local `Path`. |
| [`sql_orders_report/`](sql_orders_report/) | SQL cells with DuckDB; SQL and Python interleave through the same DAG. |
| [`review_triage/`](review_triage/) | Prompt cell with `@output_schema` — structured LLM output validated against JSON Schema. |

### Larger applied examples

| Notebook | What it shows |
|---|---|
| [`news_alpha_trader/`](news_alpha_trader/) | Multi-cell finance pipeline — news → sentiment → signals → trades. |
| [`arxiv_classifier/`](arxiv_classifier/) | Distributed embedding + clustering over arXiv abstracts. Larger DAG. |

---

## SDK scripts

These talk to a running Strata server through `StrataClient`. Start the
server first:

```bash
uv run python -m strata
```

Then run any of the scripts below:

```bash
uv run python examples/01_basic_usage.py
```

### Core usage

| File | Description |
|---|---|
| [01_basic_usage.py](01_basic_usage.py) | Connect to Strata and scan a table |
| [02_column_projection.py](02_column_projection.py) | Select specific columns to reduce data transfer |
| [03_filtering.py](03_filtering.py) | Predicates for row-group pruning |
| [04_time_travel.py](04_time_travel.py) | Query historical Iceberg snapshots |

### Integrations

| File | Description |
|---|---|
| [05_duckdb_integration.py](05_duckdb_integration.py) | SQL over Strata-served tables with DuckDB |
| [08_polars_integration.py](08_polars_integration.py) | Zero-copy Arrow → Polars DataFrames |
| [09_s3_storage.py](09_s3_storage.py) | Iceberg tables backed by S3 |

### Advanced features

| File | Description |
|---|---|
| [06_cache_management.py](06_cache_management.py) | Monitor and manage the Strata cache |
| [07_error_handling.py](07_error_handling.py) | Handle common errors gracefully |
| [10_artifacts.py](10_artifacts.py) | Materialize, chain, and track transform artifacts |
| [11_async_client.py](11_async_client.py) | Non-blocking async operations for high throughput |
| [12_delibera_integration.py](12_delibera_integration.py) | Direct artifact upload via `put()` for non-Strata producers |

### Demo helpers

| File | Description |
|---|---|
| [setup_demo.py](setup_demo.py) | Create a demo Iceberg table for testing |
| [hello_world.py](hello_world.py) | Benchmark cold/warm/restart performance |

---

## SDK quick start

```python
from strata.client import StrataClient, gt

client = StrataClient(base_url="http://127.0.0.1:8765")

batches = list(client.scan(
    "file:///warehouse#db.events",
    columns=["id", "value", "timestamp"],
    filters=[gt("timestamp", 1704067200000000)]
))

import pyarrow as pa
df = pa.Table.from_batches(batches).to_pandas()
print(df.head())

client.close()
```

### Async

```python
import asyncio
from strata.client import AsyncStrataClient, gt

async def main():
    async with AsyncStrataClient() as client:
        table = await client.scan_to_table(
            "file:///warehouse#db.events",
            columns=["id", "value"],
            filters=[gt("value", 100.0)],
        )
        print(f"Got {table.num_rows} rows")

asyncio.run(main())
```

### Artifact workflow

```python
from strata.client import StrataClient

with StrataClient(base_url="http://127.0.0.1:8765") as client:
    artifact = client.materialize(
        inputs=["file:///warehouse#db.events"],
        transform={
            "ref": "duckdb_sql@v1",
            "params": {"sql": "SELECT category, COUNT(*) FROM input0 GROUP BY 1"}
        },
        name="daily_summary",
    )
    print(f"Artifact URI: {artifact.uri}")
    print(f"Cache hit: {artifact.cache_hit}")
    print(artifact.to_pandas())
```
