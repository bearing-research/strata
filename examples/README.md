# Strata Examples

Two kinds of examples live here:

- **Notebooks:** directories holding a `notebook.toml` plus Python, SQL,
  R or prompt cells. Open them in the notebook UI
  (`strata-notebook --notebook-dir ./examples`) or run them headlessly
  with `strata run`.
- **SDK scripts:** standalone `*.py` files that talk to a running Strata
  server through `StrataClient`. Start the server first, then run them
  with `uv run` or `python`.

If you're not sure where to start, **the notebooks are the main UX surface
of Strata**. The SDK scripts are for users who want to query Iceberg tables
from their own Python programs.

---

## Notebooks

Open with the Notebook UI:

```bash
strata-notebook --notebook-dir ./examples   # every example appears on the home page at http://127.0.0.1:8765
```

Or run headlessly:

```bash
strata run examples/iris_classification
```

### Where to start

| Notebook | What it shows |
|---|---|
| [`pandas_basics/`](pandas_basics/) | Linear Pandas pipeline: load, select, group, summarize. Smallest end-to-end notebook. |
| [`iris_classification/`](iris_classification/) | sklearn classifier on the Iris dataset. Shows how artifacts flow between cells. |
| [`titanic_ml/`](titanic_ml/) | Survival prediction end-to-end: load, feature-engineer, train, score. |

### Feature showcases

| Notebook | What it shows |
|---|---|
| [`markdown_showcase/`](markdown_showcase/) | Markdown cells, prose-and-code interleaving. |
| [`data_viewer/`](data_viewer/) | Interactive DataFrame viewer: paging and click-to-sort over a 2,000-row frame. |
| [`widget_playground/`](widget_playground/) | `# widget` cell: a slider/number/dropdown control panel driving a DataFrame. |
| [`library_cells/`](library_cells/) | A cell exports `def`s/`class`es as a shared library across the notebook. |
| [`model_variants/`](model_variants/) | `# @variant` annotation: three classifiers sharing one DAG slot. |
| [`model_variants_sweep/`](model_variants_sweep/) | Sweep mode: every variant runs and one downstream cell compares them, plus a `# @per_variant` fan-out. |
| [`loop_hill_climb/`](loop_hill_climb/) | `# @loop` annotation: iterative refinement with carried state. |
| [`s3_mount/`](s3_mount/) | `# @mount` annotation: read data from an S3 bucket as a local `Path`. |
| [`sql_orders_report/`](sql_orders_report/) | SQL cells over a local SQLite file; SQL and Python interleave through the same DAG. Needs the `sql` and `sql-sqlite` extras. |
| [`review_triage/`](review_triage/) | Prompt cell with `@output_schema`: structured LLM output validated against JSON Schema. |
| [`r_lm_vs_sklearn/`](r_lm_vs_sklearn/) | R cells next to Python: fit `lm()` in R, the same model in sklearn, compare side-by-side over Arrow. |
| [`r_mtcars_analysis/`](r_mtcars_analysis/) | A pure-R notebook: `lm()`, `aggregate()` and inline ggplot2 and base-graphics plots. |
| [`agent_demo/`](agent_demo/) | The notebook a coding agent builds live in the `strata agent` demo: the training cell stays cached while only the evaluation re-runs. |

### Larger applied examples

| Notebook | What it shows |
|---|---|
| [`news_alpha_trader/`](news_alpha_trader/) | Multi-cell finance pipeline: news, sentiment, signals, trades. |
| [`arxiv_classifier/`](arxiv_classifier/) | Distributed embedding + clustering over arXiv abstracts. Larger DAG. |

---

## SDK scripts

These talk to a running Strata server through `StrataClient`. From a Strata
checkout, start the server, build the demo table the scripts read, then run a
script:

```bash
uv run strata-notebook
uv run python examples/setup_demo.py
uv run python examples/01_basic_usage.py
```

The scan scripts read `setup_demo.py`'s `analytics.events` table; pass another
table URI as the first argument to point one elsewhere (`04_time_travel.py`
reads the demo table's snapshots and takes no argument). Every script here runs
as is against the demo table except `09_s3_storage.py`, a template that needs an
Iceberg warehouse in S3 and your credentials filled in.

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
| [09_s3_storage.py](09_s3_storage.py) | Iceberg tables backed by S3 (template: needs an S3 warehouse and credentials) |

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
| [setup_demo.py](setup_demo.py) | Create the demo Iceberg table the scripts read (`demo-warehouse/`, two snapshots) |
| [hello_world.py](hello_world.py) | Time a cold scan against two artifact-cache hits |

---

## SDK quick start

```python
from strata_client import StrataClient

with StrataClient(base_url="http://127.0.0.1:8765") as client:
    artifact = client.materialize(
        inputs=["file:///warehouse#db.events"],
        transform={
            "executor": "scan@v1",
            "params": {
                "columns": ["id", "value", "timestamp"],
                "filters": [{"column": "timestamp", "op": ">", "value": 1704067200000000}],
            },
        },
    )
    print(client.fetch(artifact.uri).to_pandas().head())
```

### Async

```python
import asyncio
from strata_client import AsyncStrataClient

async def main():
    async with AsyncStrataClient() as client:
        artifact = await client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={
                "executor": "scan@v1",
                "params": {
                    "columns": ["id", "value"],
                    "filters": [{"column": "value", "op": ">", "value": 100.0}],
                },
            },
        )
        table = await client.fetch(artifact.uri)
        print(f"Got {table.num_rows} rows")

asyncio.run(main())
```

### Artifact workflow

```python
from strata_client import StrataClient

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
