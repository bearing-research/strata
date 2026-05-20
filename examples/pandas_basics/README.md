# Pandas Basics — the core DataFrame operations

A guided tour of the pandas operations you reach for every day, split
into one cell per concept. Running cell N re-uses the artifact from
cell N-1, so edits stay fast and the DAG stays honest.

## What it shows

- A linear chain where each cell reads the previous cell's output.
- Cache-hit behavior: re-running a cell after its upstream hasn't
  changed finishes in a few milliseconds.
- How Strata's **staleness propagation** works — edit cell 2 and cells
  3-7 turn yellow automatically.

## Cells

| Cell | What it does |
|---|---|
| `create_data` | Builds a small sales DataFrame as the root of the chain. |
| `select_filter` | Column selection + boolean indexing. |
| `add_columns` | Derived columns (e.g. `total = price * quantity`). |
| `groupby` | `groupby` + aggregate. |
| `pivot` | Pivot from long to wide. |
| `merge` | Join two DataFrames. |
| `summary` | Describe and basic stats. |

## Running

From the project root:

```bash
uv run strata-notebook --host 127.0.0.1 --port 8765
```

Then open `examples/pandas_basics` from the Strata home page.

## Try this

1. Run all cells top-to-bottom.
2. Edit `create_data` (for example, change a price).
3. Watch cells 2-7 turn stale automatically.
4. Run cell 7. Strata re-executes only the cells that need it — you
   should see cache hits reported for any intermediate cell whose
   inputs didn't actually change.
