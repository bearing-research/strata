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

## Per-cell unit tests

Four cells ship with pytest tests next to them
(`cells/<cell-id>.test.py`) — a worked example of Strata's built-in
**cell unit tests**:

| Test file | Pins |
|---|---|
| `create-data.test.py` | row count, columns, no nulls, value ranges |
| `add-columns.test.py` | `revenue == units * price`, `month` format |
| `select-filter.test.py` | the high-value filter invariant holds for every row |
| `summary.test.py` | exactly one winning product per region |

Tests run in the notebook's own venv and receive a `cell` fixture whose
attributes are the cell's namespace **after it runs against its real
upstream input** — so `cell.sales` is the actual DataFrame, `cell.revenue`
a value it computed, `cell.my_func` a function it defined:

```python
def test_revenue_is_units_times_price(cell):
    expected = cell.sales["units"] * cell.sales["price"]
    assert (cell.sales["revenue"] == expected).all()
```

Open the Tests panel on a cell to run them, or send the `cell_run_tests`
WS request. A cell with no tests simply carries no `.test.py` file.

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
