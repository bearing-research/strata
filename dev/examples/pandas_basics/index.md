# Pandas Basics: the core DataFrame operations

A guided tour of the pandas operations you reach for every day, split
into one cell per concept. Each cell re-uses its upstream cell's
artifact, so edits stay fast and the DAG stays honest.

## What it shows

- A fan-out: `create_data` feeds `select_filter` and `add_columns`, and
  `add_columns` feeds `groupby`, `pivot`, `merge` and `summary`.
- Cache-hit behavior: re-running a cell after its upstream hasn't
  changed finishes in a few milliseconds.
- How Strata's **staleness propagation** works, edit `add_columns` and
  the four cells it feeds turn yellow automatically, while
  `select_filter` stays ready.

## Cells

| Cell | What it does |
|---|---|
| `create_data` | Builds a small sales DataFrame as the root of the chain. |
| `select_filter` | Column selection + boolean indexing. |
| `add_columns` | A new frame, `orders`, with derived columns (`revenue = units * price`, `month`). |
| `groupby` | `groupby` + aggregate. |
| `pivot` | Pivot from long to wide. |
| `merge` | Join two DataFrames. |
| `summary` | Describe and basic stats. |

## Per-cell unit tests

Four cells ship with pytest tests next to them
(`cells/<cell-id>.test.py`), a worked example of Strata's built-in
**cell unit tests**:

| Test file | Pins |
|---|---|
| `create-data.test.py` | row count, columns, no nulls, value ranges |
| `add-columns.test.py` | `revenue == units * price`, `month` format |
| `select-filter.test.py` | the high-value filter invariant holds for every row |
| `summary.test.py` | exactly one winning product per region |

Tests run in the notebook's own venv and receive a `cell` fixture whose
attributes are the cell's namespace **after it runs against its real
upstream input**: so `cell.sales` is the actual DataFrame, `cell.orders`
a frame it computed, `cell.my_func` a function it defined:

```python
def test_revenue_is_units_times_price(cell):
    expected = cell.orders["units"] * cell.orders["price"]
    assert (cell.orders["revenue"] == expected).all()
```

Open the Tests panel on a cell to run them, or send the `cell_run_tests`
WS request. A cell with no tests simply carries no `.test.py` file.

## Running

From the project root:

```bash
uv run strata-notebook
```

Then open `examples/pandas_basics` from the Strata home page.

## Try this

1. Run all cells top-to-bottom.
2. Edit `create_data` (for example, change a price).
3. Watch every other cell turn stale automatically.
4. Run `summary`. Strata re-executes only the cells on its path,
   `create_data` and `add_columns`, and leaves `select_filter`,
   `groupby`, `pivot` and `merge` stale until you run them.

## Create sample sales dataset

<sub>**kind** python</sub>

```python
# @name Create sample sales dataset
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

sales = pd.DataFrame(
    {
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "product": np.random.choice(["Widget", "Gadget", "Doohickey"], n),
        "units": np.random.randint(1, 50, n),
        "price": np.round(np.random.uniform(5, 100, n), 2),
    }
)

print(f"Created {len(sales)} rows")
sales.head(10)
```

## Filter high-value orders

<sub>**kind** python</sub>

```python
# @name Filter high-value orders
# Filter to high-value orders (units > 20 and price > 30).
high_value = sales[(sales["units"] > 20) & (sales["price"] > 30)].copy()

print(
    f"High-value orders: {len(high_value)} / {len(sales)} ({100 * len(high_value) / len(sales):.1f}%)"
)
high_value.head()
```

## Add computed columns

<sub>**kind** python</sub>

```python
# @name Add computed columns
orders = sales.copy()
orders["revenue"] = orders["units"] * orders["price"]
orders["month"] = orders["date"].dt.to_period("M").astype(str)

print(f"Total revenue: ${orders['revenue'].sum():,.2f}")
orders[["date", "region", "product", "units", "price", "revenue", "month"]].head()
```

## Revenue by region and product

<sub>**kind** python</sub>

```python
# @name Revenue by region and product
region_product = (
    orders.groupby(["region", "product"])["revenue"]
    .agg(["sum", "mean", "count"])
    .round(2)
    .sort_values("sum", ascending=False)
)

print(region_product)
```

## Monthly revenue pivot

<sub>**kind** python</sub>

```python
# @name Monthly revenue pivot
monthly_pivot = orders.pivot_table(
    values="revenue",
    index="month",
    columns="region",
    aggfunc="sum",
).round(2)

print(monthly_pivot)
```

## Merge with region metadata

<sub>**kind** python</sub>

```python
# @name Merge with region metadata
region_info = pd.DataFrame(
    {
        "region": ["North", "South", "East", "West"],
        "manager": ["Alice", "Bob", "Carol", "Dave"],
        "target_revenue": [50000, 45000, 55000, 40000],
    }
)

region_summary = (
    orders.groupby("region")["revenue"].sum().reset_index().merge(region_info, on="region")
)

region_summary["pct_of_target"] = (
    100 * region_summary["revenue"] / region_summary["target_revenue"]
).round(1)

print(region_summary)
```

## Top product per region

<sub>**kind** python</sub>

```python
# @name Top product per region
# Top product by revenue in each region.
top_products = (
    orders.groupby(["region", "product"])["revenue"]
    .sum()
    .reset_index()
    .sort_values("revenue", ascending=False)
    .drop_duplicates(subset="region")
    .sort_values("region")
)

print("Top product by revenue per region:")
print(top_products.to_string(index=False))
```
