# Filter, group, merge and pivot a sales table

A guided tour of the pandas operations you reach for every day, split
into one cell per concept. Each cell re-uses its upstream cell's
artifact, so edits stay fast and the DAG stays honest.

---

## Create sample sales dataset { .nb-cell }

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

**Output**

| date | region | product | units | price |
| --- | --- | --- | --- | --- |
| 2025-01-01 00:00:00 | East | Doohickey | 49 | 21.86 |
| 2025-01-02 00:00:00 | West | Doohickey | 46 | 76.31 |
| 2025-01-03 00:00:00 | North | Widget | 31 | 81.65 |
| 2025-01-04 00:00:00 | East | Widget | 35 | 99.1 |
| 2025-01-05 00:00:00 | East | Gadget | 33 | 44.2 |
| 2025-01-06 00:00:00 | West | Widget | 21 | 40.34 |
| 2025-01-07 00:00:00 | North | Doohickey | 32 | 78.76 |
| 2025-01-08 00:00:00 | North | Doohickey | 23 | 37.38 |

*…showing 8 of 10 rows*

```text title="stdout"
Created 200 rows
```

## Filter high-value orders { .nb-cell }

```python
# @name Filter high-value orders
# Filter to high-value orders (units > 20 and price > 30).
high_value = sales[(sales["units"] > 20) & (sales["price"] > 30)].copy()

print(
    f"High-value orders: {len(high_value)} / {len(sales)} ({100 * len(high_value) / len(sales):.1f}%)"
)
high_value.head()
```

**Output**

| date | region | product | units | price |
| --- | --- | --- | --- | --- |
| 2025-01-02 00:00:00 | West | Doohickey | 46 | 76.31 |
| 2025-01-03 00:00:00 | North | Widget | 31 | 81.65 |
| 2025-01-04 00:00:00 | East | Widget | 35 | 99.1 |
| 2025-01-05 00:00:00 | East | Gadget | 33 | 44.2 |
| 2025-01-06 00:00:00 | West | Widget | 21 | 40.34 |

```text title="stdout"
High-value orders: 103 / 200 (51.5%)
```

## Add computed columns { .nb-cell }

```python
# @name Add computed columns
orders = sales.copy()
orders["revenue"] = orders["units"] * orders["price"]
orders["month"] = orders["date"].dt.to_period("M").astype(str)

print(f"Total revenue: ${orders['revenue'].sum():,.2f}")
orders[["date", "region", "product", "units", "price", "revenue", "month"]].head()
```

**Output**

| date | region | product | units | price | revenue | month |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-01-01 00:00:00 | East | Doohickey | 49 | 21.86 | 1071 | 2025-01 |
| 2025-01-02 00:00:00 | West | Doohickey | 46 | 76.31 | 3510 | 2025-01 |
| 2025-01-03 00:00:00 | North | Widget | 31 | 81.65 | 2531 | 2025-01 |
| 2025-01-04 00:00:00 | East | Widget | 35 | 99.1 | 3468 | 2025-01 |
| 2025-01-05 00:00:00 | East | Gadget | 33 | 44.2 | 1459 | 2025-01 |

```text title="stdout"
Total revenue: $286,921.78
```

## Revenue by region and product { .nb-cell }

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

```text title="stdout"
                       sum     mean  count
region product                            
West   Gadget     39380.61  1575.22     25
East   Widget     36094.93  1640.68     22
North  Doohickey  35274.73  1959.71     18
       Widget     25560.20  1345.27     19
South  Widget     24480.51  1440.03     17
West   Doohickey  23145.00  1653.21     14
South  Doohickey  21616.88  1271.58     17
East   Doohickey  19491.84  1146.58     17
West   Widget     19245.32  1283.02     15
East   Gadget     18097.29  1206.49     15
South  Gadget     12689.33  1057.44     12
North  Gadget     11845.14  1316.13      9
```

## Monthly revenue pivot { .nb-cell }

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

```text title="stdout"
region       East     North     South      West
month                                          
2025-01  17464.80  10450.42   3262.09  20146.13
2025-02   9788.52   7223.75   8998.66   9319.29
2025-03  11563.50   9816.40   6487.70  15958.12
2025-04   3531.57   4690.82  22618.98  12097.91
2025-05   7846.12  24564.89   4670.42  12955.69
2025-06   8771.83  13358.63   6983.45   9664.54
2025-07  14717.72   2575.16   5765.42   1629.25
```

## Merge with region metadata { .nb-cell }

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

```text title="stdout"
  region   revenue manager  target_revenue  pct_of_target
0   East  73684.06   Carol           55000          134.0
1  North  72680.07   Alice           50000          145.4
2  South  58786.72     Bob           45000          130.6
3   West  81770.93    Dave           40000          204.4
```

## Top product per region { .nb-cell }

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

```text title="stdout"
Top product by revenue per region:
region   product  revenue
  East    Widget 36094.93
 North Doohickey 35274.73
 South    Widget 24480.51
  West    Gadget 39380.61
```

---

## About this example

### What it shows

- A fan-out: `create_data` feeds `select_filter` and `add_columns`, and
  `add_columns` feeds `groupby`, `pivot`, `merge` and `summary`.
- Cache-hit behavior: re-running a cell after its upstream hasn't
  changed finishes in a few milliseconds.
- How Strata's **staleness propagation** works: edit `add_columns` and
  the four cells it feeds turn yellow automatically, while
  `select_filter` stays ready.

### Cells

| Cell | What it does |
|---|---|
| `create_data` | Builds a small sales DataFrame as the root of the chain. |
| `select_filter` | Column selection + boolean indexing. |
| `add_columns` | A new frame, `orders`, with derived columns (`revenue = units * price`, `month`). |
| `groupby` | `groupby` + aggregate. |
| `pivot` | Pivot from long to wide. |
| `merge` | Join two DataFrames. |
| `summary` | Describe and basic stats. |

### Per-cell unit tests

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

### Running

From the project root:

```bash
uv run strata-notebook --notebook-dir examples
```

Then open **Pandas Basics** from the notebook list on the Strata home page.

### Try this

1. Run all cells top-to-bottom.
2. Edit `create_data` (for example, change a price).
3. Watch every other cell turn stale automatically.
4. Run `summary`. Strata re-executes only the cells on its path,
   `create_data` and `add_columns`, and leaves `select_filter`,
   `groupby`, `pivot` and `merge` stale until you run them.
