# Data Viewer: paging and sorting a large DataFrame

A two-cell notebook that produces DataFrames larger than the inline
preview, so the notebook's **interactive data viewer** switches on.

## What it shows

- A cell whose displayed output is a **2,000-row** DataFrame. Instead of
  a static 20-row table, the output renders as a scrollable grid with a
  footer showing `1–50 of 2,000 rows` and **Prev / Next** paging.
- **Click a column header to sort**: ascending, then descending, then
  clear. Sorting runs server-side over the whole frame (not just the
  visible page), reading the cached Arrow artifact directly.
- A **page-size** selector (25 / 50 / 100 / 250).
- Columns of every dtype: integer (`id`), datetime (`ts`), string
  (`region`, `product`), and float (`unit_price`, `revenue`).

The viewer only appears for table-shaped outputs (pandas / polars
DataFrames + Series, pyarrow Tables, and SQL / R tabular results).
Tensors and scalars keep their existing display.

## Cells

| Cell | What it does |
|---|---|
| `generate` | Builds a 2,000-row transactions frame and displays it. |
| `ranked` | Sorts by revenue into a derived frame, also gets the grid. |

## Running

From the project root:

```bash
uv run strata-notebook
```

Then open `examples/data_viewer` from the Strata home page and run both
cells.

## Try this

1. Run `generate`. Scroll the grid; click **Next** to page through all
   2,000 rows.
2. Click the `revenue` header twice to sort descending, the top row is
   now the largest revenue in the *entire* frame, not just this page.
3. Change the page size to 250 and page again.
4. Run `ranked` and sort it by `ts`: the server re-sorts the derived
   frame on demand.

## Generate transactions

<sub>**kind** python</sub>

```python
# @name Generate transactions
# The interactive grid keys off the cell's *displayed* value, so the last
# expression is the full frame (not `.head()`). 2,000 rows > the 20-row
# inline preview, so the viewer switches on paging + click-to-sort headers.
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 2000

transactions = pd.DataFrame(
    {
        "id": np.arange(1, n + 1),
        "ts": pd.date_range("2026-01-01", periods=n, freq="h"),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "product": rng.choice(["Widget", "Gadget", "Doohickey", "Sprocket"], n),
        "units": rng.integers(1, 100, n),
        "unit_price": np.round(rng.uniform(4.0, 250.0, n), 2),
    }
)
transactions["revenue"] = np.round(transactions["units"] * transactions["unit_price"], 2)

print(f"{len(transactions):,} transactions across {transactions['region'].nunique()} regions")
transactions
```

## Rank by revenue

<sub>**kind** python</sub>

```python
# @name Rank by revenue
# A derived frame gets the same viewer. The grid's column sort is applied
# server-side over the *whole* frame, so it re-orders independently of this
# initial sort; click any header to prove it.
ranked = transactions.sort_values("revenue", ascending=False).reset_index(drop=True)

print(f"Top revenue: ${ranked['revenue'].iloc[0]:,.2f}")
ranked
```
