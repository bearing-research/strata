# Page through 2,000 transactions

A two-cell notebook that produces DataFrames larger than the 20-row
inline preview, so the notebook's **interactive data viewer** pages
through the full frame.

---

## Generate transactions { .nb-cell }

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

**Output**

| id | ts | region | product | units | unit_price | revenue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-01-01 00:00:00 | West | Doohickey | 56 | 53.75 | 3010 |
| 2 | 2026-01-01 01:00:00 | East | Sprocket | 23 | 204.3 | 4698 |
| 3 | 2026-01-01 02:00:00 | East | Gadget | 90 | 180.8 | 1.627e+04 |
| 4 | 2026-01-01 03:00:00 | West | Sprocket | 49 | 225.2 | 1.104e+04 |
| 5 | 2026-01-01 04:00:00 | East | Widget | 41 | 231.4 | 9489 |
| 6 | 2026-01-01 05:00:00 | West | Doohickey | 94 | 62.23 | 5850 |
| 7 | 2026-01-01 06:00:00 | West | Gadget | 46 | 230.8 | 1.062e+04 |
| 8 | 2026-01-01 07:00:00 | North | Doohickey | 73 | 103.2 | 7533 |

*…showing 8 of 2000 rows*

```text title="stdout"
2,000 transactions across 4 regions
```

## Rank by revenue { .nb-cell }

```python
# @name Rank by revenue
# A derived frame gets the same viewer. The grid's column sort is applied
# server-side over the *whole* frame, so it re-orders independently of this
# initial sort; click any header to prove it.
ranked = transactions.sort_values("revenue", ascending=False).reset_index(drop=True)

print(f"Top revenue: ${ranked['revenue'].iloc[0]:,.2f}")
ranked
```

**Output**

| id | ts | region | product | units | unit_price | revenue |
| --- | --- | --- | --- | --- | --- | --- |
| 647 | 2026-01-27 22:00:00 | West | Widget | 99 | 249.8 | 2.473e+04 |
| 1276 | 2026-02-23 03:00:00 | East | Sprocket | 99 | 249 | 2.465e+04 |
| 86 | 2026-01-04 13:00:00 | East | Widget | 99 | 244.3 | 2.418e+04 |
| 1356 | 2026-02-26 11:00:00 | East | Gadget | 98 | 246.6 | 2.417e+04 |
| 1029 | 2026-02-12 20:00:00 | East | Gadget | 98 | 244.3 | 2.394e+04 |
| 728 | 2026-01-31 07:00:00 | East | Sprocket | 96 | 249.4 | 2.394e+04 |
| 1397 | 2026-02-28 04:00:00 | East | Widget | 98 | 243.5 | 2.386e+04 |
| 1459 | 2026-03-02 18:00:00 | East | Sprocket | 99 | 239.7 | 2.373e+04 |

*…showing 8 of 2000 rows*

```text title="stdout"
Top revenue: $24,731.19
```

---

## About this example

### What it shows

- A cell whose displayed output is a **2,000-row** DataFrame. Every
  table-shaped output renders in the grid; one larger than the preview
  also gets a footer showing `1–50 of 2,000 rows` and **Prev / Next**
  paging.
- **Click a column header to sort**: ascending, then descending, then
  clear. Sorting runs server-side over the whole frame (not just the
  visible page), reading the cached Arrow artifact directly.
- A **page-size** selector (25 / 50 / 100 / 250).
- Columns of every dtype: integer (`id`), datetime (`ts`), string
  (`region`, `product`), and float (`unit_price`, `revenue`).

The viewer only appears for table-shaped outputs (pandas / polars
DataFrames + Series, pyarrow Tables, and SQL / R tabular results).
Tensors and scalars keep their existing display.

### Cells

| Cell | What it does |
|---|---|
| `generate` | Builds a 2,000-row transactions frame and displays it. |
| `ranked` | Sorts by revenue into a derived frame, also gets the grid. |

### Running

From the project root:

```bash
uv run strata-notebook --notebook-dir examples
```

Then open **Data Viewer** from the notebook list on the Strata home page and run both
cells.

### Try this

1. Run `generate`. Scroll the grid; click **Next** to page through all
   2,000 rows.
2. Click the `revenue` header twice to sort descending, the top row is
   now the largest revenue in the *entire* frame, not just this page.
3. Change the page size to 250 and page again.
4. Run `ranked` and sort it by `ts`: the server re-sorts the derived
   frame on demand.
