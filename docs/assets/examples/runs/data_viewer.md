<!-- strata example run, source 1477e1fd0c2ff4ca -->
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
