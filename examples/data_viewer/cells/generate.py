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
