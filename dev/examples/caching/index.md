# Tutorial: what the cache saves

A two-minute tour of caching. Run it twice and read what the second run saved,
edit one cell and watch only what depends on it run again, then open the graph.

Open a copy with `strata tutorial caching`, or from **Open a sample** on the home page.

## Cells

1. `load.py`: builds a small table, after a two-second wait that stands in for a slow query
2. `summarize.py`: revenue by region
3. `plot.py`: a bar chart of the summary

The markdown cells between them are the instructions.

# What the cache saves

Strata stores each cell's result under a hash of its source, its inputs and its
environment. If none of those changed, the cell doesn't run again. This takes about
two minutes.

## Step 1: run it

Press **Run All** at the top of the page. The cell below waits two seconds
on purpose, standing in for a slow query.

## Load the data

<sub>**kind** python</sub>

```python
# @name Load the data
import time

import pandas as pd

time.sleep(2)

sales = pd.DataFrame(
    {
        "region": ["North", "North", "South", "South", "East", "East", "West", "West"],
        "quarter": ["Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2"],
        "revenue": [120, 135, 98, 110, 143, 151, 87, 95],
    }
)
sales
```

## Step 2: run it again

Press **Run All** again. Nothing runs this time. Every code cell's badge reads `cached`,
and the load cell's says how much time that saved. For the total, press **expand** at the
bottom right of the page and read **Cache savings** in the **Profiling** panel.

## Summarize by region

<sub>**kind** python</sub>

```python
# @name Summarize by region
by_region = sales.groupby("region", as_index=False)["revenue"].sum()
by_region
```

## Step 3: edit one cell

In the cell above, change `.sum()` to `.mean()`. It reads `stale`, and so does the chart
below, which uses its result. The load cell stays `cached`. Press **Run All**: only those
two run.

Now change it back to `.sum()` and press **Run All** once more. Both read `cached` at once,
because Strata still has that result.

## Plot it

<sub>**kind** python</sub>

```python
# @name Plot it
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(by_region["region"], by_region["revenue"])
ax.set_ylabel("Revenue")
plt.tight_layout()
fig
```

## Step 4: open the DAG

Press **expand** at the bottom right if the drawer is closed, and look at **Cell DAG**: load,
then summarize, then plot. Nobody declared those links. They come from the variables each
cell reads, and they are how Strata knew what your edit in step 3 affected.
