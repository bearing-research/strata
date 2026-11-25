# Drive a plot with a slider, a number and a dropdown

A three-cell notebook: a **widget cell** (a control panel) feeding two Python
cells, one that renders a DataFrame and one that draws a chart. The widget
cell is `# @live`, so dragging a control re-runs both downstream cells and
the grid and the chart update.

---

## Controls { .nb-cell }

```text
# @live
# @name Controls
alpha = slider(0, 1, step=0.05, default=0.5)
n     = number(default=20, min=5, max=100)
curve = dropdown(["linear", "sqrt", "square"], default="linear")
```

## Preview { .nb-cell }

```python
# @name Preview
# Downstream of the widget cell: drag `alpha` / change `curve` and re-run
# (or run in the live cascade) and the table updates. It renders in the
# interactive data viewer, so widgets drive the grid.
import numpy as np
import pandas as pd

x = np.arange(n)
base = {"linear": x, "sqrt": np.sqrt(x), "square": x**2}[curve]
df = pd.DataFrame({"x": x, "y": np.round(alpha * base, 3)})

print(f"alpha={alpha}  n={n}  curve={curve}")
df
```

**Output**

| x | y |
| --- | --- |
| 0 | 0 |
| 1 | 0.5 |
| 2 | 1 |
| 3 | 1.5 |
| 4 | 2 |
| 5 | 2.5 |
| 6 | 3 |
| 7 | 3.5 |

*…showing 8 of 20 rows*

```text title="stdout"
alpha=0.5  n=20  curve=linear
```

## Plot { .nb-cell }

```python
# @name Plot
# A second downstream cell: the same controls drive a matplotlib chart.
# The Controls cell ships with ⚡ Live on, so dragging `alpha` (or changing
# `curve`) updates this plot live, in the notebook or the app view.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("strata")

x = np.arange(n)
base = {"linear": x, "sqrt": np.sqrt(x), "square": x.astype(float) ** 2}[curve]
y = alpha * base

fig, ax = plt.subplots(figsize=(6, 3.2))
(line,) = ax.plot(x, y, marker="o")
ax.fill_between(x, y, alpha=0.15, color=line.get_color())
ax.set_title(f"alpha={alpha}  curve={curve}")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.tight_layout()
fig
```

![cell output](https://bearingresearch.com/strata/dev/assets/examples/runs/widget_playground-1.webp)

---

## About this example

### What it shows

- A **widget cell** with three controls:
  - `alpha`: a slider (0–1)
  - `n`: a number input (row count)
  - `curve`: a dropdown (`linear` / `sqrt` / `square`)
- A downstream Python cell that reads `alpha`, `n`, `curve` and builds a
  DataFrame, which renders in the **interactive data viewer**. So the two
  features compose: a widget drives the grid.
- A second downstream cell that plots the same curve with matplotlib.
- **Live mode**: `# @live` on the widget cell makes a control change re-run
  the cheap downstream cells instead of only marking them stale.
- **Content-addressed values**: drag `alpha` to a new value and back, the
  second time is a cache hit, no recompute.

### Cells

| Cell | What it does |
|---|---|
| `controls` | A `# @live` `widget` cell declaring the slider / number / dropdown. |
| `preview` | Reads the control values, builds a DataFrame, displays it. |
| `plot` | Reads the same values and plots `y` against `x`. |

### Running

From the project root:

```bash
uv run strata-notebook --notebook-dir examples
```

Open **Widget Playground** from the notebook list on the Strata home page.

### Try this

1. Run `preview` and `plot`. They read the widget defaults: a 20-row table
   in the data viewer and a straight line.
2. Drag the `alpha` slider. Both cells re-run on their own, and the `y`
   column and the chart reflect the new `alpha`.
3. Switch `curve` to `square`, the shape changes.
4. Drag `alpha` back to its previous value, a cache hit.
5. Turn **⚡ Live** off (or change the annotation to `# @live off`) and drag
   again: now the two cells only turn yellow (stale) until you run them.

Widget cells render in the web UI. Edit the control declaration with the
**✎ Edit controls** toggle on the cell.

### Open as an app

Click **App** in the notebook header to open this notebook as a read-only
interactive app, just the controls, the table and the chart, no editor. With
the widget cell's **⚡ Live** on, as it ships, dragging `alpha` in the app
updates them live.
