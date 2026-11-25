<!-- strata example run, source d4b4662afc098173 -->
## Controls

<sub>**kind** widget</sub>

```text
# @live
# @name Controls
alpha = slider(0, 1, step=0.05, default=0.5)
n     = number(default=20, min=5, max=100)
curve = dropdown(["linear", "sqrt", "square"], default="linear")
```

## Preview

<sub>**kind** python</sub>

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

## Plot

<sub>**kind** python</sub>

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

![cell output](../assets/examples/runs/widget_playground-1.webp)
