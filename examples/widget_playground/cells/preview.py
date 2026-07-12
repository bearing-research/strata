# @name Preview
# Downstream of the widget cell: drag `alpha` / change `curve` and re-run
# (or run in the live cascade) — the table updates. It renders in the
# interactive data viewer, so widgets drive the grid.
import numpy as np
import pandas as pd

x = np.arange(n)
base = {"linear": x, "sqrt": np.sqrt(x), "square": x**2}[curve]
df = pd.DataFrame({"x": x, "y": np.round(alpha * base, 3)})

print(f"alpha={alpha}  n={n}  curve={curve}")
df
