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
