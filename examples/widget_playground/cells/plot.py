# @name Plot
# A second downstream cell — the same controls drive a matplotlib chart.
# The Controls cell ships with ⚡ Live on, so dragging `alpha` (or changing
# `curve`) updates this plot live, in the notebook or the app view.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(n)
base = {"linear": x, "sqrt": np.sqrt(x), "square": x.astype(float) ** 2}[curve]
y = alpha * base

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(x, y, marker="o", color="#89b4fa")
ax.fill_between(x, y, alpha=0.15, color="#89b4fa")
ax.set_title(f"alpha={alpha}  curve={curve}")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.tight_layout()
fig
