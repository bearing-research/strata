# @name Plot it
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(by_region["region"], by_region["revenue"])
ax.set_ylabel("Revenue")
plt.tight_layout()
fig
