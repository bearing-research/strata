# @name plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 240)
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(x, np.sin(x), label="sin", linewidth=2)
ax.plot(x, np.cos(x), label="cos", linewidth=2)
ax.fill_between(x, np.sin(x), alpha=0.15)
ax.set_title("sin & cos — rendered in your terminal")
ax.legend(); ax.grid(alpha=0.3)
fig