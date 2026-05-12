# @name Pairwise scatter plot
# Pairwise scatter plot colored by species.
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

g = sns.pairplot(df, hue="species", diag_kind="hist", height=2)
g.figure.suptitle("Iris Feature Distributions", y=1.02)
plt.tight_layout()
plt.savefig("/tmp/iris_pairplot.png", dpi=100)
print("Saved pairplot to /tmp/iris_pairplot.png")
