# @name Feature importance from the best model
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Which weights a model exposes depends on what it is: the tree ensembles have
# feature_importances_, logistic regression has coefficients. Reading whichever
# is there keeps the cell producing a figure whoever wins, rather than printing
# an apology when the linear model does.
if hasattr(best_model, "feature_importances_"):
    weights = pd.Series(best_model.feature_importances_, index=feature_cols)
    axis_label = "Importance"
else:
    weights = pd.Series(best_model.coef_[0], index=feature_cols).abs()
    axis_label = "|Coefficient|"

importance = weights.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
importance.plot.barh(ax=ax, color="#89b4fa")
ax.set_title(f"Feature Importance ({best_name})")
ax.set_xlabel(axis_label)
plt.tight_layout()

print(importance.sort_values(ascending=False))
fig
