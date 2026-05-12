# @name Feature importance from the best model
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    importance.plot.barh(ax=ax, color="#89b4fa")
    ax.set_title(f"Feature Importance ({best_name})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("/tmp/titanic_importance.png", dpi=100)
    print("Saved feature importance to /tmp/titanic_importance.png")
    print(importance.sort_values(ascending=False))
else:
    print(f"{best_name} does not expose feature_importances_")
