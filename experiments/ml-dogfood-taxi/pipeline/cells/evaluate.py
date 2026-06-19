# @name evaluate
"""Evaluate on the holdout; metrics as a structured frame (dashboard-friendly)."""

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

pred = model.predict(X_test)

eval_metrics = pd.DataFrame(
    [
        {"metric": "mae", "value": float(mean_absolute_error(y_test, pred))},
        {"metric": "rmse", "value": float(root_mean_squared_error(y_test, pred))},
        {"metric": "r2", "value": float(r2_score(y_test, pred))},
        {"metric": "holdout_rows", "value": float(len(y_test))},
    ]
)

print(eval_metrics.to_string(index=False))
eval_metrics
