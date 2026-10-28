# @name evaluate
"""Evaluate on the holdout; metrics as a structured frame (dashboard-friendly)."""

import numpy as np
import pandas as pd
import torch

with torch.no_grad():
    pred = model(torch.from_numpy((X_test - feature_mean) / feature_std)).squeeze(1).numpy()
err = pred - y_test

eval_metrics = pd.DataFrame(
    [
        {"metric": "mae", "value": float(np.abs(err).mean())},
        {"metric": "rmse", "value": float(np.sqrt((err**2).mean()))},
        {
            "metric": "r2",
            "value": float(1 - (err**2).sum() / ((y_test - y_test.mean()) ** 2).sum()),
        },
        {"metric": "holdout_rows", "value": float(len(y_test))},
    ]
)

print(eval_metrics.to_string(index=False))
eval_metrics
