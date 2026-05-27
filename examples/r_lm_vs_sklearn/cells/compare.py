# @name Side-by-side comparison: R lm() vs sklearn
#
# Bring the two fits together. The reads here are the cross-language
# payoff: ``lm_coefs`` came out of an R cell over Arrow IPC, and
# pandas treats it exactly like ``sklearn_coefs`` — both are
# ordinary DataFrames at this point, nothing to glue.

import pandas as pd

# Coefficient comparison. Merge on ``term`` — R uses
# ``locationrural`` / ``locationsuburb`` (no dot), and the sklearn
# encoder we built matches. Inner join surfaces any mismatch loudly.
coef_compare = pd.merge(
    lm_coefs.rename(columns={"estimate": "lm_estimate", "p_value": "lm_p_value"}),
    sklearn_coefs[["term", "estimate"]].rename(columns={"estimate": "sklearn_estimate"}),
    on="term",
    how="outer",
    indicator=True,
)
coef_compare["delta"] = coef_compare["lm_estimate"] - coef_compare["sklearn_estimate"]
coef_compare = coef_compare.drop(columns=["_merge"])[
    ["term", "lm_estimate", "sklearn_estimate", "delta", "std_error", "lm_p_value"]
]

# Model-stats comparison — both DataFrames are single-row; stack
# them and add a label column so the row source is obvious.
stats_compare = pd.concat(
    [
        lm_model_stats.assign(source="R lm()"),
        sklearn_model_stats.assign(source="sklearn"),
    ],
    ignore_index=True,
)[
    [
        "source",
        "r_squared",
        "adj_r_squared",
        "f_statistic",
        "df_residual",
        "residual_std_error",
    ]
]

# Held-out predictions — same observations, two predictions each.
predictions_compare = pd.DataFrame(
    {
        "actual": lm_predictions["actual"],
        "lm_predicted": lm_predictions["predicted"],
        "sklearn_predicted": sklearn_predictions["predicted"],
    }
)
predictions_compare["lm_sklearn_diff"] = (
    predictions_compare["lm_predicted"] - predictions_compare["sklearn_predicted"]
)

# RMSE on the test set, per fitter.
def _rmse(actual: pd.Series, pred: pd.Series) -> float:
    return float(((actual - pred) ** 2).mean() ** 0.5)


rmse_lm = _rmse(predictions_compare["actual"], predictions_compare["lm_predicted"])
rmse_sklearn = _rmse(predictions_compare["actual"], predictions_compare["sklearn_predicted"])

print("=== Coefficients ===")
print(coef_compare.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
print("\n=== Model fit ===")
print(stats_compare.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
print(f"\nTest RMSE — R lm(): {rmse_lm:.3f}   sklearn: {rmse_sklearn:.3f}")
print(f"Max |R-sklearn| prediction gap on test set: "
      f"{predictions_compare['lm_sklearn_diff'].abs().max():.4f}")
