# @name Fit the same model with scikit-learn
#
# Same four predictors, same train/test split. The work is in
# encoding ``location`` so the comparison with R is apples-to-apples:
# R's ``lm()`` auto-dummies the factor with ``downtown`` as the
# baseline (alphabetical first level); we mirror that here with
# ``pd.get_dummies(drop_first=True)`` after a sort so the dropped
# column matches.
#
# sklearn doesn't surface std-errors or p-values from
# ``LinearRegression`` — its OLS implementation gives only point
# estimates. The comparison cell handles that gap by leaving those
# columns NaN on the sklearn side.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ``location`` arrives as a string column (the same shape R's
# ``factor()`` consumed).
location_levels = sorted(housing_train["location"].unique())  # baseline = first
baseline = location_levels[0]


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(
        {
            "sqft": df["sqft"].astype(float),
            "bedrooms": df["bedrooms"].astype(float),
            "age": df["age"].astype(float),
        }
    )
    # One-hot, dropping the baseline level (R's behaviour). Coerce to
    # the same column order R uses (``locationrural``, ``locationsuburb``
    # for baseline = ``downtown``) so the side-by-side comparison
    # lines up.
    for lvl in location_levels:
        if lvl == baseline:
            continue
        X[f"location{lvl}"] = (df["location"] == lvl).astype(float)
    return X


X_train = _encode(housing_train)
X_test = _encode(housing_test)
y_train = housing_train["price"].to_numpy()
y_test = housing_test["price"].to_numpy()

reg = LinearRegression()
reg.fit(X_train, y_train)

# Match R's coefficient row layout: intercept first, then predictors
# in design-matrix column order. Standard errors / p-values aren't
# computable from sklearn's LinearRegression, so we leave them NaN —
# the compare cell drops them out of the side-by-side view.
sklearn_coefs = pd.DataFrame(
    {
        "term": ["(Intercept)"] + list(X_train.columns),
        "estimate": [float(reg.intercept_)] + [float(c) for c in reg.coef_],
        "std_error": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
    }
)

# Glance row matching R's lm_model_stats shape — R² we can compute,
# adj R² needs df-residual which is n - p - 1, F statistic ditto.
n = len(y_train)
p = X_train.shape[1]
y_pred_train = reg.predict(X_train)
resid_train = y_train - y_pred_train
ss_res = float(np.sum(resid_train**2))
ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
r2 = 1 - ss_res / ss_tot
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
residual_std_err = float(np.sqrt(ss_res / (n - p - 1)))
f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))

sklearn_model_stats = pd.DataFrame(
    [
        {
            "r_squared": r2,
            "adj_r_squared": adj_r2,
            "f_statistic": f_stat,
            "df_residual": n - p - 1,
            "residual_std_error": residual_std_err,
            "n_train": n,
        }
    ]
)

sklearn_predictions = pd.DataFrame(
    {
        "actual": y_test,
        "predicted": reg.predict(X_test),
    }
)

print(f"sklearn: R²={r2:.4f}, F={f_stat:.1f} on {n - p - 1} df")
