# R `lm()` vs scikit-learn: side-by-side linear regression

Fit the same housing-price model with R's `lm()` and Python's
`scikit-learn`. Hand the coefficients + held-out predictions back to
a Python cell over Arrow IPC and read them side by side. This is the
mixed-language notebook that #59 (the R Phase 1 capstone) was
designed for, Python data prep, R modeling, Python comparison +
viz, with the artifact store gluing everything together.

## What it shows

- **R's formula syntax in one line.** `lm(price ~ sqft + bedrooms +
  age + location, data = ...)`: auto-dummy-encodes `location`,
  picks a baseline level, fits with std-errors and p-values
  attached. No ColumnTransformer, no design-matrix construction.
- **Cross-language Arrow handoff.** R returns three `data.frame`s
  (`lm_coefs`, `lm_model_stats`, `lm_predictions`); the next Python
  cell reads them as pandas DataFrames with no glue code.
- **Apples-to-apples comparison.** Same data, same train/test split,
  same encoding (we mirror R's `drop_first` factor behaviour in
  pandas). The numbers should match to ~1e-12; the demo's value is
  *how* each toolkit expresses the fit, not which is more accurate.
- **R surfaces stats Python doesn't.** Std-errors, t-statistics,
  p-values for every coefficient, sklearn's `LinearRegression`
  ships none of that. The comparison cell leaves those columns NaN
  on the sklearn side.

## Cells

| Cell | Language | What it does |
|---|---|---|
| `build-data` | Python | Synthesize 240 rows of housing data; split 200 train / 40 test. |
| `fit-lm` | R | `lm(price ~ sqft + bedrooms + age + location)`, return tidy coefficients + model stats + test predictions. |
| `fit-sklearn` | Python | Same fit with `LinearRegression` + a hand-encoded design matrix; return the same three DataFrame shapes. |
| `compare` | Python | Merge the R and sklearn outputs, print a side-by-side coefficient + fit-stats table, compute test RMSEs. |

## What you need

- **R** on `PATH` (`Rscript`). The notebook's `arrow` and `jsonlite`
  come from `renv.lock`: opening the notebook restores them into a
  project-scoped library automatically; no system installs beyond R
  itself. On macOS: `brew install r`. On Ubuntu: see
  [CRAN](https://cran.r-project.org/). The strata-notebook server
  surfaces a clean skip / error if R is missing, no crash.
- **Python deps** declared in this notebook's `pyproject.toml`
  (`pandas`, `numpy`, `scikit-learn`). Strata's per-notebook `uv
  sync` handles them automatically the first time you open the
  notebook.

## Running

From the project root:

```bash
uv run strata-notebook
```

Then open `examples/r_lm_vs_sklearn` from the Strata home page and
run the cells top-to-bottom.

## Expected output

The `compare` cell prints something like:

```
=== Coefficients ===
          term  lm_estimate  sklearn_estimate     delta  std_error  lm_p_value
   (Intercept)     135.56          135.56          -0.00       7.36      0.0000
           age      -1.20           -1.20          -0.00       0.08      0.0000
      bedrooms      15.12           15.12           0.00       1.26      0.0000
 locationrural    -137.39         -137.39          -0.00       5.35      0.0000
locationsuburb     -85.30          -85.30          -0.00       3.93      0.0000
          sqft       0.18            0.18          -0.00       0.00      0.0000

=== Model fit ===
 source  r_squared  adj_r_squared  f_statistic  df_residual  residual_std_error
 R lm()     0.9698         0.9690    1246.94            194             25.19
sklearn     0.9698         0.9690    1246.94            194             25.19

Test RMSE, R lm(): 21.97   sklearn: 21.97
Max |R-sklearn| prediction gap on test set: 0.0000
```

The 0.00 deltas are the point: both toolkits compute the same OLS
solution; the `lm_p_value` column is R-only.

## Try this

1. **Edit the model.** Drop `bedrooms` from the R formula
   (`lm(price ~ sqft + age + location)`) and watch the comparison
   reshape, `compare` shows the row dropping out.
2. **Misalign the baseline level.** In `fit-sklearn`, replace the
   `sorted(...)` in `location_levels = sorted(...)` with an unsorted
   list, `list(housing_train["location"].unique())`. When the first
   level seen is not the alphabetical first, the baseline no longer
   matches R's and the sklearn coefficients will swap signs vs R's, a
   real risk in any "translate this R analysis to Python" workflow.
3. **Add interaction terms.** R: `price ~ sqft * location`. sklearn:
   you'll have to add the interaction columns by hand. The contrast
   gets sharper.

## Synthesize a small housing dataset

<sub>**kind** python</sub>

```python
# @name Synthesize a small housing dataset
#
# 240 rows, four features (sqft, bedrooms, age, location), one target
# (price). Generated with a known linear-with-noise structure so both
# R and sklearn should recover similar coefficients; the demo is
# about *how* each toolkit expresses the fit, not which is more
# accurate.
#
# Split into train (200) + test (40). The test split is held out so
# downstream cells can compare R's predicted prices against sklearn's
# on the same observations.

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)
n = 240

sqft = rng.uniform(600, 3200, size=n).round(0)
bedrooms = rng.integers(1, 6, size=n)
age = rng.uniform(0, 80, size=n).round(1)
location = rng.choice(
    ["downtown", "suburb", "rural"],
    size=n,
    p=[0.35, 0.5, 0.15],
)

# Known generating coefficients (in $1k units): intercept 50,
# sqft 0.18, bedrooms 15, age -1.2, downtown +85, suburb 0,
# rural -45. Gaussian noise σ=25.
location_premium = pd.Series(location).map(
    {"downtown": 85.0, "suburb": 0.0, "rural": -45.0}
).to_numpy()
price_thousands = (
    50
    + 0.18 * sqft
    + 15 * bedrooms
    - 1.2 * age
    + location_premium
    + rng.normal(0, 25, size=n)
).round(2)

housing = pd.DataFrame(
    {
        "sqft": sqft,
        "bedrooms": bedrooms,
        "age": age,
        "location": location,
        "price": price_thousands,
    }
)

# Hold out the last 40 rows as a test set. Deterministic split: no
# need to re-shuffle since the rows are already in random order.
housing_train = housing.iloc[:200].reset_index(drop=True)
housing_test = housing.iloc[200:].reset_index(drop=True)

print(f"train: {len(housing_train)} rows, test: {len(housing_test)} rows")
print(housing_train.head())
```

## Fit lm() with R's formula syntax

<sub>**kind** r</sub>

```text
# @name Fit lm() with R's formula syntax
#
# The marquee R-vs-Python contrast lives in this one expression:
#
#     model <- lm(price ~ sqft + bedrooms + age + location, ...)
#
# R reads ``price ~ sqft + bedrooms + age + location`` as: predict
# price from these four predictors; auto-dummy-encode the
# ``location`` character vector (factor) so the model picks
# ``rural`` / ``suburb`` indicator coefficients with ``downtown`` as
# the baseline level. No design-matrix construction, no
# OneHotEncoder, no ColumnTransformer: just the formula.
#
# Returns three data.frames so downstream Python cells consume them
# over Arrow IPC unchanged:
#
#   lm_coefs        one row per coefficient (term, estimate, std
#                      error, t value, p value).
#   lm_model_stats  single-row summary (r², adj r², F, df,
#                      residual std err).
#   lm_predictions  test-set predictions per row.
#
# All three are bare data.frames, so harness.R's serializer takes
# the Arrow tier (not the JSON or RDS fallback).

# ``housing_train`` and ``housing_test`` arrive as ``data.frame``s
# already; that's what the Arrow IPC reader hands us when an
# upstream Python cell stored a ``pandas.DataFrame``. The character
# column ``location`` is what R needs to fit dummy-coded effects;
# converting once here lets ``lm()`` treat it as a categorical.
housing_train$location <- factor(housing_train$location)
housing_test$location <- factor(
  housing_test$location,
  levels = levels(housing_train$location)
)

model <- lm(price ~ sqft + bedrooms + age + location, data = housing_train)
fit_summary <- summary(model)

# Hand-build a tidy coefficients data.frame. ``broom::tidy`` would
# give the same shape in one line but adds a dependency outside the
# harness's ``arrow`` + ``jsonlite`` baseline. Base R is enough.
coef_matrix <- fit_summary$coefficients
lm_coefs <- data.frame(
  term = rownames(coef_matrix),
  estimate = coef_matrix[, "Estimate"],
  std_error = coef_matrix[, "Std. Error"],
  t_stat = coef_matrix[, "t value"],
  p_value = coef_matrix[, "Pr(>|t|)"],
  row.names = NULL
)

# Single-row "glance" data.frame: the model-level fit stats that
# you'd typically print at the top of summary().
lm_model_stats <- data.frame(
  r_squared = fit_summary$r.squared,
  adj_r_squared = fit_summary$adj.r.squared,
  f_statistic = fit_summary$fstatistic[["value"]],
  df_residual = model$df.residual,
  residual_std_error = fit_summary$sigma,
  n_train = nrow(housing_train)
)

# Predictions on the held-out test set. ``predict.lm`` accepts a
# data.frame keyed by the formula's predictor names + applies the
# factor encoding learned during ``lm()``.
lm_predictions <- data.frame(
  actual = housing_test$price,
  predicted = unname(predict(model, newdata = housing_test))
)

cat(sprintf(
  "R lm(): R²=%.4f, F=%.1f on %d df\n",
  lm_model_stats$r_squared,
  lm_model_stats$f_statistic,
  lm_model_stats$df_residual
))
```

## Fit the same model with scikit-learn

<sub>**kind** python</sub>

```python
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
# ``LinearRegression``; its OLS implementation gives only point
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
# computable from sklearn's LinearRegression, so we leave them NaN and
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

# Glance row matching R's lm_model_stats shape. R² we can compute,
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
```

## Side-by-side comparison: R lm() vs sklearn

<sub>**kind** python</sub>

```python
# @name Side-by-side comparison: R lm() vs sklearn
#
# Bring the two fits together. The reads here are the cross-language
# payoff: ``lm_coefs`` came out of an R cell over Arrow IPC, and
# pandas treats it exactly like ``sklearn_coefs``: both are
# ordinary DataFrames at this point, nothing to glue.

import pandas as pd

# Coefficient comparison. Merge on ``term``: R uses
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

# Model-stats comparison. Both DataFrames are single-row; stack
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

# Held-out predictions: same observations, two predictions each.
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
print(f"\nTest RMSE for R lm(): {rmse_lm:.3f}   sklearn: {rmse_sklearn:.3f}")
print(f"Max |R-sklearn| prediction gap on test set: "
      f"{predictions_compare['lm_sklearn_diff'].abs().max():.4f}")
```
