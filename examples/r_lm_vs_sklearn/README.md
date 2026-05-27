# R `lm()` vs scikit-learn — side-by-side linear regression

Fit the same housing-price model with R's `lm()` and Python's
`scikit-learn`. Hand the coefficients + held-out predictions back to
a Python cell over Arrow IPC and read them side by side. This is the
mixed-language notebook that #59 (the R Phase 1 capstone) was
designed for — Python data prep, R modeling, Python comparison +
viz, with the artifact store gluing everything together.

## What it shows

- **R's formula syntax in one line.** `lm(price ~ sqft + bedrooms +
  age + location, data = ...)` — auto-dummy-encodes `location`,
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
  p-values for every coefficient — sklearn's `LinearRegression`
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

- **R + the `arrow` and `jsonlite` R packages** for the cross-
  language handoff. On macOS: `brew install r` then `Rscript -e
  'install.packages(c("arrow", "jsonlite"))'`. On Ubuntu: see
  [CRAN](https://cran.r-project.org/). The strata-notebook server
  surfaces a clean skip / error if R is missing — no crash.
- **Python deps** declared in this notebook's `pyproject.toml`
  (`pandas`, `numpy`, `scikit-learn`). Strata's per-notebook `uv
  sync` handles them automatically the first time you open the
  notebook.

## Running

From the project root:

```bash
uv run strata-notebook --host 127.0.0.1 --port 8765
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

Test RMSE — R lm(): 21.97   sklearn: 21.97
Max |R-sklearn| prediction gap on test set: 0.0000
```

The 0.00 deltas are the point: both toolkits compute the same OLS
solution; the `lm_p_value` column is R-only.

## Try this

1. **Edit the model.** Drop `bedrooms` from the R formula
   (`lm(price ~ sqft + age + location)`) and watch the comparison
   reshape — `compare` shows the row dropping out.
2. **Misalign the baseline level.** Comment out the
   `location_levels.sort(...)` line in `fit-sklearn`. The sklearn
   coefficients will swap signs vs R's — a real risk in any
   "translate this R analysis to Python" workflow.
3. **Add interaction terms.** R: `price ~ sqft * location`. sklearn:
   you'll have to add the interaction columns by hand. The contrast
   gets sharper.
