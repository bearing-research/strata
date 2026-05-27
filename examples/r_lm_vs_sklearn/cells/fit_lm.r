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
# OneHotEncoder, no ColumnTransformer — just the formula.
#
# Returns three data.frames so downstream Python cells consume them
# over Arrow IPC unchanged:
#
#   lm_coefs        — one row per coefficient (term, estimate, std
#                      error, t value, p value).
#   lm_model_stats  — single-row summary (r², adj r², F, df,
#                      residual std err).
#   lm_predictions  — test-set predictions per row.
#
# All three are bare data.frames, so harness.R's serializer takes
# the Arrow tier (not the JSON or RDS fallback).

# ``housing_train`` and ``housing_test`` arrive as ``data.frame``s
# already — that's what the Arrow IPC reader hands us when an
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
# harness's ``arrow`` + ``jsonlite`` baseline — base R is enough.
coef_matrix <- fit_summary$coefficients
lm_coefs <- data.frame(
  term = rownames(coef_matrix),
  estimate = coef_matrix[, "Estimate"],
  std_error = coef_matrix[, "Std. Error"],
  t_stat = coef_matrix[, "t value"],
  p_value = coef_matrix[, "Pr(>|t|)"],
  row.names = NULL
)

# Single-row "glance" data.frame — the model-level fit stats that
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
