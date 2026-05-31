# @name Fit mpg ~ wt + hp + cyl
#
# R's formula syntax in one line: predict mpg from weight, horsepower,
# and the cylinder factor (auto dummy-encoded, 4-cyl as the baseline
# level). Three outputs leave this cell with three different fates in
# the artifact store:
#
#   model        — the lm object itself. Not tabular, so the harness
#                  stores it as RDS and tags it `r_only`. A downstream
#                  *R* cell reads it straight back with readRDS (see the
#                  diagnostics cell); a Python cell consuming it would
#                  instead get a structured "re-export as a data.frame"
#                  error rather than a confusing NameError.
#   coefs        — tidy coefficient table (data.frame -> Arrow IPC).
#   model_stats  — one-row fit summary (data.frame -> Arrow IPC).

model <- lm(mpg ~ wt + hp + cyl, data = cars)
s <- summary(model)

cm <- s$coefficients
coefs <- data.frame(
  term = rownames(cm),
  estimate = cm[, "Estimate"],
  std_error = cm[, "Std. Error"],
  t_stat = cm[, "t value"],
  p_value = cm[, "Pr(>|t|)"],
  row.names = NULL
)

model_stats <- data.frame(
  r_squared = s$r.squared,
  adj_r_squared = s$adj.r.squared,
  sigma = s$sigma,
  df_residual = model$df.residual
)

cat(sprintf(
  "Fitted mpg ~ wt + hp + cyl: R2=%.3f (adj %.3f), residual SE=%.2f mpg\n",
  model_stats$r_squared, model_stats$adj_r_squared, model_stats$sigma
))
