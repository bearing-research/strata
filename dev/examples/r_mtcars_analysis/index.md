# R: mtcars regression + inline plots

A **pure-R notebook**: every cell is R. Load the built-in `mtcars`
dataset, summarise it, fit a linear model, and draw two plots that
render inline as PNG. Variables flow cell-to-cell through the same
content-addressed artifact store the Python cells use; here both ends
of every edge happen to be R.

## What it shows

- **R cells stand on their own.** No Python anywhere. The DAG, the
  provenance cache, and the cascade all work exactly as they do for
  Python, language is per-cell, not per-notebook.
- **Inline plots (0.2.0).** A ggplot scatter and R's base-graphics
  2×2 `plot(lm)` diagnostic panel both render as PNG in the cell, just
  like a Python matplotlib figure. A bare trailing ggplot object
  auto-prints, no explicit `print()`.
- **Two kinds of R→R handoff.** `data.frame`s (`cars`, `by_cyl`) cross
  as Arrow IPC. The `lm` object itself isn't tabular, so it's stored as
  RDS (`r_only`) and read straight back by the diagnostics cell with
  full fidelity, an R-only object flowing between R cells that the
  Arrow tier couldn't carry. (A *Python* cell consuming it would get a
  structured "re-export as a data.frame" error instead of a crash.)
- **renv, one click.** The plotting cell needs `ggplot2`, which isn't
  in the harness baseline. It's pinned in `renv.lock` and restored
  automatically when you open the notebook, or via the Environment
  panel's **Initialize renv**. A missing package surfaces a structured
  install hint, not a stack trace.

## Cells

| Cell | What it does |
|---|---|
| `prep` | Tidy `mtcars` into a `data.frame` (`cars`), model name column, `cyl` as a factor. |
| `summarise` | `aggregate()` mean mpg / hp / wt by cylinder count → `by_cyl`. |
| `fit` | `lm(mpg ~ wt + hp + cyl)`; emit the `model` (RDS), a tidy `coefs` table, and one-row `model_stats`. |
| `plot-mpg` | ggplot2 scatter of mpg vs weight, coloured by cylinder, with per-group fits → inline PNG. |
| `diagnostics` | Read `model` back from RDS; base-graphics 2×2 residual diagnostics → inline PNG. |

## What you need

- **R** on `PATH` (`Rscript`). The notebook's `arrow`, `jsonlite`, and
  `ggplot2` come from `renv.lock`: opening the notebook restores them
  into a project-scoped library automatically; no system installs
  beyond R itself.
- The uv-managed Python venv carries only the notebook harness baseline
  (`pyarrow` / `orjson` / `cloudpickle`); **no Python runs any cell
  here**.

## Running

From the project root:

```bash
uv run strata-notebook
```

Open `examples/r_mtcars_analysis` from the Strata home page and run the
cells top to bottom. Or run it headlessly, `strata run` executes R cells
and restores the notebook's `renv.lock` on the way in:

```bash
uv run strata run examples/r_mtcars_analysis
```

## Try this

1. **Swap in another plot.** Replace `plot-mpg` with
   `ggplot(by_cyl, aes(factor(cyl), mean_mpg)) + geom_col()`: a bar of
   mean economy by cylinder. Edit, Shift+Enter, watch it re-render.
2. **Reach a non-tabular object across cells.** Add a cell with
   `confint(model)`: `model` resolves from the RDS artifact and you get
   coefficient confidence intervals, no re-fit.
3. **Break, then fix, the environment.** Delete the `renv/` directory
   and reopen: the ggplot cell shows the install hint; click **Initialize
   renv** (or **Install ggplot2**) and re-run.

## Prepare the mtcars data

<sub>**kind** r</sub>

```text
# @name Prepare the mtcars data
#
# A pure-R notebook: every cell is R. Variables flow cell-to-cell
# through the same content-addressed artifact store the Python cells
# use: a data.frame crosses as Arrow IPC, so the next R cell receives
# it as a data.frame with no glue code.
#
# `mtcars` ships with R (1974 Motor Trend, 32 cars). Tidy it into the
# frame the rest of the notebook builds on: keep the model name as a
# real column, make the cylinder count a factor (so the model and the
# plots treat it as categorical), and keep the columns we care about.

cars <- data.frame(
  model = rownames(mtcars),
  mpg = mtcars$mpg,
  wt = mtcars$wt,
  hp = mtcars$hp,
  cyl = factor(mtcars$cyl),
  row.names = NULL
)

cat(sprintf(
  "Prepared %d cars across %d cylinder classes (%s)\n",
  nrow(cars), nlevels(cars$cyl), paste(levels(cars$cyl), collapse = ", ")
))
```

## Summarise by cylinder count

<sub>**kind** r</sub>

```text
# @name Summarise by cylinder count
#
# Split-apply-combine in base R: group the cars by cylinder class and
# average economy, power, and weight per group. `aggregate()` is the
# base-R group-by; no dplyr needed. `cars` arrives as a data.frame
# (the Arrow IPC handoff from the prep cell), and `by_cyl` leaves as
# one too, ready for any downstream cell.

by_cyl <- aggregate(
  cbind(mpg, hp, wt) ~ cyl,
  data = cars,
  FUN = mean
)
by_cyl$n <- as.integer(table(cars$cyl))
names(by_cyl) <- c("cyl", "mean_mpg", "mean_hp", "mean_wt", "n")

print(by_cyl)
```

## Fit mpg ~ wt + hp + cyl

<sub>**kind** r</sub>

```text
# @name Fit mpg ~ wt + hp + cyl
#
# R's formula syntax in one line: predict mpg from weight, horsepower,
# and the cylinder factor (auto dummy-encoded, 4-cyl as the baseline
# level). Three outputs leave this cell with three different fates in
# the artifact store:
#
#   model        the lm object itself. Not tabular, so the harness
#                  stores it as RDS and tags it `r_only`. A downstream
#                  *R* cell reads it straight back with readRDS (see the
#                  diagnostics cell); a Python cell consuming it would
#                  instead get a structured "re-export as a data.frame"
#                  error rather than a confusing NameError.
#   coefs        tidy coefficient table (data.frame -> Arrow IPC).
#   model_stats  one-row fit summary (data.frame -> Arrow IPC).

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
```

## Plot mpg vs weight (ggplot2)

<sub>**kind** r</sub>

```text
# @name Plot mpg vs weight (ggplot2)
#
# The headline 0.2.0 R feature: plots render inline as PNG, exactly
# like a Python matplotlib figure. A bare trailing ggplot object
# auto-prints (REPL-style), so no explicit `print()` is needed: the
# plot is the cell's last expression and it just shows up.
#
# ggplot2 isn't part of the harness baseline (arrow + jsonlite); it
# comes from this notebook's renv.lock, restored automatically when you
# open the notebook (or one click via the Environment panel's
# "Initialize renv"). If it's somehow missing you'll see a structured
# "install ggplot2" hint, not a crash.

library(ggplot2)

ggplot(cars, aes(x = wt, y = mpg, colour = cyl)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.7) +
  labs(
    title = "Fuel economy falls with weight",
    subtitle = "mtcars, coloured by cylinder count",
    x = "Weight (1000 lbs)",
    y = "Miles per gallon",
    colour = "Cylinders"
  ) +
  theme_minimal(base_size = 13)
```

## Residual diagnostics (base graphics)

<sub>**kind** r</sub>

```text
# @name Residual diagnostics (base graphics)
#
# `plot()` on an lm object draws R's canonical 2x2 diagnostic panel
# (residuals vs fitted, Q-Q, scale-location, residuals vs leverage).
# It's a base-graphics plot, and it's captured to PNG the same way the
# ggplot is: no extra code, no device juggling. `par(mfrow)` tiles the
# four panels onto one image.
#
# `model` arrives by reading the RDS artifact the fit cell stored: an
# R-only object (an lm, S3 class and all) flowing from one R cell to
# another with full fidelity, something the tabular Arrow tier can't
# carry. This is the R-to-R counterpart of the cross-language Arrow
# handoff.

op <- par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(model)
par(op)
```
