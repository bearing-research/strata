# R: mtcars regression + inline plots

A **pure-R notebook** — every cell is R. Load the built-in `mtcars`
dataset, summarise it, fit a linear model, and draw two plots that
render inline as PNG. Variables flow cell-to-cell through the same
content-addressed artifact store the Python cells use; here both ends
of every edge happen to be R.

## What it shows

- **R cells stand on their own.** No Python anywhere. The DAG, the
  provenance cache, and the cascade all work exactly as they do for
  Python — language is per-cell, not per-notebook.
- **Inline plots (0.2.0).** A ggplot scatter and R's base-graphics
  2×2 `plot(lm)` diagnostic panel both render as PNG in the cell, just
  like a Python matplotlib figure. A bare trailing ggplot object
  auto-prints — no explicit `print()`.
- **Two kinds of R→R handoff.** `data.frame`s (`cars`, `by_cyl`) cross
  as Arrow IPC. The `lm` object itself isn't tabular, so it's stored as
  RDS (`r_only`) and read straight back by the diagnostics cell with
  full fidelity — an R-only object flowing between R cells that the
  Arrow tier couldn't carry. (A *Python* cell consuming it would get a
  structured "re-export as a data.frame" error instead of a crash.)
- **renv, one click.** The plotting cell needs `ggplot2`, which isn't
  in the harness baseline. It's pinned in `renv.lock` and restored
  automatically when you open the notebook — or via the Environment
  panel's **Initialize renv**. A missing package surfaces a structured
  install hint, not a stack trace.

## Cells

| Cell | What it does |
|---|---|
| `prep` | Tidy `mtcars` into a `data.frame` (`cars`) — model name column, `cyl` as a factor. |
| `summarise` | `aggregate()` mean mpg / hp / wt by cylinder count → `by_cyl`. |
| `fit` | `lm(mpg ~ wt + hp + cyl)`; emit the `model` (RDS), a tidy `coefs` table, and one-row `model_stats`. |
| `plot-mpg` | ggplot2 scatter of mpg vs weight, coloured by cylinder, with per-group fits → inline PNG. |
| `diagnostics` | Read `model` back from RDS; base-graphics 2×2 residual diagnostics → inline PNG. |

## What you need

- **R** on `PATH` (`Rscript`). The notebook's `arrow`, `jsonlite`, and
  `ggplot2` come from `renv.lock` — opening the notebook restores them
  into a project-scoped library automatically; no system installs
  beyond R itself.
- The uv-managed Python venv carries only the notebook harness baseline
  (`pyarrow` / `orjson` / `cloudpickle`); **no Python runs any cell
  here**.

## Running

From the project root:

```bash
uv run strata-notebook --host 127.0.0.1 --port 8765
```

Open `examples/r_mtcars_analysis` from the Strata home page and run the
cells top to bottom. (Note: the headless `strata run` CLI skips R cells
— R executes through the notebook server.)

## Try this

1. **Swap in another plot.** Replace `plot-mpg` with
   `ggplot(by_cyl, aes(factor(cyl), mean_mpg)) + geom_col()` — a bar of
   mean economy by cylinder. Edit, Shift+Enter, watch it re-render.
2. **Reach a non-tabular object across cells.** Add a cell with
   `confint(model)` — `model` resolves from the RDS artifact and you get
   coefficient confidence intervals, no re-fit.
3. **Break, then fix, the environment.** Delete the `renv/` directory
   and reopen: the ggplot cell shows the install hint; click **Initialize
   renv** (or **Install ggplot2**) and re-run.
