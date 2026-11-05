# R cells

R is a first-class notebook language alongside Python. R cells run R source via the system `Rscript`, with the same provenance + caching + Arrow-IPC artifact pipeline as Python cells. Data crosses the language boundary as `arrow/ipc`: a `pandas.DataFrame` produced by a Python cell becomes a `data.frame` in the next R cell; an R `data.frame` (or `tibble`) becomes a `pandas.DataFrame` in the next Python cell. Non-tabular R values (S3 objects, lists with classes, environments) are stored as RDS and tagged `r_only=true` - a downstream Python cell that consumes one fails with a structured `StrataRArtifactError` rather than a confusing `NameError`.

## R environments

R environments are managed from the **Environment** panel, at parity with Python's uv-backed venv:

- **System R by default.** A notebook with no `renv.lock` runs R cells against your system R library. The R card reads **System R** and shows the detected R version - cells work immediately as long as `arrow` + `jsonlite` are available system-wide.
- **One-click renv bootstrap.** **Initialize renv** creates a project-scoped, reproducible environment: it installs `renv` if missing, inits a bare project library, installs `jsonlite` + `arrow`, and snapshots to `renv.lock`. Progress streams live on the card (the first run compiles `arrow` from source, which takes a few minutes). When it finishes the card flips to **In sync**.
- **Per-package install.** A missing-package error in an R cell surfaces a structured hint with an **Install** button that runs `renv::install()` + `renv::snapshot()` for the named package.
- **Automatic restore on open.** Opening a notebook that has an `renv.lock` restores the project library automatically - the `uv sync` analogue for R. Editing `renv.lock` invalidates R cells' cache the same way editing `uv.lock` invalidates Python cells'.

`renv.lock` is committed config (like `uv.lock`); the built `renv/library/` is gitignored.

On a server that isolates cell code (service mode, or any server with `STRATA_NOTEBOOK_HARNESS_USER` set), **Initialize renv** and **Install** are refused: renv builds packages from source, which runs their code, and writes into the notebook directory. Add packages to `renv.lock` where the notebook is authored and commit it; the server restores that lock as the harness user when the notebook opens. See [What a cell can read](../../deployment/service-mode.md#what-a-cell-can-read).

## Plots

R cells display plots inline, like a Python cell's matplotlib figure. Base graphics (`plot()`, `hist()`, …) and grid-based plots (ggplot2, lattice) are captured to PNG and rendered in the cell output. A bare trailing plot object auto-renders - a last-line `p` where `p <- ggplot(...)` shows the plot without an explicit `print(p)`, mirroring the R console. A cell that draws several plots produces an ordered list of image outputs.

## Warm Rscript pool

**Warm Rscript pool.** Notebooks containing R cells keep a small pool of
pre-spawned R workers (R startup, `.Rprofile`/renv activation, and
`jsonlite`/`arrow` loads already paid), so an R cell run skips the ~1–2s
interpreter cold-start. Workers are single-shot - one cell each, then
replaced - preserving per-cell isolation; editing `renv.lock` drains and
respawns the pool. Adding the first R cell starts the pool, so a notebook
that began as pure Python does not need reopening. Pure-Python notebooks
and machines without `Rscript` never start one.

R execution and display are complete; remaining R polish is tracked on GitHub: [#84](https://github.com/bearing-research/strata/issues/84) (cross-language run-all batching).

## Functions

A cell can call a function another cell defined, or pass it by name to `sapply` and the like. The function travels with what it names from its own cell, and the rest of that cell stays behind, so a function beside a large data frame stays small.

Two limits:

- Python cells can't read an R function.
- Names read only inside a function's body aren't dependencies yet. If `g <- function() f(1)` needs `f` from another cell, call `f` at the top level of the cell too, or pass it to `g` as an argument.

## What you need today

- R `>= 4.2` on `PATH` (Strata calls `Rscript`).
- The `arrow` and `jsonlite` R packages - either available in the system R library, or installed into a project library via **Initialize renv** in the Environment panel (which installs both for you).

## Example

See [`examples/r_lm_vs_sklearn/`](https://github.com/bearing-research/strata/tree/main/examples/r_lm_vs_sklearn) for a working notebook: a Python cell synthesises a housing dataset, an R cell fits `lm(price ~ sqft + bedrooms + age + location, data = housing_train)` and returns tidy coefficient + model-stats + prediction `data.frame`s, a Python cell fits the same model with `LinearRegression` and prints a side-by-side comparison. The R formula's auto-factor-encoding of `location` is the one-line stats-edge moment.


## What if Rscript isn't installed?

An R cell whose harness can't find `Rscript` on `PATH` returns a clean cell-level failure with the message `Rscript not found on PATH. Install R (https://cran.r-project.org/) and reopen the notebook.` No server crash, no other cells affected.


## R-cell annotations

See [R annotations](../../reference/annotations/r.md).
