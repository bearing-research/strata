# R annotations

The same annotation parser handles both Python and R cells (`#`-prefixed comments at the top of the source). Supported on R cells:

| Annotation                 | Effect                                                                  |
| -------------------------- | ----------------------------------------------------------------------- |
| `# @name <text>`           | Cell display name in the UI.                                            |
| `# @env KEY=value`         | Sets `Sys.getenv("KEY")` for the cell's process.                        |
| `# @mount data file:///x`  | Binds `data` inside the R cell to the mount's local path (a character string - R has no `pathlib.Path`).  |
| `# @timeout 60`            | Per-cell execution timeout in seconds.                                  |
| `# @fetch data <url>`      | Downloads a URL as a recorded input; `data` is the file's path as a string. |
| `# @output_dir ckpt`       | Gives the cell an empty output folder; `ckpt` is its path as a string. An R cell cannot read another cell's folder yet. |
| `# @nocache`               | Always re-execute.                                                      |
| `# @stochastic`            | Re-execute every time and keep each run's outputs as their own version, as on a [Python cell](https://bearingresearch.com/strata/dev/reference/annotations/python/#stochastic). |
| `# @worker gpu-box`        | Runs the cell on an executor worker with R (see below).                 |
| `# @write_table df lake:lab.t append` | Writes a variable into an Iceberg table when the run succeeds, as on a [Python cell](https://bearingresearch.com/strata/dev/reference/annotations/python/#write_table). |
| `# @after seed`            | An edge from another cell with no variable, keyed on its last run, as on a [Python cell](https://bearingresearch.com/strata/dev/reference/annotations/python/#after). |
| `# @variant <group> <name>` | Makes the cell one alternative in a [variant group](https://bearingresearch.com/strata/dev/reference/annotations/loops-and-variants/#variant). |

An R cell with a worker runs `harness.R` on that worker (a `strata-worker` of
this version or later), under the worker's
`Rscript` and its R library, which needs `jsonlite` and `arrow`
(`docker build -f worker.Dockerfile --build-arg WITH_R=true`). Its outputs come
back the way a local run's do, so it is a cache hit here afterwards, a
downstream Python cell reads its data frame, and an R-only value returns as RDS.
A worker whose R has `renv` restores the notebook's `renv.lock` once per lock
and runs the cell against that library; one without `renv` uses its own R
library (see [Workers](https://bearingresearch.com/strata/dev/notebook/workers/)). An `@fetch` on an R cell
is read on this machine, so an R cell with both `@fetch` and `@worker` fails;
run it locally, or fetch in an upstream Python cell.

Prompt-cell annotations (`@output_schema` etc.) do not apply to R cells, `@loop` and `@loop_until` are an error there (`loop_not_python`) and the cell does not run, and an R cell that declares `@dataset` or `@table` fails; read the dataset or table in an upstream Python cell.
