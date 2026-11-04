# Examples

Every notebook under `examples/` in the repo demonstrates a specific
Strata capability. Each example is also rendered into the docs site
automatically: click through to read the cell sources and any cached
outputs without cloning the repo.

To run examples locally, point the server's storage root at the
repo's `examples/` directory and start in personal mode:

```bash
STRATA_NOTEBOOK_STORAGE_DIR=$PWD/examples \
STRATA_DEPLOYMENT_MODE=personal \
  uv run strata-notebook
```

Every example then appears on the Strata home page. No copying
required: open the notebook, click Run, edit, watch the cascade.

(If you'd rather keep examples separate from your own work, copy
the directory you want under your existing storage root instead:
`cp -R examples/iris_classification ~/.strata/notebooks/`.)

## Walkthroughs: start here

| Notebook | What you'll see |
| --- | --- |
| [`caching`](https://bearingresearch.com/strata/dev/dev/examples/caching/) | A two-minute tutorial: what the cache saves, and what an edit runs again. `strata tutorial caching` opens a copy. |
| [`iris_classification`](https://bearingresearch.com/strata/dev/dev/examples/iris_classification/) | End-to-end ML in seven cells: load → split → train → evaluate → plot. The canonical "multi-cell DAG with caching" demo. |
| [`pandas_basics`](https://bearingresearch.com/strata/dev/dev/examples/pandas_basics/) | Core DataFrame operations and rich display outputs. |
| [`titanic_ml`](https://bearingresearch.com/strata/dev/dev/examples/titanic_ml/) | Feature engineering plus comparison of two classifiers, closer to a real ML workflow. |
| [`data_viewer`](https://bearingresearch.com/strata/dev/dev/examples/data_viewer/) | A DataFrame larger than the inline preview, in the interactive viewer: page through and sort the full cached artifact. |
| [`agent_demo`](https://bearingresearch.com/strata/dev/dev/examples/agent_demo/) | The notebook a coding agent builds live in the `strata agent` demo: an expensive training cell stays cached while only the evaluation re-runs. |

## Variant cells

| Notebook | What you'll see |
| --- | --- |
| [`model_variants`](https://bearingresearch.com/strata/dev/dev/examples/model_variants/) | Three classifier variants (logistic regression, random forest, gradient boosting) sharing one DAG slot. Switch tabs, re-cascade downstream; the others stay cached. |
| [`model_variants_sweep`](https://bearingresearch.com/strata/dev/dev/examples/model_variants_sweep/) | Sweep mode: every variant runs and one downstream cell compares them as a `{variant: value}` dict, plus a `# @per_variant` fan-out. |

## Prompt cells and AI

| Notebook | What you'll see |
| --- | --- |
| [`arxiv_classifier`](https://bearingresearch.com/strata/dev/dev/examples/arxiv_classifier/) | AI-powered paper classification: prompt cells + distributed workers in one pipeline. |
| [`review_triage`](https://bearingresearch.com/strata/dev/dev/examples/review_triage/) | Structured-output prompt cells with `@output_schema` + the validate-and-retry loop. |
| [`news_alpha_trader`](https://bearingresearch.com/strata/dev/dev/examples/news_alpha_trader/) | Secret manager + AI pricing lookup + multi-cell DAG. |

## SQL cells

The server needs the `sql` and `sql-sqlite` extras for this one
(`uv tool install "strata-notebook[sql,sql-sqlite]"`); the Docker image
carries both.

| Notebook | What you'll see |
| --- | --- |
| [`sql_orders_report`](https://bearingresearch.com/strata/dev/dev/examples/sql_orders_report/) | Named connections, bind parameters from Python upstream, schema-aware caching. |

## Loop cells

| Notebook | What you'll see |
| --- | --- |
| [`loop_hill_climb`](https://bearingresearch.com/strata/dev/dev/examples/loop_hill_climb/) | `# @loop` with carry state and an early-termination predicate. |

## Widget cells

| Notebook | What you'll see |
| --- | --- |
| [`widget_playground`](https://bearingresearch.com/strata/dev/dev/examples/widget_playground/) | A widget cell (slider, number, dropdown) driving a downstream DataFrame, with `# @live` recomputing on every change. |

## Library cells

| Notebook | What you'll see |
| --- | --- |
| [`library_cells`](https://bearingresearch.com/strata/dev/dev/examples/library_cells/) | Cross-cell `def` / `class` sharing via the synthetic-module slicing path. |

## Mounts

| Notebook | What you'll see |
| --- | --- |
| [`each_pipeline`](https://bearingresearch.com/strata/dev/dev/examples/each_pipeline/) | `# @each` over 287 real files from a public bucket, each cached on its own, and `# @write_table ... sync` keeping a table in a REST catalog in step. Needs Docker. |
| [`s3_mount`](https://bearingresearch.com/strata/dev/dev/examples/s3_mount/) | `# @mount` annotation makes an S3 prefix available as a local `pathlib.Path` inside the cell. |

## Markdown rendering

| Notebook | What you'll see |
| --- | --- |
| [`markdown_showcase`](https://bearingresearch.com/strata/dev/dev/examples/markdown_showcase/) | Every markdown rendering path: headings, lists, tables, code blocks, security guards, dynamic `Markdown(...)` output. |

## R cells

| Notebook | What you'll see |
| --- | --- |
| [`r_mtcars_analysis`](https://bearingresearch.com/strata/dev/dev/examples/r_mtcars_analysis/) | A pure-R notebook - `lm()`, `aggregate()`, and inline ggplot2 + base-graphics plots, with `data.frame` and R-only (RDS) handoff between R cells. |
| [`r_lm_vs_sklearn`](https://bearingresearch.com/strata/dev/dev/examples/r_lm_vs_sklearn/) | Mixed Python + R: fit a model with R's `lm()` and compare side-by-side with scikit-learn over a cross-language Arrow handoff. |
