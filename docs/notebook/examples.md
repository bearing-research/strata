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
| [`iris_classification`](../examples/iris_classification.md) | End-to-end ML in seven cells: load → split → train → evaluate → plot. The canonical "multi-cell DAG with caching" demo. |
| [`pandas_basics`](../examples/pandas_basics.md) | Core DataFrame operations and rich display outputs. |
| [`titanic_ml`](../examples/titanic_ml.md) | Feature engineering plus comparison of two classifiers, closer to a real ML workflow. |
| [`data_viewer`](../examples/data_viewer.md) | A DataFrame larger than the inline preview, in the interactive viewer: page through and sort the full cached artifact. |
| [`agent_demo`](../examples/agent_demo.md) | The notebook a coding agent builds live in the `strata agent` demo: an expensive training cell stays cached while only the evaluation re-runs. |

## Variant cells

| Notebook | What you'll see |
| --- | --- |
| [`model_variants`](../examples/model_variants.md) | Three classifier variants (logistic regression, random forest, gradient boosting) sharing one DAG slot. Switch tabs, re-cascade downstream; the others stay cached. |
| [`model_variants_sweep`](../examples/model_variants_sweep.md) | Sweep mode: every variant runs and one downstream cell compares them as a `{variant: value}` dict, plus a `# @per_variant` fan-out. |

## Prompt cells and AI

| Notebook | What you'll see |
| --- | --- |
| [`arxiv_classifier`](../examples/arxiv_classifier.md) | AI-powered paper classification: prompt cells + distributed workers in one pipeline. |
| [`review_triage`](../examples/review_triage.md) | Structured-output prompt cells with `@output_schema` + the validate-and-retry loop. |
| [`news_alpha_trader`](../examples/news_alpha_trader.md) | Secret manager + AI pricing lookup + multi-cell DAG. |

## SQL cells

| Notebook | What you'll see |
| --- | --- |
| [`sql_orders_report`](../examples/sql_orders_report.md) | Named connections, bind parameters from Python upstream, schema-aware caching. |

## Loop cells

| Notebook | What you'll see |
| --- | --- |
| [`loop_hill_climb`](../examples/loop_hill_climb.md) | `# @loop` with carry state and an early-termination predicate. |

## Widget cells

| Notebook | What you'll see |
| --- | --- |
| [`widget_playground`](../examples/widget_playground.md) | A widget cell (slider, number, dropdown) driving a downstream DataFrame, with `# @live` recomputing on every change. |

## Library cells

| Notebook | What you'll see |
| --- | --- |
| [`library_cells`](../examples/library_cells.md) | Cross-cell `def` / `class` sharing via the synthetic-module slicing path. |

## Mounts

| Notebook | What you'll see |
| --- | --- |
| [`s3_mount`](../examples/s3_mount.md) | `# @mount` annotation makes an S3 prefix available as a local `pathlib.Path` inside the cell. |

## Markdown rendering

| Notebook | What you'll see |
| --- | --- |
| [`markdown_showcase`](../examples/markdown_showcase.md) | Every markdown rendering path: headings, lists, tables, code blocks, security guards, dynamic `Markdown(...)` output. |

## R cells

| Notebook | What you'll see |
| --- | --- |
| [`r_mtcars_analysis`](../examples/r_mtcars_analysis.md) | A pure-R notebook - `lm()`, `aggregate()`, and inline ggplot2 + base-graphics plots, with `data.frame` and R-only (RDS) handoff between R cells. |
| [`r_lm_vs_sklearn`](../examples/r_lm_vs_sklearn.md) | Mixed Python + R: fit a model with R's `lm()` and compare side-by-side with scikit-learn over a cross-language Arrow handoff. |
