# Example gallery

Every notebook under `examples/` in the repo demonstrates a specific
Strata capability. A card's picture is what that notebook makes, taken
from a real run. Click through for its cell sources and its README,
rendered into this site without cloning the repo.

To run the examples, start the server on the repo's `examples/` directory
from a clone:

```bash
uv run strata-notebook --notebook-dir examples
```

Every example then appears under **Open Notebook** on the Strata home page:
open one, click **Run All**, edit a cell and watch only what depends on it
run again. To keep the examples apart from your own work, copy the one you
want under your storage root instead:
`cp -R examples/iris_classification ~/.strata/notebooks/`.

## Walkthroughs: start here

<div class="grid cards" markdown>

-   [![The caching tutorial: a bar chart of revenue by region under the cell that plots it.](../assets/examples/caching-light.webp#only-light)![The caching tutorial: a bar chart of revenue by region under the cell that plots it.](../assets/examples/caching-dark.webp#only-dark)](../examples/caching.md)

    **[caching](../examples/caching.md)**: a two-minute tutorial on what the
    cache saves and what an edit runs again. `strata tutorial caching` opens
    a copy.

-   [![The Keeling curve notebook: the plotting cell's code above CO₂ at Mauna Loa rising from about 315 to 430 ppm since 1958, beside the average year's cycle.](../assets/examples/keeling_curve-light.webp#only-light)![The Keeling curve notebook: the plotting cell's code above CO₂ at Mauna Loa rising from about 315 to 430 ppm since 1958, beside the average year's cycle.](../assets/examples/keeling_curve-dark.webp#only-dark)](../examples/keeling_curve.md)

    **[keeling_curve](../examples/keeling_curve.md)**: NOAA's Mauna Loa
    record, downloaded with `# @fetch`, the curve and its yearly cycle, then a
    published figure whose page shows the code and NOAA's bytes behind it.
    `strata tutorial keeling_curve` opens a copy.

-   [![Iris classification: a pairplot of the four flower measurements, coloured by species.](../assets/examples/iris_classification-light.webp#only-light)![Iris classification: a pairplot of the four flower measurements, coloured by species.](../assets/examples/iris_classification-dark.webp#only-dark)](../examples/iris_classification.md)

    **[iris_classification](../examples/iris_classification.md)**: end-to-end
    ML in seven cells (load, split, train, evaluate, plot). The canonical
    multi-cell DAG with caching.

-   [![Pandas basics: a sales DataFrame in the notebook's table viewer.](../assets/examples/pandas_basics-light.webp#only-light)![Pandas basics: a sales DataFrame in the notebook's table viewer.](../assets/examples/pandas_basics-dark.webp#only-dark)](../examples/pandas_basics.md)

    **[pandas_basics](../examples/pandas_basics.md)**: core DataFrame
    operations fanning out from one dataset, with per-cell unit tests.

-   [![Titanic: a bar chart of feature importance for the best of three classifiers.](../assets/examples/titanic_ml-light.webp#only-light)![Titanic: a bar chart of feature importance for the best of three classifiers.](../assets/examples/titanic_ml-dark.webp#only-dark)](../examples/titanic_ml.md)

    **[titanic_ml](../examples/titanic_ml.md)**: feature engineering, then
    three classifiers trained and the best one evaluated. Closer to a real
    ML workflow.

-   [![Data viewer: a 2,000-row transactions table with search, filter, CSV and Parquet buttons and paging.](../assets/examples/data_viewer-light.webp#only-light)![Data viewer: a 2,000-row transactions table with search, filter, CSV and Parquet buttons and paging.](../assets/examples/data_viewer-dark.webp#only-dark)](../examples/data_viewer.md)

    **[data_viewer](../examples/data_viewer.md)**: a DataFrame larger than
    the inline preview, in the interactive viewer. Page through and sort the
    full cached artifact.

-   [![Agent demo: the notebook's introduction above the cell that trains a random forest.](../assets/examples/agent_demo-light.webp#only-light)![Agent demo: the notebook's introduction above the cell that trains a random forest.](../assets/examples/agent_demo-dark.webp#only-dark)](../examples/agent_demo.md)

    **[agent_demo](../examples/agent_demo.md)**: the notebook a coding agent
    builds live in the `strata agent` demo. The expensive training cell stays
    cached while only the evaluation runs again.

</div>

## Variant cells

<div class="grid cards" markdown>

-   [![Model variants: a confusion matrix for the active logistic-regression variant.](../assets/examples/model_variants-light.webp#only-light)![Model variants: a confusion matrix for the active logistic-regression variant.](../assets/examples/model_variants-dark.webp#only-dark)](../examples/model_variants.md)

    **[model_variants](../examples/model_variants.md)**: three classifier
    variants (logistic regression, random forest, gradient boosting) sharing
    one DAG slot. Switch tabs and the downstream cells run again; the other
    variants stay cached.

-   [![Model variants sweep: three variant cells and the cell that compares their accuracy.](../assets/examples/model_variants_sweep-light.webp#only-light)![Model variants sweep: three variant cells and the cell that compares their accuracy.](../assets/examples/model_variants_sweep-dark.webp#only-dark)](../examples/model_variants_sweep.md)

    **[model_variants_sweep](../examples/model_variants_sweep.md)**: sweep
    mode. Every variant runs and one downstream cell compares them as a
    `{variant: value}` dict, plus a `# @per_variant` fan-out.

</div>

## Prompt cells and AI

These call an LLM: set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in the
notebook's **Runtime** panel first.

<div class="grid cards" markdown>

-   [![Review triage: a structured JSON verdict from a prompt cell.](../assets/examples/review_triage-light.webp#only-light)![Review triage: a structured JSON verdict from a prompt cell.](../assets/examples/review_triage-dark.webp#only-dark)](../examples/review_triage.md)

    **[review_triage](../examples/review_triage.md)**: a prompt cell with
    `@output_schema`, validated against JSON Schema and retried on a bad
    answer.

-   **[arxiv_classifier](../examples/arxiv_classifier.md)**: arXiv papers
    aggregated on a DataFusion worker, embedded on a GPU worker, classified,
    and summarized by prompt cells. Needs two local workers
    (`run_local_workers.sh`) and an LLM key.

-   **[news_alpha_trader](../examples/news_alpha_trader.md)**: a secret
    manager, an AI pricing lookup and a multi-cell DAG over market news.
    Needs Alpaca keys and an LLM key.

</div>

## SQL cells

The server needs the `sql` and `sql-sqlite` extras for this one
(`uv tool install "strata-notebook[sql,sql-sqlite]"`); the development
environment and the Docker image carry both.

<div class="grid cards" markdown>

-   [![SQL orders report: the Python cell that combines two SQL results, above the report it prints.](../assets/examples/sql_orders_report-light.webp#only-light)![SQL orders report: the Python cell that combines two SQL results, above the report it prints.](../assets/examples/sql_orders_report-dark.webp#only-dark)](../examples/sql_orders_report.md)

    **[sql_orders_report](../examples/sql_orders_report.md)**: named
    connections, bind parameters from a Python cell upstream, and
    schema-aware caching.

</div>

## Loops, widgets and library cells

<div class="grid cards" markdown>

-   [![Hill climb loop: the loop's final state as JSON, above the markdown that explains the loop cell.](../assets/examples/loop_hill_climb-light.webp#only-light)![Hill climb loop: the loop's final state as JSON, above the markdown that explains the loop cell.](../assets/examples/loop_hill_climb-dark.webp#only-dark)](../examples/loop_hill_climb.md)

    **[loop_hill_climb](../examples/loop_hill_climb.md)**: `# @loop` with
    carried state and an early-termination predicate.

-   [![Widget playground: a slider, number and dropdown above the table they drive.](../assets/examples/widget_playground-light.webp#only-light)![Widget playground: a slider, number and dropdown above the table they drive.](../assets/examples/widget_playground-dark.webp#only-dark)](../examples/widget_playground.md)

    **[widget_playground](../examples/widget_playground.md)**: a widget cell
    (slider, number, dropdown) driving a downstream DataFrame and plot, with
    `# @live` recomputing on every change.

-   [![Library cells: helper functions defined in one cell and called from others.](../assets/examples/library_cells-light.webp#only-light)![Library cells: helper functions defined in one cell and called from others.](../assets/examples/library_cells-dark.webp#only-dark)](../examples/library_cells.md)

    **[library_cells](../examples/library_cells.md)**: `def` and `class`
    shared across cells through the synthetic-module slicing path, and the
    diagnostic when a helper closes over a runtime value.

</div>

## Mounts and tables

<div class="grid cards" markdown>

-   [![S3 mount: LaGuardia weather rows read from a public bucket, with the mount listed in the notebook settings.](../assets/examples/s3_mount-light.webp#only-light)![S3 mount: LaGuardia weather rows read from a public bucket, with the mount listed in the notebook settings.](../assets/examples/s3_mount-dark.webp#only-dark)](../examples/s3_mount.md)

    **[s3_mount](../examples/s3_mount.md)**: a `[[mounts]]` entry makes a
    public S3 prefix a local `pathlib.Path` inside the cell. Needs the `s3`
    extra.

-   **[each_pipeline](../examples/each_pipeline.md)**: `# @each` over 287
    real files from a public bucket, each cached on its own, and
    `# @write_table ... sync` keeping a table in a REST catalog in step.
    Needs Docker and the `s3` and `sql` extras.

</div>

## Markdown

<div class="grid cards" markdown>

-   [![Markdown showcase: the notebook's rendered introduction.](../assets/examples/markdown_showcase-light.webp#only-light)![Markdown showcase: the notebook's rendered introduction.](../assets/examples/markdown_showcase-dark.webp#only-dark)](../examples/markdown_showcase.md)

    **[markdown_showcase](../examples/markdown_showcase.md)**: every
    markdown rendering path: headings, lists, tables, code blocks, the
    security guards, and dynamic `Markdown(...)` output.

</div>

## R cells

These need R (`Rscript` on the server's `PATH`).

<div class="grid cards" markdown>

-   **[r_mtcars_analysis](../examples/r_mtcars_analysis.md)**: a pure-R
    notebook: `lm()`, `aggregate()`, and inline ggplot2 and base-graphics
    plots, with `data.frame` and R-only (RDS) handoff between R cells.

-   **[r_lm_vs_sklearn](../examples/r_lm_vs_sklearn.md)**: Python and R in
    one graph. Fit a model with R's `lm()` and compare it with scikit-learn
    over a cross-language Arrow handoff.

</div>
