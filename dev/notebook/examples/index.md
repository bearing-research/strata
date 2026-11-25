# Example gallery

Every notebook under `examples/` in the repo demonstrates a specific
Strata capability. A card's picture is that notebook in the editor
after a real run. Click through for its cell sources and its README,
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

-   [![The caching tutorial: a bar chart of revenue by region under the cell that plots it.](https://bearingresearch.com/strata/dev/assets/examples/caching-light.webp#only-light)![The caching tutorial: a bar chart of revenue by region under the cell that plots it.](https://bearingresearch.com/strata/dev/assets/examples/caching-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/caching/)

    **[caching](https://bearingresearch.com/strata/dev/examples/caching/)**: a two-minute tutorial on what the
    cache saves and what an edit runs again. `strata tutorial caching` opens
    a copy.

-   [![The Keeling curve notebook: the plotting cell's code above CO₂ at Mauna Loa rising from about 315 to 430 ppm since 1958, beside the average year's cycle.](https://bearingresearch.com/strata/dev/assets/examples/keeling_curve-light.webp#only-light)![The Keeling curve notebook: the plotting cell's code above CO₂ at Mauna Loa rising from about 315 to 430 ppm since 1958, beside the average year's cycle.](https://bearingresearch.com/strata/dev/assets/examples/keeling_curve-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/keeling_curve/)

    **[keeling_curve](https://bearingresearch.com/strata/dev/examples/keeling_curve/)**: NOAA's Mauna Loa
    record, downloaded with `# @fetch`, the curve and its yearly cycle, then a
    published figure whose page shows the code and NOAA's bytes behind it.
    `strata tutorial keeling_curve` opens a copy.

-   [![Iris classification: a pairplot of the four flower measurements, coloured by species.](https://bearingresearch.com/strata/dev/assets/examples/iris_classification-light.webp#only-light)![Iris classification: a pairplot of the four flower measurements, coloured by species.](https://bearingresearch.com/strata/dev/assets/examples/iris_classification-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/iris_classification/)

    **[iris_classification](https://bearingresearch.com/strata/dev/examples/iris_classification/)**: end-to-end
    ML in seven cells (load, split, train, evaluate, plot). The canonical
    multi-cell DAG with caching.

-   [![Pandas basics: a sales DataFrame in the notebook's table viewer.](https://bearingresearch.com/strata/dev/assets/examples/pandas_basics-light.webp#only-light)![Pandas basics: a sales DataFrame in the notebook's table viewer.](https://bearingresearch.com/strata/dev/assets/examples/pandas_basics-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/pandas_basics/)

    **[pandas_basics](https://bearingresearch.com/strata/dev/examples/pandas_basics/)**: core DataFrame
    operations fanning out from one dataset, with per-cell unit tests.

-   [![Titanic: a bar chart of feature importance for the best of three classifiers.](https://bearingresearch.com/strata/dev/assets/examples/titanic_ml-light.webp#only-light)![Titanic: a bar chart of feature importance for the best of three classifiers.](https://bearingresearch.com/strata/dev/assets/examples/titanic_ml-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/titanic_ml/)

    **[titanic_ml](https://bearingresearch.com/strata/dev/examples/titanic_ml/)**: feature engineering, then
    three classifiers trained and the best one evaluated. Closer to a real
    ML workflow.

-   [![Data viewer: a 2,000-row transactions table with search, filter, CSV and Parquet buttons and paging.](https://bearingresearch.com/strata/dev/assets/examples/data_viewer-light.webp#only-light)![Data viewer: a 2,000-row transactions table with search, filter, CSV and Parquet buttons and paging.](https://bearingresearch.com/strata/dev/assets/examples/data_viewer-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/data_viewer/)

    **[data_viewer](https://bearingresearch.com/strata/dev/examples/data_viewer/)**: a DataFrame larger than
    the inline preview, in the interactive viewer. Page through and sort the
    full cached artifact.

-   [![Agent demo: the notebook's introduction above the cell that trains a random forest.](https://bearingresearch.com/strata/dev/assets/examples/agent_demo-light.webp#only-light)![Agent demo: the notebook's introduction above the cell that trains a random forest.](https://bearingresearch.com/strata/dev/assets/examples/agent_demo-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/agent_demo/)

    **[agent_demo](https://bearingresearch.com/strata/dev/examples/agent_demo/)**: the notebook a coding agent
    builds live in the `strata agent` demo. The expensive training cell stays
    cached while only the evaluation runs again.

</div>

## Variant cells

<div class="grid cards" markdown>

-   [![Model variants: a confusion matrix for the active logistic-regression variant.](https://bearingresearch.com/strata/dev/assets/examples/model_variants-light.webp#only-light)![Model variants: a confusion matrix for the active logistic-regression variant.](https://bearingresearch.com/strata/dev/assets/examples/model_variants-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/model_variants/)

    **[model_variants](https://bearingresearch.com/strata/dev/examples/model_variants/)**: three classifier
    variants (logistic regression, random forest, gradient boosting) sharing
    one DAG slot. Switch tabs and the downstream cells run again; the other
    variants stay cached.

-   [![Model variants sweep: three variant cells and the cell that compares their accuracy.](https://bearingresearch.com/strata/dev/assets/examples/model_variants_sweep-light.webp#only-light)![Model variants sweep: three variant cells and the cell that compares their accuracy.](https://bearingresearch.com/strata/dev/assets/examples/model_variants_sweep-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/model_variants_sweep/)

    **[model_variants_sweep](https://bearingresearch.com/strata/dev/examples/model_variants_sweep/)**: sweep
    mode. Every variant runs and one downstream cell compares them as a
    `{variant: value}` dict, plus a `# @per_variant` fan-out.

</div>

## Prompt cells and AI

These call an LLM: set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in the
notebook's **Runtime** panel first.

<div class="grid cards" markdown>

-   [![Review triage: a structured JSON verdict from a prompt cell.](https://bearingresearch.com/strata/dev/assets/examples/review_triage-light.webp#only-light)![Review triage: a structured JSON verdict from a prompt cell.](https://bearingresearch.com/strata/dev/assets/examples/review_triage-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/review_triage/)

    **[review_triage](https://bearingresearch.com/strata/dev/examples/review_triage/)**: a prompt cell with
    `@output_schema`, validated against JSON Schema and retried on a bad
    answer.

-   **[arxiv_classifier](https://bearingresearch.com/strata/dev/examples/arxiv_classifier/)**: arXiv papers
    aggregated on a DataFusion worker, embedded on a GPU worker, classified,
    and summarized by prompt cells. Needs two local workers
    (`run_local_workers.sh`) and an LLM key.

-   **[news_alpha_trader](https://bearingresearch.com/strata/dev/examples/news_alpha_trader/)**: a secret
    manager, an AI pricing lookup and a multi-cell DAG over market news.
    Needs Alpaca keys and an LLM key.

</div>

## SQL cells

The server needs the `sql` and `sql-sqlite` extras for this one
(`uv tool install "strata-notebook[sql,sql-sqlite]"`); the development
environment and the Docker image carry both.

<div class="grid cards" markdown>

-   [![SQL orders report: the Python cell that combines two SQL results, above the report it prints.](https://bearingresearch.com/strata/dev/assets/examples/sql_orders_report-light.webp#only-light)![SQL orders report: the Python cell that combines two SQL results, above the report it prints.](https://bearingresearch.com/strata/dev/assets/examples/sql_orders_report-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/sql_orders_report/)

    **[sql_orders_report](https://bearingresearch.com/strata/dev/examples/sql_orders_report/)**: named
    connections, bind parameters from a Python cell upstream, and
    schema-aware caching.

</div>

## Loops, widgets and library cells

<div class="grid cards" markdown>

-   [![Hill climb loop: the loop's final state as JSON, above the markdown that explains the loop cell.](https://bearingresearch.com/strata/dev/assets/examples/loop_hill_climb-light.webp#only-light)![Hill climb loop: the loop's final state as JSON, above the markdown that explains the loop cell.](https://bearingresearch.com/strata/dev/assets/examples/loop_hill_climb-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/loop_hill_climb/)

    **[loop_hill_climb](https://bearingresearch.com/strata/dev/examples/loop_hill_climb/)**: `# @loop` with
    carried state and an early-termination predicate.

-   [![Widget playground: a slider, number and dropdown above the table they drive.](https://bearingresearch.com/strata/dev/assets/examples/widget_playground-light.webp#only-light)![Widget playground: a slider, number and dropdown above the table they drive.](https://bearingresearch.com/strata/dev/assets/examples/widget_playground-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/widget_playground/)

    **[widget_playground](https://bearingresearch.com/strata/dev/examples/widget_playground/)**: a widget cell
    (slider, number, dropdown) driving a downstream DataFrame and plot, with
    `# @live` recomputing on every change.

-   [![Library cells: helper functions defined in one cell and called from others.](https://bearingresearch.com/strata/dev/assets/examples/library_cells-light.webp#only-light)![Library cells: helper functions defined in one cell and called from others.](https://bearingresearch.com/strata/dev/assets/examples/library_cells-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/library_cells/)

    **[library_cells](https://bearingresearch.com/strata/dev/examples/library_cells/)**: `def` and `class`
    shared across cells through the synthetic-module slicing path, and the
    diagnostic when a helper closes over a runtime value.

</div>

## Mounts and tables

<div class="grid cards" markdown>

-   [![S3 mount: LaGuardia weather rows read from a public bucket, with the mount listed in the notebook settings.](https://bearingresearch.com/strata/dev/assets/examples/s3_mount-light.webp#only-light)![S3 mount: LaGuardia weather rows read from a public bucket, with the mount listed in the notebook settings.](https://bearingresearch.com/strata/dev/assets/examples/s3_mount-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/s3_mount/)

    **[s3_mount](https://bearingresearch.com/strata/dev/examples/s3_mount/)**: a `[[mounts]]` entry makes a
    public S3 prefix a local `pathlib.Path` inside the cell. Needs the `s3`
    extra.

-   **[each_pipeline](https://bearingresearch.com/strata/dev/examples/each_pipeline/)**: `# @each` over 287
    real files from a public bucket, each cached on its own, and
    `# @write_table ... sync` keeping a table in a REST catalog in step.
    Needs Docker and the `s3` and `sql` extras.

</div>

## Markdown

<div class="grid cards" markdown>

-   [![Markdown showcase: the notebook's rendered introduction.](https://bearingresearch.com/strata/dev/assets/examples/markdown_showcase-light.webp#only-light)![Markdown showcase: the notebook's rendered introduction.](https://bearingresearch.com/strata/dev/assets/examples/markdown_showcase-dark.webp#only-dark)](https://bearingresearch.com/strata/dev/examples/markdown_showcase/)

    **[markdown_showcase](https://bearingresearch.com/strata/dev/examples/markdown_showcase/)**: every
    markdown rendering path: headings, lists, tables, code blocks, the
    security guards, and dynamic `Markdown(...)` output.

</div>

## R cells

These need R (`Rscript` on the server's `PATH`).

<div class="grid cards" markdown>

-   **[r_mtcars_analysis](https://bearingresearch.com/strata/dev/examples/r_mtcars_analysis/)**: a pure-R
    notebook: `lm()`, `aggregate()`, and inline ggplot2 and base-graphics
    plots, with `data.frame` and R-only (RDS) handoff between R cells.

-   **[r_lm_vs_sklearn](https://bearingresearch.com/strata/dev/examples/r_lm_vs_sklearn/)**: Python and R in
    one graph. Fit a model with R's `lm()` and compare it with scikit-learn
    over a cross-language Arrow handoff.

</div>
