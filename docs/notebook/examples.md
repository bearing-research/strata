# Examples

Every notebook under `examples/` in the repo demonstrates a specific
Strata capability. Each example is also rendered into the docs site
automatically — click through to read the cell sources and any cached
outputs without cloning the repo.

To run an example locally, copy it under your notebook storage root
(`STRATA_NOTEBOOK_STORAGE_DIR`, default `/tmp/strata-notebooks`) and
open it from the Strata home page:

```bash
cp -R examples/iris_classification /tmp/strata-notebooks/
```

## Walkthroughs — start here

| Notebook | What you'll see |
| --- | --- |
| [`iris_classification`](../examples/iris_classification.md) | End-to-end ML in seven cells — load → split → train → evaluate → plot. The canonical "multi-cell DAG with caching" demo. |
| [`pandas_basics`](../examples/pandas_basics.md) | Core DataFrame operations and rich display outputs. |
| [`titanic_ml`](../examples/titanic_ml.md) | Feature engineering plus comparison of two classifiers — closer to a real ML workflow. |

## Variant cells

| Notebook | What you'll see |
| --- | --- |
| [`model_variants`](../examples/model_variants.md) | Three classifier variants (logistic regression, random forest, gradient boosting) sharing one DAG slot. Switch tabs, re-cascade downstream; the others stay cached. |

## Prompt cells and AI

| Notebook | What you'll see |
| --- | --- |
| [`arxiv_classifier`](../examples/arxiv_classifier.md) | AI-powered paper classification — prompt cells + distributed workers in one pipeline. |
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
| [`markdown_showcase`](../examples/markdown_showcase.md) | Every markdown rendering path — headings, lists, tables, code blocks, security guards, dynamic `Markdown(...)` output. |
