# Cell kinds

Strata Notebook has seven cell kinds:

| Kind         | What it runs                                     | Created by                                                    |
| ------------ | ------------------------------------------------ | ------------------------------------------------------------- |
| [**Python**](https://bearingresearch.com/strata/dev/guides/cells/python/)   | Python source in the notebook's venv             | The default, pick **Python** from the **+ Add cell** menu    |
| [**Prompt**](https://bearingresearch.com/strata/dev/guides/cells/prompt/)   | A text template sent to an AI model              | Pick **Prompt** from the **+ Add cell** menu                  |
| [**SQL**](https://bearingresearch.com/strata/dev/guides/cells/sql/)      | A query against a connected database             | Pick **SQL** from the **+ Add cell** menu                     |
| [**R**](https://bearingresearch.com/strata/dev/guides/cells/r/)        | R source via the system `Rscript`                | Pick **R** from the **+ Add cell** menu                      |
| [**Widget**](https://bearingresearch.com/strata/dev/guides/cells/widget/)   | A declarative control panel (slider, dropdown …) | Pick **Widget** from the **+ Add cell** menu                  |
| **Markdown** | Prose between cells, rendered as markdown        | Pick **Markdown** from the **+ Add cell** menu                |
| [**Loop**](https://bearingresearch.com/strata/dev/guides/cells/loops-and-variants/)     | A Python cell executed N times in a row          | Add a Python cell, then put a `# @loop` annotation at the top |

Compute cells participate in the DAG, cache by provenance hash, and can be routed to remote workers; widget and markdown cells are declarative (no subprocess). Pick the kind that matches the shape of the computation; each has its own page.

See [The DAG and staleness](https://bearingresearch.com/strata/dev/concepts/dag-and-staleness/) for the execution model; see [Cell Annotations](https://bearingresearch.com/strata/dev/reference/annotations/) for the full per-annotation reference.

## Markdown Cells

Plain prose between cells, rendered with `markdown-it` + `DOMPurify` for safe HTML output. Useful for section headings, methodology notes, and annotating decision points in a notebook. Markdown cells are **not** part of the DAG - they don't produce artifacts, don't participate in cascade execution, and don't have an `id` / variable that downstream cells can reference. They survive saves verbatim; a markdown export escapes raw HTML, as the UI does.

```markdown
## Stage 1: Load + Clean

The next two cells pull last quarter's events and drop rows with
missing timestamps. The clean DataFrame `events_clean` is what
everything downstream reads.
```

Create a markdown cell via the **+ Add cell** menu (pick "Markdown") or by setting `language = "markdown"` on a cell entry in `notebook.toml`. The cell source lives in `cells/<id>.md` rather than `.py`.

See [Markdown showcase](https://bearingresearch.com/strata/dev/examples/markdown_showcase/) for what the supported renderers handle (headings, lists, tables, code fences, fenced HTML attributes) and what's stripped by the security guard.

## Choosing between kinds

| Reach for a… | When you want…                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| Python cell  | Ordinary computation. Default.                                                                         |
| Prompt cell  | An AI response as a first-class, cached, DAG-participating artifact.                                   |
| SQL cell     | A query against a connected database, with bind parameters, schema discovery, and probe-based caching. |
| Loop cell    | Iterative refinement where pausing or forking from an intermediate state matters.                      |
| R cell       | A computation where R's stats / formula syntax / domain packages are the right tool, while keeping the rest of the pipeline in Python. |

Mixing is encouraged: a typical pipeline might be a SQL cell for extraction → Python cells for transformation → an R cell for the linear model → a prompt cell for narrative summarisation.

Any kind of cell can also live inside a [variant group](https://bearingresearch.com/strata/dev/guides/cells/loops-and-variants/#variant-cells): a tabbed slot where multiple cells share one place in the DAG and you switch between them without forking the notebook.
