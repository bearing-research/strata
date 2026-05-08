# Notebook Quickstart

Strata Notebook is an interactive notebook with content-addressed caching, automatic dependency tracking, and cascade execution.

## 1. Start the Server

=== "Docker"

    ```bash
    docker compose up -d --build
    ```

=== "From source"

    ```bash
    uv sync
    cd frontend && npm ci && npm run build && cd ..
    STRATA_DEPLOYMENT_MODE=personal uv run strata-server
    ```

Open [http://localhost:8765](http://localhost:8765).

## 2. Create a Notebook

Click **New Notebook** on the landing page. Choose a name and a parent directory under the notebook storage root.

Each notebook gets its own Python environment (managed by `uv`), so packages installed in one notebook don't affect others.

## 3. Write and Run Cells

Add a cell and type:

```python
x = 1
x + 1
```

Press ++shift+enter++ to run. The result appears below the cell.

## 4. Multi-Cell Dependencies

Add a second cell:

```python
y = x * 10
print(f"y = {y}")
```

Strata automatically detects that this cell references `x` from the first cell. The DAG in the sidebar shows the dependency arrow.

!!! info "Cascade execution"
If you change the first cell and re-run the second, Strata detects the staleness and offers to cascade-execute both cells in the correct order.

## 5. Rich Display Outputs

### Markdown

The `Markdown` helper is injected into every cell's namespace:

```python
Markdown("# Hello\n\n- item one\n- item two")
```

### Matplotlib

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
```

!!! tip "Install matplotlib first"
Open the **Environment** panel in the sidebar and add `matplotlib` before running plot cells.

### Multiple Outputs

A cell can produce multiple visible outputs:

```python
display(Markdown("## Summary"))
42
```

### DataFrames

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol"],
    "score": [95, 87, 92],
})
df
```

DataFrames render as scrollable tables with column headers and row counts.

## 6. Manage Packages

Open the **Environment** panel in the sidebar to:

- Install and remove packages
- Import from `requirements.txt`
- Export dependencies
- Sync or rebuild the environment

See [Environment Management](../notebook/environment.md) for details.

## 7. Caching

When you re-run a cell with the same inputs and source, Strata returns the cached result instantly. The cell shows a **⚡ cached** badge with timing.

Change the source or any upstream cell, and the cache is automatically invalidated.

## 8. AI Assistant

The top-right **AI Assistant** panel is a conversational sidebar that can read
your notebook, answer questions, and autonomously edit or run cells. It's
separate from prompt cells — the assistant lives outside the DAG and doesn't
create artifacts.

- **Chat mode** (++enter++): stream a response with notebook context included.
- **Agent mode** (++shift+enter++): the assistant takes actions on the notebook
  (add/edit/run cells, install packages) with a 10-step limit and a Cancel
  button.

Requires `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, or
`MISTRAL_API_KEY` in the Runtime panel. See
[LLM Integration](../notebook/llm.md) for full details.

## 9. Appearance

The top-right theme toggle cycles **system → light → dark**. Your choice
persists per browser via `localStorage`; the **system** mode follows your
OS's `prefers-color-scheme` and flips automatically when it changes.

## 10. Try an Example

Example notebooks must live under the configured notebook storage root
(`STRATA_NOTEBOOK_STORAGE_DIR`, default `/tmp/strata-notebooks`).

Copy one of the bundled examples there first:

```bash
cp -R examples/iris_classification /tmp/strata-notebooks/
```

Then use **Open Existing** and open:

```text
/tmp/strata-notebooks/iris_classification
```

Other bundled examples:

| Directory                      | What it shows                                           |
| ------------------------------ | ------------------------------------------------------- |
| `examples/iris_classification` | Basic ML workflow with sklearn                          |
| `examples/pandas_basics`       | DataFrame manipulation and plotting                     |
| `examples/titanic_ml`          | End-to-end classifier                                   |
| `examples/arxiv_classifier`    | LLM-powered paper classification (prompt cells)         |
| `examples/review_triage`       | Structured-output prompt cells with `@output_schema`    |
| `examples/loop_hill_climb`     | `@loop` cell with carry state and termination predicate |
| `examples/news_alpha_trader`   | Secret manager, LLM pricing lookup, multi-cell DAG      |
| `examples/s3_mount`            | `@mount` annotations for S3 / file access               |

See the `examples/` directory in the repo for the full list.

## Cell Operations

| Action         | How                              |
| -------------- | -------------------------------- |
| Run cell       | ++shift+enter++ or ▶ button      |
| Add cell       | **+** button in gutter or header |
| Delete cell    | **×** button in gutter           |
| Duplicate cell | **⎘** button in gutter           |
| Move cell      | **▲** / **▼** buttons in gutter  |
| Keyboard help  | Press ++question++               |

## What's Next

- [Concepts](../notebook/concepts.md) — how the DAG, caching, and cascade work
- [Environment](../notebook/environment.md) — package management and Python versions
- [Keyboard Shortcuts](../notebook/keyboard.md) — all available shortcuts
- [Docker deployment](../deployment/docker.md) — run in a container
