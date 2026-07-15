# Notebook Quickstart

Strata Notebook is an interactive notebook with content-addressed
caching, automatic dependency tracking, and cascade execution. This
quickstart walks through a real three-cell pipeline, then surfaces
the distinctive features (prompt cells, AI assistant, cascade,
cache hits) on top of it.

## 1. Start the server

=== "Docker"

    ```bash
    docker compose up -d --build
    ```

    The compose file runs Strata in personal mode (single-user,
    writes enabled). See [Docker deployment](../deployment/docker.md)
    for details.

=== "From source (one-time setup)"

    Clone the repo, then build the Rust extension and the frontend:

    ```bash
    uv sync                      # Python deps + Rust extension
    cd frontend && npm ci && npm run build && cd ..
    ```

    Run the server (this is the command you'll re-run on subsequent
    sessions):

    ```bash
    uv run strata-notebook
    ```

    Personal mode is the default now; no env vars needed for a
    local single-user run. The npm step above is one-time setup,
    not part of the recurring start command.

Open [http://localhost:8765](http://localhost:8765).

## 2. Create a notebook

Click **New Notebook** on the landing page. Pick a name and a
parent directory under the notebook storage root. Strata creates a
directory containing `notebook.toml`, a per-notebook `pyproject.toml`,
a `cells/` folder, and an empty first cell ready to type into.

By default the storage root is `~/.strata/notebooks` — **not** the
directory you launched from — so new notebooks land there regardless of
your shell's working directory. To put them somewhere else (the current
directory is a common choice), start the server with `--notebook-dir`:

```bash
strata-notebook --notebook-dir .          # current directory
strata-notebook --notebook-dir ~/work/nb  # or any path
```

(or set `STRATA_NOTEBOOK_STORAGE_DIR`). The server prints the active
location on startup.

Each notebook gets its own Python environment, managed by `uv`, so
installing pandas in one notebook doesn't touch another. Everything
autosaves: source goes to `cells/*.py`, runtime state to `.strata/`.
Both diff cleanly in git, no JSON blobs in commits.

### Cell types

Every cell is one of these kinds. The default is Python; the others
are selected via the "+ Add cell" menu or by typing the kind name
in the cell language picker.

| Kind | What it's for |
| --- | --- |
| **Python** | Regular Python code. Most cells. |
| **Prompt** | LLM call as a DAG node. The body is a template with `{{ variable }}` substitution from upstream cells; the response is cached as an artifact like any other cell output. |
| **SQL** | A SQL query against a declared connection. Connection name is an annotation; the result is a pyarrow Table available downstream. |
| **Widget** | A declarative control panel — one control per line (`alpha = slider(0, 1)`, plus number/dropdown/checkbox/text). Each control is an input downstream cells consume; with **⚡ Live** on, dragging one recomputes the cells that depend on it. |
| **Markdown** | Prose between cells. Rendered as HTML; not part of the DAG. |
| **Loop** | A Python cell that re-runs with a `carry` variable threaded across iterations. Annotate with `# @loop max_iter=N carry=state`. |
| **Variant** | Multiple cells share one DAG slot (`# @variant group name`); only the active variant is in the DAG at any time. For A/B-ing different implementations of the same step. |

Library, mount, and worker shapes are annotation-driven on top of
Python cells. See [Cell Types](../notebook/cells.md) for the full
surface.

## 3. Walk through a pipeline

Three cells, one real DAG. We'll load the iris dataset, summarize
by species, and plot a scatter.

**Before typing**, open the **Environment** panel in the sidebar and
add `scikit-learn`, `pandas`, and `matplotlib`. Each `uv add` runs
in the background and the notebook surfaces a sync banner while it's
working. Wait for the banner to clear before running the first cell.

!!! note "Display rendering"
    The output blocks below are shown as text for the doc. In the
    UI, DataFrames render in an interactive grid (page, sort, filter,
    and search over the full cached artifact, with CSV / Parquet
    export), matplotlib figures render as inline PNGs, prompt-cell
    responses render as structured JSON (when `@output_schema` is set)
    or markdown.

### Load the data

```python
import time
import pandas as pd
from sklearn.datasets import load_iris

time.sleep(2)  # simulate the latency of a real fetch
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = pd.Categorical.from_codes(df["target"], iris.target_names)
feature_names = iris.feature_names
df.head()
```

```text title="Output"
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target species
0                5.1               3.5                1.4               0.2       0  setosa
1                4.9               3.0                1.4               0.2       0  setosa
2                4.7               3.2                1.3               0.2       0  setosa
3                4.6               3.1                1.5               0.2       0  setosa
4                5.0               3.6                1.4               0.2       0  setosa
```

Press ++shift+enter++. The first run pauses ~2 seconds (the simulated
fetch) and the DataFrame preview renders below the cell.

### Summarize by species

```python
stats = df.groupby("species", observed=True)[feature_names].mean().round(2)
stats
```

```text title="Output"
            sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
species
setosa                   5.01              3.43               1.46              0.25
versicolor               5.94              2.77               4.26              1.33
virginica                6.59              2.97               5.55              2.03
```

Strata reads this cell's AST, sees it references `df` and `feature_names`
from the loader, and wires an edge. The DAG view in the sidebar
shows the dependency.

### Plot

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
for species, group in df.groupby("species", observed=True):
    ax.scatter(
        group["sepal length (cm)"],
        group["petal length (cm)"],
        label=str(species),
        alpha=0.7,
    )
ax.set_xlabel("Sepal length (cm)")
ax.set_ylabel("Petal length (cm)")
ax.legend()
fig
```

The matplotlib figure renders inline as a PNG.

## 4. Re-run for cache hits

Press ++shift+enter++ on the loader cell again. The 2-second pause
is gone, Strata returned the cached `df` instantly, and the cell
header shows a **cached** badge alongside the duration ("cached ·
5ms" or similar).

The cache is content-addressed: the cache key is a hash of the
cell's source, its upstream artifacts, and the environment lockfile.
Re-running with the same three is always a cache hit. No `@memoize`,
no manual invalidation, and the cached result is byte-identical to
what produced it the first time.

## 5. Edit upstream, watch the cascade

Edit the loader, say, change `time.sleep(2)` to `time.sleep(1)`.
Strata re-analyzes the source, computes a new provenance hash, and
marks the loader **stale**. The summary and plot cells flip stale
too: they referenced `df`, which is no longer the cached value.

Now press ++shift+enter++ on the plot cell. A confirmation banner
appears at the top of the notebook: "2 upstream cells need to
re-run, proceed?" Accept it. Strata runs loader → summary → plot
in topological order, with a live progress strip showing which cell
is executing.

Revert the edit (`time.sleep(1)` back to `time.sleep(2)`) and
re-run: every cell becomes a cache hit on the way through, no work
happens, the cascade short-circuits to milliseconds.

## 6. Add a prompt cell (optional)

Prompt cells are the most distinctive Strata feature beyond caching.
The body is a template; the response is a cached DAG node like any
other cell.

In the cell menu, click **+ Add cell** and pick **Prompt**. Paste
this in:

```text
# @model claude-sonnet-4-6
# @temperature 0.3
# @system You are a botanist describing distinguishing features of iris species.

Given these per-species feature means, write one sentence per
species describing its most distinguishing trait:

{{ stats }}
```

Hit ++shift+enter++. The `{{ stats }}` placeholder gets substituted
with the upstream DataFrame, the prompt goes to the configured LLM,
and the response is stored as an artifact keyed by `(template,
inputs, model config)`. Running again with the same upstream
`stats` and the same template is a cache hit, no second API call.

Needs `ANTHROPIC_API_KEY` (or another provider's key) configured in
the Runtime panel. See [AI Integration](../notebook/ai.md) for
provider setup and [Prompt cells](../notebook/cells.md#prompt-cells)
for the full annotation surface (`@output_schema`, `@validate_retries`,
multi-turn conversations).

## 7. AI Assistant

The top-right **AI Assistant** panel is a conversational sidebar
that can read your notebook, answer questions, and autonomously
edit or run cells. It's separate from prompt cells: the assistant
lives outside the DAG and doesn't create artifacts.

- **Chat mode** (++enter++): stream a response with notebook
  context included.
- **Agent mode** (++shift+enter++): the assistant takes actions
  on the notebook (add/edit/run cells, install packages) with a
  10-step limit and a Cancel button.

Same provider key as prompt cells. See
[AI Integration](../notebook/ai.md) for full details.

## 8. Other display types

The cells above used trailing expressions for the DataFrame and
the matplotlib `Figure`. Other shapes you can put at the end of a
cell, or call from `display()`:

| Trailing expression | Renders as |
| --- | --- |
| pandas DataFrame / Series | Interactive grid: page, sort, filter, and search over the full cached artifact, with CSV / Parquet export. |
| matplotlib `Figure` | Inline PNG. |
| PIL `Image` | Inline PNG. |
| dict / list / primitive | Fenced JSON block. |
| numpy ndarray | Preview header (`shape`, `dtype`) + first few rows. |
| `Markdown("**hi**")` | The `Markdown` helper renders inline HTML. |
| `Image.open(...)`, video, audio | Inline rendering of the appropriate type. |

Use `display(x)` for **multiple** outputs in one cell (each call
adds a new render below the cell). For one trailing expression at
the end, no `display()` needed: the harness auto-displays it.

## 9. Try an example

The repo ships ~12 example notebooks covering every cell type. To
browse them in the UI, stop the running server first (Ctrl+C), then
point the storage root at `examples/`:

```bash
strata-notebook --notebook-dir ./examples
# equivalently: STRATA_NOTEBOOK_STORAGE_DIR=$PWD/examples uv run strata-notebook
```

(For Docker, edit `docker-compose.yml` to mount `./examples:/data/notebooks`
and set the env var to that path.)

Every example is also rendered into the docs site so you can read
the cells before deciding which to open. See the
[Examples catalog](../notebook/examples.md) for the full list,
grouped by feature.

## Cell operations

| Action | How |
| --- | --- |
| Run cell | ++shift+enter++ or ▶ button |
| Add cell | **+** button in gutter or header |
| Delete cell | **×** button in gutter |
| Duplicate cell | **⎘** button in gutter |
| Move cell | **▲** / **▼** buttons in gutter |
| Change cell kind | Language picker in cell header |
| Keyboard help | Press ++question++ |

## What's next

- [Concepts](../notebook/concepts.md) for how the DAG, caching, and cascade work
- [Cell Types](../notebook/cells.md) for the full surface (Python, prompt, SQL, R, widget, markdown, loop, variant)
- [Cell Annotations](../notebook/annotations.md) for `@worker`, `@mount`, `@loop`, and friends
- [Distributed Workers](../notebook/workers.md) for `# @worker gpu-fly` and dispatching to remote compute
- [Import from Jupyter](../notebook/import.md) for `strata import nb.ipynb`
- [Export](../notebook/export.md) for sharing a notebook as a single self-contained markdown or HTML file
- [Environment Management](../notebook/environment.md) for package management and Python versions
- [AI Integration](../notebook/ai.md) for prompt cells + the assistant in depth
- [Comparison with Jupyter, Marimo, Pluto](../notebook/comparison.md)
- [Keyboard Shortcuts](../notebook/keyboard.md)
