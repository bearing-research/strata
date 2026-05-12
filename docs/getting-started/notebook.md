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

## 3. Walk Through a Pipeline

We'll load the classic iris dataset, summarize it by species, and plot a scatter. Three cells, one real DAG — enough to exercise caching, cascading, and rich displays in motion.

Open the **Environment** panel in the sidebar and add `scikit-learn`, `pandas`, and `matplotlib`.

### Load the data

```python
import time
import pandas as pd
from sklearn.datasets import load_iris

time.sleep(2)  # pretend this is an expensive fetch
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

Press ++shift+enter++. The first run pauses ~2 seconds (the simulated fetch) and the DataFrame preview renders below the cell.

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

Strata reads this cell's AST, sees it references `df` and `feature_names` from the loader, and wires an edge. The DAG view in the sidebar shows the dependency.

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

Press ++shift+enter++ on the loader cell again. The 2-second pause is gone — Strata returned the cached `df` instantly and the cell badge reads **⚡ cached**.

Caching is content-addressed: the cache key is a hash of the cell's source, its upstream artifacts, and the environment lockfile. Re-running with the same three is always a cache hit. No `@memoize`, no manual invalidation, and the cached result is byte-identical to what produced it.

## 5. Edit upstream, watch the cascade

Edit the loader — say, change `time.sleep(2)` to `time.sleep(1)`. Strata re-analyzes the source, computes a new provenance hash, and marks the loader **stale**. The summary and plot cells flip stale too: they referenced `df`, which is no longer the cached value.

Now press ++shift+enter++ on the plot cell. Strata builds a **cascade plan** — loader → summary → plot — and runs them in topological order. Revert the edit and re-run: every cell becomes a cache hit on the way through, no work happens, the cascade short-circuits to milliseconds.

## 6. Other display types

The loader and summary cells above used a trailing expression for the DataFrame render; the plot cell did the same with a matplotlib `Figure`. A few more shapes you can put at the end of any cell:

| Use this              | Renders as                                                                                |
| --------------------- | ----------------------------------------------------------------------------------------- |
| Any value (trailing)  | Its `repr`. DataFrames → scrollable tables, matplotlib `Figure` → inline PNG, dict → JSON. |
| `display(x)`          | Emits one display output; call it multiple times in one cell to stack outputs.            |
| `Markdown("**hi**")`  | The `Markdown` helper is injected into every cell's namespace and renders as HTML.        |

## 7. Manage Packages

The **Environment** panel in the sidebar lets you install/remove packages, import from `requirements.txt`, export dependencies, and sync the environment. Each notebook has its own venv (managed by `uv`) so packages don't cross-contaminate.

See [Environment Management](../notebook/environment.md) for details.

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
[AI Integration](../notebook/ai.md) for full details.

## 9. Appearance

The top-right theme toggle cycles **system → light → dark**. Your choice
persists per browser via `localStorage`; the **system** mode follows your
OS's `prefers-color-scheme` and flips automatically when it changes.

## 10. Try an Example

Point the server's storage root at the repo's `examples/` directory
and every bundled example appears on the home page:

```bash
STRATA_NOTEBOOK_STORAGE_DIR=$PWD/examples \
STRATA_DEPLOYMENT_MODE=personal \
  uv run strata-server
```

See the [Examples catalog](../notebook/examples.md) for the full list,
grouped by which feature each one demonstrates (variant cells, prompt
cells, SQL cells, loop cells, library cells, mounts, AI integration).
Every example is also rendered into the docs site so you can read the
cells before deciding which to open.

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
