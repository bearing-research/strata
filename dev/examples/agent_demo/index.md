# Agent Demo: a coding agent builds this notebook

This is the notebook from the
[`strata agent`](https://bearingresearch.com/strata/latest/notebook/agent/)
demo: a coding agent (Claude Code) builds it live while you watch in the
terminal UI. It's a plain, runnable notebook, the agent-driven part is just how
it gets written.

## What it shows

- A random forest fit **once** in `train.py`: the "expensive" step.
- An evaluation downstream in `evaluate.py`.
- The point: because Strata is **content-addressed**, editing the evaluation
  recomputes **only** that cell. The model above is reused from cache, never
  retrained. That's the difference from a plain agent + Jupyter, where a rerun
  redoes everything.

## Cells

| cell | what it does |
| --- | --- |
| `intro.md` | Title / one-line framing |
| `train.py` | Fit a `RandomForestClassifier` on synthetic data → `model` |
| `evaluate.py` | `accuracy` + `conf_matrix` from `model` |

## Running

From the repository root (the notebook's own venv does not include Strata;
`strata run` builds it from `pyproject.toml` before the first cell):

```bash
uv run strata run examples/agent_demo --force   # run every cell
```

## Reproducing the demo

To watch a coding agent build this notebook from scratch, see `RECORDING.md` in
this directory, it has the exact `strata agent` command and the two prompts,
plus the screen-capture settings.

# Agent demo: a coding agent builds this notebook

A random forest, trained once and cached. Because Strata is content-addressed,
editing the evaluation below recomputes only that cell, the model is never
retrained.

## train

<sub>**kind** python</sub>

```python
# The expensive step: fit the model. Strata caches this by provenance, so a
# downstream edit reuses the trained model instead of retraining it.
import numpy as np
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(8_000, 20))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = RandomForestClassifier(
    n_estimators=100, max_depth=8, n_jobs=-1, random_state=0
).fit(X, y)
```

## evaluate

<sub>**kind** python</sub>

```python
# Downstream of `model`. Edit this cell and only this cell recomputes; the
# model above stays cached (no retraining).
from sklearn.metrics import confusion_matrix

accuracy = float(model.score(X, y))
conf_matrix = confusion_matrix(y, model.predict(X))
```
