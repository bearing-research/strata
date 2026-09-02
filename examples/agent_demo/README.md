# Agent Demo — a coding agent builds this notebook

This is the notebook from the
[`strata agent`](https://bearing-research.github.io/strata/latest/notebook/agent/)
demo: a coding agent (Claude Code) builds it live while you watch in the
terminal UI. It's a plain, runnable notebook — the agent-driven part is just how
it gets written.

## What it shows

- A random forest fit **once** in `train.py` — the "expensive" step.
- An evaluation downstream in `evaluate.py`.
- The point: because Strata is **content-addressed**, editing the evaluation
  recomputes **only** that cell. The model above is reused from cache — never
  retrained. That's the difference from a plain agent + Jupyter, where a rerun
  redoes everything.

## Cells

| cell | what it does |
| --- | --- |
| `intro.md` | Title / one-line framing |
| `train.py` | Fit a `RandomForestClassifier` on synthetic data → `model` |
| `evaluate.py` | `accuracy` + `conf_matrix` from `model` |

## Running

```bash
uv sync                       # build the notebook's venv from pyproject.toml
uv run strata run . --force   # run every cell
```

## Reproducing the demo

To watch a coding agent build this notebook from scratch, see `RECORDING.md` in
this directory — it has the exact `strata agent` command and the two prompts,
plus the screen-capture settings.
