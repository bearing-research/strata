# Iris Classification — end-to-end ML in seven cells

Classic scikit-learn tutorial, rewritten as a Strata notebook. Shows
what the graph looks like for a realistic ML pipeline: load → explore
→ split → train → evaluate → visualize.

## What it shows

- **DAG branching.** `scatter_plot` and `train_test` both read from
  `load_data`, so editing `load_data` invalidates both branches.
- **Mixed output types.** Cells produce DataFrames (arrow/ipc), trained
  models (pickle), and matplotlib figures (image/png) — all stored
  natively by Strata's serializer.
- **Display outputs.** `_` at the end of a cell (or a trailing
  expression) becomes an inline preview on the cell.

## Cells

| Cell | What it does |
|---|---|
| `load_data` | Loads the iris dataset into a DataFrame. |
| `explore_stats` | Per-feature summary stats. |
| `scatter_plot` | Pair-plot of the four features, colored by class. |
| `train_test` | 80/20 train/test split. |
| `train_model` | Fits a `RandomForestClassifier` (100 trees). |
| `evaluate` | Accuracy + per-class precision/recall. |
| `confusion` | Confusion-matrix heatmap. |

## Running

From the project root:

```bash
uv run strata-notebook --host 127.0.0.1 --port 8765
```

Then open `examples/iris_classification` from the Strata home page.

## Try this

- Change the `test_size` in `train_test`. Cells downstream go stale;
  `scatter_plot` stays ready (it doesn't depend on the split).
- Re-run `evaluate` without re-training — the trained model is cached,
  so only evaluation runs.
