# Titanic — feature engineering + model comparison

A canonical "first Kaggle notebook" rewritten for Strata. Shows a
feature-engineering stage that fans out into two model trainers whose
metrics get compared in a final cell.

## What it shows

- **Shared upstream with two downstream branches** — `features` feeds
  both `train_model` and a second branch (implicit via `compare`), so
  editing `features` invalidates both branches.
- **Ordered display outputs** — `explore` produces multiple charts
  side-by-side, each rendered as a separate display output.
- **Typed primitives round-trip** — accuracy floats, confusion matrices,
  feature importances all flow through the artifact store without
  pickle.

## Cells

| Cell | What it does |
|---|---|
| `load_data` | Loads Titanic CSV (seaborn built-in). |
| `explore` | Survival rate by sex, class, age bucket. |
| `features` | Engineered features (family size, title from name, age bins). |
| `train_model` | Fits a random forest. |
| `evaluate` | Accuracy, precision, recall, confusion matrix. |
| `compare` | Side-by-side metrics for the trained model vs a baseline. |

## Running

From the project root:

```bash
uv run strata-notebook --host 127.0.0.1 --port 8765
```

Then open `examples/titanic_ml` from the Strata home page.

## Try this

- Add a feature in `features` (e.g. `fare_per_person`). `train_model`
  and `evaluate` go stale but the exploration charts stay ready.
- Run `evaluate` alone. Strata re-runs only the chain that's changed.
