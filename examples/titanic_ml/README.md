# Titanic: feature engineering + model comparison

A canonical "first Kaggle notebook" rewritten for Strata. A
feature-engineering stage feeds three classifiers; the best one is
evaluated and its feature weights plotted.

## What it shows

- **Two branches off one load**: `explore` and `features` both read
  `load_data`'s DataFrame, so editing `features` leaves `explore` ready.
- **One upstream feeding several cells**: `features` produces the split,
  which `train_model` and `evaluate` read, and `feature_cols`, which
  `compare` reads, so editing it invalidates all three.
- **Objects that need pickle**: `results` holds the fitted scikit-learn
  models next to their accuracies, so it is stored with pickle rather than
  as Arrow or JSON.

## Cells

| Cell | What it does |
|---|---|
| `load_data` | Loads the Titanic dataset bundled with seaborn and prints missing-value counts. |
| `explore` | Prints the survival rate grouped by class and sex, and the overall rate. |
| `features` | Keeps six columns, drops rows with missing values, encodes `sex` as 0/1 and makes a stratified 75/25 train/test split. |
| `modeling_intro` | Markdown introducing the modeling cells. |
| `train_model` | Fits logistic regression, random forest and gradient boosting, and keeps each model with its train and test accuracy in `results`. |
| `evaluate` | Picks the model with the best test accuracy and prints its classification report. |
| `compare` | Plots the best model's feature importances (or absolute coefficients for logistic regression). |

## Running

From the project root:

```bash
uv run strata-notebook
```

Then open `examples/titanic_ml` from the Strata home page.

## Try this

- Add a feature in `features` (e.g. `fare_per_person`). `train_model`,
  `evaluate` and `compare` go stale but `explore` stays ready.
- Run `evaluate` alone. Strata re-runs only the chain that's changed.
