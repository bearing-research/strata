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

## Load Titanic dataset

<sub>**kind** python</sub>

```python
# @name Load Titanic dataset
# Bundled with seaborn, so no network fetch is needed.
import seaborn as sns

df = sns.load_dataset("titanic")

print(f"Loaded {len(df)} passengers")
print(f"Survival rate: {df['survived'].mean():.1%}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
df.head()
```

## Survival rates by class and sex

<sub>**kind** python</sub>

```python
# @name Survival rates by class and sex
survival_rates = df.groupby(["pclass", "sex"])["survived"].mean().round(3)
print(survival_rates)
print(f"\nOverall: {df['survived'].mean():.3f}")
```

## Feature engineering

<sub>**kind** python</sub>

```python
# @name Feature engineering
from sklearn.model_selection import train_test_split

feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
clean = df[feature_cols + ["survived"]].dropna().copy()

# Encode sex as numeric
clean["sex"] = clean["sex"].map({"male": 0, "female": 1})

X = clean[feature_cols]
y = clean["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Features: {feature_cols}")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Dropped {len(df) - len(clean)} rows with missing values")
```

# Model the survivors

We've engineered features from the raw passenger data. Now train
three classifiers (logistic regression, random forest, gradient
boosting), evaluate the best one, and compare feature importance.

Each classifier runs once and gets cached. Tweaking
`features.py` invalidates all three downstream cells; the
provenance hash on each training cell changes because its `X_train`
input changed.

## Train classifiers

<sub>**kind** python</sub>

```python
# @name Train classifiers
# Train logistic regression, random forest, and gradient boosting; keep all three.
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    results[name] = {"train": train_acc, "test": test_acc, "model": model}
    print(f"{name:25s}  train={train_acc:.3f}  test={test_acc:.3f}")
```

## Evaluate best model

<sub>**kind** python</sub>

```python
# @name Evaluate best model
# Detailed evaluation of the best model.
from sklearn.metrics import classification_report

best_name = max(results, key=lambda k: results[k]["test"])
best_model = results[best_name]["model"]
y_pred = best_model.predict(X_test)

print(f"Best model: {best_name} (test accuracy: {results[best_name]['test']:.3f})\n")
print(classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"]))
```

## Feature importance from the best model

<sub>**kind** python</sub>

```python
# @name Feature importance from the best model
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Which weights a model exposes depends on what it is: the tree ensembles have
# feature_importances_, logistic regression has coefficients. Reading whichever
# is there keeps the cell producing a figure whoever wins, rather than printing
# an apology when the linear model does.
if hasattr(best_model, "feature_importances_"):
    weights = pd.Series(best_model.feature_importances_, index=feature_cols)
    axis_label = "Importance"
else:
    weights = pd.Series(best_model.coef_[0], index=feature_cols).abs()
    axis_label = "|Coefficient|"

importance = weights.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
importance.plot.barh(ax=ax, color="#89b4fa")
ax.set_title(f"Feature Importance ({best_name})")
ax.set_xlabel(axis_label)
plt.tight_layout()

print(importance.sort_values(ascending=False))
fig
```
