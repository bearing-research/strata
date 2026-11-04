# Iris Classification: end-to-end ML in seven cells

Classic scikit-learn tutorial, rewritten as a Strata notebook. Shows
what the graph looks like for a realistic ML pipeline: load → explore
→ split → train → evaluate → visualize.

## What it shows

- **DAG branching.** `scatter_plot` and `train_test` both read from
  `load_data`, so editing `load_data` invalidates both branches.
- **Mixed output types.** Cells produce DataFrames (arrow/ipc), trained
  models (pickle), and matplotlib figures (image/png), all stored
  natively by Strata's serializer.
- **Display outputs.** `_` at the end of a cell (or a trailing
  expression) becomes an inline preview on the cell.

## Cells

| Cell | What it does |
|---|---|
| `load_data` | Loads the iris dataset into a DataFrame. |
| `explore_stats` | Per-feature summary stats. |
| `scatter_plot` | Pair-plot of the four features, colored by class. |
| `train_test` | 70/30 train/test split. |
| `train_model` | Fits a `RandomForestClassifier` (100 trees). |
| `evaluate` | Accuracy + per-class precision/recall. |
| `confusion` | Confusion-matrix heatmap. |

## Running

From the project root:

```bash
uv run strata-notebook
```

Then open `examples/iris_classification` from the Strata home page.

## Try this

- Change the `test_size` in `train_test`. Cells downstream go stale;
  `scatter_plot` stays ready (it doesn't depend on the split).
- Re-run `evaluate` without re-training, the trained model is cached,
  so only evaluation runs.

## Load iris dataset

<sub>**kind** python</sub>

```python
# @name Load iris dataset
# Load the Iris dataset into a pandas DataFrame.
import pandas as pd
from sklearn.datasets import load_iris

iris_bunch = load_iris()
df = pd.DataFrame(iris_bunch.data, columns=iris_bunch.feature_names)
df["species"] = pd.Categorical.from_codes(iris_bunch.target, iris_bunch.target_names)
feature_names = iris_bunch.feature_names

print(f"Loaded {len(df)} samples, {len(feature_names)} features")
print(f"Species: {df['species'].unique().tolist()}")
df.head()
```

## Explore per-feature stats

<sub>**kind** python</sub>

```python
# @name Explore per-feature stats
# Summary statistics grouped by species.
stats = df.groupby("species").agg(["mean", "std"]).round(2)
print(stats)
```

## Pairwise scatter plot

<sub>**kind** python</sub>

```python
# @name Pairwise scatter plot
# Pairwise scatter plot colored by species.
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

g = sns.pairplot(df, hue="species", diag_kind="hist", height=2)
g.figure.suptitle("Iris Feature Distributions", y=1.02)
plt.tight_layout()
g.figure
```

# Train and evaluate

The exploration cells above don't feed the model directly, they're
sanity checks. The cells below are the ML pipeline proper: split the
data, fit a random forest, score it against the held-out test set,
and visualize the confusion matrix.

Edit any cell in this section and the downstream cells go stale.
Re-running the *test set* cell after an upstream change cascades the
whole subtree in topological order, hitting cache for everything that
hasn't changed.

## Train / test split

<sub>**kind** python</sub>

```python
# @name Train / test split
# Split into training and test sets.
from sklearn.model_selection import train_test_split

X = df[feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")
```

## Train random forest classifier

<sub>**kind** python</sub>

```python
# @name Train random forest classifier
# Train a Random Forest classifier.
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train accuracy: {train_score:.3f}")
print(f"Test accuracy:  {test_score:.3f}")
```

## Evaluate

<sub>**kind** python</sub>

```python
# @name Evaluate
# Per-class precision, recall, and f1 against the test set.
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
```

## Confusion matrix

<sub>**kind** python</sub>

```python
# @name Confusion matrix
# Confusion matrix heatmap.
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
fig
```
