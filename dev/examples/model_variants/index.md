# Model Variants: switch classifiers without forking the notebook

The same iris pipeline as `iris_classification`, but with three
alternative training cells grouped as variants of the same DAG slot.
Pick a classifier from the variant tabs and the rest of the notebook
re-runs against it; switch back and the previous result is a cache hit.

## What it shows

- **Variant cells.** `train_logreg`, `train_rf`, and `train_gbm` all
  carry `# @variant classifier <name>` and define the same contract
  (`model`). Only the active variant participates in the DAG.
- **Switching is cheap.** Each variant has its own provenance hash, so
  re-running an already-trained variant is a cache hit. Flip-flopping
  between two variants is free after each has run once.
- **Strict contract.** All variants must produce the same value
  bindings (imports don't count, they're scaffolding). If one variant
  accidentally adds an extra value, you get a
  `variant_contract_mismatch` diagnostic, the contract is what makes
  downstream cells correct under any selection.

## Cells

| Cell | What it does |
|---|---|
| `load_data` | Loads iris into `df` + `feature_names`. |
| `train_test` | 80/20 stratified split. |
| `train_logreg` | **Variant `classifier=logreg`**: `LogisticRegression`. Active by default. |
| `train_rf` | **Variant `classifier=rf`**: `RandomForestClassifier`. |
| `train_gbm` | **Variant `classifier=gbm`**: `GradientBoostingClassifier`. |
| `evaluate` | Test accuracy + per-class precision/recall against whichever `model` is active. |
| `confusion` | Confusion-matrix heatmap for the active classifier. |

## Running

From the project root:

```bash
uv run strata-notebook
```

Open `examples/model_variants` from the Strata home page. The
classifier group renders as a tab strip on the train cell, click a
tab to switch.

## Try this

- Run the notebook on `logreg`, then click the `rf` tab. `evaluate`
  and `confusion` go stale; re-run them to see the random-forest
  numbers. Click `logreg` again, both downstream cells become cache
  hits.
- Add a fourth variant: click the `+` on the tab strip, which clones
  the active variant, then change its annotation to
  `# @variant classifier svc` and its body to `model = SVC(...)`. A
  file dropped into `cells/` on its own is ignored: only cells listed
  in `notebook.toml` load, so a hand-made `cells/train_svc.py` also
  needs a `[[cells]]` entry there.
- Edit a variant cell to define an extra variable (say,
  `feature_importance = ...`). The header pill flags
  `variant_contract_mismatch`: siblings disagree on what they
  expose. Remove the extra binding to clear it.

## Notes

The active selection is committed in `notebook.toml`'s
`[[variant_group]]` table, flipping variants from the UI shows up as a
git diff on that one line. That's intentional: the notebook records
which experiment you ran.

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
df.head()
```

## Train / test split

<sub>**kind** python</sub>

```python
# @name Train / test split
# 80/20 train-test split, stratified on species.
from sklearn.model_selection import train_test_split

X = df[feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")
```

# Pick a classifier

The three cells below are alternatives. They share the same `model`
defines contract, each one trains a different classifier on the same
inputs and binds the result to `model`. Only the active variant
participates in the DAG; the others render as inactive tabs in the
notebook UI.

Switching variants is cheap. Each variant has its own provenance
hash, so re-running an already-trained variant is a cache hit. The
downstream cells (`Evaluate active classifier`, `Confusion matrix`)
re-cascade against whichever variant is active.

## Logistic regression

<sub>**kind** python  ·  **variant** logreg of classifier</sub>

```python
# @variant classifier logreg
# @name Logistic regression
# Logistic regression baseline.
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print(f"Trained {type(model).__name__} (train acc {model.score(X_train, y_train):.3f})")
```

## Evaluate active classifier

<sub>**kind** python</sub>

```python
# @name Evaluate active classifier
# Test-set classification report; works for any classifier.
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
test_acc = model.score(X_test, y_test)

print(f"=== {type(model).__name__} ===")
print(f"Test accuracy: {test_acc:.3f}\n")
print(classification_report(y_test, y_pred))
```

## Confusion matrix

<sub>**kind** python</sub>

```python
# @name Confusion matrix
# Confusion-matrix heatmap for the active classifier.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
ax.set_title(f"Confusion Matrix: {type(model).__name__}")
plt.tight_layout()
fig
```
