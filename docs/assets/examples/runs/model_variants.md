<!-- strata example run, source 43c836e249ab95a0 -->
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

**Output**

| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | species |
| --- | --- | --- | --- | --- |
| 5.1 | 3.5 | 1.4 | 0.2 | setosa |
| 4.9 | 3 | 1.4 | 0.2 | setosa |
| 4.7 | 3.2 | 1.3 | 0.2 | setosa |
| 4.6 | 3.1 | 1.5 | 0.2 | setosa |
| 5 | 3.6 | 1.4 | 0.2 | setosa |

```text title="stdout"
Loaded 150 samples, 4 features
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

```text title="stdout"
Train: 120 samples
Test:  30 samples
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

```text title="stdout"
Trained LogisticRegression (train acc 0.975)
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

```text title="stdout"
=== LogisticRegression ===
Test accuracy: 0.967

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
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

plt.style.use("strata")

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

![cell output](../assets/examples/runs/model_variants-1.webp)
