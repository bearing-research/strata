<!-- strata example run, source 2b09a5882fac59a1 -->
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

**Output**

| survived | pclass | sex | age | sibsp | parch | fare | embarked | class | who | adult_male | deck | embark_town | alive | alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3 | male | 22 | 1 | 0 | 7.25 | S | Third | man | True |  | Southampton | no | False |
| 1 | 1 | female | 38 | 1 | 0 | 71.28 | C | First | woman | False | C | Cherbourg | yes | False |
| 1 | 3 | female | 26 | 0 | 0 | 7.925 | S | Third | woman | False |  | Southampton | yes | True |
| 1 | 1 | female | 35 | 1 | 0 | 53.1 | S | First | woman | False | C | Southampton | yes | False |
| 0 | 3 | male | 35 | 0 | 0 | 8.05 | S | Third | man | True |  | Southampton | no | True |

```text title="stdout"
Loaded 891 passengers
Survival rate: 38.4%

Missing values:
age            177
embarked         2
deck           688
embark_town      2
dtype: int64
```

## Survival rates by class and sex

<sub>**kind** python</sub>

```python
# @name Survival rates by class and sex
survival_rates = df.groupby(["pclass", "sex"])["survived"].mean().round(3)
print(survival_rates)
print(f"\nOverall: {df['survived'].mean():.3f}")
```

```text title="stdout"
pclass  sex   
1       female    0.968
        male      0.369
2       female    0.921
        male      0.157
3       female    0.500
        male      0.135
Name: survived, dtype: float64

Overall: 0.384
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

```text title="stdout"
Features: ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
Train: 535, Test: 179
Dropped 177 rows with missing values
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

```text title="stdout"
Logistic Regression        train=0.807  test=0.810
Random Forest              train=0.991  test=0.771
Gradient Boosting          train=0.925  test=0.793
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

```text title="stdout"
Best model: Logistic Regression (test accuracy: 0.810)

                 precision    recall  f1-score   support

Did not survive       0.84      0.84      0.84       106
       Survived       0.77      0.77      0.77        73

       accuracy                           0.81       179
      macro avg       0.80      0.80      0.80       179
   weighted avg       0.81      0.81      0.81       179
```

## Feature importance from the best model

<sub>**kind** python</sub>

```python
# @name Feature importance from the best model
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("strata")

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
importance.plot.barh(ax=ax)
ax.grid(axis="y", visible=False)
ax.set_title(f"Feature Importance ({best_name})")
ax.set_xlabel(axis_label)
plt.tight_layout()

print(importance.sort_values(ascending=False))
fig
```

![cell output](../assets/examples/runs/titanic_ml-1.webp)

```text title="stdout"
sex       2.419319
pclass    1.232228
sibsp     0.227804
parch     0.102860
age       0.044134
fare      0.001285
dtype: float64
```
