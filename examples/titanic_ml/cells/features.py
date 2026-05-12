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
