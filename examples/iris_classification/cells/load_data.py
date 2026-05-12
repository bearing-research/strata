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
