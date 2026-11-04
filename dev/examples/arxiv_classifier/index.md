# arXiv Paper Classifier (distributed demo)

A full ML workflow that exercises every differentiated feature of Strata
Notebook at once:

- **Distributed workers.** Nine cells, three different workers: `local`,
  `df-cluster` (CPU-heavy DataFusion), `gpu-fly` (GPU for embeddings).
  Each cell declares its target worker with a single annotation;
  no deployment code.
- **Content-addressed caching.** Re-run the notebook after an edit; every
  unchanged cell hits cache instantly regardless of which worker it runs on.
- **Prompt cells.** Two cells use the LLM assistant with `{{ variable }}`
  injection to narrate the data at key points.
- **DAG invalidation.** Change the model in the training cell, and every
  upstream cell (load, aggregate, embed) still hits cache, only training,
  evaluation and the plot re-execute.

## The workflow

| # | Cell | Worker | What it does |
|---|---|---|---|
| 1 | `load` | local | Load arXiv metadata |
| 2 | `aggregate` | df-cluster | Group by category and year via DataFusion |
| 3 | `themes` | local (LLM) | Prompt: identify research themes from stats |
| 4 | `sample` | df-cluster | Stratified sample for the training set |
| 5 | `embed` | gpu-fly | Generate sentence-transformer embeddings |
| 6 | `clusters` | local (LLM) | Prompt: describe each paper cluster |
| 7 | `train` | local | Train a classifier on the embeddings |
| 8 | `evaluate` | local | Compute accuracy and classification report |
| 9 | `plot` | local | Visualize the confusion matrix |

Cells 1, 3, 6, 7, 8, 9 run on the user's machine. Cells 2 and 4 run on the
DataFusion worker, close to the data, avoids round-tripping the full
dataset. Cell 5 runs on the GPU worker; training a logistic regression on
the embeddings takes under a second on a CPU, so it stays local.

## Running it locally

`notebook.toml` points `df-cluster` and `gpu-fly` at two local HTTP
executors, `http://127.0.0.1:9000/v1/execute` and
`http://127.0.0.1:9001/v1/execute`, the ports `run_local_workers.sh` binds.
To run them on hosted machines instead, deploy `df-cluster/` to Fly and
`modal_gpu_worker.py` to Modal (`MODAL_DEPLOY.md`), then change the `url`
under each `[workers.config]` to the deployed one. `df-cluster` sends the
token in `STRATA_DF_CLUSTER_TOKEN` when that is set.

```bash
# Terminal 1: start the two local worker processes
./examples/arxiv_classifier/run_local_workers.sh

# Terminal 2: start Strata
uv run strata-notebook

# Terminal 3: open the notebook in the UI
# http://localhost:8765
# Click "Open Notebook" and select examples/arxiv_classifier
```

Once the notebook is open, run all cells. You should see:

- The `worker: df-cluster` badge next to cells 2 and 4
- The `worker: gpu-fly` badge next to cell 5
- A live `dispatching → df-cluster` or `dispatching → gpu-fly` badge
  (pulsing yellow) while each remote cell is executing
- Green `✓` status on every cell after the first run
- Green `✓ cached` on every cell on the second run

## What it needs

The cells run the real workload: `load` downloads about 20K arXiv abstracts
from Hugging Face, and `embed` encodes them with the `all-MiniLM-L6-v2`
sentence-transformer, so the worker running `embed` needs
`sentence-transformers` (and with it `torch`). The two prompt cells, `themes`
and `clusters`, need an LLM provider key such as `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`, set in the Runtime panel.

## Load arXiv Papers

<sub>**kind** python  ·  **worker** local</sub>

```python
# @name Load arXiv Papers
# @worker local
# Load arXiv ML papers from Hugging Face and assign topic categories.
# 118K real papers with titles and abstracts. We sample 20K and assign
# topics via keyword matching, a common first step when you have text
# data but no labels.
import re

import pandas as pd

DATASET_URL = (
    "https://huggingface.co/api/datasets/CShorten/ML-ArXiv-Papers"
    "/parquet/default/train/0.parquet"
)
SAMPLE_SIZE = 20_000

raw = pd.read_parquet(DATASET_URL, columns=["title", "abstract"])
papers = raw.dropna(subset=["abstract"]).head(SAMPLE_SIZE).reset_index(drop=True)

TOPIC_RULES = [
    ("reinforcement-learning", r"reinforcement|reward|policy gradient|Q-learning|MDP"),
    ("nlp", r"\bNLP\b|language model|translation|transformer|text classif|sentiment"),
    ("computer-vision", r"image|object detection|segmentation|convolutional|visual"),
    ("optimization", r"convex|gradient descent|convergence|optimization|stochastic"),
    ("generative", r"generative|GAN|diffusion|variational|autoencoder|VAE"),
]


def _assign_topic(text: str) -> str:
    lower = text.lower()
    for topic, pattern in TOPIC_RULES:
        if re.search(pattern, lower, re.IGNORECASE):
            return topic
    return "other"


papers["topic"] = papers["abstract"].apply(_assign_topic)
print(f"Loaded {len(papers):,} papers, {papers['topic'].nunique()} topics")
print(papers["topic"].value_counts().to_string())
papers
```

## Aggregate by Topic

<sub>**kind** python  ·  **worker** df-cluster</sub>

```python
# @name Aggregate by Topic
# @worker df-cluster
# Aggregate paper counts per topic using DataFusion SQL.
# Dispatched to the df-cluster worker which has DataFusion installed.
import pyarrow as pa
from datafusion import SessionContext

ctx = SessionContext()
table = pa.Table.from_pandas(papers)
ctx.register_record_batches("papers", [table.to_batches()])

category_stats = ctx.sql(
    """
    SELECT
        topic,
        COUNT(*) AS paper_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM papers
    GROUP BY topic
    ORDER BY paper_count DESC
    """
).to_pandas()

print("Topic distribution (DataFusion SQL):")
print(category_stats.to_string(index=False))
category_stats
```

## research_themes

<sub>**kind** prompt  ·  **worker** local</sub>

*Prompt cell: response intentionally excluded from export.*

```text
# @name research_themes
# @worker local
# Prompt cell: LLM analysis of topic distribution from the DataFusion aggregation.
Given these arXiv ML paper topic counts derived from keyword classification:

{{ category_stats }}

For each topic, write one sentence describing what kinds of papers fall into it
and why this topic matters for the ML research community. Return as a numbered list.
```

## Stratified Sample

<sub>**kind** python  ·  **worker** df-cluster</sub>

```python
# @name Stratified Sample
# @worker df-cluster
# Stratified sample: take up to 500 papers per topic so the embedding
# step runs in seconds, not minutes. Uses DataFusion window functions.
import pyarrow as pa
from datafusion import SessionContext

ctx = SessionContext()
table = pa.Table.from_pandas(papers)
ctx.register_record_batches("papers", [table.to_batches()])

PER_TOPIC = 500

sampled_papers = ctx.sql(
    f"""
    WITH ranked AS (
        SELECT
            title,
            abstract,
            topic,
            ROW_NUMBER() OVER (PARTITION BY topic ORDER BY title) AS rn
        FROM papers
    )
    SELECT title, abstract, topic
    FROM ranked
    WHERE rn <= {PER_TOPIC}
    ORDER BY topic, title
    """
).to_pandas()

print(
    f"Sampled {len(sampled_papers):,} papers across "
    f"{sampled_papers['topic'].nunique()} topics (DataFusion SQL)"
)
print(sampled_papers["topic"].value_counts().to_string())
sampled_papers
```

## Embed Abstracts

<sub>**kind** python  ·  **worker** gpu-fly</sub>

```python
# @name Embed Abstracts
# @worker gpu-fly
# @timeout 300
# Generate sentence-transformer embeddings for each paper's abstract.
# This is the expensive step: ~3K abstracts × 384-dim on an A10G GPU
# takes ~5 seconds. On CPU it would take ~90 seconds. On re-run it's
# instant: the artifact store caches the result keyed by the exact
# input data + model identity.
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = sampled_papers["abstract"].tolist()
embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
embeddings = np.array(embeddings, dtype="float32")

print(f"Generated embeddings: {embeddings.shape} ({embeddings.nbytes / 1e6:.1f} MB)")
embeddings
```

## cluster_descriptions

<sub>**kind** prompt  ·  **worker** local</sub>

*Prompt cell: response intentionally excluded from export.*

```text
# @name cluster_descriptions
# @worker local
# Prompt cell: generate a practical summary of the research themes.
The following research themes were identified from analyzing {{ category_stats }}:

{{ research_themes }}

Write a short paragraph (4-5 sentences) for a technical audience explaining
the practical significance of these themes. Focus on how they relate to each
other and what trends they suggest for the ML research community.
```

## Train Classifier

<sub>**kind** python  ·  **worker** local</sub>

```python
# @name Train Classifier
# @worker local
# Train a logistic regression classifier: embeddings → topic label.
# Runs locally. Logistic regression on 3K × 384 takes <1s on CPU, so
# there is no reason to ship the data to a GPU worker and back.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = embeddings
y = sampled_papers["topic"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
classifier = LogisticRegression(max_iter=1000, n_jobs=-1)
classifier.fit(X_train, y_train)

train_acc = classifier.score(X_train, y_train)
test_acc = classifier.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")
print(f"Classes: {list(classifier.classes_)}")

train_test_split_info = {
    "train_size": len(X_train),
    "test_size": len(X_test),
    "train_acc": round(train_acc, 4),
    "test_acc": round(test_acc, 4),
}
train_test_split_info
```

## Evaluate Model

<sub>**kind** python  ·  **worker** local</sub>

```python
# @name Evaluate Model
# @worker local
# Evaluation on the held-out test set. Small data, runs locally.
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

X = embeddings
y = sampled_papers["topic"].to_numpy()
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

print(f"Test accuracy: {accuracy:.3f}")
print()
print(classification_report(y_test, predictions, zero_division=0))

eval_results = {"accuracy": round(accuracy, 4), "per_class": report}
eval_results
```

## Confusion Matrix

<sub>**kind** python  ·  **worker** local</sub>

```python
# @name Confusion Matrix
# @worker local
# Visualize the confusion matrix on the held-out test set.
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

X = embeddings
y = sampled_papers["topic"].to_numpy()
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

classes = sorted(set(y))
predictions = classifier.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=classes)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=classes).plot(
    ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30
)
ax.set_title("arXiv Topic Classification: Confusion Matrix")
plt.tight_layout()
fig
```
