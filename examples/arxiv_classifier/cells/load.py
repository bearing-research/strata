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
