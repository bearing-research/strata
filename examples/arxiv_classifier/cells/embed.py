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
