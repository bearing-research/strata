# arXiv Paper Classifier (distributed demo)

A full ML workflow that exercises every differentiated feature of Strata
Notebook at once:

- **Distributed workers.** Nine cells, three different workers: `local`,
  `df-cluster` (CPU-heavy DataFusion), `gpu-fly` (GPU for embeddings and
  training). Each cell declares its target worker with a single annotation;
  no deployment code.
- **Content-addressed caching.** Re-run the notebook after an edit; every
  unchanged cell hits cache instantly regardless of which worker it runs on.
- **Prompt cells.** Two cells use the LLM assistant with `{{ variable }}`
  injection to narrate the data at key points.
- **DAG invalidation.** Change the model in the training cell, and every
  upstream cell (load, aggregate, embed) still hits cache — only training
  and evaluation re-execute.

## The workflow

| # | Cell | Worker | What it does |
|---|---|---|---|
| 1 | `load` | local | Load arXiv metadata |
| 2 | `aggregate` | df-cluster | Group by category and year via DataFusion |
| 3 | `themes` | local (LLM) | Prompt: identify research themes from stats |
| 4 | `sample` | df-cluster | Stratified sample for the training set |
| 5 | `embed` | gpu-fly | Generate sentence-transformer embeddings |
| 6 | `clusters` | local (LLM) | Prompt: describe each paper cluster |
| 7 | `train` | gpu-fly | Train a classifier on the embeddings |
| 8 | `evaluate` | local | Compute accuracy and classification report |
| 9 | `plot` | local | Visualize the confusion matrix |

Cells 1, 3, 6, 8, 9 run on the user's machine. Cells 2 and 4 run on the
DataFusion worker — close to the data, avoids round-tripping the full
dataset. Cells 5 and 7 run on the GPU worker.

## Running it locally

Two local HTTP executors stand in for the cloud workers during development.

```bash
# Terminal 1 — start the two local worker processes
./examples/arxiv_classifier/run_local_workers.sh

# Terminal 2 — start Strata
uv run strata-notebook

# Terminal 3 — open the notebook in the UI
# http://localhost:8765
# Click "Open Notebook" and select examples/arxiv_classifier
```

Once the notebook is open, run all cells. You should see:

- The `worker: df-cluster` badge next to cells 2 and 4
- The `worker: gpu-fly` badge next to cells 5 and 7
- A live `dispatching → df-cluster` or `dispatching → gpu-fly` badge
  (pulsing yellow) while each remote cell is executing
- Green `✓` status on every cell after the first run
- Green `✓ cached` on every cell on the second run

## Day 1 vs. real workload

As of Day 1 (April 2026), the cells run placeholder workloads — tiny
DataFrames, fake embeddings, a trivial classifier — so we can validate the
distributed plumbing without downloading a 500 MB dataset or setting up
GPU deps. The workload is swapped in over the following days as the cloud
workers come online. The worker annotations and cell structure stay the
same.
