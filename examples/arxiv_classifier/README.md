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

`notebook.toml` points `df-cluster` and `gpu-fly` at hosted workers. To use
two local HTTP executors instead, change the `url` under each
`[workers.config]`: `df-cluster` to `http://127.0.0.1:9000/v1/execute` and
`gpu-fly` to `http://127.0.0.1:9001/v1/execute`, the ports
`run_local_workers.sh` binds.

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
