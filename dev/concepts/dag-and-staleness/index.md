# The DAG and staleness

## DAG and variable analysis

Each cell's source code is analyzed via Python's AST to extract:

- **Defines**: top-level variable assignments (`x = 1`, `df = pd.read_csv(...)`)
- **References**: free variables used but not defined in this cell

The DAG builder connects references to producers:

- The **last cell** that defines a variable is its producer (handles shadowing)
- Edges flow from producer cells to consumer cells
- **Cycle detection** prevents circular dependencies

The DAG is rebuilt automatically on every cell source change.

## Cell execution flow

When you run a cell, this happens:

1. **Compute provenance hash**: `sha256(sorted_input_hashes + source_hash + env_hash)`
2. **Cache check**: Look up the hash in the artifact store → return immediately on hit
3. **Resolve inputs**: Load upstream variable artifacts into a temp directory
4. **Execute**: Spawn a subprocess running the cell harness in the notebook's
   venv, or, for a cell annotated with a [`# @worker`](https://bearingresearch.com/strata/dev/notebook/workers/), POST the
   work to that worker over HTTP instead. Everything either side of this step
   is identical; only where the Python runs changes. In service mode a cell
   on the server's own host runs as a separate OS user
   (`STRATA_NOTEBOOK_HARNESS_USER`) or not at all; see
   [What a cell can read](https://bearingresearch.com/strata/dev/deployment/service-mode/#what-a-cell-can-read).
5. **Harness**: Deserializes inputs → `exec(source, namespace)` → serializes new variables
6. **Store outputs**: Each consumed variable becomes an artifact
7. **Broadcast**: WebSocket sends status, output, and console messages to the UI

## Cascade execution

When a cell's upstream dependencies aren't ready, the **cascade planner** generates an execution plan:

1. BFS backwards from the target cell to find all upstream cells needing execution
2. Returns cells in topological order with reasons (stale, missing, or target)
3. The frontend auto-accepts the cascade and executes cells sequentially

This means you can edit an early cell and run a downstream cell, and Strata will re-execute the pipeline up to it. Within one run each cell executes at most once, even when two branches of the DAG both need it.

## Run All

Run All runs every cell in notebook order. Consecutive plain Python cells that run locally share one harness process (a batch); a cell with a worker, a timeout, a read-write mount, `@loop`, `@fetch`, `@dataset` or `@output_dir`, a sweep variant or a consumer of one, and every non-Python cell runs on its own.

When a cell fails, Run All carries on, but not through the failure: every cell that reads from a failed cell, directly or further down, is skipped and marked stale with the reason that an upstream failed. Cells the failure does not reach run as normal. Rerun All does the same while bypassing the cache.

## Staleness

A cell is out of date when no stored result matches its current provenance. This happens when:

- Its source code changed
- An upstream cell's output changed
- The environment (`uv.lock`, or an `@env` it declares) changed
- A declared outside input moved: a mounted file, an `@table` snapshot, an `@fetch` URL's bytes, an `@dataset` alias

A cell that still has a result and whose upstream moved reads **stale · upstream changed**. A cell you edited after it ran reads **stale · source changed**, and undoing the edit brings it back to **ready**, shown as cached, along with the cells downstream of it. A cell whose environment or declared outside input changed reads **stale · environment changed** or **stale · input changed**, and the cells downstream of it read stale too. When the change goes away, as when `uv.lock` returns to what it was, they read ready again without running. If an input cannot be read at all, or a run's record cannot say what moved, nothing is known to be wrong, so the cell reads **idle**. The `@fetch` and `@dataset` checks run at most once a minute while you edit, and always right before the cell runs; these lookups and the `@table` catalog read happen outside the lock that serializes the staleness walk, so one slow host does not stall the server.

The **causality chain** explains why a cell is stale. Each good run saves the parts its key was built from (source, environment, upstream outputs and outside inputs), and the chain lists the parts that differ now: "the source of X changed", "upstream X changed", "mount `raw` changed". An input that cannot be read now is reported as that, and a local mount still being hashed as "checking mount `raw`". When the record cannot explain the change, for example a run from before parts were saved, the chain says it is incomplete rather than guessing.

## Cell status lifecycle

```
idle ──→ running ──→ ready ──→ stale
              └───→ error
```

- **idle**: no result, or no way to tell whether it still holds: never executed, an input that cannot be read now, or, after a reopen, a leaf cell that keeps no record of its run (`# @nocache`, or a read-write mount)
- **running**: currently executing. If an upstream cell is edited while it runs, it stays running and finishes **stale**, since it read the old value.
- **ready**: last execution succeeded (or was a cache hit), artifact is current
- **stale**: it has a result known not to match: its source was edited since, something upstream has moved, its environment or a declared outside input changed, or Run All skipped it because a cell it reads from failed
- **error**: last execution failed. The traceback and the console output before it are kept against the source that failed, so `strata cell show`, the MCP view, a reopen, a snapshot and an export all still report them. The error stands until the cell is edited or runs successfully (a cache hit counts).
