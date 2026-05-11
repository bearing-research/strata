# Notebook Concepts

## Architecture

Strata Notebook is a content-addressed compute graph over Python. Every cell output is an artifact, and every cell execution is a `materialize(inputs, transform) → artifact` operation.

```
┌─────────────────────────────────────────────┐
│ Notebook UI (Vue.js + WebSocket)            │
│ (cell editing, run buttons, DAG view)       │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ Notebook Backend (FastAPI + WebSocket)       │
│ (session mgmt, cascade planner, executor)   │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ Strata Artifact Store                       │
│ (SQLite metadata + blob storage, provenance │
│  dedup, lineage)                            │
└─────────────────────────────────────────────┘
```

The notebook is an **orchestration layer** — it decides what to run next. The cell harness is an **executor** — it runs Python code. The artifact store decides whether a result already exists and persists it.

## Why "every output is an artifact"

The guarantees fall out of two design choices:

1. **Provenance is identity.** An artifact's ID is a hash of the cell's source
   (AST-normalized), its resolved inputs, and the environment fingerprint.
   Identical computations produce identical IDs, so the cache hit check is
   "have we computed this exact hash before?" — not "do two things look
   similar?"
2. **Artifacts are immutable.** Once stored, a version never mutates. A
   downstream cell reading artifact `@v=3` today reads the same bytes it
   would have read yesterday, even if upstream code has since changed and
   a new `@v=4` exists.

Together: the cache isn't an optimization layer that *might* be stale — it's
the single source of truth for "has this work been done." That's why cache
hits are safe to serve in milliseconds, why renaming a cell doesn't
invalidate a downstream artifact (the provenance hash depends on semantic
source, not whitespace or comments), and why you can fork a loop from
iteration 17 without having to re-run iterations 0–16.

## Notebook File Format

Each notebook is a directory on disk:

```
my_notebook/
├── notebook.toml          # Stable config: ID, name, cell list, workers, mounts, env, ai
├── pyproject.toml         # Python dependencies (uv-managed)
├── uv.lock                # Locked dependencies
├── cells/
│   ├── a1b2c3d4.py        # Cell source files
│   └── e5f6g7h8.py
└── .strata/               # Gitignored — runtime state, not committed
    ├── runtime.json           # Display outputs, provenance hashes, env metadata
    ├── console/               # Per-cell stdout/stderr ({cell_id}.json)
    └── artifacts/
        ├── artifacts.sqlite   # Artifact metadata
        └── blobs/             # Serialized cell outputs
```

`notebook.toml` holds stable config you'd commit to git; `.strata/` holds runtime state that changes on every execution (display outputs, console snapshots, the last `uv sync` timestamp, per-cell provenance hashes). `notebook.toml`'s `updated_at` tracks only structural edits — adding/removing cells, changing workers or mounts — so example notebooks don't churn under version control.

`notebook.toml` defines the notebook identity and cell ordering:

```toml
notebook_id = "f7bd9094-..."
name = "my_analysis"

[[cells]]
id = "a1b2c3d4"
file = "a1b2c3d4.py"
language = "python"
order = 0

[[cells]]
id = "e5f6g7h8"
file = "e5f6g7h8.py"
language = "python"
order = 1
```

## Version Control

Strata notebooks are designed to live in git. Unlike Jupyter `.ipynb`
files — which co-mingle source, outputs, and execution counts in one
JSON blob and produce a multi-kilobyte diff every time a cell is
re-run — a Strata notebook is just a directory of plain text:

- **Cells are `.py` files.** Normal `git diff`, `git blame`, code review
  on a pull request, syntax highlighting in every IDE. Reordering a cell
  edits one number in `notebook.toml`, not a giant JSON re-serialize.
- **`notebook.toml` is the manifest.** Stable config only — cell list,
  workers, mounts, env, AI defaults, and the active variant per group
  (see [Variant Cells](annotations.md#variant-cells)). Reviewers see
  exactly what changed about the notebook's *shape*, not its execution
  history.
- **`.strata/` is gitignored.** Display outputs, console snapshots, the
  `uv sync` timestamp, per-cell provenance hashes, the artifact store —
  none of it touches commits. Re-running a cell never changes the tracked
  tree.
- **`updated_at` only bumps on structural edits.** Adding/removing a
  cell, changing a worker, mounting a path, picking a different variant —
  those bump the timestamp. Editing source or running cells does not.
- **Secrets stay off disk.** Env keys matching `KEY`/`SECRET`/`TOKEN`/
  `PASSWORD`/`CREDENTIAL` are blanked before persisting, so the writer
  can't accidentally commit an API key. The name survives (so the
  Runtime panel still knows the slot exists), the value doesn't.
- **uv lockfile in committed config.** `pyproject.toml` + `uv.lock` pin
  the Python environment exactly the same way the rest of your repo
  does — collaborators get a reproducible environment from a fresh clone.

Put together: a Strata notebook commit shows *what changed about the
work*, not *what happened during the last run*.

## DAG and Variable Analysis

Each cell's source code is analyzed via Python's AST to extract:

- **Defines** — top-level variable assignments (`x = 1`, `df = pd.read_csv(...)`)
- **References** — free variables used but not defined in this cell

The DAG builder connects references to producers:

- The **last cell** that defines a variable is its producer (handles shadowing)
- Edges flow from producer cells to consumer cells
- **Cycle detection** prevents circular dependencies

The DAG is rebuilt automatically on every cell source change.

## Cell Execution Flow

When you run a cell, this happens:

1. **Compute provenance hash**: `sha256(sorted_input_hashes + source_hash + env_hash)`
2. **Cache check**: Look up the hash in the artifact store → return immediately on hit
3. **Resolve inputs**: Load upstream variable artifacts into a temp directory
4. **Execute**: Spawn a subprocess running the cell harness in the notebook's venv
5. **Harness**: Deserializes inputs → `exec(source, namespace)` → serializes new variables
6. **Store outputs**: Each consumed variable becomes an artifact
7. **Broadcast**: WebSocket sends status, output, and console messages to the UI

## Caching and Provenance

The provenance hash determines cache identity. It includes:

| Component | In hash? | Why |
|-----------|----------|-----|
| Source code | Yes | Different code = different result |
| Upstream artifact hashes | Yes | Different inputs = different result |
| Environment lockfile hash | Yes | Different packages = different result |
| Cell ID | No | Same code in a different cell = same result |
| Execution time | No | Same inputs should produce same output |

When you change a cell's source, its provenance hash changes, and all downstream cells become **stale**.

## Serialization

Cell outputs are serialized based on their Python type:

| Type | Format | File extension |
|------|--------|---------------|
| PyArrow tables, pandas DataFrames, numpy arrays | Arrow IPC | `.arrow` |
| Dicts, lists, scalars (int, float, str, bool, None) | JSON | `.json` |
| Everything else | Pickle | `.pickle` |

The content type is stored in the artifact metadata so the read side knows how to deserialize.

## Cascade Execution

When a cell's upstream dependencies aren't ready, the **cascade planner** generates an execution plan:

1. BFS backwards from the target cell to find all upstream cells needing execution
2. Returns cells in topological order with reasons (stale, missing, or target)
3. The frontend auto-accepts the cascade and executes cells sequentially

This means you can edit an early cell and run a downstream cell — Strata will automatically re-execute the full pipeline.

## Staleness

A cell is **stale** when its cached artifact no longer matches its current provenance. This happens when:

- Its source code changed
- An upstream cell's output changed
- The environment (uv.lock) changed

The **causality chain** explains why a cell is stale, tracing the change back to its root cause (e.g., "upstream cell X changed its source").

## Cell Status Lifecycle

```
idle → running → ready
                ↗
idle → running → error
```

- **idle** — never executed, or stale (needs re-execution)
- **running** — currently executing
- **ready** — last execution succeeded, artifact is current
- **error** — last execution failed
