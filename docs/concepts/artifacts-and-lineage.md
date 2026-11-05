# Artifacts and lineage

## Architecture

Strata Notebook is a content-addressed compute graph over Python. Every cell output is an artifact, and every cell execution is a `materialize(inputs, transform, environment) → artifact` operation.

```
┌───────────────┬───────────────┬─────────────────────────────┐
│ Web UI (Vue)  │ Terminal TUI  │ Coding agent (CLI / MCP)    │
│ edit and run  │ read-only     │ drives a session remotely   │
└───────────────┴───────────────┴─────────────────────────────┘
        │               │                      │
        └───────────────┴──────────────────────┘
                        │  REST + WebSocket
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ Notebook Backend (FastAPI + WebSocket)                       │
│ (session mgmt, DAG, cascade planner, executor)               │
└──────────────────────────────────────────────────────────────┘
            │                                │
            │ local subprocess               │ HTTP
            ▼                                ▼
   ┌──────────────────┐         ┌──────────────────────────┐
   │ Cell harness     │         │ Remote worker            │
   │ (notebook venv)  │         │ (SSH / Fly / Modal /     │
   │                  │         │  a pool-run machine)     │
   └──────────────────┘         └──────────────────────────┘
            └───────────────┬────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Strata Artifact Store                                        │
│ metadata: SQLite, or Postgres when several servers share one │
│ blobs: local disk, S3, GCS, Azure                            │
│ provenance dedup, lineage, and an optional team cache that   │
│ answers a local miss from a shared store                     │
└──────────────────────────────────────────────────────────────┘
```

The notebook is an **orchestration layer**: it decides what to run next. The cell harness is an **executor**: it runs Python code. The artifact store decides whether a result already exists and persists it.

The three surfaces are **peers on one protocol**, not a primary and two
add-ons: they speak the same REST and WebSocket API. A cell an agent runs
against a live session appears in the web UI and the [terminal viewer](../notebook/tui.md)
as it happens - though only when the agent addresses the session rather than
the notebook directory, since [the local backend edits files and the server
learns nothing](../notebook/cli.md#working-against-a-live-session-server-session). Of the
three, only the TUI is deliberately read-only.

## Why "every output is an artifact"

The guarantees fall out of two design choices:

1. **Provenance is identity.** An artifact is keyed by a hash of the cell's
   source (AST-normalized), its resolved inputs, and the environment
   fingerprint. Identical computations produce identical hashes, so the cache hit check is
   "have we computed this exact hash before?", not "do two things look
   similar?"
2. **Artifacts are immutable.** Once stored, a version never mutates. A
   downstream cell reading artifact `@v=3` today reads the same bytes it
   would have read yesterday, even if upstream code has since changed and
   a new `@v=4` exists.

Together: the cache isn't an optimization layer that *might* be stale, it's
the single source of truth for "has this work been done." That's why cache
hits are safe to serve in milliseconds, why renaming a cell doesn't
invalidate a downstream artifact (the provenance hash depends on semantic
source, not whitespace or comments), and why you can fork a loop from
iteration 17 without having to re-run iterations 0–16.

## Stateful objects and value semantics (ML training)

The flip side of "every variable is an independent, immutable artifact" is that
Strata is a **value-semantics** engine: a cell receives *copies* of its inputs
(deserialized from artifacts), not the caller's live objects. A linear Jupyter
kernel shares one memory space, so a model trained in one cell is visible
everywhere; Strata's cells are isolated, so two patterns that work in Jupyter
break here - both silently, because the run still goes green:

1. **In-place mutation that isn't exported.** A cell that trains a model with
   `optimizer.step()` (a method call, no `model = …` reassignment) mutates
   `model` *in place*. Strata's data-flow only treats a variable as a cell's
   output if it's assigned, subscripted, attr-set or passed `inplace=True`, so
   the trained `model` is never re-exported, and downstream cells read the
   **pre-training** version.

2. **Shared mutable references split across cells.** `optimizer = Adam(model.parameters())`
   makes the optimizer hold references to the *same tensors* as the model.
   Stored as two separate artifacts and reloaded into a later cell, they become
   two *different* tensor sets - so `optimizer.step()` updates the optimizer's
   private copies and the `model` you evaluate/save never changes. Training is a
   silent no-op.

Both are the same root: **shared, mutable state doesn't survive a boundary where
each variable is an independent immutable artifact.** Strata deliberately does
*not* try to fake it (preserving cross-artifact object identity, or silently
re-exporting a mutated input, would make provenance lie).

### The pattern that works

Express training as one content-addressed transform - **data in, trained model
out** - by keeping the model, its optimizer, and the training loop in **one
cell**, and making the trained model that cell's output:

```python
# @name train
model = build_model()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(epochs):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
# `model` (trained) is this cell's output - downstream cells get the trained one.
trained_model = model
```

For a large handoff, export a **checkpoint** instead of a live object: save a
`state_dict` (or a file path) from the training cell and load it downstream with
explicit `map_location` - see the device/placement guidance in
[Workers](../notebook/workers.md).

## Serialization

Cell outputs are serialized based on their Python type:

| Type | Format | File extension |
|------|--------|---------------|
| PyArrow tables, pandas and polars frames and series, numpy arrays and scalars, torch and jax tensors, `datetime` / `Decimal` / `UUID` / `bytes` / `complex` values | Arrow IPC | `.arrow` |
| Any other type exporting `__arrow_c_stream__` or `__dlpack__` | Arrow IPC | `.arrow` |
| Dicts, lists, scalars (int, float, str, bool, None) | JSON | `.json` |
| Everything else | Pickle | `.pickle` |

The content type is stored in the artifact metadata so the read side knows how to deserialize.

### JAX and 64-bit values

JAX defaults to 32 bits: it turns a `float64` into a `float32` when it builds
an array, unless `jax_enable_x64` is on. Every other library here keeps the
dtype it was given, so this is the one case where a value can come back
narrower than it was stored.

Strata will not hand you the narrowed value. Reading a stored 64-bit array
turns the setting on and converts again, and if that still does not reproduce
the dtype it raises rather than returning something plausible and wrong.

That switch is process-wide, and jax offers no per-array alternative. The warm
pool and Run All reuse one process across cells, so once it is on, later cells
in that worker get 64-bit defaults too, and no cell's provenance records which
width it got, so a cached result can depend on what ran before it. When this
happens the cell's console says so.

If a notebook works in 64-bit, say so explicitly:

```python
# @env JAX_ENABLE_X64=1
import jax.numpy as jnp
```

jax reads that at import, every execution path applies the environment before
deserializing anything, and it becomes part of the cell's env hash. The
width is then the same on every path and is recorded in what the result is keyed on.

The second row is what keeps a library Strata has never heard of out of the
pickle path. DuckDB relations, cuDF frames, Ibis tables and anything else
implementing the [Arrow PyCapsule
interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
are stored as real Arrow, so the artifact stays readable by other tools and the
notebook can show its schema and row count.

Two things to know about that row. The first is type identity: the libraries
named in the first row are handed back as themselves, while a capsule-sourced
value comes back as a `pyarrow.Table` and a DLPack-sourced one as a
`numpy.ndarray`. The second matters for libraries whose values are lazy query
expressions rather than data, Ibis among them. Storing one executes it, because
an artifact holds data and a handle is not data. A downstream cell therefore
receives the result rather than the expression, and cannot keep composing on it.
Pin the query where you want it evaluated and store that.

One-shot streams are refused rather than stored. A `pyarrow.RecordBatchReader`
is consumed by reading, so storing one would leave the object empty for whatever
reads it next; it is not picklable either, so the cell fails instead. Call
`.read_all()` and store the table. The check looks for an iterator, so a wrapper
that presents the Arrow protocol over a one-shot reader without being iterable
itself is not caught: it is stored, correctly, but the object is spent
afterwards.

Each value a cell produces is serialized once, even when it is both a variable
and the cell's last expression. That matters for a lazy handle, where writing is
what runs the query: a cell ending in a bare `df` evaluates it once, not once
per artifact.

## Artifact URIs

Cells expose their outputs as artifacts under a stable URI scheme:

```
strata://artifact/<artifact_id>@v=<version>
```

For loop cells, each iteration gets a suffix:

```
strata://artifact/<artifact_id>@iter=<k>@v=<version>
```

A notebook's `<artifact_id>` names a cell variable, not its content: `nb_<notebook_id>_cell_<cell_id>_var_<variable>`. Each new result for that variable is stored as the next `@v=N` of the same id, with its provenance hash (code + inputs + env) recorded on the version. The cache lookup searches by that hash, so same code + same inputs + same env finds the stored version whatever its number. A notebook keeps each id's current value plus its last 3 earlier versions, so reverting a recent edit is still a cache hit ([`STRATA_NOTEBOOK_KEEP_SUPERSEDED_VERSIONS`](../reference/configuration.md#notebook) sets how many).

You don't usually need to construct these ids by hand; the inspect panel surfaces them and `# @loop start_from=<cell-id>@iter=k` references them by cell ID + iteration index, not the full URI.
