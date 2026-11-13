# Python cells

The default. A Python cell is just Python source: assignments at module scope become the cell's outputs, and free variables become inputs pulled from upstream cells.

## Writing a Python cell

```python
import pandas as pd

sales = pd.read_parquet("https://example.com/sales.parquet")
by_region = sales.groupby("region")["total"].sum()
```

This cell _defines_ `sales` and `by_region`. A downstream cell that references either name will automatically depend on this one.

```python
# downstream cell, reads by_region from upstream
top_region = by_region.idxmax()
print(f"Top region: {top_region}")
```

```text title="Output"
Top region: West
```

## Variable flow and the DAG

Strata analyzes each cell's AST to extract:

- **Defines**: top-level assignments (`x = 1`, `df = pd.read_csv(...)`)
- **References**: free variables used but not defined locally

The DAG builder links references back to the **last** cell that defined each name (shadowing is handled by order). Edges flow producer → consumer. When you edit an upstream cell, every downstream cell that depends on it becomes stale automatically.

Only variables that a downstream cell references get stored as artifacts. Intermediate scratch variables stay in the subprocess and are discarded when the cell finishes.

## The ambient `strata` client

Every **locally-executed** Python cell gets a ready `strata` client in its namespace, already bound to the running server - no import or construction needed:

```python
# no `from strata_client import StrataClient` / `StrataClient(base_url=...)`
art = strata.materialize(
    inputs=[trips],
    transform={"executor": "scan@v1", "params": {"snapshot_id": trips_snapshot}},
)
strata.set_alias("taxi/tip-model", "champion", art.artifact_id, art.version)
```

It's a lightweight client with the same API surface for the common operations (`materialize`, `put`, `set_alias`, `set_tag`, `resolve_alias`, …); explicit `from strata_client import StrataClient` still works if you want the full client. The ambient `strata` is created fresh per cell run and closed automatically - you don't manage its lifecycle. It's an injected runtime tool, not a cell variable: it doesn't flow downstream and isn't part of provenance (using it has no effect on staleness), exactly like a `@mount` path or an `@table` variable.

Cells routed to a **remote executor worker** (not local execution) don't get the ambient `strata`: one that uses it fails with a sentence saying so. Scan or read artifacts in a cell on the server and pass the result on, or import a client explicitly where the worker can reach the server.

On a server that authenticates callers (`auth_mode` `trusted_proxy` or
`api_key`), the client holds no proxy token or API key. Instead, the server
mints a token for each run and hands it to the cell on its stdin, so it never
lands in a file, argv or the environment. With it the cell acts as the member who started
the run, minus any `admin:` scope, on the client's own routes (materialize,
streams, artifact reads, `put`, tags, names and aliases, the registry audit) and
on promote for its own notebook session; any other route answers 403. The token
stops working when the run ends. Run All uses one token for the whole batch.
On any server the client also sends the run's token and its cell id so the
server can count what the cell's scans read from storage for the
[reads index](../../notebook/lake-aware-cells.md#what-a-run-moved); where callers are not
authenticated, the token grants nothing.

`strata.promote()` sends an upstream result, and the whole chain behind it, to
the team store - by the variable name the cell reads it under, not by an
artifact id you never see:

```python
# `rows` is an upstream cell's variable, so it is one of this cell's inputs.
strata.promote("rows", name="taxi/rows", alias="champion")
```

This is the deliberate half of
[`STRATA_NOTEBOOK_TEAM_CACHE_PUBLISH`](../share/team-store.md#sharing-on-purpose-promoted-and-strata-artifact-promote):
under `promoted`, nothing reaches the team on its own and this is how something
does. It needs a team store configured (`STRATA_NOTEBOOK_REMOTE_STORE_URL`) and
raises if there is none. Only *upstream* results can be promoted from inside a
cell - a cell's own outputs are stored after it returns, so promoting one is a
job for the strip's **Promote to team…** or `strata artifact promote` once it has run.

The ambient client points at this server, never at the team store, and it
carries no team-store credential. When a cell publishes with a name
(`strata.put(inputs=[], transform=…, data=…, name="taxi/tip-model")`), the
artifact lands in the notebook's own store and reaches the team by promotion.
It appears in the [registry dashboard](../../core/registry.md#in-the-notebook-the-registry-dashboard): a promote strip under the cell, and the Registry tab in the bottom drawer (promote, approve, lineage - all in the UI).

> Calls through `strata` are **side effects**. On a cache hit the cell body doesn't re-run, so a `strata.set_alias(...)` won't re-fire - fine for idempotent calls (setting an alias to the version it already points at is a no-op). A cell that stores no variable is cached too, keyed on its own provenance, so a cell whose whole point is the side effect needs `# @nocache` to re-fire.

### Checkpoints: `strata.checkpoint`, `strata.restore`, `strata.head`

A long cell can publish its state as it goes. Each local Python run is an
execution of the notebook's ledger, and three calls on the ambient client speak
to it from inside the cell:

```python
state = strata.restore()  # None on a fresh run
model, epoch = (state["model"], state["epoch"]) if state else (make_model(), 0)
while epoch < 20:
    epoch += 1
    train_one_epoch(model)
    sequence = strata.checkpoint(model=model, epoch=epoch)  # returns 1, 2, 3, ...
    if is_best(model):
        strata.head("model", sequence)  # a named pointer at this checkpoint
```

- `strata.checkpoint(**parts)` stores the parts (any value a cell variable can
  hold) as one checkpoint of the run and returns its sequence number. It does
  not block the cell: the bytes are written beside the run, and the server
  accepts the checkpoint while the cell keeps going.
- `strata.restore()` returns the parts of the checkpoint this run continues
  from as `{name: value}`, or `None` when it starts from nothing. A run that
  died after a checkpoint (a crash, a cancel, a server restart) continues from
  its latest checkpoint on the next ordinary run; the sequence numbers carry
  on from there. A rerun (`↻`, `--rerun`) starts fresh and never restores.
  Only a run of the same computation continues it: after an edit to the cell
  or an upstream, the cell starts fresh, and the interrupted run continues
  only if the edit is undone. It stays resumable for a week after it was
  interrupted; then prune ends it, and its checkpoints are collected unless a
  head points at one.
- `strata.head(name, sequence)` points the run's named head (`model`,
  `best`, ...) at one of its checkpoints. Prune keeps every checkpoint a head
  points at and the latest checkpoint of a run still open; the rest are
  collected like any other version.

The cell's variables are stored as before once it returns; they are the run's
final result, and a cell downstream sees only those. To start a new run from a
checkpoint's parts instead of from nothing (a branch: try another learning rate
from epoch 12), pass the release to run with: `strata cell run <dir> <cell>
--init <release_id>` or `?init=` on the execute route. The branch is its own
run under its own provenance key, so it never serves or overwrites the plain
run's cache. The calls exist only on a local Python run; on a remote worker, an
R cell or a prompt cell they raise a `RuntimeError` saying so.

A PyTorch cell can save through
[vestigo.torch](https://github.com/bearing-research/vestigo/blob/main/docs/torch.md)
instead (`vestigo[torch]` in the notebook's dependencies). Hand it the ambient
client, and torch-checkpointing writes each checkpoint to disk, one file per
key of the state:

```python
from vestigo.torch import HostedRun

run = HostedRun(strata)
state = run.restore({"model": model, "optimizer": optimizer, "epoch": 0})
for epoch in range(state["epoch"] + 1, 21):
    train_one_epoch(model, optimizer)
    sequence = run.save(epoch, {"model": model, "optimizer": optimizer, "epoch": epoch})
run.close()
```

It loads the model and optimizer in place, and resume, branch, heads and prune
work as above. Each file of a checkpoint is its own version, in the layout
vestigo.torch reads, so `vestigo.torch.Ledger(".strata/artifacts").load(release_id,
state)` loads a cell's checkpoint outside Strata. Underneath are two more calls,
which any writer can use:

- `strata.checkpoint_files(step, write)` calls `write(folder)` with an empty
  folder, publishes what it leaves there as one checkpoint, and returns the
  checkpoint's sequence number.
- `strata.restore_files()` returns the folder the checkpoint this run continues
  from was written out to, or `None` on a fresh run.

## Library cells (cross-cell defs and classes)

Top-level `def` and `class` definitions are shared across cells via a synthetic Python module: write a helper once, call it anywhere.

```python
import math

CIRCLE_PRECISION = 4


def area(r):
    return round(math.pi * r * r, CIRCLE_PRECISION)


def perimeter(r):
    return round(2 * math.pi * r, CIRCLE_PRECISION)
```

Downstream cells reference `area(7.5)`, `perimeter(7.5)`, and `CIRCLE_PRECISION` directly.

### How sharing works (slicing)

Defs and classes don't pickle reliably across the subprocess boundary, so they round-trip via **source reconstitution**: Strata writes a slice of the cell's source to disk, re-executes it in a fresh module on the consumer side, and hands the downstream cell the resulting attribute. The slice must be side-effect-free.

The slice keeps the module docstring, `import` / `from import` (no `from X import *`), `def` / `async def`, `class`, and assignments whose RHS is a **literal constant**: numbers, strings, bools, `None`, bytes, negations of literals, and nested tuples/lists/sets/dicts of literals. Everything else (non-literal assignments, augmented assigns, expression statements, control flow, bare annotations) is dropped from the slice but stays in the cell's runtime execution.

A single cell can therefore mix runtime work and library code:

```python
# Runtime, dropped from the slice; flows through the artifact path.
raw_min = round(-math.tau * 7, 2)
raw_max = round(math.tau * 16, 2)
print(f"loaded raw bounds: [{raw_min}, {raw_max}]")

# Library, kept in the slice, exported as a synthetic module.
CLAMP_MIN = 0.0
CLAMP_MAX = 100.0


def clamp(value):
    return max(CLAMP_MIN, min(CLAMP_MAX, value))
```

```text title="Output"
loaded raw bounds: [-43.98, 100.53]
```

A downstream cell can call `clamp(raw_max)`: `clamp` and `CLAMP_MIN/MAX` come from the synthetic module, `raw_max` from the artifact path.

### When the slice isn't self-contained

Every name a kept def or class references must be bound by something else in the slice (or a Python builtin). When it isn't, Strata blocks the export with a `module_export_blocked` diagnostic, surfaced pre-flight, not just at run time.

```python
runtime_threshold = math.sqrt(9)  # dropped, non-literal RHS


def is_outlier(value):
    return value > runtime_threshold
```

> `is_outlier` references names not defined or imported in this cell: runtime_threshold

Other shapes that block on the same principle:

- **Decorators / default values / base classes** evaluated at module load: `@my_decorator` where `my_decorator` isn't imported in the same cell, or `class Child(Parent)` where `Parent` is computed at runtime.
- **Divergence**: a name kept by the slice is also reassigned by dropped runtime code, so the synthetic module's value would differ from the cell's final state. `def f(): ...; f = wrap(f)` exports the unwrapped `f`.
- **Lambda assignments**: `add = lambda x: x + 1`. Even though `cloudpickle` could serialize the value, the synthetic-module path is reserved for source-backed library code.

The fix is usually one of: move the runtime line into its own cell, add the missing import to the same cell as the def, or take the dependency as a function argument.

### Single-cell scope

The synthetic module is built from one cell's source only, with no transitive composition across cells. A def can't reach a name imported or defined in a different cell; each cell that hosts library code carries its own imports.

One concession: annotations that reference names outside the slice would normally block, but adding `from __future__ import annotations` relaxes this. PEP 563 stringifies annotations and the free-variable check drops them, so cross-cell type hints "just work" with the future import.

Walked through end-to-end in the [`library_cells`](../../examples/library_cells.md) example notebook.

## Mutation warnings

A cell that changes an upstream value in place, where the source shows it (`df["col"] = ...`, `obj.attr = ...`, `df.drop(columns=[...], inplace=True)`), is read as producing a new version of that variable: the cell becomes its producer for downstream cells, and the upstream artifact is untouched.

Some mutations can't be seen in the source: an alias (`d = df; d["col"] = ...`), a helper function that mutates its argument, a bare mutator like `rows.append(x)`. Strata compares each input before and after the cell runs and raises a **mutation warning** for those, because the change is not stored: a downstream cell run on its own reads the value from before.

The fix is to copy, or rebind, before mutating:

```python
df = upstream_df.copy()  # make a private copy
df["total"] = df["a"] + df["b"]
```

Warnings show in the cell's header pill, beside its annotation issues, and in the run
report (`strata run`, `strata cell run`, MCP `run_cell`) as `mutated_input` rows of
`run_diagnostics`. A cache hit shows them again, since the result it serves is the same.

## Reads the cell does not declare

A cell's key holds what it declares: its source, upstream values, the lockfile, its mounts,
`@table`, `@fetch` and `@dataset`. A file it opens around them is not in the key, so its
cached result stays when that file changes. Each run records what it read, listed and
imported, and says which of those nothing declares, with the fix:

| Code | When | Fix |
| --- | --- | --- |
| `undeclared_read` | a file outside every mount | `# @mount <name> <dir> ro`, and read it through `<name>` |
| `undeclared_listing` | a directory listed outside every mount | the same mount, over the directory |
| `undeclared_import` | a module loaded from a file outside the environment | make its import root a package and `uv add --editable <dir>` |
| `unlocked_package` | a module whose package `uv.lock` does not hold | `uv add <package>` |
| `work_dir_read` | a file in `work/` this run did not write | return the data from the cell that writes it |
| `subprocess_run` | the cell started a program | its reads are not seen; declare them as mounts |

```text
This cell read `/data/events.csv`, which no mount covers, so its cached result will not
change when the file does. Add `# @mount data /data ro` and read `data / "events.csv"`.
```

The mount is named after its folder, never with a name the cell already uses, and a
folder in or beside the notebook is written as a [relative path](../data/mounts.md)
(`../data`); one with a space in it as a `file:` URI (`file:../My%20Data`). Files share
the mount of their nearest common folder, though never the root, the home folder, the
notebook's folder or one above it: a mount over the notebook would key the cell on every
other cell. A file the run wrote before reading it is its own, and the interpreter, the
venv, system folders and the dot-folders tools keep in the home folder are left out.

The findings show in the header pill and the run report, as mutation warnings do,
and a cache hit shows them again. `subprocess_run` and a failed audit are notes, not
warnings. They never block a run or change its status or key. They describe the result
the cell holds, so a failed run replaces them only when it found something itself.
pyarrow, polars and DuckDB read files in native code, where Python sees no open, so
their readers that take a path are wrapped in the cell's process and say what they were
given: pyarrow's `parquet`, `dataset`, `feather`, `orc`, `csv`, `json` and `ipc` readers
and `memory_map`, `pd.read_parquet`, polars' file readers (`read_*` and `scan_*`, not
`read_database` or `scan_iceberg`), and DuckDB's `read_parquet`, `read_csv` and
`read_json` (on the module or a connection). A folder counts as each file in it but the
`_` and `.` ones a dataset skips (`_SUCCESS`, `.crc`), though a Delta table's
`_delta_log` is read, since it says which files are the table. A glob counts as what it
matches. Their writers (pyarrow's, `DataFrame.to_parquet`, polars' `write_*` and
`sink_*`, a DuckDB relation's `write_parquet` and `write_csv`) make files and folders that
are the run's own, as `open(path, "w")` does, so reading them back is not a finding. A file named
in DuckDB SQL (`select * from 'x.parquet'`, `COPY ... TO`), one opened with
`pyarrow.OSFile`, and a path on another filesystem (`filesystem=` other than the local
one) are not seen.

### When such a file changes

A local run also notes each such file as it read it (its size and modification time) and
each folder it listed by the names in it, but not the `.` ones a tool drops (a
`.DS_Store`). When one of them changes, the cell stays ready,
since its key did not move and a plain run would get the same cached result, but it says
the result may be out of date: a **may be out of date** badge in the UI, `changed_reads`
in `strata status`, `strata cell show` and MCP `status`, and a hint naming the files. A
plain run that hits the cache keeps saying it. A file that changed while the run had it
is out of date as soon as the run ends; one the run wrote itself is not. Rerun the cell
(↻, `strata cell run --rerun`) to read them again, or mount them so the key follows
them. Nothing watches the files: a ready cell's are checked whenever staleness is
worked out (opening the notebook, an edit, after a run) and whenever its status is
read (`strata status`, `strata cell show`, MCP `status`, a page reload), over up to 100
per cell, the files it read before the modules it imported. A remote worker's run is not
checked: its files are on the worker.

## Where a cell runs: `work/`

A cell runs in its notebook's `work/` directory, not beside `notebook.toml`. A file it writes by a relative path (`plt.savefig("fig.png")`) lands in `work/fig.png`, and a later cell reads it back by the same path. `work/` is made on the first run and kept between runs. The `.gitignore` that `strata new` writes leaves it out of git, exports and snapshots leave it out too, and it is never on `sys.path`, so a file a cell wrote there is not imported in place of a library. R cells run there as well.

A file beside the notebook is one directory up: read it as `../data.csv`. A Python cell that opens `data.csv` and finds it only beside the notebook fails with a hint saying so. A SQL connection's relative `path` still resolves against the notebook's directory.

A cell on a [worker](../../notebook/workers.md) runs in a `work/` of its own, made for the run and removed with it. To keep a file from a worker run, write it to an [`# @output_dir`](../../reference/annotations/python.md#output_dir) folder.

While a Python cell runs, `STRATA_CELL_ID` holds its id, in the cell and in any process it starts. Code that runs both in cells and from plain scripts can read it to tell the two apart, for example to log which cell did a slow load.

## Python-cell annotations

| Annotation         | What it does                                     |
| ------------------ | ------------------------------------------------ |
| `# @name X`        | Display name for the DAG view                    |
| `# @worker X`      | Route execution to a named remote worker         |
| `# @timeout 60`    | Override execution timeout (seconds, default 300) |
| `# @env KEY=value` | Set an env var for this cell only - non-sensitive values only; literal lands in committed source. For secrets use `notebook.toml [env]` or the Runtime panel. |
| `# @mount …`       | Attach a filesystem mount (see [Annotations][a]) |
| `# @table …`       | Declare an Iceberg table input; its snapshot joins the provenance |
| `# @fetch …`       | Download a URL as a recorded input; its digest joins the provenance |
| `# @dataset …`     | Bind a registry name as a recorded input; its version joins the provenance |
| `# @write_table …` | Write a variable into an Iceberg table when the run succeeds (see [`@write_table`](../../reference/annotations/python.md#write_table)) |
| `# @output_dir X`  | Give the cell a folder whose contents are an output (see [`@output_dir`](../../reference/annotations/python.md#output_dir)) |
| `# @nocache`       | Always re-execute; never serve a cache hit       |
| `# @stochastic`    | Every run persists beside the cache and never serves it (see [`@stochastic`](../../reference/annotations/python.md#stochastic)) |
| `# @loop …`        | Turn the cell into a [loop cell](loops-and-variants.md#loop-cells)    |

See [Cell Annotations][a] for the full reference.

[a]: ../../reference/annotations/index.md
