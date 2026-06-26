# Authoring Notebooks Programmatically

A Strata notebook is a **plain-text directory** — TOML config plus one
source file per cell. Anything that can write files can author one: a
script, a code generator, or a coding agent (Claude Code, Codex, …). No
server, no SDK, and no JSON blobs are involved.

This page is the contract for external authors. If you follow it and the
result doesn't open, validate, and run, that's a Strata bug — please
report it.

## The loop

```bash
strata new "My Analysis" --no-env     # 1. scaffold
# 2. write cells/*.py, list them in notebook.toml
strata validate my_analysis           # 3. static checks — cheap, run after every edit
strata run my_analysis                # 4. execute (syncs the venv on first run)
```

`validate` and `run` both take `--format json` and use the same exit
codes: `0` success (warnings allowed), `1` failure with structured
findings on stdout, `2` invocation error on stderr. Iterate on the
diagnostics until both exit `0`.

## Editing files vs. driving commands

Writing the files directly — as below — is the canonical contract, and the
rest of this page documents it. If you'd rather not hand-edit `notebook.toml`
or mint cell ids yourself, the `strata cell` and `strata dep` commands perform
the same edits and print structured JSON:

```bash
strata cell add  my_analysis --file step.py --after load   # mint + insert a cell
strata cell edit my_analysis <id> --file step.py           # replace a cell's source
strata cell rm   my_analysis <id>                          # delete a cell
strata cell mv   my_analysis <id> --to 2                   # reorder
strata dep add   my_analysis pandas                        # uv add
```

They write the same plain-text files described here, so the two approaches are
interchangeable, and the same commands (plus `cell list/show`, `dag`, `status`,
`cell run/test`) also drive a *running* session over `--server/--session`. See
the [Notebook CLI](cli.md) for the full command surface. The rest of this page
is the file-format contract underneath both.

## On-disk layout

```
my_analysis/
├── notebook.toml          # committed config: cells, env, workers, mounts
├── pyproject.toml         # uv-managed dependencies
├── uv.lock                # written by uv; do not hand-edit
├── cells/
│   ├── load.py
│   ├── notes.md
│   └── stats.py
└── .strata/               # runtime state — NEVER create, edit, or commit
```

**`.strata/` is hands-off.** Display outputs, provenance hashes, console
snapshots, and the artifact store live there; the server and CLI manage
it entirely. It's gitignored. If you are generating a notebook from
scratch, simply don't create it.

## notebook.toml — the minimum that works

```toml
notebook_id = "my-analysis-001"
name = "My Analysis"
cells = [
  { id = "load",  file = "load.py",  language = "python",   order = 0 },
  { id = "notes", file = "notes.md", language = "markdown", order = 1 },
  { id = "stats", file = "stats.py", language = "python",   order = 2 },
]
```

**What `strata new` actually writes is a superset of this.** The scaffold
emits a UUID `notebook_id`, `created_at` / `updated_at` timestamps, and
empty `workers = []` / `mounts = []` arrays. **Edit the generated file in
place** — replace `cells = []` with your cell list and leave the other
fields alone. Don't overwrite it with the minimal template above; the
minimum is what *you* must provide when writing from scratch, not what
the scaffold looks like.

The full schema (env, workers, mounts, connections, AI config) is
[notebook.toml Schema](../reference/notebook-toml.md). Rules that matter
when writing it by hand:

- **`notebook_id`** — any stable, unique string. The server generates
  UUIDs; hand-written IDs just need to never change afterwards (artifacts
  are keyed to it — renaming orphans the cache).
- **Cell `id`** — unique within the notebook, used in artifact keys and
  API routes. The server generates 8-char UUID prefixes; hand-written
  short names (`load`, `stats`) are fine. Don't reuse an ID after
  deleting a cell.
- **`file`** — relative to `cells/`. Conventional extensions: `.py`
  (python, prompt), `.R` (r), `.sql` (sql), `.md` (markdown).
- **`order`** — display order *and* reference-resolution order (see
  below). Keep it consistent with the list order.
- **`language`** — one of `python`, `r`, `sql`, `prompt`, `markdown`.

## How variables flow between cells

Each cell's top-level assignments are its **defines**; the free variables
in its source are its **references**. A reference binds to the **nearest
earlier cell** (by `order`) that defines that name — never to a later
cell. There is no notebook-global mutable namespace: every cross-cell
value is an immutable, content-addressed artifact.

Practical consequences for generated code:

- Put producers before consumers in `order`. A reference with no earlier
  definer isn't an error at validate time (it could be an import or a
  builtin), but it will `NameError` at run time.
- Only variables that downstream cells actually reference are persisted.
- Re-running with unchanged source + inputs + environment is a cache hit;
  cells re-execute only when something upstream changed.
- Values cross cells by serialization (Arrow for tabular/numpy, JSON for
  plain data, pickle otherwise) — write cells as if their inputs were
  freshly deserialized, because they are.

## Per-cell configuration: annotations, not TOML

Cell-level settings are `# @` comment lines at the top of the cell
source — they always win over anything persisted elsewhere. The full
surface is [Cell Annotations](annotations.md); the most common:

```python
# @name revenue_model     ← stable display name
# @timeout 120            ← per-cell timeout (seconds)
# @worker gpu-box         ← run on a named worker
# @env API_BASE=https://… ← per-cell env var
# @mount data s3://bucket/path ro   ← injects `data` as a pathlib.Path
# @table trips file:///wh#nyc.trips ← lake input: injects `trips` (URI) and
#                                     `trips_snapshot`; new table data makes
#                                     the cell stale
```

Prompt cells (LLM calls) use the same mechanism:

```python
# @name summary
# @output_schema {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]}
Summarize {{ findings }} as a verdict.
```

`{{ var }}` interpolates upstream variables. `strata validate` checks
annotation syntax (`loop_missing_carry`, malformed `@output_schema`,
unknown workers, …) without calling any LLM.

## What validate catches vs. what run catches

| Failure | Caught by |
| --- | --- |
| Malformed `notebook.toml` (with TOML line numbers) | `validate` |
| Missing / unreadable cell file | `validate` |
| DAG cycle | `validate` |
| Bad annotation (`@loop` without `carry`, schema typos, unknown worker) | `validate` |
| Reference that resolves to nothing (import? typo?) | `run` (NameError) |
| Wrong logic, missing dependency in `pyproject.toml` | `run` |

`run --format json` reports per-cell `status`, `error`, and `cache_hit`,
and skips downstream cells when an upstream fails (`"reason": "upstream
failed"`) — fix the first error and re-run; everything already correct is
a cache hit.

## Verifying computed values

Per-cell `stdout` / `stderr` are included in the `run --format json`
payload (truncated at 10k chars), so the way to check a result is to
**print it** and read it back from the JSON:

```python
# cells/stats.py
total = sum(s["revenue"] for s in sales)
print(f"total={total}")
```

```bash
strata run my_analysis --format json | jq -r '.cells[] | select(.id == "stats").stdout'
# total=93900
```

One caveat: a **cache hit replays the stored artifact without re-running
the cell**, so `stdout` can be absent on warm runs. Pass `--force` when
you need fresh console output. Don't read `.strata/` directly — it's
machine-managed runtime state with no stability guarantees.

## Don'ts

- **Don't create or edit `.strata/`** — runtime state, machine-managed.
- **Don't hand-edit `uv.lock`** — declare dependencies in
  `pyproject.toml` and let `strata run` / `uv sync` resolve. The
  scaffolded `pyproject.toml` is **not empty**: it pins the notebook's
  Python version and pre-seeds the runtime packages the cell harness
  needs (`pyarrow`, `orjson`, `cloudpickle`) — append your dependencies
  to the existing list.
- **Don't reuse cell IDs** or change `notebook_id` after artifacts exist.
- **Don't encode per-cell config in `notebook.toml`** when an annotation
  exists — annotations are the single per-cell configuration surface.
- **Don't write outputs into the notebook directory** from cell code
  unless you mean to commit them; artifacts already persist results.

## Worked example

This exact notebook is pinned by Strata's test suite
(`TestHandWrittenNotebookContract`) — if it ever stops working, CI fails:

`notebook.toml`:

```toml
notebook_id = "agent-handwritten-001"
name = "Hand-written by an agent"
cells = [
  { id = "load",  file = "load.py",  language = "python",   order = 0 },
  { id = "doc",   file = "doc.md",   language = "markdown", order = 1 },
  { id = "stats", file = "stats.py", language = "python",   order = 2 },
]
```

`cells/load.py`:

```python
# @name load
numbers = [1, 2, 3, 4]
```

`cells/stats.py`:

```python
total = sum(numbers)
mean = total / len(numbers)
```

```bash
$ strata validate ./handwritten && strata run ./handwritten
✓ valid — 3 cell(s)
...
3 ran in 1.2s
```
