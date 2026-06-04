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

## Don'ts

- **Don't create or edit `.strata/`** — runtime state, machine-managed.
- **Don't hand-edit `uv.lock`** — declare dependencies in
  `pyproject.toml` and let `strata run` / `uv sync` resolve.
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
