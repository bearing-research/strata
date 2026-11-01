# cellgraph conformance traces

What today's Strata does, step by step, on small scenario notebooks, recorded as
JSON. `cellgraph` must reproduce these traces with its legacy formats
(`docs/internal/cell-graph-package.md`, section 10). `test_cellgraph_traces.py`
re-records every scenario and requires an exact match, so the baseline stays
true. `test_replay.py` replays every step through `cellgraph` and Strata's
vocabulary (`src/strata/notebook/vocabulary.py`) and compares:

- the DAG and `dag_error`;
- names and diagnostics;
- plannability and plans;
- definition, execution and instance keys;
- cache decisions.

`KNOWN_RECORDED_BUGS` there lists what a trace records that is a Strata bug.

```
scenarios/<name>/          notebook directory (cells/, notebook.toml, data) + steps.json
traces/<name>.json         recorded output
record.py                  recorder
test_cellgraph_traces.py   guard
test_cellgraph_r_directives_trace.py   the R scenario's guard and replay, apart for CI's R job
replay.py                  replays a trace through cellgraph
test_replay.py             the replay of every other scenario
test_vocabulary_parity.py  cellgraph's diagnostics against Strata's, on cells no scenario has
```

```bash
uv run python -m tests.contract.cellgraph.record            # compare every trace
uv run python -m tests.contract.cellgraph.record --update   # rewrite them
uv run python -m tests.contract.cellgraph.record --update tables   # one scenario
```

A PR that changes a trace says which behavior changed and why, as for any
snapshot test.

## How a scenario is recorded

`steps.json` is an ordered list of steps, each applied through the code Strata
itself uses:

| Step | Through |
|---|---|
| `edit_cell`, `add_cell`, `remove_cell` | `writer.write_cell`, `add_cell_to_notebook`, `remove_cell_from_notebook` |
| `write_file`, `remove_file` | a file in the notebook directory (a mounted folder, a plain file) |
| `toml_set` (`ai.model`), `set_variant_active`, `set_variant_mode` | `writer._apply_notebook_toml_update`, `set_variant_active`, `set_variant_mode` |
| `set_env` | the server process's environment (`STRATA_NOTEBOOK_CREDENTIALS`) |
| `iceberg_create`, `iceberg_append` | pyiceberg, on the catalog the scenario's `pyproject.toml` declares |
| `seed_dataset` | a named version in the store `@dataset` reads |
| `run_all` | `strata run --no-sync --format json` |
| `run_cell` | `LocalNotebookOps.run_cell` (`strata cell run`) |
| `validate` | nothing; the snapshot only |

After every step the recorder takes a snapshot. A run step's snapshot is taken in
the session that ran, as the UI would see it right after the run. Every other
step's is taken in a freshly opened session, as `strata status` would see it. A
snapshot holds:

- **analysis:** per cell, defines, references, builtin references, upstream ids and
  diagnostics (code, severity, line); the DAG's edges, topological order, variant
  groups, inactive cells, fan-out cells, consumed variables, and `dag_error`;
- **keys:** the definition key (`compute_source_hash`), and the execution key with
  its parts (source hash, env hash, the upstream hashes as
  `session._collect_input_hashes` assembles them, and the full sorted input list),
  per cell and per fan-out instance (`instances`). The parts come from the calls
  `compute_staleness` makes;
- **host-keyed:** for SQL and prompt cells, on run steps, every
  `compute_sql_provenance_hash` / `compute_prompt_provenance_hash` call their
  executor made and its inputs, recorded by a spy that calls through. These keys
  exist only inside a run, so other steps record `null`;
- **resolved:** every mount, table, fetch and dataset component as computed,
  unresolved ones included, and whether the cell has a read-write mount;
- **published:** `artifact_uris` (artifact, version, provenance hash,
  `content_sha256`), the cell-level `artifact_uri`, and fan-out instances
  (`<var>@variant=<name>`);
- **run:** on run steps, per cell: status (ok, error, skipped), cache hit, error
  code, skip reason, the provenance hash the run recorded, and stdout;
- **staleness:** status and reasons per cell;
- **causality:** the causality map, outside the exact gate (design 13.1 replaces it).

## Determinism

- **Nonces.** The six places an unresolved input draws a random component take it
  from `strata.notebook.nonce.unresolved_nonce`. The recorder replaces that one
  function with a counter seeded per scenario. pyiceberg's snapshot id generator is
  replaced the same way, since a table's snapshot id is in the keys of the cells
  that read it.
- **Paths.** A `file://` mount's URI is in its key as written, so every scenario is
  recorded at `/tmp/strata-cellgraph/<name>` on every machine. Scenario files write
  it as `@NB@`. The recorder refuses a trace that names the repository, the home
  directory or the interpreter. Since the path is fixed, a lock per scenario keeps
  two runs on one machine (two checkouts, two users) from deleting each other's copy.
- **Mtimes.** A local mount's listing hashes each file's mtime, so the recorder
  gives every file one mtime after each step.
- **Environment.** No scenario has a `uv.lock`. Cells run with the dev interpreter
  and `--no-sync`, as the harness tests do. The recorder clears `STRATA_*` and proxy
  variables, sets `STRATA_ARTIFACT_DIR` to a per-scenario registry and `HOME` to an
  empty per-scenario directory (so no `~/.strata` cache is read or written), and
  turns warm pools off.
- **Content digests.** Values are ints, strings, lists and dicts (`json/object`),
  except where a feature needs Arrow: `@write_table` (`tables`) and the SQL result
  (`sql`). Those digests follow the locked pyarrow.
- **Time.** No scenario uses `@cache ttl`. `@fetch` is pinned with `sha256=`, so
  no stale check reads the clock.
- **Fetch.** The `fetch_dataset` scenario serves its `serve/` folder on
  `127.0.0.1:47613`, a fixed port, because the URL is in the key.

## Scenarios

| Scenario | Covers |
|---|---|
| `linear` | Basic cells; edit a middle cell; Run All, then a single cache-hit run |
| `mounts` | One mount name set at notebook level, in a cell's `mounts` and by `# @mount` (annotation wins); a `file://` listing change; an `rw` mount whose downstream reader is content-keyed |
| `variants_switch` | Switch group; change the active variant and back (cache hits) |
| `variants_sweep` | Sweep group; add a variant; edit one variant; a collapse consumer |
| `fanout` | `@per_variant`; chained fan-out; a consumer of the fan-out; per-instance keys |
| `shadowing` | Two producers of `state` with a wired consumer; `input` defined upstream and bound by a mount downstream (#1342) |
| `nocache` | A `@nocache` producer whose source stays the same while the file it reads changes; a downstream reader keyed by content; `@stochastic` with `@nocache off` (an error, so the reader is blocked) |
| `tables` | A local Iceberg catalog; append a snapshot; `@write_table`; a table in a warehouse that does not exist (unresolved) |
| `fetch_dataset` | `@fetch` resolved (pinned, local server) and unresolved; `@dataset` resolved, moved to a new version, and unresolved |
| `credentials` | A mount whose credential is not configured, then configured, then removed |
| `blocked` | A malformed annotation (#1343) with a variable reader and an `@after` dependent; a single-cell run of the reader; a dangling `@after` (#1344) |
| `seeded_loop` | `start_from` with a competing ordinary producer of the carry (#1345): editing the producer keeps the fork a cache hit, editing the seed reruns it |
| `cell_level_ref` | A cell whose record has only the cell-level `artifact_uri`, and an `@after` dependent that keys on it |
| `structural` | A cycle and a variant collision: `dag = None` with `dag_error` |
| `sql` | A SQLite cell reading an upstream variable through `:n`; edit the query; edit the upstream |
| `prompt` | A prompt cell with a stubbed model; change the notebook-level `[ai]` model |
| `markdown` | Markdown cells with `# @after`, `# @variant` and `# @app hide` before a Python cell; only `@app` is read there, so the first two add no edge and no group; move the `@after` |
| `r_directives` | An R cell whose source does not parse as Python; edit `@timeout`, add `@name`, add a plain comment (each rekeys it today) |

## Left out or reduced

- **`cell_level_ref` is built by today's code,** not by writing state the old way.
  `artifact_uri` is never written to disk, but a loop cell whose carry no cell reads
  keeps only its carry's cell-level `artifact_uri` (`executor.py`,
  `_set_loop_artifact_uris`), so its `@after` dependent's key takes the cell-level
  branch of `_collect_input_hashes`. That record lives only in the session that ran,
  so it shows in run-step snapshots, not after a reopen.
- **`prompt` uses a stub,** not a model: `chat_completion` and
  `chat_completion_stream` echo the model name and the rendered prompt, as the
  prompt-executor tests' stubs do.
- **`sql` uses SQLite on a file,** not `:memory:`: a `:memory:` path is resolved
  against the notebook directory and fails to open.
- **Remote mounts are not covered.** A failed listing (`mounts.py`, the two
  listing nonces) needs a remote URI; offline it would also need a stub of the
  filesystem library. The nonce routing of both sites is pinned by
  `tests/notebook/test_unresolved_nonce.py`.
- **`r_directives` needs `Rscript`.** Its guard is its own file, skipped where R is
  not installed and run in CI's "R Tests" job.
