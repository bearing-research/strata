# Formal verification: handoff

Written for the next agent (or person) to pick up this work cold. The
findings themselves, with traces and proposed fixes, are in
[`README.md`](README.md). This file covers the state of the work, how to
run it, the conventions, what went wrong along the way, and what to do
next.

## State

- **Branch:** merged to `main` in #867. Fixes land as separate PRs.
- **Findings:** 12, all reproduced against the real code. The status table below marks which are fixed.
  See the status table below.
- **Property tests:** two, both passing apart from known findings.

## Running everything

```bash
uv sync --all-extras                       # as CLAUDE.md says

# Replays: one pytest per finding, each PASSES while its bug exists
uv run pytest formal/ -v                   # ~15 s; property tests skip without
                                           # Hypothesis, finding 8's test on 3.13+
uv run --no-project --python 3.12 --with pytest \
    pytest formal/test_admission_counterexamples.py -k wakeup   # finding 8 on 3.12

# Property tests (Hypothesis is not a project dependency)
uv run --with hypothesis pytest formal/test_pruning_properties.py formal/test_staleness_properties.py
PRUNING_EXAMPLES=60000 uv run --with hypothesis pytest formal/test_pruning_properties.py --timeout=0
STALENESS_EXAMPLES=40 uv run --with hypothesis pytest formal/test_staleness_properties.py --timeout=0

# Model checking (Java is preinstalled in the cloud container)
curl -sSLO https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
cd formal/tla
java -cp ../../tla2tools.jar tlc2.TLC -deadlock -workers 1 -config <Config>.cfg <Spec>.tla
```

`formal/` is outside `testpaths`, so the normal `uv run pytest` never runs
any of this, and CI doesn't either.

## Conventions

- **One model per mechanism, one config per finding.** Each spec has a
  `Patched` (or similar) constant: `FALSE` models the code as it is,
  `TRUE` models the proposed fix. A `*_Patched.cfg` must check
  exhaustively with no violation. The comment on each action names the
  Python function it mirrors; keep that true when the code moves.
- **Every violation gets a replay** against the real code in
  `formal/test_*_counterexamples.py`. The replay **asserts the buggy
  outcome**, so it passes while the bug exists. When a fix lands, invert
  the assertion and move the test into `tests/` as a regression test.
- **Replays reuse the suite's fixtures** by importing them from `tests/`
  (autouse ones included, e.g. `fast_notebook_env` for notebook tests),
  with `# ruff: noqa: F811` at the top of the module.
- **Property-test oracles must not share Strata's logic.** Pruning uses
  exact Python comparisons, and staleness evaluates the current sources
  from scratch.
- **Severity** is a judgment call: how likely the trigger is, and whether
  the result is silent wrong data (high) or a visible failure or a wrong
  label (lower).

## Gotchas (each cost time once)

- TLC `.cfg` files can't hold function or record literals. Put structure
  such as the cell chain in `Staleness.tla` in the spec as definitions.
- With `-workers auto`, TLC traces aren't always the shortest. Use
  `-workers 1` to get a trace worth quoting.
- A model can overstate impact. Finding 11 looked like wrong results in
  TLC, but the replay showed the executor re-checks provenance, so only
  the reported status is wrong. Always replay before rating severity.
- Hypothesis with plain `st.floats()` / `st.integers()` almost never
  produces equal values or NaN next to a number. Mix in a small pool of
  boundary values; `test_pruning_properties.py` shows how. Without it the
  known NaN bug went unfound in 3,000 examples.
- The suite sets a 180 s per-test timeout (`pytest-timeout`, thread
  method). Long property runs need `--timeout=0` or
  `@pytest.mark.timeout(...)`.
- Only variables that some other cell reads are stored as artifacts, so a
  leaf cell's value has no artifact. The staleness property adds a sink
  cell for that reason.
- pyiceberg's `SqlCatalog` scopes tables by catalog name. Strata builds
  warehouse catalogs as `SqlCatalog("strata")`; match it in fixtures.
- Postgres-backed tests (`tests/test_artifact_store_postgres.py`) start a
  container through Docker and skip when the daemon isn't reachable.
- CPython 3.13+ fixed a lost-wakeup in `asyncio.Condition`, and the
  project still supports 3.12. Check concurrency findings on both.

## Findings status

Numbers match `README.md`. A fixed finding keeps its row, and its Replay
column points at the regression test that replaced the replay.

| # | Finding | Replay | Proposed fix (verified in the model where marked) | Smallest first step |
| --- | --- | --- | --- | --- |
| 1 | GC deletes a notebook's current value during a rebuild | `test_artifact_counterexamples.py::test_gc_during_rebuild` | GC also protects the latest ready/superseded version ✓ | Same change in `garbage_collect` |
| 2 | Promoting one notebook's output strands another's | `…::test_cross_id_dedup` | `get_latest_version` accepts `superseded` ✓ | Audit the other callers first |
| 3 | `!=` pruning drops NaN rows | `…::test_nan_ne_pruning` | Don't prune `!=` on float columns | One-line guard in `matches_stats` |
| 4 | Projection fingerprint not injective | `…::test_projection_fingerprint_collision` | Hash `json.dumps(columns)` | Changes every projection cache key once |
| 5 | Stale runner publishes; winner rewrites a ready artifact | `test_build_runner_counterexamples.py` | Per-attempt blob keys + one fenced promote ✓ | Largest change on the list |
| 6 | Stale runner fails the build that replaced it | `tests/test_build_runner.py::TestALeaseDecidesWhoMayFail` | Fence `fail_build` / `fail_artifact` on the lease ✓ | **Fixed** |
| 7 | Old manifest's upload URL still writes | `test_build_pull_counterexamples.py` | Per-attempt blob keys ✓ | Needs the same change as 5 |
| 8 | Lost wakeup on Python 3.12 | `test_admission_counterexamples.py` (3.12 only) | Re-notify on cancel in `ResizableLimiter.acquire` ✓ | ~5 lines |
| 9 | Evicted limiter lets a tenant exceed its quota | `test_admission_counterexamples.py` | Evict only idle limiters ✓ | `get_or_create_quotas` |
| 10 | Shutdown drain misses evicted limiters' streams | `test_admission_counterexamples.py` | Same as 9 ✓ | Same as 9 |
| 11 | Mid-run upstream edit leaves the downstream READY | `test_staleness_counterexamples.py` | Keep the walk's verdict; leave running cells alone ✓ | `_refresh_and_broadcast_changed_staleness` |
| 12 | Deny rule sidestepped by another address form | `test_acl_counterexamples.py` | Name tables by serving catalog | Docs done (`*:ns.*` deny patterns); code fix open |

Suggested fix order, cheapest and safest first: 6, 8, 3, 12 (docs), 1,
9/10, 11, 2, 4, then 5/7.

## Already checked; don't redo

- Interactive vs bulk starvation: impossible by construction, since the
  tiers use separate limiters.
- The deny-first loop in `AclEvaluator.authorize` is correct as written.
  The weakness is naming (finding 12).
- `derive_subkey` label collisions: none possible as used.
- `CacheKey.to_hex` `|` separator: not collidable with real file paths.
- Signed URLs sign `json.dumps(data, sort_keys=True)`, which is canonical.
- Row-group pruning for float64, int64, string, timestamp and decimal:
  60,000 random examples; only finding 3.
- Sequential edits and runs on a diamond of plain Python cells: 40 random
  sequences, all consistent.

## Next verifications, in priority order

Each item says what to prove, where, and how to start. "Lead" marks a
suspicion from reading code that hasn't been confirmed.

1. **Artifact store, non-atomic steps (lead).** `ArtifactLifecycle.tla`
   treats each store method as one atomic step, which the code doesn't
   guarantee:
   - `finalize_artifact` calls `begin_write` on neither backend. It reads
     its row, calls `find_by_provenance` on a *second* connection, then
     updates, so the check and the act are separate. The partial unique
     index is the only backstop.
   - On SQLite, the single-writer lock narrows the window. On Postgres,
     `begin_write` is a keyed advisory lock taken only by
     `create_artifact` and `force_finalize_canonical`, and the rest runs
     under READ COMMITTED.
   - Split `Finalize` and `ForcePromote` into their real read and write
     steps, with locks only where the code takes them, and re-check the
     same three invariants on both backends.
   - Replay any violation with threads on SQLite, and with
     `tests/test_artifact_store_postgres.py`'s Docker fixture on Postgres.
2. **Notebook staleness, wider and concurrent.** Extend
   `test_staleness_properties.py` with more cell kinds: leaf-only cells,
   `@nocache`, `@loop max_iter= carry=`, a failing cell, prompt and SQL
   cells (which use alternate cache schemes), and `# @per_variant`.
   Finding 11 shows the bugs are in the overlaps. To generate overlapping
   steps, drive the handlers directly with `FakeNotebookWebSocket`
   (`tests/notebook/e2e_fixtures.py`) so an edit can land while a run is
   awaiting. Oracle: evaluate the sources from scratch, as now.
3. **Run-all batching equals single-cell runs.** CLAUDE.md names
   `CellExecutor.execute_batch` as the deliberate exception to "the
   artifact store is the sole source of truth". Write a differential
   property test: for random notebooks, `notebook_run_all` must leave the
   same stored values and statuses as running each cell in order.
4. **ACL naming across URI forms.** Generalize finding 12 into a property
   test. For each catalog shape (local warehouse, `catalog_properties`
   SQL catalog, named catalog, S3 warehouse), every URI form that
   `PyIcebergCatalog.load_table` resolves to the same `metadata_location`
   must get the same `TableRef`. It will fail today. It is the acceptance
   test for the fix.
5. **Iceberg manifest pruning.** `test_pruning_properties.py` covers
   Parquet row groups only. File-level skipping uses Iceberg manifest
   bounds, which have their own NaN counts and string truncation. Build
   real Iceberg tables with `temp_warehouse`-style fixtures and check that
   the scan with filters equals the scan without filters, filtered in
   Python.
6. **Disk cache eviction vs concurrent readers (lead).** `cache.py`
   evicts to a size limit, and the Rust extension (`rust/src/lib.rs`)
   mmaps cache files. A small TLA+ model of write (temp file + rename),
   evict (unlink) and read (open, then mmap) would settle whether a
   reader can see a partial or deleted file. Replay any violation with
   threads.
7. **Stream lifecycle.** `QoSAdmission._release`'s docstring says
   `GET /v1/streams/{id}` has no already-consumed guard, so two handlers
   can serve one stream. Model create, attach, stream, disconnect,
   cleanup and QoS release. Invariants: each admission is released
   exactly once, and a stream's bytes are sent at most once per attach.
8. **Per-client semaphore eviction.** `QoSAdmission._get_client_semaphore`
   LRU-evicts semaphores still in use, the same pattern as finding 9.
   Replay it the way `test_admission_counterexamples.py` does for
   tenants (10,000 entries).
9. **`transform_spec.to_json` canonical form.** A property test that
   equal specs always serialize to the same bytes (key order, float
   formatting, tagged filter values) and that different specs differ.
   Core provenance and dedup rest on it.
10. **CI.** Once fixes land, run the `*_Patched.cfg` configs and the
    replays, inverted into `tests/`, in CI. All the patched configs
    except `Artifact_Patched` (about 2 minutes) finish in seconds.
    `Artifact_Patched` with `MaxVer = 2` takes about 2 seconds and still
    catches findings 1 and 2.
