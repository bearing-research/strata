# Formal verification: handoff

For the next agent (or person) picking this up cold. What the systems must
do and what checks it is in [`SYSTEMS.md`](SYSTEMS.md); how to run things is
in [`README.md`](README.md).

## State

- **2026-09:** the first investigation found twelve bugs, all fixed
  (#868-#879); [`FINDINGS.md`](FINDINGS.md) has them.
- **2026-11-22:** a review against `main` (667 commits later) found the two
  Strata models still finding their bugs and their patched configurations
  passing, but nothing ran them, and much of the system had no
  specification. It added `SYSTEMS.md`, `run_models.py`, the `Formal`
  workflow and the `Mirrors:` drift test, and listed the open properties
  below. Each lead in `SYSTEMS.md` is unconfirmed until replayed.
- **Phase 1 (local runs):** `LocalRun.tla` found finding 13, a lapsed lease
  losing a run nothing else wanted, and a race in the first fix; both fixed.
  LR-2 (the check-then-act in `open` across processes) and LR-4 (sequence
  collision) remain leads; LR-5 was fixed in #1613.
- **Phase 2 (staleness):** `Staleness.tla` now has REST/MCP edits and Run All.
  It found findings 14 and 15, both fixed. Two survey leads did not hold up on
  replay: a batched cell edited while it waits is corrected by the next cell's
  recompute (only the last cell stayed wrong, finding 14), and the MCP reload's
  new cell objects still receive the run's outputs.
- **Phase 3 (cache soundness):** `test_cache_soundness_properties.py`, a
  Hypothesis state machine over a real Iceberg table, found finding 16 (four
  caches keyed a data file by path and size); fixed. Lead CK-1 (b) rests on
  writers never reusing a delete file's path; MD-2 rests on random snapshot
  ids; MD-1 is still a lead (pyiceberg has no catalog that reuses metadata
  names).
- **Phase 4 (streams):** `Streams.tla` found finding 17 (a stream expired under a
  GET waiting for a slot), and a replay confirmed lead ADM-5 as finding 18; both
  fixed. Two leads did not hold: ADM-6 (no cancel can land in `release()`, which
  never suspends; now pinned by a test) and STR-2 (stream ids are fresh uuids, so a
  second miss never replaces another's cleanup).
- **Phase 5 (Run All):** `test_run_all_properties.py`, a differential property over
  random notebooks run with Run All and cell by cell, found findings 19 and 20 (the
  statuses of cells skipped behind a failure) and confirmed the mutation lead as
  finding 21 (a batched cell read an input the cell before changed in place); all
  fixed.
- **Phase 6 (encodings and naming):** `test_encoding_properties.py` checks PROV-3 for
  transform specs and ACL-2 against an oracle of which URIs load one table. It found
  finding 22 (a deny matched one spelling of a name, and Glue and Hive resolve names
  case-insensitively); fixed. The `scan@v1` identity hash is a lead (PROV-3).

## Conventions

- **One model per mechanism, one configuration per finding.** A spec keeps
  a `Patched` (or similar) constant: `FALSE` models the code before a fix,
  `TRUE` the code after. Pin every configuration's outcome in
  `run_models.py`, and list the code each action mirrors under `Mirrors:`.
- **An action is one atomic step of the code**: a database transaction, or
  a stretch of a coroutine between awaits. Splitting a Python operation into
  the steps it really takes is where the bugs are (findings 1, 5 and 9 all
  came from a step the code thought was one).
- **Every violation gets a replay** against the real code, asserting the
  buggy outcome, before it counts or is rated. The fix inverts it into a
  regression test under `tests/` and marks the property's row in
  `SYSTEMS.md`.
- **Property-test oracles must not share Strata's logic.** Pruning uses
  exact Python comparisons; staleness evaluates the sources from scratch.
- **Severity:** silent wrong data is high; a leaked resource, a wrong label
  or a visible failure is lower. Replay before rating: a model can overstate
  impact (finding 11 looked like wrong results; only the status was wrong).

## Gotchas (each cost time once)

- TLC `.cfg` files can't hold function or record literals; put structure in
  the spec as definitions.
- With `-workers auto`, TLC traces aren't always the shortest. Use
  `-workers 1` for a trace worth quoting.
- Hypothesis with plain `st.floats()` / `st.integers()` almost never
  produces equal values or NaN next to a number. Mix in a pool of boundary
  values, as `test_pruning_properties.py` does.
- The suite's 180 s per-test timeout applies to `formal/` too; long runs
  need `--timeout=0`.
- Only variables some other cell reads are stored, so a leaf's value has no
  artifact; the staleness property adds a sink cell for that reason.
- pyiceberg's `SqlCatalog` scopes tables by catalog name. Strata builds
  warehouse catalogs as `SqlCatalog("strata")`; match it in fixtures.
- CPython 3.13+ fixed a lost wakeup in `asyncio.Condition`, and 3.12 is
  supported. Check concurrency findings on both.
- Pytest temp dirs from repeated property runs fill the disk; pass
  `--basetemp` in a scratch directory and delete it after.

## Already checked; don't redo

- Interactive vs bulk starvation: impossible by construction (ADM-4).
- The deny-first loop in `AclEvaluator.authorize` (ACL-1).
- `derive_subkey` labels, the `to_hex` separator, signed-URL encoding
  (PROV-1, PROV-2, PROV-4).
- Row-group pruning for five types: 60,000 examples (PR-1).
- Sequential edits and runs on a diamond of Python cells (NB-3).

## Plan for the open properties

In order of consequence over effort. Each step is one PR: the model or
property, the replays of what it finds, then a PR per fix.

1. **Local runs** (SYSTEMS 10). Done for the lease (LR-3, LR-6). Open: LR-2,
   two processes opening one computation (extend `LocalRun.tla` with `open`
   split into its transactions, and two fresh opens); LR-4, the checkpoint
   sequence after an abandoned release (a replay is enough).
2. **Staleness** (SYSTEMS 8 and 9). Done for NB-1 and NB-2. Open: BAT-2 (a
   batched `@nocache` producer run again by a later single-cell consumer; a
   replay through `_execute_run_all` is enough), and content-key readiness and
   uncached leaves in the model (see SYSTEMS 8, what the model leaves out).
3. **Row-group cache soundness** (SYSTEMS 3 and 4). Done for CK-1, MD-2 and
   PR-2 on integer columns. Open: MD-1 (a catalog that writes `vN.metadata.json`,
   dropped with purge and recreated), CK-4 (`clear()` racing a `put`), and string
   and NaN manifest bounds for PR-2.
4. **Streams** (SYSTEMS 1 and 2). Done for ADM-5, ADM-6, STR-1 and STR-2.
   Open: the passthrough path (no artifact store) and multi-node ownership are
   outside `Streams.tla`.
5. **Run All equals single-cell runs** (BAT-1). Done for Python cells over ints
   and lists. Open: module state, `@nocache` (BAT-2), workers and loops in the
   generator.
6. **Encodings and naming** (PROV-3, ACL-2). Done for transform specs, the SQL
   catalog shape and spellings. Open: the `scan@v1` identity hash (encoding its
   fields as JSON re-keys every stored scan once).
