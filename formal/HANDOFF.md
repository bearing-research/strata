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
- **Phase 7 (replays):** four leads replayed against the real code. LR-4, BAT-2 and
  CK-4 held (findings 23, 24 and 25), all fixed; MD-1's short read did not (both S3
  readers raise on a truncated body).

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
property, the replays of what it finds, then the fixes.

1. **Pruning over strings and NaN** (PR-2). Extend `test_cache_soundness_properties.py`
   with string columns (Iceberg truncates their manifest bounds) and float columns
   with NaN counts: the one place pruning can still drop rows unseen.
2. **Two processes on one store** (LR-2, EACH-2). Half of LR-2 is fixed (finding
   26, #1653): a rerun no longer fails a build another process just resumed. The other
   half, two processes that both find no open build each creating one, waits on a
   computation-keyed open in vestigo (`open_first_attempt`; the handoff is
   `docs/internal/one-open-execution.md` in vestigo's checkout, its HANDOFF item 8).
   After that release, extend `LocalRun.tla` with `open` split into its transactions
   and two processes. EACH-2's guard against collecting during a run is a
   process-local counter.
3. **A wider Run All property** (BAT-1). Module state, `@nocache`, `@loop` cells and
   worker routing in the generator.
4. **Smaller open items.** MD-1 (a catalog that writes `vN.metadata.json`, dropped
   with purge and recreated); LR-3 (the close path checked against the process, not
   the attempt); the passthrough stream path and multi-node ownership (outside
   `Streams.tla`); the `scan@v1` identity hash (PROV-3, a decision: encoding its
   fields as JSON re-keys every stored scan once).
