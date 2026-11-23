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
2. **Staleness, current code** (SYSTEMS 8 and 9: NB-1, NB-2, BAT-2). Rework
   `Staleness.tla` to the key rule, inline upstream runs, Run All batches
   and REST/MCP edits. Replays drive the handlers with
   `FakeNotebookWebSocket` (`tests/notebook/e2e_fixtures.py`) so an edit
   lands while a run awaits.
3. **Row-group cache soundness** (SYSTEMS 3 and 4: CK-1, MD-1, MD-2, PR-2).
   A stateful Hypothesis test over a real Iceberg table: appends, deletes
   (positional and equality), overwrites, schema evolution, `add_files`, drop
   and recreate at the same location, with scans in between through one
   long-lived planner and cache. Oracle: the same scan with every cache
   cleared, and the rows read straight from pyiceberg. Filtered scans also
   cover PR-2.
4. **Streams** (SYSTEMS 1 and 2: ADM-5, ADM-6, STR-1, STR-2). A
   `Streams.tla` of materialize, GET, admit, release (with its lock as a
   step), cleanup and expiry, with two GETs and two materialize misses.
5. **Run All equals single-cell runs** (BAT-1). A differential property:
   random notebooks run with `notebook_run_all` and cell by cell must leave
   the same stored values and statuses.
6. **Encodings and naming** (PROV-3, ACL-2 beyond the SQL-catalog shape).
   Property tests: transform specs round-trip canonically; for each catalog
   shape, every URI form that loads one table is refused by a deny on any.
