# Formal verification

Strata's stateful systems, the properties they must keep, and the machinery
that checks them.

- [`SYSTEMS.md`](SYSTEMS.md) is the specification: each system's state and
  actors, its properties (ADM-1, CK-1, …), and what checks each one, or that
  nothing does yet.
- [`FINDINGS.md`](FINDINGS.md) is the record of the 2026-09 investigation,
  whose twelve findings are all fixed.
- [`HANDOFF.md`](HANDOFF.md) is for picking the work up: state, conventions,
  gotchas and the plan for the open properties.

The approach is lightweight and aimed at specific mechanisms. Proving the
Python correct (Coq, Lean, Dafny) would cost far more than it returns; two
cheaper techniques do pay off:

- **Model checking** (TLA+ with TLC) for concurrent state machines: each
  action is one step the code takes atomically (a transaction, a stretch of
  coroutine with no await), so TLC explores every interleaving of the others.
- **Property tests** (Hypothesis) for the assumptions behind pure functions:
  the real code runs on generated inputs and an oracle that does not share
  its logic judges the result.

Every counterexample is replayed against the real code before it counts, and
the replay becomes the fix's regression test.

## What's here

| File | Purpose |
| --- | --- |
| `tla/Admission.tla` | One tenant's `ResizableLimiter` (acquire with deadline, release, cancel while queued) and the `TenantRegistry` LRU that owns it (SYSTEMS 1) |
| `tla/Admission_Py312.cfg` | CPython 3.12 `asyncio.Condition` semantics. Violates `NoLostWakeup` (finding 8) |
| `tla/Admission_Eviction.cfg` | LRU eviction of a limiter in use. Violates `DrainSeesAll` and `WithinQuota` (findings 9, 10) |
| `tla/Admission_Patched.cfg` | The code as it is. All invariants hold |
| `tla/Staleness.tla` | Cell status for a chain `a → b → c` under WebSocket, REST and MCP edits, single runs and Run All (SYSTEMS 8) |
| `tla/Staleness_EditDuringRun.cfg` | An upstream edited while a downstream runs. Violates `RunningShown` (finding 11) |
| `tla/Staleness_ApiEdit.cfg` | A REST or MCP edit during a run. Violates `RunningShown` (finding 15) |
| `tla/Staleness_BatchEdit.cfg` | A Run All cell edited while it waits. Violates `ReadyMeansCurrent` (finding 14) |
| `tla/Staleness_Patched.cfg` | The code as it is. All invariants hold |
| `tla/LocalRun.tla` | A local cell run holding its build's lease: releases, the renewal task, lapses, a rival process that resumes the build, the run's close (SYSTEMS 10) |
| `tla/LocalRun_Lapse.cfg` | The code before the fix. Violates `NoNeedlessFence` (finding 13) |
| `tla/LocalRun_Race.cfg` | The fix without its lock. Violates `NoNeedlessFence` |
| `tla/LocalRun_SharedOwner.cfg` | The rival shares the run's `hostname:pid` and the owner is the process. Violates `CloseLeavesOthers` (finding 32) |
| `tla/LocalRun_Patched.cfg` | The code as it is. All invariants hold |
| `tla/Streams.tla` | One stream-mode miss: GETs waiting for admission, 429s and cancels, the background build, the cleanup timer and expiry (SYSTEMS 2) |
| `tla/Streams_Retry.cfg` | The code before the fix. Violates `NoFailedServe` (finding 17) |
| `tla/Streams_Patched.cfg` | The code as it is. All invariants hold |
| `run_models.py` | Runs every configuration and checks it ends as pinned: passes, or violates its named invariant |
| `test_pruning_properties.py` | Property: a pruned row group holds no matching row (PR-1) |
| `test_staleness_properties.py` | Property: random edits and runs on a real notebook match a fresh evaluation (NB-3) |
| `test_run_all_properties.py` | Differential property: random notebooks (in-place mutation, failures, functions and modules, `@nocache`, `@loop`, cells Run All runs on their own) run with Run All and cell by cell, fresh and after an edit, leave the same statuses and stored values (BAT-1); Run All runs a counted `@nocache` cell once (BAT-2) |
| `test_bounds_pruning_properties.py` | Pruning by truncated string bounds and NaN-bearing doubles, files, row groups and equality deletes, keeps every live row (PR-2) |
| `test_encoding_properties.py` | Transform specs read back from their canonical JSON (PROV-3); a deny on any name of a table refuses every URI that loads it (ACL-2) |
| `test_cache_soundness_properties.py` | Stateful property: a long-lived planner and row-group cache serve exactly the live rows of a real Iceberg table under writes, deletes, schema evolution and recreation (CK-1, PR-2, MD-2) |

The artifact lifecycle and build lease models live in the
`bearing-research/vestigo` repository with the ledger, under `formal/tla/`.

## Running

```bash
uv run python formal/run_models.py           # every TLA+ configuration (Java 11+)
uv run python formal/run_models.py Admission # one spec's configurations
uv run --with hypothesis pytest formal/ -v --timeout=0
PRUNING_EXAMPLES=60000 uv run --with hypothesis pytest formal/test_pruning_properties.py --timeout=0
```

`run_models.py` fetches the pinned TLC release (tla2tools v1.7.4, TLC 2.19)
into `~/.cache/strata-formal`, checking its digest; set `TLA2TOOLS` to use
another jar. It runs TLC with `-deadlock`, because a bounded model that has
used up its bound has no enabled action, which is expected. A configuration
expected to violate runs with one worker, so the violation TLC reports first
is the same every time.

The `Formal` workflow (`.github/workflows/formal.yml`) runs both on changes
to `formal/`, on `main` after any source change, and weekly with a larger
property budget. `formal/` is outside `testpaths`, so plain `uv run pytest`
does not run it. `tests/test_formal_models.py`, which does run with the unit
suite, checks that every `strata/<path>.py::Qualname` a spec lists under
`Mirrors:` still exists.

## Keeping models honest

A model helps only while it matches the code.

- Each spec's header says which code each action mirrors and lists it under
  `Mirrors:`. When that code changes meaning, not just name, re-read the
  model: the drift test catches a rename, not a new code path.
- Every configuration is pinned in `run_models.py`'s `MODELS`, and an
  unlisted configuration fails the run.
- A configuration that shows a bug stays after the fix, as the record of what
  the model catches; its `Patched` twin must pass.
