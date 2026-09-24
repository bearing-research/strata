# Formal verification of Strata: investigation

**Question:** can formal methods usefully model and verify Strata's core components?

**Answer:** yes, as long as the tools are lightweight and aimed at specific
components. Proving the Python implementation correct (Coq, Lean or Dafny
against the real code) would cost far more than it returns. Two
cheaper approaches do pay off:

- **Model checking** (TLA+ with TLC) for the concurrent state machines.
- **Checking the assumptions behind the pure functions** that Strata's
  invariants depend on (pruning soundness, cache-key injectivity), using
  SMT solvers or property-based tests.

As a proof of concept, one TLA+ model (under 200 lines) of the artifact
lifecycle found two real bugs in about a second. A review of the
assumptions behind the pruning and cache-key invariants found two more.
All four reproduce against the real code (`replay_counterexamples.py`).

## What's here

| File | Purpose |
| --- | --- |
| `tla/ArtifactLifecycle.tla` | Model of `artifact_versions`: create, write blob, fail, `finalize_artifact`, `force_finalize_canonical`, `garbage_collect` |
| `tla/GC_Rebuild.cfg` | One id with GC on. Finds finding 1 |
| `tla/CrossIdDedup.cfg` | Two ids sharing a provenance, GC off. Finds finding 2 |
| `tla/Patched.cfg` | Both ids, GC on, with the proposed fix. **All invariants hold** (5,997,420 distinct states, exhaustive, about 2 min) |
| `replay_counterexamples.py` | Runs each counterexample against the real `ArtifactStore` / `ReadPlanner` / `CacheKey` |

```bash
curl -sSLO https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
cd formal/tla
java -cp ../../tla2tools.jar tlc2.TLC -config GC_Rebuild.cfg   -deadlock ArtifactLifecycle.tla   # violation
java -cp ../../tla2tools.jar tlc2.TLC -config CrossIdDedup.cfg -deadlock ArtifactLifecycle.tla   # violation
java -cp ../../tla2tools.jar tlc2.TLC -config Patched.cfg      -deadlock -workers auto ArtifactLifecycle.tla  # passes
cd ../.. && uv run python formal/replay_counterexamples.py
```

`-deadlock` disables deadlock checking, because a bounded model with
every version slot used has no enabled action. That is expected, not a bug.

## The model

Each SQL transaction is one atomic TLA+ action. The notebook write path
(`artifact_integration.store_cell_output`) is **two** transactions,
`finalize_artifact` followed by `force_finalize_canonical`, so other
actions can run between them. The model abstracts away tenants
(single tenant), names and aliases (every version is unnamed, which is
exactly the case for notebook cell outputs), and time (GC may treat any
row as older than the cutoff).

Invariants checked:

- `UniqueReadyPerProv`: at most one `ready` row per provenance
  (the partial unique index). **Holds.**
- `ServableHasBlob`: `ready` and `superseded` rows always have their bytes. **Holds.**
- `CurrentValueDurable`: once an id has had a current value,
  `get_latest_version(id)` keeps returning one. Notebook cells load their
  inputs this way (`executor._load_input_blobs`), and a `None` silently
  leaves the variable out of the downstream cell's namespace. **Violated.**

## Findings

### 1. GC deletes a notebook's current value while a rebuild is in flight

`garbage_collect` protects "the latest version of its id" as
`MAX(version)` over rows in **any** state
(`artifact_store.py`, `garbage_collect`). A rebuild creates a
`building` row first, so while it runs, the previous `ready` version
is no longer the maximum and can be collected. TLC trace (5 steps):

```
a@v1 ready → create a@v2 (building) → GC deletes a@v1 → a@v2 fails
⇒ get_latest_version(a) = None, permanently
```

Even if the rebuild succeeds, the value is missing for the whole build
window. A failed rebuild also leaves a `failed` row as `MAX(version)`,
so the old value stays unprotected from then on. This is the same
class of bug as the "GC wiped week-old notebooks" fix described in the
method's comment: that fix covered the steady state but not an in-flight
rebuild. It triggers when periodic GC
(`artifact_gc_interval_seconds`) or `POST /v1/artifacts/gc` runs while someone
re-runs a cell whose last successful run is older than
`artifact_gc_max_age_days`.

### 2. `force_finalize_canonical` strands the *other* notebook's output

Two notebook cells with the same provenance (same source, inputs and
lockfile, e.g. a duplicated notebook) produce different artifact ids.
The second finalize is deduplicated and marked `failed`. The notebook
path then calls `force_finalize_canonical`, which supersedes the first
id's `ready` row to promote its own. `get_latest_version` accepts only
`ready`, so the first notebook's downstream cells now lose their input.
TLC trace (7 steps), with no GC involved:

```
A@v1 ready (prov p) → B@v1 finalize → deduped, failed → force_finalize_canonical(B)
⇒ A@v1 superseded, get_latest_version(A) = None
```

Running A again restores A and strands B, so the two notebooks keep
invalidating each other.

### 3. `!=` pruning drops NaN rows (pruning soundness)

`Filter.matches_stats` is sound only if `[min, max]` bounds every value
in the row group. Parquet writers leave NaN out of min/max statistics,
so for `[5.0, NaN]` the stats are `min == max == 5.0`, and
`value != 5.0` prunes the row group. That drops the NaN row, even though
`NaN != 5.0` is true in Arrow, DuckDB and Python. This violates
invariant 2 (*conservative pruning*). A model of `matches_stats` alone
would prove it correct; the bug is in the assumption about what the
statistics contain. The same review would cover null-only row groups,
`_convert_stats` coercions and mixed-type comparisons (the last are
already caught by the `except` and not pruned).

Suggested fix: for `NE` on floating-point columns, never prune (or prune
only when the column's `nan_count` is known to be 0).

### 4. Projection fingerprint is not injective (cache-key soundness)

`CacheKey.compute_projection_fingerprint` hashes `",".join(columns)`, so
`["a,b"]` (one column) and `["a", "b"]` produce the same key, and one
projection's cached Arrow IPC can be served for the other. Iceberg and
Parquet both allow commas in column names. It's rare, but it breaks invariant 1
(*immutability ⇒ correctness*), which assumes the key identifies the
content. The `|`-joined `to_hex` key string has the same shape, but its
integer fields make a collision there much harder to construct.

Suggested fix: hash a length-prefixed or JSON encoding (for example
`json.dumps(columns)`). This changes every projection cache key once,
which is safe because the cache is content-addressed and simply refills.

## Proposed fix for 1 and 2 (verified in the model)

`Patched = TRUE` in the spec:

1. `get_latest_version` treats `superseded` as current along with `ready`.
   A superseded row is already defined as "still fetchable by id+version,
   excluded only from provenance lookups", so this fits its meaning, and a
   refresh rebuild still resolves to the newer `ready` version because its
   version number is higher.
2. `garbage_collect` additionally skips each id's latest current
   (`ready`/`superseded`) version. It **must keep** the existing
   `MAX(version)` rule as well. An earlier draft of this patch dropped
   that rule, and TLC found a new 11-step bug: deleting the highest row
   lets `create_artifact` (`MAX(version)+1`) reuse its version number,
   and a still-pending `force_finalize_canonical` for the old row then
   promotes the new row, which has no blob, to `ready`. This is the kind
   of interleaving that is hard to find by reading or testing.

With both changes, TLC exhaustively checks all four invariants for 2 ids,
2 provenances and 3 versions per id. The other callers of
`get_latest_version` (names, registry, CLI) need to be checked before
change 1 ships. If any of them relies on "ready only", add a separate
`get_current_version` for the notebook instead.

## Where else formal methods would pay off

Ranked by (likely bugs × consequence) ÷ modelling effort:

| Component | Technique | Why |
| --- | --- | --- |
| **v2 pull build protocol** (`transforms/runner.py`, `build_store.py`, `signed_urls.py`, finalize callback) | TLA+ | Distributed: executor retries, duplicate or late `finalize_url` POSTs, lease expiry versus `sweep_zombie_builds`, crash between upload and finalize. The same style as the model here, and it could reuse the lifecycle actions. |
| **Notebook cascade + staleness** (`cascade.py`, `dag.py`, `session.compute_staleness`) | TLA+ or Hypothesis stateful testing | Invariants such as "after a cascade the target's inputs are READY and match current provenance", and "a source edit marks exactly the downstream closure stale". The concurrency comes from WS source flushes arriving during execution. |
| **Two-tier QoS / per-tenant limiters** (`rate_limiter.py`, `transforms/build_qos.py`) | TLA+ with fairness (liveness) | "Interactive requests are never starved by bulk" is a liveness property. Tests can't show it; TLC can, under weak fairness. |
| **Pruning soundness** (`filters.py`, manifest pruning, `_convert_stats`) | Z3 / CrossHair over the comparison logic, plus Hypothesis round-trips through real Parquet files | Finding 3 shows the value is in encoding the stats assumptions (NaN, null, type coercion) explicitly. |
| **Hash/key encodings** (`CacheKey.to_hex`, `derive_subkey`, provenance, `transform_spec.to_json`) | Property tests for injectivity | Finding 4. Cheap, and a collision means serving wrong data. |
| **ACL deny-first** (`auth.AclEvaluator`) | Z3 or exhaustive enumeration | Small and pure. Prove "a matching deny cannot be overridden by any allow set" and pattern/tenant corner cases. Low risk today, but a cheap guard against regressions. |

Not worth it: the data plane's streaming and memory bounds (better
covered by benchmarks and fuzzing) and the Rust IPC concat
(better covered by `cargo fuzz` or Kani if it grows).

## Keeping models honest

A model helps only while it matches the code. Suggested practice:

- Keep the model next to this README and name the Python functions each
  action mirrors, as `ArtifactLifecycle.tla` does, so a review of
  `artifact_store.py` knows to look here.
- Run `Patched.cfg` (renamed to the default config once the fix lands) in
  CI. It takes about 2 minutes. A smaller bound (`MaxVer = 2`) runs in
  about 2 seconds (27k states) and still catches findings 1 and 2.
- Add each TLC counterexample to `tests/` as a regression test (as
  `replay_counterexamples.py` does), so the code is checked even when
  the model isn't.
