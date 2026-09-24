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

As a proof of concept, three TLA+ models of 160–260 lines each found
eight real bugs, each in about a second of model checking:

- The **artifact lifecycle** model found findings 1–2.
- The **build lease protocol** model found findings 5–7.
- The **admission limiter** model found findings 8–10.

A review of the assumptions behind the pruning and cache-key invariants
found two more (findings 3–4). All ten reproduce against the real code
(`uv run pytest formal/`; finding 8 needs CPython 3.12, see below).

## What's here

| File | Purpose |
| --- | --- |
| `tla/ArtifactLifecycle.tla` | Model of `artifact_versions`: create, write blob, fail, `finalize_artifact`, `force_finalize_canonical`, `garbage_collect` |
| `tla/Artifact_GCRebuild.cfg` | One id with GC on. Finds finding 1 |
| `tla/Artifact_CrossIdDedup.cfg` | Two ids sharing a provenance, GC off. Finds finding 2 |
| `tla/Artifact_Patched.cfg` | Both ids, GC on, with the proposed fix. **All invariants hold** (5,997,420 distinct states, exhaustive, about 2 min) |
| `tla/BuildLease.tla` | Model of one build and its lease: `BuildRunner` claim / reclaim / heartbeat / publish / finalize / complete / fail, and the v2 pull routes (manifest, upload, finalize) |
| `tla/Build_RunnerPath.cfg` | Two runners, lease can expire mid-build. Finds findings 5 and 6 |
| `tla/Build_PullPath.cfg` | Two executors fetch the same build's manifest. Finds finding 7 |
| `tla/Build_Patched.cfg` | Runners and executors with the proposed fix. **All invariants hold** (exhaustive) |
| `test_artifact_counterexamples.py` | Findings 1–4 against the real `ArtifactStore` / `ReadPlanner` / `CacheKey` |
| `test_build_runner_counterexamples.py` | Findings 5–6 against two real `BuildRunner`s (only the executor HTTP call is stubbed) |
| `test_build_pull_counterexamples.py` | Finding 7 through the real HTTP routes (`TestClient`) |
| `tla/Admission.tla` | Model of one tenant's `ResizableLimiter` (acquire with deadline, release, cancel while queued) and the `TenantRegistry` LRU that owns it |
| `tla/Admission_Py312.cfg` | CPython 3.12 `asyncio.Condition` semantics. Finds finding 8 |
| `tla/Admission_Eviction.cfg` | LRU eviction of a limiter in use. Finds findings 9 and 10 |
| `tla/Admission_Patched.cfg` | Both fixes. **All invariants hold** (exhaustive) |
| `test_admission_counterexamples.py` | Findings 8–10 against the real `ResizableLimiter` / `TenantRegistry` |

Each `test_*` file **passes while its bug exists**, because it asserts
the violating outcome. When a fix lands, invert its assertion and move
it into `tests/` as a regression test. `formal/` sits outside
`testpaths`, so the normal `uv run pytest` does not run these.

```bash
curl -sSLO https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
cd formal/tla
TLC="java -cp ../../tla2tools.jar tlc2.TLC -deadlock -workers auto"
$TLC -config Artifact_GCRebuild.cfg    ArtifactLifecycle.tla   # violation
$TLC -config Artifact_CrossIdDedup.cfg ArtifactLifecycle.tla   # violation
$TLC -config Artifact_Patched.cfg      ArtifactLifecycle.tla   # passes
$TLC -config Build_RunnerPath.cfg      BuildLease.tla          # violation
$TLC -config Build_PullPath.cfg        BuildLease.tla          # violation
$TLC -config Build_Patched.cfg         BuildLease.tla          # passes
$TLC -config Admission_Py312.cfg       Admission.tla           # violation
$TLC -config Admission_Eviction.cfg    Admission.tla           # violation
$TLC -config Admission_Patched.cfg     Admission.tla           # passes
cd ../.. && uv run pytest formal/ -v
# finding 8 only reproduces on CPython 3.12 (skipped on 3.13+):
uv run --no-project --python 3.12 --with pytest pytest formal/test_admission_counterexamples.py -k wakeup
```

`-deadlock` disables deadlock checking, because a bounded model that
has used every version or lease epoch has no enabled action. That is
expected, not a bug. TLC stops at the first violated invariant. To see a
particular one, list only that invariant in the config's `INVARIANTS` line.

## Model 1: artifact lifecycle

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

### Proposed fix for 1 and 2 (verified in the model)

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

## Assumption checks: data plane

These two are not model-checking results. They came from asking what
the pure functions behind invariants 1 and 2 assume, then testing each
assumption.

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

## Model 2: build lease protocol

`BuildLease.tla` models one transform build and the lease that is
supposed to guarantee a single writer. It covers two ways the build can
be executed:

- **Runners**, in-process `BuildRunner._execute_build`: claim or reclaim
  a lease, renew it from the heartbeat loop, `publish_blob_from_path`,
  `finalize_artifact`, `complete_build(lease_owner=me)`, and on error
  `fail_build` + `fail_artifact`.
- **Executors**, the v2 pull routes: `GET …/manifest` claims or renews
  the lease as `external:manifest` and mints the signed URLs plus a lease
  token; `POST /v1/artifacts/upload` writes the blob; `POST …/finalize`
  checks the lease token, runs `finalize_and_set_name`, then
  `complete_build`.

A lease token is modelled as a lease *epoch*: every claim, reclaim and
manifest re-fetch changes `(lease_owner, lease_expires_at)`. A runner's
lease may expire at any moment, which stands in for a GC pause, a blocked
event loop, or a database outage longer than the lease. Blob bytes are
modelled as "which attempt wrote them".

Invariants:

- `ReadyBuildHasArtifact`: a completed build has a ready artifact. **Holds.**
- `ReadyBytesStable`: a ready artifact's bytes never change, so they always
  match the digest recorded at finalize (what `verify_artifacts` checks). **Violated.**
- `NoStaleBytesPublished`: the bytes that get published come from the
  attempt that passed the fence. **Violated.**
- `OnlyLeaseHolderFails`: only the attempt holding the lease can fail the
  build or its artifact. **Violated.**

Findings 5–7 matter most when two attempts produce different bytes: a
nondeterministic transform, `now()`, sampling, a moving input, or a
different executor version. For a byte-identical transform, findings 5
and 7 are harmless, but finding 6 is not.

### 5. A runner that lost its lease still publishes, and the winner rewrites a ready artifact

`complete_build` is the only step fenced on the lease, and
`publish_blob_from_path` and `finalize_artifact` both run before it. So
a runner whose lease was reclaimed still writes its bytes and makes them
the **ready** artifact, with their digest recorded. Only afterwards is it
told it lost ("discarding the result", which by then is too late). The
rightful owner then writes its own bytes to the same `(artifact_id,
version)` key. `finalize_artifact` treats the artifact as already ready
and does nothing. The result is a ready artifact whose bytes changed
under readers and no longer match the recorded digest
(`verify_artifacts` reports `digest_mismatch`). TLC trace (6 steps):

```
r1 claim → lease expires → r1 publish → r1 finalize (ready, digest=r1)
→ r2 reclaim → r2 publish            ⇒ ready bytes are now r2's
```

The replay runs two real `BuildRunner`s and stubs only the executor
HTTP call.

### 6. A runner that lost its lease can fail the build that replaced it

`fail_build` and `fail_artifact` don't check the lease. If the stale
runner's executor times out after another runner has reclaimed the
build, the stale runner marks the build and its artifact `failed`. The
new owner then sees `failed` and stops, so a build that would have
succeeded is reported as failed, and the error is the stale attempt's.
`BuildRunner.stop()` has the same pattern: on shutdown it fails every
in-flight build without checking the lease. The replay does not cover
that path. TLC trace (5 steps): `r1 claim → expire → r2 reclaim → r1 error → r1 fail_build`.

### 7. A retired manifest's upload URL still writes the blob

Re-fetching a manifest renews the lease, which retires the previous
*finalize* URL: its lease token no longer matches, and
`test_refetching_a_manifest_retires_the_previous_capability` checks
this. The *upload* URL carries no lease token, so it stays valid.
`verify_upload_signature` checks only the build id, the size and the
expiry. As a result, the "one live capability set at a time … makes two
writers impossible by construction" claim in that test does not hold for
uploads. TLC trace (5 steps):

```
e1 manifest → e2 manifest (renews; e1's finalize URL retired)
→ e2 upload → e1 upload (accepted) → e2 finalize   ⇒ e1's bytes published under e2's claim
```

The model also finds a narrower variant in which an upload lands between
`finalize_and_set_name` and `complete_build` and rewrites a ready
artifact (`ReadyBytesStable`). With `artifact_presigned_urls` on, uploads
go straight to the object store and never reach Strata. So a route-side
lease check can't close this, and the fix has to change *where* the
bytes land.

### Proposed fix for 5–7 (verified in the model)

`Patched = TRUE` in `BuildLease.tla`:

1. **Each attempt writes its own blob key**, scoped by lease epoch or
   token (for example a per-attempt staging key), instead of the shared
   `(artifact_id, version)` key. A stale attempt, including a presigned
   upload, can then only write bytes nobody will read.
2. **One fenced step promotes the attempt**: in a single transaction,
   check that the lease is still held, record which attempt's key the
   artifact reads from, mark the artifact ready and complete the build.
   Otherwise discard. This replaces the unfenced `finalize_artifact`
   followed by the fenced `complete_build`.
3. **`fail_build` / `fail_artifact` take the lease owner**, the same
   way `complete_build` already does.

With all three, TLC finds no violation of any invariant, for two runners
and two executors (exhaustive). The state space is small because only
one kind of claimant can hold a build at a time.

Change 3 is small and independent and fixes finding 6 completely. The
cheap alternative for 1 and 2 is to re-check the lease right before each
write. That narrows the window but can't close it, because the lease can
expire between the check and the write (the model has no variant for
this; it follows from reading the code). It also does nothing for
presigned uploads.

## Model 3: admission limiter

`Admission.tla` models one tenant's stream admission for one tier. The
limiter is `ResizableLimiter` (`adaptive_concurrency.py`), an
`asyncio.Condition` guarding `(capacity, in_use)`:

- `acquire(timeout)` waits while the limiter is full, and its timeout
  handler re-checks `in_use`.
- `release()` decrements `in_use` and calls `notify(1)`.

A queued request can also be cancelled outright, for example by a
client disconnect or by shutdown (the #238 path in `QoSAdmission.admit`).
The limiter belongs to the tenant's quotas in `TenantRegistry`, which
LRU-evicts them once more than `MAX_TRACKED_TENANTS` (1000) tenants are
tracked. The model abstracts that pressure into an `Evict` action that
may fire at any time.

The property the earlier ranking listed, that interactive requests are
never starved by bulk ones, **holds by construction**: the two tiers
use separate limiters and never wait on each other, so it needed no
model. The model looked inside a tier instead.

Invariants:

- `NoLostWakeup`: whenever a slot is free and a request is queued, some
  queued request has been notified. **Violated on CPython 3.12.**
- `WithinQuota`: a tenant never holds more slots than its quota. **Violated.**
- `DrainSeesAll`: `aggregate_limiter_usage()`, which graceful shutdown
  uses to wait for streams, counts every live stream. **Violated.**

### 8. On Python 3.12, a cancelled waiter swallows the wakeup

When a waiter that `notify(1)` picked is cancelled before it runs,
CPython 3.13+ `Condition.wait()` notifies another waiter, but 3.12 does
not. `ResizableLimiter` relies on the condition alone, so on 3.12 the
wakeup is lost. The other queued requests sleep until their deadline
(`interactive_queue_timeout` 10 s, `bulk_queue_timeout` 30 s) while the
slot is free. A request arriving in that time takes the free slot
straight away, ahead of them. If that leaves the limiter full when a
queued request's deadline passes, the re-check fails and it gets a 429.
Strata supports 3.12 (`requires-python >= 3.12`) and CI tests it. The
Docker image uses 3.13, so this affects installs from PyPI or source on 3.12.
TLC trace (5 steps):

```
r1 admit (holds) → r2 admit (queued) → r3 admit (queued)
→ r1 release (notifies r2) → r2 cancelled   ⇒ slot free, r3 not notified
```

The replay runs the real `ResizableLimiter` on CPython 3.12. C waits the
full 0.5 s deadline with `in_use == 0`, while on 3.13.7 the same scenario
returns in 0.0 s.

Suggested fix: in `ResizableLimiter.acquire`, catch `BaseException`
around the wait and call `self._cv.notify(1)` before re-raising, which
is the same re-notify 3.13 added. `Admission_Patched` models exactly
that behaviour.

### 9. An evicted limiter lets a tenant exceed its quota

`get_or_create_quotas` evicts the least recently used tenant regardless
of whether its limiters have slots in use. The tenant's next request
builds a fresh limiter at full capacity, while the evicted one is still
held by live streams, so the tenant runs up to twice its quota, and more
after each further eviction. This needs more than 1000 tenants active
at once, so it only affects multi-tenant service deployments. TLC trace
(3 steps): `r1 admit → evict → r2 admit ⇒ 2 slots held on a quota of 1`.

`QoSAdmission._get_client_semaphore` LRU-evicts per-client semaphores
(10,000 entries) the same way, which can let one client exceed
`per_client_*`. I found that by reading the code; it is not modelled
or replayed.

### 10. Graceful shutdown doesn't count streams on an evicted limiter

`_graceful_shutdown` waits while `aggregate_limiter_usage()` reports
in-flight scans, then cancels the remaining stream tasks. That count
only sums the limiters the registry still tracks, so a stream on an
evicted limiter is invisible to it. Shutdown can then report "drained"
and cancel that stream mid-response, which is exactly what #185 set
out to prevent. TLC trace (2 steps): `r1 admit → evict`.

### Proposed fix for 8–10 (verified in the model)

`Admission_Patched` (4 requests, capacity 2, 3 generations) finds no
violation with both changes:

1. Re-notify on cancellation in `ResizableLimiter.acquire` (above).
2. Evict only quotas whose limiters are idle: `in_use == 0` and no
   waiters. The registry can then briefly exceed 1000 entries, which is
   a much smaller risk than an unbounded quota.

## Where else formal methods would pay off

Ranked by (likely bugs × consequence) ÷ modelling effort. The build
protocol and the QoS limiters were on this list and are now done
(models 2 and 3):

| Component | Technique | Why |
| --- | --- | --- |
| **Notebook cascade + staleness** (`cascade.py`, `dag.py`, `session.compute_staleness`) | TLA+ or Hypothesis stateful testing | Invariants such as "after a cascade the target's inputs are READY and match current provenance", and "a source edit marks exactly the downstream closure stale". The concurrency comes from WS source flushes arriving during execution. |
| **Pruning soundness** (`filters.py`, manifest pruning, `_convert_stats`) | Z3 / CrossHair over the comparison logic, plus Hypothesis round-trips through real Parquet files | Finding 3 shows the value is in encoding the stats assumptions (NaN, null, type coercion) explicitly. |
| **Hash/key encodings** (`CacheKey.to_hex`, `derive_subkey`, provenance, `transform_spec.to_json`) | Property tests for injectivity | Finding 4. Cheap, and a collision means serving wrong data. |
| **ACL deny-first** (`auth.AclEvaluator`) | Z3 or exhaustive enumeration | Small and pure. Prove "a matching deny cannot be overridden by any allow set" and pattern/tenant corner cases. Low risk today, but a cheap guard against regressions. |

Not worth it: the data plane's streaming and memory bounds (better
covered by benchmarks and fuzzing) and the Rust IPC concat
(better covered by `cargo fuzz` or Kani if it grows).

## Keeping models honest

A model helps only while it matches the code. Suggested practice:

- Keep each model next to this README and name the Python functions each
  action mirrors, as both specs do, so a review of `artifact_store.py`
  or `transforms/runner.py` knows to look here.
- Run the `*_Patched.cfg` configs in CI once the fixes land.
  `Artifact_Patched` takes about 2 minutes. With `MaxVer = 2` it runs in
  about 2 seconds (27k states) and still catches findings 1 and 2.
  `Build_Patched` and `Admission_Patched` take about a second each.
- CI runs on 3.12, 3.13 and 3.14. Run the finding 8 test on the 3.12 job;
  it skips itself on 3.13+.
- Move each counterexample test into `tests/` with its assertion
  inverted, so the code is checked even when the model isn't.
