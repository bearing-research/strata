# Formal verification of Strata: findings

Findings 1–12 are the record of the first investigation (2026-09), which
asked whether formal methods could usefully model and verify Strata's core
components; later findings follow at the end.
[`SYSTEMS.md`](SYSTEMS.md) is the specification these findings fed into, and
[`README.md`](README.md) says how to run everything.

**Answer:** yes, as long as the tools are lightweight and aimed at specific
components. Proving the Python implementation correct (Coq, Lean or Dafny
against the real code) would cost far more than it returns. Two
cheaper approaches do pay off:

- **Model checking** (TLA+ with TLC) for the concurrent state machines.
- **Checking the assumptions behind the pure functions** that Strata's
  invariants depend on (pruning soundness, cache-key injectivity), using
  SMT solvers or property-based tests.

As a proof of concept, four TLA+ models of about 140–260 lines each found
nine real bugs, each in about a second of model checking:

- The **artifact lifecycle** model found findings 1–2.
- The **build lease protocol** model found findings 5–7.
- The **admission limiter** model found findings 8–10.
- The **notebook staleness** model found finding 11.

A review of the assumptions behind the pruning, cache-key and access
control invariants found three more (findings 3, 4 and 12). All twelve
reproduced against the real code
(`uv run pytest formal/`; finding 8 needed CPython 3.12).
Fixed findings are marked **Fixed** below; their replays are now
regression tests under `tests/`.

The artifact lifecycle and build lease models (findings 1, 2 and 5–7) moved
to the `bearing-research/vestigo` repository with the ledger (ADR 002),
under its `formal/tla/`, and have changed there since. Models 1 and 2 below
describe them as they were when the findings were made.

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
(`artifact/store.py`, `garbage_collect`). A rebuild creates a
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

**Fixed.** `garbage_collect` also spares each id's current value, the
version `get_latest_version` resolves, and keeps the `MAX(version)` rule.
It first shipped as "newest `ready`", matching `get_latest_version` at
the time; finding 2's fix widened both to `ready`/`superseded` together,
as in change 2 below. Regression tests:
`tests/test_artifact_store.py::TestGcSparesCurrentValues::test_gc_spares_the_current_value_while_a_rebuild_is_in_flight`,
and the same scenario on Postgres in `tests/test_artifact_store_postgres.py`.

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

**Fixed.** `get_latest_version` (and its prefix twin
`list_latest_by_id_prefix`) returns the newest `ready` or `superseded`
version, change 1 below, and GC spares that version, change 2. The other
callers were audited first: outside the notebook they need only a row
with bytes (the tenant check on append, CLI and `latest` URI
resolution); inside it they compare provenance or read the blob, and
several (loop seeds, iteration lookups, the team store) already accepted
`superseded`, which `ready`-only never returned. Regression tests:
`tests/test_artifact_store.py::TestTwoIdsOneComputation`.

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

**Fixed.** `matches_stats` never prunes `!=` when the stats are floats.
Regression tests: `tests/test_filters.py::TestPruningKeepsNaNRows` and
`TestFilterMatching::test_ne_all_same_float_value`. The pruning property
test no longer excludes this shape.

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

**Fixed** as suggested: the fingerprint hashes `json.dumps(columns)`.
Regression test:
`tests/test_smoke.py::TestCacheKey::test_a_comma_in_a_column_name_is_not_a_separator`.

## Assumption checks: access control

`AclEvaluator.authorize` is deny-first by construction: deny rules, then
allow rules, then the default. There is nothing to model in that loop.
Invariant 7 (*explicit denies cannot be bypassed by allows*) also rests
on an assumption outside it: that a table has one ACL name. `TableRef`
names a table by the *form of the URI it was requested under*: the store
prefix of a warehouse URI, `file:` for anything else, or a named
catalog's name.

### 12. A deny rule can be sidestepped by addressing the table another way

In a service deployment with a SQL catalog (`catalog_properties["uri"]`,
Postgres in production), every warehouse URI builds `SqlCatalog("strata")`
over that one database, whatever path or scheme comes before `#`. A bare
`namespace.table` reads the default catalog. So one table in S3 can be
requested as:

| URI | ACL name |
| --- | --- |
| `s3://bucket/wh#finance.ledger` | `s3:finance.ledger` |
| `/not/a/real/path#finance.ledger` | `file:finance.ledger` |
| `finance.ledger` | `file:finance.ledger` |

A deny on `s3:finance.*` refuses only the first. The replay runs the
real `authorize_table_access` gate: the S3 form gets a 403, the other two
pass, and `PyIcebergCatalog.load_table` returns the same metadata
location for all three. Only default-allow configurations are exposed;
with `default = "deny"`, the aliases fall back to the default and are
refused.

This is partly documented. `docs/reference/configuration.md` says a
table reachable under two prefixes needs both patterns, and its example
denies `file:`, `s3:` and `lake:` together. But it also says `file:` is
"for a local warehouse". An operator whose data is all in S3 has no
reason to write a `file:` rule, and here `file:` names S3 data behind a
path that doesn't exist.

Suggested fix: name a table by the catalog that actually serves it, not
by the address form. When `catalog_properties` is set, every warehouse
URI and the bare form resolve to the same catalog, so they should share
one ACL name. Until then, the docs should recommend `*:finance.*` for
deny rules. fnmatch lets `*` match any prefix, which covers every alias.

**Fixed**, with the names kept. Renaming every alias to one name would
have turned a working `s3:finance.*` deny into a rule that matches nothing,
so instead a deny is checked against every name the table answers to.
`shared_catalog_stores` (in `iceberg/catalog.py`) lists them: with
`catalog_properties["uri"]` set, every warehouse URI's `s3:`, `gs:`, `az:`
and `file:` names, and the bare form's too when `catalog_name` is `strata`.
Allow rules still match only the name requested, so an alias never grants
anything. A named catalog pointing at the same data is not detected; the
docs keep recommending `*:` deny patterns for that. Regression test:
`tests/test_auth.py::TestADenyOnOneAddressCoversTheTable`.

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

**Fixed.** `fail_build` takes `lease_owner` like `complete_build`, and
both the runner's failure path and `stop()` pass it. The artifact is
failed only when the fenced `fail_build` succeeds, so a runner that lost
its lease leaves the build and artifact to the owner. Regression tests:
`tests/test_build_runner.py::TestALeaseDecidesWhoMayFail`.

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

**Fixed (5 and 7)** as proposed. Each attempt writes under its own blob
id, `{artifact_id}~{attempt}`, and `artifact_versions.blob_attempt`
records the one promoted; `ArtifactStore` resolves every read, delete and
verification through it, so the blob backends are unchanged. A runner
takes a fresh attempt per execution. An executor's attempt is a digest of
its manifest's lease token: signed into the upload URL and the presigned
POST key, and the only upload finalize reads. `finalize_artifact` and
`finalize_and_set_name` take a fence, `BuildStore.complete_within`, that
completes the build under its lease (owner, and for executors the
deadline) inside the same transaction, so publishing and completing
commit together or not at all. Paths with no lease (the notebook's
in-process signed path, legacy rows) keep the shared key and the old
completion. An attempt that loses, or deduplicates to an existing
artifact, deletes its bytes. Not cleaned: a stale executor's upload,
which Strata never sees for a presigned URL, and an attempt whose runner
fails after publishing. Regression tests:
`tests/test_build_runner.py::TestOnlyTheLeaseHolderPublishes` and
`tests/test_pull_model.py::TestARetiredManifestCannotWriteTheOutput`.

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

**Fixed.** `ResizableLimiter.acquire` re-notifies on any exception out of
the wait, as proposed. Regression test:
`tests/test_adaptive_concurrency.py::TestResizableLimiter::test_a_cancelled_waiter_passes_its_wakeup_on`,
which fails without the fix on the CI 3.12 job and passes either way on
3.13+.

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

**Fixed (9 and 10).** `get_or_create_quotas` evicts, oldest first, only
tenants whose limiters are idle: no slot held and no `acquire()` pending,
which `ResizableLimiter` now counts. The code needed one change the model
could not show: `QoSAdmission.admit` looked up the tenant's limiters and
then awaited the per-client semaphore before acquiring, and a tenant
idle at the lookup could be evicted during that wait. `admit` now looks
the limiters up immediately before acquiring, with no await between, so
the acquire it counts is on the pair the registry tracks. The model
treats admit as one step, which is why it did not find this. Regression
tests: `tests/test_multitenancy.py::TestEvictionSparesBusyTenants` and
`tests/test_qos_admission.py::test_admit_acquires_the_limiter_the_registry_counts`.

## Model 4: notebook staleness

`Staleness.tla` models the status of a chain of cells `a → b → c`, with
one run at a time. It follows `notebook/ws.py` and `session.py`:

- **Edit** is `cell_source_update`. The cell that is executing is
  refused (`running_cell` / `requested_cell`), and any other cell is
  accepted. `compute_staleness_async()` then writes the walk's verdict
  into every cell's status.
- **Start**: a run reads its upstream's current result when it starts.
- **Finish**: the run stores its result, then
  `_refresh_and_broadcast_changed_staleness` recomputes the walk and,
  through `preserve_ready_cell_id`, marks the cell READY.

The walk is modelled by what it decides: a cell is ready when its latest
result was computed from its current source and its upstream's latest
result, and that upstream is ready too.

Invariants:

- `ReadyMeansCurrent`: a cell reported READY holds a result computed from
  its current source and its upstream's current result. **Violated.**
- `RunningShown`: the executing cell is reported as running. **Violated.**

### 11. Editing an upstream mid-run leaves the downstream reported READY

The source-update guard locks only the executing cell, so while `b`
runs, its upstream `a` can be edited. Two things then go wrong:

1. The flush's walk writes a status for every cell, the running one
   included, so `b` stops being reported as running while it still runs.
2. When `b` finishes, the walk correctly finds it stale because its
   upstream is. `preserve_ready_cell_id` then sets it READY
   unconditionally, and that override is broadcast. `a` reads stale and
   `b` reads ready, although `b` was built from the old `a`.

TLC trace (5 steps):

```
a start → a finish → b start → edit a → b finish
⇒ a stale, b READY (built from the old a)
```

Model checking alone overstated the impact here. The replay showed that
the harm stops at the reported status. When `c` runs next, no cascade is
offered, but the executor re-checks provenance on its own, silently
rebuilds `a` and `b`, and computes `c` from the new source. The replay
asserts both halves.

What reads the wrong status: the UI, `GET /cells`, agents and MCP tools,
the impact preview (`impact.py` lists only READY downstream cells), and
the cascade planner's decision whether to *offer* a cascade. It
self-corrects on the next recompute. Severity: low.

`preserve_ready_cell_id` exists so that leaf cells whose output isn't
cached still show READY after a successful run. The fix keeps that and
stops it from overriding a stale upstream.

### Proposed fix for 11 (verified in the model)

`Patched = TRUE`, which passes exhaustively with 2 edits and 3 runs per
cell:

1. `_apply_staleness_map` leaves the executing cell's status alone.
2. After a run, preserve READY only when the walk did not find a stale
   upstream (no `UPSTREAM` reason and every upstream READY). Otherwise
   keep the walk's verdict.

The model has no uncached leaf cells, so in the model change 2 simply
keeps the walk's verdict. The leaf case needs a test against the real
walk.

**Fixed.** Change 1 applies where the model's Edit is: the source flush
and the variant handlers pass the running cell to the recompute, which
leaves its status alone and keeps it out of the broadcast map. It is
not applied to every recompute, because a run's own finish relies on
the walk to move its cell off running. Change 2: after a run, a cell
whose upstream the walk finds not ready becomes stale with reason
`upstream` instead of ready. It is set explicitly rather than taken from
the walk, since a cached cell records no `last_provenance_hash` and the
walk calls it idle, which would read as never run. An uncached leaf whose
upstreams are ready is still kept ready. Regression test:
`tests/notebook/test_e2e_staleness.py::TestAnEditDuringARun`. Now that b
reads stale, running `c` offers the cascade that the replay showed
missing.

## Property tests

Where a hand-written model would miss too many special cases, the next
step is a property test: Hypothesis generates inputs, the real code runs,
and an oracle that doesn't share Strata's logic checks the result. Both
tests read their example count from the environment
(`PRUNING_EXAMPLES`, default 3000; `STALENESS_EXAMPLES`, default 15).

**Pruning soundness** (`test_pruning_properties.py`). Random columns of
float64, int64, string, timestamp and decimal values, with nulls, are
written through pyarrow's real Parquet writer with 1–4 rows per group.
Each row group that `_should_prune_row_group` prunes, given the file's
real statistics, must contain no row that an exact Python comparison
would keep. The generators mix a small pool of boundary values (±0.0,
NaN, ±inf, 2⁵³ and 2⁵³+1, int64 limits, long shared string prefixes,
one-microsecond timestamp steps) with the full range. That made the
difference: with plain random values, Hypothesis did not produce the
finding 3 shape even in 3,000 examples, and with the pool it
rediscovered it on its own after about 30,000.

- Result: with finding 3 excluded as known, **60,000 examples pass**
  (4 min). Since the fix, they pass with no exclusion. No other pruning violation turned up for these types and
  operators, including an int64 column filtered with a float (compared
  exactly, so 2⁵³+1 is not confused with 2⁵³). Not covered: mixing
  naive and tz-aware timestamps, and types beyond these five.

**Notebook staleness** (`test_staleness_properties.py`). A diamond
`a → (b, c) → d` of integer cells, plus a sink that reads `d`. Each
example is a random sequence of up to six source edits and runs, sent
through the real notebook WebSocket with real cell execution. The oracle
evaluates the current sources from scratch. After each step, a run's
stored value must equal that evaluation, and every cell reported READY
must hold the value the evaluation gives.

- Result: **40 sequences pass** (3 min). Sequential edits and runs keep
  statuses and values consistent, including cascades and re-runs of
  cells whose upstream changed. Finding 11 needs an edit to land
  *during* a run, which this sequential test doesn't generate. Covering
  that needs concurrent steps, which the WebSocket harness can't easily
  drive.

**Checked and found safe** (by reading the code; no test needed):

- `derive_subkey` joins labels with `:`, but no label can be ambiguous.
  Variable names can't contain `:` or `=`, and names starting with `_`
  are never stored (`analyzer.py`, `harness.py`), so a variable can't
  collide with the `__display__N` or `__console__` keys. The other labels
  have their own prefixes (`variant=`, `content=`, `iter=`).
- `CacheKey.to_hex` joins fields with `|`. A collision would need a
  table name containing `|` and a data file path starting with digits
  followed by `|`. Real file paths start with a scheme or `/`, so this
  can't happen in practice. The projection fingerprint (finding 4) is
  the real hole in the key.

## Findings since the 2026-11 review

### 13. A lapsed lease loses a run that nothing else wanted

**Model:** `tla/LocalRun.tla` (SYSTEMS 10, LR-6). A local run's lease is wall-clock,
60 s, renewed every 20 s by a task on the event loop, and a lapsed lease stays
lapsed. A lease lapses with nothing wrong whenever the process cannot renew for a
minute: a laptop asleep through the cell (the renewal timer is monotonic and stops,
the wall clock does not), or the event loop blocked, which it is while a run's
outputs are stored. Every later release of the run is then refused, though no
other process took the build. TLC trace (4 states): `lapse → work → release`
fenced.

The replay jumped the ledger's clock two minutes mid-run. A cell that checkpoints
was fenced and killed at its next checkpoint; a cell that does not ran to the end
and then lost its outputs ("executed successfully but failed to store output
artifacts"). Personal mode on a laptop is the main deployment, and checkpoints exist
for exactly the long runs a sleep interrupts. Severity: high (a finished run's
result discarded).

**Fixed.** Before each release, and on each renewal, the run holds its lease
(`LocalRuns._hold`): it renews, and if the lease lapsed and nobody took the build,
it goes on in a new attempt of the same build. A process that did take it keeps
it, since `open_attempt` refuses while another attempt's lease is live. The model
found a hole in the first version of this fix: with the renewal task and a release
holding at once, one opened a new attempt and, before it was recorded on the run,
the other's renewal succeeded, because `renew_lease` checks the process and not
the attempt; the release then went out under the fenced attempt (`LocalRun_Race`,
one lapse, 7 states). Holds are serialized per run. `LocalRun_Patched` (a rival
process that may resume the build, three lapses) passes. Regression tests:
`tests/notebook/test_local_runs.py::TestALapsedLeaseWithNoRival`, also on Postgres.

### 14. A Run All cell edited while it waits is shown ready on its old source

**Model:** `tla/Staleness.tla`, configuration `Staleness_BatchEdit` (SYSTEMS 8, NB-1).
Run All freezes every batched cell's source when it plans the batch, and only the
first cell is requested, so the WebSocket accepts an edit of any later cell while
it waits. The cell then runs its frozen source, and `preserve_ready_cell_id` marks
it ready, checking its upstreams but not its own key. TLC trace (7 states): plan,
`a` runs, edit `b`, `b` runs its frozen source, `b` ready.

The replay showed the walk does find the cell stale: the next batched cell's
recompute corrects it, so the ready frame is transient, except for the batch's last
cell, which stays ready on a source it never ran. As with finding 11 the harm is the
reported status; the executor re-checks provenance before using it. Severity: low.

**Fixed.** After a run, a cell the walk finds stale keeps the walk's verdict.
Regression test:
`tests/notebook/test_batch_running_status.py::test_a_batched_cell_edited_while_it_waits_is_not_shown_ready`.

### 15. REST and MCP edits change a running cell

**Model:** `Staleness_ApiEdit` (NB-2). The WebSocket refuses an edit of the running
or requested cell and leaves the running cell's status alone in the recompute that
follows any edit (finding 11's fix). `PUT /cells/{id}` and the MCP `edit_cell`
tool did neither. The replay ran `b`, edited it over REST (accepted), and found `b`
taken off running mid-run and then ready on a source it never ran. An MCP edit of
another cell reloads the session, whose fresh cells parse idle, so the running cell
read idle. (The reload also replaces the cell objects mid-run; a probe showed the
run's outputs still land on the live ones, so no key names the wrong upstream.)
Agents read status through these paths. Severity: low to medium.

**Fixed.** REST answers `409 cell_busy` and MCP raises, as the WebSocket refuses;
REST passes the running cell to its recompute, and a reload keeps a running cell
running. Regression tests: `tests/notebook/test_e2e_staleness.py::TestARestEditDuringARun`,
`tests/notebook/test_mcp_server.py::test_an_agent_edit_of_the_running_cell_is_refused`.

### 16. A file registered again at the same path is served its old rows

**Property:** `test_cache_soundness_properties.py` (SYSTEMS 3, CK-1). Shrunk case: a
Hive-layout file registered at a fixed path, rewritten there at the same size with other
rows, and registered again in place of the old entry. The full scan returned the old
rows. The row-group key named the file by table, uuid, path and size, all unchanged;
the existing test of a rewritten path passed only because it recreated the table, which
changes the uuid, and only scanned unfiltered.

The same `(path, size)` keyed three more caches, each with its own symptom once the
row-group key was fixed: the planner's footers, whose old statistics pruned the new rows
from a filtered scan; its column layouts, which filled a partition column the file omits
with the value it was first registered under (lead CK-1 (a)); and the fetcher's footers,
which read the new bytes by the old file's schema and failed the scan after a column
rename (`KeyError`). Re-registering a rewritten file is outside Iceberg's rule that files
never change, but `add_files` over Hive-layout data does it, and Strata already meant to
handle a rewritten path. Severity: high (wrong rows, silently).

**Fixed.** Every cache of a data file also names the snapshot that added it (the
manifest entry's `snapshot_id`): registering a file again changes it; an append, a delete
or a manifest rewrite keeps it, so those still hit. Regression tests:
`tests/test_row_group_cache_key.py::TestAPathRegisteredAgainInTheSameTable`,
`tests/test_metadata_cache.py::TestMetadataStore::test_remote_parquet_meta_of_another_registration_is_stale`.

### 17. A stream expires under a GET waiting for a slot, which then answers 500

**Model:** `Streams.tla` (SYSTEMS 2, STR-1), configuration `Streams_Retry`. A GET
cancels the stream's cleanup and queues for admission; a retry of the same stream meets
the per-client cap the first holds and gets a 429, whose exit re-arms the cleanup. If
the TTL passes before the first is admitted, expiry drops the stream and, no build
having started, fails its placeholder artifact. The first GET is then admitted, starts
the build, whose finalize is refused (`not in building state`), and answers 500
`scan_build_failed`. Replayed against the real handler with a 0.3 s TTL. It needs a TTL
shorter than the queue wait (defaults: 300 s against 10 s and 30 s queue timeouts), which
the config allows. Severity: low (a visible failure, under an unusual config).

**Fixed.** `StreamState.waiting` counts the GETs between cancelling the cleanup and
settling admission, and expiry leaves a stream with one alone; each re-arms the cleanup
as it leaves the queue, and one admitted has started the build, so the stream is still
expired once nobody waits (`NeverStranded`). Regression test:
`tests/test_stream_registry.py::test_a_stream_is_not_expired_under_a_get_waiting_for_a_slot`.

### 18. A caller holding a per-client slot gets a fresh cap after 10,000 others

**Replay** of lead ADM-5 (SYSTEMS 1). The per-client semaphores are an LRU of 10,000
callers per tier, and eviction took the oldest whether or not it was held. A caller
streaming a long bulk scan while 10,000 others arrived lost its semaphore; its next
request made a new one at full count and was admitted past its `per_client_*` cap.
Severity: low (fairness only; the tenant limiter still bounds the tenant).

**Fixed.** Eviction skips semaphores with a slot held; past the bound, the held ones are
live streams, so the table stays bounded by them. Regression test:
`tests/test_qos_admission.py::test_a_held_client_cap_outlives_other_callers`.

### 19. A cascade marks a step stale that reads nothing that failed

**Property:** `test_run_all_properties.py` (SYSTEMS 9, BAT-1). Shrunk case: `c1` reads a
variable defined only below it (unresolved, so it fails), `c2` fails on its own, and a
sink reads both. Run cell by cell, the sink's cascade ran `c1`, which failed, then skipped
`c2` and marked it stale: once a step fails the cascade skips the rest and marked every
one stale, overwriting `c2`'s own error though nothing it reads had failed. Run All left
`c2` in error. Severity: low (a wrong status).

**Fixed.** The cascade still stops at a failure, since its target cannot run, but marks
blocked (stale, reason upstream, as Run All does) only the steps that read the failure;
the others keep their status. Regression test:
`tests/notebook/test_ws.py::test_a_cascade_blocks_only_what_reads_the_failure`.

### 20. A cell blocked behind a failure is shown idle after the next run

**Property:** `test_run_all_properties.py` (SYSTEMS 9, BAT-1). Run All and a cascade mark
a cell that cannot run because what it reads failed as stale, reason upstream, to say why
it did not run. The staleness walk after any later successful run recomputed every cell
and called one that had never run idle ("no result to invalidate"), so the status a
blocked cell ended with depended on whether anything ran after it: in Run All a later
independent cell did, cell by cell the sink's cascade marked it again. Severity: low (a
wrong status).

**Fixed.** A blocked mark stands while the failure it waits on does: the walk keeps it on
a cell it would call idle when an upstream's failure still stands. A never-run cell that
no run tried to run (below a cell that failed alone, or on a cold open) is still idle. Regression test:
`tests/notebook/test_ws.py::test_a_cell_blocked_by_a_failure_stays_blocked_after_another_run`.

### 21. A batched cell reads an input the cell before it changed in place

**Property:** `test_run_all_properties.py` (SYSTEMS 9, BAT-1), confirming a survey lead.
Shrunk case: `xs = [0]`; `xs.append(0); n = len(xs)`; `m = len(xs)`. Run cell by cell,
the second cell changes only its own copy of `xs`, and the third loads the stored `[0]`:
`m = 1`. In a Run All batch the cells share one namespace, so the third read the appended
list: `m = 2`, stored and cached under a key over `xs = [0]`, which a later single-cell
run of it then hits. The batch detected the mutation and only warned. Severity: medium
(a wrong value, silently cached, behind a mutation warning).

**Fixed.** After a batched cell changes an input in place that it does not publish, the
harness loads that input again from the file it was stored as (seeded from the store, or
written by an earlier cell of the batch), so later cells read what a single-cell run
would. Regression test:
`tests/notebook/test_mutation_parity.py::test_a_batched_cell_after_an_input_mutation_reads_the_stored_value`.

### 22. A deny rule refuses one spelling of a table's name

**Property:** `test_encoding_properties.py` (SYSTEMS 6, ACL-2). ACL rules match the name a
table was requested under, by `fnmatch`, which is case-sensitive. Glue and Hive resolve
names case-insensitively, so a named catalog of either loads `lake:DB.EVENTS` as
`db.events`, and a deny on `lake:db.events` (or `*:db.*`) did not match it. The replay,
a catalog that lowercases names as Glue does behind a service-mode server, scanned the
denied table under two other spellings. Severity: high (an ACL bypass).

**Fixed.** Deny rules match a name in any case; allow rules still match only the spelling
they were written in, so a spelling never grants. Windows CI then showed the matching had
differed by platform: `fnmatch.fnmatch` folds case on Windows only, so there every table
and principal pattern, allow rules included, matched any case. Both now use `fnmatchcase`.
Regression tests: `tests/test_auth.py::TestADenyCoversEverySpellingOfTheName`.

### 23. A resumed run drops its first checkpoint

**Replay** of lead LR-4 (SYSTEMS 10). A run killed while a checkpoint was staging, or
whose checkpoint failed to stage, leaves that release abandoned, holding its sequence in
the ledger's unique `(build, kind, sequence)`. The resumed attempt numbered its
checkpoints from the restored release, so its first one took the same sequence:
`prepare` raised, the checkpoint was logged "not begun" and skipped, and a
`strata.head` naming it did not advance. A run that keeps dying in that stretch (a
preempted instance) resumed from the same checkpoint every time and never got past it.
Severity: medium (lost progress, and a run that cannot advance).

**Fixed.** A resumed run numbers past every checkpoint row of its build, whatever its
state. Regression test:
`tests/notebook/test_local_runs.py::TestAResumeAfterAnUnfinishedCheckpoint`.

### 24. Run All runs a batched `@nocache` producer again

**Replay** of lead BAT-2 (SYSTEMS 9). Run All records each cell it runs so a later
upstream materialization returns the first result, but a batch's cells were not
recorded. A cell Run All runs on its own (`@timeout`, a worker) after a batch that
held a `@nocache` producer materialized the producer again: its side effects ran twice,
and the batch's consumer and the later one read different values. Severity: medium (a
repeated side effect, two values of one input in one run).

**Fixed.** Inside a run, each cell a batch ran counts as executed. Regression test:
`tests/notebook/test_fresh_values.py::test_run_all_executes_a_batched_fresh_producer_once`.

### 25. A failed cache write fails the scan

**Replay** of lead CK-4 (SYSTEMS 3). A `clear()` between a put's `mkdir` and its
temp-file write removed the directory, and the `FileNotFoundError` reached the scan,
which had already read its row group. A full disk does the same. Severity: low (a
visible failure).

**Fixed.** The fetcher logs a failed cache write and returns the row group. Review of the
fix found the write's own gap: a put whose data rename succeeded and whose sidecar rename
failed left the data without its sidecar, and the bytes of the entry it replaced still
counted toward the cache's size. Such a put now removes the data it placed. Regression
tests: `tests/test_cache_optimized.py::TestAFailedCacheWriteIsAMiss`.

### 26. A rerun fails a build another process has just resumed

**Survey lead** (SYSTEMS 10, LR-2), replayed. `LocalRuns.open` reads the computation's
open build, checks its lease and then acts, in separate transactions. With the lease
lapsed, a second process on the store (a CLI run beside the server) resumed the build
between the read and the write. A rerun then failed the build anyway, since
`fail_build` was not pinned to the lease it had read, and the second process's live
run was fenced at its next release. A normal run instead raised vestigo's `ValueError`
from `resume`, where one process alone gives `RunRefused`. Severity: medium (another
process's run killed; two processes on one store).

**Fixed.** `open` pins the fail to the claim it read (`lease_owner`,
`lease_expires_at`). When the fail or the resume misses, `open` reads the build again,
so a build resumed elsewhere refuses this run. Regression tests:
`tests/notebook/test_local_runs.py::TestAnotherProcessBetweenTheReadAndTheWrite`, also
run on Postgres. The other half, two processes that both find no open build each
creating one, is finding 35.

### 27. A NaN filter value fails planning on a cold cache

**Property:** `test_bounds_pruning_properties.py` (SYSTEMS 5, PR-2). A scan filtered
on `f = NaN` (or any operator, with a NaN value) raised `ValueError: Cannot create
expression literal from NaN` from pyiceberg while the planner built its file-level
expression. The same scan succeeded once an unfiltered scan of the snapshot had cached
its manifest resolution, since a filtered plan reuses that and builds no expression, so
the outcome hung on what ran before. Severity: low (a client must send a NaN value).

**Fixed.** `filters_to_iceberg_expression` leaves a NaN-valued filter out, as it does a
nested column's, which only widens the file-level pruning; row-group pruning still
applies it. Regression test:
`tests/test_filters.py::TestTwoTierPruning::test_a_nan_filter_plans_on_a_cold_cache`.

### 28. A batched cell changes an upstream value it never names

**Property:** `test_run_all_properties.py` (SYSTEMS 7, BAT-1), with functions added to
the generator. A cell in a Run All batch can change a value it does not reference: by
calling a function that appends to an upstream list through its globals, or through a
container that holds the value (`box = [xs]`, then `box[0].append(0)`). Later cells in
the batch read the change, and their results were cached under a key over the stored,
unchanged value. Run cell by cell, each cell changes only its own copy. Finding 21's
fix reloaded only the inputs the cell names. Severity: medium (a Run All result differs
from a single run and is cached as if it did not).

**Fixed.** After each batched cell, the harness puts back from its file every value the
batch has published that changed, not only the cell's references. Mutation warnings
still name only what the cell references, as a single run's do. Each published value is
fingerprinted per batched cell (pandas sampled, numpy and pickled values hashed whole),
by the user's decision. Regression test:
`tests/notebook/test_mutation_parity.py::test_a_batched_cell_changing_an_input_it_does_not_name`.

### 29. A stored function in a batch cannot reach the values it reads

**Property:** `test_run_all_properties.py`, the edit pass. A stored function (a
`module/cell` export) names the upstream values it reads in `injected`, and a single
run's inputs load them into its module. A batch loaded the function without them, on
both its paths: an upstream cell run on its own, and a cache hit inside the batch. A
later cell calling it raised `NameError` under Run All and succeeded cell by cell.
Severity: medium (Run All fails a notebook that runs).

**Fixed.** The batch attaches the injected values to the function's spec on both paths
(`attach_injected` in `cell_paths/batch.py`), and the harness loads it through
`_deserialize_one`, as a single run's inputs are. Regression tests:
`test_mutation_parity.py::test_a_stored_function_called_twice_in_a_batch`,
`::test_a_batch_cache_hit_loads_a_function_with_what_it_reads`.

### 30. A stored function's own globals carry a change to the next batched call

**Property:** `test_run_all_properties.py`, pinned after 29's fix. A stored function
keeps its injected values in its own module, outside the batch's namespace, so 28's
reload never sees them. A function that appends to one returned 2 then 3 across two
batched cells, and 2 and 2 cell by cell. Severity: medium (as 28).

**Fixed.** The harness tracks the injected globals of each stored function it loads
(`_StoredGlobals`) and puts back, after each cell, any that changed. The pool worker
runs one cell per process, so nothing carries over there. Regression test:
`test_mutation_parity.py::test_a_stored_function_called_twice_in_a_batch`.

### 31. A recreated Hadoop-style table plans as the table it replaced

**Replay** (phase 7, item 4; MD-1). The metadata cache kept every remote
`*.metadata.json` and `*.avro` by location, on the claim that Iceberg never reuses a
name. A Hadoop-style table names its metadata `vN.metadata.json`, and one dropped with
purge and created again at its location writes `v1.metadata.json` again. Registered in
a catalog Strata reads (a `SqlCatalog` over a remote scheme in the replay), the new
table loaded with the old table's uuid, schema and snapshots from the cache, and its
scan read the old manifests (cached too) and failed on the purged data files. With the
old row groups in the row-group cache, whose key takes the uuid, paths and sizes from
that metadata, the scan would serve the old rows (follows from the key; not replayed).
Severity: high (silent wrong data when the row groups are cached; otherwise a table
that fails every scan until the entry is evicted). Narrow: it needs a Hadoop-style
table, which pyiceberg cannot write, behind a catalog.

**Fixed.** Only names with a uuid are cached, as Iceberg's writers name metadata,
manifest lists and manifests; any other name is read from the store every time.
Regression tests: `tests/test_iceberg_metadata_files.py::test_a_hadoop_style_name_a_recreated_table_reuses_is_read_from_the_store`,
`::test_the_names_iceberg_writers_give_are_cached`.

### 32. A lapsed run's close drops the lease of the process that resumed it

**Replay and model** (phase 7, item 4; LR-3). A local run's lease owner was its
process, `hostname:pid`, and the close of a run that ended without its final release
(`_end_without_final`) dropped the lease, or failed the build, by that owner. Two
processes on one store whose hostname and pid agree (two containers given one hostname,
both pid 1) are one owner. Run A lapsed after a checkpoint, B resumed the build, and A's
close then dropped B's live lease: a third run resumed the build and fenced B mid-run.
A's renewal also renewed B's lease. `LocalRun.tla` gained the close and a rival sharing
the owner string; `LocalRun_SharedOwner` violates the new `CloseLeavesOthers`. Severity:
medium (a live run fenced and run again; the fence keeps its releases out, so no wrong
data). Needs a shared hostname and pid.

**Fixed.** The lease owner names the attempt (`lease_owner(attempt)`), so a renewal, a
drop or a fail by an attempt that is no longer current is refused. `LocalRun_Patched`
holds with the rival sharing the run's hostname and pid. With the owner per attempt, the
close must name the attempt a hold in flight is opening, so it now takes the run's hold
lock; the model's close runs from any state with no hold in flight (a fenced run, or one
whose cell failed while working). Regression tests:
`tests/notebook/test_local_runs.py::TestTwoProcessesWithOneHostAndPid` (also on
Postgres).

### 33. A live stream's ownership claim lapses, and other nodes 404 it

**Replay** (phase 7, item 4; STR-5). With `node_advertised_url` set, a node claims each
stream it registers, and the claim expires one stream TTL later. Nothing renewed it,
while the stream lives on past that: a GET cancels its cleanup and re-arms it for
another TTL when it ends, whether it served the stream or was refused 429. Past the
first TTL, a retry routed to another node found no owner and got 404 for a stream its
node still held (the replay drives `StreamRegistry` and `StreamOwnershipStore` through
a re-arm). Nothing called `sweep_expired`, so a dead node's rows stayed in the table.
Severity: low (a visible 404 on a retry, multi-node only).

**Fixed.** A server loop renews every live stream's claim three times per TTL
(`StreamRegistry.renew_claims`) and sweeps expired rows; a node that dies stops
renewing, so its claims lapse within a TTL. A stream that ends while its renewal writes
is released again, or the renewal would restore its row after the release (the code
review caught it). Regression tests:
`tests/test_stream_ownership.py::TestRegistryHooks::test_a_live_stream_stays_claimed_past_its_first_ttl`,
`::test_a_stream_ended_during_its_renewal_is_left_unclaimed`,
`::test_the_server_renews_claims_and_sweeps_expired_ones`.

### 34. Two scans could share a `scan@v1` identity hash

**Property** (phase 7, item 4; PROV-3). `compute_identity_provenance` fed sha256 the
labelled fields one after another, `table:{table}@{snapshot}`, `schema:…`, `reading:…`,
`columns:{repr}`, `filters:…`, with no separator. Text in one field could stand in for
the next: table `t@1schema:Nonereading:A` with reading `B` (snapshot 5, schema 0) and
table `t` with reading `A@5schema:0reading:B` (snapshot 1, schema none) hashed alike.
Its filter fingerprint encoded each value as its wire spelling, so the string
`"__date__:2024-01-01"` and the date it spells had one fingerprint. The fingerprint was
also an MD5 prefix of 64 bits, and the projection and reading fingerprints (in the
row-group key, the reading also in the identity) SHA-256 prefixes of 64 bits.
Severity: low (a hash shared by two scans serves one's result for the other, but the
planner makes the `reading` string, and the wire turns such a string into the date
before the server sees it; neither pair arrives from a request).

**Fixed.** The identity hash is sha256 of canonical JSON over named fields; the filter
fingerprint adds each value's type (the type it is encoded as, so `numpy.float64` is a
float), and the three fingerprints are full SHA-256 digests. Every stored scan is keyed
anew once; the row-group cache moves to format version 7, whose start removes the old
entries (filtered manifest plans were never stored). The property
`test_two_scans_share_an_identity_hash_only_when_they_are_one_scan` pins both pairs as
examples, and its "spelled" pairs find the second without one. Regression tests:
`tests/test_materialize_service.py::TestComputeIdentityProvenance::test_one_fields_text_cannot_run_into_the_next`,
`tests/test_filters_conformance.py::test_fingerprint_a_string_spelling_a_tagged_value_is_not_that_value`.

### 35. Two processes that both find no open build each open one

**Survey lead** (SYSTEMS 10, LR-2), the half finding 26 left. `LocalRuns.open` reads the
computation's open build and, finding none, creates one (`create_build`) and opens its
first attempt (`open_attempt`): two more transactions, neither of which looks again.
Two processes on one store (a CLI run beside the server) that both read nothing open
each created a build, and the computation ran twice, each run's final release
repointing its outputs. Model `LocalOpen_Split` shows it in four steps. Severity: low
(two processes starting one computation within a transaction of each other).

**Fixed** with vestigo 0.8.0's `BuildStore.open_build`: the create and the first
attempt's open in one transaction under a lock keyed on the computation, returning the
open build when there is one. `_new_build` calls it; when another build comes back it
fails its own placeholder and returns None, and `open` reads the build again, as one
found (a live lease refuses the run), while a revert or a team-cache pull waits for that
run. An open that raises fails the placeholder too, unless its build committed. Model
`LocalOpen_Patched` (`OneOpenBuild`, the whole `open` loop of two processes with lapses,
resumes and reruns). Regression tests:
`tests/notebook/test_local_runs.py::TestTwoProcessesThatBothFindNoOpenBuild`, also run
on Postgres.

