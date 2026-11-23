# Strata's systems and their invariants

What each of Strata's stateful systems is, the properties it must keep, and
what checks each property. A property is only as good as its check, so every
one names a checker, or says plainly that it has none.

**Checked by** takes one of these forms:

- **model**: a TLA+ configuration under `tla/` that TLC checks exhaustively
  within its bounds (`formal/run_models.py`, in CI).
- **property**: a Hypothesis test under `formal/` that runs the real code
  against an oracle that does not share its logic (in CI).
- **test**: a regression test under `tests/`, usually the replay of a model's
  counterexample, inverted when the fix landed.
- **construction**: true by the structure of the code, with the reason given.
- **none**: unchecked. A **lead** is a suspicion from reading the code, with
  the interleaving or input that would break the property; it is not a
  confirmed bug until a replay against the real code shows it.

Findings 1–12 are in [`FINDINGS.md`](FINDINGS.md). The ledger's own systems
(artifact lifecycle, build leases, releases) are specified and modelled in
the `bearing-research/vestigo` repository, under `formal/`; section 7 says
which of their properties Strata relies on.

Reviewed against `main` on 2026-11-22.

## 1. Stream admission (QoS)

**What it is.** Every `GET /v1/streams/{id}` is admitted before it streams:
first a per-client semaphore, then the tenant's limiter for the request's
tier (interactive or bulk). A limiter is a `ResizableLimiter`
(`adaptive_concurrency.py`): an `asyncio.Condition` over `(capacity,
in_use)`. Tenants' limiters live in `TenantRegistry` (`tenancy/registry.py`),
an LRU of at most 1000 tenants; per-client semaphores live in
`QoSAdmission` (`streaming/qos.py`), an LRU of 10,000 per tier.

**Actors.** Concurrent requests, each of which can time out in the queue or
be cancelled (client disconnect, shutdown); release on every exit path;
registry eviction under tenant pressure; graceful shutdown, which drains on
`active_scan_count()`.

| ID | Property | Checked by |
| --- | --- | --- |
| ADM-1 | Whenever a slot is free and a request is queued, some queued request has been woken. | model `Admission_Patched` (finding 8; `Admission_Py312` shows the bug) |
| ADM-2 | A tenant never holds more slots than its quota. | model `Admission_Patched` (finding 9) |
| ADM-3 | The shutdown drain counts every live stream. | model `Admission_Patched` (finding 10) |
| ADM-4 | Bulk requests never starve interactive ones. | construction: the tiers have separate limiters and never wait on each other |
| ADM-5 | A client never holds more than its `per_client_*` slots. | none. **Lead:** `_get_client_semaphore` evicts the LRU semaphore even while it is held; after 10,000 other callers, a client's next request gets a fresh semaphore at full count. |
| ADM-6 | Each admission is released exactly once. | none. **Lead:** `_release` sets `_released` before awaiting `limiter.release()`, which takes the condition's lock; a cancel while it waits for the lock leaks the tenant slot and the client semaphore, and the flag stops a retry. |

## 2. Stream lifecycle

**What it is.** `POST /v1/materialize` on a miss registers a stream (its id is
the artifact id unless it is a refresh), registers the scan, starts the
prefetch and schedules a cleanup after the stream TTL. `GET
/v1/streams/{id}` cancels the cleanup, admits (section 1), serves (plan
passthrough, or a background build into the store followed by the blob), and
re-arms the cleanup on exit. Expiry discards the prefetch, pops the scan and
the stream, releases ownership and, if no build ran, fails the placeholder
artifact (`streaming/registry.py`, `api/routers/streams.py`).

**Actors.** Any number of GETs for one stream (a retried GET is allowed:
there is no consumed guard), the cleanup timer, a second materialize miss for
the same artifact id.

| ID | Property | Checked by |
| --- | --- | --- |
| STR-1 | A scan's state is not expired while a GET is serving it. | none. **Lead:** each GET re-arms the cleanup when it exits; a second GET that gets a 429 or finishes re-arms it while the first is still queued or streaming, and a TTL shorter than that wait expires the scan under it. |
| STR-2 | Every registered stream is expired once, and its unbuilt placeholder failed. | none. **Lead:** a second miss for the same artifact id registers the same stream id, and its `schedule_cleanup` cancels the first's, whose scan id is then never expired. |
| STR-3 | Only one background build runs per stream. | construction: the build is created check-then-set with no await between |

## 3. Row-group cache

**What it is.** A scan's Parquet row groups are cached as Arrow IPC on disk
(`cache/disk.py`), keyed by `CacheKey` (`types.py`). Since #1214 the key has
no snapshot id: it names what a snapshot can change about a row group's rows
instead, `hash(tenant | table_identity | table_uuid | file_path | file_size |
row_group_id | projection | schema_id | deletes | reading)`. `deletes` is a
digest of the row group's deleted positions and the paths of the equality
delete files that may apply; `reading` is the format version and name
mapping (`planner.py`).

**Actors.** Concurrent scans in one process, other processes sharing the
cache directory, eviction to the size limit, `clear()`.

| ID | Property | Checked by |
| --- | --- | --- |
| CK-1 | Equal keys mean equal cached content: every input that decides a cached row group's bytes is in its key. | none as a whole. Projection, schema evolution, defaults, name mapping, nanosecond handling, positional deletes and which equality deletes apply are in the key. **Leads:** (a) identity-partition values used to fill a column missing from the file come from the manifest, not the key, so a file re-added at the same path with another partition value hits the old entry; (b) the process-lifetime caches of delete rows (`DeletedRows`, `EqualityDeleteSets`) and of file columns are keyed by path only, so a table recreated at the same location can mix the old delete rows into a new-uuid key in a long-lived process. |
| CK-2 | The projection fingerprint is injective. | test `tests/test_smoke.py::TestCacheKey` (finding 4) |
| CK-3 | A reader never sees a partial or torn entry. | construction: `put` writes a temp file, fsyncs and `os.replace`s it; an unlinked file stays readable through an open descriptor on POSIX. Not checked on Windows, where replace over an open file fails. |
| CK-4 | A cache failure is a miss, never a failed scan. | none. **Lead:** `clear()` racing a `put` makes the temp write raise, and `_fetch_from_storage` does not catch it. |

## 4. Planning metadata

**What it is.** Two caches make planning cheap and survive an object-store
outage: the metadata FileIO (`iceberg/metadata_files.py`) keeps
`*.metadata.json` and `*.avro` files on disk on the claim that Iceberg writes
each once under a name it never reuses, and the planner caches manifest
plans by `(catalog, identity, snapshot_id, filter)`.

| ID | Property | Checked by |
| --- | --- | --- |
| MD-1 | A cached metadata file holds the bytes its location holds. | test `tests/test_iceberg_metadata_files.py` for the write and read path. **Leads:** a Hadoop-style catalog names metadata `vN.metadata.json`, and drop with purge then recreate reuses `v1.metadata.json`; a short read is cached without a length check. |
| MD-2 | A cached plan belongs to the table it is served for. | none. **Lead:** the manifest-plan key has no table uuid, so a recreated table that reuses a snapshot id gets the old plan. |

## 5. Pruning

**What it is.** Filters skip data in two places: Iceberg manifest bounds skip
files, Parquet row-group statistics skip row groups
(`planner._should_prune_row_group`, `Filter.matches_stats`). Pruning must be
conservative: when safety cannot be proved, read the data.

| ID | Property | Checked by |
| --- | --- | --- |
| PR-1 | A pruned row group holds no row the filter keeps. | property `test_pruning_properties.py` (float64, int64, string, timestamp, decimal, with nulls; finding 3) |
| PR-2 | A pruned data file holds no row the filter keeps. | none. Manifest bounds have their own NaN counts and string truncation. |

## 6. Access control

**What it is.** `AclEvaluator.authorize` (`auth/principal.py`) evaluates deny
rules, then allow rules, then the default, against a table's ACL name, which
comes from the form of the URI it was requested under.

| ID | Property | Checked by |
| --- | --- | --- |
| ACL-1 | An explicit deny is never overridden by an allow. | construction: denies are evaluated first and return |
| ACL-2 | A deny that matches one name of a table refuses every URI form that loads it. | test `tests/test_auth.py::TestADenyOnOneAddressCoversTheTable` for a SQL catalog shared by every warehouse URI (finding 12). none for other shapes: a named catalog over the same data is not detected. |
| ACL-3 | Cache contents are shared across principals; admission and retrieval are gated. | construction: the cache key has no principal; the gates are on the routes |

## 7. The ledger, as Strata uses it

The `vestigo` package specifies and model-checks the artifact lifecycle
(`ArtifactLifecycle.tla`) and builds with their leases, attempts and releases
(`BuildLease.tla`). Strata relies on these of its properties:

- At most one ready row per provenance; a ready row's bytes never change; an
  id's current value stays resolvable through a rebuild and through a
  provenance held by another id (findings 1, 2, 5, 7).
- Only the attempt holding a build's lease can complete, fail or release it,
  and a lapsed lease is never renewed (findings 5, 6).

Strata-side properties of how it drives the ledger are section 10.

## 8. Notebook staleness

**What it is.** After every edit and run, the session's walk
(`NotebookSession._compute_staleness_locked`) decides each cell's status.
A cell is ready when every consumed variable's latest stored version has the
provenance its current key derives (`derive_subkey(K, var)`), where `K` is
cellgraph's `parts_digest` over its definition, environment, upstream
outputs and inputs (mounts, tables, fetches, datasets). A leaf, or a SQL,
prompt or per-variant cell, keeps a ready status it already had while its key
is unchanged. After a run, `preserve_ready_cell_id` marks the cell ready
unless a direct upstream is not ready (then stale, reason upstream) or the walk
finds the cell itself stale (then the walk's verdict). An edit of the cell
being executed or requested is refused on every path (`cell_busy`), and every
edit's recompute leaves the running cell's status alone.

**Actors.** One execution per notebook (`NotebookExecutionState`, shared by
the WebSocket, REST and MCP), but one execution runs several cells: a target
runs its stale upstreams inline, and Run All batches cells. Edits arrive from
the WebSocket, REST (`PUT /cells/{id}`) and MCP (`ops.edit_cell`, which
reloads the session).

| ID | Property | Checked by |
| --- | --- | --- |
| NB-1 | A cell shown ready holds the result its current key names. | model `Staleness_Patched` (findings 11 and 14; `Staleness_BatchEdit` shows 14); test `tests/notebook/test_batch_running_status.py::test_a_batched_cell_edited_while_it_waits_is_not_shown_ready` |
| NB-2 | The executing cell is shown running, and no edit changes the source its run has. | model `Staleness_Patched` (findings 11 and 15; `Staleness_ApiEdit` shows 15); tests `tests/notebook/test_e2e_staleness.py::TestAnEditDuringARun` (WebSocket), `::TestARestEditDuringARun` (REST), `tests/notebook/test_mcp_server.py::test_an_agent_edit_of_the_running_cell_is_refused` (MCP) |
| NB-3 | Sequential edits and runs keep statuses and values consistent. | property `test_staleness_properties.py` (diamond of plain Python cells, through the real WebSocket) |

**What `Staleness.tla` leaves out.** It models the three edit paths, single
runs that start on a ready upstream, and Run All with its frozen sources, over
a chain of three cells. It leaves out: upstreams run inline (and "run this
only" on stale ones); readiness by content key, under which an edit back to
an old source is ready again once that result is latest (the model's source
versions only grow); uncached leaves, which the walk calls idle after a run
and the code keeps ready; and the ERROR and blocked-by-failure statuses.

## 9. Run All batching

**What it is.** `notebook_run_all` groups cells into batches that share one
harness process and one Python namespace (`CellExecutor.execute_batch`,
`cell_paths/batch.py`); the store is where their outputs spill for later
single-cell runs. This is the one deliberate exception to "the artifact store
is the sole source of truth".

| ID | Property | Checked by |
| --- | --- | --- |
| BAT-1 | Run All leaves the same stored values and statuses as running each cell in order. | none. **Leads:** an in-place mutation of an upstream value is visible to later cells in the batch (detected, warned, not keyed); a cache hit loads only consumed variables, so other names and module state depend on which cells hit; a consumer above its producer in display order. |
| BAT-2 | A `@nocache` cell runs at most once per run. | none. **Lead:** batch results are not recorded in the run's at-most-once record, so a later single-cell consumer runs a batched `@nocache` producer again. |

## 10. Local cell runs and the ledger

**What it is.** Every local Python run is a ledger build found again by its
computation key, with an attempt whose lease the server renews every 20 s
(`notebook/local_runs.py`). `strata.checkpoint` is an accepted checkpoint
release, the stored outputs are the final release, `strata.head` advances a
reference (`notebook/run_channel.py`). `LocalRuns.open` resumes an
interrupted build from its newest accepted checkpoint, or fails it and starts
fresh for a rerun. A run's token (`auth/run_tokens.py`) stands in for the
proxy token on the ambient client's routes until the run ends.

**Actors.** The run's harness (cold, over a pipe; warm, over the worker's
stdout), the renewal task, cancel, another process or session opening the
same key, server restart, and the wall clock: the lease is wall-clock time,
so it lapses whenever the process cannot renew for a minute (a laptop
asleep, an event loop blocked while outputs are stored). Every ledger call is
its own transaction; `LocalRuns.open` is several, with thread hops between.

**Holding the lease** (`LocalRuns._hold`, before every release and on every
renewal): renew it, or, when it lapsed and no other process took the build,
go on in a new attempt of the same build. A lapsed lease stays lapsed, and
`open_attempt` refuses while another attempt's lease is live, so a process
that took the build keeps it. Holds are serialized per run, because a renewal
is the process's, not the attempt's.

| ID | Property | Checked by |
| --- | --- | --- |
| LR-1 | A run interrupted after a checkpoint resumes from its newest accepted checkpoint; a rerun starts fresh. | test `tests/notebook/test_local_runs.py` (and on Postgres) |
| LR-2 | A live run of a key refuses a second. | test `tests/notebook/test_local_runs.py`, sequentially. **Lead:** `open` checks and acts in separate transactions, so two processes can both find no open build and create two, or one can fail a build the other just resumed. |
| LR-3 | Only the attempt that holds the lease releases, renews or ends its build. | the ledger's fence for releases (section 7); model `LocalRun_Patched` (`RunAttemptIsOwn`, `OneLiveHolder`). **Lead:** the lease owner is the process (`hostname:pid`), not the attempt, and the close path (`renew_lease(…, 0)`, `fail_build(lease_owner=…)`) is not checked against the attempt; within one process, runs of one computation are serialized, so this needs two processes sharing a pid. |
| LR-6 | A lease that lapses while no other process wants the build never costs the run. | model `LocalRun_Patched` (`NoNeedlessFence`; `LocalRun_Lapse` shows finding 13, `LocalRun_Race` the fix without its lock); test `tests/notebook/test_local_runs.py::TestALapsedLeaseWithNoRival`, also on Postgres. Assumes a hold and its release take far less than the 60 s lease. |
| LR-4 | A resumed attempt can always release its next checkpoint. | none. **Lead:** the sequence restarts after the newest *accepted* checkpoint, but the unique `(build, kind, sequence)` also counts abandoned and rejected rows; the colliding checkpoint is logged and skipped (`prepare` raises `ReleaseError`), and a `strata.head` naming its sequence can point at the earlier attempt's release. |
| LR-5 | A cell executes at most once per request. | construction: a fenced, timed-out or failed warm run is never run again cold; only a worker that never got the manifest is a miss. Test `tests/notebook/test_subprocess_line_limits.py::test_an_event_that_fails_is_the_run_s_error_not_a_pool_miss` (#1613). |
| TOK-1 | A run's token is refused once its run ends, and never carries `admin:` scopes. | test `tests/test_run_tokens.py`, `tests/notebook/test_run_token_cells.py` |

## 11. `@each` cells

**What it is.** An `@each` cell runs one instance per file or row, each keyed
on its own content and stored at `@variant=each-<sha16>`. When every instance
succeeds, one manifest per output names each instance's version and requires
it (`notebook/each.py`, `cell_paths/fanout.py`).

| ID | Property | Checked by |
| --- | --- | --- |
| EACH-1 | A manifest exists only when every instance succeeded; on a partial failure the cell's previous outputs stand. | test `tests/notebook/test_each.py` |
| EACH-2 | An instance a kept manifest names is never collected. | test `tests/notebook/test_each.py`, `test_each_postgres.py`. **Lead:** the guard against collecting during a run is a process-local counter, so a sweep from another process sees only `min_idle_seconds`. |

## 12. Core provenance encodings

| ID | Property | Checked by |
| --- | --- | --- |
| PROV-1 | `derive_subkey` labels cannot collide. | construction: variable names cannot contain `:` or `=` and `_` names are never stored; other labels carry their own prefixes |
| PROV-2 | `CacheKey.to_hex`'s `\|` separator cannot be forged. | construction: a collision needs a file path starting with digits and `\|` |
| PROV-3 | Equal transform specs serialize to the same bytes, and different ones differ. | none |
| PROV-4 | Signed URLs sign a canonical encoding. | construction: `json.dumps(…, sort_keys=True)` |
