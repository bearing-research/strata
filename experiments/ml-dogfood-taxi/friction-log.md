# Friction log — ml-dogfood-taxi

Every gap hit while using Strata as the only data/artifact tool, in order
encountered. Severity: **blocker** (needed a non-Strata tool or gave up) /
**major** (ugly workaround, would annoy a real user daily) / **minor**
(papercut).

Format per entry: what I was trying to do → what happened → severity →
workaround used → what would fix it.

---

## 1. Package name ≠ import name when declaring the client dep

Trying to: give notebook cells the Strata SDK by adding `strata` to the
notebook's `pyproject.toml` (it's the import name).
What happened: `uv sync` → "there are no versions of strata"; the PyPI
package is `strata-notebook`, the import is `strata`. Nothing in the error
points at the right name.
Severity: **minor** (papercut, but it's the literal first step of SDK use).
Workaround: depend on `strata-notebook`.
Fix: slim `strata-client` package whose name matches usage, or at minimum a
PyPI `strata` stub that errors with the correct name.

## 2. Full server package required just to be a client

Trying to: call `client.materialize` / `client.put` from analysis code.
What happened: the only way to get `strata.client` is installing
`strata-notebook` — the entire server (FastAPI, uvicorn, pyiceberg, duckdb,
Rust extension, …) lands in every analysis venv that only needs an HTTP
client + pyarrow.
Severity: **major** (slow env builds everywhere; a real team would balk at
the server in their training image).
Workaround: accept the heavy install.
Fix: publish a slim `strata-client` (httpx + pyarrow only).

## 3. Personal mode and server-side transforms are mutually exclusive — and the failure is silent

Trying to: run feature engineering server-side via the embedded
`duckdb_sql@v1` (the workflow examples/10_artifacts.py showcases), against
the default `python -m strata` personal server.
What happened: `POST /v1/materialize` returned **200** with an artifact URI;
the artifact sat in `state=building` forever; the client's data fetch died
with a bare `400 Artifact is not ready (state=building)`. Root cause:
`server_transforms_enabled` requires `deployment_mode == "service"` +
`transforms.enabled` (`config.py:564`) — the build runner never starts in
personal mode, but materialize happily queues into the void.
And the trap is two-sided: `client.put` (needed to persist locally-computed
results) requires `writes_enabled`, which is **personal-mode only** — so no
single deployment mode can run the canonical ML loop
(server-side transform → local train → put model).
Severity: **blocker** (first pipeline step; silent hang masquerading as
success; no mode supports the whole workflow).
Workaround: stay in personal mode; demote feature engineering from
server-side SQL to client-side pandas; keep provenance by `put`-ing the
features with `inputs=[scan_uri]`.
Fix (layered): (1) materialize should fail fast (4xx) when the transform
has no executor in this mode, not 200-then-forever-building; (2) personal
mode should run embedded transforms in-process — a personal server that
can't execute `duckdb_sql@v1` makes the flagship artifact workflow
service-only; (3) revisit writes-vs-transforms mode coupling.
Refinement (found later): the protocol DOES have a personal-mode answer —
the server returns a build spec and upload/finalize endpoints exist for the
client to execute locally — but `StrataClient.materialize` never implements
that client-side build path. The integration tests hand-roll it, and the
whole real-table duckdb test class is skipped with reason "Requires
client-side local execution for duckdb_sql\@v1". So this is a known,
half-built seam, not a design hole.

## 4. CRITICAL BUG: silent data loss on multi-row-group scans (65% of rows dropped)

Trying to: scan the 2,964,624-row trips table via `materialize(scan@v1)` +
`to_pandas()` — the docs' example-1 flow.
What happened: got exactly 1,048,576 rows (= row group 1 of 3). **No error,
no warning.** Trained a model on silently truncated data — and would never
have known if the row count weren't a suspicious power of two.
Root cause chain:
- `server.py:682` `_build_identity_artifact` persists the artifact blob as
  `b"".join(all_chunks)` where each chunk is a complete per-row-group IPC
  stream (schema + batches + EOS) — three streams butted together.
- `client.py:704` reads with `ipc.open_stream(...).read_all()`, which stops
  at the first EOS. Every standard Arrow reader does the same.
- `fast_io.concat_stream_bytes` / `stream_concat_ipc_segments` — the
  purpose-built (Rust-accelerated) stream mergers — are referenced by
  NOTHING in src/. Dead code.
- Tests never catch it: the canonical `test_db.events` fixture is far too
  small to span multiple row groups.
- Artifact metadata `row_count` is correct (2,964,624) while the readable
  data is 1,048,576 — the inconsistency is cheaply detectable at finalize
  time and nothing checks it.
Severity: **critical** (silent correctness violation of invariant #1 on the
flagship path; an ML user gets a biased model with zero signal anything is
wrong).
Workaround: none acceptable; fix required before continuing the dogfood
(loop-reading successive IPC streams client-side would mask a server bug).
Fix: use proper IPC concatenation when finalizing multi-task scan
artifacts; add a finalize-time check (readable rows == row_count); add a
multi-row-group warehouse fixture so tests exercise this forever.

## 5. No way to heal a poisoned artifact: refresh forks instead of repairing

Trying to: after fixing #121 server-side, get the pipeline to re-derive from
a complete scan. Ran `materialize(refresh=True)` then `strata run --force`.
What happened: refresh created a **parallel artifact id** with the complete
data, while provenance-cache hits kept resolving to the old truncated
artifact (same provenance hash, two ready artifacts, lookup returns the
stale one). `--force` on the notebook re-executes cells but can't reach
through to the server's provenance cache. Only `delete_artifact` on the
poisoned version unwedged it. Also: the forever-`building` zombie from
friction #3 still sits in the store; `cleanup_failed` exists but nothing
surfaces it.
Severity: **major** (cache-integrity recovery is undiscoverable; dedup
invariant — one provenance hash, one artifact — silently violated by
refresh).
Workaround: `client.delete_artifact(id, version)` by hand.
Fix: refresh should supersede the canonical artifact for that provenance
(new version of same id, lookup returns newest ready); an
`strata artifact verify/repair` surface; janitor for zombie `building`
rows.

## 6. Slash-namespaced artifact names are write-only

Trying to: check `is_artifact_stale("taxi/trips-raw")` — names were created
fine by `materialize(name="taxi/trips-raw")` / `put(name=...)`.
What happened: every name *read* endpoint is path-routed
(`/v1/artifacts/names/{name}/status`), so a name containing `/` 404s —
URL-encoded `%2F` too (FastAPI path params don't match encoded slashes by
default). Write side takes the name in a request body, so creation
succeeds: the name is stored but unreachable.
Severity: **major** (namespaced names like `team/dataset` are the natural
registry convention — every model registry uses them; here they silently
become write-only).
Workaround: dot convention (`taxi.trips-raw`).
Fix: `:path` converter on name routes (or query-param form) + client-side
encoding; validate/normalize at set_name time so write and read agree on
the alphabet.

## 7. Tenant bookkeeping leaks into single-user personal mode

Trying to: `set_name` on the features/model artifacts I'd just created via
`client.put` in the same personal-mode session.
What happened: `400 — Artifact ...@v=1 belongs to tenant _default, cannot
assign name in tenant None`. The upload path stamps `_default` as tenant;
the names path resolves no-header requests to tenant `None`; and
`StrataClient` has no tenant parameter at all, so the SDK cannot even
express the header that would reconcile them. Inconsistent default-tenant
resolution across endpoints, surfaced to a user who never asked for
multi-tenancy.
Severity: **major** (artifacts you just made can be unnameable; personal
mode should never mention tenants).
Workaround: raw httpx call with `X-Tenant-ID: _default`.
Fix: one default-tenant constant applied uniformly when multi-tenancy is
off; tenant plumb-through on StrataClient for when it's on.
Refinement: with multi-tenancy disabled the X-Tenant-ID header is ignored
entirely, so there is NO HTTP workaround — upload route stamps `_default`,
scan route stamps NULL, names route resolves None. Naming at put-time works
(same transaction); naming after the fact is impossible for put-created
artifacts. Three different "no tenant" spellings on three paths.

## 8. Notebook is blind to lake staleness — fresh data never triggers recompute

Trying to: after month 2 landed (snapshot S2), re-run the pipeline and have
it notice the training data changed.
What happened: the artifact layer KNOWS — `get_name_status("taxi.trips-raw")`
returns `is_stale: true` with the exact S1→S2 transition in
`changed_inputs`. But `strata run` (no --force) serves every cell from
cache: cell provenance is `(inputs, source, env)` and the resolved lake
snapshot isn't in it. The orchestration layer and the persistence layer
don't talk; the notebook's cascade/staleness machinery only sees cell→cell
edges.
Severity: **major** (the all-in-one pitch is "Strata notices your data
changed"; today the notebook quietly trains on stale data while the server
holds the staleness fact one query away).
Workaround: `strata run --force` (recomputes everything, including what
wasn't stale).
Fix: cells that materialize against lake tables get the resolved snapshot
folded into cell provenance (scan-level `input_versions` already records
it), so S2 → upstream cell goes stale → existing cascade machinery handles
the rest. This is the highest-leverage integration gap found so far.

## 9. Promotion is a silent pointer swap — no history, no champion/challenger

Trying to: after retraining on S2, compare the new model against the
promoted one before deciding.
What happened: the promote cell's `put(name="taxi/tip-model")` silently
repointed the name to the challenger. The S1 champion still exists but is
reachable only by raw artifact id — which I knew solely because it was in
my terminal scrollback. No alias history ("what did this name point to
yesterday"), no holding two pointers (`champion`/`candidate`) on one model
line, no tags to record WHY a version was promoted (its MAE lives in a
separate metrics artifact you must know to look for).
Severity: **major** (this is the registry gap, experienced concretely: a
real team loses track of the previous production model the moment someone
re-runs promote).
Workaround: keep artifact ids in a text file. Seriously.
Fix: Phase 2 registry layer — aliases + tags + append-only audit of name
moves (exactly the planned design; the dogfood confirms it's the right
shape).

## 10. Lineage is plumbing-complete but has no human surface

Trying to: answer "which snapshot trained this model?" for two models.
What happened: the answer IS in `artifact.lineage()` — the table edge
carries the snapshot id in `input_version` — but extracting it took
hand-written graph-walking code; nothing labels the edge as a snapshot,
and there's no CLI (`strata artifact lineage <name>`) or formatted view.
A compliance officer could not self-serve this.
Severity: **minor-major** (data is right; surface is missing — pure UX
debt on top of the strongest part of the system).
Workaround: 15 lines of edge-filtering Python.
Fix: artifact CLI with a lineage subcommand that renders the chain
(model <- features <- scan <- table @ snapshot), plus typed edge kinds in
the payload.

---

# Synthesis (2026-06-05)

Ten entries from one day of honest use. One critical bug found and fixed
(#121 / PR #122). The static gap list from the pre-dogfood audit gets
substantially re-ranked.

## What the dogfood VALIDATED (the pitch is real)

- **Snapshot-pinned scans + staleness detection**: append month 2 →
  `is_stale: true` with the exact S1→S2 transition, zero manual
  bookkeeping. The lake seam is the strongest part of the system.
- **Provenance lineage is complete**: model → features → scan → table @
  snapshot, walkable for both champion (S1) and challenger (S2). The
  EU-AI-Act audit answer exists today (just unsurfaceable, see #10).
- **Content-addressed dedup across re-runs** worked exactly as promised.
- **Eval-metrics-as-structured-artifact** (dashboard-friendly constraint)
  was natural to write — no API needed, a 4-row DataFrame did it.
- 213MB artifact blobs round-tripped fine; chunked upload was never missed
  at this scale.

## Re-ranked gap list (by experienced severity, was → now)

1. **Integrity hardening** (frictions 4, 5 — NOT on the static list).
   #121-class silent corruption, no finalize-time validation, no
   verify/repair, refresh forks instead of healing, zombie `building`
   rows. "Production-level artifact versioning" is an integrity promise
   before it is a feature list.
2. **Notebook×lake staleness integration** (friction 8 — NOT on the static
   list). The server knows data changed; the notebook trains on stale
   cache anyway. Fold resolved snapshots into cell provenance and the
   existing cascade machinery does the rest. Highest product leverage.
3. **Registry layer: aliases + tags + audit history** (friction 9; was
   "P2"). Confirmed in its planned shape by concretely losing the
   champion pointer. Promote = silent name swap today.
4. **Mode coherence for the ML loop** (friction 3; was implicit). The
   canonical workflow spans personal-only writes and service-only
   transforms; materialize 200s into a forever-`building` void. Fail-fast
   + embedded transforms in personal mode (or finish the half-built
   client-side build path).
5. **SDK packaging** (frictions 1, 2; was unranked): slim `strata-client`,
   name parity on PyPI.
6. **Names + tenancy hygiene** (frictions 6, 7; was unranked): slash names
   are write-only; three spellings of "no tenant" across routes. Small
   fixes, outsized trust damage.
7. **Artifact/lineage CLI surface** (friction 10; was #4 "artifact CLI").
   The data model already answers the questions; nothing renders it.
8. **Experiment/run grouping** (was #1 — DEMOTED). Single-model-line work
   never missed it; it matters at many-experiments scale. Build after the
   above.
9. **Chunked/resumable upload** (was #5 — demoted until multi-GB models
   are actually exercised).
10. **Params-as-queryable-keys** (was #3 — partially fine): transform
    params carried hyperparameters acceptably; queryability is the
    missing half, fold into the registry/dashboard-friendly metadata work.

## Headline takeaways

- The differentiator (lake-native snapshot provenance) is REAL and worked
  first try. The liabilities are integrity guarantees and surface polish,
  not the core model.
- The most valuable single feature to build next is #2 (notebook reacts to
  lake staleness) — it turns three disconnected good behaviors into the
  demo that sells the whole thesis: append data → open notebook → "your
  model is stale" → one click → retrained + promoted with full lineage.
- Phase 2 registry design (aliases/tags/audit) survives contact with
  reality unchanged.

---

# Follow-ups landed (2026-06-06)

- **Friction #4/#5 → FIXED** via #121 (PR #122) + #123 (PR #124): scans never
  silently truncate; finalize validates blobs; refresh supersedes instead of
  forking; zombie builds swept; `strata artifact verify` exists.
- **Friction #8 → FIXED** via the `@table` annotation (PR #125), demoed on
  this very pipeline:
  1. scan.py declares `# @table trips file://...#nyc.trips`; the cell scans
     pinned via the injected `trips_snapshot`.
  2. Unchanged lake → `strata run` → every cell cache-hits.
  3. Month 3 appended (S2 → S3, +3,582,628 rows).
  4. Plain `strata run` — **no --force** — scan recomputed at S3 with all
     9,554,778 rows, cascade retrained + promoted automatically.
  The "append data → notebook notices → retrain with lineage" demo is real.
- Still open from the log: #3 mode coherence, #1/#2 SDK packaging, #6/#7
  name/tenant hygiene, #9 registry layer (Phase 2), #10 lineage CLI.

---

# Follow-ups landed (2026-06-11) — all 10 frictions closed

Everything still-open as of 2026-06-06 has now shipped to `main`:

- **#1/#2 (SDK packaging) → FIXED** via the new **`strata-client`** distribution
  (#159–#162). `pip install strata-client` → `from strata_client import
  StrataClient` on **httpx + pyarrow only** — no pyiceberg/fastapi/duckdb/
  pydantic, no Rust extension. Client and server are now **fully independent**
  (they share only the JSON wire protocol; neither imports the other), and the
  package-name dead-end ("no versions of strata") is gone — the import is
  `strata_client`. Integrations ship as `strata-client[duckdb|pandas|polars|
  datafusion]`. **Caveat: not yet dogfood-re-exercised** — verified by tests +
  the client-only CI job, but no run has actually *installed and used*
  strata-client (the original dogfood worked around #1/#2 by installing the
  whole server).
- **#3 (mode coherence) → FIXED** via #126: personal mode runs the embedded
  transforms in-process, and an unknown/unrunnable transform now fails fast with
  a 400 instead of parking in `building` forever. The full
  scan → transform → put loop works in personal mode. The re-audit (#155)
  un-skipped and ran the contract suite that proves it.
- **#6 (slash-namespaced names write-only) → FIXED** via #127: name routes use
  `:path` converters, so `team/dataset/raw` reads, resolves, and deletes.
  Re-audit added `test_slash_name_round_trip`.
- **#7 (tenant bookkeeping leaks into personal mode) → FIXED** via #126/#127:
  put-created artifacts resolve to a nameable tenant, so `set_name` after the
  fact works with no header. Re-audit added `test_put_then_name_after_the_fact`.
- **#9 (promotion = silent pointer swap) → FIXED**: the registry layer (#129)
  added aliases / tags / append-only audit + approval gates (#136/#137), and the
  notebook **registry dashboard** (#147–#151) surfaces promote / approve / audit
  in the UI — promotion no longer has to be code.
- **#10 (lineage has no human surface) → FIXED**: `strata artifact lineage` (CLI)
  plus the notebook **lineage view** (#154) render
  `model ← features ← scan ← table @ snapshot`.

**Status: 10/10 frictions resolved.** The re-audit (#155) verified #3/#4/#5/#6/#7
with tests and recovered ~19 contract/lifecycle/staleness tests that had been
needlessly skipped. The one remaining gap is **empirical, not code**: the
strata-client fix (#1/#2) has never been exercised by an actual run — see the
re-run plan below.

---

# Reassessment + next step (2026-06-11)

**Where we are.** The dogfood did its job: it surfaced 10 frictions, all now
closed. The *differentiator* (lake-native snapshot provenance + staleness) was
validated first-try and is unchanged; the *liabilities* it exposed — integrity
guarantees (#4/#5), the notebook×lake staleness gap (#8), registry/lineage
surface (#9/#10), mode coherence (#3), name/tenant hygiene (#6/#7), and SDK
packaging (#1/#2) — have each been fixed and (mostly) test-verified. The
"ML all-in-one" thesis is now substantially *built*: append data → notebook
notices → retrain → promote/approve → full lineage, plus a slim client on-ramp.

**What's NOT yet validated.** Three surfaces shipped *after* the original run and
have never been dogfooded:
1. **`strata-client`** — the #1/#2 fix; nobody has installed it and run an
   analysis against a server with it.
2. **The registry dashboard** (#147–#151) — promote/approve/audit/lineage in the
   UI; the first run predated it and used the CLI/SDK.
3. **The ambient `strata` client in cells** (#146) — cells calling
   `strata.put(name=)` directly.

**Plan — a confirming "Phase 1" re-run.** Re-run the taxi pipeline on current
`main` with the explicit goals of (a) empirically closing #1/#2 by *actually
using* `strata-client` from the analysis code, and (b) exercising the new
dashboard + ambient-client surfaces, recording any *new* friction the fixes or
the next layer reveal. Demoted/backlog items (experiment-run grouping,
params-as-queryable-keys, chunked upload) stay deferred — they only bite at
many-experiments / multi-GB scale, which this single-model line doesn't hit.
Acceptance: the full loop runs against `strata-client` with no server install in
the analysis venv, and the dashboard drives at least one promote+approve.

---

# Confirming re-run — strata-client driven (2026-06-11)

Ran the loop against current `main` from a venv containing **only**
`strata-client` (httpx + pyarrow) — no server stack — talking to a personal-mode
server over HTTP. Server scanned the live `nyc.trips` warehouse (13.07M rows,
4 months loaded).

**Result: #1/#2 EMPIRICALLY CLOSED.** The full loop ran end-to-end via the slim
client:
- `StrataClient()` with **no args** resolved `http://127.0.0.1:8765` (config=None)
  — no boilerplate, the env-var/pyproject resolver works.
- `materialize(scan@v1)` (no snapshot_id → latest), `to_table()` (pyarrow),
  `put(name=…)`, `set_tag`, `set_alias("champion")` → applied,
  `resolve_alias`, `get_tags`, `get_registry_audit` (2 entries) — all worked.
- `get_artifact_by_name(...).lineage()` returns the full chain
  `model ← scan ← table` (3 nodes / 2 edges). champion + candidate coexist.
- **Slimness verified**: `find_spec` for strata / pyiceberg / fastapi / uvicorn /
  duckdb / pydantic all `None` in the analysis venv. The thing the original
  dogfood worked around (the whole server in the analysis venv) is gone.

**SDK completeness**: materialize/put/fetch, to_table/to_pandas/to_polars,
get_artifact[_by_name], lineage, dependents, info, is_artifact_stale,
list_artifacts, delete_artifact, and the full registry (set_alias / resolve_alias
/ set_tag / get_tags / get_registry_audit / list_pending_changes / approve /
reject). No missing accessor hit during the loop.

**New friction found (1):**
- **F11 (migration) — `from strata.client import StrataClient` now hard-breaks.**
  The client moved to `strata_client`; existing code (incl. this dogfood's own
  `pipeline/cells/*.py`, examples, any user script) raises a bare
  `ModuleNotFoundError: No module named 'strata.client'` with no pointer to the
  rename. Severity: **major** for anyone upgrading. **RESOLVED 2026-06-11 (user):
  option A — accept the clean break pre-1.0; the changelog breaking-change entry
  is the migration note (`from strata.client import StrataClient` →
  `from strata_client import StrataClient`).** No `strata.client` error-shim
  (pre-1.0, few external users; a permanent server-side shim is clutter). Folds
  into the already-inventoried "SDK version-skew error story" (0.4.0) if a
  friendlier upgrade story is wanted later. NOTE: this experiment's
  `pipeline/cells/*.py` still import `strata.client` and would need updating to
  `strata_client` (or the ambient `strata` cell client) before any notebook
  re-run.

**Not re-exercised in this run (flagged, not closed):**
- The **registry dashboard** (promote/approve/audit/lineage in the notebook UI)
  and the **ambient `strata` cell client** — both are notebook-session / Vue
  surfaces, not headless-checkable; they need a browser eyeball against a
  notebook session (this run used the standalone server + SDK).
- The **protected-alias approval flow** end to end (set `STRATA_REGISTRY_
  PROTECTED_ALIASES=champion` → move returns 202 pending → approve). The SDK
  methods exist and are unit-tested; not re-driven here to avoid a server
  restart + 13M-row rescan.

**Verdict:** the dogfood's headline gap (#1/#2) is now closed empirically, not
just by tests. Remaining is a migration papercut (F11) and two UI surfaces that
want a human eyeball.

---

# Ambient-client notebook re-run (2026-06-11)

Repointed `pipeline/cells/*.py` off `from strata.client import StrataClient`
(F11) to the **ambient `strata` client** (no import — injected into the cell).
`strata validate` clean; full headless `strata run` works end-to-end against the
live server (scan → train → evaluate → promote, ~18s warm). The promote cell
published `taxi/tip-model` with real tags (mae=1.23, r2=0.64) + champion/candidate
aliases, **cell-stamped** (`nb_cell`=scan,promote) so the per-cell promote strip
populates. The dogfood notebook now drives the dashboard.

**Side benefit (friction #2, concrete):** with cells on the ambient client, the
notebook venv no longer needs `strata-notebook` at all — its `pyproject.toml`
still lists the whole server as a dep (legacy); it could be just
`pyarrow + pandas + scikit-learn`. The slim-notebook win, demonstrable.

**New friction found (F12) — ambient client `IncompleteRead` on a large *cold*
scan.** The first run failed at the scan cell: the cold scan of the full
13.07M-row / ~941 MB table over the live `/v1/streams/{id}` endpoint raised
`IncompleteRead(0 bytes read)` in the harness subprocess after ~35s, and the
artifact was left `state=failed`. Isolated: the **identical** scan succeeds via
`strata-client` (httpx), and via the ambient client's urllib path once the
parquet is **warm** (read 941 MB in 3.2s); smaller live streams (≤~420 MB)
always work. So it's a **cold-scan timing edge case on very large live streams**,
specific to the ambient (urllib) client in the harness — not the server, not the
repointing, not request size per se. Severity: **minor–major** (a user whose
*first* scan is huge hits a confusing failure + a poisoned `failed` artifact;
re-run succeeds warm). **RESOLVED 2026-06-12 — root-caused + fixed.** Not a
timeout and not the size: deterministically reproduced by **clearing the server's
row-group cache** (forcing a fresh multi-row-group scan over the live
`/v1/streams` endpoint) — `IncompleteRead` in ~1.6s, partial bytes vary
(0 → 511 MB). Decisive isolation: the **identical** fresh scan succeeds via
`strata-client` (httpx) and via the ambient client's urllib path **when read in
chunks**; only a single all-at-once `resp.read()` fails. Mechanism: one big
blocking read lets the server's send buffer fill, the live-stream generator's
`request.is_disconnected()` check trips (false positive — no error logged, the
artifact is finalized `failed`), and the client sees `IncompleteRead`. Fix:
`notebook_client.py::_get_bytes` now **drains the response in 1 MiB chunks** (like
httpx). Validated: `strata run --force` on a cache-cleared fresh 12.8s scan now
runs clean end-to-end. Regression test asserts the chunked drain (rejects a
revert to `resp.read()`). **Candidate server-side follow-up (not done):** the
`/v1/streams` generator finalizing a `failed` artifact on a transient
`is_disconnected()` is fragile — any slow-reading client could poison an
artifact; consider not persisting `failed` on disconnect, or hardening the
disconnect check.
