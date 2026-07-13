# Changelog

All notable changes to Strata will be documented in this file.

Entries focus on user-visible changes and release framing rather than
exhaustive commit history.

## Unreleased

### Added

- **App view — open a notebook as a read-only interactive app.** Click **App**
  in the notebook header (or visit `/app/<sessionId>`) to render just the widget
  control panels + display outputs — no editor, DAG, or toolbars. The connection
  is read-only (edits and cell runs are rejected server-side); viewers can still
  drive widgets, so with a widget's **⚡ Live** toggle on it's an interactive
  "tweak a parameter, see the result" dashboard. `# @app hide` keeps a cell out
  of the view.
- **Interactive widget cells.** A new `widget` cell kind is a declarative
  control panel — one control per line (`alpha = slider(0, 1, default=0.5)`,
  plus `number` / `dropdown` / `checkbox` / `text`). Each control defines a
  variable downstream cells consume; dragging a control marks those cells stale
  (run them, or opt into auto-run later). Values are content-addressed, so
  returning a slider to a prior value is a cache hit. Add one from the **+**
  menu; see [Widget Cells](docs/notebook/cells.md) and the
  `examples/widget_playground` notebook. (#420–#423)
- **An interactive data viewer for DataFrame outputs.** Table cell outputs now
  render in a scrollable grid you can page, sort (click a header), search, and
  filter per column — backed by the full cached artifact, not a 20-row preview —
  with CSV / Parquet export. The terminal viewer (TUI) gains the same paging /
  sort / CSV-export over the grid. (#416, #417)
- **An MCP server for driving a live notebook from a coding agent.** Enable
  `mcp_enabled` (personal mode only, behind the `[mcp]` extra) and Strata mounts
  a Model Context Protocol endpoint at `/mcp` — `claude mcp add --transport http
  strata http://localhost:8765/mcp`. An external agent (Claude Code, any MCP
  client) gets the same operations the `strata` CLI drives — read
  (`list_notebooks` / `get_notebook` / `get_cell` / `dag` / `status`), run
  (`run_cell` / `run_tests`), author (`add_cell` / `edit_cell` / `remove_cell` /
  `move_cell`), and dependencies (`add_dependency` / `remove_dependency`) —
  against a **warm session**, not an offline copy. Because the tools reuse
  Strata's broadcasting execution paths, the browser UI and the terminal viewer
  become a live view of the agent at work. (#117)

### Fixed

- **No more spurious `display` mutation warning.** Any cell whose last line was
  a bare expression (e.g. a trailing `df` to show it) emitted a false
  "'display' was mutated in place" warning — the harness's own injected display
  helper being flagged. Injected/ambient names are now excluded from mutation
  detection. (#418)
- **Downstream cells now read "stale", not "idle", when an upstream changes.**
  When you edit an upstream cell (or its inputs, mount, or environment change),
  a downstream cell that already holds a result is now marked **stale** with an
  "upstream changed" reason, instead of a bare "idle" with no explanation. The
  web UI surfaces this as `stale · upstream changed` and the terminal viewer
  shows the stale glyph. A never-run downstream stays **idle** — there is no
  cached result to invalidate until its upstream produces inputs. (#361)

## 0.4.0 — 2026-07-01

0.4.0 is a **consolidation and hardening** cycle. The headlines are a new
**read-only terminal viewer** for notebooks and a **full agent-facing notebook
CLI**, alongside an internal restructuring of the server, per-cell unit tests,
broader value serialization with in-place mutation tracking, concurrency-bug
fixes, and CI hardening. No breaking changes.

### Added

- **A terminal viewer for notebooks (`strata-notebook-tui`).** A read-only,
  full-screen spectator that attaches to a running notebook session over the
  WebSocket protocol and renders it live — cells flip status as they run, with
  the detail view following the action. The detail pane splits into a **code**
  group — **Source** (syntax-highlighted with the one-dark theme, matching the
  web UI) and **Tests** (the cell's test source) — and a **runtime** group:
  **Output** (markdown cells and markdown outputs render as markdown, a
  DataFrame/table renders as a real table, images render inline via the
  terminal's graphics protocol — enlarge one full-screen with `i`), **Console**,
  **Agent** (an AI agent's reasoning streams here as it drives the notebook), and
  **Results** (each unit test's outcome + failure diff). Plus a layered ASCII
  **DAG** view (`d`), a per-cell run-time column, cascade / environment-job
  progress in the header, follow mode, per-cell unit-test result badges, a `?`
  keybinding reference, and background auto-resync so the view stays live
  without manual refresh. Ships behind the `[tui]` extra
  (`uv tool install "strata-notebook[tui]"`); it never edits or runs cells —
  purely for watching, e.g. an agent build a notebook in one terminal while you
  watch in another. See [Terminal Viewer](notebook/tui.md).

- **A full notebook CLI for agents (`strata cell`, `dag`, `status`, `dep`).**
  The `strata` command grew an agent-facing surface over one shared
  `NotebookOps` core: **inspect** (`cell list/show`, `dag`, `status`),
  **execute** one cell at a time (`cell run` in normal / `--rerun` / `--force`
  mode, `cell test`), and **author** (`cell add/edit/rm/mv`, `cell annotate` to
  splice `# @key` annotations, `dep add/rm`) — all with `--format json` and a
  stable exit-code contract (`0` ok / `1` operation failure / `2` invocation
  error), so an agent can drive a notebook as a first-class tool. Every command
  runs **either offline against a notebook directory or against a live session
  on a running server** via `--server/--session` (the same session a human
  watches in the TUI). See [Notebook CLI](notebook/cli.md) and [Authoring
  Programmatically](notebook/agent-authoring.md).

- **Per-cell unit tests in the notebook.** Every Python code cell gets a
  **Tests** panel (the `🧪` toggle next to Inspect, which doubles as a health
  badge: `✓ 4/4` green, failing red, errored amber, `· stale` when the cell or
  its tests changed since the last run). Write `pytest`-style tests against the
  functions a cell defines — they run as **real pytest** against a re-executed
  copy of the cell with its upstream inputs injected, so assertion rewriting,
  fixtures, parametrize, and marks all work (`def test_x(cell): assert
cell.featurize(cell.trips)…` — `cell.X` is any def or input after the cell
  ran). Test source is a committed `cells/{id}.test.py`; results persist in
  `.strata/runtime.json` and rehydrate on reopen. Driveable over WebSocket
  (`cell_run_tests` → `cell_test_status` / `cell_test_results`). Python cells
  only; `pytest` must be in the notebook's environment (a missing-pytest run
  surfaces an actionable message). The generated-conftest runner is written to
  be liftable to a standalone plugin for CI/pre-commit later. See the
  `pandas_basics` example for a worked set of cell tests.

- **Cell-test tooling auto-provisions.** Running a cell's tests when `pytest`
  isn't in the notebook environment now installs it on demand (a generic
  dev-tool provisioning path) and retries, instead of failing; a failing test's
  captured stdout/stderr is surfaced in the result message. Dev-group
  dependencies are excluded from the cell-provenance environment hash, so adding
  a test tool doesn't invalidate cached cell outputs.

- **Broader value serialization with in-place mutation tracking.** Cell outputs
  now serialize polars, torch, and jax values through a unified Arrow type
  registry (alongside the existing pandas / numpy / pyarrow support). In-place
  mutations — `df.sort_values(inplace=True)`, or mutating a numpy array / dict /
  list / set / torch tensor received from upstream — are detected (statically
  recaptured into the DAG and verified at runtime via a fingerprint registry),
  so provenance stays correct when a cell mutates a value it didn't define.
- **General mutation detection for stateful (ML) workloads.** Runtime mutation
  detection is no longer limited to a hand-written type registry — it falls back
  to a serializer-based fingerprint, so an in-place mutation of *any* serializable
  object (a `torch.nn.Module` trained via `optimizer.step()`, an sklearn
  estimator, a custom class) is caught with no per-library rule. A cell that
  mutates an input in place **but doesn't export it** now warns (downstream would
  otherwise read the pre-mutation value), and `strata run` surfaces those
  warnings in both human and JSON output. Strata also warns when two of a cell's
  outputs **share a mutable object** (the optimizer-over-a-model footgun: stored
  as separate artifacts they decouple downstream). New
  [Stateful objects & value semantics](https://bearing-research.github.io/strata/notebook/concepts/)
  docs cover the one-cell training pattern.

- **Variant sweep mode.** A variant group can now run in **sweep mode**
  (`mode = "sweep"` in `notebook.toml`): instead of only the active variant
  executing, *every* variant of the group runs on each execution and the
  downstream cell receives the group's variable as a `{variant_name: value}`
  dict — for comparing alternatives (models, hyperparameters, prompts) side by
  side in one downstream cell. The default stays **switch mode** (one active
  variant, single value). Per-variant input hashes are grouped into the
  provenance key so caching stays correct across the fan-out. In the UI the
  group renders as a tab strip with a **sweep** badge, a run-all button, and a
  readiness rollup; clicking a tab shows that variant's source while all still
  run. The CLI and WebSocket protocol expose the mode toggle. See the
  `model_variants_sweep` example.

- **Live-mirror: REST/CLI notebook edits stream to spectators.** Cell edits and
  runs driven over REST — e.g. by the agent CLI — are now broadcast to connected
  WebSocket clients, so a human watching in the TUI or the web UI sees an agent's
  changes appear live without a manual refresh.

### Changed

- **Server internals decomposed (gates-first).** `server.py` went from a
  ~6,950-line module — where all ~76 routes hand-wired their own
  mode/auth/tenant/QoS gates, the shape behind the service-mode gate bug the
  0.3.0 registry review caught — to ~3,210 lines, in three layers:
  **typed dependencies** (`strata/api/dependencies.py`: distinct
  `ReadStore`/`WriteStore`/`PersonalModeStore` types + principal/scope/tenant
  gates, so a route can't wire the wrong gate), **services**
  (`strata/services/`: pure, HTTP-free, unit-testable artifact/registry/build
  logic), and **per-domain routers** (`strata/api/routers/`: cache, debug,
  registry, metrics/health, admin, artifacts, names, builds, metadata — nine
  domains). A route-table snapshot test freezes the full HTTP surface
  (path, methods, gate count) so the move is provably behavior-preserving. No
  API changes. The materialize + streaming data plane stays in `server.py` for
  a later cycle (its QoS/ACL coupling needs the same gate extraction first).

- **WebSocket frame payloads are now typed.** Notebook WS payloads were inline
  dicts; the execution/test, cascade, cell-status, environment-job, dag-update,
  impact-preview and profiling-summary frames (`cell_status`, `cell_console`,
  `cell_output_delta`, `cell_iteration_progress`, `cell_test_status`,
  `cell_test_results`, `cascade_prompt`, `cascade_progress`, `environment_job_*`,
  `dag_update`, `impact_preview`, `profiling_summary`) are now `pydantic` models
  validated at the emit boundary, so the protocol is self-describing for non-Vue
  clients. A few low-traffic frames remain inline and follow in later cycles.
  Wire shapes are unchanged (one stale doc field, `cascade_prompt.steps` →
  `cells_to_run`, was corrected to match what's actually sent).

- **Runs on CPython 3.14; notebook server uses the sans-I/O WebSocket backend.**
  The server now drives uvicorn with `ws="websockets-sansio"` and raises the
  uvicorn floor to `>=0.35.0` (the first release with `WebSocketsSansIOProtocol`).
  This replaces uvicorn's deprecated legacy-asyncio WebSocket protocol, whose
  `_drain_helper` trips an `AssertionError` on CPython 3.14 the first time the
  server sends a frame — which broke the notebook WebSocket entirely on 3.14. No
  new dependency (`websockets` was already required); 3.14 is now in the tested
  and classified matrix.

### Fixed

- **QoS slots no longer leak when a request is cancelled.** Two admission paths
  caught `except Exception`, which misses `asyncio.CancelledError` (a
  `BaseException`) — a client disconnect mid-acquire would leak the interactive
  slot, and enough of them could wedge the limiter. Both now release on
  cancellation.

- **QoS admission limiters reconcile with config + adaptive sizing** (#185): the
  interactive/bulk limiter pools now track the configured and adaptively-tuned
  slot counts instead of drifting from them.

- **No more error-level log spam for re-importable module inputs.** A cell that
  uses an `import numpy as np` from an upstream cell would log an error-level
  "artifact still missing" line per module when that binding wasn't materialised
  (e.g. a producer running on a remote `@worker` that doesn't ship module blobs
  back). Module bindings are re-importable by name, so the consuming cell just
  re-imports them — that case now logs at debug; a genuinely missing *data*
  artifact still logs at error.

- **Actionable error when `uv` isn't on PATH.** A headless `strata run` (ssh,
  cron) where uv's install dir (`~/.local/bin`) isn't on PATH failed *every*
  Python cell with a bare `[Errno 2] No such file or directory: 'uv'`. uv is now
  resolved via `shutil.which` with a fallback probe of the standard installer
  dirs (`~/.local/bin`, `~/.cargo/bin`), and if it still can't be found, cells
  fail with an actionable message ("uv not found on PATH. Install uv … or add it
  to PATH …") — matching the existing `Rscript not found` guard.

- **Terminal viewer no longer storms on image-heavy notebooks.** The TUI's
  WebSocket client used the `websockets` default 1 MiB frame cap; a
  `notebook_state` frame carrying base64 plot/image outputs exceeds that, so the
  client rejected the first frame and wedged in a fast reconnect loop
  (`ConnectionClosedError` in the status pill). It now connects with no
  frame-size cap, matching the browser client. The `[tui]` extra also declares
  `Pillow` explicitly, so `uv tool install "strata-notebook[tui]"` launches
  instead of failing with `ModuleNotFoundError: No module named 'PIL'`.

- **GC tracker no longer self-deadlocks.** The GC callback took a non-reentrant
  lock that a GC pause triggered during the callback could re-enter; it's now
  non-blocking, removing a rare hang.

- **Filter values serialize correctly through the client integrations** (#193):
  the filter-value serializer is now wired through the duckdb/pandas/polars
  adapters and the server, so typed filter literals round-trip consistently.

### Internal

- CI hardening: a per-test `pytest-timeout` plus a job backstop convert
  multi-hour hangs into named 3-minute failures; process-global server state
  (including the rate limiter) is reset between tests so the parallel (`xdist`)
  unit-test runs are isolated; flaky wall-clock timing assertions were replaced
  with structural checks; `ty` is scoped to shipped code and the type-check is
  clean including warnings; the notebook WebSocket tests no longer drive
  `TestClient`'s portal (a py3.12/macOS hang).

## 0.3.0 — 2026-06-17

### Security

- **Table ACL is enforced on transform inputs** (service-mode hardening): under
  trusted-proxy auth, `AclEvaluator` previously gated only the direct `scan@v1`
  path, so a principal denied a table could still read it by passing it as an
  input to a transform (`/v1/artifacts/materialize`, `explain-materialize`,
  name-status). The scan path and every table-input resolution now share one
  `_authorize_table_access` gate, so a transform input can't bypass the ACL.
  Personal mode (no auth) is unaffected.

- **Registry authorization hardening** (pre-release security review): the
  registry audit read is now tenant-scoped — a principal sees only its own
  tenant's history (`admin:*` sees the whole store), matching every other
  registry route. Deciding protected-alias changes (approve/reject)
  requires the `admin:registry` scope under trusted-proxy auth, and
  approval enforces separation of duty (the requester cannot self-approve
  without `admin:*`). These close latent cross-tenant-disclosure and
  self-approval gaps before the registry is exposed in multi-tenant
  service mode; personal mode (single operator) is unaffected.

### Added

- **`strata-client`: a slim, independent client distribution.** Using Strata as
  a library no longer means installing the whole server. The new `strata-client`
  package depends only on **httpx + pyarrow** (no pyiceberg / fastapi / duckdb /
  pydantic, no Rust extension) and ships the full client —
  `materialize`/`put`/`fetch`/scan, the registry surface (aliases, tags, names,
  audit), the `Filter` helpers, and the duckdb/pandas/polars/datafusion
  integrations (as extras, e.g. `strata-client[duckdb]`): `pip install
strata-client`, then `from strata_client import StrataClient`. It resolves its
  server URL from `STRATA_SERVER_URL` / `STRATA_HOST` / `STRATA_PORT` /
  `pyproject.toml` with no pydantic. The client and the server (`strata-notebook`)
  are **independent** — they share only the JSON wire protocol, neither depends
  on the other.

  **Breaking (import paths):** the client moved out of the `strata` namespace.
  `from strata.client import StrataClient` → `from strata_client import
StrataClient`; the integrations moved from `strata.integration.*` /
  `strata.duckdb_ext` / `strata.polars_ext` to `strata_client.integration.*`.
  The server keeps `from strata.types import Filter` working (it owns its own
  copy of the dependency-free `Filter` wire types).

- **Registry dashboard in the notebook** (#147–#150): the registry is now a
  first-class UI surface, so promotion and approvals don't have to be code. A
  cell that publishes with a name (`strata.put(model, name="taxi/tip-model")`)
  is stamped with its cell and shows a **promote strip** right below it; the
  bottom drawer gains a **Registry tab** with the pending-approval queue
  (Approve / Reject — the human gate, in the UI), a names table (alias chips,
  latest version, tags, and a `[Promote▾]` champion/candidate menu), and a
  collapsible audit timeline; and a **lineage view** renders
  `model ← features ← scan ← table @ snapshot`. Promote toasts the result
  (`✓ applied` or `⏳ pending` for a protected alias). New reads:
  `GET /v1/notebooks/{sid}/artifacts` and `GET /v1/registry/summary`.
  Personal-mode only — the dashboard hides itself in service mode.

- **Ambient `strata` client in notebook cells** (#146): every locally-executed
  Python cell gets a ready `strata` client in its namespace — no
  `from strata.client import StrataClient` / `StrataClient(base_url=…)` /
  `close()`. It covers the common operations (`materialize`, `put`, `set_alias`,
  `set_tag`, `resolve_alias`, …) over a lightweight stdlib client path-loaded
  into the notebook venv (no new dependency), is created fresh per run and
  closed automatically, and — like a mount or `@table` variable — is an injected
  tool, not a cell input, so it never affects provenance. Local execution only;
  remote-executor cells import a client explicitly.

- **Warm Rscript pool** (#81): notebooks with R cells pre-spawn R workers
  that have already paid interpreter startup, renv activation, and
  `jsonlite`/`arrow` loads — an R cell run skips the ~1–2s cold-start tax
  and reports `execution_method: "warm"`. Single-shot workers preserve
  per-cell isolation; `renv.lock` edits drain and respawn the pool;
  pure-Python notebooks and machines without `Rscript` never start one.
  The pool machinery is the existing Python warm pool with a
  parameterized worker command — the stdin/stdout frame protocol is
  language-agnostic.

- **Live-provider LLM tests** (opt-in): `STRATA_TEST_LIVE_LLM=1` runs
  integration tests against the real Anthropic and OpenAI APIs — unary
  completions, schema enforcement (native tool-use / strict
  `json_schema`), and streaming with usage accounting — catching
  provider contract drift the mocked tests cannot. Each provider class
  skips unless its API key is present; models are overridable via
  `STRATA_TEST_LIVE_{ANTHROPIC,OPENAI}_MODEL`.

- **Structured output degrades gracefully on minimal providers**: some
  OpenAI-compatible servers reject `response_format` or `stream_options`
  with a 400 — schema-constrained prompt cells used to die on the raw
  provider error. Strata now retries once without the extensions,
  steering the model with a schema-guidance system turn and marking the
  result degraded; the client-side validate-and-retry loop carries full
  enforcement. Validation is also lenient about packaging: JSON wrapped
  in code fences or prose is extracted before validating instead of
  burning a retry on the wrapper.

- **Notebook cells can target a remote shared store** (`notebook_remote_store_url`,
  shared research store): the ambient `strata` client injected into cells can now
  point at a central deployment instead of the local notebook server, so a team
  of researchers publishes/consumes datasets against one store.
  `notebook_remote_store_headers` carries the auth the remote store needs (e.g.
  trusted-proxy identity/token) — set via env so secrets stay out of committed
  config. Unset → the ambient client targets the local server as before.

- **Authenticated write-back in service mode** (`service_writes_enabled`, shared
  research store) — **preview**: an opt-in capability letting authenticated clients _publish_
  to a service-mode store — `put`, `set_name`, `set_alias`, tags — so a team can
  share processed datasets through one central deployment. Each write requires
  trusted-proxy auth and the `artifacts:write` scope, lands in the caller's
  tenant (team = tenant; can't target another team), and is attributed to the
  publishing principal in the registry audit. Default is **off** — service mode
  stays read-only unless you enable it, and it requires `auth_mode='trusted_proxy'`
  (enforced at startup) so every write is attributable. Pairs with the
  now-resolvable registry names (above) to make a published dataset "always
  available to the team."

- **Configurable pull-model signing secret** (`STRATA_TRANSFORM_SIGNING_SECRET`):
  the HMAC secret that signs v2-pull build URLs can now be pinned via config
  instead of being a random per-process value. Without it, the secret was
  regenerated on every restart — so in-flight signed download/upload/finalize
  URLs broke on restart and never matched across replicas. Set a stable value for
  any multi-replica or restart-surviving deployment; if `pull_model_enabled` is on
  without it, the server logs a warning at startup.

- **Approval gates on protected aliases**: set
  `STRATA_REGISTRY_PROTECTED_ALIASES=champion,production` and moves or
  deletes of those aliases queue for approval (HTTP 202) instead of
  applying — `POST /v1/registry/pending/approve` applies the change with
  the approver as the audit actor, `…/reject` discards it, and every step
  (request, approval/rejection, the applied move) lands in the registry
  audit. Unprotected aliases are unaffected; the default is no gating.
  SDK: `list_pending_changes` / `approve_alias_change` /
  `reject_alias_change`; CLI: `strata artifact pending`.

- **Registry layer: aliases, tags, and an append-only audit log** (#129):
  promotion is no longer a silent pointer swap. A registry name can hold
  many **aliases** (`taxi/tip-model @ champion`, `@ candidate`) following
  the post-stages industry model; artifact versions carry queryable
  **tags** (`auc=0.91`, `validated_by=…`); and every name/alias/tag
  mutation lands in an immutable **audit** written in the same transaction
  (`who, what, from → to, when`) — including names set by `materialize`
  itself. New SDK methods (`set_alias`, `resolve_alias`, `set_tag`,
  `get_registry_audit`, …), REST routes under `/v1/names/{name}/aliases`
  and `/v1/artifacts/{id}/v/{n}/tags`, and `strata artifact audit [name]`
  which renders the history (`old-id@v1 -> new-id@v1`). Alias refs
  (`name@alias`) are accepted anywhere the artifact CLI takes a reference.

- **Prompt cells stream live** (#111): LLM output renders token-by-token
  on the cell card as the model generates, instead of appearing all at
  once on completion. Schema-validation retries surface as a badge on
  the stream. New `cell_output_delta` WebSocket frame (ephemeral — not
  persisted or replayed; external WS clients that ignore unknown frame
  types are unaffected).
- **Structured streams render as partial JSON** (#113): prompt cells
  with an `@output_schema` show fields popping in as the model finishes
  them — a lenient partial-JSON parser pretty-prints the valid prefix,
  with a character ticker and raw-tail fallback while a field is still
  in flight.
- **`strata validate`** (#115): static notebook checks without executing
  anything — TOML parse (with line numbers), DAG cycle detection, and
  the same per-cell annotation diagnostics the server runs on open.
  `--format json` carries per-cell defines/references. Exit codes mirror
  `strata run`.
- **`strata new`** (#115): scaffold a notebook directory from the CLI
  without the server. Idempotent on existing notebooks — re-running
  never orphans artifacts.
- **Programmatic authoring guide** (#115): `docs/notebook/agent-authoring.md`
  is the contract for scripts and coding agents writing notebooks as
  plain files — the worked example is pinned by the test suite.
- **Per-cell `stdout` / `stderr` in `strata run --format json`** (#116):
  read computed values back from the run payload (truncated at 10k
  chars) instead of screen-scraping.
- **Agent conversation memory survives restarts** (#119): per-notebook
  agent history now persists to `.strata/agent_history.json` (atomic
  writes, 12-turn window, tool traces never persisted) so a server
  restart no longer wipes the conversation. Destructive-tool approval
  prompts also get a configurable timeout
  (`STRATA_AI_APPROVAL_TIMEOUT_SECONDS` / `[ai] approval_timeout_seconds`,
  default 120s; expiry counts as a decline).
- **Lake-aware cells: `@table` annotation**: declare an Iceberg table input
  on a cell (`# @table trips file:///wh#nyc.trips`) and the table's snapshot
  id joins the cell's provenance — new data landing in the lake makes the
  cell stale and the normal cascade re-runs it, with `<name>` (URI) and
  `<name>_snapshot` injected so the cell scans exactly the snapshot its
  provenance recorded. `snapshot=<id>` pins a cell to one snapshot forever.
- **Artifact inspection CLI**: `strata artifact list / show / lineage /
pull` work directly against a local store, no server needed. `lineage`
  renders the provenance chain down to the lake — `model ← features ←
scan ← table @ snapshot` — answering "which snapshot trained this
  model?" in one command. References accept a name, `id@v=N`, or a bare
  artifact id; name resolution is tenant-agnostic so legacy stores
  inspect cleanly.
- **Personal mode executes transforms**: the embedded build runner now runs
  in personal mode, so `materialize` with `duckdb_sql@v1` executes
  server-side out of the box — previously the request was accepted and then
  sat in `building` forever (no mode could run the full
  scan → transform → train → put workflow). Unknown transforms fail fast
  with a 400 listing what's available, and a `name=` on an async
  materialize is now set when the build completes.
- **Artifact store integrity hardening** (#123): artifacts are validated at
  finalize time (the blob must be exactly one readable Arrow IPC stream
  matching the recorded row count — a mismatch becomes a `failed` artifact,
  never a serveable one); `refresh=True` now rebuilds the _same_ artifact as
  a new version and supersedes the old one instead of forking a parallel
  identity the cache never returns; builds stuck in `building` are swept to
  `failed` at startup; and `strata artifact verify` checks a whole store's
  blobs against metadata after the fact.

- **`strata-notebook --notebook-dir`**: control where new notebooks are
  created. They default to `~/.strata/notebooks` — not the directory you
  launched from — so pass `--notebook-dir .` to use the current directory, or
  any path (equivalently, set `STRATA_NOTEBOOK_STORAGE_DIR`). The server now
  also prints the active notebook location on startup.

### Changed

- **Registry name resolution works in service mode** (shared-store groundwork):
  resolving a published dataset by name — `GET /v1/names/{name}`, alias
  resolution, name-status, and tag reads — used to 403 in service mode (gated as
  a write). These are reads, so they're now enabled and tenant-scoped (a team
  resolves its own namespace; cross-team is not found). Registry _writes_
  (`set_name`/`set_alias`/tags) and _listing_ all names stay blocked — those are
  the next step (authenticated write-back).

- **Service-mode config coherence is checked at startup** (hardening): three
  misconfigurations now fail fast instead of silently misbehaving at runtime —
  `multi_tenant_enabled` with `auth_mode='none'` (the tenant header would be
  unauthenticated and spoofable, and reads aren't tenant-filtered without auth, so
  multi-tenancy now requires `auth_mode='trusted_proxy'`), ACL rules with
  `auth_mode='none'` (ACL is only enforced under trusted-proxy auth, so the rules
  would never run), and transforms enabled without an `artifact_dir` (builds
  persist artifacts and need a store).

- **Artifact-mode scan builds are bounded-memory**: the background build for
  `materialize(mode="artifact")` now writes each row-group chunk straight to the
  blob store (write-through) instead of accumulating the whole result in memory
  before persisting. A multi-GB scan no longer holds the full result resident on
  the server. (Part of decoupling the scan build from the client — see _A client
  never poisons a scan artifact_ under Fixed.)

- **Default cell timeout raised from 30 s to 300 s**: the previous default
  was an easy footgun for I/O-bound cells (network pulls, slow APIs), which
  timed out at exactly 30 s unless a `# @timeout` annotation was added. The
  new default matches the core scan timeout; a genuinely hung cell is still
  killed at the wall, and per-cell / per-notebook overrides are unchanged.

### Fixed

- **`materialize(mode="artifact")` without a store fails fast** (service-mode
  hardening): requesting artifact (persisted) mode in a deployment with no
  `artifact_dir` used to return a `build_id` that never resolved — the background
  build silently no-ops with no store to write to, so the client polled forever.
  It now returns `400` up front, pointing at `mode="stream"` (scan without
  persistence) or configuring `artifact_dir`.

- **Materialized results are readable in service mode** (service-mode
  hardening): `GET /v1/artifacts/{id}/v/{n}/data` and the artifact metadata GET
  were gated as writes, so they returned 403 in service mode — an identity-scan
  **cache hit returned a `/data` URL the client couldn't fetch**, and a build
  service couldn't serve its results. Reads are now allowed in service mode,
  gated by tenant (`_ensure_artifact_access`) and the table ACL of the
  artifact's inputs (so a principal denied a table can't read it back via a
  cached scan result — "result retrieval is ACL-gated"). Personal mode is
  unchanged.

- **Ambient cell `strata` client survives large materialize streams** (ML
  dogfood): a cell scanning a big lake table via the injected `strata` client
  could fail with `IncompleteRead` — and leave the artifact `failed` — on a
  _fresh_ multi-row-group scan. The client read the stream in one blocking
  `resp.read()`, which let the server's send buffer fill and tripped its
  `is_disconnected()` check, aborting the stream. The client now drains the
  response in chunks (as httpx does), so large scans complete. Cell execution
  and warm/cached scans were unaffected.

- **A client never poisons a scan artifact** (server-side root fix for the
  above): the `/v1/streams` GET no longer scans-and-persists inside the response
  generator. The build now runs as a decoupled, bounded-memory background task
  and finalizes the artifact on its own merits; the GET waits for it and then
  streams the persisted blob. A slow or dropped reader can no longer abort the
  build or mark the artifact `failed` — a mid-stream disconnect leaves it
  `ready`. The QoS scan slot is released the moment the build completes.

- **Headless `strata run` no longer drops console output on cache hits**
  (ML dogfood): a re-run whose cells hit cache carried no fresh stdout, and
  the empty-console write then _unlinked_ the file the producing run had
  persisted — so `.strata/console/` ended up holding only the cell that
  actually re-executed. Cache hits now leave the persisted console
  untouched, so `print()` output stays recoverable across runs.

- **Registry hardening (pre-release review)**: garbage collection and
  `delete_artifact` now respect **alias** pointers — a champion alias
  pinning an old (even superseded) version protects it from collection,
  and deleting an artifact cleans its aliases (audited) and tags.
  `approve_alias_change` is fully transactional: the pending-consumption,
  approval audit, and the alias move itself commit together, and a
  pending change whose target vanished fails cleanly with the entry
  intact for an explicit reject. Concurrent refresh rebuilds of one
  artifact no longer race version allocation. Alias writes targeting the
  version already pointed at are idempotent no-ops (`status:
"unchanged"`) — re-running a promote cell doesn't refile approvals or
  spam the audit.

- **Namespaced artifact names are no longer write-only** (friction from the
  ML dogfood): names containing `/` (`team/dataset/raw`) could be created
  but every read route 404'd on them. The name routes now use path
  converters, so slash-namespaced names — the natural registry convention —
  resolve, report status, and delete normally.
- **Legacy `_default`-tenant artifacts stay nameable**: artifacts written by
  pre-fix `PUT /v1/artifacts` carry tenant `_default`; single-tenant name
  requests (no tenant) may now point names at them instead of being
  rejected with a tenant mismatch. Real cross-tenant mismatches are still
  rejected.
- **Put-created artifacts can be named after the fact**: `PUT /v1/artifacts`
  stamped artifacts with tenant `_default` while the name routes resolve
  no-header requests to no tenant — so `set_name` on an artifact you had
  just created was rejected with a tenant mismatch. The put route now
  resolves the tenant the same way materialize does.
- **Multi-input transforms bind inputs in caller order**: the stored
  transform spec sorted its inputs "for deterministic hashing", so the
  build runner bound `input0` / `input1` by lexicographic artifact id —
  joins could silently swap their operands depending on generated UUIDs,
  and `f(a, b)` deduplicated against `f(b, a)`. Input order is part of the
  computation now (and of provenance). Existing caches of multi-input
  transforms whose caller order differed from sorted order will rebuild
  once.
- **Multi-row-group scans no longer silently truncate** (#121): scanning
  a table whose plan spans multiple Parquet row groups or files produced
  an Arrow IPC body that standard readers stopped reading after the
  first row group — `materialize` + `fetch` returned ~1M rows from a
  2.9M-row table with no error. Both the streamed response and the
  persisted artifact blob are now a single valid IPC stream, with
  regression tests over a multi-file warehouse.
- **Cross-process lock around renv mutations** (#109): concurrent
  `strata run` invocations and the server no longer race on the same
  notebook's R environment — renv init/install/restore now take a file
  lock on `.strata/renv-process.lock`, and a held lock surfaces as a
  structured failure instead of corrupted state.

## 0.2.0 — 2026-06-03

Second release. Headline: **R cells** alongside Python in the same
notebook with cross-language Arrow handoff — first-class in the UI, with
an R environment panel (one-click renv bootstrap + package install),
automatic `renv::restore()` on open, and inline plot output (ggplot2 /
base graphics render to PNG). Plus run-all batching that amortises
subprocess cost across consecutive Python cells, a 60-second WS reconnect
grace so a flaky network doesn't kill a running execution, real-emulator
integration tests for the S3 / Azure / GCS mount schemes, and versioned
docs via `mike`.

Upgrading from 0.1.0 is non-breaking. The artifact cache stays valid —
`compute_lockfile_hash` was extended to fold `renv.lock` content but
yields byte-identical output for notebooks without one (every existing
Python-only notebook). The WS protocol gained reconnect + MessageType
frames; external WS clients that ignore unknown frame types are
unaffected. No Python API surface removed.

### Added

#### R cells

R is a first-class notebook language alongside Python: cells execute
end-to-end, cross-language Arrow exchange works, provenance/caching is
language-agnostic, and the full UX layer ships in this release — R cells
in the Add-cell menu, an R environment panel with one-click renv
bootstrap + package install, and automatic `renv::restore()` on open.
The example notebook below shows the shape.

- `LanguageExecutor` + `LanguageAnalyzer` protocols + registries under
  `src/strata/notebook/languages/` — generalises the cell-language story
  beyond Python.
- R DAG analyzer (`languages/r/analyze_cell.R`) — defines/references via
  Rscript walking the parsed expression tree; source-hash cache keeps
  re-analysis cheap.
- R harness (`languages/r/harness.R`) — manifest-driven cell-execution
  subprocess. Reads inputs via `arrow::read_ipc_stream`, runs the cell
  body, writes outputs as Arrow IPC (for `data.frame` / tibble), JSON
  (for atomic scalars / lists), or RDS (everything else, tagged
  `r_only=true`).
- `ContentType.RDS_OBJECT = "application/x-r-rds"` + the
  `StrataRArtifactError` exception — Python cells consuming an R-only
  RDS artifact fail with a structured "re-export as `data.frame`" hint
  instead of a `NameError`. Same gating in the batch harness and the
  warm-pool worker.
- `renv.lock` content participates in the env hash via
  `compute_lockfile_hash`, so editing the lockfile invalidates R cells'
  cache the same way `uv.lock` invalidates Python cells'. Backward-
  compatible: notebooks without `renv.lock` see byte-identical hashes.
- `_renv_sync` helper + `[r]` block schema in `notebook.toml`, wired into
  session open: opening a notebook with an `renv.lock` restores the
  project library automatically (the `uv sync` analogue for R).
- R cells in the Add-cell menu with the correct `.R` file extension —
  no more hand-editing `notebook.toml` to add one.
- R environment panel at parity with Python: a stacked R card shows the
  current renv state (System R vs in-sync vs lockfile-edited), a one-click
  **Initialize renv** bootstrap (install renv → bare project library →
  `jsonlite` + `arrow` → snapshot) with live streamed progress, and a
  per-package **Install** action driven off the missing-package hint. R
  environment jobs stream stdout/stderr over the WS and persist a synced
  R runtime (lock hash + timestamp + R version) on success.
- A missing-package error in an R cell surfaces a structured install hint;
  an erroring R cell now marks its READY downstream cells stale instead of
  leaving them green.
- Inline plot output for R cells: base graphics and grid-based plots
  (ggplot2 / lattice) render to PNG and display in the cell like a Python
  matplotlib figure. A bare trailing plot object auto-renders (REPL-style);
  multiple plots in one cell produce ordered displays.
- `# @mount`, `# @env KEY=VAL`, and `# @name` annotations work on R
  cells with no R-specific parser changes — the annotation parser is
  language-agnostic.
- Headless `strata run` executes R cells (previously skipped as an
  unsupported language) and restores the notebook's `renv.lock` into a
  project library first — so R and mixed notebooks run end-to-end from
  the CLI for CI / scheduled jobs, not only through the server.
- New `examples/r_lm_vs_sklearn/` notebook — Python cell builds a
  housing DataFrame, R cell fits `lm(price ~ sqft + bedrooms + age +
location)`, Python cell fits the same model with sklearn and prints
  a side-by-side comparison.
- New `examples/r_mtcars_analysis/` notebook — a pure-R analysis (every
  cell R): `lm()` + `aggregate()` + inline ggplot2 and base-graphics
  plots, showing the R DAG, `data.frame` Arrow handoff, and R-only (RDS)
  object handoff between R cells.
- CI `r-tests` job runs on Ubuntu + macOS via `r-lib/actions/setup-r`
  with the `arrow` + `jsonlite` packages installed from posit/RSPM
  binaries, against both R `release` and `oldrel-1`. The cross-language
  suite (`tests/notebook/test_r_cells.py`) exercises Py→R→Py Arrow
  round-trip, R-only RDS refusal, mount injection, error shapes, cache
  hit/miss, `renv.lock` change invalidation, inline plot capture, real
  `renv::restore`, and env-annotation injection; a separate `r-examples`
  job runs the R example notebooks end-to-end via `strata run`.

#### Run-all batching

Consecutive Python cells share a single harness subprocess on
`run all` / `rerun all`, amortising the ~150ms cold-start across the
batch. R cells are still single-cell (Phase 2). Mixed notebooks
partition into per-language runs automatically.

- `harness.execute_batch` library entry point + `--batch` CLI flag —
  one subprocess executes a sequence of cells against a shared
  namespace, communicating cache-check / persist requests with the
  parent over JSON-line pipes.
- `CellExecutor.execute_batch` orchestration with per-cell timeout
  watchdog inside the batch subprocess (a hung cell can't take down
  the whole run).
- `is_cell_batchable` gate keeps the partitioner conservative —
  prompts, SQL, R cells, and any cell with `# @worker` / `# @mount
rw` opt out automatically.

#### Reconnect resilience

- 60-second WS reconnect grace before the server tears down a session's
  execution state — a Wi-Fi blip mid-cell no longer kills the run.
- `MessageType` StrEnum extracted to `protocol.py` — single canonical
  source for every C↔S frame name, removes string-literal drift across
  the codebase + the docs.
- New `docs/reference/notebook-protocol.md` — the full client-author
  reference (bootstrap, auth model, reconnect grace, cold-start payload,
  every message type) so external clients can target the WS protocol
  without reading server code.
- `notebook.toml` write path preserves TOML datetime values and the
  `array-of-tables` shape, so a saved-and-reopened notebook produces a
  byte-identical TOML for unchanged sections.

#### Mount integration tests

- S3 mount tests against MinIO via `testcontainers`.
- Azure mount tests against the Azurite emulator.
- GCS mount tests against `fake-gcs-server`.
- The notebook-side mount-credentials hook (`MountResolver`) gets
  exercised against all three, so credential resolution + path
  normalisation regressions surface in CI rather than at first remote
  upload.

#### Rerun cells

- `↻` button + Cmd+Shift+Enter rerun a single cell bypassing its cache
  (and rerunning stale upstreams).
- `notebook_rerun_all` WS message + UI "Rerun all" entry — cascade with
  cache disabled, useful when you've changed something the provenance
  hash can't see (a non-deterministic data source, an outside-the-
  notebook file the cell reads, etc.).

#### Versioned docs

- Documentation site is now version-aware via `mike`. Visit
  `https://bearing-research.github.io/strata/` — the version dropdown
  in the header lets readers pick `latest` (always the current
  release) or a pinned version (`0.2.0`, `0.1.0`, ...). Pre-release
  preview lives under `dev`.

### Changed

- Docs site builds + deploys via `mike` instead of `mkdocs gh-deploy`.
  PRs that touch `docs/` still validate `--strict` without touching
  `gh-pages`; main pushes update the `dev` alias; release tags pin
  a versioned snapshot + the `latest` alias.
- `create_notebook`'s `pyproject.toml` shape is built from metadata
  rather than templated as a string — adding a default dep is now a
  one-line list edit instead of a multi-place template change.
- `MountResolver` derives its TOML on-disk shape from
  `MountSpec.model_dump()` rather than a hand-rolled mapping, so
  schema changes only touch one place.
- `test_routes.py` + `test_ws.py` boilerplate collapses into shared
  helpers / fixtures — net subtraction in the test suite, fewer places
  to drift on protocol changes.
- README's Highlights section calls out R cells, DAG view, loop cells,
  prompt-cell variable resolution, and auto-install hints. The buried
  feature list under "Quick Start" is gone.

### Fixed

- `strata run` without `--no-sync` no longer fails at environment sync
  with "env sync finished without a status snapshot". The headless runner
  read the session's currently-running-job slot (reset to `None` on
  completion) instead of the returned job's terminal status, so the
  default invocation aborted on every notebook before running a cell.
- `notebook.toml` TOML datetime values and `array-of-tables` rows no
  longer churn on a round-trip save (#45). Pre-fix, saving a notebook
  with no edits would rewrite datetime fields as strings + collapse
  array-of-tables into inline tables, polluting git diffs.
- `/{session_id}/...` routes are now owner-gated in personal mode with
  `STRATA_PERSONAL_MODE_USER_HEADER` set (#41). Pre-fix, a request
  with the proxy-supplied user header could read another user's session
  state via `GET /v1/notebooks/{id}/cells`.
- Three correctness gaps in the run-all dispatcher and three in the
  batch dispatcher, caught by review (#34, #35, #36).
- The 1-element identifier collapse in the R analyzer's JSON emit
  (`auto_unbox = TRUE` was eating single-name vectors). Wrapped
  `defines` / `references` in `jsonlite::I()`.
- The R analyzer walker mis-attributing reads under in-place mutations
  (`df <- df[complete.cases(df), ]` correctly keeps `df` in
  references).
- A Python-only artifact (a `pickle` value or a `module/*` content type)
  consumed by an R cell now fails with a structured error naming the
  variable and the re-export fix, instead of aborting the R subprocess
  and surfacing a generic "Rscript exited without producing a result
  manifest" — symmetric with the existing R-only (RDS) → Python guard
  (#107).
- A Python numpy array / non-tabular scalar read into an R cell now
  warns that the value is flattened into a `data.frame` (the Arrow shape
  metadata can't round-trip into R) instead of changing shape silently
  (#107).
- macOS / Linux + Python 3.14 SQLite I/O flake in cache_warm tests gets
  one retry (`tests/conftest.py`); Iceberg `temp_warehouse` fixture
  disposes its catalog engine before yield to close the related flake.

### Security

- **Phase A scorecard hygiene** — `SECURITY.md`, Dependabot config,
  least-privilege workflow permissions.
- **Phase C SHA pinning** — every GitHub Action across every workflow
  pinned to a 40-char commit SHA with the version annotation in a
  trailing comment. Docker base images (the shipped image + the
  df-cluster example) are pinned by `sha256` digest. A Dependabot
  `docker` ecosystem keeps both digests and Action SHAs current via
  weekly group PRs.
- **Token-permission least privilege** — `docs.yml` and `release.yml`
  default to read-only, escalating to `contents: write` only in the
  single job that pushes the rendered site (mike → gh-pages) or creates
  the GitHub Release.
- WS upgrade owner-gating closes the cross-session-read path noted
  above.

### Compatibility

- **Cache:** non-breaking for Python-only notebooks. `compute_lockfile_hash`
  was extended to fold `renv.lock`, but the extension is a no-op when
  the file is absent (every Python-only notebook gets byte-identical
  hash output).
- **WS protocol:** the new `MessageType` extraction is purely a
  refactor — frame strings are unchanged. The new reconnect-grace
  - per-cell-watchdog frames are additive; clients that ignore unknown
    frames continue to work.
- **REST API:** unchanged.
- **Wheel ABI:** still `abi3-py312` (one wheel per platform covers 3.12+).
- **Python deps:** no breaking changes; R support is fully optional
  (Python-only users don't need R installed).

## 0.1.0 — 2026-05-20

First stable release of Strata Notebook. The package is published on
PyPI as `strata-notebook`; the Python module is imported as `strata`.
Wheels ship for Linux (x86_64, aarch64), macOS (x86_64, arm64), and
Windows (x86_64) and are abi3-compatible from Python 3.12 through 3.14.

Strata refuses to start outside a uv-managed Python environment;
`uv tool install strata-notebook` is the canonical install path,
with `uv add strata-notebook` for project-style installs. `pip
install` into a hand-rolled `python -m venv` is rejected by the
startup guard. The notebook app boots via `strata-notebook` (or
`python -m strata`); `strata-worker` boots a remote worker; and
`strata run | export | import` covers headless notebook tooling.

### Added

#### Notebook UI and lifecycle

- notebook home / create / open flows with recent-notebook tracking
- notebook rename, delete, duplicate, and management improvements
- per-notebook Python environments (managed by `uv`) with status, sync,
  import / export, and async environment jobs
- Python-version selection in the new-notebook flow
- inline cell display outputs: PNG images, markdown, `display(...)` side
  effects, `plt.show()` / `Figure.show()`, ordered multiple visible outputs
  per cell
- markdown cells for prose / documentation
- timing instrumentation and a browser benchmark for create / open flows

#### SQL cells

- SQL cell language with `# @sql connection=<name>` annotation, named-bind
  parameters resolved from upstream cells, and an Arrow-IPC artifact
  produced per query
- per-driver `DriverAdapter` Protocol with capability flags (per-table
  freshness, snapshot support, separate probe connection requirement)
- five built-in driver adapters:
  - **PostgreSQL** via ADBC, freshness via `pg_stat_user_tables`
  - **SQLite** via ADBC, freshness via `PRAGMA data_version` /
    `schema_version`, read-only via URI `mode=ro` plus `PRAGMA query_only`
  - **Snowflake** via ADBC, URI-as-identity, runtime schema resolution,
    `write_role` for read / write principal split
  - **BigQuery** via ADBC, credentials principal in identity, ambient-ADC
    sentinel, notebook-relative credential paths, `write_credentials_path`
    for read / write principal split
  - **DuckDB** (embedded) via the native DuckDB DBAPI, layered RO
    enforcement (file flag + cursor-level `BEGIN TRANSACTION READ ONLY`)
- write cells via `# @sql write=true`, with per-statement status tables
- `# @cache fingerprint | forever | session | ttl=N | snapshot` policies
- `# @after <cell>` ordering-only DAG annotation
- Connections panel + REST API for managing `[connections.<name>]` blocks,
  with literal auth values blanked on disk during the write round-trip
- schema-discovery sidebar enumerating tables and columns visible through
  each connection
- `sql_orders_report` example notebook demonstrating a five-cell SQL pipeline

#### Module export and cross-cell library code

- cells that mix runtime work and library code (defs, classes, literal
  constants) can now share the library code across cells; the planner
  slices the cell's AST, keeps the shareable parts, and validates the
  slice with `symtable` so each kept def / class is self-contained
- `module_export_blocked` diagnostic surfaces pre-flight on cell open and
  names the specific function and unresolved variable
- `from __future__ import annotations` correctly relaxes cross-cell
  type-hint references (PEP 563 stringifies annotations, so the
  free-variable check drops them)
- module-level globals written from inside a function are detected
- comprehension elements walk with loop targets locally scoped
- Python 3.14 / PEP 749 deferred-annotation behavior: annotation references
  go through an explicit AST walk so the cross-cell check is consistent
  across `symtable`'s version-dependent free-variable reporting
- `library_cells` example notebook walking through cross-cell library code

#### Deployment

- local service-mode demo stack, smoke script, and deployment guide
- Fly-hosted notebook defaults use persistent notebook storage and a
  larger auto-extending volume configuration
- Docker builds reuse uv and Cargo caches more effectively for faster
  local iteration

#### Release infrastructure

- `pip install strata-notebook` / `uv add strata-notebook` (the bare
  `strata` name on PyPI was held by an unrelated config framework)
- wheel ships the frontend SPA bundled at `strata/_frontend/`, so
  `strata-notebook` works out of the box without a separate frontend build
- abi3-py312 wheel format — one wheel per platform covers Python 3.12+
- tag-driven release workflow with TestPyPI auto-publish and
  PyPI publish gated by a protected GitHub Environment
- post-build wheel smoke test (`wheel-test` job) installs the Linux
  x86_64 wheel into a fresh uv venv, exercises console scripts and
  `/health`, asserts the bundled SPA is served at `/`, and runs the
  matrix against Python 3.12 / 3.13 / 3.14 — packaging bugs fail
  the run before the artifact reaches the index

### Changed

- markdown rendering uses `markdown-it` + `DOMPurify` rather than a
  hand-rolled renderer, with consistent output between in-place cell
  preview and `Markdown(...)` display outputs
- docs split into separate Strata Core and Strata Notebook quickstarts;
  the root README is an umbrella landing page
- notebook create bootstraps the initial environment asynchronously,
  making first open substantially faster
- notebook open / create flows reuse prefetched state and lazy-load
  secondary panels to reduce perceived latency
- add-cell UI replaces per-type buttons with a unified menu
- write-cell status table preserves per-statement rowcounts and is no
  longer truncated to a default cap
- connection-editor UI fixed for round-trip fidelity (auth blanking,
  driver-specific extras, theme correctness) and dark-mode parity

### Fixed

- service-mode session discovery / reconnect policy and related UX
  regressions
- reconnect metadata loss for remote execution state
- run-all only executing the first cell
- missing-package install UX in the cell output area
- local service-mode browser routing and notebook creation flow
- relative connection paths now resolve against the notebook directory,
  not the server CWD
- recent-notebook list is server-validated on home-page load so deleted
  notebooks no longer land the user on a "session not found" toast; a
  Clear button next to the section title wipes the local list without
  touching on-disk notebooks
- `update_notebook_connections` is now idempotent when no `[connections]`
  block exists and the request is empty — saves with no change no longer
  churn `updated_at` or rewrite the on-disk TOML shape (invariant 6)
- timing-based perf assertion in `test_concurrent_scans_dont_block_each_other`
  replaced with a structural correctness check (no more CI flakes from
  runner load)

### Security

- read-only enforcement for SQL cells is layered (file-handle flag +
  session-level guard) rather than SQL-text keyword filtering — a SQL
  cell cannot write to the database regardless of how the connection
  was specified
- BigQuery / Snowflake adapters route reads and writes through different
  principals when configured (`write_credentials_path`, `write_role`),
  with `read_only` kwarg on `canonicalize_connection_id` so changing the
  write principal does not invalidate read-cell caches

## 0.1.0a2 — 2026-05-20

Third release-validation dry-run. Four changes from `0.1.0a1`:

- **Wheel smoke-test job** added to the release workflow. After the
  five wheel matrix jobs finish, a new `wheel-test` job downloads the
  Linux x86_64 wheel, installs it into a fresh uv-managed venv, and
  exercises import + console scripts (`strata`, `strata-worker`) +
  server boot + `/health` + the served SPA at `/`. The TestPyPI
  publish job now depends on `wheel-test`, so a packaging bug fails
  the CI run before the artifact reaches the index. Catches the
  class of bug we hit on `0.1.0a0` (missing `packaging` dep would
  have been caught locally instead of in the smoke test we ran
  after the publish failed).
- **`GET /` assertion in the smoke test.** `server.py::_mount_frontend()`
  silently skips mounting the SPA when `src/strata/_frontend/index.html`
  is absent, so a wheel without the bundle would still pass `/health`.
  The smoke now also fetches `/` and asserts the response is the SPA
  index (grep for `<!doctype html`).
- **`abi3-py312` forward-compat matrix** on `wheel-test`. Same wheel
  is installed and smoke-tested against Python 3.12, 3.13, and 3.14
  via a job-level matrix. The release contract is "one wheel per
  platform covers 3.12+"; this validates it against every minor uv
  knows about.
- **`workflow_dispatch` recovery now checks out the tagged ref.**
  Previously the manual-rerun path checked out whatever branch the
  user dispatched from — if `main` had moved since the tag, the
  rebuilt wheels would have the tagged version label but `main`'s
  source. Now every checkout uses
  `${{ inputs.version }}` → `v${inputs.version}` for dispatch,
  falling back to `github.ref` for the tag-push path.

This alpha will **approve the PyPI gate** (unlike `a0` / `a1` which
rejected it) to validate the PyPI trusted-publisher config + the
GitHub Release creation job before claiming the stable `0.1.0` slot.

## 0.1.0a1 — 2026-05-19

Second release-validation dry-run. `0.1.0a0` uploaded all 5 platform
wheels to TestPyPI successfully but the sdist was rejected with
HTTP 400 ("License-File LICENSE does not exist in distribution
file") — maturin's sdist is built via `cargo package` rooted at
`rust/` and didn't pick up `LICENSE` and `README.md` from the repo
root. Added both to `[tool.maturin] include` with `format = ["sdist"]`
so they land in the archive matching the PEP 639 metadata.

The pipeline never published to PyPI on `0.1.0a0` because the
TestPyPI failure short-circuited the run. `0.1.0a1` is the retry
with the fix; no other changes from `0.1.0a0`.

## 0.1.0a0 — 2026-05-19

Release-validation dry-run. The first tagged release in the project's
history; exercises the full publish pipeline (multi-platform wheel
matrix, TestPyPI auto-publish, manually-gated PyPI publish) before
the stable 0.1.0 cut. The wheel content is identical to what 0.1.0
will ship; only the version label differs. Anyone installing
`strata-notebook==0.1.0a0` from PyPI will get a working install with
the feature surface planned for 0.1.0 (described above); the alpha
label exists so the version slot can be discarded if the dry-run
surfaces any release-pipeline bugs.

The first stable release is still planned as 0.1.0. See the section
above for the feature inventory; this dry-run aims to validate that
the inventory ships correctly.
