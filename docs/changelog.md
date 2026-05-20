# Changelog

All notable changes to Strata will be documented in this file.

The project is still in alpha. Entries here focus on user-visible changes and
release framing rather than exhaustive commit history.

The authoritative copy of this file lives at [`CHANGELOG.md`](https://github.com/bearing-research/strata/blob/main/CHANGELOG.md) in the repo root; this docs page mirrors it. Maintainers: keep the two in sync when editing.

## Unreleased

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
the feature surface planned for 0.1.0 (described below); the alpha
label exists so the version slot can be discarded if the dry-run
surfaces any release-pipeline bugs.

The first stable release is still planned as 0.1.0. See the section
below for the feature inventory; this dry-run aims to validate that
the inventory ships correctly.

## Planned for 0.1.0

First public release of Strata Notebook — in flight, **not yet
published**. The package will be published on PyPI as `strata-notebook`
once the release ships; the Python module is imported as `strata`.
Wheels will ship for Linux (x86_64, aarch64), macOS (x86_64, arm64),
and Windows (x86_64) and will be abi3-compatible from Python 3.12 onward.

Strata refuses to start outside a uv-managed Python environment;
`uv tool install strata-notebook` will be the canonical install path
once 0.1.0 ships. Until then, install from a git checkout
(`uv sync` in a cloned repo).

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
  `strata-server` works out of the box without a separate frontend build
- abi3-py312 wheel format — one wheel per platform covers Python 3.12+
- tag-driven release workflow with TestPyPI auto-publish and
  PyPI publish gated by a protected GitHub Environment

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
