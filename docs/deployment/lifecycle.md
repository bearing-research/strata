# Operations & Lifecycle

This page covers the operational disk-and-state story: where notebook data lives, how big it gets, how to back it up, how to move it, and how to clean up.

If you're configuring caps and tuning, see [Configuration](../reference/configuration.md). If you're shipping to production, see [Deployment Modes](modes.md).

## Where everything lives

A Strata deployment has three persistent locations:

| Location | Default | Contents | When to back up |
| --- | --- | --- | --- |
| **Notebook storage** | `~/.strata/notebooks/` | One subdirectory per notebook: `notebook.toml`, `cells/*.py`, `pyproject.toml`, `uv.lock`, `.strata/` (per-notebook runtime), `.venv/` (per-notebook venv) | Always - this is your work |
| **Iceberg row-group cache** | `~/.strata/cache/` | Arrow-IPC files keyed by Parquet row-group. The Parquet/Iceberg metadata cache sits beside it at `~/.strata/meta.sqlite` (or `STRATA_METADATA_DB` if set) | Optional - purely a perf cache, safe to delete |
| **Server-side artifact store** | `~/.strata/artifacts/` (or `STRATA_ARTIFACT_DIR`) | The Core SDK's artifact blobs + metadata SQLite (the metadata moves to Postgres with `STRATA_ARTIFACT_METADATA_DSN`, and `strata migrate` copies an existing SQLite store across). **Distinct from the per-notebook `.strata/artifacts/`** below. | If you use `StrataClient.materialize`, named artifact pointers, publications or pins you care about, or API keys (they live in the same database) |

Inside each notebook directory:

```
mynotebook/
├── notebook.toml             # committed config - schema: notebook-toml.md
├── pyproject.toml            # uv-managed deps for this notebook
├── uv.lock                   # pinned versions
├── cells/                    # one .py per cell - committed source
└── .strata/                  # runtime state - gitignored
    ├── runtime.json          # display outputs, provenance hashes, env metadata
    ├── console/              # per-cell stdout/stderr (one JSON per cell)
    └── artifacts/            # SQLite + blobs (cached cell outputs)
└── .venv/                    # uv-materialized venv - gitignored, host-specific
```

The `.strata/artifacts/` directory is the **per-notebook** artifact store. It grows as cells produce outputs and is the thing that makes "re-run an unchanged cell" instant.

## Backup

A notebook is its committed files. To back one up, archive everything **except** `.strata/` and `.venv/`:

```bash
cd ~/.strata/notebooks
tar --exclude='.strata' --exclude='.venv' -czf mynotebook.tar.gz mynotebook/
```

The excluded directories are runtime state (regenerable) and a host-specific venv (rebuildable with `uv sync`). Skipping them keeps the backup small (typical: tens of KB instead of hundreds of MB).

If you'd rather not exclude `.strata/`, you can include it for a "warm restore" - cached cell outputs survive the trip and downstream cells stay green on the destination. Just expect the archive to be larger.

Copying `.strata/` while a cell is finishing can capture the new `runtime.json` with the old artifacts. To copy a notebook the server has open, hold it still first:

```bash
curl -X POST 'http://localhost:8765/v1/notebooks/<session_id>/quiesce'
# ...copy the directory...
curl -X POST 'http://localhost:8765/v1/notebooks/<session_id>/release'
```

Quiesce waits for running cells (cancelling any still running after `timeout_seconds`, default 30), then refuses runs and edits with a 409 until release or `max_hold_seconds` (default 600). `POST /v1/projects/{path}/quiesce` and `.../release` do the same for every notebook under a directory, open or not. Under principal auth both need the `admin:notebooks` scope.

## Moving between machines

Same idea: copy the notebook directory minus `.venv/`. Optionally minus `.strata/` if you want a clean cache.

```bash
# On source
tar --exclude='.venv' -czf mynotebook.tar.gz ~/.strata/notebooks/mynotebook/

# On destination
mkdir -p ~/.strata/notebooks
tar xzf mynotebook.tar.gz -C ~/.strata/notebooks/
cd ~/.strata/notebooks/mynotebook
uv sync       # rebuilds .venv from pyproject.toml + uv.lock
```

The `uv.lock` ensures the rebuilt venv pins identical versions to the source machine. The Rust toolchain on the destination needs to match Strata's source requirements only if you're upgrading Strata at the same time; for an existing wheel install it's not needed.

`strata export <dir> --to snapshot --include all --out <file>.zip` packs the committed files, runtime state and every artifact into one zip, and `strata import <file>.zip` unpacks it on the other side. See [Snapshots](../notebook/export.md#snapshots).

**What doesn't transfer.** Mounted external storage (`s3://`, `gs://`) is referenced by URI, so cells that use mounts work on any machine with the right credentials. Mounts with `file://` URIs pointing at machine-local paths don't.

## Deleting a notebook

Three options, depending on the surface:

| From | How | Effect |
| --- | --- | --- |
| **UI** | "Delete notebook" in the notebook menu | Removes the directory and closes the open session. Confirm prompt. |
| **REST** | `DELETE /v1/notebooks/{session_id}` for an open session, or `POST /v1/notebooks/delete-by-path` for a path-based delete. Both are personal mode only. | Same as the UI |
| **Filesystem** | `rm -rf ~/.strata/notebooks/mynotebook` while the server isn't running | Same outcome, no graceful session close |

Deleting a notebook also deletes its local store, `.strata/artifacts/`. Copies that already left it stay where they went: results you [published](../notebook/publishing.md) live in the server's store, and results promoted or offered to a [team store](service-mode.md#the-team-cache-sharing-results-nobody-named) live there.

## Cleaning up the Core artifact store

The **server-side** artifact store (driven by `StrataClient.materialize`) keeps
every distinct result it computes, so it grows with every new query. In
personal mode it looks after itself: every hour the server collects what
nothing needs, least recently used first:

- anything unused for 30 days;
- and, when the store is over 20 GiB, the least recently used until it is down
  to 80% of that.

Service mode does this only when an operator sets
`STRATA_ARTIFACT_GC_INTERVAL_SECONDS`. Every limit is a setting; see
[Configuration](../reference/configuration.md#artifact-storage).

A version is collected only when nothing holds it:

- nothing named, aliased, pinned, published, awaiting alias approval or still
  building is, or was built from, it. A named result keeps its whole chain, so
  its lineage stays walkable and a refresh can re-read its inputs;
- it is not the current value of an id somebody chose, nor something one was
  built from. A notebook stores each cell output under its own id and reads the
  latest version back, so that version stays. An id the store made up for one
  `materialize` has no such value;
- it has not been used in the last hour: finishing its build, a cache hit, a
  read of its data, or a request that names it as an input all count.

So **an unnamed result is a cache entry**. Its URI keeps working while it is
used, and once it is collected the same request computes it again. To keep a
result regardless, name it (`name=` on `materialize`) or pin it.

Run a sweep yourself, or preview one, from the command line (no server
needed) or over HTTP:

```bash
strata artifact gc --dry-run              # what the configured retention would take
strata artifact gc --max-bytes 5G         # bring the store under 5 GiB now
curl -X POST 'http://localhost:8765/v1/artifacts/gc?dry_run=true'
```

Or from Python:

```python
from strata_client import StrataClient

client = StrataClient(base_url="http://localhost:8765")
client.garbage_collect(max_idle_days=7)
# {"deleted_count": 14, "deleted_bytes": 8429283, "store_bytes": 51239012, "dry_run": false}
```

Each limit you leave out takes the configured one. On the command line and the
route a limit of `0` means "everything past the recent-use floor", unlike the
settings, where `0` turns the limit off. `collect_latest=true`
(`--collect-latest`) also collects the current value of caller-chosen ids,
which deletes live notebook state; use it only on a store you are deliberately
reclaiming. In service mode the route needs a principal holding `admin:*`, and
collects within the caller's tenant.

A publication (withdrawn ones included) or a pin protects its whole lineage, not
only the version. A page or a snapshot needs every step behind the result.

### Pins

A pin holds a version and its chain for a reason the store has no other way to
know, such as a snapshot that must stay restorable or a review that's still
open:

```bash
curl -X POST 'http://localhost:8765/v1/artifacts/figure/v/3/pin' \
  -H 'Content-Type: application/json' -d '{"reason": "snapshot:s1"}'
curl -X DELETE 'http://localhost:8765/v1/artifacts/figure/v/3/pin?reason=snapshot:s1'
```

There is one pin per reason: two holders use two reasons and release them
independently, and pinning again under the same reason only refreshes it. In
service mode pins need the `artifacts:pin` scope (or `admin:*`) and are scoped to
the caller's tenant.

A **notebook's own** artifact store (`.strata/artifacts/`) keeps each cell
output's current value plus its three most recent earlier values, so reverting
a recent edit is still a cache hit. Older values are pruned in the background
each time the server opens the notebook, including a reopen of a notebook that
is already open, and a value a crash left half-written for over an hour is
marked failed then. Set `STRATA_NOTEBOOK_KEEP_SUPERSEDED_VERSIONS`
to keep more, or to `0` to keep every value. Deleting the notebook deletes its
store.

Every sweep, of either store, also removes the temporary files of blob writes
that died part way (a killed process), once they have gone untouched for an
hour.

## Cleaning up the Iceberg row-group cache

```bash
curl -X POST 'http://localhost:8765/v1/cache/clear'
```

Clears the in-memory + on-disk Iceberg cache. Personal mode is unrestricted; service mode requires the `admin:cache` scope. Safe to run at any time - the worst case is the next read repopulates from Parquet.

## Disk-usage budgeting

There are **two** caps to understand, and they don't cover everything.

| Knob | Default | What it caps | What it doesn't cap |
| --- | --- | --- | --- |
| `STRATA_MAX_CACHE_SIZE_BYTES` | 10 GB | The Iceberg row-group cache (`~/.strata/cache/`) - LRU-evicted to stay under the cap | Anything else |
| `STRATA_ARTIFACT_GC_MAX_BYTES` | 20 GiB in personal mode, off in service mode | The Core artifact store (`~/.strata/artifacts/`), least recently used first, on the hourly sweep | The notebook-scoped artifact stores, and anything named, pinned or published |

Things with **no built-in size limit**:

- `~/.strata/notebooks/*/​.strata/artifacts/` - per-notebook artifact stores. Each cell output keeps its current value and a few earlier ones (`STRATA_NOTEBOOK_KEEP_SUPERSEDED_VERSIONS`), so a store grows with the number and size of a notebook's outputs rather than with every run.
- `~/.strata/notebooks/*/​.venv/` - per-notebook venvs. Grow with each `uv add`; the heaviest notebooks (torch + cuda) can run to several GB each. Use shared system packages or smaller deps if disk is tight.

Practical guidance:

- In personal mode the Core store is capped out of the box; lower `STRATA_ARTIFACT_GC_MAX_BYTES` if 20 GiB is too much for the disk. In service mode set `STRATA_ARTIFACT_GC_INTERVAL_SECONDS` and a limit, or run `POST /v1/artifacts/gc` on a cron.
- The Iceberg cache is self-managing under its byte cap - leave it.
- If a single notebook's `.strata/artifacts/` gets uncomfortably large, the cleanest reset is to delete the notebook's `.strata/` directory while the server isn't running. Cell source survives; provenance cache resets.
- For `.venv/` sprawl: `du -sh ~/.strata/notebooks/*/.venv` is the quickest audit. Old notebooks you don't open anymore can have their `.venv/` deleted - `uv sync` will recreate it next time. With `STRATA_NOTEBOOK_ENV_BACKEND=shared`, notebooks with the same lockfile share one environment instead; the server removes shared environments nothing links to after `STRATA_NOTEBOOK_SHARED_ENV_TTL_DAYS` (default 7), and `strata env gc` does it on demand. See [Shared environments](../notebook/environment.md#shared-environments).
- `GET /v1/artifacts/usage` reports the Core store's version counts and bytes. In service mode it reports the caller's tenant (`admin:*` can name one with `?tenant=`).

## Notebook storage location

The notebook storage root is controlled by `STRATA_NOTEBOOK_STORAGE_DIR`. The default is `~/.strata/notebooks/` (matches the `~/.strata/` convention for cache + artifacts).

!!! info "Upgrading from a pre-2026-05 install?"
    Earlier Strata versions defaulted to `/tmp/strata-notebooks`, which
    most Linux distros wipe on reboot. If your notebooks are there,
    move them once:

    ```bash
    mkdir -p ~/.strata
    mv /tmp/strata-notebooks ~/.strata/notebooks
    ```

    or set `STRATA_NOTEBOOK_STORAGE_DIR=/tmp/strata-notebooks` if you
    intentionally want the legacy path (e.g. you're already mounting a
    volume at `/tmp/strata-notebooks` in Docker - see the Docker page
    for that pattern).

For multi-user deployments, see `STRATA_PERSONAL_MODE_USER_HEADER` in [Configuration](../reference/configuration.md#notebook) - it scopes each user to their own subdirectory under the storage root.
