# Operations & Lifecycle

This page covers the operational disk-and-state story: where notebook data lives, how big it gets, how to back it up, how to move it, and how to clean up.

If you're configuring caps and tuning, see [Configuration](../reference/configuration.md). If you're shipping to production, see [Deployment Modes](modes.md).

## Where everything lives

A Strata deployment has three persistent locations:

| Location | Default | Contents | When to back up |
| --- | --- | --- | --- |
| **Notebook storage** | `~/.strata/notebooks/` | One subdirectory per notebook: `notebook.toml`, `cells/*.py`, `pyproject.toml`, `uv.lock`, `.strata/` (per-notebook runtime), `.venv/` (per-notebook venv) | Always — this is your work |
| **Iceberg row-group cache** | `~/.strata/cache/` | Arrow-IPC files keyed by Parquet row-group, plus `meta.sqlite` (or `STRATA_METADATA_DB` if set) | Optional — purely a perf cache, safe to delete |
| **Server-side artifact store** | `~/.strata/artifacts/` (or `STRATA_ARTIFACT_DIR`) | The Core SDK's artifact blobs + metadata SQLite. **Distinct from the per-notebook `.strata/artifacts/`** below. | Only if you use `StrataClient.materialize` or named artifact pointers you care about |

Inside each notebook directory:

```
mynotebook/
├── notebook.toml             # committed config — schema: notebook-toml.md
├── pyproject.toml            # uv-managed deps for this notebook
├── uv.lock                   # pinned versions
├── cells/                    # one .py per cell — committed source
└── .strata/                  # runtime state — gitignored
    ├── runtime.json          # display outputs, provenance hashes, env metadata
    ├── console/              # per-cell stdout/stderr (one JSON per cell)
    └── artifacts/            # SQLite + blobs (cached cell outputs)
└── .venv/                    # uv-materialized venv — gitignored, host-specific
```

The `.strata/artifacts/` directory is the **per-notebook** artifact store. It grows as cells produce outputs and is the thing that makes "re-run an unchanged cell" instant.

## Backup

A notebook is its committed files. To back one up, archive everything **except** `.strata/` and `.venv/`:

```bash
cd ~/.strata/notebooks
tar --exclude='.strata' --exclude='.venv' -czf mynotebook.tar.gz mynotebook/
```

The excluded directories are runtime state (regenerable) and a host-specific venv (rebuildable with `uv sync`). Skipping them keeps the backup small (typical: tens of KB instead of hundreds of MB).

If you'd rather not exclude `.strata/`, you can include it for a "warm restore" — cached cell outputs survive the trip and downstream cells stay green on the destination. Just expect the archive to be larger.

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

**What doesn't transfer.** Mounted external storage (`s3://`, `gs://`) is referenced by URI, so cells that use mounts work on any machine with the right credentials. Mounts with `file://` URIs pointing at machine-local paths obviously don't.

## Deleting a notebook

Three options, depending on the surface:

| From | How | Effect |
| --- | --- | --- |
| **UI** | "Delete notebook" in the notebook menu | Removes the directory and closes the open session. Confirm prompt. |
| **REST** | `DELETE /v1/notebooks/{session_id}` for an open session, or `POST /v1/notebooks/delete-by-path` for a path-based delete (personal mode only) | Same as the UI |
| **Filesystem** | `rm -rf ~/.strata/notebooks/mynotebook` while the server isn't running | Same outcome, no graceful session close |

Deleting a notebook also deletes its `.strata/artifacts/` — there's no shared artifact store across notebooks, so nothing leaks.

## Cleaning up the Core artifact store

The **server-side** artifact store (driven by `StrataClient.materialize`) accumulates blobs that may no longer be referenced by any name pointer. There's no automatic cap on it; you GC manually:

```bash
curl -X POST 'http://localhost:8765/v1/artifacts/gc?max_age_days=7'
```

Or from Python:

```python
from strata_client import StrataClient
client = StrataClient(base_url="http://localhost:8765")
client.garbage_collect(max_age_days=7.0)
# {"deleted": 14, "bytes_freed": 8429283, ...}
```

The GC pass:

- Walks the metadata SQLite for artifacts older than `max_age_days`
- Filters to "unreferenced" — no `[name]` pointer references them
- Deletes only artifacts in `ready` or `failed` state (in-flight artifacts are safe)
- Returns counts + bytes freed

Personal mode only — service-mode deployments need the `admin:cache` scope and should usually GC per-tenant.

GC the **per-notebook** artifact store by deleting the notebook (or by deleting `.strata/artifacts/` while the server isn't running). There's no per-notebook GC endpoint — cell-output artifacts are content-addressed and pruning them would defeat the cache.

## Cleaning up the Iceberg row-group cache

```bash
curl -X POST 'http://localhost:8765/v1/cache/clear'
```

Clears the in-memory + on-disk Iceberg cache. Personal mode is unrestricted; service mode requires the `admin:cache` scope. Safe to run at any time — the worst case is the next read repopulates from Parquet.

## Disk-usage budgeting

This is the part most people get bitten by. There are **two** caps to understand and they don't cover everything.

| Knob | Default | What it caps | What it doesn't cap |
| --- | --- | --- | --- |
| `STRATA_MAX_CACHE_SIZE_BYTES` | 10 GB | The Iceberg row-group cache (`~/.strata/cache/`) — LRU-evicted to stay under the cap | Anything else |
| Artifact GC `max_age_days` | (no automatic run) | The Core artifact store, by age, when you run GC | The notebook-scoped artifact stores |

Things with **no built-in size limit**:

- `~/.strata/notebooks/*/​.strata/artifacts/` — per-notebook artifact stores. Grow with each cell run that produces new outputs (cache hits don't add bytes; only new provenance hashes do).
- `~/.strata/notebooks/*/​.venv/` — per-notebook venvs. Grow with each `uv add`; the heaviest notebooks (torch + cuda) can run to several GB each. Use shared system packages or smaller deps if disk is tight.
- The server's `~/.strata/artifacts/` until you GC it.

Practical guidance:

- Run `POST /v1/artifacts/gc` weekly (or on a cron) if you use the Core SDK.
- The Iceberg cache is self-managing under its byte cap — leave it.
- If a single notebook's `.strata/artifacts/` gets uncomfortably large, the cleanest reset is to delete the notebook's `.strata/` directory while the server isn't running. Cell source survives; provenance cache resets.
- For `.venv/` sprawl: `du -sh ~/.strata/notebooks/*/.venv` is the quickest audit. Old notebooks you don't open anymore can have their `.venv/` deleted — `uv sync` will recreate it next time.

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
    volume at `/tmp/strata-notebooks` in Docker — see the Docker page
    for that pattern).

For multi-user deployments, see `STRATA_PERSONAL_MODE_USER_HEADER` in [Configuration](../reference/configuration.md#notebook) — it scopes each user to their own subdirectory under the storage root.
