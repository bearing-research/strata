# Environment Management

Each notebook has its own isolated Python environment managed by [uv](https://docs.astral.sh/uv/).

## How It Works

When you create a notebook, Strata:

1. Generates a `pyproject.toml` with `pyarrow` as the default dependency
2. Runs `uv sync` to create a `.venv/` and `uv.lock`
3. All cell execution uses this notebook-local venv

The environment lockfile hash (`sha256(uv.lock)`) participates in provenance. Changing the environment invalidates all cached cell outputs.

## Python Version

At notebook creation time, you can select a Python version from the versions configured on the server. The default is the server's own Python version.

!!! note
    The available versions depend on the server's `STRATA_NOTEBOOK_PYTHON_VERSIONS` configuration. On the hosted preview, both 3.12 and 3.13 are available.

## Installing Packages

### From the UI

Open the **Environment** panel in the sidebar. Type a package name and click **Add**.

```
pandas>=2.0
scikit-learn
matplotlib
```

The operation runs asynchronously, you can continue editing cells while it installs.

### Import from requirements.txt

In the Environment panel, click **Import** and paste a `requirements.txt`:

```
pandas>=2.0
numpy>=2.0
scikit-learn>=1.5
matplotlib>=3.9
seaborn>=0.13
```

Strata previews the changes (additions, removals, unchanged) before applying.

### Import from environment.yaml

Conda-style `environment.yaml` files are supported on a best-effort basis. Strata extracts `pip` dependencies and ignores conda-specific packages.

### Export

Click **Export** to download the current dependencies as `requirements.txt`.

## Environment Operations

All environment mutations run as **async jobs** with four actions:

| Action | Description |
|--------|-------------|
| `add` | Install a new package |
| `remove` | Remove a package |
| `sync` | Rebuild the environment from `pyproject.toml` |
| `import` | Bulk import from requirements.txt or environment.yaml |

The UI shows:

- Current job status (running, success, failed)
- Recent operation history (persisted across server restarts)
- Resolved package count and lockfile hash

## Cache Invalidation

When you install or remove a package:

1. `uv sync` runs to update `uv.lock`
2. The lockfile hash changes
3. All cells become **stale** (their provenance no longer matches)
4. Re-running any cell recomputes with the new environment

!!! info "Switching back"
    If you remove a package and then re-add it (returning to the same `uv.lock`), the original provenance hashes match again and cells get **cache hits**. This is free by construction, no special logic needed.

## Missing Package Detection

When a cell fails with `ModuleNotFoundError`, Strata detects the missing package and offers a one-click install button:

```
ModuleNotFoundError: No module named 'pandas'
→ [Install pandas]
```

## File Layout

```
my_notebook/
├── pyproject.toml    # Package declarations
├── uv.lock           # Locked dependency graph
└── .venv/            # Virtual environment (auto-created, not committed)
```

The `pyproject.toml` and `uv.lock` are the source of truth. The `.venv/` is recreated by `uv sync` when needed.

## Shared environments

On a server with many notebooks built from the same few lockfiles, a `.venv`
per notebook installs the same packages again and again. With
`STRATA_NOTEBOOK_ENV_BACKEND=shared`, notebooks share one environment per
lockfile instead:

```
~/.strata/envs/
├── 3f9c…/            # one environment per uv.lock + interpreter build
├── 3f9c….lock        # held while that environment is installed
└── refs/3f9c…/       # one file per notebook linked to it
my_notebook/
└── .venv -> ~/.strata/envs/3f9c…
```

- **The key** is the SHA-256 of the raw `uv.lock` bytes together with the exact
  interpreter build and platform. Two notebooks with the same lock on the same
  interpreter get the same environment, and the second one's sync is only the
  link.
- **A shared environment is never changed in place.** Adding or removing a
  package updates that notebook's `pyproject.toml` and `uv.lock` without
  syncing, then syncs, which lands in another environment and moves only that
  notebook's link. The other notebooks keep theirs.
- **Clean-up.** An environment no notebook links to is removed once it has
  gone unused for `STRATA_NOTEBOOK_SHARED_ENV_TTL_DAYS` (default 7), by an
  hourly sweep in the server or by `strata env gc`. An environment a notebook
  links to is never removed.

R libraries are shared the same way. A notebook with an `renv.lock` restores
it once per server into `r/` in the same store, keyed by the raw `renv.lock`
bytes and the exact R build, and its `renv/library` is a link there:

```
~/.strata/envs/
├── r/
│   ├── 8a41…/        # one library per renv.lock + R build
│   ├── cache/        # renv's package cache (RENV_PATHS_CACHE)
│   └── refs/8a41…/
my_notebook/
└── renv/library -> ~/.strata/envs/r/8a41…
```

A second notebook with the same `renv.lock` links to the library without
running `renv::restore()`, and every Rscript (cells, the warm pool, the
environment panel) reads it through renv's project path. Installing a package
from the environment panel first moves the notebook onto a private library
restored from the package cache, and once `renv.lock` is written that library
is kept under the new lock's key, so the other notebooks keep theirs. The R and
Python keys are separate: changing one lock rebuilds only that language's
environment. The same sweep removes libraries no notebook links to.

Provenance still follows the lockfiles, as with a `.venv` and `renv/library`
per notebook. POSIX only, since the link is a symlink.
