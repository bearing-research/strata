# Environment Management

Each notebook has its own isolated Python environment managed by [uv](https://docs.astral.sh/uv/).

## How It Works

When you create a notebook, Strata:

1. Generates a `pyproject.toml` seeded with the packages a cell needs to hand values to the next one: `pyarrow`, `orjson` and `cloudpickle`
2. Runs `uv sync` to create a `.venv/` and `uv.lock`
3. All cell execution uses this notebook-local venv

Opening a notebook runs the same `uv sync` (and restores `renv.lock`, when the notebook has one) as an environment job, which keeps the server responsive to other requests and tabs while it runs; the open answers once the job finishes, so cells can run straight away. The job appears in the Environment panel's history. Reopening a notebook that is already open skips the job unless its `.venv` is missing or its `renv.lock` changed.

A hash of the lockfiles (`uv.lock`, plus `renv.lock` when the notebook has one) participates in provenance, so changing the environment invalidates all cached cell outputs. When `uv.lock` has a dev group, only the runtime dependency closure is hashed, so adding a dev tool such as `pytest` leaves the cache alone.

A cell sent to a [remote worker](workers.md) runs in this same locked environment when the worker supports it, so the lock describes what the cell ran against there too.

## Python Version

At notebook creation time, you can select a Python version from the versions configured on the server. The first one listed is the default.

To change it later, click **Requested Python** in the Environment panel. Strata rewrites `requires-python`, rebuilds `.venv/` with `uv sync`, and restores the old version if the sync fails.

!!! note
    The available versions come from the server's `STRATA_NOTEBOOK_PYTHON_VERSIONS`. Unset, they are the Python versions `uv` reports as installed on the server that Strata supports, with the server's own version included.

## Installing Packages

### From the UI

Open the **Environment** panel in the sidebar. Type a package name and click **Add**.

```
pandas>=2.0
scikit-learn
matplotlib
```

The operation runs asynchronously, you can continue editing cells while it installs.

On a server in service mode, packages install from wheels only: a package with no wheel for the notebook's Python fails to resolve, and the message says a wheel is required. R packages are not added from the notebook there, nor on a personal-mode server with `STRATA_NOTEBOOK_HARNESS_USER` set: the server restores the notebook's committed `renv.lock` as the harness user when the notebook opens. See [Service mode](../deployment/service-mode.md).

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

Conda-style `environment.yaml` files are supported on a best-effort basis. Strata translates each conda spec to a pip requirement (`numpy=2.1` becomes `numpy==2.1`, a `conda-forge::` prefix is dropped) and adds the entries of the `pip:` list as they are. It ignores `channels`, the `python` pin and the `pip` entry itself, and the preview lists what it ignored.

### Export

Click **Export** to download the current dependencies as `requirements.txt`.

## Environment Operations

All environment mutations run as **async jobs** with seven actions:

| Action | Description |
|--------|-------------|
| `add` | Install a new package |
| `remove` | Remove a package |
| `sync` | Rebuild the environment from `pyproject.toml` |
| `import` | Bulk import from requirements.txt or environment.yaml |
| `change_python` | Switch the notebook's Python version |
| `r_init` | Set up renv for R cells |
| `r_add` | Install an R package with renv |

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
lockfile instead. They live under `STRATA_NOTEBOOK_SHARED_ENV_DIR`, by default
`envs` beside the notebook storage directory (`~/.strata/envs` unless
`STRATA_NOTEBOOK_STORAGE_DIR` is set):

```
~/.strata/envs/
├── 3f9c…/            # one environment per uv.lock + interpreter build
├── 3f9c….lock        # held while that environment is installed
└── refs/3f9c…/       # one file per notebook linked to it
my_notebook/
└── .venv -> ~/.strata/envs/3f9c…
```

- **The key** is a SHA-256 of `uv.lock` together with the exact interpreter
  build and platform. It leaves out the notebook project's own name, version
  and declared version ranges, the only part of the lock that differs between
  two notebooks with the same dependencies. Everything that
  changes what is installed counts: every resolved package with its version,
  source and hashes, the markers and extras on the notebook's dependencies, and
  `requires-python`. So two notebooks with the same resolved dependencies on the
  same interpreter get the same environment, whatever they are called, and the
  second one's sync is only the link.
- **The notebook's own project is not installed** into a shared environment
  (`uv sync --no-install-project`), since other notebooks link to it. A
  notebook created by `strata new` has no build system, so uv never installs it
  anyway. If you add one, its package is not importable from a shared
  environment.
- **A shared environment is never changed in place.** Adding or removing a
  package updates that notebook's `pyproject.toml` and `uv.lock` without
  syncing, then syncs, which lands in another environment and moves only that
  notebook's link. The other notebooks keep theirs.
- **One install at a time per environment.** A notebook that needs an
  environment another notebook is still installing (an import opened straight
  away on a new server, say) waits for that install in its own environment
  job, then only links. The server keeps answering other requests meanwhile.
- **Clean-up.** An environment no notebook links to is removed once it has
  gone unused for `STRATA_NOTEBOOK_SHARED_ENV_TTL_DAYS` (default 7), by an
  hourly sweep in the server or by `strata env gc`. An environment a notebook
  links to is never removed, opening a notebook counts as using it, and the
  sweep removes only environments it built. A directory of your own under the
  store, or one whose build never finished, is left alone.

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
