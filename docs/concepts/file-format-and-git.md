# File format and git

## Notebook file format

Each notebook is a directory on disk:

```
my_notebook/
├── notebook.toml          # Stable config: ID, name, cell list, workers, mounts, env, ai
├── pyproject.toml         # Python dependencies (uv-managed)
├── uv.lock                # Locked dependencies
├── cells/
│   ├── a1b2c3d4.py        # Cell source files
│   └── e5f6g7h8.py
├── work/                  # Gitignored: where cells run and write their files
└── .strata/               # Gitignored, runtime state, not committed
    ├── runtime.json           # Display outputs, provenance hashes, env metadata
    ├── console/               # Per-cell stdout/stderr ({cell_id}.json)
    ├── fetch/                 # The bytes each @fetch last read
    ├── tables/index.json      # The tables each cell's last good run read
    ├── reads/index.json       # The bytes each cell's last good run moved, per input
    ├── writes/index.json      # The files each cell's last good run wrote through rw mounts
    └── artifacts/
        ├── artifacts.sqlite   # Artifact metadata
        └── blobs/             # Serialized cell outputs
```

`notebook.toml` holds stable config you'd commit to git; `.strata/` holds runtime state that changes on every execution (display outputs, console snapshots, the last `uv sync` timestamp, per-cell provenance hashes). `notebook.toml`'s `updated_at` tracks only structural edits (adding/removing cells, changing workers or mounts), so example notebooks don't churn under version control.

`notebook.toml` defines the notebook identity and cell ordering:

```toml
notebook_id = "f7bd9094-..."
name = "my_analysis"

[[cells]]
id = "a1b2c3d4"
file = "a1b2c3d4.py"
language = "python"
order = 0

[[cells]]
id = "e5f6g7h8"
file = "e5f6g7h8.py"
language = "python"
order = 1
```

## Version control

Strata notebooks are designed to live in git. Unlike Jupyter `.ipynb`
files, which co-mingle source, outputs, and execution counts in one
JSON blob and produce a multi-kilobyte diff every time a cell is
re-run, a Strata notebook is just a directory of plain text:

- **Cells are plain text files.** Python, prompt and SQL cells are `.py`,
  R cells `.r`, markdown cells `.md` and widget cells `.widget`. Normal
  `git diff`, `git blame`, code review on a pull request, syntax highlighting in every IDE. Reordering a cell
  edits one number in `notebook.toml`, not a giant JSON re-serialize.
- **`notebook.toml` is the manifest.** Stable config only: cell list,
  workers, mounts, env, AI defaults, and the active variant per group
  (see [Variant Cells](../guides/cells/loops-and-variants.md#variant-cells)). Reviewers see
  exactly what changed about the notebook's *shape*, not its execution
  history.
- **`.strata/` is gitignored.** Display outputs, console snapshots, the
  `uv sync` timestamp, per-cell provenance hashes, the artifact store:
  none of it touches commits. Re-running a cell never changes the tracked
  tree.
- **`updated_at` only bumps on structural edits.** Adding/removing a
  cell, changing a worker, mounting a path, picking a different variant:
  those bump the timestamp. Editing source or running cells does not.
- **Secrets stay off disk.** Env keys matching `KEY`/`SECRET`/`TOKEN`/
  `PASSWORD`/`CREDENTIAL` are blanked before persisting, so the writer
  can't accidentally commit an API key. The name survives only when the
  block also holds a non-sensitive value; an `[env]` with nothing but
  blanked secrets is not written at all. An empty value for such a key is
  a placeholder, not an override: until you re-enter it, cells see the
  server's own value for that variable.
- **uv lockfile in committed config.** `pyproject.toml` + `uv.lock` pin
  the Python environment exactly the same way the rest of your repo
  does; collaborators get a reproducible environment from a fresh clone.

Put together: a Strata notebook commit shows *what changed about the
work*, not *what happened during the last run*.
