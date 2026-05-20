# Import

`strata import` converts a Jupyter `.ipynb` file into a Strata
notebook directory. The result is a normal Strata notebook, same
DAG analysis, same artifact store, same execution model, that you
can open in the UI, run headlessly, edit, or export.

## What survives, what doesn't

Read this before you decide to import a particular notebook.

**What survives:**

- **Code and markdown cells**, in source order, including variable
  rebinding patterns like `df = df.dropna()` (Strata's DAG handles
  read-before-write correctly).
- **`;`-suppression** of the trailing display (Strata's harness
  auto-displays bare expressions; the converter detects `df;` and
  appends a `pass` so the display is skipped).
- **`%pip install`, `!pip install`, `%conda install`** — packages
  captured into the new notebook's `pyproject.toml`.
- **`%env`, `%set_env`** — translated to `# @env KEY=VAL` cell
  annotations.
- **`%run script.py`** — translated to `exec(Path(...).read_text())`.
- **`%%bash`, `%%sh`, `%%script`** — bodies wrapped in
  `subprocess.run(..., shell=True)`.
- **`%%writefile`** — translated to `Path(...).write_text(...)`.
- **`%timeit`, `%time` (line form)** — the magic prefix is dropped;
  the body keeps running.

**What you lose:**

- **All cell outputs.** Imported cells start blank; first run
  produces fresh outputs from the resolved environment. This is
  intentional — the whole point of content-addressed caching is to
  produce outputs from a known source + env, not to trust whatever
  was last serialized into the `.ipynb`.
- **Widgets** (`ipywidgets`, `tqdm.notebook`, custom display).
  Strata's display protocol covers DataFrames, plots, markdown,
  images — but interactive JS widgets don't survive.
- **`%matplotlib inline` / `notebook`.** Dropped. Strata captures
  figures via the display protocol automatically.
- **Raw cells.** Skipped entirely (counted in the import report).
- **`!shell` commands** other than `pip install`. Dropped with a
  marker comment; assignment-form (`var = !cmd`) drops the command
  and stubs `var = []` so downstream code still parses. Restore
  manually with `subprocess.run(...)` if the escape was load-bearing.
- **Non-Python `%%` cell magics** (`%%R`, `%%ruby`, `%%javascript`,
  `%%html`, `%%latex`, `%%svg`, `%%markdown`). Dropped.
- **`%%sql`.** Dropped — convert by hand to a Strata
  [SQL cell](cells.md#sql-cells) if you want it back.
- **REPL-inspection magics** (`%who`, `%whos`, `%lsmagic`, `%history`,
  `%alias`, etc.). Dropped; no Strata equivalent.

Everything that's "dropped" is reported in the import report
(`<notebook_dir>/import_report.md`) with the exact line number and
the marker comment placed in the cell source — there's no silent
data loss.

## When to use it

- **You arrived from Jupyter** and want to try Strata on a notebook
  you already have. One command and you're working in Strata's model
  (content-addressed caching, explicit DAG, distributed workers,
  prompt cells, …) without rewriting from scratch.
- **You're picking up someone else's analysis** that lives as an
  `.ipynb` on GitHub or Kaggle. Import it once, iterate normally.
- **You're stress-testing Strata** against real-world notebooks:
  the import + corpus-runner combo is the validation harness Strata
  itself uses pre-release.

The import is **one-shot**, not a live sync. The `.ipynb` is treated
as input; the resulting notebook directory is the source of truth
from then on. There's no "save back to ipynb", that's a separate
[planned feature](#round-trip-back-to-ipynb).

## Usage

```bash
strata import <path/to/notebook.ipynb> [options]
```

### Options

| Flag         | Description                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------------- |
| `--out <dir>` | Target notebook directory. Defaults to a sibling directory named after the `.ipynb` stem. |

### Examples

```bash
# Side-by-side default: ./my_analysis.ipynb → ./my_analysis/
strata import my_analysis.ipynb

# Place the result elsewhere
strata import ~/Downloads/kaggle_titanic.ipynb --out ~/work/titanic

# Then open or run normally
strata-notebook --notebook-storage-dir ~/work     # UI
strata run ~/work/titanic                       # headless
```

## REST

```
POST /v1/notebooks/import
Content-Type: multipart/form-data
```

| Field         | Type        | Description                                                                  |
| ------------- | ----------- | ---------------------------------------------------------------------------- |
| `file`        | file (required) | The `.ipynb` upload. Hard cap of 50 MB per request.                       |
| `name`        | string      | Override the notebook name. Defaults to the upload's filename stem.          |
| `parent_path` | string      | Override the storage location. Must lie inside the configured storage root. |

Returns the same notebook state shape as `POST /v1/notebooks/create`
plus an `import_report` field with the conversion details. On
malformed input, invalid JSON, non-nbformat structure, path
traversal in `name`, collision with an existing notebook, returns a
clean `400` or `409` with `detail` set.

## What the converter does

### Cell-by-cell, in source order

| Jupyter cell | Strata cell                                                  |
| ------------ | ------------------------------------------------------------ |
| Markdown     | Markdown cell with the source verbatim                       |
| Code         | Python cell (after magic translation, see below)             |
| Raw          | Skipped; counted in the report                               |

Variable rebinding (`df = df.dropna()`, `df = df[df.col > 0]`, …)
is a first-class pattern. Strata's DAG analyser handles read-before-
write semantics correctly: the cell appears as both a producer
*and* a consumer of `df`, so the upstream edge is drawn and downstream
cells see the post-mutation view.

### `;`-display-suppression preserved

`df;` in Jupyter evaluates `df` but suppresses the auto-displayed
value. Strata's harness auto-displays the last bare expression too,
so the converter detects the trailing `;` (with or without an
adjacent comment) and appends a `pass` so the harness skips display.
The cell still runs.

### Magic translation table

Single dict in `strata.notebook.jupyter_import`. Adding a row
extends support; rows can be:

| Magic | Action |
|---|---|
| `%matplotlib inline`, `%matplotlib notebook` | Dropped. Strata captures figures via the display protocol. |
| `%load_ext`, `%reload_ext`, `%autoreload`, `%config`, `%colors`, `%rerun` | Dropped. |
| `%capture`, `%xmode`, `%pdb`, `%debug`, `%tb` | Dropped. |
| `%who`, `%who_ls`, `%whos`, `%lsmagic`, `%magic`, `%history`, `%alias`, `%alias_magic` | Dropped (interactive REPL inspection has no Strata equivalent). |
| `%timeit`, `%time` (line form) | Magic stripped, body kept. |
| `%pip install <pkgs>`, `!pip install <pkgs>`, `%conda install <pkgs>` | Packages captured into `pyproject.toml`. |
| `%env KEY=VAL`, `%set_env KEY=VAL` | Translated to a `# @env KEY=VAL` cell annotation. |
| `%run script.py` | Translated to an `exec(Path("script.py").read_text())` (with a self-contained `Path` import). |
| `%%bash`, `%%sh`, `%%script` | Body wrapped in `subprocess.run(..., shell=True)`. |
| `%%writefile <path>`, `%%file <path>` | Translated to `Path(<path>).write_text(<body>)`. |
| `%%timeit`, `%%time`, `%%capture` | Recurse on body as plain code. |
| `%%javascript`, `%%js`, `%%html`, `%%latex`, `%%svg`, `%%markdown` | Dropped. Strata has no equivalent renderer. |
| `%%R`, `%%ruby`, `%%perl`, `%%cython`, `%%fortran`, `%%sql` | Dropped (non-Python languages and embedded SQL aren't auto-translatable; use Strata's SQL cell type for `%%sql` content). |
| Anything unrecognized | Dropped with a `# strata: unsupported magic '<name>' dropped` comment at the original location. |

### `!shell` commands

Auto-running shell from an untrusted notebook is a real hazard
(stress-testing public Kaggle notebooks is a primary use case), so
the converter is conservative:

- **`!pip install pkg1 pkg2`**: packages captured to deps,
  source line removed.
- **`var = !cmd`** (assignment-form shell escape) command dropped,
  `var` stub-bound to `[]` so downstream code still parses. Restore
  manually with `subprocess.run(...)` if the shell escape matters.
- **Other `!cmd`**: dropped with a marker comment. Surfaces in the
  import report's *Shell commands dropped* section.

### Dependencies

Four sources, in priority order, earlier sources shadow later ones,
so a version-pinned spec from `requirements.txt` wins over a bare
inferred-from-imports entry:

1. Sibling `requirements.txt` next to the `.ipynb`.
2. Sibling `pyproject.toml` next to the `.ipynb`.
3. `%pip install` / `!pip install` lines extracted from cells.
4. **Bare imports in cell source.** AST-walk each cell, collect
   top-level import names, filter stdlib (via `sys.stdlib_module_names`)
   and local sibling modules (anything that resolves to a `*.py` or
   `*/__init__.py` next to the notebook). Map import names to PyPI
   names via a small hand-maintained dict for common mismatches:
   `cv2 → opencv-python`, `sklearn → scikit-learn`, `PIL → Pillow`,
   `bs4 → beautifulsoup4`, `yaml → PyYAML`, etc. Anything not in the
   dict is assumed to use the same name on PyPI (right ~95% of the
   time).

The combined set is deduped with PEP 503-normalized package names
(so `scikit-learn` and `scikit_learn` collapse to one entry) and
filtered to **PEP 508 specifiers only**: `pyproject.toml`'s
`dependencies` won't accept editable installs (`-e .`), bare URLs
(`git+https://…`), or local paths. Skipped specs land in the import
report so you can address them by hand.

The deps are written to the new notebook's `pyproject.toml`. First
`uv sync` (which runs automatically when you open the notebook in
the UI, or when you invoke `strata run`) resolves them. The
importer doesn't call `uv add` itself, that's slow, networked, and
partial-failure-prone.

## The import report

Every import writes `<notebook_dir>/import_report.md` and the same
content is returned inline on the REST response (`import_report.report_text`).

Sections appear only when relevant, a clean notebook with no magics,
no shell, and no deps produces a short report with just the counts.
Sections that surface when applicable:

- **Counts**: markdown / code cells, `;`-suppression instances,
  skipped cell types.
- **Magics translated**: list of every magic the converter rewrote
  or absorbed.
- **Magics dropped**: magics with no Strata equivalent; the source
  carries a marker comment where each one used to live.
- **Shell commands dropped**: `!cmd` lines (except `!pip install`)
  that the converter removed.
- **Dependencies captured**: what landed in `pyproject.toml`,
  ready for `uv sync` to resolve.
- **Warnings**: anything noteworthy; e.g. pip-only specs skipped
  for not being valid PEP 508.

## Limitations by design

- **No output preservation.** Imported cells start blank. Running
  the notebook through Strata produces fresh outputs from a known
  environment, which is exactly the signal the
  [corpus-runner stress test](#stress-testing-with-the-corpus-runner)
  is built to validate. Pre-loaded author outputs would mask
  execution incompatibilities.
- **No widgets.** `application/vnd.jupyter.widget-view+json`
  outputs and `%%javascript` cells are dropped with a warning.
  Strata's display protocol doesn't model interactive widgets.
- **No round-trip back to `.ipynb`.** Tracked as a follow-up; out
  of scope for the import path itself.
- **Kaggle hard-coded paths.** Notebooks that reference
  `/kaggle/input/...` directly will fail on first run outside
  Kaggle. The path lives unchanged in the imported source; surface
  it in the report or rewrite by hand.

## Stress-testing with the corpus runner

The same `import_notebook` machinery the CLI and REST endpoint use
backs the corpus runner under `tests/notebook/test_jupyter_corpus.py`.
Five hand-crafted smoke fixtures get scored through:

```
parse:    can we read the .ipynb at all?
convert:  did jupyter_import produce a valid Strata notebook dir?
dag:      does the DAG build with no cycles / unbound references?
run:      does `strata run` complete with no exceptions?
artifact: do leaf cells produce non-empty artifacts?
```

The fast tier (`parse + convert + dag`) runs on every PR. The full
tier (`+ run + artifact`) is opt-in:

```bash
STRATA_CORPUS_RUN=1 uv run pytest tests/notebook/test_jupyter_corpus.py
```

The runner does a `uv sync` per fixture and a real `strata run`,
so it's slow (network) and intended for nightly schedules or
manual pre-release verification.

### Extended corpus + nightly CI

`tests/notebook/jupyter_corpus/extended.yaml` lists public `.ipynb`
URLs pinned to specific commit SHAs. The `jupyter-corpus` GitHub
Actions workflow fetches each one nightly (04:00 UTC) and runs it
through the same rubric. Failures are report-only, the workflow
uploads a junit XML artifact, doesn't gate other CI. The cache is
keyed by the manifest's content hash so once a URL has been
fetched it's served from disk on subsequent runs.

Anyone can trigger the workflow from the Actions tab via
`workflow_dispatch`. Adding a new entry: pick a stable repo, pin to
a commit SHA, run `STRATA_CORPUS_RUN=1 pytest -k <name>` locally to
verify the entry clears the bar you set, commit the manifest line.

## Round-trip back to `.ipynb`

Out of scope today. Tracked as a follow-up because preserving the
import-lossy parts (Jupyter outputs, widget metadata, original
magic text) would compromise the import itself. See
`docs/internal/design-jupyter-import.md` if you're picking this up.
