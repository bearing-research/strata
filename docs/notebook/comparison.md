# vs Jupyter, Marimo, Pluto

Strata is closest in spirit to the new generation of reactive notebooks
(Marimo, Pluto.jl) - it shares the "your DAG comes from your variable
references" idea. Where Strata steps further is in turning every cell output
into a content-addressed artifact and treating remote compute and AI calls
as first-class cell behaviors rather than escape hatches.

## Capability matrix

| Capability | Strata | Marimo | Pluto.jl | Jupyter |
|---|---|---|---|---|
| File format | Per-cell `.py` files + `notebook.toml` manifest | Single `.py` per notebook | Single `.jl` per notebook | JSON `.ipynb` |
| Git-friendly diffs | Per-cell, no embedded outputs or execution counts | Single-file but text | Single-file but text | Outputs + base64 images + execution counts embedded in the same file |
| Automatic DAG from variable references | Yes | Yes | Yes | No |
| Persistent cell-output cache | **Automatic**, content-addressed per cell, survives restarts | **Opt-in** via `mo.cache` / `mo.lru_cache` / `mo.persistent_cache` decorators (or context managers) | None, Pluto guarantees the program state is described by the visible code, no hidden cache between sessions | None |
| Distributed / remote execution | `# @worker gpu-fly` annotation dispatches a single cell to a registered worker | Via external orchestration (e.g. SkyPilot recipe); no per-cell remote annotation | Single-process | Single-process per kernel |
| First-class AI/LLM cells | Prompt cells participate in the DAG and cache by template + inputs + model config | Marimo bills itself as an "AI-native editor": cell-level code generation, inline autocompletion, and a `marimo pair` agent skill for collaborative coding (added in v0.22.5). All editor-side; an LLM call inside a cell is just a regular Python expression. | No | No |
| Built-in SQL cells | Yes (named connections, schema discovery, snapshot-aware caching) | Yes (built-in SQL engine) | Community library | Community extensions |
| Loop / iteration cells | Yes (`# @loop max_iter=N carry=var`), checkpointed per iteration | No | No | No |
| Variant cells (tabbed alternatives sharing a DAG slot) | Yes | No | No | No |
| Per-notebook Python environment | Separate `pyproject.toml` + `uv.lock` per notebook | PEP 723 inline script metadata at the top of the `.py` file (`# /// script` block); `marimo edit --sandbox` provisions the venv via uv (or pip/poetry/pixi/rye) | Julia project / Project.toml | Manual (venv / conda / kernel spec) |
| Headless / CI runner | `strata run` (executes the cascade in topological order) | Notebooks runnable as `python file.py` | None first-class - `.jl` file works ad-hoc via `julia notebook.jl`, otherwise via PlutoUtils.jl | `nbconvert --execute` |

## Where Strata is distinctive

**Caching is automatic, not opt-in.** Marimo offers persistent caching
through `mo.persistent_cache` (decorator or context manager), `mo.cache`,
and `mo.lru_cache` - the user explicitly delimits a block of code they
want cached. In Strata, every cell's output is content-addressed by
default: the provenance hash of source + upstream artifact hashes +
environment lockfile decides cache identity, and a cache hit is the path
of zero work. Re-running a notebook nobody's touched costs milliseconds.

**Remote compute is a one-line annotation.** Marimo can be run on a remote
host (SkyPilot integration, SSH port-forwarding), but the granularity is
the whole notebook process. Strata's `# @worker gpu-fly` annotation routes
a single cell, fitting one classifier on a GPU, fingerprinting one file
on a high-memory box, without rewriting the rest of the pipeline.

**AI calls are first-class DAG nodes.** Marimo's "AI-native editor"
framing - including the `marimo pair` agent skill they shipped in
v0.22.5 - covers code-authoring assistance: generating cells from a
prompt, inline autocompletion, sidebar chat. LLM responses are not
themselves DAG nodes; if a user calls an LLM from a Python cell, it's
an ordinary expression with no caching, no schema enforcement, no
retry on validation. Strata's prompt cells render a `{{ var }}`
template against upstream artifacts, send the result to an
OpenAI-compatible API (or Anthropic native tool-use when an output
schema is set), validate against an optional JSON Schema, and store
the response as a cached artifact - same caching guarantees as a
Python cell. Mixing prompt and Python cells in one DAG is the point.

**Variant cells are unique to Strata.** Three alternative training
implementations can share the same DAG slot; switching the active variant
is a one-line edit in `notebook.toml` and downstream cells re-cascade
against the new producer. The other tools require duplicating cells (and
the downstream cells that read them) per variant.

**Notebook commits show the work, not the runtime.** Strata stores cells
as one `.py` file per cell, `notebook.toml` as the manifest, and all
runtime state (display outputs, console snapshots, the artifact store) in
a gitignored `.strata/` directory. `notebook.toml`'s `updated_at` only
bumps on structural edits, adding/removing cells, changing workers:
so re-running a cell never touches the tracked tree. Jupyter `.ipynb`
files JSON-encode source, outputs (base64 images and all), and execution
counts in the same blob; Marimo and Pluto avoid the JSON issue with one
text file per notebook but still keep all cells together - and Marimo's
PEP 723 inline dependency block means dependency edits and code edits
share the same file. Strata's per-cell layout keeps a diff that touches
cell 3 from rebasing on top of changes to cell 7.

## Where other notebooks are stronger

- **Interactive UI widgets.** Marimo has `mo.ui.slider`, `mo.ui.dropdown`,
  etc., reactive widgets the user can drag/click to update a parameter,
  which then propagates through the DAG. Strata doesn't have a widget
  layer; you change a value by editing source.
- **Ecosystem maturity.** Jupyter's ecosystem of extensions, kernels (R,
  Julia, Scala, Bash, etc.), and integrations is unmatched. Strata is
  Python-only with an AI provider abstraction.
- **Reactive evaluation at the keystroke level.** Pluto and Marimo
  immediately re-run dependent cells on edit. Strata is reactive about
  *staleness* (the DAG updates, downstream cells flip to stale on every
  source change) but execution is explicit, you press Run.
- **Hosted offerings.** Google Colab, Deepnote, Hex, and Databricks
  Notebooks all bundle a hosted runtime; Strata is self-hosted (see
  the section below on where these fit).

## Where the hosted offerings fit

Most managed notebook services are JupyterLab in a hosted wrapper. Their
files are `.ipynb`, their kernels are IPython, and they differentiate on
compute provisioning (GPUs, identity, billing) rather than on the
notebook runtime itself:

| Offering | Runtime | File format |
|---|---|---|
| Google Colab | Jupyter | `.ipynb` |
| Kaggle Notebooks | Jupyter | `.ipynb` |
| AWS SageMaker Studio | JupyterLab | `.ipynb` |
| Azure ML Notebooks | Jupyter / JupyterLab | `.ipynb` |
| Databricks Notebooks | Custom UI on IPython kernel | `.ipynb` (default), `.dbc` legacy |

None of them have automatic content-addressed caching, per-cell remote
dispatch, or first-class AI cells, because the underlying Jupyter
runtime doesn't.

The smaller "we-rejected-Jupyter" cohort (Marimo, Observable, Deepnote,
Hex) explicitly stepped away from `.ipynb` to redesign the runtime:
reactive execution, real-time collaboration, multi-language cells, app
deployment. That cohort is Strata's natural competitive set; the
JupyterLab-wrapper hosted offerings are an orthogonal category whose
moat is compute provisioning, not notebook-engine innovation.

## When to pick Strata

Strata is the right fit when your notebook is:

- **Expensive to recompute**: model training, embeddings, large scans,
  long LLM chains. The automatic cache pays for itself the first time you
  reload.
- **Heterogeneous in compute**: some cells want a GPU, some want a
  warehouse, some are pure CPU. The `# @worker` annotation routes each
  cell to where it should run.
- **Iterative and branching**: variant cells let you keep three model
  candidates in one notebook without forking.
- **Version-controlled with others**: plain text, no JSON-in-git pain,
  no execution-count churn on every re-run.
- **AI-heavy**: prompt cells make LLM responses cacheable like any
  other artifact, with schema-constrained output and retry-on-validation.

For light interactive exploration where the work is a few seconds per
cell, you're not really paying for what Strata gives you, Jupyter and
Marimo are fine. The value lands when your work is too expensive to
re-run on every refresh.
