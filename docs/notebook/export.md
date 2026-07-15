# Export

`strata export` renders a notebook directory to a single self-contained
file, markdown or HTML, for sharing or archiving. The exported file
embeds cell sources, cached display outputs (DataFrame tables, images,
JSON, console snapshots), and any structural metadata worth showing,
with **no runtime dependency** on the recipient's end.

## When to use it

- **Send a notebook to a teammate without Strata.** Pipe the markdown
  into Slack / email / Confluence, or drop the HTML file on a shared
  drive. They open it and see the same cells and outputs you do.
- **Archive a notebook's state at a moment in time.** Re-runs change
  outputs; an exported file captures what *this* run produced.
- **Auto-generate documentation.** The docs site you're reading right
  now generates the [Examples catalog](examples.md) by running
  `strata export` against every `examples/*/` at mkdocs build time.

For exports inside the notebook UI itself, click the **Export** button
in the notebook header, same engine, browser triggers a file download.

## Usage

```bash
strata export <notebook_dir> [options]
```

### Options

| Flag                            | Description                                                                                          |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--to {markdown,html}`          | Output format. Default `markdown`.                                                                   |
| `--out <path>`                  | Write to a file instead of stdout.                                                                   |
| `--include-inactive-variants`   | Stack all variants of every group; otherwise only the active variant is rendered.                    |
| `--no-console`                  | Omit the per-cell stdout/stderr snapshots.                                                           |
| `--app-view`                    | App-view snapshot: render only widgets, markdown, and outputs (no cell sources). See below.          |
| `--max-output-bytes <n>`        | Per-output byte cap (default 1048576 = 1 MB). Truncates console snapshots and JSON; replaces oversized images with a size-note placeholder. Pass `0` to disable. |

### Examples

```bash
# Markdown to stdout, pipe anywhere
strata export ./my_analysis

# Standalone HTML for sharing
strata export ./my_analysis --to html --out share.html

# Include every variant of every variant group
strata export ./my_analysis --include-inactive-variants

# App-view snapshot: a portable, frozen picture of the dashboard
strata export ./my_analysis --app-view --to html --out dashboard.html
```

## App-view snapshot (`--app-view`)

A regular export is a **document** view — it shows every cell's *source*
alongside its outputs, for reading or archiving the notebook. An
app-view snapshot is a **dashboard** view: it renders only what the
[read-only app view](cells.md#app-view) shows — widget panels (as their
current control values), markdown, and display outputs — with **no cell
sources, chips, or console**.

It's the static counterpart to embedding the app view. Where an embed is
*live* (an `<iframe>` backed by a running server, interactive), a
snapshot is *frozen* and **self-contained** — a single file with images
baked in as `data:` URLs, no server or network needed. Use it to email a
report, drop a result on a shared drive, or archive exactly what a run
produced.

Widget cells render as a compact line of `control: value` chips — the
parameter settings that produced the outputs below them — read from the
notebook's persisted control values. Cells marked `# @app hide` are
omitted, matching the live app view; prompt-cell responses stay excluded
for the same privacy reason as a regular export.

In the UI, the notebook's **Export** menu offers **App snapshot**
alongside Markdown and HTML.

## What gets rendered

For each cell, the exporter emits in order:

1. **Banner**: the cell's `# @name` (if set) or its ID, plus small
   chips for `# @worker`, `# @variant`, `# @loop`, `# @mount`.
2. **Source**: fenced code block in the cell's language. The fence
   length auto-grows to cover embedded triple-backticks safely.
3. **Cached display outputs**: per content type:
    - **DataFrames / Arrow tables** render as markdown tables (or
      HTML `<table>`), truncated to 20 preview rows.
    - **PNG images** inline as `data:` URLs, loaded lazily from the
      artifact store when needed.
    - **JSON / dict / list** as fenced JSON.
    - **Markdown** rendered as content.
    - **Pickled** values become a placeholder with the type hint
      from the serializer (e.g. *Pickled output (`<MyThing object>`) not rendered in export*).
    - **Errors** render as text blocks.
4. **Console output**: `stdout` / `stderr` snapshots if present.
   ANSI escape sequences are stripped so coloured output stays
   readable in a non-terminal viewer.

## Cell-kind specifics

### Prompt cells, response excluded by design

Prompt cells render the **template only**. Cached LLM responses are
never included in exports, with or without flags. Reasons:

- LLM responses can contain PII, judgments, or context the cell
  author wouldn't want to leak in a shared file.
- A `--include-prompt-responses` opt-in is easy to forget; an opt-out
  would mean responses leak by default.

If you want a response in the export, paste it into a markdown cell
manually, that signals "yes I want this published."

### Variant cells, active member by default

A variant group renders only its **active** variant by default, plus
a small banner note ("Variant `<name>` of group `<group>`"). Pass
`--include-inactive-variants` to render all members stacked.

### Loop cells, final iteration only

Loop cells render the body and the **final iteration's display
output**. Per-iteration history is not unrolled (it would bloat the
export for limited gain). The cell banner notes `max_iter` and
`carry` so the reader knows it's a loop.

### Markdown cells, content verbatim

Markdown cells render as content, not as fenced source. In HTML
exports they currently render as preformatted text (no
markdown-to-HTML pass); for prose-heavy notebooks use the markdown
format.

## Output formats

| Format     | Best for                                                          |
| ---------- | ----------------------------------------------------------------- |
| `markdown` | Drops into GitHub PRs, mkdocs sites, Confluence pages, Notion.    |
| `html`     | Standalone shareable file. Server-side Pygments syntax highlights. Images inline. No external network requests. |

## Integration: mkdocs hook

The docs site uses a small mkdocs hook
(`docs_hooks/export_examples.py`) to render every `examples/*/`
notebook at build time. The hook calls `export_notebook` directly:
no CLI process is spawned. Generated pages are gitignored and
regenerated on every build. The hand-written
[Examples catalog](examples.md) provides the feature-grouped table of
contents.

To run something similar on your own docs build, the simplest
recipe is:

```python
# docs_hooks/export_my_notebooks.py
from pathlib import Path

from strata.notebook.export import export_notebook


def on_pre_build(config):
    out_dir = Path(config["docs_dir"]) / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for nb in Path("path/to/notebooks").iterdir():
        if (nb / "notebook.toml").is_file():
            (out_dir / f"{nb.name}.md").write_text(export_notebook(nb))
```

Register it in `mkdocs.yml`:

```yaml
hooks:
  - docs_hooks/export_my_notebooks.py
```
