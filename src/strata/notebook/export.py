"""Notebook export to shareable markdown / HTML.

Public entry: :func:`export_notebook` takes a notebook directory and
emits a single self-contained file representing the notebook — source
cells, cached display outputs, console snapshots — with no external
runtime dependencies on the receiving end.

This is the engine behind ``strata export`` (CLI) and the mkdocs hook
that auto-renders ``examples/*`` into the docs site.

Design choices captured in
``docs/internal/design-notebook-export.md``; the noteworthy ones:

- Prompt-cell *responses* are never rendered, regardless of flags.
  Privacy default — an LLM response can carry sensitive judgments
  the cell author might not want to share. The cell source template
  is always shown so the reader still sees what was asked.
- Variant cells render only the active variant by default;
  ``include_inactive_variants=True`` opts into a stacked rendering.
- Loop cells render the body + final iteration's output; per-
  iteration history is skipped to keep export size bounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from strata.notebook.models import CellOutput, CellState, NotebookState
from strata.notebook.parser import parse_notebook


@dataclass
class ExportOptions:
    """User-facing knobs for an export run."""

    output_format: Literal["markdown", "html"] = "markdown"
    include_inactive_variants: bool = False
    include_console: bool = True


def export_notebook(
    notebook_dir: Path,
    options: ExportOptions | None = None,
) -> str:
    """Render ``notebook_dir`` to a single export string.

    The notebook's existing on-disk state is used as-is — display
    outputs and console snapshots come from ``.strata/runtime.json``
    and ``.strata/console/`` respectively. Cells that have never been
    executed appear with their source only.

    Args:
        notebook_dir: directory containing ``notebook.toml``.
        options: format selection + flags; defaults to markdown.

    Returns:
        The rendered file content. Caller writes it to disk or stdout.
    """
    options = options or ExportOptions()
    notebook_dir = Path(notebook_dir)
    state = parse_notebook(notebook_dir)
    _resolve_variant_flags(state)

    blocks: list[Block] = []
    readme = _load_readme(notebook_dir)
    if readme is not None:
        # README already opens with its own h1; adding a "Notebook: <name>"
        # header on top would create two competing page titles in mkdocs.
        blocks.append(MarkdownBlock(readme))
    else:
        blocks.append(HeadingBlock(f"Notebook: {state.name}", level=1))

    for cell in state.cells:
        if not options.include_inactive_variants and cell.variant_active is False:
            continue
        blocks.extend(_render_cell(cell, state, notebook_dir, options))

    if options.output_format == "html":
        return _emit_html(blocks, title=state.name)
    return _emit_markdown(blocks)


# ---------------------------------------------------------------------------
# Block tree


@dataclass
class Block:
    """Base type for the renderer's intermediate representation."""


@dataclass
class HeadingBlock(Block):
    text: str
    level: int = 2


@dataclass
class MarkdownBlock(Block):
    """Verbatim markdown content (for the README and markdown cells)."""

    body: str


@dataclass
class CodeBlock(Block):
    language: str
    body: str
    title: str | None = None


@dataclass
class ChipsBlock(Block):
    """Small inline metadata chips shown under a cell heading."""

    items: list[tuple[str, str]] = field(default_factory=list)  # (label, value)


@dataclass
class NoteBlock(Block):
    """Single italicized sentence — context the reader needs."""

    text: str


@dataclass
class ImageBlock(Block):
    """Inline image rendered via a `data:` URL."""

    data_url: str
    alt: str = ""


@dataclass
class TableBlock(Block):
    """A tabular preview rendered as a markdown table.

    ``rows`` is a list of dicts keyed by column name. ``columns`` is
    the column order; entries missing from a row become an empty
    string in that cell.
    """

    columns: list[str]
    rows: list[dict[str, object]]
    title: str | None = None
    truncated_to: int | None = None  # row count if truncated
    total_rows: int | None = None  # rows reported by the upstream cell


# ---------------------------------------------------------------------------
# Cell rendering


def _render_cell(
    cell: CellState,
    state: NotebookState,
    notebook_dir: Path,
    options: ExportOptions,
) -> list[Block]:
    blocks: list[Block] = []

    # Banner heading uses the @name annotation when present, otherwise
    # the cell id. The intent is "what is this cell" — a short label.
    from strata.notebook.annotations import parse_annotations

    annotations = parse_annotations(cell.source)
    label = annotations.name or cell.id
    blocks.append(HeadingBlock(label, level=2))

    chips = _cell_chips(cell, annotations, state)
    if chips:
        blocks.append(ChipsBlock(chips))

    if cell.language == "markdown":
        # Markdown cells are *content*, not annotated source. Render the
        # body verbatim and skip the source-as-code path.
        blocks.append(MarkdownBlock(cell.source))
        return blocks

    # Prompt cells: source template only — never the response.
    if cell.language == "prompt":
        blocks.append(NoteBlock("Prompt cell — response intentionally excluded from export."))
        blocks.append(CodeBlock(language="text", body=cell.source))
        return blocks

    # Python / SQL / loop: source + outputs + console.
    fence_lang = _source_fence_language(cell.language)
    blocks.append(CodeBlock(language=fence_lang, body=cell.source))

    for output in cell.display_outputs or []:
        blocks.extend(
            _render_display_output(
                output,
                notebook_dir=notebook_dir,
                notebook_id=state.id,
            )
        )

    if options.include_console:
        blocks.extend(_render_console(cell))

    return blocks


_ANSI_ESCAPE_RE = None  # lazy-compiled in _strip_ansi


def _strip_ansi(text: str) -> str:
    """Remove ANSI CSI/OSC escape sequences from terminal output.

    Cells using ``rich``, ``colorama``, ``click.echo(..., color=True)``
    or progress bars emit escape sequences into stdout. They render as
    colours in a terminal but as ``\\x1b[31m...`` noise in a markdown or
    HTML reader. Strip them before persisting into the export so
    console snapshots stay readable.
    """
    global _ANSI_ESCAPE_RE
    if _ANSI_ESCAPE_RE is None:
        import re

        _ANSI_ESCAPE_RE = re.compile(
            r"\x1B"  # ESC
            r"(?:"
            r"[@-Z\\-_]"  # 2-byte CSI introducers
            r"|"
            r"\[[0-?]*[ -/]*[@-~]"  # CSI ... final byte
            r"|"
            r"\][^\x07\x1B]*(?:\x07|\x1B\\)"  # OSC ... ST/BEL
            r")"
        )
    return _ANSI_ESCAPE_RE.sub("", text)


def _render_console(cell: CellState) -> list[Block]:
    """Render persisted stdout/stderr snapshots, with ANSI codes stripped."""
    blocks: list[Block] = []
    stdout = _strip_ansi(cell.console_stdout or "").rstrip()
    stderr = _strip_ansi(cell.console_stderr or "").rstrip()
    if stdout:
        blocks.append(CodeBlock(language="text", body=stdout, title="stdout"))
    if stderr:
        blocks.append(CodeBlock(language="text", body=stderr, title="stderr"))
    return blocks


_PREVIEW_ROW_LIMIT = 20


def _render_display_output(
    output: CellOutput,
    *,
    notebook_dir: Path,
    notebook_id: str,
) -> list[Block]:
    """Per-content-type renderer for one persisted cell output.

    For image/png and text/markdown outputs the inline payload is not
    persisted in notebook.toml (it's a transient large field). When
    the artifact_uri points at a stored blob, load it lazily — same
    approach NotebookSession._hydrate_display_output uses.
    """
    if output.error:
        return [
            CodeBlock(language="text", body=output.error.rstrip(), title="Error"),
        ]

    output = _hydrate_output(output, notebook_dir=notebook_dir, notebook_id=notebook_id)
    ctype = output.content_type

    if ctype == "image/png" and output.inline_data_url:
        return [ImageBlock(data_url=output.inline_data_url, alt="cell output")]

    if ctype == "text/markdown" and output.markdown_text is not None:
        return [MarkdownBlock(output.markdown_text)]

    if ctype == "arrow/ipc":
        columns = list(output.columns or [])
        if columns:
            preview = output.preview if isinstance(output.preview, list) else []
            normalized = _normalize_table_preview(preview, columns)
            truncated_to = min(len(normalized), _PREVIEW_ROW_LIMIT)
            return [
                TableBlock(
                    columns=columns,
                    rows=normalized[:truncated_to],
                    title="Output",
                    truncated_to=truncated_to,
                    total_rows=output.rows,
                )
            ]
        # No columns — fall through to scalar/repr below.

    if ctype == "json/object":
        body = _format_json_preview(output.preview)
        return [CodeBlock(language="json", body=body, title="Output")]

    if ctype == "pickle/object":
        # serializer.py stores a "<TypeName object>" hint in preview for
        # pickled values; surface it so the reader knows what kind of
        # opaque blob the cell produced.
        hint = output.preview if isinstance(output.preview, str) else None
        if hint:
            return [NoteBlock(f"Pickled output ({hint}) — not rendered in export.")]
        return [NoteBlock("Pickled output — not rendered in export.")]

    # Fallback: render the preview as text. Covers scalars (json content
    # type with a scalar preview, plain int/str values, etc.) and any
    # exotic content type we haven't special-cased.
    preview = output.preview
    if preview is None:
        return []
    body = _format_scalar_preview(preview)
    return [CodeBlock(language="text", body=body, title="Output")]


def _format_json_preview(value: object) -> str:
    """JSON-pretty-print a preview value for a fenced ``json`` block."""
    import json

    try:
        return json.dumps(value, indent=2, default=str, sort_keys=False)
    except (TypeError, ValueError):
        return repr(value)


def _normalize_table_preview(
    preview: list,
    columns: list[str],
) -> list[dict[str, object]]:
    """Coerce serialized table-preview rows into dict-keyed rows.

    The serializer at ``serializer.py`` emits rows as positional lists
    (one entry per column). Some callers — and our own tests — emit
    them as dicts already. Accept either shape so the table emitter
    always works against the same dict-of-cells representation.

    Rows shorter than ``columns`` get missing cells coerced to None;
    rows longer are truncated. Non-list / non-dict entries are
    skipped silently.
    """
    out: list[dict[str, object]] = []
    for row in preview:
        if isinstance(row, dict):
            out.append(dict(row))
        elif isinstance(row, (list, tuple)):
            padded = list(row[: len(columns)])
            while len(padded) < len(columns):
                padded.append(None)
            out.append(dict(zip(columns, padded)))
        # Anything else (a stray scalar that snuck into the preview
        # list) is silently dropped — better to render a small table
        # than to error during export.
    return out


def _format_scalar_preview(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    return repr(value)


def _hydrate_output(output: CellOutput, *, notebook_dir: Path, notebook_id: str) -> CellOutput:
    """Re-attach transient inline fields stripped at TOML-save time.

    ``inline_data_url`` (for image/png) and ``markdown_text`` (for
    text/markdown) are dropped before persisting because they're
    large transient fields. When the cell carries an ``artifact_uri``
    we can re-fetch the blob from the local artifact store.

    Failures are silent — if the artifact store can't be opened or the
    blob is missing, the output renders as-is (no image / no markdown
    body, just whatever else the renderer can show).
    """
    if output.content_type not in {"image/png", "text/markdown"}:
        return output
    if output.content_type == "image/png" and output.inline_data_url:
        return output
    if output.content_type == "text/markdown" and output.markdown_text:
        return output
    if not output.artifact_uri:
        return output

    try:
        from strata.notebook.artifact_integration import NotebookArtifactManager

        artifact_id, version = _parse_artifact_uri(output.artifact_uri)
        manager = NotebookArtifactManager(
            notebook_id=notebook_id,
            artifact_dir=notebook_dir / ".strata" / "artifacts",
        )
        blob = manager.load_artifact_data(artifact_id, version)
    except Exception:
        return output

    hydrated = output.model_copy()
    if output.content_type == "image/png":
        import base64

        hydrated.inline_data_url = f"data:image/png;base64,{base64.b64encode(blob).decode('ascii')}"
    else:  # text/markdown
        hydrated.markdown_text = blob.decode("utf-8", errors="replace")
    return hydrated


def _parse_artifact_uri(artifact_uri: str) -> tuple[str, int]:
    """Parse a canonical ``strata://artifact/<id>@v=<n>`` URI."""
    parts = artifact_uri.split("/")
    artifact_id = parts[-1].split("@")[0]
    version = int(parts[-1].split("@v=")[1])
    return artifact_id, version


def _cell_chips(cell: CellState, annotations, state: NotebookState) -> list[tuple[str, str]]:
    """Build the small metadata chip list shown under a cell heading."""
    chips: list[tuple[str, str]] = []
    chips.append(("kind", cell.language))
    if cell.variant_group is not None and cell.variant_name is not None:
        chips.append(("variant", f"{cell.variant_name} of {cell.variant_group}"))
    if annotations.worker:
        chips.append(("worker", annotations.worker))
    if annotations.loop is not None:
        chips.append(
            ("loop", f"max_iter={annotations.loop.max_iter} carry={annotations.loop.carry}")
        )
    if annotations.mounts:
        chips.append(("mounts", ", ".join(m.name for m in annotations.mounts)))
    return chips


def _source_fence_language(language: str) -> str:
    """Map the cell's language to a markdown code-fence info string."""
    if language == "python":
        return "python"
    if language == "sql":
        return "sql"
    return "text"


# ---------------------------------------------------------------------------
# Variant resolution
#
# parse_notebook() loads cells from disk but doesn't populate
# variant_group / variant_name / variant_active — those are produced
# by NotebookSession._analyze_and_build_dag at session boot. Export
# runs without a session, so we replicate just the variant-resolution
# slice here: parse each cell's @variant annotation and decide which
# member is active per group using the same first-in-source-order
# fallback the DAG layer applies.


def _resolve_variant_flags(state: NotebookState) -> None:
    """Populate variant_group / variant_name / variant_active on each cell."""
    from strata.notebook.annotations import parse_annotations

    selections = dict(state.variant_active_selections)
    grouped: dict[str, list[CellState]] = {}
    group_order: list[str] = []

    for cell in state.cells:
        annotations = parse_annotations(cell.source)
        if annotations.variant is None:
            cell.variant_group = None
            cell.variant_name = None
            cell.variant_active = True
            continue
        cell.variant_group = annotations.variant.group
        cell.variant_name = annotations.variant.name
        cell.variant_active = True  # adjusted below if shadowed
        if annotations.variant.group not in grouped:
            group_order.append(annotations.variant.group)
        grouped.setdefault(annotations.variant.group, []).append(cell)

    for group_id in group_order:
        members = grouped[group_id]
        wanted_name = selections.get(group_id)
        active_cell = members[0]
        if wanted_name is not None:
            for cell in members:
                if cell.variant_name == wanted_name:
                    active_cell = cell
                    break
        for cell in members:
            cell.variant_active = cell.id == active_cell.id


# ---------------------------------------------------------------------------
# README discovery


def _load_readme(notebook_dir: Path) -> str | None:
    readme_path = notebook_dir / "README.md"
    if not readme_path.is_file():
        return None
    try:
        return readme_path.read_text(encoding="utf-8")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Emitters


def _emit_markdown(blocks: list[Block]) -> str:
    """Walk the block tree and emit CommonMark."""
    pieces: list[str] = []
    for block in blocks:
        if isinstance(block, HeadingBlock):
            pieces.append(f"{'#' * block.level} {block.text}")
        elif isinstance(block, MarkdownBlock):
            pieces.append(block.body.rstrip("\n"))
        elif isinstance(block, CodeBlock):
            fence = "`" * _fence_length_for(block.body)
            title_suffix = f' title="{block.title}"' if block.title else ""
            pieces.append(
                f"{fence}{block.language}{title_suffix}\n{block.body.rstrip()}\n{fence}",
            )
        elif isinstance(block, ChipsBlock):
            chip_text = "  ·  ".join(f"**{k}** {v}" for k, v in block.items)
            pieces.append(f"<sub>{chip_text}</sub>")
        elif isinstance(block, NoteBlock):
            pieces.append(f"*{block.text}*")
        elif isinstance(block, ImageBlock):
            pieces.append(f'<img src="{block.data_url}" alt="{block.alt}">')
        elif isinstance(block, TableBlock):
            pieces.append(_emit_markdown_table(block))
    return "\n\n".join(pieces) + "\n"


def _emit_markdown_table(block: TableBlock) -> str:
    """Render a TableBlock as a GitHub-flavored markdown table."""
    columns = list(block.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines: list[str] = []
    for row in block.rows:
        cells = [_format_table_cell(row.get(col)) for col in columns]
        body_lines.append("| " + " | ".join(cells) + " |")

    suffix = ""
    if (
        block.total_rows is not None
        and block.truncated_to is not None
        and block.total_rows > block.truncated_to
    ):
        suffix = f"\n\n*…showing {block.truncated_to} of {block.total_rows} rows*"

    title_line = f"**{block.title}**\n\n" if block.title else ""
    return title_line + "\n".join([header, separator, *body_lines]) + suffix


def _fence_length_for(body: str) -> int:
    """Return the minimum number of backticks needed to fence ``body``.

    CommonMark requires the closing fence to be at least as long as the
    opening one. If the body contains a run of N backticks, the fence
    must be longer than N or it'll close early and corrupt the markdown.
    Prompt cells routinely embed fenced examples inside their templates,
    so this is a real concern, not a theoretical one.
    """
    import re

    longest = 0
    for match in re.finditer(r"`+", body):
        longest = max(longest, len(match.group()))
    return max(3, longest + 1)


def _format_table_cell(value: object) -> str:
    """Coerce a single cell value to a markdown-safe inline string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("|", "\\|").replace("\n", " ")
    if isinstance(value, float):
        # Trim trailing zeros so 5.000000 → 5.0; keep enough precision
        # for stats tables (mean / std / etc.) without becoming noisy.
        return f"{value:.4g}"
    return str(value)


def _emit_html(blocks: list[Block], *, title: str) -> str:
    """Render the block tree as a standalone HTML document.

    Code blocks are syntax-highlighted via Pygments (server-side, no
    client JS). Images are inlined as ``data:`` URLs (already produced
    by the hydration step). Tables become real ``<table>`` elements.

    Markdown content (README intro, markdown cells) is rendered as
    preformatted text — adding a markdown-to-HTML library to the
    notebook runtime just for export wasn't worth the dep cost. For
    the best fidelity on prose-heavy notebooks, use ``--to markdown``
    and post-process with your tool of choice.
    """
    from html import escape

    pieces: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{escape(title)}</title>",
        f"<style>{_html_stylesheet()}</style>",
        "</head>",
        "<body>",
        '<main class="notebook-export">',
    ]

    for block in blocks:
        pieces.append(_render_block_html(block))

    pieces.extend(["</main>", "</body>", "</html>"])
    return "\n".join(pieces) + "\n"


def _render_block_html(block: Block) -> str:
    from html import escape

    if isinstance(block, HeadingBlock):
        level = max(1, min(6, block.level))
        return f"<h{level}>{escape(block.text)}</h{level}>"
    if isinstance(block, MarkdownBlock):
        # See _emit_html docstring on why we wrap in <pre> rather
        # than rendering markdown to HTML.
        return f'<pre class="markdown-source">{escape(block.body.rstrip())}</pre>'
    if isinstance(block, CodeBlock):
        return _render_code_html(block)
    if isinstance(block, ChipsBlock):
        chip_html = "".join(
            f'<span class="chip"><b>{escape(k)}</b> {escape(v)}</span>' for k, v in block.items
        )
        return f'<div class="chips">{chip_html}</div>'
    if isinstance(block, NoteBlock):
        return f'<p class="note"><em>{escape(block.text)}</em></p>'
    if isinstance(block, ImageBlock):
        src = escape(block.data_url, quote=True)
        alt = escape(block.alt)
        return f'<p class="image"><img src="{src}" alt="{alt}"></p>'
    if isinstance(block, TableBlock):
        return _render_table_html(block)
    return ""


def _render_code_html(block: CodeBlock) -> str:
    """Syntax-highlight a code block via Pygments."""
    from html import escape

    title_html = f'<div class="code-title">{escape(block.title)}</div>' if block.title else ""
    try:
        from pygments import highlight
        from pygments.formatters.html import HtmlFormatter
        from pygments.lexers import get_lexer_by_name
        from pygments.util import ClassNotFound

        try:
            lexer = get_lexer_by_name(block.language)
        except ClassNotFound:
            lexer = get_lexer_by_name("text")
        formatter = HtmlFormatter(nowrap=False, cssclass="codehilite")
        highlighted = highlight(block.body.rstrip(), lexer, formatter)
        return f'<div class="code-block">{title_html}{highlighted}</div>'
    except Exception:
        # If Pygments is missing or chokes, fall back to escaped <pre>.
        escaped = escape(block.body.rstrip())
        return (
            f'<div class="code-block">{title_html}'
            f'<pre class="codehilite"><code>{escaped}</code></pre></div>'
        )


def _render_table_html(block: TableBlock) -> str:
    from html import escape

    columns = list(block.columns)
    header_cells = "".join(f"<th>{escape(c)}</th>" for c in columns)
    body_rows: list[str] = []
    for row in block.rows:
        cells = "".join(f"<td>{escape(_format_table_cell(row.get(col)))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")

    suffix = ""
    if (
        block.total_rows is not None
        and block.truncated_to is not None
        and block.total_rows > block.truncated_to
    ):
        suffix = (
            f'<div class="table-footer">…showing {block.truncated_to} of '
            f"{block.total_rows} rows</div>"
        )

    caption = f"<caption>{escape(block.title)}</caption>" if block.title else ""
    return (
        '<div class="table-block">'
        f"<table>{caption}<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
        f"{suffix}</div>"
    )


def _html_stylesheet() -> str:
    """Embedded CSS for the standalone HTML export.

    Goals: legible, neutral, prints reasonably. Not a faithful match
    for the in-product notebook UI; we aim for "looks like a clean
    document," not "looks like the editor."
    """
    try:
        from pygments.formatters.html import HtmlFormatter

        pygments_css = HtmlFormatter(cssclass="codehilite").get_style_defs(".codehilite")
    except Exception:
        pygments_css = ""

    css_lines = [
        ":root {",
        "  --fg: #1f2328; --muted: #6b7280; --border: #d0d7de;",
        "  --bg-code: #f6f8fa; --bg-chip: #eef0f3;",
        "}",
        'body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;',
        "       color: var(--fg); margin: 0; padding: 24px; }",
        "main.notebook-export { max-width: 880px; margin: 0 auto; }",
        "h1, h2, h3 { line-height: 1.25; }",
        "h1 { font-size: 1.9rem; border-bottom: 1px solid var(--border);",
        "     padding-bottom: 8px; }",
        "h2 { font-size: 1.4rem; margin-top: 2rem; }",
        "p.note { color: var(--muted); margin: 4px 0 12px; }",
        ".chips { display: flex; gap: 6px; flex-wrap: wrap; margin: -8px 0 12px;",
        "         font-size: 12px; color: var(--muted); }",
        ".chip { background: var(--bg-chip); border-radius: 999px; padding: 2px 10px; }",
        ".chip b { color: var(--fg); margin-right: 4px; }",
        ".code-block { margin: 12px 0; }",
        ".code-title { font-size: 11px; text-transform: uppercase;",
        "              letter-spacing: 0.04em; color: var(--muted);",
        "              margin-bottom: 4px; }",
        ".codehilite { background: var(--bg-code); border: 1px solid var(--border);",
        "              border-radius: 6px; padding: 12px; overflow-x: auto;",
        '              font-family: ui-monospace, "JetBrains Mono", Menlo, monospace;',
        "              font-size: 13px; }",
        "pre.markdown-source { background: var(--bg-code);",
        "                      border: 1px solid var(--border);",
        "                      border-radius: 6px; padding: 12px;",
        "                      overflow-x: auto; white-space: pre-wrap;",
        '                      font-family: ui-monospace, "JetBrains Mono", Menlo,',
        "                                   monospace;",
        "                      font-size: 13px; }",
        ".table-block { margin: 12px 0; overflow-x: auto; }",
        "table { border-collapse: collapse; font-size: 13px; }",
        "th, td { border-bottom: 1px solid var(--border); padding: 6px 12px;",
        "         text-align: left; }",
        "th { background: var(--bg-code); }",
        "caption { caption-side: top; font-weight: 600; text-align: left;",
        "          padding-bottom: 6px; }",
        ".table-footer { font-size: 12px; color: var(--muted); padding-top: 4px; }",
        ".image img { max-width: 100%; height: auto;",
        "             border: 1px solid var(--border); border-radius: 6px; }",
    ]
    return "\n".join(css_lines) + "\n" + pygments_css
