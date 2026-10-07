"""Notebook export to a single self-contained markdown or HTML file.

Engine behind ``strata export`` and the mkdocs hook that renders ``examples/*``.
Prompt-cell responses are never rendered (an LLM answer may carry judgments the
author would not share; the template is shown). Variant cells render only the
active variant unless ``include_inactive_variants``; loop cells render the
final iteration only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from strata.notebook.models import CellLanguage, CellOutput, CellState, NotebookState
from strata.notebook.parser import parse_notebook

if TYPE_CHECKING:
    import re

_DEFAULT_MAX_OUTPUT_BYTES = 1_048_576  # per rendered output


class ExportFormat(StrEnum):
    """Target format for ``export_notebook``."""

    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class ExportOptions:
    """User-facing knobs for an export run."""

    output_format: ExportFormat = ExportFormat.MARKDOWN
    include_inactive_variants: bool = False
    include_console: bool = True
    # Render only what the read-only app view shows (widgets, markdown, display
    # outputs), with no sources, chips or console.
    app_view: bool = False
    # Per-output byte cap for console, JSON previews and inline images. DataFrame
    # previews are row-capped separately. Zero disables it.
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES


def export_notebook(
    notebook_dir: Path,
    options: ExportOptions | None = None,
) -> str:
    """Render ``notebook_dir`` to a single export string.

    Uses the on-disk state as-is (``.strata/runtime.json`` and
    ``.strata/console/``); never-run cells appear with their source only.
    """
    options = options or ExportOptions()
    notebook_dir = Path(notebook_dir)
    state = parse_notebook(notebook_dir)
    _resolve_variant_flags(state)

    blocks: list[Block] = []

    if options.app_view:
        # No README banner: the app view has none.
        blocks.append(HeadingBlock(state.name, level=1))
        for cell in state.cells:
            if not options.include_inactive_variants and cell.variant_active is False:
                continue
            if not _is_app_cell(cell):
                continue
            blocks.extend(_render_app_cell(cell, state, notebook_dir, options))
        if options.output_format == ExportFormat.HTML:
            return _emit_html(blocks, title=state.name)
        return _emit_markdown(blocks)

    readme = _load_readme(notebook_dir)
    if readme is not None:
        # README opens with its own h1; a second header would compete in mkdocs.
        blocks.append(MarkdownBlock(readme))
    else:
        blocks.append(HeadingBlock(f"Notebook: {state.name}", level=1))

    for cell in state.cells:
        if not options.include_inactive_variants and cell.variant_active is False:
            continue
        blocks.extend(_render_cell(cell, state, notebook_dir, options))

    if options.output_format == ExportFormat.HTML:
        return _emit_html(blocks, title=state.name)
    return _emit_markdown(blocks)


# --- Block tree ---


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
    """Single italicized sentence: context the reader needs."""

    text: str


@dataclass
class ImageBlock(Block):
    """Inline image rendered via a `data:` URL."""

    data_url: str
    alt: str = ""


@dataclass
class TableBlock(Block):
    """A tabular preview rendered as a markdown table.

    ``columns`` is the column order; a column missing from a row renders empty.
    """

    columns: list[str]
    rows: list[dict[str, object]]
    title: str | None = None
    truncated_to: int | None = None  # row count if truncated
    total_rows: int | None = None  # rows reported by the upstream cell


# --- Cell rendering ---


def _is_app_hidden(source: str) -> bool:
    """Whether a cell carries ``# @app hide``, as the app view filters."""
    import re

    return any(re.match(r"#\s*@app\s+hide\b", line.strip()) for line in source.splitlines())


def _is_app_cell(cell: CellState) -> bool:
    """Cells the app view shows: widget, markdown, or with a display output, minus ``@app hide``.

    Mirrors ``appCells`` in ``AppView.vue`` so a snapshot matches the live app.
    """
    if _is_app_hidden(cell.source):
        return False
    return cell.language in (CellLanguage.WIDGET, CellLanguage.MARKDOWN) or bool(
        cell.display_outputs
    )


def _render_widget_controls(cell: CellState) -> list[Block]:
    """Render a widget cell's controls as ``(name, value)`` chips, value or default."""
    from strata.notebook.widget_analyzer import analyze_widget_cell

    descriptors = analyze_widget_cell(cell.source).descriptors
    values = cell.widget_values or {}
    items = [(d.name, str(values.get(d.name, d.default))) for d in descriptors]
    return [ChipsBlock(items)] if items else []


def _render_app_cell(
    cell: CellState,
    state: NotebookState,
    notebook_dir: Path,
    options: ExportOptions,
) -> list[Block]:
    """Render one cell for an app-view snapshot: outputs only, no source, chips or console.

    Prompt cells are skipped, as their output is the model response.
    """
    if cell.language == CellLanguage.PROMPT:
        return []
    if cell.language == CellLanguage.MARKDOWN:
        return [MarkdownBlock(cell.source)]

    from strata.notebook.annotations import parse_annotations

    blocks: list[Block] = []
    name = parse_annotations(cell.source).name
    if name:
        blocks.append(HeadingBlock(name, level=2))

    if cell.language == CellLanguage.WIDGET:
        blocks.extend(_render_widget_controls(cell))
        return blocks

    for output in cell.display_outputs or []:
        blocks.extend(
            _render_display_output(
                output,
                notebook_dir=notebook_dir,
                notebook_id=state.id,
                max_bytes=options.max_output_bytes,
            )
        )
    return blocks


def _render_cell(
    cell: CellState,
    state: NotebookState,
    notebook_dir: Path,
    options: ExportOptions,
) -> list[Block]:
    blocks: list[Block] = []

    from strata.notebook.annotations import parse_annotations

    annotations = parse_annotations(cell.source)

    if cell.language == CellLanguage.MARKDOWN:
        # Markdown cells usually open with their own heading, so no banner or chips:
        # the body is the section divider.
        blocks.append(MarkdownBlock(cell.source))
        return blocks

    label = annotations.name or cell.id
    blocks.append(HeadingBlock(label, level=2))

    chips = _cell_chips(cell, annotations, state)
    if chips:
        blocks.append(ChipsBlock(chips))

    # Prompt cells: source template only, never the response.
    if cell.language == CellLanguage.PROMPT:
        blocks.append(NoteBlock("Prompt cell: response intentionally excluded from export."))
        blocks.append(CodeBlock(language="text", body=cell.source))
        return blocks

    fence_lang = _source_fence_language(cell.language)
    blocks.append(CodeBlock(language=fence_lang, body=cell.source))

    for output in cell.display_outputs or []:
        blocks.extend(
            _render_display_output(
                output,
                notebook_dir=notebook_dir,
                notebook_id=state.id,
                max_bytes=options.max_output_bytes,
            )
        )

    if options.include_console:
        blocks.extend(_render_console(cell, max_bytes=options.max_output_bytes))

    blocks.extend(_render_error(cell, max_bytes=options.max_output_bytes))
    return blocks


_ANSI_ESCAPE_RE = None


def _sanitize_markdown_body(body: str) -> str:
    """Neutralize raw HTML and script-capable link targets in user-authored markdown.

    Matches the UI, which renders with markdown-it ``html: false``: every raw ``<`` outside
    code is entity-escaped so tags show as text (http/https/mailto autolinks stay), and
    ``javascript:`` / ``vbscript:`` / non-image ``data:`` link destinations become ``#``.
    """
    import re

    pieces: list[str] = []
    fence: str | None = None
    prose: list[str] = []
    for line in body.splitlines(keepends=True):
        if fence is None:
            opener = re.match(r" {0,3}(`{3,}|~{3,})", line)
            if opener and not (opener.group(1)[0] == "`" and "`" in line[opener.end() :]):
                pieces.append(_sanitize_markdown_prose("".join(prose)))
                prose = []
                fence = opener.group(1)
                pieces.append(line)
            else:
                prose.append(line)
            continue
        pieces.append(line)
        closer = re.match(r" {0,3}(`{3,}|~{3,})[ \t]*$", line.rstrip("\r\n"))
        if closer and closer.group(1)[0] == fence[0] and len(closer.group(1)) >= len(fence):
            fence = None
    pieces.append(_sanitize_markdown_prose("".join(prose)))
    return "".join(pieces)


def _sanitize_markdown_prose(text: str) -> str:
    """Sanitize markdown outside fenced code: link targets everywhere, ``<`` outside code spans."""
    import re

    text = _neutralize_link_targets(text)
    out: list[str] = []
    pos = 0
    runs = list(re.finditer(r"`+", text))
    i = 0
    while i < len(runs):
        opener = runs[i]
        match = next(
            (j for j in range(i + 1, len(runs)) if len(runs[j].group()) == len(opener.group())),
            None,
        )
        if match is None:
            i += 1
            continue
        out.append(_escape_raw_html(text[pos : opener.start()]))
        out.append(text[opener.start() : runs[match].end()])
        pos = runs[match].end()
        i = match + 1
    out.append(_escape_raw_html(text[pos:]))
    return "".join(out)


def _escape_raw_html(text: str) -> str:
    import re

    # Keep http(s)/mailto and email autolinks; any other ``<`` could open a tag.
    return re.sub(
        r"<(?!(?:(?:https?|mailto):[^\s<>]*|[\w.+-]+@[\w-]+(?:\.[\w-]+)+)>)",
        "&lt;",
        text,
        flags=re.IGNORECASE,
    )


def _neutralize_link_targets(text: str) -> str:
    """Rewrite inline-link and reference-definition destinations with an unsafe scheme to ``#``."""
    import re

    out: list[str] = []
    pos = 0
    for opener in re.finditer(r"\]\(", text):
        start = opener.end()
        if start < pos:
            continue
        while start < len(text) and text[start] in " \t\r\n":
            start += 1
        end = _link_destination_end(text, start)
        if _is_unsafe_link(text[start:end]):
            out.append(text[pos:start])
            out.append("#")
            pos = end
    out.append(text[pos:])
    text = "".join(out)

    def _ref(m: re.Match[str]) -> str:
        return m.group(1) + ("#" if _is_unsafe_link(m.group(2)) else m.group(2))

    return re.sub(
        r"^( {0,3}\[[^\]\n]+\]:[ \t]*(?:\r?\n[ \t]*)?)(<[^>\n]*>|\S+)",
        _ref,
        text,
        flags=re.MULTILINE,
    )


def _link_destination_end(text: str, start: int) -> int:
    """Index just past a CommonMark link destination starting at ``start``."""
    if start < len(text) and text[start] == "<":
        close = text.find(">", start)
        newline = text.find("\n", start)
        if close != -1 and (newline == -1 or close < newline):
            return close + 1
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch.isspace() or ord(ch) < 0x20:
            break
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        i += 1
    return min(i, len(text))


def _is_unsafe_link(destination: str) -> bool:
    import html
    import re

    # Decode the way a markdown renderer and then a browser would before checking the scheme.
    target = html.unescape(destination.strip("<>"))
    target = re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", target)
    target = re.sub(r"[\x00-\x20\x7f]", "", target).lower()
    if re.match(r"data:image/(?:gif|png|jpeg|webp);", target):
        return False
    return re.match(r"(?:javascript|vbscript|data):", target) is not None


def _truncate_text(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to about ``max_bytes`` UTF-8 bytes, with a marker of what was dropped.

    Cuts on a character boundary. ``max_bytes <= 0`` disables truncation.
    """
    if max_bytes <= 0:
        return text
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    head = encoded[:max_bytes].decode("utf-8", errors="ignore")
    omitted = len(encoded) - len(head.encode("utf-8"))
    return f"{head}\n\n… {omitted} more bytes truncated"


def _strip_ansi(text: str) -> str:
    """Remove ANSI CSI/OSC escape sequences (colour, progress bars) from terminal output."""
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


def _render_console(cell: CellState, *, max_bytes: int) -> list[Block]:
    """Render persisted stdout/stderr snapshots, with ANSI codes stripped."""
    blocks: list[Block] = []
    stdout = _truncate_text(_strip_ansi(cell.console_stdout or "").rstrip(), max_bytes)
    stderr = _truncate_text(_strip_ansi(cell.console_stderr or "").rstrip(), max_bytes)
    if stdout:
        blocks.append(CodeBlock(language="text", body=stdout, title="stdout"))
    if stderr:
        blocks.append(CodeBlock(language="text", body=stderr, title="stderr"))
    return blocks


def _render_error(cell: CellState, *, max_bytes: int) -> list[Block]:
    """The error the cell's last run ended with, while it still describes it.

    A failed run stores no display output, so without this the export shows no
    sign of failure. ``current_error`` is empty once the source has changed.
    """
    error = cell.current_error()
    if not error:
        return []
    body = _truncate_text(_strip_ansi(error).rstrip(), max_bytes)
    return [CodeBlock(language="text", body=body, title="Error")]


_PREVIEW_ROW_LIMIT = 20


def _render_display_output(
    output: CellOutput,
    *,
    notebook_dir: Path,
    notebook_id: str,
    max_bytes: int,
) -> list[Block]:
    """Per-content-type renderer for one persisted cell output.

    Image and markdown payloads are not persisted inline; they are loaded from
    the artifact store when ``artifact_uri`` points at one.
    """
    if output.error:
        return [
            CodeBlock(language="text", body=output.error.rstrip(), title="Error"),
        ]

    output = _hydrate_output(output, notebook_dir=notebook_dir, notebook_id=notebook_id)
    ctype = output.content_type

    if ctype == "image/png" and output.inline_data_url:
        data_url = output.inline_data_url
        if max_bytes > 0 and len(data_url) > max_bytes:
            kb = len(data_url) // 1024
            return [
                NoteBlock(
                    f"Image output ({kb} KB), too large to inline at the "
                    f"current size cap. Re-export with "
                    f"`--max-output-bytes {len(data_url) + 1024}` to include it."
                )
            ]
        return [ImageBlock(data_url=data_url, alt="cell output")]

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
        # No columns: fall through to scalar/repr below.

    if ctype == "json/object":
        body = _truncate_text(_format_json_preview(output.preview), max_bytes)
        return [CodeBlock(language="json", body=body, title="Output")]

    if ctype == "pickle/object":
        # serializer.py stores a "<TypeName object>" hint in preview for
        # pickled values; surface it so the reader knows what kind of
        # opaque blob the cell produced.
        hint = output.preview if isinstance(output.preview, str) else None
        if hint:
            return [NoteBlock(f"Pickled output ({hint}), not rendered in export.")]
        return [NoteBlock("Pickled output, not rendered in export.")]

    # Fallback: render the preview as text. Covers scalars (json content
    # type with a scalar preview, plain int/str values, etc.) and any
    # exotic content type we haven't special-cased.
    preview = output.preview
    if preview is None:
        return []
    body = _truncate_text(_format_scalar_preview(preview), max_bytes)
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
    """Coerce table-preview rows (positional lists or dicts) into dict-keyed rows.

    Short rows are padded with None, long rows truncated, other entries dropped.
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
        # list) is silently dropped; better to render a small table
        # than to error during export.
    return out


def _format_scalar_preview(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    return repr(value)


def _hydrate_output(output: CellOutput, *, notebook_dir: Path, notebook_id: str) -> CellOutput:
    """Re-attach the inline image/markdown payload, which is not persisted, from the store.

    Best effort: if the store or blob is unavailable, the output is returned as-is.
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
    """Parse a canonical ``strata://artifact/<id>@v=<n>`` URI.

    Splits on the last ``@v=``: a fan-out instance's id has an ``@`` of its own.
    """
    artifact_id, sep, version = artifact_uri.split("/")[-1].rpartition("@v=")
    if not sep:
        raise ValueError(f"not a versioned artifact URI: {artifact_uri!r}")
    return artifact_id, int(version)


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


# --- Variant resolution ---
#
# parse_notebook() doesn't set variant_* fields (the session's DAG build does),
# and export runs without a session, so replicate that slice here using the
# DAG layer's first-in-source-order fallback.


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


# --- README discovery ---


def _load_readme(notebook_dir: Path) -> str | None:
    readme_path = notebook_dir / "README.md"
    if not readme_path.is_file():
        return None
    try:
        return readme_path.read_text(encoding="utf-8")
    except OSError:
        return None


# --- Emitters ---


def _emit_markdown(blocks: list[Block]) -> str:
    """Walk the block tree and emit CommonMark."""
    from html import escape

    # Names, widget values, notes and table values come from cells and data; the
    # published page renders raw HTML, so they get the markdown-cell sanitizer too.
    inline = _sanitize_markdown_prose
    pieces: list[str] = []
    for block in blocks:
        if isinstance(block, HeadingBlock):
            pieces.append(f"{'#' * block.level} {inline(block.text)}")
        elif isinstance(block, MarkdownBlock):
            pieces.append(_sanitize_markdown_body(block.body).rstrip("\n"))
        elif isinstance(block, CodeBlock):
            fence = "`" * _fence_length_for(block.body)
            title_suffix = f' title="{block.title}"' if block.title else ""
            pieces.append(
                f"{fence}{block.language}{title_suffix}\n{block.body.rstrip()}\n{fence}",
            )
        elif isinstance(block, ChipsBlock):
            chip_text = "  ·  ".join(f"**{inline(k)}** {inline(v)}" for k, v in block.items)
            pieces.append(f"<sub>{chip_text}</sub>")
        elif isinstance(block, NoteBlock):
            pieces.append(f"*{inline(block.text)}*")
        elif isinstance(block, ImageBlock):
            # An imported snapshot's display URL is untrusted: only an inline image is emitted.
            if block.data_url.startswith("data:image/"):
                src, alt = escape(block.data_url, quote=True), escape(block.alt, quote=True)
                pieces.append(f'<img src="{src}" alt="{alt}">')
            else:
                pieces.append("*Image output omitted: its source is not an inline image.*")
        elif isinstance(block, TableBlock):
            pieces.append(_emit_markdown_table(block))
    return "\n\n".join(pieces) + "\n"


def _emit_markdown_table(block: TableBlock) -> str:
    """Render a TableBlock as a GitHub-flavored markdown table."""
    columns = list(block.columns)
    header = "| " + " | ".join(_format_table_cell(str(col)) for col in columns) + " |"
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

    title_line = f"**{_sanitize_markdown_prose(block.title)}**\n\n" if block.title else ""
    return title_line + "\n".join([header, separator, *body_lines]) + suffix


def _fence_length_for(body: str) -> int:
    """Return the minimum number of backticks needed to fence ``body``.

    The fence must be longer than any backtick run in the body, or it closes
    early; prompt templates often embed fenced examples.
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
    if isinstance(value, float):
        # 5.000000 -> 5.0, with enough precision for stats tables.
        return f"{value:.4g}"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return _sanitize_markdown_prose(text)


def _emit_html(blocks: list[Block], *, title: str) -> str:
    """Render the block tree as a standalone HTML document.

    Code is highlighted server-side with Pygments; images are inline ``data:``
    URLs. Markdown content is shown as preformatted source, to avoid a
    markdown-to-HTML dependency; use ``--to markdown`` for prose fidelity.
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
        # See _emit_html for why this is <pre> rather than rendered HTML.
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
    """Embedded CSS for the standalone HTML export: a clean document, not the editor look."""
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
