"""Import .ipynb files into Strata notebook directories.

Public entry: :func:`import_notebook` takes a path to a Jupyter
notebook file and produces a runnable Strata notebook directory.

PR 1 scope (this file): parse + convert markdown and code cells only.
``;``-suppression of display is preserved. Magic translation, shell
command extraction, and dependency capture land in PR 2.

The conversion is intentionally light. Most Jupyter notebooks that
run top-to-bottom and don't depend on magics produce a valid Strata
notebook with no further intervention — variable rebinding
(``df = transform(df)``) flows through the DAG via the existing
defines/references analysis, and the harness already auto-displays
the last bare expression so no ``display(...)`` wrapping is needed.

Design doc: ``docs/internal/design-jupyter-import.md``.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)

_SUPPRESSED_COMMENT = "# strata: trailing ';' from Jupyter preserved as display-suppression"

# A trailing ';' may be followed by an inline comment ("df;  # don't print")
# or simply trailing whitespace. Both forms are common in real notebooks.
_SUPPRESSION_TAIL_RE = re.compile(r";[ \t]*(?:#[^\n]*)?\s*\Z")


@dataclass
class ImportResult:
    """Outcome of an ``.ipynb`` import.

    Surfaced through the CLI and (in a later PR) the REST endpoint so
    the user knows what landed in the new notebook directory, what
    we elided, and what they may want to fix by hand.
    """

    notebook_dir: Path
    markdown_cells: int = 0
    code_cells: int = 0
    suppressed_outputs: int = 0  # cells where we preserved Jupyter's ; suppression
    skipped_cells: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def import_notebook(
    ipynb_path: Path | str,
    out_dir: Path | str | None = None,
) -> ImportResult:
    """Convert a Jupyter ``.ipynb`` file into a Strata notebook directory.

    Args:
        ipynb_path: Path to the source ``.ipynb`` file.
        out_dir: Target notebook directory. If ``None``, a sibling
            directory named after the ``.ipynb`` stem is created next
            to the source file.

    Returns:
        An :class:`ImportResult` describing what got converted.
    """
    ipynb_path = Path(ipynb_path)
    if not ipynb_path.is_file():
        raise FileNotFoundError(f"No such file: {ipynb_path}")

    with ipynb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    if out_dir is not None:
        out_dir = Path(out_dir)
        parent = out_dir.parent
        name = out_dir.name
    else:
        parent = ipynb_path.parent
        name = ipynb_path.stem

    notebook_dir = create_notebook(parent, name, initialize_environment=False)
    result = ImportResult(notebook_dir=notebook_dir)

    prev_cell_id: str | None = None
    for cell in nb.get("cells") or []:
        cell_type = cell.get("cell_type")
        source = _source_to_text(cell.get("source", ""))

        if cell_type == "markdown":
            cell_id = _new_cell_id("md")
            add_cell_to_notebook(
                notebook_dir,
                cell_id,
                after_cell_id=prev_cell_id,
                language="markdown",
            )
            write_cell(notebook_dir, cell_id, _ensure_final_newline(source))
            result.markdown_cells += 1
            prev_cell_id = cell_id
        elif cell_type == "code":
            cell_id = _new_cell_id("cell")
            converted, suppressed = _convert_code_source(source)
            add_cell_to_notebook(
                notebook_dir,
                cell_id,
                after_cell_id=prev_cell_id,
                language="python",
            )
            write_cell(notebook_dir, cell_id, converted)
            result.code_cells += 1
            if suppressed:
                result.suppressed_outputs += 1
            prev_cell_id = cell_id
        elif cell_type is None:
            # Malformed cell; nbformat requires cell_type. Skip silently
            # rather than refusing the whole import.
            result.warnings.append("cell missing 'cell_type' was skipped")
        else:
            # Raw cells and any future cell types we don't model.
            result.skipped_cells.append(str(cell_type))

    return result


# ---------------------------------------------------------------------------
# Source conversion


def _source_to_text(source: Any) -> str:
    """nbformat allows ``source`` as either a string or a list of lines.

    The list form is the canonical on-disk shape; the string form
    appears in hand-edited notebooks and in some exporters.
    """
    if isinstance(source, list):
        return "".join(source)
    if source is None:
        return ""
    return str(source)


def _new_cell_id(prefix: str) -> str:
    """Cell IDs follow the existing 8-char UUID-prefix convention."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _ensure_final_newline(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def _convert_code_source(source: str) -> tuple[str, bool]:
    """Apply minimal source transforms to a Jupyter code cell body.

    PR 1: only the trailing-``;`` display-suppression convention is
    handled. The Strata harness already eval+displays the last bare
    expression natively, so no ``display(...)`` wrapping is required
    for cells without the suppression marker.

    Returns ``(converted_source, suppressed_flag)``.
    """
    if not source:
        return "", False
    body = source
    suppressed = False
    if _ends_with_display_suppression(body):
        body = _suppress_last_expression(body)
        suppressed = True
    return _ensure_final_newline(body), suppressed


def _ends_with_display_suppression(source: str) -> bool:
    """True if the source ends in Jupyter's ``;`` suppression idiom.

    In Jupyter, a trailing semicolon on the cell's last expression
    suppresses the auto-displayed value. Python's parser treats ``;``
    purely as a statement separator with no effect on parsed AST, so
    we detect the convention textually. Both bare (``df;``) and
    commented (``df;  # quiet``) forms count.
    """
    return _SUPPRESSION_TAIL_RE.search(source) is not None


def _suppress_last_expression(source: str) -> str:
    """Rewrite a ``;``-suppressed cell so Strata won't auto-display its last expr.

    The harness auto-displays the value of a final ``ast.Expr`` node.
    To preserve Jupyter's suppression semantics we strip the trailing
    ``;`` and append a ``pass`` statement, which becomes the new last
    node — no display, side effects of the expression itself still
    evaluate. If the body has no parsable bare-expression tail, leave
    it alone.
    """
    match = _SUPPRESSION_TAIL_RE.search(source)
    if match is None:
        return source
    body = source[: match.start()].rstrip()
    if not body:
        return source

    try:
        tree = ast.parse(body)
    except SyntaxError:
        # Don't break import on cells that won't parse; the cell will
        # surface its syntax error at run time, the user can fix it.
        return source

    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        # No bare expression to suppress; leave source as-is (just
        # without the dangling ;).
        return body + "\n"
    return f"{body}\n{_SUPPRESSED_COMMENT}\npass\n"
