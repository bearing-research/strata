"""Import .ipynb files into Strata notebook directories.

:func:`import_notebook` converts markdown and code cells, translates magics and
``!shell`` lines, captures dependencies, and writes ``import_report.md``. The
conversion is light: rebinding such as ``df = transform(df)`` already flows
through the DAG, and the harness auto-displays a final bare expression.

Design doc: ``docs/internal/design-jupyter-import.md``.
"""

from __future__ import annotations

import ast
import io
import json
import re
import shlex
import sys
import tokenize
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import tomli_w
from packaging.requirements import InvalidRequirement, Requirement

from strata.notebook.models import CellLanguage
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)

_SUPPRESSED_COMMENT = "# strata: trailing ';' from Jupyter preserved as display-suppression"

# Tokens IPython skips when it looks for the ';' that ends the last statement.
_TRAILING_TRIVIA = frozenset(
    {
        tokenize.ENDMARKER,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
        tokenize.INDENT,
        tokenize.DEDENT,
    }
)

_LINE_MAGIC_RE = re.compile(r"^(\s*)%([a-zA-Z_]\w*)([^\n]*)$")
_CELL_MAGIC_RE = re.compile(r"\A[ \t]*%%([a-zA-Z_]\w*)([^\n]*)\n?")
_SHELL_RE = re.compile(r"^(\s*)!(.*)$")
# ``files = !ls /data``: IPython binds the lhs to stdout lines. Unhandled,
# the line breaks Python's parser.
_SHELL_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_]\w*)(\s*=\s*)!(.+)$")
# %timeit's options: -n<N> -r<R> -p<P> take a number; -t -c -q -o are flags.
_TIMEIT_OPTIONS_RE = re.compile(r"\A(?:-[nrp]\s*\d+\s+|-[tcqo]\s+)+")
_PIP_INSTALL_RE = re.compile(
    r"^\s*(?:pip|pip3|python\s+-m\s+pip|uv\s+pip)\s+install\s+(.+)$",
)


# --- Result types ---


@dataclass
class _CellConversion:
    """Per-cell conversion output. Aggregated into :class:`ImportResult`."""

    source: str
    suppressed: bool = False
    deps: list[str] = field(default_factory=list)
    translated_magics: list[str] = field(default_factory=list)
    dropped_magics: list[str] = field(default_factory=list)
    dropped_shells: list[str] = field(default_factory=list)
    # ``# @env`` lines, hoisted: an annotation below the first code line is ignored.
    env_annotations: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    """Outcome of an ``.ipynb`` import: what landed, what was elided, what to fix by hand."""

    notebook_dir: Path
    markdown_cells: int = 0
    code_cells: int = 0
    suppressed_outputs: int = 0
    skipped_cells: list[str] = field(default_factory=list)
    translated_magics: list[str] = field(default_factory=list)
    dropped_magics: list[str] = field(default_factory=list)
    dropped_shells: list[str] = field(default_factory=list)
    captured_deps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Set by ``import_notebook``; the text is kept so REST can return it without re-reading.
    report_path: Path | None = None
    report_text: str = ""


# --- Public API ---


def import_notebook(
    ipynb_path: Path | str,
    out_dir: Path | str | None = None,
    *,
    check_deps: bool = False,
) -> ImportResult:
    """Convert a Jupyter ``.ipynb`` file into a Strata notebook directory.

    Args:
        ipynb_path: Path to the source ``.ipynb`` file.
        out_dir: Target directory; ``None`` creates a sibling named after the stem.
        check_deps: Run ``uv lock`` to verify the captured dependencies resolve;
            failures become warnings. Off by default: it needs uv and the network.

    Returns:
        An :class:`ImportResult`. The converted notebook is always checked for
        openability; failures land in ``result.warnings`` rather than raising.

    Raises:
        FileNotFoundError: if ``ipynb_path`` doesn't exist.
        ValueError: if the file isn't a valid nbformat object.
    """
    ipynb_path = Path(ipynb_path)
    if not ipynb_path.is_file():
        raise FileNotFoundError(f"No such file: {ipynb_path}")

    with ipynb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # Validate up front so a malformed source leaves no half-built directory and
    # surfaces as a 400, not a 500 from a mid-loop AttributeError.
    _validate_nbformat_structure(nb)

    if out_dir is not None:
        out_dir = Path(out_dir)
        parent = out_dir.parent
        name = out_dir.name
    else:
        parent = ipynb_path.parent
        name = ipynb_path.stem

    notebook_dir = create_notebook(parent, name, initialize_environment=False)
    result = ImportResult(notebook_dir=notebook_dir)

    sibling_deps = _capture_sibling_deps(ipynb_path.parent)
    local_modules = _local_module_names(ipynb_path.parent)
    is_r = _kernel_language(nb) == "r"
    if is_r:
        result.warnings.append(
            "R kernel: code cells imported as R cells, as written; R packages are not "
            "captured, so install them in system R or with renv"
        )

    prev_cell_id: str | None = None
    cell_deps: list[str] = []
    scanned_imports: set[str] = set()
    for cell in nb.get("cells") or []:
        cell_type = cell.get("cell_type")
        source = _source_to_text(cell.get("source", ""))

        if cell_type == "markdown":
            cell_id = _new_cell_id("md")
            add_cell_to_notebook(
                notebook_dir,
                cell_id,
                after_cell_id=prev_cell_id,
                language=CellLanguage.MARKDOWN,
            )
            write_cell(notebook_dir, cell_id, _ensure_final_newline(source))
            result.markdown_cells += 1
            prev_cell_id = cell_id
        elif cell_type == "code" and is_r:
            # IPython magics, `!` lines and `;` suppression are Python-kernel syntax.
            cell_id = _new_cell_id("cell")
            add_cell_to_notebook(
                notebook_dir, cell_id, after_cell_id=prev_cell_id, language=CellLanguage.R
            )
            write_cell(notebook_dir, cell_id, _ensure_final_newline(source))
            result.code_cells += 1
            prev_cell_id = cell_id
        elif cell_type == "code":
            cell_id = _new_cell_id("cell")
            conv = _convert_code_source(source)
            add_cell_to_notebook(
                notebook_dir,
                cell_id,
                after_cell_id=prev_cell_id,
                language=CellLanguage.PYTHON,
            )
            write_cell(notebook_dir, cell_id, conv.source)
            result.code_cells += 1
            if conv.suppressed:
                result.suppressed_outputs += 1
            result.translated_magics.extend(conv.translated_magics)
            result.dropped_magics.extend(conv.dropped_magics)
            result.dropped_shells.extend(conv.dropped_shells)
            cell_deps.extend(conv.deps)
            # Converted source: magics are stripped, so it parses as Python.
            scanned_imports |= _scan_imports(conv.source)
            prev_cell_id = cell_id
        elif cell_type is None:
            result.warnings.append("cell missing 'cell_type' was skipped")
        else:
            result.skipped_cells.append(str(cell_type))

    inferred_deps = _imports_to_deps(scanned_imports, local_modules)

    # Explicit sources (siblings, %pip install) come first so their pins shadow
    # scan-derived names. Dedup is by PEP 503-normalized name.
    all_deps = _dedupe_by_package([*sibling_deps, *cell_deps, *inferred_deps])
    # Drop pip-only forms (editable installs, URLs, paths) that pyproject can't
    # represent: uv rejects them, or they corrupt the TOML.
    valid_deps = [d for d in all_deps if _is_valid_pep508_dep(d)]
    rejected_deps = [d for d in all_deps if not _is_valid_pep508_dep(d)]
    if valid_deps:
        _merge_pyproject_deps(notebook_dir, valid_deps)
    result.captured_deps = valid_deps
    if rejected_deps:
        sample = ", ".join(repr(d) for d in rejected_deps[:3])
        more = "" if len(rejected_deps) <= 3 else f" (+{len(rejected_deps) - 3} more)"
        result.warnings.append(
            f"{len(rejected_deps)} pip-only dep spec(s) skipped: pyproject.toml "
            f"requires PEP 508 specifiers: {sample}{more}"
        )

    # Run the parse/analyze/DAG pass NotebookSession runs on open, so errors the
    # user would hit on open show up in the import report instead.
    _check_openable(notebook_dir, result)

    # Opt-in: seconds-slow on cold caches and requires the uv CLI.
    if check_deps:
        _check_resolvable(notebook_dir, result)

    report_text = format_import_report(result, ipynb_path)
    report_path = notebook_dir / "import_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    result.report_path = report_path
    result.report_text = report_text

    return result


# --- Post-import sanity checks ---


def _check_openable(notebook_dir: Path, result: ImportResult) -> None:
    """Parse, analyze and DAG-build the converted notebook; warn on the first failure.

    Never raises: the notebook is already on disk, and a partial success is more
    useful than a refused import.
    """
    # Local: these modules import this one transitively through parser test fixtures.
    from strata.notebook.analyzer import analyze_cell
    from strata.notebook.dag import CellAnalysisWithId, NotebookDag
    from strata.notebook.parser import parse_notebook

    try:
        nb_state = parse_notebook(notebook_dir)
    except Exception as exc:
        result.warnings.append(f"converted notebook fails to parse: {type(exc).__name__}: {exc}")
        return

    analyses: list[CellAnalysisWithId] = []
    for cell in nb_state.cells:
        if cell.language != CellLanguage.PYTHON:
            continue
        cell_analysis = analyze_cell(cell.source)
        if cell_analysis.error:
            result.warnings.append(f"cell {cell.id[:8]} fails to analyze: {cell_analysis.error}")
            continue
        analyses.append(
            CellAnalysisWithId(
                id=cell.id,
                defines=cell_analysis.defines,
                references=cell_analysis.references,
                builtin_references=cell_analysis.builtin_references,
            )
        )

    try:
        NotebookDag.from_cells(analyses)
    except Exception as exc:
        result.warnings.append(f"DAG build fails: {type(exc).__name__}: {exc}")


def _check_resolvable(notebook_dir: Path, result: ImportResult) -> None:
    """Run ``uv lock`` to verify the captured dependencies resolve; warn on failure.

    On success the written ``uv.lock`` seeds the notebook's first ``uv sync``.
    """
    import subprocess

    from strata.notebook.dependencies import uv_env

    try:
        completed = subprocess.run(
            ["uv", "lock"],
            cwd=str(notebook_dir),
            env=uv_env(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        result.warnings.append("--check-deps skipped: uv not found on PATH")
        return
    except subprocess.TimeoutExpired:
        result.warnings.append("--check-deps timed out after 120s")
        return

    if completed.returncode != 0:
        # Trimmed; ``uv lock`` in the notebook directory shows the full failure.
        detail = completed.stderr.strip() or completed.stdout.strip() or "(no output)"
        result.warnings.append(f"dependency resolution failed: {detail[:400]}")


# --- Structural validation ---


def _kernel_language(nb: dict[str, Any]) -> str:
    """Lowercased kernel language from the notebook metadata; ``""`` when absent."""
    metadata = nb.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    for key, field_name in (("kernelspec", "language"), ("language_info", "name")):
        section = metadata.get(key)
        if isinstance(section, dict) and isinstance(section.get(field_name), str):
            return section[field_name].strip().lower()
    return ""


def _validate_nbformat_structure(nb: object) -> None:
    """Reject nbformat shapes that would crash the converter mid-loop.

    Fails on a non-object top level, a non-list ``cells``, or a non-object cell
    entry. Subtler problems (missing version, unknown ``cell_type``) are converted
    as best as possible.
    """
    if not isinstance(nb, dict):
        raise ValueError(
            f"Invalid .ipynb: expected JSON object at top level, got {type(nb).__name__}"
        )
    # ``cast`` only: ty narrows the isinstance to dict[Unknown, Unknown], whose
    # ``.get`` takes ``Never``. A JSON parse is str-keyed.
    nb_dict = cast(dict[str, Any], nb)
    cells = nb_dict.get("cells")
    if cells is not None and not isinstance(cells, list):
        raise ValueError(f"Invalid .ipynb: 'cells' must be a list, got {type(cells).__name__}")
    for idx, cell in enumerate(cells or []):
        if not isinstance(cell, dict):
            raise ValueError(
                f"Invalid .ipynb: cells[{idx}] must be a JSON object, got {type(cell).__name__}"
            )


# --- Import report ---


def format_import_report(result: ImportResult, ipynb_path: Path | str) -> str:
    """Build the human-readable conversion report for one import.

    Sections appear only when they have content.
    """
    ipynb_path = Path(ipynb_path)
    lines: list[str] = [
        f"# Imported from {ipynb_path.name}",
        "",
        f"- Source: `{ipynb_path}`",
        f"- Target: `{result.notebook_dir}`",
        "",
        "## Counts",
        "",
        f"- Markdown cells: {result.markdown_cells}",
        f"- Code cells: {result.code_cells}",
    ]
    if result.suppressed_outputs:
        lines.append(
            f"- Cells with `;`-display-suppression preserved: {result.suppressed_outputs}",
        )
    if result.skipped_cells:
        kinds = ", ".join(f"`{k}`" for k in sorted(set(result.skipped_cells)))
        lines.append(
            f"- Skipped cell type(s): {kinds} ({len(result.skipped_cells)} cells)",
        )

    if result.translated_magics:
        lines.extend(
            [
                "",
                "## Magics translated",
                "",
                "These were rewritten or absorbed into the imported notebook.",
                "",
            ]
        )
        lines.extend(f"- `{m}`" for m in result.translated_magics)

    if result.dropped_magics:
        lines.extend(
            [
                "",
                "## Magics dropped",
                "",
                "Strata doesn't translate these; the source carries a "
                "`# strata: ...` marker comment where each one lived. Inspect "
                "the affected cells if behavior depends on them.",
                "",
            ]
        )
        lines.extend(f"- `{m}`" for m in result.dropped_magics)

    if result.dropped_shells:
        lines.extend(
            [
                "",
                "## Shell commands dropped",
                "",
                "Auto-running shell from an untrusted notebook is a real "
                "hazard, so `!cmd` lines (except `!pip install ...`) are "
                "dropped. Wrap in `subprocess.run(...)` by hand if needed.",
                "",
            ]
        )
        lines.extend(f"- `{s}`" for s in result.dropped_shells)

    if result.captured_deps:
        lines.extend(
            [
                "",
                "## Dependencies captured",
                "",
                "Added to `pyproject.toml`. First `uv sync` resolves them.",
                "",
            ]
        )
        lines.extend(f"- `{d}`" for d in result.captured_deps)

    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in result.warnings)

    return "\n".join(lines) + "\n"


# --- Source conversion ---


def _source_to_text(source: Any) -> str:
    """Join an nbformat ``source`` (string or list of lines) into stripped text.

    Stripping fixes hand-edited cells with a leading space (``" Image(...)"``) that
    would otherwise be an indent error.
    """
    if isinstance(source, list):
        text = "".join(source)
    elif source is None:
        text = ""
    else:
        text = str(source)
    # Fixes the common " Image(...) " case; a genuinely indented first line
    # still surfaces as a syntax error later.
    return text.strip() + "\n" if text.strip() else ""


def _new_cell_id(prefix: str) -> str:
    """A new cell id (8-char UUID prefix)."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _ensure_final_newline(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def _convert_code_source(source: str) -> _CellConversion:
    """Convert one Jupyter code cell to runnable Python.

    A ``%%`` cell magic dispatches the whole body to one handler; otherwise line
    magics and ``!cmd`` lines are translated line by line. ``;`` suppression is
    applied last, since a translated magic can change the final expression.
    """
    if not source:
        return _CellConversion(source="")

    cell_magic = _CELL_MAGIC_RE.match(source)
    if cell_magic:
        name = cell_magic.group(1)
        args = cell_magic.group(2).strip()
        body = source[cell_magic.end() :]
        return _translate_cell_magic(name, args, body)

    out_lines: list[str] = []
    conv = _CellConversion(source="")
    lines = source.splitlines(keepends=True)
    for raw_line, at_statement in zip(lines, _statement_starts(lines), strict=True):
        line_no_eol = raw_line.rstrip("\n")
        # A ``%`` or ``!`` inside a string or a bracketed expression is Python, not a magic.
        replacement = _translate_escape_line(line_no_eol, conv) if at_statement else None
        if replacement is None:
            out_lines.append(raw_line)
            continue
        indent = line_no_eol[: len(line_no_eol) - len(line_no_eol.lstrip())]
        if indent and not any(_is_code_line(line) for line in replacement):
            # The magic may have been its block's only statement.
            replacement = [*replacement, f"{indent}pass\n"]
        out_lines.extend(replacement)

    result_source = "".join(conv.env_annotations) + "".join(out_lines)
    if _ends_with_display_suppression(result_source):
        result_source = _suppress_last_expression(result_source)
        conv.suppressed = True
    conv.source = _ensure_final_newline(result_source)
    return conv


def _translate_escape_line(line: str, conv: _CellConversion) -> list[str] | None:
    """Replacement lines for a ``%magic`` or ``!shell`` line, or ``None`` for plain Python."""
    line_magic = _LINE_MAGIC_RE.match(line)
    if line_magic:
        indent, magic_name, magic_args = line_magic.groups()
        return _translate_line_magic(magic_name, magic_args.lstrip(), indent, conv)
    shell_assign = _SHELL_ASSIGN_RE.match(line)
    if shell_assign:
        indent, target, eq, cmd = shell_assign.groups()
        return _translate_shell_assignment(target, eq, cmd.strip(), indent, conv)
    shell = _SHELL_RE.match(line)
    if shell:
        indent, cmd = shell.groups()
        return _translate_shell(cmd.strip(), indent, conv)
    return None


def _is_code_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _statement_starts(lines: list[str]) -> list[bool]:
    """For each line, whether it begins a statement.

    False inside a triple-quoted string, an open bracket or after a backslash
    continuation. A magic or shell line is not scanned, so a quote in its argument
    cannot open a string. Source with magics does not tokenize, hence the hand scan.
    """
    starts: list[bool] = []
    quote: str | None = None
    depth = 0
    continued = False
    for raw in lines:
        line = raw.rstrip("\n")
        at_start = quote is None and depth == 0 and not continued
        starts.append(at_start)
        if at_start and (
            _LINE_MAGIC_RE.match(line) or _SHELL_ASSIGN_RE.match(line) or _SHELL_RE.match(line)
        ):
            continued = False
            continue
        i = 0
        while i < len(line):
            if quote is not None:
                if line[i] == "\\":
                    i += 2
                elif line.startswith(quote, i):
                    i += len(quote)
                    quote = None
                else:
                    i += 1
                continue
            char = line[i]
            if char == "#":
                break
            if char in "\"'":
                quote = line[i : i + 3] if line[i : i + 3] in ('"""', "'''") else char
                i += len(quote)
                continue
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
            i += 1
        continued = line.endswith("\\")
        if quote in ('"', "'") and not continued:
            quote = None  # an unterminated one-line string ends with its line
    return starts


def _suppression_offset(source: str) -> int | None:
    """Offset of the ``;`` that ends the last statement, as IPython finds it.

    Comments and blank lines after it do not count, so ``df;  # quiet`` is
    suppressed and ``df`` followed by a ``# done;`` comment line is not.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return None
    last = next((tok for tok in reversed(tokens) if tok.type not in _TRAILING_TRIVIA), None)
    if last is None or last.type != tokenize.OP or last.string != ";":
        return None
    row, col = last.start
    # The tokenizer's own line split: ``splitlines`` also breaks on U+2028, form feed
    # and a bare ``\r``, which would put the cut mid-statement.
    return sum(len(line) for line in io.StringIO(source).readlines()[: row - 1]) + col


def _ends_with_display_suppression(source: str) -> bool:
    """True if the source ends in Jupyter's ``;`` suppression (``df;`` or ``df;  # quiet``)."""
    return _suppression_offset(source) is not None


def _suppress_last_expression(source: str) -> str:
    """Append ``pass`` so the harness does not auto-display a ``;``-suppressed last expression."""
    offset = _suppression_offset(source)
    if offset is None:
        return source
    body = source[:offset].rstrip()
    if not body:
        return source

    try:
        tree = ast.parse(body)
    except SyntaxError:
        return source

    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return body + "\n"
    return f"{body}\n{_SUPPRESSED_COMMENT}\npass\n"


# --- Magic translation ---


def _translate_line_magic(
    name: str,
    args: str,
    indent: str,
    conv: _CellConversion,
) -> list[str]:
    """Dispatch a ``%name args`` line magic; return the replacement lines.

    Records metadata on ``conv`` in place.
    """
    handler = _LINE_MAGIC_TABLE.get(name)
    if handler is None:
        conv.dropped_magics.append(f"%{name}")
        return [f"{indent}# strata: unsupported magic '%{name}' dropped\n"]
    return handler(name, args, indent, conv)


def _translate_cell_magic(name: str, args: str, body: str) -> _CellConversion:
    handler = _CELL_MAGIC_TABLE.get(name)
    if handler is None:
        return _CellConversion(
            source=f"# strata: unsupported cell magic '%%{name}' dropped\n",
            dropped_magics=[f"%%{name}"],
        )
    return handler(name, args, body)


# --- line-magic handlers ---


def _lm_drop(name: str, args: str, indent: str, conv: _CellConversion) -> list[str]:
    conv.translated_magics.append(f"%{name}")
    return []


def _lm_strip(name: str, args: str, indent: str, conv: _CellConversion) -> list[str]:
    """``%timeit -n 10 body`` -> ``body`` (drop the timing wrapper and its options)."""
    conv.translated_magics.append(f"%{name}")
    args = _TIMEIT_OPTIONS_RE.sub("", args)
    if args:
        return [f"{indent}{args}\n"]
    return []


def _lm_pip(name: str, args: str, indent: str, conv: _CellConversion) -> list[str]:
    """``%pip install pkg`` captures packages; other subcommands are dropped."""
    parts = args.strip().split(None, 1)
    subcommand = parts[0] if parts else ""
    if subcommand != "install":
        conv.dropped_magics.append(f"%{name} {subcommand}".strip())
        return [
            f"{indent}# strata: %{name} {subcommand} dropped (only 'install' is captured)\n",
        ]
    rest = parts[1] if len(parts) > 1 else ""
    packages = _parse_pip_install(rest)
    conv.deps.extend(packages)
    conv.translated_magics.append(f"%{name} install {' '.join(packages)}")
    return []


def _lm_env(name: str, args: str, indent: str, conv: _CellConversion) -> list[str]:
    """``%env KEY=VAL`` -> a ``# @env KEY=VAL`` annotation at the top of the cell."""
    if "=" not in args:
        conv.dropped_magics.append(f"%{name} (no KEY=VALUE)")
        return [f"{indent}# strata: %env requires KEY=VALUE; dropped\n"]
    conv.translated_magics.append(f"%{name}")
    conv.env_annotations.append(f"# @env {args.strip()}\n")
    return []


def _lm_run(name: str, args: str, indent: str, conv: _CellConversion) -> list[str]:
    """``%run script.py`` becomes an ``exec`` of the script's text (best effort).

    Imports ``pathlib.Path`` under an alias, so the cell need not import it.
    """
    target = args.strip()
    if not target:
        conv.dropped_magics.append(f"%{name} (no target)")
        return [f"{indent}# strata: %run with no target dropped\n"]
    conv.translated_magics.append(f"%{name} {target}")
    return [
        f"{indent}# strata: %run translated; verify the path resolves at runtime\n",
        f"{indent}from pathlib import Path as _strata_path\n",
        f"{indent}exec(_strata_path({target!r}).read_text())\n",
    ]


_LINE_MAGIC_TABLE = {
    # Display / rendering setup (no-ops in Strata)
    "matplotlib": _lm_drop,
    "load_ext": _lm_drop,
    "autoreload": _lm_drop,
    "reload_ext": _lm_drop,
    "config": _lm_drop,
    "colors": _lm_drop,
    "rerun": _lm_drop,
    # Debugger / exception controls
    "capture": _lm_drop,
    "xmode": _lm_drop,
    "pdb": _lm_drop,
    "debug": _lm_drop,
    "tb": _lm_drop,
    # Inspection / "what's defined" magics (no Strata equivalent)
    "who": _lm_drop,
    "who_ls": _lm_drop,
    "whos": _lm_drop,
    "lsmagic": _lm_drop,
    "magic": _lm_drop,
    "history": _lm_drop,
    "alias": _lm_drop,
    "alias_magic": _lm_drop,
    # Timing wrappers, keep the body
    "timeit": _lm_strip,
    "time": _lm_strip,
    # Package management, captured as deps
    "pip": _lm_pip,
    "conda": _lm_pip,  # ``%conda install x`` is captured the same as %pip
    # Environment / runtime
    "env": _lm_env,
    "run": _lm_run,
    "set_env": _lm_env,  # alias of %env
}


# --- cell-magic handlers ---


def _cm_strip(name: str, args: str, body: str) -> _CellConversion:
    """``%%timeit body`` → recurse on the body as plain code."""
    inner = _convert_code_source(body)
    inner.translated_magics.insert(0, f"%%{name}")
    return inner


def _cm_drop(name: str, args: str, body: str) -> _CellConversion:
    return _CellConversion(
        source="# strata: cell magic dropped (body not translatable)\n",
        dropped_magics=[f"%%{name}"],
    )


def _cm_bash(name: str, args: str, body: str) -> _CellConversion:
    """``%%bash``/``%%sh``/``%%script``: dropped, body kept as comments.

    Same policy as ``!cmd`` lines: running arbitrary shell from an imported
    (possibly untrusted) notebook is a hazard, so re-enabling it is a deliberate
    uncomment.
    """
    commented = "".join(f"# {line}\n" for line in body.splitlines())
    return _CellConversion(
        source=(
            f"# strata: %%{name} cell dropped (shell is not auto-run on import); "
            "body preserved below\n" + commented
        ),
        dropped_shells=[f"%%{name}"],
    )


def _cm_writefile(name: str, args: str, body: str) -> _CellConversion:
    target = args.strip().strip("'\"")
    if not target:
        return _CellConversion(
            source="# strata: %%writefile with no path dropped\n",
            dropped_magics=["%%writefile (no path)"],
        )
    wrapped = (
        f"from pathlib import Path as _StrataPath\n_StrataPath({target!r}).write_text({body!r})\n"
    )
    return _CellConversion(
        source=wrapped,
        translated_magics=[f"%%writefile {target}"],
    )


_CELL_MAGIC_TABLE = {
    # Timing / capture wrappers, recurse on body
    "timeit": _cm_strip,
    "time": _cm_strip,
    "capture": _cm_strip,
    # Shell-out cell magics: dropped with the body preserved as comments,
    # matching the ``!cmd`` policy (no auto-run shell from imports)
    "bash": _cm_bash,
    "sh": _cm_bash,
    "script": _cm_bash,
    # File-writing magic
    "writefile": _cm_writefile,
    "file": _cm_writefile,  # alias for %%writefile in older IPython
    # Renderer cell magics that have no Strata-display equivalent
    "javascript": _cm_drop,
    "js": _cm_drop,
    "html": _cm_drop,
    "latex": _cm_drop,
    "svg": _cm_drop,
    "markdown": _cm_drop,
    # Other-language cell magics, dropped with marker
    "R": _cm_drop,
    "ruby": _cm_drop,
    "perl": _cm_drop,
    "cython": _cm_drop,
    "fortran": _cm_drop,
    "sql": _cm_drop,  # %%sql binds to a connection that Strata's SQL cell type
    # handles natively; the magic form can't auto-convert.
}


# --- shell translation ---


def _translate_shell(cmd: str, indent: str, conv: _CellConversion) -> list[str]:
    """``!cmd`` lines. Only ``pip install`` is captured; everything else dropped."""
    pip = _PIP_INSTALL_RE.match(cmd)
    if pip:
        packages = _parse_pip_install(pip.group(1))
        conv.deps.extend(packages)
        conv.translated_magics.append(f"!{cmd}")
        return []
    conv.dropped_shells.append(f"!{cmd}")
    return [f"{indent}# strata: shell command dropped: !{cmd}\n"]


def _translate_shell_assignment(
    target: str,
    eq: str,
    cmd: str,
    indent: str,
    conv: _CellConversion,
) -> list[str]:
    """``target = !cmd``: drop the command and bind ``target`` to ``[]``.

    Not run, for the same safety reason as ``!cmd``; the stub keeps later
    references to ``target`` valid. ``!pip install`` here still captures the package.
    """
    pip = _PIP_INSTALL_RE.match(cmd)
    if pip:
        packages = _parse_pip_install(pip.group(1))
        conv.deps.extend(packages)
        conv.translated_magics.append(f"{target} = !{cmd}")
        return [
            f"{indent}{target}{eq}[]  # strata: '{target} = !pip install ...' captured to deps\n",
        ]
    conv.dropped_shells.append(f"{target} = !{cmd}")
    stub = (
        f"{indent}{target}{eq}[]  "
        f"# strata: shell escape '!{cmd}' dropped; restore with subprocess.run if needed\n"
    )
    return [stub]


# ---------------------------------------------------------------------------
# Dependency capture


def _parse_pip_install(args: str) -> list[str]:
    """Extract package specifiers from a ``pip install ...`` argument string.

    Drops flags and flag arguments (``-r req.txt``, ``--index-url ...``); keeps
    version specifiers, URL specs and extras attached to their package.
    """
    try:
        tokens = shlex.split(args)
    except ValueError:
        tokens = args.split()

    consume_next = {
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-i",
        "--index-url",
        "--extra-index-url",
        "--find-links",
        "-f",
        "--no-binary",
        "--only-binary",
        "--prefer-binary",
        "--platform",
        "--python-version",
        "--implementation",
        "--abi",
    }
    skip = False
    packages: list[str] = []
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok in consume_next:
            skip = True
            continue
        if tok.startswith("-"):
            continue
        packages.append(tok)
    return packages


def _capture_sibling_deps(parent: Path) -> list[str]:
    """Read ``requirements.txt`` / ``pyproject.toml`` next to the ``.ipynb``.

    Best effort: an unreadable source contributes no deps.
    """
    deps: list[str] = []
    req = parent / "requirements.txt"
    if req.is_file():
        try:
            text = req.read_text(encoding="utf-8")
        except OSError:
            text = ""
        # pip joins backslash continuations, then drops `  # comment` and `--hash=...`.
        for raw in text.replace("\\\n", " ").splitlines():
            line = _REQUIREMENT_TAIL_RE.split(raw, maxsplit=1)[0].strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            deps.append(line)

    pyproject = parent / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project_deps = data.get("project", {}).get("dependencies", [])
            if isinstance(project_deps, list):
                deps.extend(str(d) for d in project_deps)
        except (OSError, tomllib.TOMLDecodeError):
            pass

    return deps


_REQUIREMENT_TAIL_RE = re.compile(r"\s+(?:#|--?[A-Za-z])")


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


# Import names whose PyPI name differs; anything else is assumed identical.
_IMPORT_TO_PIP: dict[str, str] = {
    # Data science / ML basics
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "PIL": "Pillow",
    # Web / scraping / serialization
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "lxml": "lxml",
    # Crypto / security
    "Crypto": "pycryptodome",
    "OpenSSL": "pyOpenSSL",
    "jwt": "PyJWT",
    # Database drivers
    "MySQLdb": "mysqlclient",
    "psycopg2": "psycopg2-binary",
    "pymongo": "pymongo",
    # Python utility libs that publish under different names
    "attr": "attrs",
    "git": "GitPython",
    "tabulate": "tabulate",
    # Bioinformatics / specialized
    "Bio": "biopython",
    # Common namespace-package collisions
    "google": "google-api-python-client",  # ``import google.auth`` etc.
    "mpl_toolkits": "matplotlib",
    "pkg_resources": "setuptools",
    # Deprecated aliases that users still write
    "gym": "gymnasium",  # gym is unmaintained; gymnasium is the maintained fork
}


# ``google.cloud.<path>`` imports (API version suffix dropped) whose distribution is
# not ``google-cloud-<path>``: the guess is missing from PyPI, or names another project.
_GOOGLE_CLOUD_TO_PIP: dict[str, str] = {
    "accessapproval": "google-cloud-access-approval",
    "alloydb.connector": "google-cloud-alloydb-connector",
    "artifactregistry": "google-cloud-artifact-registry",
    "billing.budgets": "google-cloud-billing-budgets",
    "bigtable_admin": "google-cloud-bigtable",
    # ``google-cloud-dataflow`` is the retired Beam-based SDK, not this client.
    "dataflow": "google-cloud-dataflow-client",
    "devtools.cloudbuild": "google-cloud-build",
    "devtools.containeranalysis": "google-cloud-containeranalysis",
    "dialogflowcx": "google-cloud-dialogflow-cx",
    "errorreporting": "google-cloud-error-reporting",
    "firestore_admin": "google-cloud-firestore",
    "gkehub": "google-cloud-gke-hub",
    "iam_admin": "google-cloud-iam",
    "iam_credentials": "google-cloud-iam",
    "networkconnectivity": "google-cloud-network-connectivity",
    "orgpolicy": "google-cloud-org-policy",
    "osconfig": "google-cloud-os-config",
    "oslogin": "google-cloud-os-login",
    "recaptchaenterprise": "google-cloud-recaptcha-enterprise",
    "resourcemanager": "google-cloud-resource-manager",
    "secretmanager": "google-cloud-secret-manager",
    "servicedirectory": "google-cloud-service-directory",
    "servicemanagement": "google-cloud-service-management",
    "spanner_admin_database": "google-cloud-spanner",
    "spanner_admin_instance": "google-cloud-spanner",
    "spanner_dbapi": "google-cloud-spanner",
    "sql.connector": "cloud-sql-python-connector",
    "vpcaccess": "google-cloud-vpc-access",
}


def _google_cloud_path(name: str) -> str:
    """``google.cloud.pubsub_v1`` -> ``pubsub``: one distribution holds every API version."""
    return re.sub(r"_v\d[a-z0-9]*$", "", name.removeprefix("google.cloud."))


def _scan_imports(source: str) -> set[str]:
    """Collect top-level, non-stdlib module names a cell imports.

    Returns an empty set on a syntax error; that surfaces when the cell runs.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(_distribution_key(alias.name))
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, never a third-party dependency.
            if node.module and node.level == 0:
                if node.module == "google.cloud" or node.module.startswith("google.cloud."):
                    # ``from google.cloud.devtools import cloudbuild_v1`` names its
                    # distribution in the imported name.
                    names.update(
                        _distribution_key(f"{node.module}.{alias.name}") for alias in node.names
                    )
                else:
                    names.add(_distribution_key(node.module))
    return names - sys.stdlib_module_names


def _distribution_key(module: str) -> str:
    """The part of a dotted import that names its distribution.

    The top-level package, except under the ``google.cloud`` namespace, where
    each ``google.cloud.<x>`` is its own ``google-cloud-<x>`` distribution, or
    ``google.cloud.<x>.<y>`` for a nested one such as ``sql.connector``.
    """
    parts = module.split(".")
    if parts[:2] == ["google", "cloud"] and len(parts) > 2:
        nested = ".".join(parts[:4])
        if len(parts) > 3 and _google_cloud_path(nested) in _GOOGLE_CLOUD_TO_PIP:
            return nested
        return ".".join(parts[:3])
    return parts[0]


def _local_module_names(parent_dir: Path) -> set[str]:
    """Names that resolve to local files or packages, not PyPI.

    Keeps ``import my_helpers`` next to ``my_helpers.py`` from becoming a bogus
    dependency that breaks ``uv sync``.
    """
    names: set[str] = set()
    if not parent_dir.is_dir():
        return names
    try:
        entries = list(parent_dir.iterdir())
    except OSError:
        return names
    for entry in entries:
        if entry.is_file() and entry.suffix == ".py" and entry.stem != "__init__":
            names.add(entry.stem)
        elif entry.is_dir() and (entry / "__init__.py").is_file():
            names.add(entry.name)
    return names


def _imports_to_deps(imports: set[str], local_modules: set[str]) -> list[str]:
    """Map import names to sorted pip package specifiers, skipping local modules."""
    deps: list[str] = []
    for name in sorted(imports):
        if name in local_modules:
            continue
        if name.startswith("google.cloud."):
            package = _google_cloud_path(name)
            deps.append(
                _GOOGLE_CLOUD_TO_PIP.get(package, "google-cloud-" + package.replace("_", "-"))
            )
        else:
            deps.append(_IMPORT_TO_PIP.get(name, name))
    return deps


def _normalize_pep503(name: str) -> str:
    """Canonical comparison key for a PEP 508 specifier (PEP 503 name normalization).

    Strips version, extras and markers; ``scikit_learn`` and ``Scikit-Learn`` map
    to the same key, so a pinned sibling dep shadows a bare scanned name.
    """
    head = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", name.strip())
    if not head:
        return ""
    return re.sub(r"[._-]+", "-", head.group(1)).lower()


def _dedupe_by_package(specs: list[str]) -> list[str]:
    """Dedupe in order by normalized package name, so an earlier pin beats a later bare name."""
    seen: set[str] = set()
    out: list[str] = []
    for spec in specs:
        key = _normalize_pep503(spec)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def _is_valid_pep508_dep(spec: str) -> bool:
    """Whether ``spec`` is a PEP 508 specifier ``project.dependencies`` accepts.

    Rejects pip-only forms (editable installs, bare URLs, local paths), which would
    produce invalid TOML or fail in uv.
    """
    try:
        Requirement(spec.strip())
    except InvalidRequirement:
        return False
    return True


def _merge_pyproject_deps(notebook_dir: Path, new_deps: list[str]) -> list[str]:
    """Add captured deps to the new notebook's ``pyproject.toml``; return those added.

    Round-trips through ``tomllib`` + ``tomli_w`` so markers with quotes are
    escaped correctly. Does not run ``uv add`` (slow, networked); the first
    ``uv sync`` resolves them.
    """
    pyproject = notebook_dir / "pyproject.toml"
    if not pyproject.is_file():
        return []

    with pyproject.open("rb") as f:
        data = tomllib.load(f)

    project = data.setdefault("project", {})
    existing = list(project.get("dependencies") or [])
    existing_set = {d.strip() for d in existing if isinstance(d, str)}
    additions = [d for d in new_deps if d.strip() not in existing_set]
    if not additions:
        return []

    project["dependencies"] = existing + additions
    with pyproject.open("wb") as f:
        tomli_w.dump(data, f)
    return additions
