"""Import .ipynb files into Strata notebook directories.

Public entry: :func:`import_notebook` takes a path to a Jupyter
notebook file and produces a runnable Strata notebook directory.

PR 1 scope: parse + convert markdown / code cells, ``;``-suppression.
PR 2 adds: line-magic and cell-magic translation, ``!shell`` handling,
dependency capture from sibling ``requirements.txt`` / ``pyproject.toml``
and ``pip install`` lines extracted from cells.
PR 3 adds: a human-readable import report saved as
``<notebook_dir>/import_report.md`` and returned on the
:class:`ImportResult` so the REST endpoint can serve it directly.

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
import shlex
import sys
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import tomli_w

from strata.notebook.models import CellLanguage
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)

_SUPPRESSED_COMMENT = "# strata: trailing ';' from Jupyter preserved as display-suppression"

# A trailing ';' may be followed by an inline comment ("df;  # don't print")
# or simply trailing whitespace. Both forms are common in real notebooks.
_SUPPRESSION_TAIL_RE = re.compile(r";[ \t]*(?:#[^\n]*)?\s*\Z")

_LINE_MAGIC_RE = re.compile(r"^(\s*)%([a-zA-Z_]\w*)([^\n]*)$")
_CELL_MAGIC_RE = re.compile(r"\A[ \t]*%%([a-zA-Z_]\w*)([^\n]*)\n?")
_SHELL_RE = re.compile(r"^(\s*)!(.*)$")
# Assignment-form shell escape: ``files = !ls /data``. IPython supports
# this and binds the lhs to a list of stdout lines. Without explicit
# handling the line passes through and breaks Python's parser.
_SHELL_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_]\w*)(\s*=\s*)!(.+)$")
_PIP_INSTALL_RE = re.compile(
    r"^\s*(?:pip|pip3|python\s+-m\s+pip|uv\s+pip)\s+install\s+(.+)$",
)


# ---------------------------------------------------------------------------
# Result types


@dataclass
class _CellConversion:
    """Per-cell conversion output. Aggregated into :class:`ImportResult`."""

    source: str
    suppressed: bool = False
    deps: list[str] = field(default_factory=list)
    translated_magics: list[str] = field(default_factory=list)
    dropped_magics: list[str] = field(default_factory=list)
    dropped_shells: list[str] = field(default_factory=list)


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
    suppressed_outputs: int = 0
    skipped_cells: list[str] = field(default_factory=list)
    translated_magics: list[str] = field(default_factory=list)
    dropped_magics: list[str] = field(default_factory=list)
    dropped_shells: list[str] = field(default_factory=list)
    captured_deps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Populated by ``import_notebook`` after the conversion finishes.
    # Path to the rendered report file, and the same content in memory
    # so the REST endpoint can return it without re-reading from disk.
    report_path: Path | None = None
    report_text: str = ""


# ---------------------------------------------------------------------------
# Public API


def import_notebook(
    ipynb_path: Path | str,
    out_dir: Path | str | None = None,
    *,
    owner: str | None = None,
) -> ImportResult:
    """Convert a Jupyter ``.ipynb`` file into a Strata notebook directory.

    Args:
        ipynb_path: Path to the source ``.ipynb`` file.
        out_dir: Target notebook directory. If ``None``, a sibling
            directory named after the ``.ipynb`` stem is created next
            to the source file.
        owner: Caller identity to stamp into ``notebook.toml``. The CLI
            doesn't pass this (single-user); the REST endpoint does so
            multi-user / per-user-scoped deployments don't lose owner
            attribution on imported notebooks.

    Returns:
        An :class:`ImportResult` describing what got converted.

    Raises:
        FileNotFoundError: if ``ipynb_path`` doesn't exist.
        ValueError: if the file isn't a valid nbformat object (top-level
            JSON must be a dict; anything else — a list, scalar, null —
            isn't a notebook).
    """
    ipynb_path = Path(ipynb_path)
    if not ipynb_path.is_file():
        raise FileNotFoundError(f"No such file: {ipynb_path}")

    with ipynb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # Validate the nbformat structure up-front so we don't leave a
    # half-materialized notebook directory on disk when the source is
    # malformed (and so AttributeError from a mid-loop ``.get("cell_type")``
    # surfaces as a 400 at the REST layer, not a 500).
    _validate_nbformat_structure(nb)

    if out_dir is not None:
        out_dir = Path(out_dir)
        parent = out_dir.parent
        name = out_dir.name
    else:
        parent = ipynb_path.parent
        name = ipynb_path.stem

    notebook_dir = create_notebook(parent, name, initialize_environment=False, owner=owner)
    result = ImportResult(notebook_dir=notebook_dir)

    sibling_deps = _capture_sibling_deps(ipynb_path.parent)
    local_modules = _local_module_names(ipynb_path.parent)

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
            # Scan the *converted* source — magics have been stripped,
            # so what remains is valid Python the harness will execute.
            scanned_imports |= _scan_imports(conv.source)
            prev_cell_id = cell_id
        elif cell_type is None:
            result.warnings.append("cell missing 'cell_type' was skipped")
        else:
            result.skipped_cells.append(str(cell_type))

    inferred_deps = _imports_to_deps(scanned_imports, local_modules)

    # Merge captured deps into the new notebook's pyproject.toml.
    # Order matters for the dedup: explicit sources (siblings, %pip
    # install) come first so their version pins shadow bare scan-
    # derived names. PEP 503-normalized package-name dedup catches
    # ``scikit_learn`` vs ``scikit-learn`` collisions.
    all_deps = _dedupe_by_package([*sibling_deps, *cell_deps, *inferred_deps])
    # Filter pip-only forms (editable installs, bare URLs, paths) that
    # pyproject.toml dependencies can't represent — those would either
    # be rejected by uv at sync time or, worse, slip through and corrupt
    # the TOML (a "; python_version < '3.10'" marker contains literal
    # characters that need proper escaping).
    valid_deps = [d for d in all_deps if _is_valid_pep508_dep(d)]
    rejected_deps = [d for d in all_deps if not _is_valid_pep508_dep(d)]
    if valid_deps:
        _merge_pyproject_deps(notebook_dir, valid_deps)
    result.captured_deps = valid_deps
    if rejected_deps:
        sample = ", ".join(repr(d) for d in rejected_deps[:3])
        more = "" if len(rejected_deps) <= 3 else f" (+{len(rejected_deps) - 3} more)"
        result.warnings.append(
            f"{len(rejected_deps)} pip-only dep spec(s) skipped — pyproject.toml "
            f"requires PEP 508 specifiers: {sample}{more}"
        )

    # Write the human-readable report next to notebook.toml. Same
    # content is returned on the result so callers (REST, CLI) can
    # serve it without re-reading.
    report_text = format_import_report(result, ipynb_path)
    report_path = notebook_dir / "import_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    result.report_path = report_path
    result.report_text = report_text

    return result


# ---------------------------------------------------------------------------
# Structural validation


def _validate_nbformat_structure(nb: object) -> None:
    """Reject obvious nbformat violations before we materialize anything.

    Catches:
      - Top-level value isn't a JSON object (a list, a scalar, null).
      - ``cells`` is present but isn't a list.
      - Individual entries inside ``cells`` aren't JSON objects.

    Anything more nuanced (missing nbformat version, unknown cell_type)
    we accept and convert as best we can — those don't crash the
    converter, they just produce no-op cells or warnings on the result.
    The point of this check is just to fail fast on shapes that would
    raise AttributeError mid-loop.
    """
    if not isinstance(nb, dict):
        raise ValueError(
            f"Invalid .ipynb: expected JSON object at top level, got {type(nb).__name__}"
        )
    # ``cast`` only — ty narrows the isinstance result to dict[Unknown, Unknown]
    # whose ``.get`` signature resolves to ``(key: Never) -> ...``. The runtime
    # dict is the standard JSON parse so str-keyed access is safe.
    nb_dict = cast(dict[str, Any], nb)
    cells = nb_dict.get("cells")
    if cells is not None and not isinstance(cells, list):
        raise ValueError(f"Invalid .ipynb: 'cells' must be a list, got {type(cells).__name__}")
    for idx, cell in enumerate(cells or []):
        if not isinstance(cell, dict):
            raise ValueError(
                f"Invalid .ipynb: cells[{idx}] must be a JSON object, got {type(cell).__name__}"
            )


# ---------------------------------------------------------------------------
# Import report


def format_import_report(result: ImportResult, ipynb_path: Path | str) -> str:
    """Build the human-readable conversion report for one import.

    Same content the CLI surfaces and the REST endpoint will return.
    Sections only appear when they have content — a clean notebook
    with no magics produces a short report.
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


# ---------------------------------------------------------------------------
# Source conversion


def _source_to_text(source: Any) -> str:
    """nbformat allows ``source`` as either a string or a list of lines.

    The list form is the canonical on-disk shape; the string form
    appears in hand-edited notebooks and in some exporters. Some
    hand-edited notebooks contain cells whose source has a leading
    space (the JSON looks like ``" Image(...)"``) which would fail
    Python's parser as a module-level indent error. ``dedent`` (in
    a single-line sense) is safe here: we're just normalizing the
    cell envelope, not changing intended indentation inside a
    function body.
    """
    if isinstance(source, list):
        text = "".join(source)
    elif source is None:
        text = ""
    else:
        text = str(source)
    # Strip leading newlines / spaces and trailing whitespace that
    # would confuse module-level parsing. If the first line is
    # genuinely indented (rare; cell shouldn't start that way), the
    # caller still surfaces a syntax error later, but the common
    # case of " Image(...) " is fixed.
    return text.strip() + "\n" if text.strip() else ""


def _new_cell_id(prefix: str) -> str:
    """Cell IDs follow the existing 8-char UUID-prefix convention."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _ensure_final_newline(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def _convert_code_source(source: str) -> _CellConversion:
    """Convert one Jupyter code cell to runnable Python.

    Order of operations:
      1. Cell magic (first line is ``%%name``) — applies to the whole
         body and is dispatched to a single handler.
      2. Otherwise, walk lines: translate line magics (``%name``) and
         shell escapes (``!cmd``) per the table.
      3. Apply ``;``-display-suppression rewriting after magics have
         been translated — a translated magic might have left the
         body's last expression as the new suppression target.
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
    for raw_line in source.splitlines(keepends=True):
        line_no_eol = raw_line.rstrip("\n")
        line_magic = _LINE_MAGIC_RE.match(line_no_eol)
        if line_magic:
            indent, magic_name, magic_args = line_magic.groups()
            replacement = _translate_line_magic(
                magic_name,
                magic_args.lstrip(),
                indent,
                conv,
            )
            out_lines.extend(replacement)
            continue
        shell_assign = _SHELL_ASSIGN_RE.match(line_no_eol)
        if shell_assign:
            indent, target, eq, cmd = shell_assign.groups()
            replacement = _translate_shell_assignment(
                target,
                eq,
                cmd.strip(),
                indent,
                conv,
            )
            out_lines.extend(replacement)
            continue
        shell = _SHELL_RE.match(line_no_eol)
        if shell:
            indent, cmd = shell.groups()
            replacement = _translate_shell(cmd.strip(), indent, conv)
            out_lines.extend(replacement)
            continue
        out_lines.append(raw_line)

    result_source = "".join(out_lines)
    if _ends_with_display_suppression(result_source):
        result_source = _suppress_last_expression(result_source)
        conv.suppressed = True
    conv.source = _ensure_final_newline(result_source)
    return conv


def _ends_with_display_suppression(source: str) -> bool:
    """True if the source ends in Jupyter's ``;`` suppression idiom.

    Both bare (``df;``) and commented (``df;  # quiet``) forms count.
    """
    return _SUPPRESSION_TAIL_RE.search(source) is not None


def _suppress_last_expression(source: str) -> str:
    """Rewrite a ``;``-suppressed cell so Strata won't auto-display its last expr.

    The harness auto-displays the value of a final ``ast.Expr`` node.
    Append ``pass`` so the last node becomes a ``Pass`` instead.
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
        return source

    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return body + "\n"
    return f"{body}\n{_SUPPRESSED_COMMENT}\npass\n"


# ---------------------------------------------------------------------------
# Magic translation


def _translate_line_magic(
    name: str,
    args: str,
    indent: str,
    conv: _CellConversion,
) -> list[str]:
    """Dispatch a ``%name args`` line magic.

    Returns the lines to substitute (may be empty). Mutates ``conv``
    in place to record metadata.
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
    """``%timeit body`` → ``body`` (drop the timing wrapper, keep the work)."""
    conv.translated_magics.append(f"%{name}")
    if args:
        return [f"{indent}{args}\n"]
    return []


def _lm_pip(name: str, args: str, indent: str, conv: _CellConversion) -> list[str]:
    """``%pip install pkg`` captures packages; other subcommands are dropped.

    Only the ``install`` subcommand contributes deps — ``%pip list``,
    ``%pip uninstall``, ``%pip show``, etc. are not useful at import
    time and would just leak noise into the captured-deps list.
    """
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
    """``%env KEY=VAL`` → ``# @env KEY=VAL`` cell annotation."""
    if "=" not in args:
        conv.dropped_magics.append(f"%{name} (no KEY=VALUE)")
        return [f"{indent}# strata: %env requires KEY=VALUE; dropped\n"]
    conv.translated_magics.append(f"%{name}")
    return [f"{indent}# @env {args.strip()}\n"]


def _lm_run(name: str, args: str, indent: str, conv: _CellConversion) -> list[str]:
    """``%run script.py`` → ``exec`` of the script's text (best effort).

    Uses an aliased ``pathlib.Path`` import so the generated code
    works even if the cell hasn't imported ``Path`` itself.
    """
    target = args.strip()
    if not target:
        conv.dropped_magics.append(f"%{name} (no target)")
        return [f"{indent}# strata: %run with no target dropped\n"]
    conv.translated_magics.append(f"%{name} {target}")
    return [
        f"{indent}# strata: %run translated — verify the path resolves at runtime\n",
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
    """``%%bash``/``%%sh`` → wrap the body in ``subprocess.run(..., shell=True)``."""
    wrapped = f"import subprocess as _strata_sp\n_strata_sp.run({body!r}, shell=True, check=True)\n"
    return _CellConversion(source=wrapped, translated_magics=[f"%%{name}"])


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
    # Shell-out cell magics, translated to subprocess.run
    "bash": _cm_bash,
    "sh": _cm_bash,
    "script": _cm_bash,  # ``%%script python`` and friends, best-effort as shell
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
    """``target = !cmd`` — IPython binds ``target`` to stdout lines.

    Auto-running arbitrary shell from an imported notebook is a real
    hazard (untrusted-corpus stress tests are a primary use case), so
    we don't translate to a live subprocess call. Instead we drop the
    command and stub the binding with ``[]`` so downstream Python
    still parses and references to ``target`` resolve. The user can
    swap in a real ``subprocess.run`` if the shell escape matters.

    ``!pip install`` in this form is rare but still captures the
    package — the lhs gets the same empty-list stub.
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

    Drops flag tokens (``-U``, ``--upgrade``, ``-q``, etc.) and flag-args
    pairs that consume a following positional (``-r req.txt``,
    ``--index-url ...``). Keeps version specifiers attached to their
    package name (``foo==1.2``), URL specs (``git+https://…``), and
    extras (``foo[bar]``).
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

    Best-effort: errors collapse to "no deps captured from this source".
    Most Kaggle / GitHub notebooks ship one or the other.
    """
    deps: list[str] = []
    req = parent / "requirements.txt"
    if req.is_file():
        try:
            for raw in req.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                deps.append(line)
        except OSError:
            pass

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


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


# Top-level import names whose PyPI package name differs. Anything not
# in this dict is assumed to use ``import_name == pip_name``, right
# ~95% of the time in practice. Extend by adding a row.
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
    "lxml": "lxml",  # Same name but kept here for documentation
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
    # Deprecated aliases that users still write
    "gym": "gymnasium",  # gym is unmaintained; gymnasium is the maintained fork
}


def _scan_imports(source: str) -> set[str]:
    """Collect top-level module names imported by a cell.

    Walks the AST for ``Import`` and ``ImportFrom`` nodes, takes the
    first dotted-path component, filters out stdlib names. Returns
    an empty set on syntax errors — those surface separately when
    the harness tries to execute the cell.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            # node.level > 0 is a relative import (``from . import x``),
            # which can't be a third-party dependency.
            if node.module and node.level == 0:
                names.add(node.module.split(".", 1)[0])
    return names - sys.stdlib_module_names


def _local_module_names(parent_dir: Path) -> set[str]:
    """Names that would resolve to local files / packages, not PyPI.

    Without this, a notebook that does ``import my_helpers`` next to
    a ``my_helpers.py`` file would end up with a fabricated PyPI dep
    that fails confusingly on ``uv sync``.
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
    """Map a set of import names to pip package specifiers.

    Skips anything that names a local module. Returns sorted output
    for stable, hashable results across runs.
    """
    deps: list[str] = []
    for name in sorted(imports):
        if name in local_modules:
            continue
        deps.append(_IMPORT_TO_PIP.get(name, name))
    return deps


def _normalize_pep503(name: str) -> str:
    """Canonical comparison key for PEP 508 specifiers.

    Strips version markers / extras / markers, lowercases, replaces
    ``_``/``.`` with ``-`` (PEP 503 normalization). ``scikit_learn``,
    ``scikit-learn``, and ``Scikit-Learn`` all map to the same key,
    so version-pinned siblings shadow bare scan-derived names.
    """
    head = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", name.strip())
    if not head:
        return ""
    return re.sub(r"[._-]+", "-", head.group(1)).lower()


def _dedupe_by_package(specs: list[str]) -> list[str]:
    """Dedupe in order, keying by the PEP 503-normalized package name.

    Ensures ``pandas==2.0.1`` (explicit, earlier) shadows a bare
    ``pandas`` (inferred from imports, later) — the version pin wins.
    """
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
    """Filter out specifiers that pyproject.toml ``dependencies`` won't accept.

    pyproject.toml's ``project.dependencies`` requires PEP 508
    specifiers — ``name``, ``name==1.2``, ``name[extras]``, ``name @ url``,
    optionally with a marker. Pip-only forms (editable installs,
    bare URLs, local paths) are rejected here so they don't get
    serialized into invalid TOML or get rejected later by uv.
    """
    spec = spec.strip()
    if not spec or spec.startswith("-"):
        return False
    if spec.startswith(
        ("git+", "hg+", "svn+", "bzr+", "file:", "http://", "https://", "/", "./", "../")
    ):
        return False
    # PEP 508 names start with a letter/digit. Anything else (bare URL
    # fragments, `.`-style paths sneaking past the prefix list, etc.) is
    # rejected.
    return re.match(r"^[A-Za-z0-9]", spec) is not None


def _merge_pyproject_deps(notebook_dir: Path, new_deps: list[str]) -> list[str]:
    """Add captured deps to the new notebook's ``pyproject.toml``.

    Round-trips through ``tomllib`` + ``tomli_w`` so any string with
    embedded quotes / backslashes / etc. (e.g. environment markers like
    ``importlib-metadata; python_version < "3.10"``) is properly escaped
    by the serializer — manual string interpolation would emit invalid
    TOML.

    The notebook venv hasn't been created yet, so first ``uv sync``
    will resolve the deps. We deliberately don't run ``uv add`` here —
    that's slow, networked, and partial-failure-prone.

    Returns the deps actually added (skipping ones already present).
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
