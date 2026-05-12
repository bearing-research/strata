"""Import .ipynb files into Strata notebook directories.

Public entry: :func:`import_notebook` takes a path to a Jupyter
notebook file and produces a runnable Strata notebook directory.

PR 1 scope: parse + convert markdown / code cells, ``;``-suppression.
PR 2 adds: line-magic and cell-magic translation, ``!shell`` handling,
dependency capture from sibling ``requirements.txt`` / ``pyproject.toml``
and ``pip install`` lines extracted from cells.

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
import tomllib
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

_LINE_MAGIC_RE = re.compile(r"^(\s*)%([a-zA-Z_]\w*)([^\n]*)$")
_CELL_MAGIC_RE = re.compile(r"\A[ \t]*%%([a-zA-Z_]\w*)([^\n]*)\n?")
_SHELL_RE = re.compile(r"^(\s*)!(.*)$")
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


# ---------------------------------------------------------------------------
# Public API


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

    sibling_deps = _capture_sibling_deps(ipynb_path.parent)

    prev_cell_id: str | None = None
    cell_deps: list[str] = []
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
            conv = _convert_code_source(source)
            add_cell_to_notebook(
                notebook_dir,
                cell_id,
                after_cell_id=prev_cell_id,
                language="python",
            )
            write_cell(notebook_dir, cell_id, conv.source)
            result.code_cells += 1
            if conv.suppressed:
                result.suppressed_outputs += 1
            result.translated_magics.extend(conv.translated_magics)
            result.dropped_magics.extend(conv.dropped_magics)
            result.dropped_shells.extend(conv.dropped_shells)
            cell_deps.extend(conv.deps)
            prev_cell_id = cell_id
        elif cell_type is None:
            result.warnings.append("cell missing 'cell_type' was skipped")
        else:
            result.skipped_cells.append(str(cell_type))

    # Merge captured deps into the new notebook's pyproject.toml.
    # uv sync on first run will resolve them; we deliberately don't
    # run `uv add` here — that's slow, networked, and can fail
    # partially, and our job is to set up source for the user.
    all_deps = _dedupe_preserve_order([*sibling_deps, *cell_deps])
    if all_deps:
        _merge_pyproject_deps(notebook_dir, all_deps)
        result.captured_deps = all_deps

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
    return handler(args, body)


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
    """``%run script.py`` → ``exec`` of the script's text (best effort)."""
    target = args.strip()
    if not target:
        conv.dropped_magics.append(f"%{name} (no target)")
        return [f"{indent}# strata: %run with no target dropped\n"]
    conv.translated_magics.append(f"%{name} {target}")
    return [
        f"{indent}# strata: %run translated — verify the path resolves at runtime\n",
        f"{indent}exec(Path({target!r}).read_text())\n",
    ]


_LINE_MAGIC_TABLE = {
    "matplotlib": _lm_drop,
    "load_ext": _lm_drop,
    "autoreload": _lm_drop,
    "reload_ext": _lm_drop,
    "capture": _lm_drop,
    "xmode": _lm_drop,
    "pdb": _lm_drop,
    "debug": _lm_drop,
    "config": _lm_drop,
    "timeit": _lm_strip,
    "time": _lm_strip,
    "pip": _lm_pip,
    "env": _lm_env,
    "run": _lm_run,
}


# --- cell-magic handlers ---


def _cm_strip(args: str, body: str) -> _CellConversion:
    """``%%timeit body`` → recurse on the body as plain code."""
    inner = _convert_code_source(body)
    inner.translated_magics.insert(0, "%%timeit/time")
    return inner


def _cm_drop(args: str, body: str) -> _CellConversion:
    return _CellConversion(
        source="# strata: cell magic dropped (body not translatable)\n",
        dropped_magics=["%%javascript|html|latex|svg|markdown"],
    )


def _cm_bash(args: str, body: str) -> _CellConversion:
    """``%%bash``/``%%sh`` → wrap the body in ``subprocess.run(..., shell=True)``."""
    wrapped = f"import subprocess as _strata_sp\n_strata_sp.run({body!r}, shell=True, check=True)\n"
    return _CellConversion(source=wrapped, translated_magics=["%%bash/sh"])


def _cm_writefile(args: str, body: str) -> _CellConversion:
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
    "timeit": _cm_strip,
    "time": _cm_strip,
    "capture": _cm_strip,
    "bash": _cm_bash,
    "sh": _cm_bash,
    "writefile": _cm_writefile,
    "javascript": _cm_drop,
    "html": _cm_drop,
    "latex": _cm_drop,
    "svg": _cm_drop,
    "markdown": _cm_drop,
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


def _merge_pyproject_deps(notebook_dir: Path, new_deps: list[str]) -> None:
    """Append captured deps to the new notebook's ``pyproject.toml``.

    The notebook venv hasn't been created yet (we passed
    ``initialize_environment=False`` to ``create_notebook``), so first
    ``uv sync`` will pick the deps up naturally. We don't call
    ``uv add`` ourselves — that's slow, networked, and partial-
    failure-prone, and the user is going to run sync anyway when
    they open or run the imported notebook.

    Edits the file textually to avoid round-tripping through
    ``tomli_w`` (which would normalize formatting on every import).
    """
    pyproject = notebook_dir / "pyproject.toml"
    if not pyproject.is_file():
        return
    text = pyproject.read_text(encoding="utf-8")
    # Find the dependencies = [ ... ] block. The notebook template
    # writes it as a single multi-line block.
    match = re.search(r"(dependencies\s*=\s*\[)([^\]]*)(\])", text, flags=re.DOTALL)
    if not match:
        return
    existing_block = match.group(2)
    existing = set()
    for line in existing_block.splitlines():
        m = re.match(r'\s*"([^"]+)"', line)
        if m:
            existing.add(m.group(1))
    additions = [d for d in new_deps if d not in existing]
    if not additions:
        return
    new_lines = "\n".join(f'    "{d}",' for d in additions)
    # Preserve trailing newline before the closing bracket.
    suffix = "\n" if existing_block.rstrip("\n").endswith(",") else ",\n"
    rebuilt = (
        match.group(1) + existing_block.rstrip("\n") + suffix + new_lines + "\n" + match.group(3)
    )
    pyproject.write_text(text[: match.start()] + rebuilt + text[match.end() :], encoding="utf-8")
