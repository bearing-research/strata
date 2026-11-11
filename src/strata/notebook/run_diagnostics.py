"""What a cell's run found that its cached result depends on without saying, as diagnostics.

A run's read audit (``read_audit.report``) and mutation warnings, turned into the
messages the cell's header pill and the run report show, each naming its one fix.
Never a key, a status or a cache decision: the run stands as it ran.
"""

from __future__ import annotations

import keyword
import os
import re
from collections.abc import Sequence
from pathlib import Path

from strata.notebook.immutability import MutationWarning
from strata.notebook.models import AnnotationDiagnostic, DiagnosticSeverity
from strata.notebook.read_audit import (
    AUDIT_FAILED,
    PATH_CAP,
    SUBPROCESS_RUN,
    UNDECLARED_IMPORT,
    UNDECLARED_LISTING,
    UNDECLARED_READ,
    UNLOCKED_PACKAGE,
    WORK_DIR_READ,
    ReadFinding,
)

MUTATED_INPUT = "mutated_input"
# Per diagnostic, the items its message names.
SHOWN = 5


def run_diagnostics(
    read_audit: Sequence[ReadFinding],
    mutation_warnings: Sequence[MutationWarning],
    *,
    notebook_dir: Path,
) -> list[AnnotationDiagnostic]:
    """One diagnostic per kind of finding, then one per mutation warning."""
    diagnostics = [_finding(finding, notebook_dir) for finding in read_audit]
    for warning in mutation_warnings:
        suggestion = warning.get("suggestion")
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.WARN,
                code=MUTATED_INPUT,
                message=f"{warning['message']}. {suggestion}" if suggestion else warning["message"],
            )
        )
    return diagnostics


def _finding(finding: ReadFinding, notebook_dir: Path) -> AnnotationDiagnostic:
    code = finding["code"]
    items = finding["items"]
    severity = DiagnosticSeverity.WARN
    if code == UNDECLARED_READ:
        message = _read(items, finding["total"], notebook_dir)
    elif code == UNDECLARED_LISTING:
        message = _listing(items, finding["total"], notebook_dir)
    elif code == UNDECLARED_IMPORT:
        message = _import(items, finding["total"], notebook_dir)
    elif code == UNLOCKED_PACKAGE:
        message = _unlocked(items, finding["total"])
    elif code == WORK_DIR_READ:
        message = _work_dir(items, finding["total"], notebook_dir)
    elif code == SUBPROCESS_RUN:
        severity = DiagnosticSeverity.INFO
        message = _subprocess(items, finding["total"])
    elif code == AUDIT_FAILED:
        severity = DiagnosticSeverity.INFO
        message = f"The read audit failed ({_list(items)}), so this run's reads were not checked."
    else:
        message = f"This run read {_list(items)}, which the cell does not declare."
    if finding["partial"]:
        message += f" The audit stopped recording at {PATH_CAP:,} paths, so there may be more."
    return AnnotationDiagnostic(severity=severity, code=code, message=message)


def _read(items: list[str], total: int, notebook_dir: Path) -> str:
    mounts, loose = _mounts([os.path.dirname(path) for path in items], notebook_dir)
    if total == 1:
        what = f"This cell read {_code(items[0])}, which no mount covers"
        if mounts:
            ((name, root, written),) = mounts
            relative = os.path.relpath(items[0], root)
            fix = f"Add {_annotation(name, written)} and read {_code(f'{name} / "{relative}"')}."
        else:
            fix = "Move it into a folder of its own and mount that."
        return f"{what}, so its cached result will not change when the file does. {fix}"
    what = f"This cell read {total} files no mount covers ({_shown(items, total)})"
    fix = _mount_fix(mounts, loose, "read them")
    return f"{what}, so its cached result will not change when they do. {fix}"


def _listing(items: list[str], total: int, notebook_dir: Path) -> str:
    mounts, loose = _mounts(items, notebook_dir)
    if total == 1 and mounts:
        ((name, _, written),) = mounts
        what = f"This cell listed {_code(items[0])}, which no mount covers"
        fix = f"Add {_annotation(name, written)} and list {_code(name)}."
    else:
        noun = _code(items[0]) if total == 1 else f"{total} directories"
        what = f"This cell listed {noun} no mount covers ({_shown(items, total)})"
        fix = _mount_fix(mounts, loose, "list them")
    return f"{what}, so its cached result will not change when a file is added or removed. {fix}"


def _import(items: list[str], total: int, notebook_dir: Path) -> str:
    roots = sorted(_roots(sorted({os.path.dirname(path) for path in items})))
    noun = _code(items[0]) if total == 1 else f"{total} modules ({_shown(items, total)})"
    edits = _join([_code(f"uv add --editable {_written(root, notebook_dir)}") for root in roots])
    return (
        f"This cell imported {noun} from outside the notebook's environment, so its cached "
        f"result will not change when that source does. Make it a package and {edits}, "
        "which keys its source."
    )


def _unlocked(items: list[str], total: int) -> str:
    noun = (
        f"{_code(items[0])}, which uv.lock does not hold"
        if total == 1
        else f"{total} packages uv.lock does not hold ({_shown(items, total)})"
    )
    return (
        f"This cell imported {noun}, so its cached result will not change when "
        f"{'that package does' if total == 1 else 'they do'}. "
        f"{_code('uv add ' + ' '.join(items[:SHOWN]))}{' and the rest' if total > SHOWN else ''} "
        "(or the Environment panel) keys "
        f"{'it' if total == 1 else 'them'}."
    )


def _work_dir(items: list[str], total: int, notebook_dir: Path) -> str:
    shown = [_under(path, notebook_dir) for path in items]
    noun = _code(shown[0]) if total == 1 else f"{total} files ({_shown(shown, total)})"
    return (
        f"This cell read {noun}, which an earlier run left in work/, so the DAG "
        "has no edge to the cell that wrote it and its cached result will not change when "
        "it does. Return the data from the cell that writes it, and use that variable here."
    )


def _subprocess(items: list[str], total: int) -> str:
    noun = _code(items[0]) if total == 1 else f"{total} programs ({_shown(items, total)})"
    return (
        f"This cell started {noun}, whose reads the audit does not see. Declare what "
        f"{'it reads' if total == 1 else 'they read'} as mounts, or its cached result "
        "will not change when that does."
    )


def _mounts(dirs: list[str], notebook_dir: Path) -> tuple[list[tuple[str, str, str]], list[str]]:
    """A mount ``(name, path, as written)`` per group of *dirs*, over their nearest common one,
    and the directories no mount should cover (the home folder, the root).

    Written relative to the notebook when it is in or beside it, so the fix holds
    wherever the project is checked out.
    """
    taken: set[str] = set()
    mounts, loose = [], []
    for root in sorted(_roots(sorted(set(dirs)))):
        if not _mountable(root):
            loose.append(root)
            continue
        name = _name(root, taken)
        taken.add(name)
        mounts.append((name, root, _written(root, notebook_dir)))
    return mounts, loose


def _written(root: str, notebook_dir: Path) -> str:
    """*root* relative to the notebook when it is in or beside it, else as it is."""
    rel = os.path.relpath(os.path.realpath(root), os.path.realpath(notebook_dir))
    if rel.startswith(os.path.join(os.pardir, os.pardir)):
        return root
    # ``./`` so ``uv add`` reads a path, not a package name.
    return rel if rel.startswith(os.pardir) or rel == os.curdir else f".{os.sep}{rel}"


def _roots(dirs: list[str]) -> list[str]:
    common = os.path.commonpath(dirs)
    if _mountable(common):
        return [common]
    groups: dict[str, list[str]] = {}
    for path in dirs:
        rel = os.path.relpath(path, common)
        head = "" if rel == "." else rel.split(os.sep)[0]
        groups.setdefault(head, []).append(path)
    roots = []
    for head, group in groups.items():
        # A file right in an unmountable directory (the home folder): that one, still.
        roots.extend([common] if head == "" else _roots(group))
    return roots


def _mountable(path: str) -> bool:
    """Not the root, the home folder or a folder above it: a mount reads everything in it."""
    home = os.path.expanduser("~")
    return path != os.sep and not (home == path or home.startswith(path.rstrip(os.sep) + os.sep))


def _name(root: str, taken: set[str]) -> str:
    base = re.sub(r"\W+", "_", os.path.basename(root.rstrip(os.sep))).strip("_").lower()
    if not base or base[0].isdigit() or keyword.iskeyword(base):
        base = f"data_{base}".rstrip("_")
    name, n = base, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    return name


def _annotation(name: str, root: str) -> str:
    return _code(f"# @mount {name} {root} ro")


def _mount_fix(mounts: list[tuple[str, str, str]], loose: list[str], verb: str) -> str:
    fixes = []
    if mounts:
        adds = [_annotation(name, written) for name, _, written in mounts[:SHOWN]]
        if len(mounts) > SHOWN:
            adds.append(f"{len(mounts) - SHOWN} more like them")
        through = "that mount" if len(mounts) == 1 else "those mounts"
        fixes.append(f"Add {_join(adds)}, and {verb} through {through}.")
    if loose:
        where = _join([_code(path) for path in loose])
        fixes.append(f"Move the ones directly in {where} into a folder, and mount that.")
    return " ".join(fixes)


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _under(path: str, notebook_dir: Path) -> str:
    rel = os.path.relpath(path, notebook_dir)
    return path if rel.startswith("..") else rel


def _shown(items: list[str], total: int) -> str:
    shown = ", ".join(_code(item) for item in items[:SHOWN])
    return f"{shown} and {total - SHOWN} more" if total > SHOWN else shown


def _list(items: list[str]) -> str:
    return ", ".join(_code(item) for item in items)


def _code(text: str) -> str:
    return f"`{text}`"
