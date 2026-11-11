"""What a cell's run found that its cached result depends on without saying, as diagnostics.

A run's read audit (``read_audit.report``) and mutation warnings, turned into the
messages the cell's header pill and the run report show, each naming its one fix.
Never a key, a status or a cache decision: the run stands as it ran.
"""

from __future__ import annotations

import keyword
import os
import re
import shlex
from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.parse import quote

from strata.notebook.immutability import MutationWarning
from strata.notebook.models import AnnotationDiagnostic, DiagnosticSeverity
from strata.notebook.read_audit import (
    AUDIT_FAILED,
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
    taken: Iterable[str] = (),
) -> list[AnnotationDiagnostic]:
    """One diagnostic per kind of finding, then one per distinct mutation warning.

    A suggested mount is named after its folder, never with a name in *taken* (the
    cell's mounts and variables).
    """
    names = _Names(taken)
    diagnostics = [_finding(finding, notebook_dir, names) for finding in read_audit]
    # A loop's iterations and a fan-out's instances each warn of the same mutation.
    messages = dict.fromkeys(
        f"{warning['message']}. {warning['suggestion']}"
        if warning.get("suggestion")
        else warning["message"]
        for warning in mutation_warnings
    )
    diagnostics.extend(
        AnnotationDiagnostic(severity=DiagnosticSeverity.WARN, code=MUTATED_INPUT, message=m)
        for m in messages
    )
    return diagnostics


def changed_reads_hint(paths: Sequence[str], rerun: str) -> str:
    """What a ready cell says when files its result read undeclared have changed since.

    Its key did not move, so a plain run would hit the same cache: *rerun* names the
    run that reads them again.
    """
    one = len(paths) == 1
    noun = _code(paths[0]) if one else f"{len(paths)} files ({_shown(list(paths), len(paths))})"
    them = "it" if one else "them"
    return (
        f"May be out of date: {noun} changed after this result read {them}, and the cache "
        f"does not see {them}. {rerun} reads {them} again; a mount over {them} keys {them}."
    )


def _finding(finding: ReadFinding, notebook_dir: Path, names: _Names) -> AnnotationDiagnostic:
    code = finding["code"]
    items = finding["items"]
    severity = DiagnosticSeverity.WARN
    if code == UNDECLARED_READ:
        message = _read(items, finding["total"], notebook_dir, names)
    elif code == UNDECLARED_LISTING:
        message = _listing(items, finding["total"], notebook_dir, names)
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
    if finding["partial"] and code != AUDIT_FAILED:
        # At its path cap, or on a call the hook could not read.
        message += " The audit did not record all of this run's reads, so there may be more."
    return AnnotationDiagnostic(severity=severity, code=code, message=message)


def _read(items: list[str], total: int, notebook_dir: Path, names: _Names) -> str:
    mounts, loose = _mounts([os.path.dirname(path) for path in items], notebook_dir, names)
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
    fix = _mount_fix(mounts, loose, "read them", "Move the ones directly in {} into a folder")
    return f"{what}, so its cached result will not change when they do. {fix}"


def _listing(items: list[str], total: int, notebook_dir: Path, names: _Names) -> str:
    mounts, loose = _mounts(items, notebook_dir, names)
    if total == 1:
        what = f"This cell listed {_code(items[0])}, which no mount covers"
    else:
        what = f"This cell listed {total} directories no mount covers ({_shown(items, total)})"
    if total == 1 and mounts:
        ((name, _, written),) = mounts
        fix = f"Add {_annotation(name, written)} and list {_code(name)}."
    else:
        fix = _mount_fix(mounts, loose, "list them", "In place of {}, list a folder of its own")
    return f"{what}, so its cached result will not change when a file is added or removed. {fix}"


def _import(items: list[str], total: int, notebook_dir: Path) -> str:
    # Its import root, where its package's pyproject.toml goes: above its packages.
    roots = sorted({_import_root(os.path.dirname(path)) for path in items})
    noun = _code(items[0]) if total == 1 else f"{total} modules ({_shown(items, total)})"
    fixes = []
    edits = [
        _code(f"uv add --editable {shlex.quote(_written(root, notebook_dir))}")
        for root in roots
        if _mountable(root, notebook_dir)
    ]
    if edits:
        more = [f"{len(edits) - SHOWN} more like them"] if len(edits) > SHOWN else []
        fixes.append(
            f"Make {'it' if len(edits) == 1 else 'each'} a package and "
            f"{_join(edits[:SHOWN] + more)}, which keys its source."
        )
    loose = [_code(root) for root in roots if not _mountable(root, notebook_dir)]
    if loose:
        fixes.append(
            f"Move the ones directly in {_join(loose)} into a folder, make that a package "
            f"and {_code('uv add --editable')} it."
        )
    return (
        f"This cell imported {noun} from outside the notebook's environment, so its cached "
        f"result will not change when that source does. {' '.join(fixes)}"
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


def _mounts(
    dirs: list[str], notebook_dir: Path, names: _Names
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """A mount ``(name, path, as written)`` per group of *dirs*, over their nearest common one,
    and the directories no mount should cover (see :func:`_mountable`).

    Written relative to the notebook when it is in or beside it, so the fix holds
    wherever the project is checked out.
    """
    mounts, loose = [], []
    for root in sorted(_roots(sorted(set(dirs)), notebook_dir)):
        if _mountable(root, notebook_dir):
            mounts.append((names(root), root, _written(root, notebook_dir)))
        else:
            loose.append(root)
    return mounts, loose


def _written(root: str, notebook_dir: Path) -> str:
    """*root* relative to the notebook when it is in or beside it, else as it is."""
    rel = os.path.relpath(os.path.realpath(root), os.path.realpath(notebook_dir))
    if rel.startswith(os.path.join(os.pardir, os.pardir)):
        return root
    # ``./`` so ``uv add`` reads a path, not a package name. Never ``.``: not mountable.
    return rel if rel.startswith(os.pardir) else f".{os.sep}{rel}"


def _roots(dirs: list[str], notebook_dir: Path) -> list[str]:
    common = os.path.commonpath(dirs)
    if _mountable(common, notebook_dir):
        return [common]
    groups: dict[str, list[str]] = {}
    for path in dirs:
        rel = os.path.relpath(path, common)
        head = "" if rel == "." else rel.split(os.sep)[0]
        groups.setdefault(head, []).append(path)
    roots = []
    for head, group in groups.items():
        # A file right in an unmountable directory (the home folder): that one, still.
        roots.extend([common] if head == "" else _roots(group, notebook_dir))
    return roots


def _mountable(path: str, notebook_dir: Path) -> bool:
    """Not the root, nor the home folder, the notebook's or a folder above either: a mount
    reads everything in it, and one over the notebook keys the cell on every other cell.
    """
    real = os.path.realpath(path)
    over = (os.path.realpath(os.path.expanduser("~")), os.path.realpath(notebook_dir))
    return real != os.sep and not any(
        below == real or below.startswith(real.rstrip(os.sep) + os.sep) for below in over
    )


def _import_root(directory: str) -> str:
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        directory = os.path.dirname(directory)
    return directory


class _Names:
    """A mount name per folder, the same for it in every diagnostic of a run."""

    def __init__(self, taken: Iterable[str]) -> None:
        self._taken = set(taken)
        self._named: dict[str, str] = {}

    def __call__(self, root: str) -> str:
        if root not in self._named:
            base = re.sub(r"\W+", "_", os.path.basename(root.rstrip(os.sep))).strip("_").lower()
            if not base or base[0].isdigit() or keyword.iskeyword(base):
                base = f"data_{base}".rstrip("_")
            name, n = base, 2
            while name in self._taken:
                name, n = f"{base}_{n}", n + 1
            self._taken.add(name)
            self._named[root] = name
        return self._named[root]


def _annotation(name: str, root: str) -> str:
    if any(c.isspace() for c in root):
        # The annotation splits on whitespace; a ``file:`` URI is percent-decoded.
        root = f"file://{quote(root)}" if os.path.isabs(root) else f"file:{quote(root)}"
    return _code(f"# @mount {name} {root} ro")


def _mount_fix(
    mounts: list[tuple[str, str, str]], loose: list[str], verb: str, instead: str
) -> str:
    """Add the *mounts*; and for the *loose* folders, *instead* (``{}`` for where) them."""
    fixes = []
    if mounts:
        adds = [_annotation(name, written) for name, _, written in mounts[:SHOWN]]
        if len(mounts) > SHOWN:
            adds.append(f"{len(mounts) - SHOWN} more like them")
        through = "that mount" if len(mounts) == 1 else "those mounts"
        fixes.append(f"Add {_join(adds)}, and {verb} through {through}.")
    if loose:
        fixes.append(f"{instead.format(_join([_code(path) for path in loose]))}, and mount that.")
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
