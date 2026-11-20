"""The agent hook: what the project's scratchpad holds, checked now (design 4.7).

Claude Code runs it at session start (``session-start``) and on each prompt
(``prompt``, only the cells the prompt names), with the hook's JSON on stdin. It reads
the digest the daemon wrote (``.strata-scratch/.strata/digest.json``) and prints a short
block for the agent. It never starts the daemon nor opens a session, and runs by path on
the standard library alone (``python -I agent_hook.py``): ``strata.notebook`` takes half
a second to import, and this runs before every prompt.

The digest is last-known state, so each cell's recorded reads are observed again first:
``cached`` means checked just now. A path that changed makes its cell stale, and every
cell downstream of it; the environment's watched paths, changed, make every cell stale.
What the budget leaves unchecked is said ``as of`` the digest's time, never ``cached``.
Keep ``fingerprint``, ``racy``, ``observe``, ``watch_state`` and ``find_anchor`` as
``strata.notebook.code_reads`` and ``strata.scratch`` have them: a test pins each.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import time
from typing import Any

ABSENT = "absent"
LARGE_BYTES = 64 * 1024 * 1024
RACY_NS = 2_000_000_000
SCRATCH_DIR = ".strata-scratch"
DIGEST = os.path.join(SCRATCH_DIR, ".strata", "digest.json")
# What one hook run may look at: past either, the rest is said "as of".
MAX_PATHS = 1000
MAX_SECONDS = 0.08
MAX_LINES = 15


def fingerprint(path: str, taken_ns: int) -> list[int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, taken_ns]


def racy(print_: list[int]) -> bool:
    _dev, _ino, _size, mtime, ctime, taken = print_
    return ctime == 0 or max(mtime, ctime) >= taken - RACY_NS


def observe(path: str, *, listing: bool = False) -> tuple[str, list[int] | None]:
    taken = time.time_ns()
    try:
        st = os.stat(path)
    except OSError:
        return ABSENT, None
    print_ = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, taken]
    try:
        if stat.S_ISDIR(st.st_mode):
            if not listing:
                return "dir", print_
            names = sorted(n for n in os.listdir(path) if not n.startswith("."))
            digest = hashlib.sha256("\0".join(names).encode(errors="surrogateescape"))
            return f"list:{digest.hexdigest()}", print_
        if st.st_size > LARGE_BYTES:
            return "stat:" + ":".join(map(str, print_[:5])), print_
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            while chunk := file.read(1 << 20):
                digest.update(chunk)
    except OSError:
        return ABSENT, None
    return f"sha256:{digest.hexdigest()}", print_


def watch_state(path: str, mode: bool | None) -> str:
    if mode is not None:
        return observe(path, listing=mode)[0]
    try:
        st = os.stat(path)
    except OSError:
        return ABSENT
    return f"stat:{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


def _home() -> str | None:
    home = os.path.expanduser("~")
    return os.path.realpath(home) if home != "~" else None


def find_anchor(start: str) -> str:
    """``strata.scratch.find_anchor``: the project *start* is in."""
    start = os.path.realpath(start)
    chain = [start]
    while os.path.dirname(chain[-1]) != chain[-1]:
        chain.append(os.path.dirname(chain[-1]))
    inside = [
        d
        for d in chain
        if os.path.basename(d) == SCRATCH_DIR or os.path.isfile(os.path.join(d, "notebook.toml"))
    ]
    if inside:
        start = os.path.dirname(inside[-1])
        chain = [start]
        while os.path.dirname(chain[-1]) != chain[-1]:
            chain.append(os.path.dirname(chain[-1]))
    home = _home()
    for directory in chain:
        if os.path.isfile(os.path.join(directory, "pyproject.toml")):
            return directory
        if os.path.exists(os.path.join(directory, ".git")):
            return directory if directory != home else start
        if directory == home:
            return start
    return start


class _Budget:
    def __init__(self) -> None:
        self.paths = 0
        self.deadline = time.monotonic() + MAX_SECONDS

    def spend(self) -> bool:
        """Whether one more path may be looked at."""
        self.paths += 1
        return self.paths <= MAX_PATHS and time.monotonic() < self.deadline


def _moved(entries: dict[str, list[Any]], budget: _Budget) -> tuple[str | None, bool]:
    """The first of *entries* (``{path: [seen, stat]}``) not as recorded, and whether all
    were checked."""
    taken = time.time_ns()
    for path, (seen, recorded) in entries.items():
        if not budget.spend():
            return None, False
        current = fingerprint(path, taken)
        if (
            recorded is not None
            and current is not None
            and current[:5] == recorded[:5]
            and not racy(recorded)
        ):
            continue
        if seen == ABSENT and current is None:
            continue
        if observe(path, listing=seen.startswith("list:"))[0] != seen:
            return path, True
    return None, True


def _environment_moved(environment: dict[str, Any], budget: _Budget) -> tuple[str | None, bool]:
    """The first of E's watched paths not as recorded, and whether all were checked: as
    ``project_env._holds`` checks them."""
    for path, (mode, seen) in (environment.get("observed") or {}).items():
        if not budget.spend():
            return None, False
        if watch_state(path, mode) != seen:
            return path, True
    base = environment.get("base") or {}
    anchor = environment.get("anchor")
    if anchor and observe(anchor, listing=True)[0] == environment.get("anchor_listing"):
        # Nothing appeared in the project folder: only the names something was at.
        base = {path: base.get(path, ABSENT) for path in environment.get("base_check") or []}
    for path, seen in base.items():
        if seen == ABSENT and not os.path.exists(path):
            # Still absent: one stat, not a path the budget counts.
            continue
        if not budget.spend():
            return None, False
        if observe(path)[0] != seen:
            return path, True
    return None, True


def _shown(path: str, anchor: str) -> str:
    try:
        relative = os.path.relpath(path, anchor)
    except ValueError:
        return path
    return path if relative.startswith("..") else relative


def verdicts(digest: dict[str, Any], anchor: str) -> dict[str, str]:
    """Each cell's state, checked now: ``cached``, ``stale, ...``, ``failed`` or ``as of ...``."""
    budget = _Budget()
    written = digest.get("written_at") or 0
    as_of = "as of " + time.strftime("%Y-%m-%d %H:%M", time.localtime(written / 1000))
    environment, checked = _environment_moved(digest.get("environment") or {}, budget)
    cells = digest.get("cells") or []
    found: dict[str, str] = {}
    for cell in cells:
        cell_id = cell["cell"]
        if environment is not None:
            found[cell_id] = f"stale, environment changed ({_shown(environment, anchor)})"
            continue
        if cell.get("status") == "error":
            found[cell_id] = "failed"
            continue
        if cell.get("status") == "stale":
            changed = cell.get("changed")
            found[cell_id] = f"stale, {_shown(changed, anchor)} changed" if changed else "stale"
            continue
        if not checked:
            found[cell_id] = as_of
            continue
        entries = dict(cell.get("reads") or {})
        source = cell.get("source")
        if source:
            entries = {source[0]: [source[1], source[2]], **entries}
        moved, complete = _moved(entries, budget)
        if moved is not None:
            what = (
                "appeared" if (cell.get("reads") or {}).get(moved, [""])[0] == ABSENT else "changed"
            )
            found[cell_id] = f"stale, {_shown(moved, anchor)} {what}"
        else:
            found[cell_id] = "cached" if complete else as_of
    # Downstream of a stale cell is stale too, in the digest's order (the notebook's).
    for cell in cells:
        if found[cell["cell"]] in ("cached",) or found[cell["cell"]].startswith("as of"):
            if any(found.get(up, "").startswith("stale") for up in cell.get("upstream") or []):
                found[cell["cell"]] = "stale, upstream"
    return found


def _cost(cold_ms: int | None) -> str | None:
    if not cold_ms:
        return None
    return f"{cold_ms / 1000:.0f} s to build" if cold_ms >= 1000 else f"{cold_ms} ms to build"


def render(digest: dict[str, Any], anchor: str, run_tool: str, prompt: str | None) -> str:
    """The block for the agent; empty when nothing is to be said."""
    cells = digest.get("cells") or []
    if prompt is not None:
        words = set(re.findall(r"\w+", prompt))
        cells = [c for c in cells if c["cell"] in words or words & set(c.get("defines") or [])]
        digest = {**digest, "cells": cells}
    if not cells:
        return ""
    found = verdicts(digest, anchor)
    lines = []
    for cell in cells[:MAX_LINES]:
        names = ", ".join(cell.get("defines") or []) or "no variables"
        parts = [p for p in (_cost(cell.get("cold_ms")), found[cell["cell"]]) if p]
        lines.append(f"- {names} (cell {cell['cell']}): {', '.join(parts)}")
    if len(cells) > MAX_LINES:
        lines.append(f"- and {len(cells) - MAX_LINES} more; the scratchpad tool lists them")
    note = f"Strata scratchpad for this project; build on these with {run_tool}"
    return f'<strata_scratchpad note="{note}">\n' + "\n".join(lines) + "\n</strata_scratchpad>\n"


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "session-start"
    run_tool = argv[2] if len(argv) > 2 else "mcp__strata__run"
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    anchor = find_anchor(str(payload.get("cwd") or os.getcwd()))
    try:
        with open(os.path.join(anchor, DIGEST), encoding="utf-8") as file:
            digest = json.load(file)
    except (OSError, ValueError):
        return 0
    if not isinstance(digest, dict) or digest.get("version") != 1:
        return 0
    prompt = str(payload.get("prompt") or "") if mode == "prompt" else None
    sys.stdout.write(render(digest, anchor, run_tool, prompt))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
