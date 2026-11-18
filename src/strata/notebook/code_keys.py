"""A scratchpad cell's key over what its run read: the code it loaded and the files it opened.

L, the lookup key, is the cell's provenance hash as cellgraph makes it. R, what a run read
(``code_reads.Reads``), completes it: F is L when R is empty, else L derived over R. A run's
outputs are stored under F, so a downstream cell, which keys on what its upstream stored
under, changes with it.

F cannot be known before the files are looked at, so an index maps each L to the reads of
the runs keyed under it (``.strata/reads/code.json``, the newest :data:`MAX_SETS`). A lookup
observes those paths as they are now and tries the F each set makes. The index only
points: F is checked against the store, so losing it costs a rerun, never a wrong result.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vestigo.provenance import derive_subkey

from strata.notebook.code_reads import fingerprint, observe, racy

INDEX = Path(".strata") / "reads" / "code.json"
MAX_SETS = 8


def code_key(lookup: str, reads: Mapping[str, str]) -> str:
    """F: *lookup* itself when nothing was read, else *lookup* derived over *reads*."""
    if not reads:
        return lookup
    digest = hashlib.sha256(json.dumps(sorted(reads.items())).encode()).hexdigest()
    return derive_subkey(lookup, f"code={digest}")


@dataclass(frozen=True)
class Resolved:
    """The key a cell's result is found under, and why none of its reads' keys was."""

    key: str
    # The first path the newest run under the lookup key read that is not as it read it.
    changed: str | None = None


class CodeIndex:
    """Each lookup key's runs, by what they read: ``{L: [{path: [seen, stat]}, ...]}``."""

    def __init__(self, notebook_dir: Path) -> None:
        self.path = notebook_dir / INDEX
        self._lock = threading.Lock()
        self._data: dict[str, list[dict[str, list[Any]]]] = {}
        self._loaded_as: tuple[int, int] | None = None

    def _load(self) -> dict[str, list[dict[str, list[Any]]]]:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self._data, self._loaded_as = {}, None
            return self._data
        if (st.st_mtime_ns, st.st_size) != self._loaded_as:
            try:
                data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                # A hint only: a torn or foreign file costs reruns.
                data = {}
            self._data = data if isinstance(data, dict) else {}
            self._loaded_as = (st.st_mtime_ns, st.st_size)
        return self._data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".json.partial")
        partial.write_text(json.dumps(self._data))
        os.replace(partial, self.path)
        st = self.path.stat()
        self._loaded_as = (st.st_mtime_ns, st.st_size)

    def record(self, lookup: str, reads: Mapping[str, str], stat: Mapping[str, list[int]]) -> None:
        """Note a run under *lookup* that read *reads*, newest first."""
        entry = {path: [seen, stat.get(path)] for path, seen in reads.items()}
        with self._lock:
            sets = [s for s in self._load().get(lookup, []) if s.keys() != entry.keys()]
            self._data[lookup] = [entry, *sets][:MAX_SETS]
            self._save()

    def resolve(self, lookup: str, has_result: Callable[[str], bool]) -> Resolved:
        """The key *lookup*'s result is stored under now, if any is (*has_result* says), else
        *lookup* itself: a run that read nothing is stored there."""
        changed: str | None = None
        for index, (key, moved) in enumerate(self._keys(lookup)):
            if index == 0:
                changed = moved
            if has_result(key):
                return Resolved(key, changed)
        return Resolved(lookup, changed)

    def _keys(self, lookup: str) -> Iterator[tuple[str, str | None]]:
        """The F of each run under *lookup*, as its reads are now, and the first of its paths
        that is not as that run read it."""
        with self._lock:
            sets = list(self._load().get(lookup, []))
        for entry in sets:
            now, moved, retaken = _now(entry)
            if retaken:
                self._retake(lookup, entry, retaken)
            yield code_key(lookup, now), moved

    def _retake(self, lookup: str, entry: dict[str, list[Any]], retaken: dict[str, Any]) -> None:
        """Keep the fresh stat of paths hashed again and found as they were, so the next
        check takes the fast path."""
        with self._lock:
            for stored in self._load().get(lookup, []):
                if stored.keys() == entry.keys():
                    for path, print_ in retaken.items():
                        stored[path][1] = print_
                    self._save()
                    return


def _now(
    entry: Mapping[str, list[Any]],
) -> tuple[dict[str, str], str | None, dict[str, list[int]]]:
    """Each path of *entry* as it is now, the first one changed, and the stat to keep for
    the paths hashed again and found unchanged.

    A path is hashed only when its stat differs from the one recorded, or either is racy:
    a differing stat is a reason to look, never a verdict.
    """
    now: dict[str, str] = {}
    moved: str | None = None
    retaken: dict[str, list[int]] = {}
    taken = time.time_ns()
    for path, (seen, recorded) in entry.items():
        current = fingerprint(path, taken)
        if (
            recorded is not None
            and current is not None
            and current[:5] == recorded[:5]
            and not racy(recorded)
        ):
            now[path] = seen
            continue
        if seen == "absent" and current is None:
            now[path] = seen
            continue
        observed, print_ = observe(path, listing=seen.startswith("list:"))
        now[path] = observed
        if observed != seen:
            moved = moved or path
        elif print_ is not None and print_ != recorded and not racy(print_):
            retaken[path] = print_
    return now, moved, retaken
