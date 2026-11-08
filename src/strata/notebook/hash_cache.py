"""Each local input file's sha256, remembered by its stat so an unchanged file is not read.

A local input is keyed on its content. The stat only decides whether the remembered hash
can be reused, so a stat that moves for no reason (an xattr, ``chmod``, a hard link)
costs a read, never a rerun. The size, mtime, ctime, inode and device must all match
the remembered stat. On Windows ``st_ctime`` is the creation time, so a rewrite that
keeps the size and restores the mtime goes unseen there.

A hash taken within :data:`_RACY_NS` of the file's last change is used once and checked
again next time (git's rule for its index). Without that, a write in the same timestamp
tick as the read would look like no change.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

# Wider than the coarsest mtime in use (FAT: 2 s).
_RACY_NS = 2_000_000_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_hashes (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    ctime_ns INTEGER NOT NULL,
    ino INTEGER NOT NULL,
    dev INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    hashed_at_ns INTEGER NOT NULL
)
"""


@dataclass(frozen=True)
class Hashed:
    """A file's sha256. ``settled`` is false when the file changed while it was read."""

    sha256: str
    settled: bool


def _stat_key(st: os.stat_result) -> tuple[int, int, int, int, int]:
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev)


def _read_digest(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class FileHashes:
    """The hashes of one notebook's local input files, in a SQLite file of their own.

    Paths are absolute, as given: a link and its target each get a row, both keyed on
    the target's stat. Safe to share between threads.

    *clock* is the wall clock in ns. A local file's times come from the same clock; a
    network share's server clock can drift from it.
    """

    def __init__(self, db_path: Path, *, clock: Callable[[], int] = time.time_ns) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            db_path, timeout=30.0, check_same_thread=False, isolation_level=None
        )
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def cached(self, path: Path) -> str | None:
        """*path*'s hash if the remembered one still holds, else ``None``. Never reads it.

        Raises:
            OSError: *path* cannot be stat'ed.
        """
        path = Path(os.path.abspath(path))
        return self._trusted(path, _stat_key(os.stat(path)))

    def hash_of(self, path: Path) -> Hashed:
        """*path*'s hash: the remembered one if it still holds, else read and remembered.

        A file that changed while it was read is not remembered, and its hash is
        returned unsettled.

        Raises:
            OSError: *path* cannot be stat'ed or read.
        """
        path = Path(os.path.abspath(path))
        key = _stat_key(os.stat(path))
        known = self._trusted(path, key)
        if known is not None:
            return Hashed(known, settled=True)
        # Before the read: a write during it is then never older than the hash.
        hashed_at = self._clock()
        digest = _read_digest(path)
        if _stat_key(os.stat(path)) != key:
            return Hashed(digest, settled=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO file_hashes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(path), *key, digest, hashed_at),
            )
        return Hashed(digest, settled=True)

    def prune(self, root: Path, seen: Collection[Path]) -> int:
        """Forget the files under *root* not in *seen*, what a walk of it just found.

        Returns how many were forgotten.
        """
        prefix = os.path.join(os.path.abspath(root), "")
        keep = {os.path.abspath(p) for p in seen}
        with self._lock:
            rows = self._conn.execute(
                "SELECT path FROM file_hashes WHERE substr(path, 1, ?) = ?",
                (len(prefix), prefix),
            ).fetchall()
            gone = [(path,) for (path,) in rows if path not in keep]
            self._conn.executemany("DELETE FROM file_hashes WHERE path = ?", gone)
        return len(gone)

    def _trusted(self, path: Path, key: tuple[int, int, int, int, int]) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT size, mtime_ns, ctime_ns, ino, dev, sha256, hashed_at_ns "
                "FROM file_hashes WHERE path = ?",
                (str(path),),
            ).fetchone()
        if row is None or tuple(row[:5]) != key:
            return None
        mtime_ns, ctime_ns, digest, hashed_at = row[1], row[2], row[5], row[6]
        if max(mtime_ns, ctime_ns) + _RACY_NS >= hashed_at:
            return None
        return digest
