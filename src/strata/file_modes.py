"""Owner-only modes for the state a server keeps on disk.

Under the default umask that state is world-readable, so any other account on
the host, the harness user that runs cells among them, could read artifact
metadata, blobs, the cache or a notebook's console. These helpers create it
0700 / 0600 and narrow what an earlier release left wider. They only ever
remove permission bits, so a stricter umask stands. POSIX modes do not apply on
Windows, where they do nothing.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

PRIVATE_DIR = 0o700
# Others may pass through to a known path inside, as the harness user reaches a
# notebook's per-run directories, but may not list or read the directory itself.
PASS_THROUGH_DIR = 0o711
PRIVATE_FILE = 0o600


def private_dir(path: Path, mode: int = PRIVATE_DIR) -> Path:
    """Create *path* (and its parents) with *mode* on the leaf, or narrow it to *mode*."""
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    narrow(path, mode)
    return path


def private_file(path: Path) -> Path:
    """Create *path* empty and 0600 if it is missing, or narrow it to 0600.

    Created before SQLite opens a database so it never exists wider; SQLite gives
    the ``-wal`` and ``-shm`` files the database file's mode.
    """
    if os.name != "nt" and not path.exists():
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, PRIVATE_FILE))
    if path.exists():
        narrow(path, PRIVATE_FILE)
    return path


def narrow(path: Path, mode: int) -> None:
    """Drop every permission bit of *path* that *mode* does not have; never add one.

    A path owned by another account is left alone: it is not this process's to change.
    """
    if os.name == "nt":
        return
    info = path.stat()
    if info.st_uid != os.geteuid():
        return
    current = stat.S_IMODE(info.st_mode)
    narrowed = current & (mode | ~0o777)
    if narrowed != current:
        os.chmod(path, narrowed)
