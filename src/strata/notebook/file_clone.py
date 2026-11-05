"""Copy-on-write clones, where the filesystem has them.

A clone is instant and shares disk blocks until either side writes, so unlike a hard link a
write to it never reaches the original.
"""

from __future__ import annotations

import io
import logging
import os
import stat
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DARWIN_AT_FDCWD = -2
_LINUX_FICLONE = 0x40049409


def clone_reader(source: Any, path: Path) -> bool:
    """:func:`clone_file` of what *source* reads, if it is a plain file opened to read.

    Not any reader with a ``fileno``: a gzip or decrypting one hands back the file under it,
    whose bytes are not what it reads.
    """
    if isinstance(source, io.BufferedReader) and isinstance(source.raw, io.FileIO):
        return clone_file(source.fileno(), path)
    return False


def clone_file(source_fd: int, path: Path) -> bool:
    """Make *path*, which must not exist, a clone of an open file.

    ``False``, leaving no new file, where that cannot be done: most often another filesystem
    or one without clones. The clone gets the mode and times a new file gets, not the source's.
    """
    try:
        if sys.platform == "darwin":
            _clone_darwin(source_fd, path)
        elif sys.platform == "linux":
            _clone_linux(source_fd, path)
        else:
            return False
    except OSError as exc:
        logger.debug("Not cloning to %s: %s", path, exc)
        return False
    return True


def _clone_darwin(source_fd: int, path: Path) -> None:
    import ctypes

    # fclonefileat keeps the source's mode and times, and refuses an existing file.
    path.touch(exist_ok=False)
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        path.unlink()
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.fclonefileat(source_fd, _DARWIN_AT_FDCWD, os.fsencode(path), 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(path))
        os.chmod(path, mode)
        os.utime(path)
    except OSError:
        path.unlink(missing_ok=True)
        raise


def _clone_linux(source_fd: int, path: Path) -> None:
    import fcntl

    with path.open("xb") as target:
        try:
            fcntl.ioctl(target, _LINUX_FICLONE, source_fd)
        except OSError:
            path.unlink(missing_ok=True)
            raise
