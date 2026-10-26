"""A GCS service-account key as the path ``GOOGLE_APPLICATION_CREDENTIALS`` wants.

PyIceberg and the lake-file reader open ``gs://`` through PyArrow, whose GCS client reads
credentials only from that variable, and only as a file path. The configured
``gcs_credentials_json`` is a path or the key material itself, so the material is spilled
to a private file. The artifact store's GCS backend takes the key directly and needs none
of this.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path


def resolve_gcs_credentials(credentials: str) -> str:
    """Return a filesystem path for *credentials*, writing inline JSON key material to a file.

    A value that parses as a JSON object is spilled to a 0600 file named by the key's
    digest (so a hard-killed process leaves one file, rewritten by the next start) and
    removed at normal exit; anything else is passed through as a path. On Windows the
    0600 does not apply; prefer a mounted credential file there.

    Raises:
        OSError: the predictable path holds something that is not this user's own
            regular file (a planted symlink or file, in a shared temp dir).
    """
    text = credentials.strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return credentials
    if not isinstance(parsed, dict):
        return credentials

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"strata_gcs_{digest}.json"
    # Created with mode 0o600 so the key is never world-readable; the chmod
    # below fixes a file left behind by a previous run. O_NOFOLLOW and the fstat
    # check keep the key out of another user's file at this name; the truncate
    # waits for them so the key never lands in one.
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
            raise PermissionError(f"{path} is not a regular file owned by this user")
        os.ftruncate(fd, 0)
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)
    atexit.register(lambda: path.unlink(missing_ok=True))
    return str(path)
