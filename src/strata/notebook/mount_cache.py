"""A machine's copies of read-only remote mounts, shared and bounded.

``objects/`` holds each remote file once, keyed by its URI and version (etag, size,
mtime). ``views/<key>/data`` is one mount's listing at one fingerprint, assembled from
hard links into ``objects/``, so notebooks reading the same prefix share bytes, and a
prefix with one new file costs one download. Objects are read-only on POSIX: a hard link
shares the file, so a cell writing into one would change it for every view.

A sweep keeps the objects under ``max_bytes`` and drops views unused for
``max_idle_days``, least recently used first. A view used within the last hour is kept
even over the bound, since a running cell may still be reading it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 20 * 1024**3
DEFAULT_MAX_IDLE_DAYS = 30.0
# Never evict a view used this recently: a cell may still be reading it.
_MIN_IDLE_SECONDS = 3600.0
# Sweep at least this often per cache root, even when nothing was downloaded.
_SWEEP_INTERVAL_SECONDS = 600.0
# Left behind by a process that died mid-download or mid-build.
_STALE_TEMP_SECONDS = 86400.0
_LOCK = ".lock"  # its mtime is the view's last use

_last_sweep: dict[Path, float] = {}


def default_root() -> Path:
    return Path.home() / ".strata" / "mount_cache"


def object_key(protocol: str, scope: str, name: str, info: Mapping[str, Any]) -> str | None:
    """A remote file's identity at its current version, or ``None`` when it has no version.

    *scope* names how the store is reached (a digest of the storage options), so an object
    is only reused by a caller that fetched the listing the same way.
    """
    etag = str(info.get("ETag", info.get("etag", "")))
    mtime = str(info.get("LastModified", info.get("mtime", "")))
    if not etag and not mtime:
        return None
    parts = [protocol, scope, name, str(info.get("size", "")), etag, mtime]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


class MountCache:
    """Read-only remote mount data under *root*, shared by every notebook on the machine."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_idle_days: float = DEFAULT_MAX_IDLE_DAYS,
    ):
        self.root = root
        self.objects = root / "objects"
        self.views = root / "views"
        self.max_bytes = max_bytes  # 0: no cap
        self.max_idle_seconds = max_idle_days * 86400.0  # 0: no idle limit
        self._grew = False  # downloaded since the last sweep

    @classmethod
    def from_config(cls, config: Any) -> MountCache:
        configured = getattr(config, "notebook_mount_cache_dir", None)
        storage = getattr(config, "notebook_storage_dir", None)
        if configured is not None:
            root = Path(configured)
        elif storage is not None:
            root = Path(storage).parent / "mount_cache"
        else:
            root = default_root()
        return cls(
            root,
            max_bytes=getattr(config, "notebook_mount_cache_max_bytes", DEFAULT_MAX_BYTES),
            max_idle_days=getattr(
                config, "notebook_mount_cache_max_idle_days", DEFAULT_MAX_IDLE_DAYS
            ),
        )

    def find_view(self, key: str) -> Path | None:
        """The data directory of the view built under *key*, marked used, or ``None``."""
        view = self.views / key
        try:
            os.utime(view / _LOCK)
        except FileNotFoundError:
            return None
        return view / "data"

    def build_view(
        self,
        key: str,
        files: list[tuple[Path, str | None, str]],
        fetch: Callable[[str, Path], None],
    ) -> Path:
        """Assemble the view *key* from ``(relative path, object key, remote name)`` entries.

        Downloads only the objects not already here, through ``fetch(remote name, dest)``. A
        file with no object key is downloaded into the view itself. Callers validate the
        relative paths.
        """
        _mkdir_unlisted(self.views)
        building = self.views / f".tmp-{uuid.uuid4().hex}"
        data = building / "data"
        data.mkdir(parents=True)
        downloaded = 0
        try:
            for rel, okey, remote_name in files:
                dest = data / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Two names on one path (a case-insensitive disk): the last wins.
                dest.unlink(missing_ok=True)
                if okey is None:
                    fetch(remote_name, dest)
                    downloaded += 1
                else:
                    downloaded += self._link_object(okey, remote_name, dest, fetch)
            (building / _LOCK).touch()
            try:
                building.rename(self.views / key)
            except OSError:
                # Another process built the same view first; theirs is identical.
                shutil.rmtree(building, ignore_errors=True)
        except BaseException:
            shutil.rmtree(building, ignore_errors=True)
            raise
        logger.info("mount view %s: %d of %d files downloaded", key[:12], downloaded, len(files))
        self._grew = self._grew or downloaded > 0
        found = self.find_view(key)
        if found is None:
            raise RuntimeError("the mount cache removed a view as it was built")
        return found

    def _link_object(
        self, okey: str, remote_name: str, dest: Path, fetch: Callable[[str, Path], None]
    ) -> int:
        """Hard-link object *okey* to *dest*, downloading it if absent; 1 if downloaded."""
        path = self.objects / okey[:2] / okey
        downloaded = 0
        # A sweep can remove an object between its download and the link.
        for _ in range(3):
            if not path.exists():
                self._download(path, remote_name, fetch)
                downloaded = 1
            try:
                os.link(path, dest)
                return downloaded
            except FileNotFoundError:
                continue
            except OSError as exc:
                # No hard links here (some network or FUSE disks): a copy still serves.
                logger.debug("mount cache: copying %s, no hard link: %s", path.name, exc)
                shutil.copyfile(path, dest)
                _make_read_only(dest)
                return downloaded
        raise RuntimeError(f"could not keep {remote_name} in the mount cache")

    def _download(self, path: Path, remote_name: str, fetch: Callable[[str, Path], None]) -> None:
        _mkdir_unlisted(self.objects)
        _mkdir_unlisted(path.parent)
        partial = self.objects / f".tmp-{uuid.uuid4().hex}"
        try:
            fetch(remote_name, partial)
            # Its mtime says when it arrived, so a sweep spares a fresh download.
            os.utime(partial)
            _make_read_only(partial)
            os.replace(partial, path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    def sweep(self, *, force: bool = False, now: float | None = None) -> None:
        """Evict until under the bound and nothing is idle past the limit.

        Unless forced or something was downloaded since the last one, at most once per
        ``_SWEEP_INTERVAL_SECONDS`` for this root.
        """
        now = time.time() if now is None else now
        last = _last_sweep.get(self.root)
        force = force or self._grew
        if not force and last is not None and now - last < _SWEEP_INTERVAL_SECONDS:
            return
        _last_sweep[self.root] = now
        self._grew = False

        objects: dict[int, tuple[Path, int]] = {}
        total = 0
        if self.objects.is_dir():
            for entry in self.objects.iterdir():
                if entry.name.startswith(".tmp-"):
                    _remove_if_stale(entry, now)
                    continue
                for path in entry.iterdir():
                    try:
                        st = path.lstat()
                    except FileNotFoundError:
                        continue  # a concurrent sweep took it
                    if st.st_nlink == 1 and now - st.st_mtime > _MIN_IDLE_SECONDS:
                        # In no view: an evicted view's leftover.
                        path.unlink(missing_ok=True)
                        continue
                    objects[st.st_ino] = (path, st.st_size)
                    total += st.st_size

        views: list[tuple[float, Path]] = []
        if self.views.is_dir():
            for view in self.views.iterdir():
                if view.name.startswith("."):
                    _remove_if_stale(view, now)
                    continue
                try:
                    views.append((os.stat(view / _LOCK).st_mtime, view))
                except FileNotFoundError:
                    views.append((0.0, view))
        views.sort()

        for used, view in views:
            idle = now - used
            over = self.max_bytes > 0 and total > self.max_bytes
            expired = self.max_idle_seconds > 0 and idle > self.max_idle_seconds
            if not (over or expired):
                break
            if idle <= _MIN_IDLE_SECONDS:
                break
            for ino in _remove_view(view, self.views):
                found = objects.get(ino)
                if found is None:
                    continue
                path, size = found
                try:
                    if path.lstat().st_nlink > 1:
                        continue  # another view still holds it
                    path.unlink()
                except FileNotFoundError:
                    pass
                del objects[ino]
                total -= size


def _mkdir_unlisted(path: Path) -> None:
    """Make *path* passable but not listable by others (POSIX).

    A harness user must read a view it is handed but should not find another notebook's by
    listing: a view's key hashes the storage options that reach it.
    """
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o044:
        path.chmod(mode & ~0o044)


def _remove_view(view: Path, views: Path) -> set[int]:
    """Delete *view*, returning the inodes of its files."""
    inodes: set[int] = set()
    for dirpath, _dirs, names in os.walk(view):
        for name in names:
            try:
                inodes.add((Path(dirpath) / name).lstat().st_ino)
            except FileNotFoundError:
                continue  # a concurrent sweep is removing it
    trash = views / f".trash-{uuid.uuid4().hex}"
    try:
        # Out of the way first, so a reader never finds a half-deleted view.
        view.rename(trash)
    except OSError as exc:
        logger.warning("could not evict mount view %s: %s", view.name, exc)
        return set()
    shutil.rmtree(trash, ignore_errors=True)
    return inodes


def _make_read_only(path: Path) -> None:
    """Clear the write bits (POSIX), keeping the read bits the umask left."""
    if os.name != "nt":
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)


def _remove_if_stale(path: Path, now: float) -> None:
    try:
        if now - path.lstat().st_mtime <= _STALE_TEMP_SECONDS:
            return
    except FileNotFoundError:
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
