"""File mount resolution and fingerprinting for notebook cells.

Local (``file://``) mounts are used directly; remote ones (``s3``, ``gs``,
``az``) are copied locally via ``fsspec`` before the subprocess spawns, so the
harness only sees ``pathlib.Path``: read-only ones into the machine's shared
:class:`~strata.notebook.mount_cache.MountCache`. Read-only mounts are
fingerprinted into provenance; read-write mounts are synced back after
execution and do not participate in provenance. What a cell writes through one is
recorded instead: :meth:`MountResolver.sync_back` returns the files it created or
changed, with sizes and digests.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Collection, Coroutine, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse

from pydantic import BaseModel
from vestigo.digests import FileDigests
from vestigo.provenance import derive_subkey

from strata.notebook import nonce
from strata.notebook.credentials import (
    CredentialError,
    CredentialResolver,
    credential_identity,
    read_credential_file,
)
from strata.notebook.harness_user import UnsafeRunFile, open_tree_file
from strata.notebook.models import MountMode, MountSpec
from strata.notebook.mount_cache import MountCache, object_key

logger = logging.getLogger(__name__)

DIGESTS_FILE = "hashes.sqlite"
"""Beside a resolver's ``cache_dir``: the digests of its local mounts' files, by stat."""

STALENESS_READ_BYTES = 64 * 1024 * 1024
"""What a staleness check reads of a local mount's files the cache holds no digest for.
Past it the mount's component is :func:`nonce.unhashed_nonce`'s, and the session hashes
the mount in the background (``NotebookSession._hash_mounts``)."""

_LOG_READ_BYTES = 1024 * 1024 * 1024
# A background hash's tries, and the wait between them: past the digest cache's 2 s rule.
_HASH_ATTEMPTS = 3
_SETTLE_SECONDS = 2.5
# What a hash reads between two progress reports; a hash of no more reports none.
_PROGRESS_BATCH_BYTES = 256 * 1024 * 1024
# ``digest_many``'s read threads: a batch of fewer files would leave some idle.
_READ_THREADS = min(32, os.cpu_count() or 1)

HashProgress = Callable[[int, int, bool], None]
"""``(bytes read, bytes to read, finished)``, from the hashing thread: once at the start,
after each batch, and finished last, also when the hash failed."""


def on_loop(
    loop: asyncio.AbstractEventLoop, send: Callable[[int, int, bool], Coroutine[Any, Any, None]]
) -> HashProgress:
    """A :data:`HashProgress` that runs *send* on *loop*, each report after the one before,
    so a client never gets a report after the finished one."""
    lock = asyncio.Lock()

    async def in_order(bytes_done: int, total: int, done: bool) -> None:
        async with lock:
            await send(bytes_done, total, done)

    def report(bytes_done: int, total: int, done: bool) -> None:
        coroutine = in_order(bytes_done, total, done)
        try:
            asyncio.run_coroutine_threadsafe(coroutine, loop)
        except RuntimeError:
            coroutine.close()  # the loop closed: no client to tell

    return report


def mount_progress_frame(
    cell_id: str, mount: str, bytes_done: int, total: int, done: bool
) -> dict[str, Any]:
    """A ``cell_mount_progress`` payload: *mount*'s hash, as *cell_id* names it."""
    return {
        "cell_id": cell_id,
        "mount": mount,
        "bytes_done": bytes_done,
        "bytes": total,
        "done": done,
    }


# Digest caches kept open, the least recently used idle one closed past this.
_OPEN_DIGEST_CACHES = 32


@dataclass
class ResolvedMount:
    """A mount that has been resolved to a local path."""

    spec: MountSpec
    local_path: Path
    fingerprint: str | None  # None for RW mounts: the cell is not cacheable.
    # Remote RW only: synced back after execution.
    staging_dir: Path | None = None
    # RW only: each file's (size, mtime_ns) before the cell ran, to tell what it wrote.
    baseline: dict[str, tuple[int, int]] | None = None
    # Remote RO only: the bytes downloaded to build the view (0 when it was cached).
    bytes_moved: int | None = None
    # RO only: each file's version by relative path, the listing's for a remote mount and
    # the sha256 for a local one (none for a pinned one).
    file_versions: dict[str, str] | None = None
    # Local RO only: where its files' digests are cached, for one that has no version.
    digests_path: Path | None = None


class WrittenFile(BaseModel):
    """One file a cell created or changed through a read-write mount."""

    path: str  # relative to the mount, POSIX separators
    size: int
    sha256: str


class MountWrite(BaseModel):
    """What one run wrote through one read-write mount."""

    mount: str
    uri: str
    files: list[WrittenFile]


def _file_states(root: Path) -> dict[str, tuple[int, int]]:
    """``{relative path: (size, mtime_ns)}`` for each regular file under *root*, no links."""
    states: dict[str, tuple[int, int]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            st = path.lstat()
            if stat.S_ISREG(st.st_mode):
                states[path.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return states


def _sha256(f: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := f.read(1 << 20):
        digest.update(chunk)
    return digest.hexdigest()


def _open_written(rm: ResolvedMount, rel: str) -> BinaryIO:
    """A file the run wrote, opened for reading.

    A staging copy is the cell's to write (under the harness user it owns it), so no link
    in it is followed: one pointing at the server's files would be read, or uploaded.
    """
    if rm.staging_dir is not None:
        return open_tree_file(rm.staging_dir, rel)
    return open(rm.local_path / rel, "rb")


def written_files(rm: ResolvedMount) -> list[WrittenFile]:
    """Files under a read-write mount that are new or changed since it was prepared."""
    before = rm.baseline or {}
    files: list[WrittenFile] = []
    for rel, state in sorted(_file_states(rm.local_path).items()):
        if before.get(rel) == state:
            continue
        try:
            with _open_written(rm, rel) as f:
                digest = _sha256(f)
        except UnsafeRunFile:
            logger.warning("mount %s: not recording %s, a link", rm.spec.name, rel)
            continue
        files.append(WrittenFile(path=rel, size=state[0], sha256=digest))
    return files


def _upload_staging(fs: Any, remote_uri: str, rm: ResolvedMount) -> None:
    """Upload each regular file of the staging copy to its place under *remote_uri*."""
    assert rm.staging_dir is not None
    for rel in sorted(_file_states(rm.staging_dir)):
        try:
            source = _open_written(rm, rel)
        except UnsafeRunFile:
            logger.warning("mount %s: not syncing %s, a link", rm.spec.name, rel)
            continue
        with source, fs.open(f"{remote_uri.rstrip('/')}/{rel}", "wb") as target:
            shutil.copyfileobj(source, target, 1 << 20)


# --- URI parsing ---


def parse_mount_uri(uri: str) -> tuple[str, str]:
    """Parse a mount URI into ``(scheme, path)``: ``s3://bucket/p`` gives ``("s3", "bucket/p")``.

    Raises:
        ValueError: If the URI is malformed or uses an unsupported scheme.
    """
    supported = {"file", "s3", "gs", "gcs", "az", "azure"}

    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if not scheme:
        return "file", uri

    if scheme not in supported:
        raise ValueError(
            f"Unsupported mount URI scheme '{scheme}' in '{uri}'. "
            f"Supported: {', '.join(sorted(supported))}"
        )

    if scheme == "gcs":
        scheme = "gs"
    elif scheme == "azure":
        scheme = "az"

    if scheme == "file":
        return "file", unquote(parsed.path)
    else:
        path = parsed.netloc
        if parsed.path and parsed.path != "/":
            path += parsed.path
        return scheme, path


def _is_remote(scheme: str) -> bool:
    return scheme != "file"


type MountCredentials = dict[str, dict[str, Any]]
"""Per-scheme fsspec storage options for ``MountResolver``.

Maps URI scheme (``s3``, ``gs``, ``az``) to a dict of options spread
into ``fsspec.filesystem(protocol, **opts)``. See fsspec backend docs
for valid keys (s3fs, gcsfs, adlfs). Per-mount ``options`` from
``[[mounts]]`` win on key collision.
"""


class MountResolver:
    """Resolves mount URIs to local paths for cell execution.

    Local mounts must exist. Read-only remote mounts are views in ``mount_cache``;
    read-write ones are staged under ``cache_dir``, which is the caller's own (per
    notebook or per job). ``credentials`` are per-scheme fsspec storage options.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        credentials: MountCredentials | None = None,
        credential_resolver: CredentialResolver | None = None,
        mount_cache: MountCache | None = None,
    ):
        # User-scoped, not a shared /tmp dir that invites cross-user cache poisoning.
        # Real callers always pass cache_dir.
        self.cache_dir = cache_dir or Path.home() / ".strata" / "mount_cache"
        self.mount_cache = mount_cache or MountCache(self.cache_dir)
        self.credentials = credentials or {}
        # Empty by default, so a mount naming an undefined credential fails by name
        # instead of reaching the store anonymously.
        self.credential_resolver = credential_resolver or CredentialResolver()
        self.digests_path = self.cache_dir.parent / DIGESTS_FILE
        self._fsspec_available: bool | None = None

    def storage_options(self, mount: MountSpec) -> dict[str, Any]:
        """Scheme credentials, then the scheme default and named credential, then options."""
        scheme, _ = parse_mount_uri(mount.uri)
        return {
            **self.credentials.get(scheme, {}),
            **self.credential_resolver.storage_options(scheme, mount.credential, mount.options),
        }

    def _check_fsspec(self) -> bool:
        """Check if fsspec is available."""
        if self._fsspec_available is None:
            try:
                import fsspec  # noqa: F401

                self._fsspec_available = True
            except ImportError:
                self._fsspec_available = False
        return self._fsspec_available

    async def prepare_mounts(
        self,
        mounts: list[MountSpec],
        *,
        only: Mapping[str, str] | None = None,
    ) -> dict[str, ResolvedMount]:
        """Resolve all mounts to local paths, keyed by mount name.

        *only* gives, by mount name, the glob of the files a run reads: a remote mount
        downloads no others. Its fingerprint still covers the whole listing.

        Raises:
            ValueError: If a local mount path doesn't exist.
            ImportError: If a remote mount is requested but fsspec is unavailable.
        """
        resolved: dict[str, ResolvedMount] = {}
        cached = False

        for mount in mounts:
            scheme, path = parse_mount_uri(mount.uri)

            if scheme == "file":
                resolved[mount.name] = await self._resolve_local(mount, path)
            else:
                resolved[mount.name] = await self._resolve_remote(
                    mount,
                    scheme,
                    path,
                    glob=(only or {}).get(mount.name),
                )
                cached = cached or mount.mode == MountMode.READ_ONLY

        if cached:
            # Once per prepare, not per mount: a sweep walks every object.
            await asyncio.to_thread(self.mount_cache.sweep)
        return resolved

    async def _resolve_local(
        self,
        mount: MountSpec,
        local_path: str,
    ) -> ResolvedMount:
        """Resolve a local file:// mount."""
        p = Path(local_path)

        if mount.mode == MountMode.READ_WRITE:
            # RW: no fingerprint (side effect).
            p.mkdir(parents=True, exist_ok=True)
            return ResolvedMount(
                spec=mount,
                local_path=p,
                fingerprint=None,
                baseline=await asyncio.to_thread(_file_states, p),
            )

        if not p.exists():
            raise ValueError(f"Local mount '{mount.name}' path does not exist: {p}")

        if mount.pin is not None:
            fingerprint = MountFingerprinter.fingerprint_mount_sync(mount)
            assert fingerprint is not None
            return ResolvedMount(
                spec=mount, local_path=p, fingerprint=fingerprint, digests_path=self.digests_path
            )
        digests = await asyncio.to_thread(local_digests, p, self.digests_path)
        assert digests is not None
        return ResolvedMount(
            spec=mount,
            local_path=p,
            fingerprint=_fingerprint_digests(digests),
            file_versions=digests,
            digests_path=self.digests_path,
        )

    async def _resolve_remote(
        self,
        mount: MountSpec,
        scheme: str,
        remote_path: str,
        *,
        glob: str | None = None,
    ) -> ResolvedMount:
        """Resolve a remote mount (S3, GCS, Azure) via fsspec."""
        if not self._check_fsspec():
            raise ImportError(
                f"Remote mount '{mount.name}' requires fsspec. "
                f"Install it with: pip install fsspec s3fs gcsfs adlfs"
            )

        if mount.mode == MountMode.READ_ONLY:
            return await self._resolve_remote_ro(mount, scheme, remote_path, glob)
        # RW: stage locally, sync back after execution.
        mount_hash = hashlib.sha256(mount.uri.encode()).hexdigest()[:12]
        return await self._resolve_remote_rw(
            mount,
            scheme,
            remote_path,
            self.cache_dir / f"{mount.name}_{mount_hash}",
        )

    async def _resolve_remote_ro(
        self,
        mount: MountSpec,
        scheme: str,
        remote_path: str,
        glob: str | None = None,
    ) -> ResolvedMount:
        """Resolve a read-only remote mount to a view in the shared mount cache."""
        protocol = _scheme_to_fsspec_protocol(scheme)
        storage_options = self.storage_options(mount)
        try:
            fingerprint, local_path, moved, versions = await asyncio.to_thread(
                self._prepare_view, mount, protocol, remote_path, storage_options, glob
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to materialize remote mount '{mount.name}' from {mount.uri}: {e}"
            ) from e
        resolved = ResolvedMount(
            spec=mount,
            local_path=local_path,
            fingerprint=fingerprint,
            bytes_moved=moved,
            file_versions=versions,
        )
        # Held as long as the run keeps its resolved mounts, however long it runs.
        self.mount_cache.hold(resolved, local_path)
        return resolved

    def _prepare_view(
        self,
        mount: MountSpec,
        protocol: str,
        remote_path: str,
        storage_options: dict[str, Any],
        glob: str | None = None,
    ) -> tuple[str, Path, int, dict[str, str] | None]:
        """The mount's fingerprint, its view, the bytes downloaded for it and each file's
        version, from one listing."""
        fs = _mount_filesystem(protocol, storage_options)
        options_digest = _options_digest(storage_options)
        listing: dict[str, dict[str, Any]] | None = None
        if mount.pin is not None:
            fingerprint = hashlib.sha256(f"pin:{mount.pin}".encode()).hexdigest()
        else:
            listing = _list_remote_file_info(fs, protocol, remote_path)
            fingerprint = _fingerprint_listing(listing)
        key = _view_key(protocol, remote_path, options_digest, fingerprint, glob)

        def versions_of(listed: dict[str, dict[str, Any]]) -> dict[str, str]:
            # No scope: rotating a credential changes no file.
            keys = {
                _relative_remote_path(name, protocol, remote_path).as_posix(): object_key(
                    protocol, "", name, info
                )
                for name, info in listed.items()
            }
            return {rel: version for rel, version in keys.items() if version is not None}

        found = self.mount_cache.find_view(key)
        if found is not None:
            return fingerprint, found, 0, None if listing is None else versions_of(listing)
        if listing is None:
            listing = _list_remote_file_info(fs, protocol, remote_path)
        files = []
        for remote_name, info in sorted(listing.items()):
            rel = _relative_remote_path(remote_name, protocol, remote_path)
            if glob is not None and not glob_matches(rel.as_posix(), glob):
                continue
            _check_relative(rel, mount.name, remote_name)
            files.append(
                (rel, object_key(protocol, options_digest, remote_name, info), remote_name)
            )
        view, moved = self.mount_cache.build_view(
            key, files, lambda name, dest: fs.get(f"{protocol}://{name}", str(dest))
        )
        return fingerprint, view, moved, versions_of(listing)

    async def _resolve_remote_rw(
        self,
        mount: MountSpec,
        scheme: str,
        remote_path: str,
        local_dir: Path,
    ) -> ResolvedMount:
        """Resolve a read-write remote mount with staging directory."""
        staging = local_dir / "staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        protocol = _scheme_to_fsspec_protocol(scheme)
        storage_options = self.storage_options(mount)

        try:
            fs = _mount_filesystem(protocol, storage_options)
            staging_root = staging.resolve()
            remote_uri = f"{protocol}://{remote_path}"
            if fs.exists(remote_uri):
                # Validate each name before fetching: ``recursive=True`` hides the local paths
                # it writes, so a post-hoc scan can't see a file already placed outside staging.
                for remote_name in _list_remote_files(fs, protocol, remote_path):
                    rel = _relative_remote_path(remote_name, protocol, remote_path)
                    local_file = staging / rel
                    _assert_within(local_file, staging_root, mount.name, remote_name)
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    fs.get(f"{protocol}://{remote_name}", str(local_file))
        except Exception as e:
            raise RuntimeError(
                f"Failed to stage RW mount '{mount.name}' from {mount.uri}: {e}"
            ) from e

        # Catches a fetched symlink that points outside the staging dir.
        for entry in staging.rglob("*"):
            _assert_within(entry, staging_root, mount.name, str(entry))

        return ResolvedMount(
            spec=mount,
            local_path=staging,
            fingerprint=None,
            staging_dir=staging,
            baseline=await asyncio.to_thread(_file_states, staging),
        )

    async def sync_back(
        self,
        resolved: dict[str, ResolvedMount],
    ) -> list[MountWrite]:
        """Sync read-write mounts back to their remote URIs after a successful run.

        Returns what the run wrote, one entry per mount it wrote anything through. A
        deletion is not recorded, and is not synced back to a remote mount either.
        """
        writes: list[MountWrite] = []
        for name, rm in resolved.items():
            if rm.spec.mode != MountMode.READ_WRITE:
                continue
            files = await asyncio.to_thread(written_files, rm)
            if files:
                writes.append(MountWrite(mount=name, uri=rm.spec.uri, files=files))
            if rm.staging_dir is None:
                continue

            scheme, remote_path = parse_mount_uri(rm.spec.uri)
            if scheme == "file":
                # Local RW writes go directly.
                continue

            if not self._check_fsspec():
                raise ImportError(f"Cannot sync-back RW mount '{name}': fsspec not available")

            protocol = _scheme_to_fsspec_protocol(scheme)
            storage_options = self.storage_options(rm.spec)

            try:
                fs = _mount_filesystem(protocol, storage_options)
                await asyncio.to_thread(_upload_staging, fs, f"{protocol}://{remote_path}", rm)
                logger.info(
                    "Synced RW mount '%s' back to %s",
                    name,
                    rm.spec.uri,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to sync-back RW mount '{name}' to {rm.spec.uri}: {e}"
                ) from e
        return writes


class MountFingerprinter:
    """Content fingerprints for mount URIs, so cache entries invalidate when contents change."""

    @staticmethod
    def fingerprint_local_sync(
        path: Path,
        digests_path: Path,
        *,
        read_bytes: int | None = None,
        on_progress: HashProgress | None = None,
    ) -> str | None:
        """Fingerprint a local file or directory from each file's relative path and content.

        The digests come from the cache at *digests_path*, so an unchanged file is stat'ed,
        not read. ``None`` when *read_bytes* is given and the files the cache holds no
        digest for are more than that.
        """
        if not path.exists():
            return hashlib.sha256(b"missing").hexdigest()
        digests = local_digests(path, digests_path, read_bytes=read_bytes, on_progress=on_progress)
        return None if digests is None else _fingerprint_digests(digests)

    @staticmethod
    def fingerprint_remote_sync(
        scheme: str,
        remote_path: str,
        storage_options: dict[str, Any] | None = None,
    ) -> str:
        """Fingerprint a remote path from listing metadata (ETags, sizes, mtimes); no download."""
        return MountFingerprinter.remote_listing_sync(scheme, remote_path, storage_options)[0]

    @staticmethod
    def remote_listing_sync(
        scheme: str,
        remote_path: str,
        storage_options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, dict[str, Any]] | None]:
        """A remote path's fingerprint and the listing it came from, ``None`` if unlisted."""
        try:
            protocol = _scheme_to_fsspec_protocol(scheme)
            fs = _mount_filesystem(protocol, storage_options or {})
            listing = _list_remote_file_info(fs, protocol, remote_path)
            return _fingerprint_listing(listing), listing

        except ImportError as exc:
            # fsspec or the backend driver is missing. A deterministic hash here would
            # cache-hit across environments on stale content; return a unique one instead.
            logger.warning(
                "fsspec missing while fingerprinting %s://%s (%s); "
                "treating mount as non-cacheable for this run",
                scheme,
                remote_path,
                exc,
            )
            return nonce.unresolved_nonce(), None
        except Exception as e:
            logger.warning(
                "Failed to fingerprint remote mount %s://%s: %s",
                scheme,
                remote_path,
                e,
            )
            # Unique per call: forces re-execution.
            return nonce.unresolved_nonce(), None

    @staticmethod
    async def fingerprint_remote(
        scheme: str,
        remote_path: str,
        storage_options: dict[str, Any] | None = None,
    ) -> str:
        """Async wrapper for remote mount fingerprinting."""
        return MountFingerprinter.fingerprint_remote_sync(
            scheme,
            remote_path,
            storage_options,
        )

    @staticmethod
    def fingerprint_mount_sync(
        mount: MountSpec,
        storage_options: dict[str, Any] | None = None,
        *,
        digests_path: Path | None = None,
        read_bytes: int | None = None,
        on_progress: HashProgress | None = None,
    ) -> str | None:
        """Compute the fingerprint for any mount spec.

        ``None`` for read-write mounts: the caller must skip the cache check.
        Pinned mounts hash the pin value. ``storage_options`` should be the
        merged options the data fetch uses, so the listing hits the same
        endpoint; when ``None``, ``mount.options`` is used. A local mount's file
        digests are cached at *digests_path*; past *read_bytes* of files to read, its
        fingerprint is :func:`nonce.unhashed_nonce`'s, and the session hashes it.
        """
        if mount.mode == MountMode.READ_WRITE:
            return None

        if mount.pin is not None:
            return hashlib.sha256(f"pin:{mount.pin}".encode()).hexdigest()

        scheme, path = parse_mount_uri(mount.uri)
        if scheme == "file":
            if digests_path is None:
                raise ValueError(f"mount {mount.name}: a local mount needs a digest cache")
            fingerprint = MountFingerprinter.fingerprint_local_sync(
                Path(path), digests_path, read_bytes=read_bytes, on_progress=on_progress
            )
            # Not read yet, which is not unavailable: a background job reads it.
            return nonce.unhashed_nonce() if fingerprint is None else fingerprint
        else:
            effective = storage_options if storage_options is not None else (mount.options or None)
            return MountFingerprinter.fingerprint_remote_sync(
                scheme, path, storage_options=effective
            )


def local_files(root: Path) -> list[tuple[str, Path, os.stat_result]]:
    """``(relative path, path, stat)`` of each regular file under *root*, or *root* itself.

    A link to a file is a file, as in a Hugging Face cache; a dangling one is skipped,
    and links to directories are not followed.
    """
    if not root.is_dir():
        return [(root.name, root, root.stat())]
    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                st = path.stat()
            except OSError:
                continue  # a dangling link, or gone since the walk listed it
            if stat.S_ISREG(st.st_mode):
                files.append((path.relative_to(root).as_posix(), path, st))
    return sorted(files)


class _DigestCache:
    def __init__(self, path: Path) -> None:
        self.digests = FileDigests(path)
        self.users = 0


_digest_caches: OrderedDict[Path, _DigestCache] = OrderedDict()
_digest_caches_lock = threading.Lock()


@contextmanager
def digest_cache(path: Path) -> Generator[FileDigests]:
    """The process's one :class:`FileDigests` over *path*, opened once: one that fell back
    to memory then still remembers between looks. Past :data:`_OPEN_DIGEST_CACHES`, the
    least recently used one nothing is using is closed."""
    key = path.absolute()
    with _digest_caches_lock:
        entry = _digest_caches.pop(key, None) or _DigestCache(key)
        _digest_caches[key] = entry
        entry.users += 1
    try:
        yield entry.digests
    finally:
        with _digest_caches_lock:
            entry.users -= 1
            idle = [k for k, e in _digest_caches.items() if e.users == 0]
            for k in idle[: max(0, len(_digest_caches) - _OPEN_DIGEST_CACHES)]:
                _digest_caches.pop(k).digests.close()


def close_digest_caches() -> None:
    """Close every digest cache nothing is using; the test suite does after each test."""
    with _digest_caches_lock:
        for k in [k for k, e in _digest_caches.items() if e.users == 0]:
            _digest_caches.pop(k).digests.close()


_hashing_locks: dict[tuple[Path, Path], threading.Lock] = {}


def _hashing(digests_path: Path, root: Path) -> threading.Lock:
    key = (digests_path.absolute(), root.absolute())
    with _digest_caches_lock:
        return _hashing_locks.setdefault(key, threading.Lock())


def local_digests(
    root: Path,
    digests_path: Path,
    *,
    read_bytes: int | None = None,
    on_progress: HashProgress | None = None,
) -> dict[str, str] | None:
    """Each file's sha256 under the local mount *root*, by relative POSIX path.

    A file the cache at *digests_path* holds a digest for under its current stat is not
    read. With *read_bytes*, ``None`` instead of reading more than that of the files that
    can be read. The rest is :func:`file_digests`, which reports to *on_progress*.
    """
    if read_bytes is None:
        # A run and the background job over one mount: the second finds the first's digests.
        with _hashing(digests_path, root):
            return _local_digests(root, digests_path, read_bytes=None, on_progress=on_progress)
    return _local_digests(root, digests_path, read_bytes=read_bytes, on_progress=on_progress)


def hash_local_mount(
    root: Path, digests_path: Path, *, on_progress: HashProgress | None = None
) -> None:
    """Hash *root*'s files into the cache at *digests_path* until a staleness check can key
    it, at most :data:`_HASH_ATTEMPTS` times: a file written under 2 s before it was read
    is not remembered, so it is read again once it has settled. *on_progress* hears the
    tries as one hash, finished once."""
    shown: tuple[int, int] | None = None

    def report(bytes_done: int, total: int, finished: bool) -> None:
        nonlocal shown
        shown = (bytes_done, total)
        if not finished and on_progress is not None:
            on_progress(bytes_done, total, False)

    try:
        for attempt in range(_HASH_ATTEMPTS):
            local_digests(root, digests_path, on_progress=report)
            last = attempt == _HASH_ATTEMPTS - 1
            if last or (
                local_digests(root, digests_path, read_bytes=STALENESS_READ_BYTES) is not None
            ):
                return
            time.sleep(_SETTLE_SECONDS)
    finally:
        if on_progress is not None and shown is not None:
            on_progress(*shown, True)


def _local_digests(
    root: Path, digests_path: Path, *, read_bytes: int | None, on_progress: HashProgress | None
) -> dict[str, str] | None:
    files = without_digest_cache(local_files(root), digests_path)
    with digest_cache(digests_path) as cache:
        try:
            known = cache.cached_many(path for _, path, _ in files)
        except OSError:
            # A file went between the walk and the look.
            if read_bytes is not None:
                return None
            return file_digests(cache, files, on_progress=on_progress)
        digests = {rel: sha for (rel, _, _), sha in zip(files, known) if sha is not None}
        missing = [f for f, sha in zip(files, known) if sha is None]
        unread = sum(st.st_size for _, path, st in missing if os.access(path, os.R_OK))
        if read_bytes is not None and unread > read_bytes:
            return None
        if unread >= _LOG_READ_BYTES:
            logger.info("Hashing %s: %d files, %d bytes", root, len(missing), unread)
        return digests | file_digests(cache, missing, on_progress=on_progress)


def file_digests(
    cache: FileDigests,
    files: list[tuple[str, Path, os.stat_result]],
    *,
    on_progress: HashProgress | None = None,
) -> dict[str, str]:
    """*files*' sha256 by relative path, read where *cache* holds none.

    A file that cannot be read is ``unreadable``, so its mount's key changes once it can
    be; one that changed while it was read gets a key no run reproduces; one gone since
    the walk is left out. Past :data:`_PROGRESS_BATCH_BYTES`, the files are read in
    batches of about that, and of no fewer files than read threads, each reported to
    *on_progress*.
    """
    readable = [f for f in files if os.access(f[1], os.R_OK)]
    digests = {rel: "unreadable" for rel, path, _ in files if not os.access(path, os.R_OK)}
    total = sum(st.st_size for _, _, st in readable)
    if on_progress is None or total <= _PROGRESS_BATCH_BYTES:
        return digests | _read_batch(cache, readable)
    done = 0
    on_progress(0, total, False)
    try:
        for batch in _batches(readable):
            digests |= _read_batch(cache, batch)
            done += sum(st.st_size for _, _, st in batch)
            if done < total:
                on_progress(done, total, False)
    finally:
        on_progress(done, total, True)
    return digests


def _batches(
    files: list[tuple[str, Path, os.stat_result]],
) -> Generator[list[tuple[str, Path, os.stat_result]]]:
    batch: list[tuple[str, Path, os.stat_result]] = []
    size = 0
    for file in files:
        batch.append(file)
        size += file[2].st_size
        if size >= _PROGRESS_BATCH_BYTES and len(batch) >= _READ_THREADS:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def _read_batch(
    cache: FileDigests, files: list[tuple[str, Path, os.stat_result]]
) -> dict[str, str]:
    try:
        found = cache.digest_many([path for _, path, _ in files], stats=[st for _, _, st in files])
    except OSError:
        # Gone, or unreadable after all: the rest one by one, those read so far remembered.
        return _digests_one_by_one(cache, files)
    return {
        rel: d.sha256 if d.settled else nonce.unread_nonce() for (rel, _, _), d in zip(files, found)
    }


def _digests_one_by_one(
    cache: FileDigests, files: list[tuple[str, Path, os.stat_result]]
) -> dict[str, str]:
    digests: dict[str, str] = {}
    for rel, path, _ in files:
        try:
            found = cache.digest(path)
        except FileNotFoundError:
            continue  # gone since the walk
        except OSError:
            digests[rel] = "unreadable"
        else:
            digests[rel] = found.sha256 if found.settled else nonce.unread_nonce()
    return digests


def moved_key(provenance_hash: str) -> str:
    """What a run whose input moved is recorded under: a key no check makes, so its cell
    stays stale until a run that reads still files."""
    return derive_subkey(provenance_hash, "moved")


def moved_parts(key_parts: dict[str, Any] | None, moved: str) -> dict[str, Any] | None:
    """*key_parts* with the part of the mount *moved* is in marked as changed, so the stale
    cell's explanation names that mount even once the file is put back."""
    if key_parts is None or "inputs" not in key_parts:
        return key_parts
    mount = moved.partition("/")[0]
    inputs = [
        [kind, name, f"{component}:moved" if (kind, name) == ("mount", mount) else component]
        for kind, name, component in key_parts["inputs"]
    ]
    return {**key_parts, "inputs": inputs}


def moved_line(moved: str) -> str:
    """The console line that says *moved* changed after the run began: while the cell ran,
    or, in Run All, before it ran but after the batch was keyed."""
    return f"{moved} changed after the run began; its outputs are kept for this run only.\n"


def moved_file(prepared: Mapping[str, ResolvedMount], *, skip: Collection[str] = ()) -> str | None:
    """A file of a local read-only mount in *prepared*, but those in *skip*, whose content
    is not what the run was keyed on (``mount/path``), or ``None`` when none moved.

    Read after a run: the cell reads the live folder, so a file written meanwhile gives
    outputs the key does not describe. Only its content is compared, so a file written and
    put back before this check is not seen.
    """
    for name, rm in sorted(prepared.items()):
        if name in skip or rm.file_versions is None or rm.digests_path is None:
            continue  # remote, pinned or read-write: nothing local was keyed by content
        try:
            # Mostly remembered digests, so without waiting on a background hash of the mount.
            now = local_digests(rm.local_path, rm.digests_path, read_bytes=STALENESS_READ_BYTES)
            if now is None:
                now = local_digests(rm.local_path, rm.digests_path) or {}
        except OSError:
            return name  # the folder went
        moved = sorted(
            rel
            for rel in now.keys() | rm.file_versions.keys()
            if now.get(rel) != rm.file_versions.get(rel)
        )
        if moved:
            return f"{name}/{moved[0]}"
    return None


def without_digest_cache(
    files: list[tuple[str, Path, os.stat_result]], digests_path: Path
) -> list[tuple[str, Path, os.stat_result]]:
    """*files* without the digest cache's own, which each look rewrites: a mount over the
    notebook's folder would otherwise never key the same twice."""
    own = {digests_path.absolute(), Path(f"{digests_path.absolute()}-journal")}
    return [f for f in files if f[1].absolute() not in own]


def _fingerprint_digests(digests: Mapping[str, str]) -> str:
    content = "\n".join(f"{rel}:{sha}" for rel, sha in sorted(digests.items()))
    return hashlib.sha256(content.encode()).hexdigest()


# --- Helpers ---


def _options_digest(storage_options: dict[str, Any]) -> str:
    """What keys a view and its objects to the options that reached the store.

    Neither is handed to a caller that reaches the store another way, so a credential that
    may list a prefix but not read it never gets another notebook's bytes. A rotated secret
    starts a fresh copy.
    """
    return hashlib.sha256(
        json.dumps(storage_options, sort_keys=True, default=str).encode()
    ).hexdigest()


def glob_matches(relative: str, glob: str) -> bool:
    """Whether the file at *relative* in a mount is one an ``@each`` *glob* names."""
    return fnmatch.fnmatch(relative, glob)


def _view_key(
    protocol: str, remote_path: str, options_digest: str, fingerprint: str, glob: str | None = None
) -> str:
    parts = [protocol, remote_path, options_digest, fingerprint]
    if glob is not None:
        parts.append(glob)
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _fingerprint_listing(listing: dict[str, dict[str, Any]]) -> str:
    """A remote prefix's fingerprint from its listing: names, sizes, etags and mtimes."""
    parts: list[str] = []
    for info in listing.values():
        name = info.get("name", "")
        size = info.get("size", 0)
        etag = info.get("ETag", info.get("etag", ""))
        mtime = info.get("LastModified", info.get("mtime", ""))
        parts.append(f"{name}:{size}:{etag}:{mtime}")
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()


def _check_relative(rel: Path, mount_name: str, remote_name: str) -> None:
    """Raise if a remote name's path under the mount would leave it.

    Lexical: the view is a fresh directory holding no links, so ``..``, a root or a
    drive are the only ways out.
    """
    if rel.anchor or ".." in rel.parts or not rel.parts:
        raise RuntimeError(
            f"Mount '{mount_name}' rejected: remote name {remote_name!r} is outside the "
            f"mount root. This usually means the remote returned a path-traversal name "
            f"(.. segments or an absolute path)."
        )


def _scheme_to_fsspec_protocol(scheme: str) -> str:
    """Map our URI scheme to fsspec protocol string."""
    return {
        "s3": "s3",
        "gs": "gcs",
        "az": "abfs",
    }.get(scheme, scheme)


def _mount_filesystem(protocol: str, storage_options: dict[str, Any]) -> Any:
    """``fsspec.filesystem(protocol, **storage_options)``, but not gcsfs's gRPC probe.

    Since 2026.6 gcsfs asks Google's Storage Control API (gRPC) about a bucket
    before first use; a non-Google endpoint (fake-gcs-server, a GCS-compatible
    service) cannot answer, and gcsfs retries for about two minutes per bucket.
    Such endpoints get the standard filesystem; ``*.googleapis.com`` keeps the default.
    """
    import fsspec

    storage_options = _keyless(protocol, storage_options)
    if protocol in ("gcs", "gs"):
        endpoint = str(
            storage_options.get("endpoint_url") or os.environ.get("STORAGE_EMULATOR_HOST") or ""
        )
        host = urlparse(endpoint if "://" in endpoint else f"//{endpoint}").hostname or ""
        if endpoint and not (host == "googleapis.com" or host.endswith(".googleapis.com")):
            from gcsfs.core import GCSFileSystem

            return GCSFileSystem(**storage_options)
    return fsspec.filesystem(protocol, **storage_options)


def _keyless(protocol: str, storage_options: dict[str, Any]) -> dict[str, Any]:
    """*storage_options* with a federated token file made into the credential fsspec takes.

    gcsfs cannot load a workload-identity (``external_account``) file, and adlfs has no
    field for a federated token file. Both credentials re-read the file at each refresh,
    so a rewritten token is used without a restart.
    """
    if protocol in ("gcs", "gs"):
        token = storage_options.get("token")
        info = _external_account(token) if isinstance(token, str) else None
        if info is not None:
            from google.auth import identity_pool

            access = storage_options.get("access", "full_control")
            scope = f"https://www.googleapis.com/auth/devstorage.{access}"
            credentials = identity_pool.Credentials.from_info(info, scopes=[scope])
            return {**storage_options, "token": credentials}
    elif protocol in ("abfs", "az") and "federated_token_file" in storage_options:
        from azure.identity.aio import ClientAssertionCredential

        options = dict(storage_options)
        token_file = Path(options.pop("federated_token_file"))
        tenant_id, client_id = options.pop("tenant_id", None), options.pop("client_id", None)
        if not tenant_id or not client_id:
            raise ValueError("an az mount's federated_token_file needs tenant_id and client_id")
        # WorkloadIdentityCredential's exchange, minus its ten minutes of caching the file.
        credential = ClientAssertionCredential(
            tenant_id, client_id, lambda: read_credential_file(token_file)
        )
        return {**options, "credential": credential}
    return storage_options


def _external_account(token: str) -> dict[str, Any] | None:
    """The workload-identity configuration in the file *token* names, if it is one.

    Raises:
        CredentialError: *token* names a file that is not regular, or is too large.
    """
    try:
        info = json.loads(read_credential_file(Path(token)))
    except CredentialError:
        raise
    except (OSError, ValueError):
        # A raw token, which gcsfs takes as it is.
        return None
    return info if isinstance(info, dict) and info.get("type") == "external_account" else None


def _assert_within(candidate: Path, root: Path, mount_name: str, remote_name: str) -> None:
    """Raise if ``candidate`` would resolve outside ``root``.

    ``root`` must already be resolved; ``candidate`` is resolved non-strictly
    since it may not exist yet. A remote name with ``..``, an absolute path, or
    an escaping symlink rejects the mount rather than widening what the harness reads.
    """
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise RuntimeError(
            f"Mount '{mount_name}' failed to resolve local path for "
            f"remote name {remote_name!r}: {exc}"
        ) from exc
    if not resolved.is_relative_to(root):
        raise RuntimeError(
            f"Mount '{mount_name}' rejected: remote name {remote_name!r} "
            f"resolves to {resolved}, which is outside the mount root {root}. "
            f"This usually means the remote returned a path-traversal name "
            f"(.. segments, absolute path, or a symlink pointing outside)."
        )


def _list_remote_file_info(
    fs: Any,
    protocol: str,
    remote_path: str,
) -> dict[str, dict[str, Any]]:
    """List remote files recursively with metadata."""
    remote_uri = f"{protocol}://{remote_path}"

    is_file = False
    if hasattr(fs, "isfile"):
        is_file = bool(fs.isfile(remote_uri))
    if is_file:
        info = dict(fs.info(remote_uri))
        name = _strip_protocol(str(info.get("name", remote_uri)), protocol)
        info["name"] = name
        return {name: info}

    listing = fs.find(remote_uri, withdirs=False, detail=True)
    if isinstance(listing, list):
        return {
            _strip_protocol(name, protocol): {"name": _strip_protocol(name, protocol)}
            for name in listing
        }

    normalized: dict[str, dict[str, Any]] = {}
    for name, info in listing.items():
        stripped = _strip_protocol(name, protocol)
        normalized_info = dict(info)
        normalized_info["name"] = _strip_protocol(
            str(info.get("name", stripped)),
            protocol,
        )
        normalized[stripped] = normalized_info
    return normalized


def _list_remote_files(fs: Any, protocol: str, remote_path: str) -> list[str]:
    """Return remote file names recursively."""
    return sorted(_list_remote_file_info(fs, protocol, remote_path))


def _strip_protocol(uri: str, protocol: str) -> str:
    """Remove a scheme prefix from a remote path if present."""
    return uri.removeprefix(f"{protocol}://")


def _relative_remote_path(
    remote_name: str,
    protocol: str,
    remote_path: str,
) -> Path:
    """Convert an absolute remote file name into a path under the mount root."""
    normalized = _strip_protocol(remote_name, protocol)
    base = remote_path.rstrip("/")
    if normalized == base:
        return Path(Path(normalized).name)
    prefix = f"{base}/"
    if normalized.startswith(prefix):
        return Path(normalized[len(prefix) :])
    return Path(Path(normalized).name)


def resolve_cell_mounts(
    notebook_mounts: list[MountSpec],
    cell_mounts: list[MountSpec],
    annotation_mounts: list[MountSpec],
) -> list[MountSpec]:
    """Merge notebook, cell-meta and annotation mounts, deduplicated by name.

    Precedence (highest first): ``# @mount`` annotations, ``[[cells.mounts]]``,
    then notebook-level ``[[mounts]]``.
    """
    merged: dict[str, MountSpec] = {}

    for m in notebook_mounts:
        merged[m.name] = m
    for m in cell_mounts:
        merged[m.name] = m
    for m in annotation_mounts:
        merged[m.name] = m

    return list(merged.values())


def ensure_writable(credentials: CredentialResolver, mounts: list[MountSpec]) -> None:
    """Refuse a run whose read-write mounts use a credential its principal may not write.

    Called before anything is staged or sent, so the cell is refused as a whole.
    """
    for mount in mounts:
        if mount.mode == MountMode.READ_WRITE:
            credentials.ensure_writable(parse_mount_uri(mount.uri)[0], mount.credential)


@dataclass(frozen=True)
class MountWeight:
    """A remote read-only mount's bytes, and how many are already where the cell runs.

    ``None`` is unknown: a pinned mount is not listed, and a share needs the cache.
    *listing* is kept when the share is unknown, so the machine can be asked for it.
    """

    bytes: int | None
    cached_bytes: int | None
    listing: dict[str, dict[str, Any]] | None = field(default=None, compare=False, repr=False)


def mount_fingerprint_sync(
    resolver: MountResolver, mount: MountSpec, *, read_bytes: int | None = None
) -> str | None:
    """The provenance component for one mount, shared by execution and staleness.

    Both must produce the same string, or artifacts are keyed under a hash
    staleness never reproduces. It includes the uri (two directories with the
    same files are different inputs) and the credential's name, never its values.
    An unresolvable credential gives a unique fingerprint: the cell shows stale,
    runs, and fails naming the credential. A local mount with more than *read_bytes* of
    files to read, which staleness passes, gets :func:`nonce.unhashed_nonce`'s, and the
    session hashes it in the background (:func:`hash_local_mount`).
    """
    return weigh_mount_sync(resolver, mount, None, read_bytes=read_bytes)[0]


def weigh_mount_sync(
    resolver: MountResolver,
    mount: MountSpec,
    cache: MountCache | None,
    glob: str | None = None,
    *,
    read_bytes: int | None = None,
    on_progress: HashProgress | None = None,
) -> tuple[str | None, MountWeight | None, str | None]:
    """:func:`mount_fingerprint_sync`'s component, with the mount's weight from the same
    listing: its share already in *cache*, the one the cell runs against.

    *glob* is the files a run reads of the mount (``@each``). It narrows the weight and
    the third, not the component, which is over everything listed.

    The weight is ``None`` for a mount that moves nothing to read (``file://``) or that a
    run does not read through a view (``rw``). The third is what a yes to a large read
    is kept for (``read_confirm``): the component, without the per-call nonce an
    unresolvable credential or a failed listing gets. *on_progress* hears a local mount's
    hash, when there is one to make (:data:`HashProgress`).
    """
    try:
        storage_options = resolver.storage_options(mount)
    except CredentialError as exc:
        logger.warning("mount %s: %s", mount.name, exc)
        unresolved = f"{mount.name}:unresolved:{nonce.unresolved_nonce()}"
        return unresolved, None, f"{mount.name}:{mount.uri}:unresolved"
    scheme, path = parse_mount_uri(mount.uri)
    weight: MountWeight | None = None
    unlisted = False
    if scheme == "file" or mount.mode == MountMode.READ_WRITE or mount.pin is not None:
        fingerprint = MountFingerprinter.fingerprint_mount_sync(
            mount,
            storage_options=storage_options or None,
            digests_path=resolver.digests_path,
            read_bytes=read_bytes,
            on_progress=on_progress,
        )
        if mount.pin is not None and scheme != "file":
            weight = MountWeight(bytes=None, cached_bytes=None)
    else:
        fingerprint, listing = MountFingerprinter.remote_listing_sync(
            scheme, path, storage_options or None
        )
        weight = _weigh_listing(scheme, path, storage_options, fingerprint, listing, cache, glob)
        unlisted = listing is None
    if fingerprint is None:
        return None, weight, None
    component = mount_component(mount, fingerprint)
    approval = f"{_component_prefix(mount)}:unlisted" if unlisted else component
    # A yes to some of a mount is not one to the rest of it.
    return component, weight, approval if glob is None else f"{approval}:{glob}"


def mount_component(mount: MountSpec, fingerprint: str) -> str:
    """A mount's part of a cell's provenance, for the fingerprint of what it lists."""
    return f"{_component_prefix(mount)}:{fingerprint}"


def _component_prefix(mount: MountSpec) -> str:
    identity = credential_identity(mount.credential)
    return f"{mount.name}:{mount.uri}:{identity}" if identity else f"{mount.name}:{mount.uri}"


def _weigh_listing(
    scheme: str,
    path: str,
    storage_options: dict[str, Any],
    fingerprint: str,
    listing: dict[str, dict[str, Any]] | None,
    cache: MountCache | None,
    glob: str | None = None,
) -> MountWeight:
    if listing is None:
        return MountWeight(bytes=None, cached_bytes=None)
    protocol = _scheme_to_fsspec_protocol(scheme)
    if glob is not None:
        listing = {
            name: info
            for name, info in listing.items()
            if glob_matches(_relative_remote_path(name, protocol, path).as_posix(), glob)
        }
    sizes = {name: int(info.get("size") or 0) for name, info in listing.items()}
    if cache is None:
        return MountWeight(bytes=sum(sizes.values()), cached_bytes=None, listing=listing)
    digest = _options_digest(storage_options)
    objects = [
        (object_key(protocol, digest, name, info), sizes[name]) for name, info in listing.items()
    ]
    return MountWeight(
        bytes=sum(sizes.values()),
        cached_bytes=cache.cached_bytes(
            _view_key(protocol, path, digest, fingerprint, glob), objects
        ),
    )


def listing_entries(listing: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """A listing as JSON, keeping what its fingerprint and object keys read.

    A missing size stays missing: the two read it with different defaults.
    """
    return [
        {
            "key": key,
            "name": info.get("name", ""),
            "etag": str(info.get("ETag", info.get("etag", ""))),
            "mtime": str(info.get("LastModified", info.get("mtime", ""))),
            **({"size": info["size"]} if "size" in info else {}),
        }
        for key, info in listing.items()
    ]


def cached_share(
    mount: MountSpec,
    storage_options: dict[str, Any],
    entries: list[dict[str, Any]],
    cache: MountCache,
) -> int | None:
    """How many bytes of a listing a machine's *cache* holds, for a server that asks.

    The object keys hash the storage options, so only the machine reading the mount
    can compute them. ``None`` for a mount it would not read through a view.
    """
    scheme, path = parse_mount_uri(mount.uri)
    if scheme == "file" or mount.mode == MountMode.READ_WRITE:
        return None
    listing = {
        entry["key"]: {k: entry[k] for k in ("name", "size", "etag", "mtime") if k in entry}
        for entry in entries
    }
    fingerprint = _fingerprint_listing(listing)
    return _weigh_listing(scheme, path, storage_options, fingerprint, listing, cache).cached_bytes
