"""File mount resolution and fingerprinting for notebook cells.

This module handles the full lifecycle of filesystem mounts:

1. **Resolution** — Converting mount URIs to local paths before cell execution.
   Local mounts (``file://``) are validated and used directly.  Remote mounts
   (``s3://``, ``gs://``, ``az://``) are materialised to a local cache directory
   via ``fsspec`` when available, or via PyArrow filesystems as a fallback.

2. **Fingerprinting** — Computing content hashes for read-only mounts so they
   participate in provenance-based caching.  A cell reading from
   ``s3://bucket/data`` invalidates when the data changes.

3. **Sync-back** — After cell execution, read-write mounts are synced back to
   their remote URIs.

Design decisions:

- The **harness** only sees local ``pathlib.Path`` objects — all remote
  resolution happens in the executor before the subprocess spawns.
- Read-write mounts are tracked as side-effect declarations in artifact
  metadata but don't participate in provenance hashing (you can't hash
  the output before computing it).
- ``fsspec`` is an optional dependency.  When unavailable, only ``file://``
  mounts work and remote mounts raise a clear error.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from strata.notebook.models import MountMode, MountSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolved mount — the result of preparing a mount for execution
# ---------------------------------------------------------------------------


@dataclass
class ResolvedMount:
    """A mount that has been resolved to a local path."""

    spec: MountSpec
    local_path: Path
    fingerprint: str | None  # None for RW mounts → signals non-cacheable cell
    # For remote RW mounts: the staging dir that needs sync-back
    staging_dir: Path | None = None


# ---------------------------------------------------------------------------
# URI parsing helpers
# ---------------------------------------------------------------------------


def parse_mount_uri(uri: str) -> tuple[str, str]:
    """Parse a mount URI into (scheme, path).

    Examples::

        "file:///home/user/data"   → ("file", "/home/user/data")
        "s3://bucket/prefix"       → ("s3", "bucket/prefix")
        "gs://bucket/prefix"       → ("gs", "bucket/prefix")
        "az://container/prefix"    → ("az", "container/prefix")

    Returns:
        Tuple of (scheme, path).

    Raises:
        ValueError: If the URI is malformed or uses an unsupported scheme.
    """
    supported = {"file", "s3", "gs", "gcs", "az", "azure"}

    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if not scheme:
        # Bare path — treat as local file
        return "file", uri

    if scheme not in supported:
        raise ValueError(
            f"Unsupported mount URI scheme '{scheme}' in '{uri}'. "
            f"Supported: {', '.join(sorted(supported))}"
        )

    # Normalise aliases
    if scheme == "gcs":
        scheme = "gs"
    elif scheme == "azure":
        scheme = "az"

    if scheme == "file":
        # file:///path → /path
        return "file", parsed.path
    else:
        # s3://bucket/prefix → bucket/prefix
        path = parsed.netloc
        if parsed.path and parsed.path != "/":
            path += parsed.path
        return scheme, path


def _is_remote(scheme: str) -> bool:
    return scheme != "file"


# ---------------------------------------------------------------------------
# MountResolver
# ---------------------------------------------------------------------------


class MountResolver:
    """Resolves mount URIs to local paths for cell execution.

    For local mounts, validates the path exists and returns it directly.
    For remote mounts, uses fsspec (if available) to create a cached
    local mirror, or raises an error if fsspec is not installed.

    Args:
        cache_dir: Base directory for caching remote mounts.
        credentials: Optional cloud credentials (passed to fsspec).
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        credentials: dict[str, Any] | None = None,
    ):
        # Default to a user-scoped dir (matches ~/.strata/artifacts) instead
        # of the world-writable /tmp/strata_mounts that earlier versions
        # used — that path was shared across all users on the host and
        # invited cross-user cache poisoning. Real callers (executor,
        # remote_executor, tests) always pass an explicit cache_dir; the
        # default just keeps ad-hoc REPL use from blowing up.
        self.cache_dir = cache_dir or Path.home() / ".strata" / "mount_cache"
        self.credentials = credentials or {}
        self._fsspec_available: bool | None = None

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
    ) -> dict[str, ResolvedMount]:
        """Resolve all mounts to local paths.

        Args:
            mounts: Mount specifications to resolve.

        Returns:
            Dict of {mount_name: ResolvedMount}.

        Raises:
            ValueError: If a local mount path doesn't exist.
            ImportError: If a remote mount is requested but fsspec is unavailable.
        """
        resolved: dict[str, ResolvedMount] = {}

        for mount in mounts:
            scheme, path = parse_mount_uri(mount.uri)

            if scheme == "file":
                resolved[mount.name] = await self._resolve_local(mount, path)
            else:
                resolved[mount.name] = await self._resolve_remote(
                    mount,
                    scheme,
                    path,
                )

        return resolved

    async def _resolve_local(
        self,
        mount: MountSpec,
        local_path: str,
    ) -> ResolvedMount:
        """Resolve a local file:// mount."""
        p = Path(local_path)

        if mount.mode == MountMode.READ_WRITE:
            # RW local: ensure directory exists, no fingerprint (side effect)
            p.mkdir(parents=True, exist_ok=True)
            return ResolvedMount(spec=mount, local_path=p, fingerprint=None)

        if not p.exists():
            raise ValueError(f"Local mount '{mount.name}' path does not exist: {p}")

        fingerprint = await MountFingerprinter.fingerprint_mount(mount)
        assert fingerprint is not None
        return ResolvedMount(spec=mount, local_path=p, fingerprint=fingerprint)

    async def _resolve_remote(
        self,
        mount: MountSpec,
        scheme: str,
        remote_path: str,
    ) -> ResolvedMount:
        """Resolve a remote mount (S3, GCS, Azure) via fsspec."""
        if not self._check_fsspec():
            raise ImportError(
                f"Remote mount '{mount.name}' requires fsspec. "
                f"Install it with: pip install fsspec s3fs gcsfs adlfs"
            )

        # Build the cache directory for this mount
        mount_hash = hashlib.sha256(mount.uri.encode()).hexdigest()[:12]
        local_dir = self.cache_dir / f"{mount.name}_{mount_hash}"

        if mount.mode == MountMode.READ_ONLY:
            # Use fsspec's filecache for read-only mounts
            return await self._resolve_remote_ro(
                mount,
                scheme,
                remote_path,
                local_dir,
            )
        else:
            # RW: stage locally, sync back after execution
            return await self._resolve_remote_rw(
                mount,
                scheme,
                remote_path,
                local_dir,
            )

    async def _resolve_remote_ro(
        self,
        mount: MountSpec,
        scheme: str,
        remote_path: str,
        local_dir: Path,
    ) -> ResolvedMount:
        """Resolve a read-only remote mount recursively."""
        import fsspec

        protocol = _scheme_to_fsspec_protocol(scheme)
        storage_options = {**self.credentials.get(scheme, {}), **mount.options}
        fingerprint = await MountFingerprinter.fingerprint_mount(mount)
        assert fingerprint is not None

        fs = fsspec.filesystem(protocol, **storage_options)
        snapshot_dir = local_dir / fingerprint[:12]
        local_mirror = snapshot_dir / "data"
        complete_marker = snapshot_dir / ".complete"

        if not complete_marker.exists():
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            local_mirror.mkdir(parents=True, exist_ok=True)
            mirror_root = local_mirror.resolve()
            try:
                for remote_name in _list_remote_files(fs, protocol, remote_path):
                    rel = _relative_remote_path(remote_name, protocol, remote_path)
                    local_file = local_mirror / rel
                    # Path-traversal guard: a remote backend (or
                    # attacker-controlled bucket) can return a name
                    # containing ``..`` segments or an absolute path,
                    # which would escape local_mirror once joined. The
                    # harness only sees pathlib.Path objects rooted
                    # under local_mirror, so an escape silently widens
                    # what the cell can read.
                    _assert_within(local_file, mirror_root, mount.name, remote_name)
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    fs.get(f"{protocol}://{remote_name}", str(local_file))
                complete_marker.write_text("", encoding="utf-8")
            except Exception as e:
                shutil.rmtree(snapshot_dir, ignore_errors=True)
                raise RuntimeError(
                    f"Failed to materialize remote mount '{mount.name}' from {mount.uri}: {e}"
                ) from e

        return ResolvedMount(
            spec=mount,
            local_path=local_mirror,
            fingerprint=fingerprint,
        )

    async def _resolve_remote_rw(
        self,
        mount: MountSpec,
        scheme: str,
        remote_path: str,
        local_dir: Path,
    ) -> ResolvedMount:
        """Resolve a read-write remote mount with staging directory."""
        # Stage: download current contents to local staging dir
        staging = local_dir / "staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        protocol = _scheme_to_fsspec_protocol(scheme)
        storage_options = {**self.credentials.get(scheme, {}), **mount.options}

        try:
            import fsspec

            fs = fsspec.filesystem(protocol, **storage_options)
            remote_uri = f"{protocol}://{remote_path}"
            if fs.exists(remote_uri):
                fs.get(remote_uri, str(staging), recursive=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to stage RW mount '{mount.name}' from {mount.uri}: {e}"
            ) from e

        # Defence-in-depth path-traversal scan: fsspec's recursive=True
        # is opaque about the local paths it writes; if the remote
        # backend returned names with ``..`` segments or symlinks
        # pointing outside, files could land outside the staging dir
        # and the harness wouldn't know. Walk after the fact and
        # reject the mount if anything escaped.
        staging_root = staging.resolve()
        for entry in staging.rglob("*"):
            _assert_within(entry, staging_root, mount.name, str(entry))

        return ResolvedMount(
            spec=mount,
            local_path=staging,
            fingerprint=None,  # RW mounts → cell is non-cacheable
            staging_dir=staging,
        )

    async def sync_back(
        self,
        resolved: dict[str, ResolvedMount],
    ) -> None:
        """Sync read-write mounts back to their remote URIs.

        Called after cell execution completes successfully.
        """
        for name, rm in resolved.items():
            if rm.spec.mode != MountMode.READ_WRITE:
                continue
            if rm.staging_dir is None:
                continue

            scheme, remote_path = parse_mount_uri(rm.spec.uri)
            if scheme == "file":
                # Local RW — nothing to sync (writes go directly)
                continue

            if not self._check_fsspec():
                raise ImportError(f"Cannot sync-back RW mount '{name}': fsspec not available")

            import fsspec

            protocol = _scheme_to_fsspec_protocol(scheme)
            storage_options = {**self.credentials.get(scheme, {}), **rm.spec.options}

            try:
                fs = fsspec.filesystem(protocol, **storage_options)
                remote_uri = f"{protocol}://{remote_path}"
                fs.put(str(rm.staging_dir), remote_uri, recursive=True)
                logger.info(
                    "Synced RW mount '%s' back to %s",
                    name,
                    rm.spec.uri,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to sync-back RW mount '{name}' to {rm.spec.uri}: {e}"
                ) from e


# ---------------------------------------------------------------------------
# MountFingerprinter
# ---------------------------------------------------------------------------


class MountFingerprinter:
    """Compute content fingerprints for mount URIs.

    Fingerprints are used in provenance hashing so that cache entries
    invalidate when mount contents change.
    """

    @staticmethod
    def fingerprint_local_sync(path: Path) -> str:
        """Fingerprint a local directory using file mtimes and sizes.

        This is fast but not cryptographic — suitable for local dev.
        For production, consider content hashing.
        """
        if not path.exists():
            return hashlib.sha256(b"missing").hexdigest()

        if path.is_file():
            stat = path.stat()
            content = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
            return hashlib.sha256(content.encode()).hexdigest()

        # Directory: hash the tree structure
        parts: list[str] = []
        try:
            for root, _dirs, files in os.walk(path):
                for fname in sorted(files):
                    fpath = Path(root) / fname
                    try:
                        stat = fpath.stat()
                        rel = fpath.relative_to(path)
                        parts.append(f"{rel}:{stat.st_size}:{stat.st_mtime_ns}")
                    except OSError:
                        pass
        except OSError:
            pass

        content = "\n".join(sorted(parts))
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    async def fingerprint_local(path: Path) -> str:
        """Async wrapper for local mount fingerprinting."""
        return MountFingerprinter.fingerprint_local_sync(path)

    @staticmethod
    def fingerprint_remote_sync(
        scheme: str,
        remote_path: str,
        storage_options: dict[str, Any] | None = None,
    ) -> str:
        """Fingerprint a remote path using object listing metadata.

        Uses ETags/sizes/mtimes from the remote listing — no data download.
        """
        try:
            import fsspec

            protocol = _scheme_to_fsspec_protocol(scheme)
            fs = fsspec.filesystem(protocol, **(storage_options or {}))
            listing = _list_remote_file_info(fs, protocol, remote_path)
            parts: list[str] = []
            for info in listing.values():
                name = info.get("name", "")
                size = info.get("size", 0)
                etag = info.get("ETag", info.get("etag", ""))
                mtime = info.get("LastModified", info.get("mtime", ""))
                parts.append(f"{name}:{size}:{etag}:{mtime}")

            content = "\n".join(sorted(parts))
            return hashlib.sha256(content.encode()).hexdigest()

        except ImportError as exc:
            # fsspec or the backend-specific driver is missing. The
            # earlier code returned a *deterministic* hash here, which
            # silently produced cache HITs across environments — one
            # machine without fsspec would see the same fingerprint
            # forever even when the remote content changed. Match the
            # generic Exception branch and return a unique-per-call
            # hash so the cell is treated as uncacheable rather than
            # cached on stale content.
            logger.warning(
                "fsspec missing while fingerprinting %s://%s (%s); "
                "treating mount as non-cacheable for this run",
                scheme,
                remote_path,
                exc,
            )
            return hashlib.sha256(os.urandom(32)).hexdigest()
        except Exception as e:
            logger.warning(
                "Failed to fingerprint remote mount %s://%s: %s",
                scheme,
                remote_path,
                e,
            )
            # Return a unique-per-call hash to force re-execution
            return hashlib.sha256(os.urandom(32)).hexdigest()

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
    def fingerprint_mount_sync(mount: MountSpec) -> str | None:
        """Compute fingerprint for any mount spec.

        Returns ``None`` for read-write mounts — the caller must treat
        the cell as non-cacheable (skip cache check entirely).
        Pinned mounts return a hash of the pin value.
        """
        if mount.mode == MountMode.READ_WRITE:
            return None

        if mount.pin is not None:
            return hashlib.sha256(f"pin:{mount.pin}".encode()).hexdigest()

        scheme, path = parse_mount_uri(mount.uri)
        if scheme == "file":
            return MountFingerprinter.fingerprint_local_sync(Path(path))
        else:
            return MountFingerprinter.fingerprint_remote_sync(
                scheme, path, storage_options=mount.options or None
            )

    @staticmethod
    async def fingerprint_mount(mount: MountSpec) -> str | None:
        """Async wrapper for mount fingerprinting."""
        return MountFingerprinter.fingerprint_mount_sync(mount)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scheme_to_fsspec_protocol(scheme: str) -> str:
    """Map our URI scheme to fsspec protocol string."""
    return {
        "s3": "s3",
        "gs": "gcs",
        "az": "abfs",
    }.get(scheme, scheme)


def _assert_within(candidate: Path, root: Path, mount_name: str, remote_name: str) -> None:
    """Raise if ``candidate`` would resolve outside ``root``.

    ``root`` must already be resolved by the caller (so the comparison
    handles symlinks consistently). The candidate is resolved with
    ``strict=False`` because it may not exist yet (we check before
    creation). A remote backend that returns a name containing ``..``
    segments, an absolute path, or a symlink pointing outside the
    mirror gets the mount rejected rather than silently widening what
    the harness can read.
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
    """Merge notebook, cell-meta, and annotation mounts.

    Priority order (highest wins):
    1. Annotation mounts (``# @mount`` in cell source)
    2. Cell-level mounts (``[[cells.mounts]]`` in notebook.toml)
    3. Notebook-level mounts (``[[mounts]]`` in notebook.toml)

    Returns:
        Merged list of MountSpec, deduplicated by name.
    """
    merged: dict[str, MountSpec] = {}

    for m in notebook_mounts:
        merged[m.name] = m
    for m in cell_mounts:
        merged[m.name] = m
    for m in annotation_mounts:
        merged[m.name] = m

    return list(merged.values())
