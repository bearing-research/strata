"""File mount resolution and fingerprinting for notebook cells.

Local (``file://``) mounts are used directly; remote ones (``s3``, ``gs``,
``az``) are mirrored to a local cache via ``fsspec`` before the subprocess
spawns, so the harness only sees ``pathlib.Path``. Read-only mounts are
fingerprinted into provenance; read-write mounts are synced back after
execution and do not participate in provenance.
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

from strata.notebook.credentials import CredentialError, CredentialResolver, credential_identity
from strata.notebook.models import MountMode, MountSpec

logger = logging.getLogger(__name__)


@dataclass
class ResolvedMount:
    """A mount that has been resolved to a local path."""

    spec: MountSpec
    local_path: Path
    fingerprint: str | None  # None for RW mounts: the cell is not cacheable.
    # Remote RW only: synced back after execution.
    staging_dir: Path | None = None


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
        return "file", parsed.path
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

    Local mounts must exist; remote mounts are mirrored under ``cache_dir`` via
    fsspec. ``credentials`` are per-scheme fsspec storage options.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        credentials: MountCredentials | None = None,
        credential_resolver: CredentialResolver | None = None,
    ):
        # User-scoped, not a shared /tmp dir that invites cross-user cache poisoning.
        # Real callers always pass cache_dir.
        self.cache_dir = cache_dir or Path.home() / ".strata" / "mount_cache"
        self.credentials = credentials or {}
        # Empty by default, so a mount naming an undefined credential fails by name
        # instead of reaching the store anonymously.
        self.credential_resolver = credential_resolver or CredentialResolver()
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
    ) -> dict[str, ResolvedMount]:
        """Resolve all mounts to local paths, keyed by mount name.

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
            # RW: no fingerprint (side effect).
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

        mount_hash = hashlib.sha256(mount.uri.encode()).hexdigest()[:12]
        local_dir = self.cache_dir / f"{mount.name}_{mount_hash}"

        if mount.mode == MountMode.READ_ONLY:
            return await self._resolve_remote_ro(
                mount,
                scheme,
                remote_path,
                local_dir,
            )
        else:
            # RW: stage locally, sync back after execution.
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
        protocol = _scheme_to_fsspec_protocol(scheme)
        storage_options = self.storage_options(mount)
        fingerprint = await MountFingerprinter.fingerprint_mount(
            mount, storage_options=storage_options
        )
        assert fingerprint is not None

        fs = _mount_filesystem(protocol, storage_options)
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
                    # Path-traversal guard: a remote name with ``..`` or an absolute path would
                    # escape local_mirror and silently widen what the cell can read.
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
        )

    async def sync_back(
        self,
        resolved: dict[str, ResolvedMount],
    ) -> None:
        """Sync read-write mounts back to their remote URIs after a successful run."""
        for name, rm in resolved.items():
            if rm.spec.mode != MountMode.READ_WRITE:
                continue
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


class MountFingerprinter:
    """Content fingerprints for mount URIs, so cache entries invalidate when contents change."""

    @staticmethod
    def fingerprint_local_sync(path: Path) -> str:
        """Fingerprint a local directory from file mtimes and sizes (fast, not content-based)."""
        if not path.exists():
            return hashlib.sha256(b"missing").hexdigest()

        if path.is_file():
            stat = path.stat()
            content = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
            return hashlib.sha256(content.encode()).hexdigest()

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
        """Fingerprint a remote path from listing metadata (ETags, sizes, mtimes); no download."""
        try:
            protocol = _scheme_to_fsspec_protocol(scheme)
            fs = _mount_filesystem(protocol, storage_options or {})
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
            # fsspec or the backend driver is missing. A deterministic hash here would
            # cache-hit across environments on stale content; return a unique one instead.
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
            # Unique per call: forces re-execution.
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
    def fingerprint_mount_sync(
        mount: MountSpec,
        storage_options: dict[str, Any] | None = None,
    ) -> str | None:
        """Compute the fingerprint for any mount spec.

        ``None`` for read-write mounts: the caller must skip the cache check.
        Pinned mounts hash the pin value. ``storage_options`` should be the
        merged options the data fetch uses, so the listing hits the same
        endpoint; when ``None``, ``mount.options`` is used.
        """
        if mount.mode == MountMode.READ_WRITE:
            return None

        if mount.pin is not None:
            return hashlib.sha256(f"pin:{mount.pin}".encode()).hexdigest()

        scheme, path = parse_mount_uri(mount.uri)
        if scheme == "file":
            return MountFingerprinter.fingerprint_local_sync(Path(path))
        else:
            effective = storage_options if storage_options is not None else (mount.options or None)
            return MountFingerprinter.fingerprint_remote_sync(
                scheme, path, storage_options=effective
            )

    @staticmethod
    async def fingerprint_mount(
        mount: MountSpec,
        storage_options: dict[str, Any] | None = None,
    ) -> str | None:
        """Async wrapper for mount fingerprinting."""
        return MountFingerprinter.fingerprint_mount_sync(mount, storage_options)


# --- Helpers ---


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

    if protocol in ("gcs", "gs"):
        endpoint = str(
            storage_options.get("endpoint_url") or os.environ.get("STORAGE_EMULATOR_HOST") or ""
        )
        host = urlparse(endpoint if "://" in endpoint else f"//{endpoint}").hostname or ""
        if endpoint and not (host == "googleapis.com" or host.endswith(".googleapis.com")):
            from gcsfs.core import GCSFileSystem

            return GCSFileSystem(**storage_options)
    return fsspec.filesystem(protocol, **storage_options)


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


def mount_fingerprint_sync(resolver: MountResolver, mount: MountSpec) -> str | None:
    """The provenance component for one mount, shared by execution and staleness.

    Both must produce the same string, or artifacts are keyed under a hash
    staleness never reproduces. It includes the uri (two directories with the
    same files are different inputs) and the credential's name, never its values.
    An unresolvable credential gives a unique fingerprint: the cell shows stale,
    runs, and fails naming the credential.
    """
    try:
        storage_options = resolver.storage_options(mount)
    except CredentialError as exc:
        logger.warning("mount %s: %s", mount.name, exc)
        return f"{mount.name}:unresolved:{hashlib.sha256(os.urandom(32)).hexdigest()}"
    fingerprint = MountFingerprinter.fingerprint_mount_sync(
        mount, storage_options=storage_options or None
    )
    if fingerprint is None:
        return None
    identity = credential_identity(mount.credential)
    if identity:
        return f"{mount.name}:{mount.uri}:{identity}:{fingerprint}"
    return f"{mount.name}:{mount.uri}:{fingerprint}"


async def mount_fingerprint(resolver: MountResolver, mount: MountSpec) -> str | None:
    """Async form of :func:`mount_fingerprint_sync`, for the executor."""
    return mount_fingerprint_sync(resolver, mount)
