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
import hashlib
import json
import logging
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse

from pydantic import BaseModel

from strata.notebook.credentials import CredentialError, CredentialResolver, credential_identity
from strata.notebook.harness_user import UnsafeRunFile, open_tree_file
from strata.notebook.models import MountMode, MountSpec
from strata.notebook.mount_cache import MountCache, object_key

logger = logging.getLogger(__name__)


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

        if mount.mode == MountMode.READ_ONLY:
            return await self._resolve_remote_ro(mount, scheme, remote_path)
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
    ) -> ResolvedMount:
        """Resolve a read-only remote mount to a view in the shared mount cache."""
        protocol = _scheme_to_fsspec_protocol(scheme)
        storage_options = self.storage_options(mount)
        try:
            fingerprint, local_path, moved = await asyncio.to_thread(
                self._prepare_view, mount, protocol, remote_path, storage_options
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to materialize remote mount '{mount.name}' from {mount.uri}: {e}"
            ) from e
        resolved = ResolvedMount(
            spec=mount, local_path=local_path, fingerprint=fingerprint, bytes_moved=moved
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
    ) -> tuple[str, Path, int]:
        """The mount's fingerprint, its view and the bytes downloaded for it, from one listing."""
        fs = _mount_filesystem(protocol, storage_options)
        options_digest = _options_digest(storage_options)
        listing: dict[str, dict[str, Any]] | None = None
        if mount.pin is not None:
            fingerprint = hashlib.sha256(f"pin:{mount.pin}".encode()).hexdigest()
        else:
            listing = _list_remote_file_info(fs, protocol, remote_path)
            fingerprint = _fingerprint_listing(listing)
        key = _view_key(protocol, remote_path, options_digest, fingerprint)

        found = self.mount_cache.find_view(key)
        if found is not None:
            return fingerprint, found, 0
        if listing is None:
            listing = _list_remote_file_info(fs, protocol, remote_path)
        files = []
        for remote_name, info in sorted(listing.items()):
            rel = _relative_remote_path(remote_name, protocol, remote_path)
            _check_relative(rel, mount.name, remote_name)
            files.append(
                (rel, object_key(protocol, options_digest, remote_name, info), remote_name)
            )
        view, moved = self.mount_cache.build_view(
            key, files, lambda name, dest: fs.get(f"{protocol}://{name}", str(dest))
        )
        return fingerprint, view, moved

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
            return hashlib.sha256(os.urandom(32)).hexdigest(), None
        except Exception as e:
            logger.warning(
                "Failed to fingerprint remote mount %s://%s: %s",
                scheme,
                remote_path,
                e,
            )
            # Unique per call: forces re-execution.
            return hashlib.sha256(os.urandom(32)).hexdigest(), None

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


def _options_digest(storage_options: dict[str, Any]) -> str:
    """What keys a view and its objects to the options that reached the store.

    Neither is handed to a caller that reaches the store another way, so a credential that
    may list a prefix but not read it never gets another notebook's bytes. A rotated secret
    starts a fresh copy.
    """
    return hashlib.sha256(
        json.dumps(storage_options, sort_keys=True, default=str).encode()
    ).hexdigest()


def _view_key(protocol: str, remote_path: str, options_digest: str, fingerprint: str) -> str:
    return hashlib.sha256(
        "\0".join([protocol, remote_path, options_digest, fingerprint]).encode()
    ).hexdigest()


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
        credential = ClientAssertionCredential(tenant_id, client_id, token_file.read_text)
        return {**options, "credential": credential}
    return storage_options


def _external_account(token: str) -> dict[str, Any] | None:
    """The workload-identity configuration in the file *token* names, if it is one."""
    try:
        info = json.loads(Path(token).read_text())
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
    """

    bytes: int | None
    cached_bytes: int | None


def mount_fingerprint_sync(resolver: MountResolver, mount: MountSpec) -> str | None:
    """The provenance component for one mount, shared by execution and staleness.

    Both must produce the same string, or artifacts are keyed under a hash
    staleness never reproduces. It includes the uri (two directories with the
    same files are different inputs) and the credential's name, never its values.
    An unresolvable credential gives a unique fingerprint: the cell shows stale,
    runs, and fails naming the credential.
    """
    return weigh_mount_sync(resolver, mount, None)[0]


def weigh_mount_sync(
    resolver: MountResolver, mount: MountSpec, cache: MountCache | None
) -> tuple[str | None, MountWeight | None]:
    """:func:`mount_fingerprint_sync`'s component, with the mount's weight from the same
    listing: its share already in *cache*, the one the cell runs against.

    The weight is ``None`` for a mount that moves nothing to read (``file://``) or that a
    run does not read through a view (``rw``).
    """
    try:
        storage_options = resolver.storage_options(mount)
    except CredentialError as exc:
        logger.warning("mount %s: %s", mount.name, exc)
        unresolved = f"{mount.name}:unresolved:{hashlib.sha256(os.urandom(32)).hexdigest()}"
        return unresolved, None
    scheme, path = parse_mount_uri(mount.uri)
    weight: MountWeight | None = None
    if scheme == "file" or mount.mode == MountMode.READ_WRITE or mount.pin is not None:
        fingerprint = MountFingerprinter.fingerprint_mount_sync(
            mount, storage_options=storage_options or None
        )
        if mount.pin is not None and scheme != "file":
            weight = MountWeight(bytes=None, cached_bytes=None)
    else:
        fingerprint, listing = MountFingerprinter.remote_listing_sync(
            scheme, path, storage_options or None
        )
        weight = _weigh_listing(scheme, path, storage_options, fingerprint, listing, cache)
    if fingerprint is None:
        return None, weight
    identity = credential_identity(mount.credential)
    if identity:
        return f"{mount.name}:{mount.uri}:{identity}:{fingerprint}", weight
    return f"{mount.name}:{mount.uri}:{fingerprint}", weight


def _weigh_listing(
    scheme: str,
    path: str,
    storage_options: dict[str, Any],
    fingerprint: str,
    listing: dict[str, dict[str, Any]] | None,
    cache: MountCache | None,
) -> MountWeight:
    if listing is None:
        return MountWeight(bytes=None, cached_bytes=None)
    sizes = {name: int(info.get("size") or 0) for name, info in listing.items()}
    if cache is None:
        return MountWeight(bytes=sum(sizes.values()), cached_bytes=None)
    protocol = _scheme_to_fsspec_protocol(scheme)
    digest = _options_digest(storage_options)
    objects = [
        (object_key(protocol, digest, name, info), sizes[name]) for name, info in listing.items()
    ]
    return MountWeight(
        bytes=sum(sizes.values()),
        cached_bytes=cache.cached_bytes(_view_key(protocol, path, digest, fingerprint), objects),
    )
