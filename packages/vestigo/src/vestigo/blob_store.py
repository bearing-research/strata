"""Blob storage backends for artifact data: local disk, S3, GCS and Azure.

Blobs are keyed by ``(artifact_id, version)``; metadata lives separately in the
artifact store's database.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Buffer, Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from azure.storage.blob import StorageStreamDownloader

BLOB_STREAM_CHUNK_BYTES = 64 * 1024
"""Default read/write chunk size for blob streaming callers.

Matches httpx's multipart ``FileField.CHUNK_SIZE`` and FastAPI's
``FileResponse`` default, so the HTTP-boundary streams line up with
the blob-store streams without extra copying.
"""


OBJECT_STORE_OPERATIONS = ("put", "get", "head", "list", "delete")

GCS_READ_CHUNK_BYTES = 8 * 1024 * 1024
"""What a GCS reader fetches per request; the SDK's default of 40 MiB is what a reader
holds in memory, whatever the caller's read size."""


@dataclass(frozen=True)
class BlobSettings:
    """Credentials and endpoints for the object-store backends, built by the application.

    Every field is optional: a backend reads only its own, and ``None`` means the
    provider's own defaults (ambient credentials, the public endpoint).
    """

    s3_region: str | None = None
    s3_endpoint_url: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_anonymous: bool = False
    gcs_credentials_json: str | None = None
    gcs_anonymous: bool = False
    gcs_endpoint_override: str | None = None
    azure_account_name: str | None = None
    azure_account_key: str | None = None
    azure_connection_string: str | None = None
    azure_sas_token: str | None = None
    azure_use_default_credential: bool = False
    azure_endpoint_url: str | None = None


class ObjectStoreErrors:
    """Failed object-store calls by operation, for the consumer's error metric."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(OBJECT_STORE_OPERATIONS, 0)

    def record(self, operation: str) -> None:
        with self._lock:
            self._counts[operation] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


_object_store_errors = ObjectStoreErrors()


def get_object_store_errors() -> ObjectStoreErrors:
    """The process-wide object-store error counts."""
    return _object_store_errors


class BlobStore(ABC):
    """Abstract base for blob storage backends, keyed by ``(artifact_id, version)``."""

    @abstractmethod
    def open_blob_reader(
        self, artifact_id: str, version: int
    ) -> AbstractContextManager[BinaryIO] | None:
        """Open a streaming reader for a blob, or return ``None`` if it does not exist.

        The context manager yields a binary file-like object; callers should rely only on
        ``read(size)``.
        """
        ...

    @abstractmethod
    def open_blob_writer(self, artifact_id: str, version: int) -> AbstractContextManager[BinaryIO]:
        """Open a streaming writer for a blob.

        Commits atomically on clean context exit; on exception the partial write is discarded.
        """
        ...

    def write_blob(self, artifact_id: str, version: int, data: bytes) -> None:
        """Write a blob from bytes, atomically where the backend allows."""
        with self.open_blob_writer(artifact_id, version) as writer:
            writer.write(data)

    def read_blob(self, artifact_id: str, version: int) -> bytes | None:
        """Read a blob's bytes, or return None if it does not exist."""
        reader = self.open_blob_reader(artifact_id, version)
        if reader is None:
            return None
        with reader as f:
            return f.read()

    @abstractmethod
    def blob_exists(self, artifact_id: str, version: int) -> bool:
        """Return whether the blob exists."""
        ...

    @abstractmethod
    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return the blob's size in bytes without reading it, or ``None`` if it does not exist."""
        ...

    def publish_blob_from_path(self, artifact_id: str, version: int, source_path: Path) -> None:
        """Atomically publish a blob from a complete local file.

        The default pipes through ``open_blob_writer``; backends may upload directly.
        The source file is not consumed: the caller still owns and removes it.
        """
        with open(source_path, "rb") as src, self.open_blob_writer(artifact_id, version) as dst:
            while True:
                chunk = src.read(BLOB_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)

    @abstractmethod
    def delete_blob(self, artifact_id: str, version: int) -> bool:
        """Delete a blob; return False if it did not exist."""
        ...

    def remove_stale_temp_files(self, idle_seconds: float) -> int:
        """Remove partial writes a killed process left, untouched for ``idle_seconds``.

        Returns how many went. Only the local store stages writes beside its blobs; the
        others stage in the system temp dir.
        """
        return 0

    def presign_get(self, artifact_id: str, version: int, ttl_seconds: int) -> str | None:
        """Return a URL that reads this blob straight from the object store, or ``None``.

        ``None`` means the backend cannot sign one (local disk, or credentials this
        process cannot sign with), and the caller serves the bytes itself.
        """
        return None

    def presign_post(
        self, artifact_id: str, version: int, max_bytes: int, ttl_seconds: int
    ) -> tuple[str, dict[str, str]] | None:
        """Return ``(url, fields)`` for a form upload of at most ``max_bytes``, or ``None``.

        POST the fields plus the body as the ``file`` part. A POST policy rather than a
        presigned PUT because it lets the object store refuse an oversized upload.
        """
        return None

    def presign_put(
        self, artifact_id: str, version: int, ttl_seconds: int
    ) -> tuple[str, dict[str, str]] | None:
        """Return ``(url, headers)`` to PUT the body to, or ``None``.

        For stores with no form upload (Azure). Nothing bounds the body, so finalize
        enforces the size limit after the fact.
        """
        return None

    def _blob_key(self, artifact_id: str, version: int) -> str:
        """Return the storage key ``{artifact_id}@v={version}.arrow``.

        An id containing uppercase gets a short hash of its exact bytes appended, so
        ids differing only in case (``Widget`` vs ``widget``) do not collide on a
        case-insensitive filesystem. All-lowercase ids keep the plain key.
        """
        key = artifact_id
        if key != key.lower():
            key = f"{key}-{hashlib.sha256(artifact_id.encode()).hexdigest()[:8]}"
        return f"{key}@v={version}.arrow"

    @staticmethod
    @contextmanager
    def _staged_local_writer(
        commit: Callable[[Path], None],
        *,
        prefix: str = "vestigo_blob_",
    ) -> Generator[BinaryIO]:
        """Stage writes through a local tempfile and pass it to ``commit(path)`` on clean exit.

        On exception the tempfile is removed and ``commit`` never runs, so a partial
        write is never observable. The handle is closed before ``commit`` so backends
        can reopen the file on Windows.
        """
        fd, tmp_name = tempfile.mkstemp(prefix=prefix, suffix=".tmp")
        tmp_path = Path(tmp_name)
        handle = os.fdopen(fd, "w+b")
        try:
            yield handle
            handle.flush()
            handle.close()
            commit(tmp_path)
        finally:
            if not handle.closed:
                handle.close()
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


class LocalBlobStore(BlobStore):
    """Local filesystem blob storage, written atomically.

    Paths are ``{blobs_dir}/{artifact_id}@v={version}.arrow``.
    """

    def __init__(self, blobs_dir: Path):
        """Initialize the store over ``blobs_dir``."""
        from vestigo.file_modes import private_dir

        self.blobs_dir = blobs_dir
        private_dir(self.blobs_dir)

    def _blob_path(self, artifact_id: str, version: int) -> Path:
        """Get filesystem path for a blob.

        Raises:
            ValueError: the id names a path, which would read or write a file outside the
                blob directory, another artifact's included.
        """
        path = self.blobs_dir / self._blob_key(artifact_id, version)
        if path.parent != self.blobs_dir:
            raise ValueError(f"an artifact id cannot name a path: {artifact_id!r}")
        return path

    def open_blob_reader(
        self, artifact_id: str, version: int
    ) -> AbstractContextManager[BinaryIO] | None:
        """Open a streaming reader for a local blob."""
        path = self._blob_path(artifact_id, version)
        if not path.exists():
            return None
        return open(path, "rb")

    def open_blob_writer(self, artifact_id: str, version: int) -> AbstractContextManager[BinaryIO]:
        """Open a streaming writer for a local blob with atomic commit."""
        path = self._blob_path(artifact_id, version)

        @contextmanager
        def _writer() -> Generator[BinaryIO]:
            fd, tmp_name = tempfile.mkstemp(
                prefix=path.name + ".",
                suffix=".tmp",
                dir=path.parent,
            )
            tmp_path = Path(tmp_name)
            handle = os.fdopen(fd, "wb")
            try:
                yield handle
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                os.replace(tmp_path, path)
                # The rename is in the directory, which has its own write-back; without
                # this a power loss can keep the metadata row and lose the file.
                _fsync_dir(path.parent)
            except BaseException:
                if not handle.closed:
                    handle.close()
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
                raise

        return _writer()

    def blob_exists(self, artifact_id: str, version: int) -> bool:
        """Check if blob exists on local filesystem."""
        return self._blob_path(artifact_id, version).exists()

    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return blob size from filesystem metadata."""
        path = self._blob_path(artifact_id, version)
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return None

    def delete_blob(self, artifact_id: str, version: int) -> bool:
        """Delete blob from local filesystem."""
        path = self._blob_path(artifact_id, version)
        try:
            path.unlink()
        except FileNotFoundError:
            return False  # absent, or a concurrent sweep got there first
        return True

    def remove_stale_temp_files(self, idle_seconds: float) -> int:
        """Remove the ``*.tmp`` files of writes that died, untouched for ``idle_seconds``."""
        cutoff = time.time() - idle_seconds
        removed = 0
        for path in self.blobs_dir.glob("*.tmp"):
            try:
                # A live write keeps touching its file, so an old mtime means nobody is writing.
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except FileNotFoundError:
                continue  # its writer finished or gave up meanwhile
        return removed


def _fsync_dir(path: Path) -> None:
    """Flush a directory's entries (a rename or unlink in it) to disk; a no-op on Windows."""
    if sys.platform == "win32":
        return  # directories cannot be opened for fsync there
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _endpoint_url(endpoint: str) -> str:
    """``host:port`` as a URL: the SDKs take a URL where a bare host was once accepted."""
    return endpoint if "://" in endpoint else f"https://{endpoint}"


@contextmanager
def _object_store_call(
    operation: str, absent: Callable[[BaseException], bool] | None = None
) -> Generator[None]:
    """One call to an object store's SDK.

    An error ``absent`` recognises is the answer "not there" and is raised as
    ``FileNotFoundError``; any other is counted for the consumer's metric and re-raised
    as the SDK raised it.
    """
    try:
        yield
    except Exception as exc:
        if absent is not None and absent(exc):
            raise FileNotFoundError(str(exc)) from exc
        _object_store_errors.record(operation)
        raise


def _s3_absent(exc: BaseException) -> bool:
    """Whether a botocore error says the object is not there.

    ``NoSuchKey`` on a GET; a HEAD carries no body, so its bare ``404`` is taken too, and a
    missing bucket reads as a missing object there. ``NoSuchBucket`` on a GET propagates.
    """
    response = getattr(exc, "response", None) or {}
    return response.get("Error", {}).get("Code") in ("NoSuchKey", "404")


class S3BlobStore(BlobStore):
    """S3 or S3-compatible (MinIO, LocalStack) blob storage through boto3.

    Keys are ``{prefix}/{artifact_id}@v={version}.arrow``. The one client reads, writes
    and signs, so a role is resolved once and its session token signs the URLs.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "artifacts",
        region: str | None = None,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        anonymous: bool = False,
    ):
        """Initialize the S3 store.

        Without ``access_key``/``secret_key`` credentials come from boto3's chain (the
        environment, a profile, a role); ``endpoint_url`` targets an S3-compatible
        service, addressed path-style.

        Raises:
            ValueError: only one of ``access_key`` and ``secret_key`` was given, which
                would otherwise fall back to ambient credentials without a word.
        """
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError as exc:
            raise ImportError("The S3 backend needs boto3: install vestigo[s3]") from exc

        if bool(access_key) != bool(secret_key):
            raise ValueError("S3 needs both access_key and secret_key, or neither")

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._anonymous = anonymous

        session: dict[str, Any] = {}
        if region:
            session["region_name"] = region
        if access_key and secret_key:
            session["aws_access_key_id"] = access_key
            session["aws_secret_access_key"] = secret_key
        self._session = boto3.session.Session(**session)
        config = Config(
            signature_version=UNSIGNED if anonymous else "s3v4",
            # A custom endpoint is one host; a bucket subdomain of it rarely resolves.
            s3={"addressing_style": "path" if endpoint_url else "auto"},
        )
        self._client = self._session.client(
            "s3", config=config, endpoint_url=_endpoint_url(endpoint_url) if endpoint_url else None
        )
        # Resolved once here: botocore keeps a hit but walks the whole chain again on a
        # miss, an instance-metadata probe included, and presigning asks per URL.
        self._signs = not anonymous and self._session.get_credentials() is not None

    def _object_key(self, artifact_id: str, version: int) -> str:
        blob_key = self._blob_key(artifact_id, version)
        return f"{self.prefix}/{blob_key}" if self.prefix else blob_key

    def presign_get(self, artifact_id: str, version: int, ttl_seconds: int) -> str | None:
        if not self._signs:
            return None
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._object_key(artifact_id, version)},
            ExpiresIn=int(ttl_seconds),
        )

    def presign_post(
        self, artifact_id: str, version: int, max_bytes: int, ttl_seconds: int
    ) -> tuple[str, dict[str, str]] | None:
        if not self._signs:
            return None
        signed = self._client.generate_presigned_post(
            self.bucket,
            self._object_key(artifact_id, version),
            Conditions=[["content-length-range", 1, int(max_bytes)]],
            ExpiresIn=int(ttl_seconds),
        )
        return signed["url"], signed["fields"]

    def open_blob_reader(
        self, artifact_id: str, version: int
    ) -> AbstractContextManager[BinaryIO] | None:
        """Open a streaming reader for an S3 blob; a backend error propagates."""
        key = self._object_key(artifact_id, version)
        try:
            with _object_store_call("get", _s3_absent):
                response = self._client.get_object(Bucket=self.bucket, Key=key)
        except FileNotFoundError:
            return None
        return response["Body"]  # a file-like that closes itself on context exit

    def _upload_from_path(self, key: str, source_path: Path) -> None:
        """Upload a local file to ``key``, in parts past the SDK's threshold.

        A failure leaves the key as it was: an aborted multipart upload is nothing, and
        a single PUT lands whole or not at all.
        """
        with open(source_path, "rb") as src, _object_store_call("put"):
            self._client.upload_fileobj(src, self.bucket, key)

    def open_blob_writer(self, artifact_id: str, version: int) -> AbstractContextManager[BinaryIO]:
        """Open a streaming writer for an S3 blob, staged locally so failed writes never upload."""
        key = self._object_key(artifact_id, version)
        return self._staged_local_writer(
            lambda staged: self._upload_from_path(key, staged),
            prefix="vestigo_s3_blob_",
        )

    def publish_blob_from_path(self, artifact_id: str, version: int, source_path: Path) -> None:
        """Stream ``source_path`` straight to S3, skipping the local staging copy."""
        self._upload_from_path(self._object_key(artifact_id, version), source_path)

    def blob_exists(self, artifact_id: str, version: int) -> bool:
        """Check if blob exists in S3; a backend error propagates."""
        return self.blob_size(artifact_id, version) is not None

    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return blob size from S3 object metadata; a backend error propagates."""
        key = self._object_key(artifact_id, version)
        try:
            with _object_store_call("head", _s3_absent):
                head = self._client.head_object(Bucket=self.bucket, Key=key)
        except FileNotFoundError:
            return None
        return int(head["ContentLength"])

    def delete_blob(self, artifact_id: str, version: int) -> bool:
        """Delete blob from S3; False when it was not there, and a backend error propagates.

        S3 answers the delete of a missing key as a success, so presence is checked first.
        """
        if not self.blob_exists(artifact_id, version):
            return False
        key = self._object_key(artifact_id, version)
        with _object_store_call("delete"):
            self._client.delete_object(Bucket=self.bucket, Key=key)
        return True

    @classmethod
    def from_settings(
        cls, settings: BlobSettings, bucket: str, prefix: str = "artifacts"
    ) -> S3BlobStore:
        """An S3BlobStore on the ``s3_*`` fields of *settings*."""
        return cls(
            bucket=bucket,
            prefix=prefix,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            anonymous=settings.s3_anonymous,
        )


def azure_account_url(endpoint_url: str, account_name: str) -> str:
    """The account's blob URL: the endpoint host with the account as its first path segment.

    An endpoint that already ends in ``/<account>`` is taken as that URL, not doubled.
    """
    endpoint = endpoint_url.rstrip("/")
    if endpoint.endswith(f"/{account_name}"):
        return endpoint
    return f"{endpoint}/{account_name}"


def _gcs_absent(exc: BaseException) -> bool:
    """Whether a google-api-core error is a 404."""
    from google.api_core.exceptions import NotFound

    return isinstance(exc, NotFound)


_GCS_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _gcs_credentials(credentials_json: str | None, anonymous: bool) -> Any:
    """The credentials a GCS store runs on.

    A credentials file of any kind ``GOOGLE_APPLICATION_CREDENTIALS`` accepts (a
    service-account key, workload identity federation, an authorized user), given as its
    JSON or as its path; none, for a public bucket; else Application Default Credentials.

    Raises:
        google.auth.exceptions.DefaultCredentialsError: neither credentials nor anonymous,
            and the environment holds none: a store that could not read or write.
    """
    import google.auth
    from google.auth.credentials import AnonymousCredentials

    if anonymous:
        return AnonymousCredentials()
    if credentials_json:
        try:
            info = json.loads(credentials_json)
        except ValueError:
            info = None
        if not isinstance(info, dict):
            info = json.loads(Path(credentials_json).read_text(encoding="utf-8"))
        return _gcs_credentials_from_info(info)
    credentials, _ = google.auth.default(scopes=_GCS_SCOPES)
    return credentials


def _gcs_credentials_from_info(info: dict[str, Any]) -> Any:
    """Credentials from a credentials file's contents, by its ``type``.

    The library's generic loaders are deprecated for not validating what they load; the
    per-type constructors are what they recommend. The types are the ones
    ``GOOGLE_APPLICATION_CREDENTIALS`` accepts.

    Raises:
        ValueError: a ``type`` none of them covers.
    """
    kind = info.get("type")
    if kind == "service_account":
        from google.oauth2.service_account import Credentials

        return Credentials.from_service_account_info(info, scopes=_GCS_SCOPES)
    if kind == "authorized_user":
        from google.oauth2.credentials import Credentials as UserCredentials

        return UserCredentials.from_authorized_user_info(info, scopes=_GCS_SCOPES)
    if kind == "impersonated_service_account":
        from google.auth.impersonated_credentials import Credentials as Impersonated

        return Impersonated.from_impersonated_service_account_info(info, scopes=_GCS_SCOPES)
    if kind == "external_account":
        source = info.get("credential_source") or {}
        if "executable" in source:
            from google.auth.pluggable import Credentials as External
        elif str(source.get("environment_id", "")).startswith("aws"):
            from google.auth.aws import Credentials as External
        else:
            from google.auth.identity_pool import Credentials as External
        return External.from_info(info, scopes=_GCS_SCOPES)
    raise ValueError(f"Unsupported GCS credentials type {kind!r}")


class GCSBlobStore(BlobStore):
    """Google Cloud Storage blob storage through google-cloud-storage.

    Keys are ``{prefix}/{artifact_id}@v={version}.arrow``. The one client reads, writes
    and signs, on a service-account key, Application Default Credentials, or none.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "artifacts",
        credentials_json: str | None = None,
        anonymous: bool = False,
        endpoint_override: str | None = None,
    ):
        """Initialize the GCS store.

        ``credentials_json`` is a service-account key: a file path, or the key's JSON
        itself, which is what a container's environment holds. ``endpoint_override``
        is an emulator's URL.

        Raises:
            google.auth.exceptions.DefaultCredentialsError: see ``_gcs_credentials``.
        """
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ImportError(
                "The GCS backend needs google-cloud-storage: install vestigo[gcs]"
            ) from exc

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._anonymous = anonymous

        options = {"api_endpoint": _endpoint_url(endpoint_override)} if endpoint_override else None
        self._client = storage.Client(
            project=None,
            credentials=_gcs_credentials(credentials_json, anonymous),
            client_options=options,
        )

    def _object_name(self, artifact_id: str, version: int) -> str:
        blob_key = self._blob_key(artifact_id, version)
        return f"{self.prefix}/{blob_key}" if self.prefix else blob_key

    def _blob(self, name: str) -> Any:
        return self._client.bucket(self.bucket).blob(name)

    def _signing_kwargs(self) -> dict[str, str] | None:
        """How to sign: locally with a key (``{}``), through IAM signBlob, or ``None``."""
        from google.auth.credentials import Signing

        if self._anonymous:
            return None
        credentials = self._client._credentials
        if isinstance(credentials, Signing):
            return {}
        # Workload identity and the GCE metadata server hold no key; IAM signs as the
        # attached service account (it needs roles/iam.serviceAccountTokenCreator on itself).
        if not hasattr(credentials, "service_account_email"):
            return None
        if not credentials.valid:
            from google.auth.transport.requests import Request

            credentials.refresh(Request())
        return {
            "service_account_email": credentials.service_account_email,
            "access_token": credentials.token,
        }

    def presign_get(self, artifact_id: str, version: int, ttl_seconds: int) -> str | None:
        from datetime import timedelta

        signing = self._signing_kwargs()
        if signing is None:
            return None
        return self._blob(self._object_name(artifact_id, version)).generate_signed_url(
            version="v4", expiration=timedelta(seconds=int(ttl_seconds)), method="GET", **signing
        )

    def presign_post(
        self, artifact_id: str, version: int, max_bytes: int, ttl_seconds: int
    ) -> tuple[str, dict[str, str]] | None:
        from datetime import timedelta

        signing = self._signing_kwargs()
        if signing is None:
            return None
        policy = self._client.generate_signed_post_policy_v4(
            self.bucket,
            self._object_name(artifact_id, version),
            expiration=timedelta(seconds=int(ttl_seconds)),
            conditions=[["content-length-range", 1, int(max_bytes)]],
            **signing,
        )
        return policy["url"], policy["fields"]

    def open_blob_reader(
        self, artifact_id: str, version: int
    ) -> AbstractContextManager[BinaryIO] | None:
        """Open a streaming reader for a GCS blob; a backend error propagates.

        The metadata read first answers absence and pins the reader to one generation.
        The reader fetches ``GCS_READ_CHUNK_BYTES`` at a time, whatever the caller reads.
        """
        blob = self._blob(self._object_name(artifact_id, version))
        try:
            with _object_store_call("get", _gcs_absent):
                blob.reload()
        except FileNotFoundError:
            return None
        return blob.open("rb", chunk_size=GCS_READ_CHUNK_BYTES)

    def _upload_from_path(self, name: str, source_path: Path) -> None:
        """Upload a local file to ``name``: one request to 8 MiB, resumable past it.

        The size is given, or the SDK opens a resumable session for every blob. A failure
        leaves the name as it was: an unfinished resumable upload is nothing. Retried on
        transient errors, since the same bytes land at the same name.
        """
        from google.cloud.storage.retry import DEFAULT_RETRY

        size = source_path.stat().st_size
        with open(source_path, "rb") as src, _object_store_call("put"):
            self._blob(name).upload_from_file(src, size=size, retry=DEFAULT_RETRY)

    def open_blob_writer(self, artifact_id: str, version: int) -> AbstractContextManager[BinaryIO]:
        """Open a streaming writer for a GCS blob, staged locally so failed writes never upload."""
        name = self._object_name(artifact_id, version)
        return self._staged_local_writer(
            lambda staged: self._upload_from_path(name, staged),
            prefix="vestigo_gcs_blob_",
        )

    def publish_blob_from_path(self, artifact_id: str, version: int, source_path: Path) -> None:
        """Stream ``source_path`` straight to GCS, skipping the local staging copy."""
        self._upload_from_path(self._object_name(artifact_id, version), source_path)

    def blob_exists(self, artifact_id: str, version: int) -> bool:
        """Check if blob exists in GCS; a backend error propagates."""
        with _object_store_call("head"):
            return bool(self._blob(self._object_name(artifact_id, version)).exists())

    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return blob size from GCS object metadata; a backend error propagates."""
        blob = self._blob(self._object_name(artifact_id, version))
        try:
            with _object_store_call("head", _gcs_absent):
                blob.reload()
        except FileNotFoundError:
            return None
        return int(blob.size) if blob.size is not None else None

    def delete_blob(self, artifact_id: str, version: int) -> bool:
        """Delete blob from GCS; False when it was not there, and a backend error propagates."""
        try:
            with _object_store_call("delete", _gcs_absent):
                self._blob(self._object_name(artifact_id, version)).delete()
        except FileNotFoundError:
            return False
        return True

    @classmethod
    def from_settings(
        cls, settings: BlobSettings, bucket: str, prefix: str = "artifacts"
    ) -> GCSBlobStore:
        """A GCSBlobStore on the ``gcs_*`` fields of *settings*."""
        return cls(
            bucket=bucket,
            prefix=prefix,
            credentials_json=settings.gcs_credentials_json,
            anonymous=settings.gcs_anonymous,
            endpoint_override=settings.gcs_endpoint_override,
        )


def _azure_absent(exc: BaseException) -> bool:
    from azure.core.exceptions import ResourceNotFoundError

    return isinstance(exc, ResourceNotFoundError)


class _AzureDownloadReader(io.RawIOBase):
    """Raw stream over an Azure ``StorageStreamDownloader``'s chunks.

    Wrapped in ``io.BufferedReader`` it behaves as a real binary file.
    """

    def __init__(self, downloader: StorageStreamDownloader) -> None:
        self._chunks: Iterator[bytes] = iter(downloader.chunks())
        # The unread rest of the current chunk. A memoryview, so consuming the
        # front makes a view rather than copying the rest of a 32 MiB chunk.
        self._buffer = memoryview(b"")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        while not self._buffer:
            chunk = next(self._chunks, None)
            if chunk is None:
                return 0
            self._buffer = memoryview(chunk)
        view = memoryview(buffer).cast("B")
        n = min(len(view), len(self._buffer))
        view[:n] = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return n

    def readall(self) -> bytes:
        # ``BufferedReader.read()`` lands here; the inherited loop would copy
        # through a 128 KiB buffer, so join the chunks instead.
        rest = [self._buffer, *self._chunks]
        self._buffer = memoryview(b"")
        return b"".join(rest)


class AzureBlobStore(BlobStore):
    """Azure Blob Storage backend.

    Keys are ``{prefix}/{artifact_id}@v={version}.arrow``. Authenticates with a
    connection string, account key, SAS token, or DefaultAzureCredential. Requires
    the ``azure`` extra.
    """

    def __init__(
        self,
        account_name: str,
        container_name: str,
        prefix: str = "artifacts",
        account_key: str | None = None,
        connection_string: str | None = None,
        sas_token: str | None = None,
        use_default_credential: bool = False,
        endpoint_url: str | None = None,
    ):
        """Initialize the Azure store; ``endpoint_url`` targets the Azurite emulator."""
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as e:
            raise ImportError(
                "Azure Blob Storage support requires the 'azure' extra. "
                "Install with: pip install 'vestigo[azure]'"
            ) from e

        self.account_name = account_name
        self.container_name = container_name
        self.prefix = prefix.strip("/")
        # ``endpoint_url`` is the blob host with the account as the first path segment
        # (Azurite's form); the lake path derives its connection string the same way.
        if endpoint_url and account_name:
            self._account_url = azure_account_url(endpoint_url, account_name)
        else:
            self._account_url = endpoint_url or f"https://{account_name}.blob.core.windows.net"
        self._delegation_key: Any = None
        self._delegation_key_expiry = 0.0

        if connection_string:
            self._client = ContainerClient.from_connection_string(
                conn_str=connection_string,
                container_name=container_name,
            )
        elif use_default_credential:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            self._client = ContainerClient(
                account_url=self._account_url,
                container_name=container_name,
                credential=credential,
            )
        elif sas_token:
            # SAS token can be passed as credential
            self._client = ContainerClient(
                account_url=self._account_url,
                container_name=container_name,
                credential=sas_token,
            )
        elif account_key:
            self._client = ContainerClient(
                account_url=self._account_url,
                container_name=container_name,
                credential=account_key,
            )
        else:
            raise ValueError(
                "Azure Blob Storage requires one of: connection_string, account_key, "
                "sas_token, or use_default_credential=True"
            )

    def _azure_key(self, artifact_id: str, version: int) -> str:
        """Get full Azure blob key."""
        blob_key = self._blob_key(artifact_id, version)
        if self.prefix:
            return f"{self.prefix}/{blob_key}"
        return blob_key

    def _sas_signing(self, ttl_seconds: int) -> dict[str, Any] | None:
        """Key material for a blob SAS: the account key, a user delegation key, or ``None``.

        A configured SAS token cannot mint narrower ones, and handing it out would grant
        the worker everything it grants.
        """
        from datetime import UTC, datetime, timedelta

        credential = self._client.credential
        account_key = getattr(credential, "account_key", None)
        if account_key:
            return {"account_key": account_key}
        if not hasattr(credential, "get_token"):
            return None
        # A delegation key costs a round trip, so reuse one while it outlives the SAS.
        if self._delegation_key is None or self._delegation_key_expiry < time.time() + ttl_seconds:
            from azure.storage.blob import BlobServiceClient

            now = datetime.now(UTC)
            expiry = now + timedelta(seconds=ttl_seconds + 3600)
            self._delegation_key = BlobServiceClient(
                self._account_url, credential=credential
            ).get_user_delegation_key(now - timedelta(minutes=5), expiry)
            self._delegation_key_expiry = expiry.timestamp()
        return {"user_delegation_key": self._delegation_key}

    def _presign(
        self, artifact_id: str, version: int, ttl_seconds: int, permission: Any
    ) -> str | None:
        from datetime import UTC, datetime, timedelta

        from azure.storage.blob import generate_blob_sas

        signing = self._sas_signing(int(ttl_seconds))
        if signing is None:
            return None
        key = self._azure_key(artifact_id, version)
        sas = generate_blob_sas(
            account_name=self._client.account_name or self.account_name,
            container_name=self.container_name,
            blob_name=key,
            permission=permission,
            expiry=datetime.now(UTC) + timedelta(seconds=int(ttl_seconds)),
            **signing,
        )
        return f"{self._client.get_blob_client(key).url}?{sas}"

    def presign_get(self, artifact_id: str, version: int, ttl_seconds: int) -> str | None:
        from azure.storage.blob import BlobSasPermissions

        return self._presign(artifact_id, version, ttl_seconds, BlobSasPermissions(read=True))

    def presign_put(
        self, artifact_id: str, version: int, ttl_seconds: int
    ) -> tuple[str, dict[str, str]] | None:
        from azure.storage.blob import BlobSasPermissions

        url = self._presign(
            artifact_id, version, ttl_seconds, BlobSasPermissions(create=True, write=True)
        )
        return (url, {"x-ms-blob-type": "BlockBlob"}) if url is not None else None

    def open_blob_reader(
        self, artifact_id: str, version: int
    ) -> AbstractContextManager[BinaryIO] | None:
        """Open a streaming reader for an Azure blob."""
        key = self._azure_key(artifact_id, version)
        blob_client = self._client.get_blob_client(key)
        try:
            with _object_store_call("get", _azure_absent):
                downloader = blob_client.download_blob()
        except FileNotFoundError:
            return None

        @contextmanager
        def _reader() -> Generator[BinaryIO]:
            stream = io.BufferedReader(_AzureDownloadReader(downloader))
            try:
                yield stream
            finally:
                stream.close()

        return _reader()

    def _upload_from_path(self, key: str, source_path: Path) -> None:
        """Stream a local file to ``key`` via the Azure SDK."""
        blob_client = self._client.get_blob_client(key)
        with open(source_path, "rb") as src, _object_store_call("put"):
            blob_client.upload_blob(src, overwrite=True)

    def open_blob_writer(self, artifact_id: str, version: int) -> AbstractContextManager[BinaryIO]:
        """Open a streaming writer for an Azure blob, uploaded on clean exit.

        Staged through a local tempfile because the SDK streams uploads from a file
        handle, not from a caller-written stream.
        """
        key = self._azure_key(artifact_id, version)
        return self._staged_local_writer(
            lambda staged: self._upload_from_path(key, staged),
            prefix="vestigo_azure_blob_",
        )

    def publish_blob_from_path(self, artifact_id: str, version: int, source_path: Path) -> None:
        """Stream ``source_path`` straight to Azure, skipping the local staging copy."""
        key = self._azure_key(artifact_id, version)
        self._upload_from_path(key, source_path)

    def blob_exists(self, artifact_id: str, version: int) -> bool:
        """Check if blob exists in Azure Blob Storage."""
        key = self._azure_key(artifact_id, version)
        blob_client = self._client.get_blob_client(key)
        with _object_store_call("head"):
            return blob_client.exists()

    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return blob size from Azure Blob Storage metadata."""
        key = self._azure_key(artifact_id, version)
        blob_client = self._client.get_blob_client(key)
        try:
            with _object_store_call("head", _azure_absent):
                props = blob_client.get_blob_properties()
        except FileNotFoundError:
            return None
        size = getattr(props, "size", None)
        return int(size) if size is not None else None

    def delete_blob(self, artifact_id: str, version: int) -> bool:
        """Delete blob from Azure Blob Storage."""
        key = self._azure_key(artifact_id, version)
        blob_client = self._client.get_blob_client(key)
        try:
            with _object_store_call("delete", _azure_absent):
                blob_client.delete_blob()
        except FileNotFoundError:
            return False
        return True

    @classmethod
    def from_settings(
        cls, settings: BlobSettings, container_name: str, prefix: str = "artifacts"
    ) -> AzureBlobStore:
        """An AzureBlobStore on the ``azure_*`` fields of *settings*."""
        return cls(
            account_name=settings.azure_account_name or "",
            container_name=container_name,
            prefix=prefix,
            account_key=settings.azure_account_key,
            connection_string=settings.azure_connection_string,
            sas_token=settings.azure_sas_token,
            use_default_credential=settings.azure_use_default_credential,
            endpoint_url=settings.azure_endpoint_url,
        )
