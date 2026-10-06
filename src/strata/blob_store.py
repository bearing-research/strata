"""Blob storage backends for artifact data: local disk, S3, GCS and Azure.

Blobs are keyed by ``(artifact_id, version)``; metadata lives separately in the
artifact store's database.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Buffer, Callable, Iterator
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from azure.storage.blob import StorageStreamDownloader

    from strata.config import StrataConfig

BLOB_STREAM_CHUNK_BYTES = 64 * 1024
"""Default read/write chunk size for blob streaming callers.

Matches httpx's multipart ``FileField.CHUNK_SIZE`` and FastAPI's
``FileResponse`` default, so the HTTP-boundary streams line up with
the blob-store streams without extra copying.
"""


def _sigv4_context(secret_key: str, region: str) -> tuple[str, str, bytes]:
    """``(amz_date, credential scope, signing key)`` for an S3 SigV4 signature now."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    datestamp = now.strftime("%Y%m%d")
    key = f"AWS4{secret_key}".encode()
    for part in (datestamp, region, "s3", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return now.strftime("%Y%m%dT%H%M%SZ"), f"{datestamp}/{region}/s3/aws4_request", key


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
        process cannot sign with), and the caller keeps serving bytes through Strata.
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
        prefix: str = "strata_blob_",
    ) -> Iterator[BinaryIO]:
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
        self.blobs_dir = blobs_dir
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    def _blob_path(self, artifact_id: str, version: int) -> Path:
        """Get filesystem path for a blob."""
        return self.blobs_dir / self._blob_key(artifact_id, version)

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
        def _writer() -> Iterator[BinaryIO]:
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
        if not path.exists():
            return False
        path.unlink()
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


class S3BlobStore(BlobStore):
    """S3 or S3-compatible (MinIO, LocalStack) blob storage via PyArrow's S3FileSystem.

    Keys are ``{prefix}/{artifact_id}@v={version}.arrow``.
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

        Without ``access_key``/``secret_key`` credentials come from the environment or
        IAM; ``endpoint_url`` targets an S3-compatible service.
        """
        import pyarrow.fs as pafs

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        # Kept for presigning, which PyArrow's filesystem does not expose.
        self._region = region
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key
        self._anonymous = anonymous

        kwargs = {}
        if region:
            kwargs["region"] = region
        if endpoint_url:
            kwargs["endpoint_override"] = endpoint_url
        if access_key and secret_key:
            kwargs["access_key"] = access_key
            kwargs["secret_key"] = secret_key
        if anonymous:
            kwargs["anonymous"] = True

        self._fs = pafs.S3FileSystem(**kwargs)

    def _object_key(self, artifact_id: str, version: int) -> str:
        blob_key = self._blob_key(artifact_id, version)
        return f"{self.prefix}/{blob_key}" if self.prefix else blob_key

    def _signing_credentials(self) -> tuple[str, str, str | None] | None:
        """Return the keys to sign with: the configured pair, the environment, else a role.

        PyArrow keeps the credentials it resolves to itself, so a role (instance profile,
        ECS task, web identity) is resolved again here through botocore.
        """
        if self._anonymous:
            return None
        if self._access_key and self._secret_key:
            return self._access_key, self._secret_key, None
        access = os.environ.get("AWS_ACCESS_KEY_ID")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if access and secret:
            return access, secret, os.environ.get("AWS_SESSION_TOKEN")
        credentials = self._role_credentials
        if credentials is None:
            return None
        # Temporary credentials: frozen refreshes them when they near expiry.
        frozen = credentials.get_frozen_credentials()
        return frozen.access_key, frozen.secret_key, frozen.token

    @cached_property
    def _role_credentials(self) -> Any:
        """botocore's credential chain, resolved once (a miss probes instance metadata)."""
        try:
            import botocore.session
        except ImportError:
            logger.warning(
                "S3 presigning with role credentials needs botocore: "
                "pip install 'strata-notebook[s3]'; keeping Strata URLs"
            )
            return None
        return botocore.session.get_session().get_credentials()

    def _signing_region(self) -> str:
        return (
            self._region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )

    def _object_location(self, key: str) -> tuple[str, str, str]:
        """Return ``(scheme, host, path)``: path-style on custom endpoints, else virtual-hosted."""
        from urllib.parse import quote, urlsplit

        quoted = quote(key, safe="/-_.~")
        if self._endpoint_url:
            endpoint = (
                self._endpoint_url
                if "://" in self._endpoint_url
                else f"https://{self._endpoint_url}"
            )
            parts = urlsplit(endpoint)
            return parts.scheme, parts.netloc, f"/{self.bucket}/{quoted}"
        return "https", f"{self.bucket}.s3.{self._signing_region()}.amazonaws.com", f"/{quoted}"

    def presign_get(self, artifact_id: str, version: int, ttl_seconds: int) -> str | None:
        from urllib.parse import quote

        credentials = self._signing_credentials()
        if credentials is None:
            return None
        access_key, secret_key, token = credentials
        scheme, host, path = self._object_location(self._object_key(artifact_id, version))
        amz_date, scope, signing_key = _sigv4_context(secret_key, self._signing_region())
        params = {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": f"{access_key}/{scope}",
            "X-Amz-Date": amz_date,
            "X-Amz-Expires": str(int(ttl_seconds)),
            "X-Amz-SignedHeaders": "host",
        }
        if token:
            params["X-Amz-Security-Token"] = token
        query = "&".join(
            f"{quote(k, safe='-_.~')}={quote(v, safe='-_.~')}" for k, v in sorted(params.items())
        )
        canonical_request = f"GET\n{path}\n{query}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD"
        string_to_sign = (
            f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n"
            f"{hashlib.sha256(canonical_request.encode()).hexdigest()}"
        )
        signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
        return f"{scheme}://{host}{path}?{query}&X-Amz-Signature={signature}"

    def presign_post(
        self, artifact_id: str, version: int, max_bytes: int, ttl_seconds: int
    ) -> tuple[str, dict[str, str]] | None:
        import base64
        import json
        from datetime import UTC, datetime, timedelta

        credentials = self._signing_credentials()
        if credentials is None:
            return None
        access_key, secret_key, token = credentials
        key = self._object_key(artifact_id, version)
        scheme, host, _ = self._object_location(key)
        url = f"{scheme}://{host}/{self.bucket}" if self._endpoint_url else f"{scheme}://{host}"
        amz_date, scope, signing_key = _sigv4_context(secret_key, self._signing_region())
        fields = {
            "key": key,
            "x-amz-algorithm": "AWS4-HMAC-SHA256",
            "x-amz-credential": f"{access_key}/{scope}",
            "x-amz-date": amz_date,
        }
        if token:
            fields["x-amz-security-token"] = token
        expiration = datetime.now(UTC) + timedelta(seconds=int(ttl_seconds))
        policy = {
            "expiration": expiration.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conditions": [
                {"bucket": self.bucket},
                *({name: value} for name, value in fields.items()),
                ["content-length-range", 1, int(max_bytes)],
            ],
        }
        encoded_policy = base64.b64encode(json.dumps(policy).encode()).decode()
        fields["policy"] = encoded_policy
        fields["x-amz-signature"] = hmac.new(
            signing_key, encoded_policy.encode(), hashlib.sha256
        ).hexdigest()
        return url, fields

    def _s3_key(self, artifact_id: str, version: int) -> str:
        """Get full S3 key for a blob."""
        blob_key = self._blob_key(artifact_id, version)
        if self.prefix:
            return f"{self.bucket}/{self.prefix}/{blob_key}"
        return f"{self.bucket}/{blob_key}"

    def open_blob_reader(
        self, artifact_id: str, version: int
    ) -> AbstractContextManager[BinaryIO] | None:
        """Open a streaming reader for an S3 blob."""
        import pyarrow as pa

        key = self._s3_key(artifact_id, version)
        try:
            return self._fs.open_input_stream(key)
        except (FileNotFoundError, pa.ArrowIOError):
            return None

    def _upload_from_path(self, key: str, source_path: Path) -> None:
        """Stream a local file to ``key`` via the PyArrow S3 filesystem."""
        with open(source_path, "rb") as src, self._fs.open_output_stream(key) as dst:
            while True:
                chunk = src.read(BLOB_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)

    def open_blob_writer(self, artifact_id: str, version: int) -> AbstractContextManager[BinaryIO]:
        """Open a streaming writer for an S3 blob, staged locally so failed writes never upload."""
        key = self._s3_key(artifact_id, version)
        return self._staged_local_writer(
            lambda staged: self._upload_from_path(key, staged),
            prefix="strata_s3_blob_",
        )

    def publish_blob_from_path(self, artifact_id: str, version: int, source_path: Path) -> None:
        """Stream ``source_path`` straight to S3, skipping the local staging copy."""
        key = self._s3_key(artifact_id, version)
        self._upload_from_path(key, source_path)

    def blob_exists(self, artifact_id: str, version: int) -> bool:
        """Check if blob exists in S3."""
        import pyarrow.fs as pafs

        key = self._s3_key(artifact_id, version)
        try:
            info = self._fs.get_file_info(key)
            return info.type == pafs.FileType.File
        except Exception:
            # A backend error is not "absent"; log it rather than report a
            # confident False.
            logger.exception(
                "blob_exists failed for %s@v=%d; reporting absent", artifact_id, version
            )
            return False

    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return blob size from S3 object metadata."""
        import pyarrow.fs as pafs

        key = self._s3_key(artifact_id, version)
        try:
            info = self._fs.get_file_info(key)
        except Exception:
            logger.exception(
                "blob_size failed for %s@v=%d; reporting unknown", artifact_id, version
            )
            return None
        if info.type != pafs.FileType.File:
            return None
        return int(info.size) if info.size is not None else None

    def delete_blob(self, artifact_id: str, version: int) -> bool:
        """Delete blob from S3."""
        key = self._s3_key(artifact_id, version)
        try:
            if not self.blob_exists(artifact_id, version):
                return False
            self._fs.delete_file(key)
            return True
        except Exception:
            # The caller removes the metadata row regardless, so a silent False
            # would orphan the object with nothing left to retry it.
            logger.exception(
                "delete_blob failed for %s@v=%d; object orphaned", artifact_id, version
            )
            return False

    @classmethod
    def from_config(
        cls, config: StrataConfig, bucket: str, prefix: str = "artifacts"
    ) -> S3BlobStore:
        """Create an S3BlobStore from Strata configuration's S3 settings."""
        return cls(
            bucket=bucket,
            prefix=prefix,
            region=config.s3_region,
            endpoint_url=config.s3_endpoint_url,
            access_key=config.s3_access_key,
            secret_key=config.s3_secret_key,
            anonymous=config.s3_anonymous,
        )


def azure_account_url(endpoint_url: str, account_name: str) -> str:
    """The account's blob URL: the endpoint host with the account as its first path segment.

    An endpoint that already ends in ``/<account>`` is taken as that URL, not doubled.
    """
    endpoint = endpoint_url.rstrip("/")
    if endpoint.endswith(f"/{account_name}"):
        return endpoint
    return f"{endpoint}/{account_name}"


def _resolve_gcs_credentials(credentials: str) -> str:
    """Return a filesystem path for *credentials*, writing inline JSON key material to a file.

    ``GOOGLE_APPLICATION_CREDENTIALS`` accepts only a path. A value that parses as
    a JSON object is spilled to a 0600 file named by the key's digest (so a
    hard-killed process leaves one file, rewritten by the next start) and removed
    at normal exit; anything else is passed through as a path. On Windows the 0600
    does not apply; prefer a mounted credential file there.
    """
    import atexit
    import json

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
    # below fixes a file left behind by a previous run.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)
    atexit.register(lambda: path.unlink(missing_ok=True))
    return str(path)


class GCSBlobStore(BlobStore):
    """Google Cloud Storage blob storage via PyArrow's GcsFileSystem.

    Keys are ``{prefix}/{artifact_id}@v={version}.arrow``. Supports Application
    Default Credentials, service-account keys, or anonymous access.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "artifacts",
        default_bucket_location: str | None = None,
        credentials_json: str | None = None,
        anonymous: bool = False,
        endpoint_override: str | None = None,
    ):
        """Initialize the GCS store.

        ``default_bucket_location`` is a location for new buckets (``US``,
        ``europe-west1``), not a project id. ``credentials_json`` is a key file path
        or the JSON itself.
        """
        import pyarrow.fs as pafs

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._anonymous = anonymous
        self._endpoint_override = endpoint_override

        kwargs = {}
        if default_bucket_location:
            kwargs["default_bucket_location"] = default_bucket_location
        if credentials_json:
            kwargs["access_token"] = None  # Disable token auth
            # GOOGLE_APPLICATION_CREDENTIALS must be a file path, but operators
            # paste key material into ...CREDENTIALS_JSON (an env var in most
            # containers). Spill it to a private file so both readings work.
            import os

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _resolve_gcs_credentials(
                credentials_json
            )
        if anonymous:
            kwargs["anonymous"] = True
        if endpoint_override:
            kwargs["endpoint_override"] = endpoint_override

        self._fs = pafs.GcsFileSystem(**kwargs)

    def _gcs_key(self, artifact_id: str, version: int) -> str:
        """Get full GCS key for a blob."""
        blob_key = self._blob_key(artifact_id, version)
        if self.prefix:
            return f"{self.bucket}/{self.prefix}/{blob_key}"
        return f"{self.bucket}/{blob_key}"

    def _object_name(self, artifact_id: str, version: int) -> str:
        return self._gcs_key(artifact_id, version).removeprefix(f"{self.bucket}/")

    @cached_property
    def _signing_client(self) -> Any:
        """A storage client over the ambient credentials, or ``None`` if there are none."""
        if self._anonymous:
            return None
        try:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError
            from google.cloud import storage
        except ImportError:
            logger.warning(
                "GCS presigning needs google-cloud-storage: "
                "pip install 'strata-notebook[gcs]'; keeping Strata URLs"
            )
            return None
        try:
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except DefaultCredentialsError:
            return None
        options = {"api_endpoint": self._endpoint_override} if self._endpoint_override else None
        return storage.Client(project="-", credentials=credentials, client_options=options)

    def _signing_kwargs(self) -> dict[str, str] | None:
        """How to sign: locally with a key (``{}``), through IAM signBlob, or ``None``."""
        from google.auth.credentials import Signing

        if self._signing_client is None:
            return None
        credentials = self._signing_client._credentials
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
        blob = self._signing_client.bucket(self.bucket).blob(
            self._object_name(artifact_id, version)
        )
        return blob.generate_signed_url(
            version="v4", expiration=timedelta(seconds=int(ttl_seconds)), method="GET", **signing
        )

    def presign_post(
        self, artifact_id: str, version: int, max_bytes: int, ttl_seconds: int
    ) -> tuple[str, dict[str, str]] | None:
        from datetime import timedelta

        signing = self._signing_kwargs()
        if signing is None:
            return None
        policy = self._signing_client.generate_signed_post_policy_v4(
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
        """Open a streaming reader for a GCS blob."""
        import pyarrow as pa

        key = self._gcs_key(artifact_id, version)
        try:
            return self._fs.open_input_stream(key)
        except (FileNotFoundError, pa.ArrowIOError):
            return None

    def _upload_from_path(self, key: str, source_path: Path) -> None:
        """Stream a local file to ``key`` via the PyArrow GCS filesystem."""
        with open(source_path, "rb") as src, self._fs.open_output_stream(key) as dst:
            while True:
                chunk = src.read(BLOB_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)

    def open_blob_writer(self, artifact_id: str, version: int) -> AbstractContextManager[BinaryIO]:
        """Open a streaming writer for a GCS blob, staged locally so failed writes never upload."""
        key = self._gcs_key(artifact_id, version)
        return self._staged_local_writer(
            lambda staged: self._upload_from_path(key, staged),
            prefix="strata_gcs_blob_",
        )

    def publish_blob_from_path(self, artifact_id: str, version: int, source_path: Path) -> None:
        """Stream ``source_path`` straight to GCS, skipping the local staging copy."""
        key = self._gcs_key(artifact_id, version)
        self._upload_from_path(key, source_path)

    def blob_exists(self, artifact_id: str, version: int) -> bool:
        """Check if blob exists in GCS."""
        import pyarrow.fs as pafs

        key = self._gcs_key(artifact_id, version)
        try:
            info = self._fs.get_file_info(key)
            return info.type == pafs.FileType.File
        except Exception:
            return False

    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return blob size from GCS object metadata."""
        import pyarrow.fs as pafs

        key = self._gcs_key(artifact_id, version)
        try:
            info = self._fs.get_file_info(key)
        except Exception:
            return None
        if info.type != pafs.FileType.File:
            return None
        return int(info.size) if info.size is not None else None

    def delete_blob(self, artifact_id: str, version: int) -> bool:
        """Delete blob from GCS."""
        key = self._gcs_key(artifact_id, version)
        try:
            if not self.blob_exists(artifact_id, version):
                return False
            self._fs.delete_file(key)
            return True
        except Exception:
            # The caller removes the metadata row regardless, so a silent False
            # would orphan the object with nothing left to retry it.
            logger.exception(
                "delete_blob failed for %s@v=%d; object orphaned", artifact_id, version
            )
            return False

    @classmethod
    def from_config(
        cls, config: StrataConfig, bucket: str, prefix: str = "artifacts"
    ) -> GCSBlobStore:
        """Create a GCSBlobStore from Strata configuration's GCS settings."""
        return cls(
            bucket=bucket,
            prefix=prefix,
            default_bucket_location=config.gcs_default_bucket_location,
            credentials_json=config.gcs_credentials_json,
            anonymous=config.gcs_anonymous,
            endpoint_override=config.gcs_endpoint_override,
        )


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
                "Install with: pip install strata-notebook[azure]"
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
        from azure.core.exceptions import ResourceNotFoundError

        key = self._azure_key(artifact_id, version)
        blob_client = self._client.get_blob_client(key)
        try:
            downloader = blob_client.download_blob()
        except ResourceNotFoundError:
            return None

        @contextmanager
        def _reader() -> Iterator[BinaryIO]:
            stream = io.BufferedReader(_AzureDownloadReader(downloader))
            try:
                yield stream
            finally:
                stream.close()

        return _reader()

    def _upload_from_path(self, key: str, source_path: Path) -> None:
        """Stream a local file to ``key`` via the Azure SDK."""
        blob_client = self._client.get_blob_client(key)
        with open(source_path, "rb") as src:
            blob_client.upload_blob(src, overwrite=True)

    def open_blob_writer(self, artifact_id: str, version: int) -> AbstractContextManager[BinaryIO]:
        """Open a streaming writer for an Azure blob, uploaded on clean exit.

        Staged through a local tempfile because the SDK streams uploads from a file
        handle, not from a caller-written stream.
        """
        key = self._azure_key(artifact_id, version)
        return self._staged_local_writer(
            lambda staged: self._upload_from_path(key, staged),
            prefix="strata_azure_blob_",
        )

    def publish_blob_from_path(self, artifact_id: str, version: int, source_path: Path) -> None:
        """Stream ``source_path`` straight to Azure, skipping the local staging copy."""
        key = self._azure_key(artifact_id, version)
        self._upload_from_path(key, source_path)

    def blob_exists(self, artifact_id: str, version: int) -> bool:
        """Check if blob exists in Azure Blob Storage."""
        key = self._azure_key(artifact_id, version)
        blob_client = self._client.get_blob_client(key)
        return blob_client.exists()

    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return blob size from Azure Blob Storage metadata."""
        from azure.core.exceptions import ResourceNotFoundError

        key = self._azure_key(artifact_id, version)
        blob_client = self._client.get_blob_client(key)
        try:
            props = blob_client.get_blob_properties()
        except ResourceNotFoundError:
            return None
        size = getattr(props, "size", None)
        return int(size) if size is not None else None

    def delete_blob(self, artifact_id: str, version: int) -> bool:
        """Delete blob from Azure Blob Storage."""
        from azure.core.exceptions import ResourceNotFoundError

        key = self._azure_key(artifact_id, version)
        blob_client = self._client.get_blob_client(key)
        try:
            blob_client.delete_blob()
            return True
        except ResourceNotFoundError:
            return False

    @classmethod
    def from_config(
        cls, config: StrataConfig, container_name: str, prefix: str = "artifacts"
    ) -> AzureBlobStore:
        """Create an AzureBlobStore from Strata configuration's Azure settings."""
        return cls(
            account_name=config.azure_account_name or "",
            container_name=container_name,
            prefix=prefix,
            account_key=config.azure_account_key,
            connection_string=config.azure_connection_string,
            sas_token=config.azure_sas_token,
            use_default_credential=config.azure_use_default_credential,
            endpoint_url=config.azure_endpoint_url,
        )


def create_blob_store(config: StrataConfig) -> BlobStore:
    """Create a blob store from the ``STRATA_ARTIFACT_*`` environment variables only.

    .. warning::

       ``config``'s own backend fields are ignored (it supplies credentials only),
       so a backend set in ``pyproject.toml`` silently falls back to local disk.
       The server uses :meth:`StrataConfig.create_blob_store` instead; prefer it.

    Reads ``STRATA_ARTIFACT_BLOB_BACKEND`` (``local``, ``s3``, ``gcs``, ``azure``)
    and the matching ``*_BUCKET``/``*_CONTAINER`` and ``*_PREFIX`` variables.

    Raises:
        ValueError: If required configuration is missing.
    """
    backend = os.environ.get("STRATA_ARTIFACT_BLOB_BACKEND", "local").lower()

    if backend == "s3":
        bucket = os.environ.get("STRATA_ARTIFACT_S3_BUCKET")
        if not bucket:
            raise ValueError(
                "S3 blob backend requires STRATA_ARTIFACT_S3_BUCKET environment variable"
            )
        prefix = os.environ.get("STRATA_ARTIFACT_S3_PREFIX", "artifacts")
        return S3BlobStore.from_config(config, bucket=bucket, prefix=prefix)

    if backend == "gcs":
        bucket = os.environ.get("STRATA_ARTIFACT_GCS_BUCKET")
        if not bucket:
            raise ValueError(
                "GCS blob backend requires STRATA_ARTIFACT_GCS_BUCKET environment variable"
            )
        prefix = os.environ.get("STRATA_ARTIFACT_GCS_PREFIX", "artifacts")
        return GCSBlobStore.from_config(config, bucket=bucket, prefix=prefix)

    if backend == "azure":
        container = os.environ.get("STRATA_ARTIFACT_AZURE_CONTAINER")
        if not container:
            raise ValueError(
                "Azure blob backend requires STRATA_ARTIFACT_AZURE_CONTAINER environment variable"
            )
        prefix = os.environ.get("STRATA_ARTIFACT_AZURE_PREFIX", "artifacts")
        return AzureBlobStore.from_config(config, container_name=container, prefix=prefix)

    if config.artifact_dir is None:
        raise ValueError("Local blob store requires artifact_dir in configuration")
    blobs_dir = config.artifact_dir / "blobs"
    return LocalBlobStore(blobs_dir)
