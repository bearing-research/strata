"""Blob storage abstraction for artifact data.

This module provides a protocol for blob storage, enabling artifact data
to be stored on different backends (local filesystem, S3, GCS, etc.).

The BlobStore protocol exposes streaming and convenience operations:

- ``open_blob_reader`` / ``open_blob_writer`` — chunked streaming I/O
- ``write_blob`` / ``read_blob`` — bytes-in, bytes-out convenience wrappers
- ``blob_exists`` / ``delete_blob`` — metadata ops

Implementations:
- LocalBlobStore: Local filesystem storage (default)
- S3BlobStore: Amazon S3 / S3-compatible storage (MinIO, LocalStack)
- GCSBlobStore: Google Cloud Storage
- AzureBlobStore: Azure Blob Storage
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

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


class BlobStore(ABC):
    """Abstract base class for blob storage backends.

    Blob stores persist artifact data (Arrow IPC files) separately from
    metadata (SQLite). This separation enables:
    - Horizontal scaling of blob storage (S3/GCS)
    - Efficient data locality in distributed setups
    - Independent lifecycle management of data vs metadata

    All blob operations use (artifact_id, version) as the key, which maps
    to a unique blob path/key in the underlying storage.
    """

    @abstractmethod
    def open_blob_reader(
        self, artifact_id: str, version: int
    ) -> AbstractContextManager[BinaryIO] | None:
        """Open a streaming reader for a blob.

        Returns ``None`` if the blob does not exist. Otherwise returns a
        context manager that yields a binary file-like object supporting
        at least ``read(size)``. Implementations may provide additional
        file-like methods (``seek``, iteration) but callers should only
        rely on ``read(size)``.

        Args:
            artifact_id: Unique artifact identifier
            version: Artifact version number
        """
        ...

    @abstractmethod
    def open_blob_writer(self, artifact_id: str, version: int) -> AbstractContextManager[BinaryIO]:
        """Open a streaming writer for a blob.

        The caller writes to the yielded handle. On clean context exit
        the blob is committed atomically; on exception the partial write
        is discarded.

        Args:
            artifact_id: Unique artifact identifier
            version: Artifact version number
        """
        ...

    def write_blob(self, artifact_id: str, version: int, data: bytes) -> None:
        """Write artifact data to storage (bytes convenience wrapper).

        Implementations should ensure atomic writes where possible (write to
        temp location then rename) to prevent partial writes on failure.

        Args:
            artifact_id: Unique artifact identifier
            version: Artifact version number
            data: Arrow IPC stream bytes to store
        """
        with self.open_blob_writer(artifact_id, version) as writer:
            writer.write(data)

    def read_blob(self, artifact_id: str, version: int) -> bytes | None:
        """Read artifact data from storage (bytes convenience wrapper).

        Args:
            artifact_id: Unique artifact identifier
            version: Artifact version number

        Returns:
            Arrow IPC stream bytes, or None if blob doesn't exist
        """
        reader = self.open_blob_reader(artifact_id, version)
        if reader is None:
            return None
        with reader as f:
            return f.read()

    @abstractmethod
    def blob_exists(self, artifact_id: str, version: int) -> bool:
        """Check if a blob exists in storage.

        Args:
            artifact_id: Unique artifact identifier
            version: Artifact version number

        Returns:
            True if the blob exists, False otherwise
        """
        ...

    @abstractmethod
    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return the size of a blob in bytes without materializing it.

        Returns ``None`` if the blob does not exist.
        """
        ...

    def publish_blob_from_path(self, artifact_id: str, version: int, source_path: Path) -> None:
        """Atomically publish a blob from a prepared local file.

        Intended for callers that already have the full payload on disk
        (e.g. an async request handler that spooled the upload into a
        tempfile). The default implementation pipes ``source_path``
        through ``open_blob_writer`` so it inherits the same atomic
        semantics. Backends may override to upload directly and skip the
        intermediate staging copy.

        The source file is not consumed — the caller retains ownership
        and is responsible for removing it.
        """
        with open(source_path, "rb") as src, self.open_blob_writer(artifact_id, version) as dst:
            while True:
                chunk = src.read(BLOB_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)

    @abstractmethod
    def delete_blob(self, artifact_id: str, version: int) -> bool:
        """Delete a blob from storage.

        Args:
            artifact_id: Unique artifact identifier
            version: Artifact version number

        Returns:
            True if blob was deleted, False if it didn't exist
        """
        ...

    def _blob_key(self, artifact_id: str, version: int) -> str:
        """Generate storage key for a blob.

        Default key format: {artifact_id}@v={version}.arrow

        Args:
            artifact_id: Unique artifact identifier
            version: Artifact version number

        Returns:
            Storage key string
        """
        return f"{artifact_id}@v={version}.arrow"

    @staticmethod
    @contextmanager
    def _staged_local_writer(
        commit: Callable[[Path], None],
        *,
        prefix: str = "strata_blob_",
    ) -> Iterator[BinaryIO]:
        """Stage writes through a local tempfile and commit atomically.

        Yields a binary handle for the caller to write into. On clean
        context exit the handle is flushed and **closed**, then the
        tempfile is passed to ``commit(path)`` which should publish the
        blob at its final location. On exception the tempfile is removed
        and commit is never invoked, guaranteeing that a partial write
        is not observable.

        The handle is closed before ``commit`` runs so that backends may
        safely reopen the tempfile on Windows, where open write handles
        block concurrent reads of the same path.
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
    """Local filesystem blob storage.

    Stores blobs in a directory structure:
        {blobs_dir}/{artifact_id}@v={version}.arrow

    Uses atomic writes (write to .tmp, then rename) to prevent partial writes.
    """

    def __init__(self, blobs_dir: Path):
        """Initialize local blob store.

        Args:
            blobs_dir: Directory for storing blob files
        """
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


class S3BlobStore(BlobStore):
    """Amazon S3 / S3-compatible blob storage.

    Stores blobs in an S3 bucket with key structure:
        {prefix}/{artifact_id}@v={version}.arrow

    Uses PyArrow's S3FileSystem for efficient Arrow IPC I/O.
    Supports S3-compatible services like MinIO, LocalStack, etc.
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
        """Initialize S3 blob store.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix within bucket (default: "artifacts")
            region: AWS region (e.g., "us-east-1")
            endpoint_url: Custom endpoint for S3-compatible services
            access_key: AWS access key ID (or use env/IAM)
            secret_key: AWS secret access key (or use env/IAM)
            anonymous: Use anonymous access for public buckets
        """
        import pyarrow.fs as pafs

        self.bucket = bucket
        self.prefix = prefix.strip("/")

        # Build S3FileSystem
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
        """Open a streaming writer for an S3 blob.

        Stages writes through a local tempfile so that a failed context
        exit discards the data before anything is pushed to S3, honoring
        the ``BlobStore.open_blob_writer`` atomicity contract.
        """
        key = self._s3_key(artifact_id, version)
        return self._staged_local_writer(
            lambda staged: self._upload_from_path(key, staged),
            prefix="strata_s3_blob_",
        )

    def publish_blob_from_path(self, artifact_id: str, version: int, source_path: Path) -> None:
        """Stream ``source_path`` straight to S3, skipping the local staging copy.

        The default implementation in the base class pipes through
        ``open_blob_writer`` which would stage the already-staged source
        file through another local tempfile. Since the caller has already
        produced a complete local payload, we can upload it directly and
        save one disk-to-disk copy per upload.
        """
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
            return False

    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return blob size from S3 object metadata."""
        import pyarrow.fs as pafs

        key = self._s3_key(artifact_id, version)
        try:
            info = self._fs.get_file_info(key)
        except Exception:
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
            return False

    @classmethod
    def from_config(
        cls, config: StrataConfig, bucket: str, prefix: str = "artifacts"
    ) -> S3BlobStore:
        """Create S3BlobStore from Strata configuration.

        Args:
            config: Strata configuration with S3 settings
            bucket: S3 bucket name
            prefix: Key prefix within bucket

        Returns:
            Configured S3BlobStore instance
        """
        return cls(
            bucket=bucket,
            prefix=prefix,
            region=config.s3_region,
            endpoint_url=config.s3_endpoint_url,
            access_key=config.s3_access_key,
            secret_key=config.s3_secret_key,
            anonymous=config.s3_anonymous,
        )


class GCSBlobStore(BlobStore):
    """Google Cloud Storage blob storage.

    Stores blobs in a GCS bucket with key structure:
        {prefix}/{artifact_id}@v={version}.arrow

    Uses PyArrow's GcsFileSystem for efficient Arrow IPC I/O.
    Supports Application Default Credentials, service account keys, or anonymous access.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "artifacts",
        project_id: str | None = None,
        credentials_json: str | None = None,
        anonymous: bool = False,
        endpoint_override: str | None = None,
    ):
        """Initialize GCS blob store.

        Args:
            bucket: GCS bucket name
            prefix: Key prefix within bucket (default: "artifacts")
            project_id: GCP project ID (uses default if not specified)
            credentials_json: Path to service account JSON key file
            anonymous: Use anonymous access for public buckets
            endpoint_override: Custom endpoint for GCS-compatible services (e.g., fake-gcs-server)
        """
        import pyarrow.fs as pafs

        self.bucket = bucket
        self.prefix = prefix.strip("/")

        # Build GcsFileSystem
        kwargs = {}
        if project_id:
            kwargs["default_bucket_location"] = project_id
        if credentials_json:
            kwargs["access_token"] = None  # Disable token auth
            # GcsFileSystem uses GOOGLE_APPLICATION_CREDENTIALS env var
            # or explicit credentials via access_token
            # For service account key files, set the env var before creating
            import os

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_json
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
        """Open a streaming writer for a GCS blob.

        Stages writes through a local tempfile so that a failed context
        exit discards the data before anything is pushed to GCS, honoring
        the ``BlobStore.open_blob_writer`` atomicity contract.
        """
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
            return False

    @classmethod
    def from_config(
        cls, config: StrataConfig, bucket: str, prefix: str = "artifacts"
    ) -> GCSBlobStore:
        """Create GCSBlobStore from Strata configuration.

        Args:
            config: Strata configuration with GCS settings
            bucket: GCS bucket name
            prefix: Key prefix within bucket

        Returns:
            Configured GCSBlobStore instance
        """
        return cls(
            bucket=bucket,
            prefix=prefix,
            project_id=config.gcs_project_id,
            credentials_json=config.gcs_credentials_json,
            anonymous=config.gcs_anonymous,
            endpoint_override=config.gcs_endpoint_override,
        )


class _AzureDownloadReader:
    """Minimal file-like wrapper over an Azure ``StorageStreamDownloader``."""

    def __init__(self, downloader: StorageStreamDownloader) -> None:
        self._downloader = downloader
        self._chunks: Iterator[bytes] = iter(downloader.chunks())
        self._buffer = b""
        self._closed = False

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise ValueError("read from closed Azure blob reader")
        if size < 0:
            parts = [self._buffer]
            self._buffer = b""
            for chunk in self._chunks:
                parts.append(chunk)
            return b"".join(parts)
        while len(self._buffer) < size:
            try:
                self._buffer += next(self._chunks)
            except StopIteration:
                break
        out, self._buffer = self._buffer[:size], self._buffer[size:]
        return out

    def close(self) -> None:
        self._closed = True


class AzureBlobStore(BlobStore):
    """Azure Blob Storage backend.

    Stores blobs in an Azure Storage container with key structure:
        {prefix}/{artifact_id}@v={version}.arrow

    Supports multiple authentication methods:
    - Connection string (easiest for local development)
    - Account key (account name + key)
    - SAS token (shared access signature)
    - DefaultAzureCredential (managed identity, environment vars, CLI, etc.)

    Requires the `azure` optional dependency:
        pip install strata-notebook[azure]
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
        """Initialize Azure Blob Storage backend.

        Args:
            account_name: Azure Storage account name
            container_name: Azure Blob container name
            prefix: Key prefix within container (default: "artifacts")
            account_key: Account access key (or use connection_string/sas_token)
            connection_string: Full connection string (alternative to account_key)
            sas_token: Shared Access Signature token (alternative to account_key)
            use_default_credential: Use Azure DefaultAzureCredential for auth
            endpoint_url: Custom endpoint URL (for Azurite emulator)
        """
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

        # Build the container client based on auth method
        if connection_string:
            self._client = ContainerClient.from_connection_string(
                conn_str=connection_string,
                container_name=container_name,
            )
        elif use_default_credential:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            account_url = endpoint_url or f"https://{account_name}.blob.core.windows.net"
            self._client = ContainerClient(
                account_url=account_url,
                container_name=container_name,
                credential=credential,
            )
        elif sas_token:
            account_url = endpoint_url or f"https://{account_name}.blob.core.windows.net"
            # SAS token can be passed as credential
            self._client = ContainerClient(
                account_url=account_url,
                container_name=container_name,
                credential=sas_token,
            )
        elif account_key:
            account_url = endpoint_url or f"https://{account_name}.blob.core.windows.net"
            self._client = ContainerClient(
                account_url=account_url,
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

    def open_blob_reader(
        self, artifact_id: str, version: int
    ) -> AbstractContextManager[BinaryIO] | None:
        """Open a streaming reader for an Azure blob.

        Wraps the SDK's ``StorageStreamDownloader`` chunk iterator in a
        minimal file-like so callers can use ``f.read(size)`` / iterate.
        """
        from azure.core.exceptions import ResourceNotFoundError

        key = self._azure_key(artifact_id, version)
        blob_client = self._client.get_blob_client(key)
        try:
            downloader = blob_client.download_blob()
        except ResourceNotFoundError:
            return None

        @contextmanager
        def _reader() -> Iterator[BinaryIO]:
            stream = _AzureDownloadReader(downloader)
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
        """Open a streaming writer for an Azure blob.

        Stages writes through a local tempfile and uploads via the SDK
        on clean context exit. The Azure SDK supports streaming uploads
        from a file handle but not from a caller-written stream, so the
        staging write keeps peak RAM at one chunk while still honoring
        the discard-on-exception contract.
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
        """Create AzureBlobStore from Strata configuration.

        Args:
            config: Strata configuration with Azure settings
            container_name: Azure Blob container name
            prefix: Key prefix within container

        Returns:
            Configured AzureBlobStore instance
        """
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
    """Create appropriate blob store based on configuration.

    Factory function that creates the correct blob store implementation
    based on configuration settings.

    Configuration options (in pyproject.toml or environment):
        # Local filesystem (default)
        artifact_dir = "/path/to/artifacts"

        # S3 storage
        artifact_blob_backend = "s3"
        artifact_s3_bucket = "my-bucket"
        artifact_s3_prefix = "artifacts"

        # GCS storage
        artifact_blob_backend = "gcs"
        artifact_gcs_bucket = "my-bucket"
        artifact_gcs_prefix = "artifacts"

        # Azure Blob Storage
        artifact_blob_backend = "azure"
        artifact_azure_container = "my-container"
        artifact_azure_prefix = "artifacts"

    Environment variables:
        STRATA_ARTIFACT_BLOB_BACKEND: "local" | "s3" | "gcs" | "azure"
        STRATA_ARTIFACT_S3_BUCKET: S3 bucket name
        STRATA_ARTIFACT_S3_PREFIX: Key prefix in bucket
        STRATA_ARTIFACT_GCS_BUCKET: GCS bucket name
        STRATA_ARTIFACT_GCS_PREFIX: Key prefix in bucket
        STRATA_ARTIFACT_AZURE_CONTAINER: Azure Blob container name
        STRATA_ARTIFACT_AZURE_PREFIX: Key prefix in container

    Args:
        config: Strata configuration

    Returns:
        Configured BlobStore instance

    Raises:
        ValueError: If required configuration is missing
    """
    # Check environment variables first
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

    # Default: local filesystem
    if config.artifact_dir is None:
        raise ValueError("Local blob store requires artifact_dir in configuration")
    blobs_dir = config.artifact_dir / "blobs"
    return LocalBlobStore(blobs_dir)
