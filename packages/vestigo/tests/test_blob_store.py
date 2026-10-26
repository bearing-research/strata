"""Tests for blob storage backends."""

import io
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vestigo import blob_store as blob_store_module
from vestigo.blob_store import (
    BlobStore,
    GCSBlobStore,
    LocalBlobStore,
    S3BlobStore,
    get_object_store_errors,
)


class TestLocalBlobStore:
    @pytest.mark.parametrize("artifact_id", ["../blobs/victim", "../../outside", "a/b"])
    def test_an_id_that_names_a_path_reaches_no_file(self, tmp_path: Path, artifact_id: str):
        """``../blobs/victim`` would be another artifact's bytes; ``../..`` leaves the store."""
        store = LocalBlobStore(tmp_path / "blobs")
        store.write_blob("victim", 1, b"theirs")

        with pytest.raises(ValueError, match="cannot name a path"):
            store.write_blob(artifact_id, 1, b"mine")
        with pytest.raises(ValueError, match="cannot name a path"):
            store.read_blob(artifact_id, 1)

        assert store.read_blob("victim", 1) == b"theirs"
        assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == ["victim@v=1.arrow"]

    def test_write_and_read_blob(self, tmp_path: Path):
        store = LocalBlobStore(tmp_path / "blobs")
        data = b"test artifact data"

        store.write_blob("artifact-1", 1, data)
        result = store.read_blob("artifact-1", 1)

        assert result == data

    def test_read_nonexistent_blob(self, tmp_path: Path):
        store = LocalBlobStore(tmp_path / "blobs")

        result = store.read_blob("nonexistent", 1)

        assert result is None

    def test_blob_key_is_case_collision_proof(self, tmp_path: Path):
        """Ids differing only in case must not collapse to one file.

        On a case-insensitive filesystem (macOS, Windows) ``..._var_Widget`` and ``..._var_widget``
        would share a blob. Asserted on the key so Linux CI catches it too.
        """
        store = LocalBlobStore(tmp_path / "blobs")
        k_upper = store._blob_key("nb_x_cell_c_var_Widget", 1)
        k_lower = store._blob_key("nb_x_cell_c_var_widget", 1)
        assert k_upper != k_lower
        # They must not collide under case folding.
        assert k_upper.lower() != k_lower.lower()
        # All-lowercase ids keep their stable name (cached blobs survive upgrades).
        assert k_lower == "nb_x_cell_c_var_widget@v=1.arrow"

    def test_case_differing_ids_round_trip_to_distinct_data(self, tmp_path: Path):
        """Case-differing ids read back distinct data (fails on a case-insensitive FS without the
        key fix).
        """
        store = LocalBlobStore(tmp_path / "blobs")
        store.write_blob("nb_var_Widget", 1, b"the class")
        store.write_blob("nb_var_widget", 1, b"the instance")
        assert store.read_blob("nb_var_Widget", 1) == b"the class"
        assert store.read_blob("nb_var_widget", 1) == b"the instance"

    def test_blob_exists(self, tmp_path: Path):
        store = LocalBlobStore(tmp_path / "blobs")
        data = b"test data"

        assert not store.blob_exists("artifact-1", 1)

        store.write_blob("artifact-1", 1, data)

        assert store.blob_exists("artifact-1", 1)
        assert not store.blob_exists("artifact-1", 2)

    def test_delete_blob(self, tmp_path: Path):
        store = LocalBlobStore(tmp_path / "blobs")
        data = b"test data"

        store.write_blob("artifact-1", 1, data)
        assert store.blob_exists("artifact-1", 1)

        result = store.delete_blob("artifact-1", 1)

        assert result is True
        assert not store.blob_exists("artifact-1", 1)

    def test_delete_nonexistent_blob(self, tmp_path: Path):
        store = LocalBlobStore(tmp_path / "blobs")

        result = store.delete_blob("nonexistent", 1)

        assert result is False

    def test_a_delete_that_loses_the_race_reports_absent(self, tmp_path: Path, monkeypatch):
        """Two sweeps on one blob dir: the loser gets False, as on the object stores."""
        store = LocalBlobStore(tmp_path / "blobs")
        store.write_blob("artifact-1", 1, b"x")
        real_unlink = Path.unlink

        def unlink_after_the_other_sweep(self, *args, **kwargs):
            os.unlink(self)  # the other sweep gets there first
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", unlink_after_the_other_sweep)

        assert store.delete_blob("artifact-1", 1) is False
        assert not store.blob_exists("artifact-1", 1)

    @pytest.mark.skipif(sys.platform == "win32", reason="directories cannot be fsynced there")
    def test_the_directory_is_fsynced_once_the_blob_is_in_place(self, tmp_path: Path, monkeypatch):
        """A durable file under a directory entry that is not is lost at power loss."""
        store = LocalBlobStore(tmp_path / "blobs")
        synced: list[tuple[bool, bool]] = []
        real_fsync = os.fsync

        def recording_fsync(fd):
            is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
            synced.append((is_dir, store.blob_exists("artifact-1", 1)))
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", recording_fsync)

        store.write_blob("artifact-1", 1, b"payload")

        # The file first (not yet renamed), then its directory with the blob in place.
        assert synced == [(False, False), (True, True)]

    def test_multiple_versions(self, tmp_path: Path):
        """Multiple versions of one artifact are stored separately."""
        store = LocalBlobStore(tmp_path / "blobs")

        store.write_blob("artifact-1", 1, b"version 1")
        store.write_blob("artifact-1", 2, b"version 2")
        store.write_blob("artifact-1", 3, b"version 3")

        assert store.read_blob("artifact-1", 1) == b"version 1"
        assert store.read_blob("artifact-1", 2) == b"version 2"
        assert store.read_blob("artifact-1", 3) == b"version 3"

    def test_blob_key_format(self, tmp_path: Path):
        store = LocalBlobStore(tmp_path / "blobs")

        key = store._blob_key("abc123", 5)

        assert key == "abc123@v=5.arrow"

    def test_creates_directory(self, tmp_path: Path):
        """The store creates the blobs directory."""
        blobs_dir = tmp_path / "new" / "nested" / "blobs"
        assert not blobs_dir.exists()

        LocalBlobStore(blobs_dir)

        assert blobs_dir.exists()

    def test_atomic_write(self, tmp_path: Path):
        """Writes are atomic: no partial files on failure."""
        store = LocalBlobStore(tmp_path / "blobs")
        data = b"test data" * 1000

        store.write_blob("artifact-1", 1, data)

        residual_tmp = list((tmp_path / "blobs").glob("*.tmp"))
        assert residual_tmp == []

        assert store.read_blob("artifact-1", 1) == data

    def test_streaming_writer_commits_atomically(self, tmp_path: Path):
        """open_blob_writer publishes the blob only on clean context exit."""
        store = LocalBlobStore(tmp_path / "blobs")

        with store.open_blob_writer("artifact-1", 1) as writer:
            writer.write(b"chunk1")
            writer.write(b"chunk2")
            assert not store.blob_exists("artifact-1", 1)

        assert store.read_blob("artifact-1", 1) == b"chunk1chunk2"

    def test_streaming_writer_discards_on_exception(self, tmp_path: Path):
        """A writer exiting with an exception leaves no blob or tmp file."""
        store = LocalBlobStore(tmp_path / "blobs")

        class BoomError(RuntimeError):
            pass

        with pytest.raises(BoomError):
            with store.open_blob_writer("artifact-1", 1) as writer:
                writer.write(b"partial")
                raise BoomError()

        assert not store.blob_exists("artifact-1", 1)
        assert list((tmp_path / "blobs").glob("*.tmp")) == []

    def test_streaming_reader_reads_chunks(self, tmp_path: Path):
        """open_blob_reader yields a file-like that streams chunks."""
        store = LocalBlobStore(tmp_path / "blobs")
        payload = b"abcdefghij" * 10
        store.write_blob("artifact-1", 1, payload)

        reader_cm = store.open_blob_reader("artifact-1", 1)
        assert reader_cm is not None

        collected = []
        with reader_cm as reader:
            while True:
                chunk = reader.read(17)
                if not chunk:
                    break
                collected.append(chunk)

        assert b"".join(collected) == payload

    def test_streaming_reader_missing_returns_none(self, tmp_path: Path):
        """open_blob_reader returns None for missing blobs."""
        store = LocalBlobStore(tmp_path / "blobs")
        assert store.open_blob_reader("absent", 1) is None

    def test_streaming_writer_preserves_existing_on_error(self, tmp_path: Path):
        """A failed writer context must not corrupt a previously committed blob."""
        store = LocalBlobStore(tmp_path / "blobs")
        store.write_blob("artifact-1", 1, b"original")

        with pytest.raises(RuntimeError):
            with store.open_blob_writer("artifact-1", 1) as writer:
                writer.write(b"garbage")
                raise RuntimeError("simulated failure")

        assert store.read_blob("artifact-1", 1) == b"original"

    def test_blob_size_reports_size_without_materializing(self, tmp_path: Path):
        """blob_size returns the length from filesystem metadata."""
        store = LocalBlobStore(tmp_path / "blobs")
        payload = b"x" * 12345
        store.write_blob("artifact-1", 1, payload)

        assert store.blob_size("artifact-1", 1) == len(payload)

    def test_blob_size_missing_returns_none(self, tmp_path: Path):
        """blob_size returns None for missing blobs."""
        store = LocalBlobStore(tmp_path / "blobs")
        assert store.blob_size("absent", 1) is None


class TestStagedLocalWriter:
    """``BlobStore._staged_local_writer``, behind the S3/GCS/Azure writers' atomicity.

    It stages to a local tempfile, calls ``commit(path)`` only on clean exit, and discards the
    tempfile otherwise.
    """

    def test_commit_invoked_on_clean_exit(self, tmp_path: Path):
        """``commit`` runs exactly once on clean exit, with a readable tempfile."""
        captured: dict[str, object] = {}

        def _commit(path: Path) -> None:
            captured["path"] = path
            captured["bytes"] = path.read_bytes()

        with BlobStore._staged_local_writer(_commit) as writer:
            writer.write(b"hello ")
            writer.write(b"world")

        assert captured["bytes"] == b"hello world"
        # Tempfile must be removed after commit completes.
        assert not Path(str(captured["path"])).exists()

    def test_commit_not_invoked_on_exception(self, tmp_path: Path):
        """``commit`` does not run if the writer context raises."""
        calls: list[Path] = []

        def _commit(path: Path) -> None:
            calls.append(path)

        class BoomError(RuntimeError):
            pass

        with pytest.raises(BoomError):
            with BlobStore._staged_local_writer(_commit) as writer:
                writer.write(b"partial")
                raise BoomError()

        assert calls == []

    def test_tempfile_discarded_on_exception(self, tmp_path: Path, monkeypatch):
        """The staged tempfile is removed even when commit never runs."""
        captured_path: list[Path] = []
        from vestigo import blob_store as blob_store_module

        real_mkstemp = blob_store_module.tempfile.mkstemp

        def _recording_mkstemp(*args, **kwargs):
            fd, name = real_mkstemp(*args, **kwargs)
            captured_path.append(Path(name))
            return fd, name

        monkeypatch.setattr(blob_store_module.tempfile, "mkstemp", _recording_mkstemp)

        class BoomError(RuntimeError):
            pass

        with pytest.raises(BoomError):
            with BlobStore._staged_local_writer(lambda _p: None) as writer:
                writer.write(b"partial")
                raise BoomError()

        assert captured_path, "staged tempfile was never created"
        assert not captured_path[0].exists()

    def test_commit_exception_propagates_and_cleans_up(self, tmp_path: Path, monkeypatch):
        """If commit raises, the tempfile is still removed."""
        captured_path: list[Path] = []
        from vestigo import blob_store as blob_store_module

        real_mkstemp = blob_store_module.tempfile.mkstemp

        def _recording_mkstemp(*args, **kwargs):
            fd, name = real_mkstemp(*args, **kwargs)
            captured_path.append(Path(name))
            return fd, name

        monkeypatch.setattr(blob_store_module.tempfile, "mkstemp", _recording_mkstemp)

        def _failing_commit(_path: Path) -> None:
            raise RuntimeError("remote commit rejected")

        with pytest.raises(RuntimeError, match="remote commit rejected"):
            with BlobStore._staged_local_writer(_failing_commit) as writer:
                writer.write(b"payload")

        assert captured_path and not captured_path[0].exists()

    def test_handle_is_closed_before_commit(self, tmp_path: Path):
        """The writer handle is closed before ``commit`` runs.

        Remote backends reopen the tempfile in ``commit``, which fails on Windows with
        ERROR_SHARING_VIOLATION while a write handle is open.
        """
        observed_state: dict[str, object] = {}

        with BlobStore._staged_local_writer(
            lambda _path: None, prefix="vestigo_handle_test_"
        ) as writer:
            observed_state["handle"] = writer

        def _commit(path: Path) -> None:
            observed_state["closed_before_commit"] = observed_state["handle"].closed  # type: ignore[union-attr]
            # Reopen the path: this exercises the same pattern that would
            # fail on Windows if the writer handle were still open.
            with open(path, "rb") as reread:
                observed_state["reread_len"] = len(reread.read())

        with BlobStore._staged_local_writer(_commit) as writer:
            writer.write(b"payload-bytes")
            observed_state["handle"] = writer

        assert observed_state["closed_before_commit"] is True
        assert observed_state["reread_len"] == len(b"payload-bytes")


class TestPublishBlobFromPath:
    def test_round_trip_via_local_backend(self, tmp_path: Path):
        """The default implementation publishes a staged file atomically."""
        store = LocalBlobStore(tmp_path / "blobs")
        staging = tmp_path / "incoming.bin"
        payload = b"hello " * 1000
        staging.write_bytes(payload)

        store.publish_blob_from_path("artifact-1", 1, staging)

        assert store.read_blob("artifact-1", 1) == payload
        # Source is not consumed; the caller owns its lifecycle.
        assert staging.exists()

    def test_publish_overwrites_existing_blob(self, tmp_path: Path):
        """Publishing a new file replaces the previously committed blob."""
        store = LocalBlobStore(tmp_path / "blobs")
        store.write_blob("artifact-1", 1, b"v1-bytes")

        staging = tmp_path / "incoming.bin"
        staging.write_bytes(b"v2-bytes")

        store.publish_blob_from_path("artifact-1", 1, staging)

        assert store.read_blob("artifact-1", 1) == b"v2-bytes"


def _client_error(status: int, code: str, operation: str):
    """A botocore ``ClientError`` as the SDK raises it for an HTTP ``status``."""
    from botocore.exceptions import ClientError

    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


@pytest.fixture
def s3_store():
    """An S3 store over moto's in-process S3, with its bucket made."""
    from moto import mock_aws

    with mock_aws():
        store = S3BlobStore(
            bucket="fake-bucket",
            prefix="artifacts",
            region="us-east-1",
            access_key="testing",
            secret_key="testing",
        )
        store._client.create_bucket(Bucket="fake-bucket")
        yield store


class TestS3BlobStore:
    """The S3 I/O paths over moto, and the SDK's errors where moto cannot produce them."""

    def test_object_key_format(self):
        store = S3BlobStore(bucket="test-bucket", prefix="artifacts", anonymous=True)

        assert store._object_key("abc123", 5) == "artifacts/abc123@v=5.arrow"

    def test_object_key_without_prefix(self):
        store = S3BlobStore(bucket="test-bucket", prefix="", anonymous=True)

        assert store._object_key("abc123", 5) == "abc123@v=5.arrow"

    @pytest.mark.parametrize("half", [{"access_key": "AKIA"}, {"secret_key": "s3cret"}])
    def test_half_a_key_pair_is_refused(self, half):
        # One key alone would fall back to ambient credentials, for I/O and signing both.
        with pytest.raises(ValueError, match="both access_key and secret_key"):
            S3BlobStore(bucket="test-bucket", **half)

    def test_a_custom_endpoint_is_a_url_addressed_path_style(self):
        store = S3BlobStore(bucket="b", endpoint_url="minio.internal:9000", anonymous=True)

        assert store._client.meta.endpoint_url == "https://minio.internal:9000"
        assert store._client.meta.config.s3 == {"addressing_style": "path"}

    def test_write_publish_exists_size_delete(self, s3_store, tmp_path: Path):
        store = s3_store

        assert store.read_blob("artifact-1", 1) is None
        assert store.blob_exists("artifact-1", 1) is False
        assert store.blob_size("artifact-1", 1) is None
        assert store.delete_blob("artifact-1", 1) is False

        store.write_blob("artifact-1", 1, b"streamed")
        assert store.read_blob("artifact-1", 1) == b"streamed"
        assert store.blob_exists("artifact-1", 1) is True
        assert store.blob_size("artifact-1", 1) == len(b"streamed")

        source = tmp_path / "incoming.bin"
        source.write_bytes(b"published")
        store.publish_blob_from_path("artifact-1", 1, source)
        assert store.read_blob("artifact-1", 1) == b"published"
        assert source.exists(), "the source is the caller's to remove"

        assert store.delete_blob("artifact-1", 1) is True
        assert store.blob_exists("artifact-1", 1) is False
        assert store.delete_blob("artifact-1", 1) is False

    def test_a_reader_streams_in_sizes_the_caller_asks_for(self, s3_store):
        s3_store.write_blob("artifact-1", 1, bytes(range(256)) * 1000)

        reader = s3_store.open_blob_reader("artifact-1", 1)
        assert reader is not None
        with reader as f:
            parts = [f.read(100000), f.read(100000), f.read(100000), f.read(100000)]

        assert b"".join(parts) == bytes(range(256)) * 1000
        assert parts[-1] == b""

    def test_a_blob_past_the_multipart_threshold_round_trips(self, s3_store, tmp_path: Path):
        """The SDK uploads 8 MiB and above in parts; the parts must come back as one object."""
        payload = os.urandom(1024) * (9 * 1024)  # 9 MiB
        source = tmp_path / "big.bin"
        source.write_bytes(payload)

        s3_store.publish_blob_from_path("big", 1, source)

        assert s3_store.blob_size("big", 1) == len(payload)
        assert s3_store.read_blob("big", 1) == payload

    def test_a_failed_writer_uploads_nothing(self, s3_store):
        with pytest.raises(RuntimeError, match="boom"):
            with s3_store.open_blob_writer("artifact-1", 1) as writer:
                writer.write(b"partial")
                raise RuntimeError("boom")

        assert s3_store.blob_exists("artifact-1", 1) is False

    def test_a_failed_upload_leaves_the_previous_object(self, s3_store, tmp_path: Path):
        """An upload lands whole or not at all; the object under the key is kept either way."""
        s3_store.write_blob("artifact-1", 1, b"the old bytes")
        source = tmp_path / "incoming.bin"
        source.write_bytes(b"the new bytes")

        def refused(fileobj, bucket, key, **kwargs):
            raise _client_error(500, "InternalError", "PutObject")

        s3_store._client.upload_fileobj = refused
        with pytest.raises(Exception, match="InternalError"):
            s3_store.publish_blob_from_path("artifact-1", 1, source)

        assert s3_store.read_blob("artifact-1", 1) == b"the old bytes"

    def test_a_failed_upload_cleans_up_the_staged_tempfile(self, s3_store, monkeypatch):
        captured: list[Path] = []
        real_mkstemp = blob_store_module.tempfile.mkstemp

        def recording_mkstemp(*args, **kwargs):
            fd, name = real_mkstemp(*args, **kwargs)
            captured.append(Path(name))
            return fd, name

        monkeypatch.setattr(blob_store_module.tempfile, "mkstemp", recording_mkstemp)

        def refused(fileobj, bucket, key, **kwargs):
            raise _client_error(503, "SlowDown", "PutObject")

        s3_store._client.upload_fileobj = refused
        with pytest.raises(Exception, match="SlowDown"):
            with s3_store.open_blob_writer("artifact-1", 1) as writer:
                writer.write(b"payload")

        assert captured and not captured[0].exists()

    def test_publish_from_path_skips_double_staging(self, s3_store, tmp_path: Path, monkeypatch):
        prefixes: list[str] = []
        real_mkstemp = blob_store_module.tempfile.mkstemp

        def counting_mkstemp(*args, **kwargs):
            prefixes.append(kwargs.get("prefix", ""))
            return real_mkstemp(*args, **kwargs)

        monkeypatch.setattr(blob_store_module.tempfile, "mkstemp", counting_mkstemp)
        source = tmp_path / "incoming.bin"
        source.write_bytes(b"abc123" * 500)

        s3_store.publish_blob_from_path("artifact-1", 1, source)

        assert not [p for p in prefixes if p.startswith("vestigo_s3_blob_")]
        assert s3_store.read_blob("artifact-1", 1) == b"abc123" * 500

    def test_a_backend_error_on_read_is_not_absent(self, s3_store):
        """ACCESS_DENIED, a bad region or a timeout used to read as None: an artifact whose
        blob is "gone", a 404, an integrity finding, all silent."""
        from botocore.exceptions import ClientError

        def denied(**kwargs):
            raise _client_error(403, "AccessDenied", "GetObject")

        s3_store._client.get_object = denied
        before = get_object_store_errors().snapshot()["get"]

        with pytest.raises(ClientError, match="AccessDenied"):
            s3_store.read_blob("artifact-1", 1)
        with pytest.raises(ClientError, match="AccessDenied"):
            s3_store.open_blob_reader("artifact-1", 1)

        assert get_object_store_errors().snapshot()["get"] == before + 2

    def test_exists_size_and_delete_propagate_a_backend_error(self, s3_store):
        """``delete_blob`` matters most: the caller removes the metadata row regardless, so a
        False that meant "could not reach the bucket" orphaned the object for good."""
        from botocore.exceptions import ClientError

        def denied(**kwargs):
            raise _client_error(403, "AccessDenied", "HeadObject")

        s3_store._client.head_object = denied
        before = get_object_store_errors().snapshot()["head"]

        for call in (s3_store.blob_exists, s3_store.blob_size, s3_store.delete_blob):
            with pytest.raises(ClientError, match="AccessDenied"):
                call("artifact-1", 1)

        assert get_object_store_errors().snapshot()["head"] == before + 3

    def test_a_missing_bucket_on_read_is_an_error(self, s3_store):
        """A 404 is not always a missing object: a misnamed bucket must not read as empty."""
        from botocore.exceptions import ClientError

        def no_bucket(**kwargs):
            raise _client_error(404, "NoSuchBucket", "GetObject")

        s3_store._client.get_object = no_bucket

        with pytest.raises(ClientError, match="NoSuchBucket"):
            s3_store.read_blob("artifact-1", 1)

    def test_an_absent_object_is_not_an_error(self, s3_store):
        before = get_object_store_errors().snapshot()

        assert s3_store.read_blob("missing", 1) is None
        assert s3_store.blob_size("missing", 1) is None

        assert get_object_store_errors().snapshot() == before

    def test_presigned_urls_come_from_the_client_s_own_credentials(self, s3_store):
        url = s3_store.presign_get("fig", 1, ttl_seconds=60)
        signed = s3_store.presign_post("out", 1, max_bytes=16, ttl_seconds=60)

        assert url is not None
        assert url.startswith("https://fake-bucket.s3.amazonaws.com/artifacts/fig%40v%3D1.arrow?")
        assert "X-Amz-Credential=testing%2F" in url
        assert signed is not None
        assert signed[1]["key"] == "artifacts/out@v=1.arrow"
        assert signed[1]["x-amz-credential"].startswith("testing/")

    def test_an_anonymous_store_does_not_presign(self):
        store = S3BlobStore(bucket="public", anonymous=True)

        assert store.presign_get("fig", 1, 60) is None
        assert store.presign_post("fig", 1, 16, 60) is None

    def test_a_store_without_credentials_does_not_presign(self):
        # The suite's conftest keeps the developer's AWS credentials out of the environment.
        store = S3BlobStore(bucket="b", region="us-east-1")

        assert store.presign_get("fig", 1, 60) is None
        assert store.presign_post("fig", 1, 16, 60) is None


class _FakeGcsBlob:
    """One object of a fake GCS bucket, over a local file, raising as the SDK does."""

    def __init__(self, path: Path, record: list[str], error: Exception | None):
        self._path = path
        self._record = record
        self._error = error
        self.size: int | None = None

    def _absent(self):
        from google.api_core.exceptions import NotFound

        return NotFound(f"{self._path.name} not found")

    def reload(self) -> None:
        if self._error is not None:
            raise self._error
        if not self._path.exists():
            raise self._absent()
        self.size = self._path.stat().st_size

    def open(self, mode: str, chunk_size: int | None = None):
        assert mode == "rb" and chunk_size is not None
        return open(self._path, "rb")

    def upload_from_file(self, src, size=None, retry=None) -> None:
        if self._error is not None:
            raise self._error
        self._record.append(f"upload:size={size}")
        self._path.write_bytes(src.read(size))

    def exists(self) -> bool:
        if self._error is not None:
            raise self._error
        return self._path.exists()

    def delete(self) -> None:
        if self._error is not None:
            raise self._error
        if not self._path.exists():
            raise self._absent()
        self._record.append("delete")
        self._path.unlink()


class _FakeGcsClient:
    """``client.bucket(name).blob(name)`` over a local directory.

    ``error`` is raised by every call, as an unreachable or forbidden bucket would.
    """

    def __init__(self, root: Path, error: Exception | None = None):
        self.root = root
        self.error = error
        self.record: list[str] = []

    def bucket(self, name: str):
        return SimpleNamespace(
            blob=lambda key: _FakeGcsBlob(
                self.root / key.replace("/", "_"), self.record, self.error
            )
        )


class TestGCSBlobStore:
    """The GCS I/O paths over a fake client, keyed like the real one."""

    def _store(self, tmp_path: Path, error: Exception | None = None) -> GCSBlobStore:
        store = GCSBlobStore.__new__(GCSBlobStore)
        store.bucket = "fake-bucket"
        store.prefix = "artifacts"
        store._anonymous = False
        store._client = _FakeGcsClient(tmp_path, error)
        return store

    def test_object_name_format(self):
        store = GCSBlobStore(bucket="test-bucket", prefix="artifacts", anonymous=True)

        assert store._object_name("abc123", 5) == "artifacts/abc123@v=5.arrow"

    def test_object_name_without_prefix(self):
        store = GCSBlobStore(bucket="test-bucket", prefix="", anonymous=True)

        assert store._object_name("abc123", 5) == "abc123@v=5.arrow"

    def test_an_emulator_endpoint_is_a_url(self):
        store = GCSBlobStore(bucket="b", anonymous=True, endpoint_override="localhost:4443")

        assert store._client._connection.API_BASE_URL == "https://localhost:4443"

    def test_a_workload_identity_credentials_file_is_accepted(self, tmp_path: Path):
        """The setting takes any file ``GOOGLE_APPLICATION_CREDENTIALS`` would, not only a key."""
        from google.auth.identity_pool import Credentials

        info = {
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "token_url": "https://sts.googleapis.com/v1/token",
            "credential_source": {"file": str(tmp_path / "token")},
        }
        key_file = tmp_path / "wif.json"
        key_file.write_text(json.dumps(info))

        for given in (json.dumps(info), str(key_file)):
            store = GCSBlobStore(bucket="bkt", credentials_json=given)
            assert isinstance(store._client._credentials, Credentials)

    def test_an_unknown_credentials_type_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported GCS credentials type 'oauth1'"):
            GCSBlobStore(bucket="bkt", credentials_json='{"type": "oauth1"}')

    def test_without_credentials_or_anonymous_there_is_no_store(self, monkeypatch):
        """A store that could neither read nor write fails at construction, not at first use."""
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        def no_credentials(scopes=None):
            raise DefaultCredentialsError("none")

        monkeypatch.setattr(google.auth, "default", no_credentials)

        with pytest.raises(DefaultCredentialsError):
            GCSBlobStore(bucket="bkt")

    def test_write_publish_exists_size_delete(self, tmp_path: Path):
        store = self._store(tmp_path)

        assert store.read_blob("artifact-1", 1) is None
        assert store.blob_exists("artifact-1", 1) is False
        assert store.blob_size("artifact-1", 1) is None
        assert store.delete_blob("artifact-1", 1) is False

        store.write_blob("artifact-1", 1, b"streamed")
        assert store.read_blob("artifact-1", 1) == b"streamed"
        assert store.blob_exists("artifact-1", 1) is True
        assert store.blob_size("artifact-1", 1) == len(b"streamed")
        assert (tmp_path / "artifacts_artifact-1@v=1.arrow").exists()

        source = tmp_path / "incoming.bin"
        source.write_bytes(b"published")
        store.publish_blob_from_path("artifact-1", 1, source)
        assert store.read_blob("artifact-1", 1) == b"published"
        assert store._client.record == ["upload:size=8", "upload:size=9"]

        assert store.delete_blob("artifact-1", 1) is True
        assert store.blob_exists("artifact-1", 1) is False
        assert store.delete_blob("artifact-1", 1) is False

    def test_a_failed_writer_uploads_nothing(self, tmp_path: Path):
        store = self._store(tmp_path)

        with pytest.raises(RuntimeError, match="boom"):
            with store.open_blob_writer("artifact-1", 1) as writer:
                writer.write(b"partial")
                raise RuntimeError("boom")

        assert store._client.record == []
        assert store.blob_exists("artifact-1", 1) is False

    def test_a_backend_error_on_read_is_not_absent(self, tmp_path: Path):
        from google.api_core.exceptions import Forbidden

        store = self._store(tmp_path, Forbidden("denied"))
        before = get_object_store_errors().snapshot()["get"]

        with pytest.raises(Forbidden, match="denied"):
            store.read_blob("artifact-1", 1)
        with pytest.raises(Forbidden, match="denied"):
            store.open_blob_reader("artifact-1", 1)

        assert get_object_store_errors().snapshot()["get"] == before + 2

    def test_exists_size_and_delete_propagate_a_backend_error(self, tmp_path: Path):
        from google.api_core.exceptions import ServiceUnavailable

        store = self._store(tmp_path, ServiceUnavailable("backend"))

        for call in (store.blob_exists, store.blob_size, store.delete_blob):
            with pytest.raises(ServiceUnavailable, match="backend"):
                call("artifact-1", 1)

    def test_a_failed_upload_leaves_the_previous_object(self, tmp_path: Path):
        from google.api_core.exceptions import ServiceUnavailable

        store = self._store(tmp_path)
        store.write_blob("artifact-1", 1, b"the old bytes")
        source = tmp_path / "incoming.bin"
        source.write_bytes(b"the new bytes")

        store._client.error = ServiceUnavailable("backend")
        with pytest.raises(ServiceUnavailable):
            store.publish_blob_from_path("artifact-1", 1, source)
        store._client.error = None

        assert store.read_blob("artifact-1", 1) == b"the old bytes"


class _FakeBlobClient:
    """One blob of a fake Azure container, over a local file."""

    def __init__(self, path: Path, record: list[str]):
        from azure.core.exceptions import ResourceNotFoundError

        self._path = path
        self._record = record
        self._absent = ResourceNotFoundError

    def upload_blob(self, data, overwrite: bool = False) -> None:
        self._record.append(f"upload:overwrite={overwrite}")
        self._path.write_bytes(data.read())

    def download_blob(self):
        if not self._path.exists():
            raise self._absent(f"{self._path.name} not found")
        content = self._path.read_bytes()
        return SimpleNamespace(chunks=lambda: iter([content[:3], content[3:]]))

    def exists(self) -> bool:
        return self._path.exists()

    def get_blob_properties(self):
        if not self._path.exists():
            raise self._absent(f"{self._path.name} not found")
        return SimpleNamespace(size=self._path.stat().st_size)

    def delete_blob(self) -> None:
        if not self._path.exists():
            raise self._absent(f"{self._path.name} not found")
        self._path.unlink()


class _FakeContainerClient:
    def __init__(self, root: Path):
        self.root = root
        self.record: list[str] = []

    def get_blob_client(self, key: str) -> _FakeBlobClient:
        return _FakeBlobClient(self.root / key.replace("/", "_"), self.record)


class TestAzureBackendOverAFakeClient:
    """The Azure I/O paths over a fake ``ContainerClient``, keyed like the real one."""

    def _store(self, tmp_path: Path):
        pytest.importorskip("azure.core.exceptions")
        from vestigo.blob_store import AzureBlobStore

        store = AzureBlobStore.__new__(AzureBlobStore)
        store.prefix = "artifacts"
        store._client = _FakeContainerClient(tmp_path)
        return store

    def test_write_publish_exists_size_delete(self, tmp_path: Path):
        store = self._store(tmp_path)

        assert store.read_blob("artifact-1", 1) is None
        assert store.blob_exists("artifact-1", 1) is False
        assert store.blob_size("artifact-1", 1) is None
        assert store.delete_blob("artifact-1", 1) is False

        store.write_blob("artifact-1", 1, b"streamed")
        assert store.read_blob("artifact-1", 1) == b"streamed"
        assert store.blob_exists("artifact-1", 1) is True
        assert store.blob_size("artifact-1", 1) == len(b"streamed")
        assert (tmp_path / "artifacts_artifact-1@v=1.arrow").exists()

        source = tmp_path / "incoming.bin"
        source.write_bytes(b"published")
        store.publish_blob_from_path("artifact-1", 1, source)
        assert store.read_blob("artifact-1", 1) == b"published"
        assert store._client.record == ["upload:overwrite=True"] * 2

        assert store.delete_blob("artifact-1", 1) is True
        assert store.blob_exists("artifact-1", 1) is False
        assert store.delete_blob("artifact-1", 1) is False

    def test_a_failed_writer_uploads_nothing(self, tmp_path: Path):
        store = self._store(tmp_path)

        with pytest.raises(RuntimeError, match="boom"):
            with store.open_blob_writer("artifact-1", 1) as writer:
                writer.write(b"partial")
                raise RuntimeError("boom")

        assert store._client.record == []
        assert store.blob_exists("artifact-1", 1) is False


class TestAzureDownloadReader:
    """The file-like over an Azure download's chunks returns the blob's bytes.

    Chunk sizes vary (the SDK's first GET is larger) and a read may straddle any number of chunks.
    """

    DATA = bytes(range(256)) * 400  # 102400 bytes
    CHUNK_SIZES = (40000, 0, 7, 30000, 1, 32392)

    class Downloader:
        def __init__(self, data: bytes, sizes: tuple[int, ...]) -> None:
            assert sum(sizes) == len(data)
            self._data = data
            self._sizes = sizes

        def chunks(self):
            start = 0
            for size in self._sizes:
                yield self._data[start : start + size]
                start += size

    def _raw(self):
        from vestigo.blob_store import _AzureDownloadReader

        return _AzureDownloadReader(self.Downloader(self.DATA, self.CHUNK_SIZES))

    def test_read_returns_the_whole_blob(self):
        assert io.BufferedReader(self._raw()).read() == self.DATA

    @pytest.mark.parametrize("size", [1, 3, 4096, 39999, 40008, 70007, 200000])
    def test_read_size_fills_across_chunks(self, size: int):
        stream = io.BufferedReader(self._raw())
        pieces = []
        remaining = len(self.DATA)
        while piece := stream.read(size):
            assert len(piece) == min(size, remaining)
            remaining -= len(piece)
            pieces.append(piece)
        assert b"".join(pieces) == self.DATA

    def test_readinto_with_odd_buffer_sizes(self):
        raw = self._raw()
        sizes = [5, 39990, 13, 1, 30001, 65536]
        out = bytearray()
        n = 1
        i = 0
        while n:
            buffer = bytearray(sizes[i % len(sizes)])
            n = raw.readinto(buffer)
            out += buffer[:n]
            i += 1
        assert bytes(out) == self.DATA
        assert raw.readall() == b""

    def test_readall_after_partial_reads_returns_the_rest(self):
        raw = self._raw()
        head = bytearray(40003)
        assert raw.readinto(head) == 40000  # stops at the first chunk's end
        assert raw.readinto(head) == 7
        assert raw.readall() == self.DATA[40007:]
        assert raw.readinto(head) == 0

        stream = io.BufferedReader(self._raw())
        assert stream.read(12345) == self.DATA[:12345]
        assert stream.read() == self.DATA[12345:]
        assert stream.read() == b""

    def test_copyfileobj_copies_the_blob(self):
        out = io.BytesIO()
        shutil.copyfileobj(io.BufferedReader(self._raw()), out, length=9999)
        assert out.getvalue() == self.DATA


class TestAzureBlobStore:
    """Key generation and errors for AzureBlobStore; full coverage needs Azure or Azurite."""

    def test_azure_key_format_with_prefix(self):
        """The Azure key includes the prefix."""
        pytest.importorskip("azure.storage.blob")
        from vestigo.blob_store import AzureBlobStore

        # A fake connection string: key generation needs no real connection.
        store = AzureBlobStore(
            account_name="testaccount",
            container_name="test-container",
            prefix="artifacts",
            connection_string="DefaultEndpointsProtocol=https;AccountName=testaccount;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
        )

        key = store._azure_key("abc123", 5)

        assert key == "artifacts/abc123@v=5.arrow"

    def test_azure_key_format_without_prefix(self):
        pytest.importorskip("azure.storage.blob")
        from vestigo.blob_store import AzureBlobStore

        store = AzureBlobStore(
            account_name="testaccount",
            container_name="test-container",
            prefix="",
            connection_string="DefaultEndpointsProtocol=https;AccountName=testaccount;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
        )

        key = store._azure_key("abc123", 5)

        assert key == "abc123@v=5.arrow"

    def test_blob_key_format(self):
        pytest.importorskip("azure.storage.blob")
        from vestigo.blob_store import AzureBlobStore

        store = AzureBlobStore(
            account_name="testaccount",
            container_name="test-container",
            prefix="artifacts",
            connection_string="DefaultEndpointsProtocol=https;AccountName=testaccount;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
        )

        key = store._blob_key("abc123", 5)

        assert key == "abc123@v=5.arrow"

    def test_requires_auth_method(self):
        """The Azure store raises without an auth method."""
        pytest.importorskip("azure.storage.blob")
        from vestigo.blob_store import AzureBlobStore

        with pytest.raises(ValueError, match="requires one of"):
            AzureBlobStore(
                account_name="testaccount",
                container_name="test-container",
                prefix="artifacts",
                # No auth method provided
            )

    def test_endpoint_url_is_the_host_and_the_account_is_its_first_path_segment(self):
        """The documented meaning of ``azure_endpoint_url``."""
        pytest.importorskip("azure.storage.blob")
        from vestigo.blob_store import AzureBlobStore

        store = AzureBlobStore(
            account_name="devstoreaccount1",
            container_name="test-container",
            account_key="dGVzdGtleQ==",
            endpoint_url="http://127.0.0.1:10000/",
        )

        assert store._account_url == "http://127.0.0.1:10000/devstoreaccount1"
        assert store._client.url == "http://127.0.0.1:10000/devstoreaccount1/test-container"
        assert store._client.account_name == "devstoreaccount1"

    def test_endpoint_url_already_naming_the_account_is_not_doubled(self):
        pytest.importorskip("azure.storage.blob")
        from vestigo.blob_store import AzureBlobStore

        store = AzureBlobStore(
            account_name="devstoreaccount1",
            container_name="test-container",
            account_key="dGVzdGtleQ==",
            endpoint_url="http://127.0.0.1:10000/devstoreaccount1/",
        )

        assert store._account_url == "http://127.0.0.1:10000/devstoreaccount1"
        assert store._client.account_name == "devstoreaccount1"

    def test_endpoint_url_without_an_account_name_is_used_as_given(self):
        pytest.importorskip("azure.storage.blob")
        from vestigo.blob_store import AzureBlobStore

        store = AzureBlobStore(
            account_name="",
            container_name="test-container",
            sas_token="sv=2024&sig=abc",
            endpoint_url="https://acct.blob.core.windows.net",
        )

        assert store._account_url == "https://acct.blob.core.windows.net"
        assert (
            store._client.url == "https://acct.blob.core.windows.net/test-container?sv=2024&sig=abc"
        )
