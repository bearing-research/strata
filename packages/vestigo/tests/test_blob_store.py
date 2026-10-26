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


class _FakeOutputStream:
    """Completes on exit even after an error, as PyArrow's object-store streams do."""

    def __init__(
        self,
        sink: Path,
        record: list[str],
        *,
        fail_on_open: bool = False,
        fail_on_write: int | None = None,
    ):
        self._sink = sink
        self._record = record
        self._writes = 0
        self._fail_on_write = fail_on_write
        if fail_on_open:
            self._record.append("open_failed")
            raise OSError("simulated s3 open failure")
        self._handle = open(sink, "wb")
        self._record.append(f"open:{sink.name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._handle.close()
        self._record.append("close")

    def write(self, data: bytes) -> None:
        self._writes += 1
        if self._writes == self._fail_on_write:
            raise OSError("simulated s3 write failure")
        self._handle.write(data)


class _FakeArrowFileSystem:
    """The PyArrow filesystem surface the S3 and GCS stores use, over a local directory.

    ``backend_error`` is raised by every call, as an unreachable or forbidden bucket would.
    """

    def __init__(
        self,
        root: Path,
        *,
        open_fails: bool = False,
        open_raises_on_call: int | None = None,
        fail_on_write: int | None = None,
        backend_error: Exception | None = None,
    ):
        self.root = root
        self._fail_on_write = fail_on_write
        self._record: list[str] = []
        self.open_fails = open_fails
        self._open_raises_on_call = open_raises_on_call
        self._open_calls = 0
        self.backend_error = backend_error

    def _sink(self, key: str) -> Path:
        if self.backend_error is not None:
            raise self.backend_error
        return self.root / key.replace("/", "_")

    def open_output_stream(self, key: str):
        self._open_calls += 1
        sink = self._sink(key)
        if self.open_fails or self._open_calls == self._open_raises_on_call:
            return _FakeOutputStream(sink, self._record, fail_on_open=True)
        return _FakeOutputStream(sink, self._record, fail_on_write=self._fail_on_write)

    def open_input_stream(self, key: str):
        sink = self._sink(key)
        if not sink.exists():
            raise FileNotFoundError(key)
        return open(sink, "rb")

    def get_file_info(self, key: str):
        import pyarrow.fs as pafs

        sink = self._sink(key)
        if not sink.exists():
            return pafs.FileInfo(key, type=pafs.FileType.NotFound)
        return pafs.FileInfo(key, type=pafs.FileType.File, size=sink.stat().st_size)

    def delete_file(self, key: str) -> None:
        self._record.append("delete")
        sink = self._sink(key)
        if not sink.exists():
            raise FileNotFoundError(key)
        sink.unlink()


def _arrow_store(cls, fake_fs, prefix: str = "artifacts"):
    """An S3 or GCS store over ``fake_fs``, without the real client its constructor builds."""
    store = cls.__new__(cls)
    store.bucket = "fake-bucket"
    store.prefix = prefix
    store._fs = fake_fs
    return store


class TestS3BackendAtomicity:
    """Writer atomicity for S3BlobStore over a fake PyArrow filesystem."""

    def _build_store_with_fake_fs(self, fake_fs, prefix: str = "artifacts") -> S3BlobStore:
        return _arrow_store(S3BlobStore, fake_fs, prefix)

    def test_writer_commit_produces_remote_object(self, tmp_path: Path):
        """A clean writer context publishes the full payload."""
        fake_fs = _FakeArrowFileSystem(tmp_path)
        store = self._build_store_with_fake_fs(fake_fs)

        with store.open_blob_writer("artifact-1", 1) as writer:
            writer.write(b"part1")
            writer.write(b"part2")

        sink = tmp_path / "fake-bucket_artifacts_artifact-1@v=1.arrow"
        assert sink.read_bytes() == b"part1part2"
        assert fake_fs._record[0].startswith("open:")
        assert fake_fs._record[-1] == "close"

    def test_writer_exception_never_opens_remote_stream(self, tmp_path: Path):
        """An exception inside the writer context skips the remote upload entirely."""
        fake_fs = _FakeArrowFileSystem(tmp_path)
        store = self._build_store_with_fake_fs(fake_fs)

        class BoomError(RuntimeError):
            pass

        with pytest.raises(BoomError):
            with store.open_blob_writer("artifact-1", 1) as writer:
                writer.write(b"partial")
                raise BoomError()

        assert fake_fs._record == []
        # No sink file should exist in the fake bucket.
        assert list(tmp_path.iterdir()) == []

    def test_writer_open_failure_cleans_up_staged_tempfile(self, tmp_path: Path, monkeypatch):
        """If the upload fails at open, the local tempfile is still removed."""
        fake_fs = _FakeArrowFileSystem(tmp_path, open_fails=True)
        store = self._build_store_with_fake_fs(fake_fs)

        from vestigo import blob_store as blob_store_module

        captured_tmp: list[Path] = []
        real_mkstemp = blob_store_module.tempfile.mkstemp

        def _recording_mkstemp(*args, **kwargs):
            fd, name = real_mkstemp(*args, **kwargs)
            captured_tmp.append(Path(name))
            return fd, name

        monkeypatch.setattr(blob_store_module.tempfile, "mkstemp", _recording_mkstemp)

        with pytest.raises(OSError, match="simulated s3 open failure"):
            with store.open_blob_writer("artifact-1", 1) as writer:
                writer.write(b"payload")

        assert captured_tmp and not captured_tmp[0].exists()
        assert fake_fs._record == ["open_failed", "delete"]

    def test_a_failed_upload_leaves_no_object_at_the_key(self, tmp_path: Path):
        """Leaving the stream completes the upload, so a copy that failed midway used to leave
        a truncated object, which blob_exists reported and an import retry then kept."""
        fake_fs = _FakeArrowFileSystem(tmp_path, fail_on_write=2)
        store = self._build_store_with_fake_fs(fake_fs)
        source = tmp_path / "incoming.bin"
        source.write_bytes(b"x" * (3 * blob_store_module.BLOB_STREAM_CHUNK_BYTES))

        with pytest.raises(OSError, match="simulated s3 write failure"):
            store.publish_blob_from_path("artifact-1", 1, source)

        assert not (tmp_path / "fake-bucket_artifacts_artifact-1@v=1.arrow").exists()
        assert fake_fs._record[-2:] == ["close", "delete"]

    def test_publish_from_path_skips_double_staging(self, tmp_path: Path, monkeypatch):
        """publish_blob_from_path uploads the source directly, without mkstemp."""
        fake_fs = _FakeArrowFileSystem(tmp_path)
        store = self._build_store_with_fake_fs(fake_fs)

        from vestigo import blob_store as blob_store_module

        mkstemp_calls: list[str] = []
        real_mkstemp = blob_store_module.tempfile.mkstemp

        def _counting_mkstemp(*args, **kwargs):
            mkstemp_calls.append(kwargs.get("prefix", ""))
            return real_mkstemp(*args, **kwargs)

        monkeypatch.setattr(blob_store_module.tempfile, "mkstemp", _counting_mkstemp)

        source = tmp_path / "incoming.bin"
        payload = b"abc123" * 500
        source.write_bytes(payload)

        store.publish_blob_from_path("artifact-1", 1, source)

        blob_prefix_calls = [p for p in mkstemp_calls if p.startswith("vestigo_s3_blob_")]
        assert blob_prefix_calls == [], (
            "publish_blob_from_path should not route through the staged-writer tempfile"
        )
        sink = tmp_path / "fake-bucket_artifacts_artifact-1@v=1.arrow"
        assert sink.read_bytes() == payload
        # Source file is not consumed; the caller owns cleanup.
        assert source.exists()


class TestS3BlobStore:
    """Key generation and construction for S3BlobStore.

    PyArrow's S3FileSystem uses its own C++ AWS SDK, which moto cannot mock; the I/O paths run
    over ``_FakeArrowFileSystem`` below, and the round trip needs LocalStack or MinIO.
    """

    def test_s3_key_format(self):
        """The S3 key includes bucket and prefix."""
        store = S3BlobStore(bucket="test-bucket", prefix="artifacts", anonymous=True)

        assert store._s3_key("abc123", 5) == "test-bucket/artifacts/abc123@v=5.arrow"

    def test_s3_key_without_prefix(self):
        store = S3BlobStore(bucket="test-bucket", prefix="", anonymous=True)

        assert store._s3_key("abc123", 5) == "test-bucket/abc123@v=5.arrow"

    def test_blob_key_format(self):
        store = S3BlobStore(bucket="test-bucket", prefix="artifacts", anonymous=True)

        assert store._blob_key("abc123", 5) == "abc123@v=5.arrow"

    @pytest.mark.parametrize("half", [{"access_key": "AKIA"}, {"secret_key": "s3cret"}])
    def test_half_a_key_pair_is_refused(self, half):
        # One key alone would fall back to ambient credentials, for I/O and signing both.
        with pytest.raises(ValueError, match="both access_key and secret_key"):
            S3BlobStore(bucket="test-bucket", **half)

    @pytest.mark.skipif(
        not os.environ.get("VESTIGO_S3_ENDPOINT_URL"),
        reason="set VESTIGO_S3_ENDPOINT_URL (LocalStack or MinIO) to run the S3 round trip",
    )
    def test_write_and_read_blob_integration(self):
        """Round-trip against an S3 endpoint with the ``test``/``test`` key pair.

        Set VESTIGO_S3_ENDPOINT_URL=http://localhost:4566 and VESTIGO_S3_REGION=us-east-1.
        """
        store = S3BlobStore(
            bucket="test-bucket",
            prefix="artifacts",
            region=os.environ.get("VESTIGO_S3_REGION", "us-east-1"),
            endpoint_url=os.environ["VESTIGO_S3_ENDPOINT_URL"],
            access_key="test",
            secret_key="test",
        )

        data = b"test artifact data"
        store.write_blob("artifact-1", 1, data)
        result = store.read_blob("artifact-1", 1)

        assert result == data


class TestGCSCredentialResolution:
    """``GOOGLE_APPLICATION_CREDENTIALS`` gets a path and nothing else.

    Operators paste key material into the consumer's credentials setting, often an env var;
    handed straight to Google it fails only at first blob access.
    """

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows has no Unix mode bits; chmod only toggles the read-only flag",
    )
    def test_inline_key_material_is_written_to_a_private_file(self, tmp_path):
        from vestigo.blob_store import _resolve_gcs_credentials

        key = '{"type": "service_account", "project_id": "p", "private_key": "k"}'

        path = _resolve_gcs_credentials(key)

        assert path != key, "inline JSON must become a path"
        written = Path(path)
        assert json.loads(written.read_text()) == json.loads(key)
        # Key material on disk: readable by its owner and nobody else.
        assert stat.S_IMODE(written.stat().st_mode) == 0o600

    def test_the_same_key_reuses_one_file(self):
        # atexit does not run on SIGKILL or an OOM kill, so a random filename
        # would leave one private key behind per hard-killed process. The name
        # is derived from the key, so a restart rewrites the same file.
        from vestigo.blob_store import _resolve_gcs_credentials

        key = '{"type": "service_account", "private_key": "k"}'

        first = _resolve_gcs_credentials(key)
        second = _resolve_gcs_credentials(key)
        other = _resolve_gcs_credentials('{"type": "service_account", "private_key": "j"}')

        assert first == second
        assert other != first

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows has no Unix mode bits; chmod only toggles the read-only flag",
    )
    def test_a_file_left_world_readable_is_tightened(self):
        # A leftover from a previous run, or a temp dir with a permissive
        # umask: the mode has to be fixed rather than assumed.
        from vestigo.blob_store import _resolve_gcs_credentials

        key = '{"type": "service_account", "private_key": "loose"}'
        path = Path(_resolve_gcs_credentials(key))
        path.chmod(0o644)

        _resolve_gcs_credentials(key)

        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    @pytest.mark.skipif(sys.platform == "win32", reason="no O_NOFOLLOW or file owners there")
    def test_a_symlink_planted_at_the_predictable_name_is_not_followed(self, tmp_path, monkeypatch):
        # The name is derived from the key and visible in the temp dir; another user who
        # saw it plants a link there and the next start would write the key through it.
        from vestigo.blob_store import _resolve_gcs_credentials

        monkeypatch.setattr(blob_store_module.tempfile, "gettempdir", lambda: str(tmp_path))
        key = '{"type": "service_account", "private_key": "SECRET"}'
        target = tmp_path / "theirs.txt"
        target.write_text("")
        planted = Path(_resolve_gcs_credentials(key))
        planted.unlink()
        planted.symlink_to(target)

        with pytest.raises(OSError):
            _resolve_gcs_credentials(key)

        assert target.read_text() == ""

    def test_a_path_is_passed_through_untouched(self):
        from vestigo.blob_store import _resolve_gcs_credentials

        assert _resolve_gcs_credentials("/etc/vestigo/gcs-key.json") == "/etc/vestigo/gcs-key.json"

    def test_a_json_scalar_is_treated_as_a_path(self):
        # Parsed rather than sniffed for a leading brace, but a bare JSON
        # scalar still parses. A relative filename must not be mistaken for
        # key material just because json.loads accepts it.
        from vestigo.blob_store import _resolve_gcs_credentials

        assert _resolve_gcs_credentials("123") == "123"


class TestGCSBlobStore:
    """Key generation for GCSBlobStore.

    PyArrow's GcsFileSystem uses its own C++ SDK, which mocks cannot reach; full coverage needs real
    GCS or fake-gcs-server.
    """

    def test_gcs_key_format(self):
        """The GCS key includes bucket and prefix."""
        # Create store with anonymous access to avoid credential errors
        store = GCSBlobStore(
            bucket="test-bucket",
            prefix="artifacts",
            anonymous=True,
        )

        key = store._gcs_key("abc123", 5)

        assert key == "test-bucket/artifacts/abc123@v=5.arrow"

    def test_gcs_key_without_prefix(self):
        store = GCSBlobStore(
            bucket="test-bucket",
            prefix="",
            anonymous=True,
        )

        key = store._gcs_key("abc123", 5)
        assert key == "test-bucket/abc123@v=5.arrow"

    def test_blob_key_format(self):
        store = GCSBlobStore(
            bucket="test-bucket",
            prefix="artifacts",
            anonymous=True,
        )

        key = store._blob_key("abc123", 5)

        assert key == "abc123@v=5.arrow"

    @pytest.mark.skipif(
        not os.environ.get("VESTIGO_GCS_ENDPOINT_OVERRIDE"),
        reason="set VESTIGO_GCS_ENDPOINT_OVERRIDE (fake-gcs-server) to run the GCS round trip",
    )
    def test_write_and_read_blob_integration(self):
        """Round-trip against a GCS endpoint, anonymously.

        For fake-gcs-server: VESTIGO_GCS_ENDPOINT_OVERRIDE=localhost:4443.
        """
        store = GCSBlobStore(
            bucket="test-bucket",
            prefix="artifacts",
            endpoint_override=os.environ["VESTIGO_GCS_ENDPOINT_OVERRIDE"],
            anonymous=True,
        )

        data = b"test artifact data"
        store.write_blob("artifact-1", 1, data)
        result = store.read_blob("artifact-1", 1)

        assert result == data


@pytest.mark.parametrize("cls", [S3BlobStore, GCSBlobStore], ids=["s3", "gcs"])
class TestArrowBackendsOverAFakeFilesystem:
    """The S3 and GCS I/O paths, and their one error contract, over ``_FakeArrowFileSystem``."""

    def test_write_publish_exists_size_delete(self, cls, tmp_path: Path):
        store = _arrow_store(cls, _FakeArrowFileSystem(tmp_path / "bucket"))
        (tmp_path / "bucket").mkdir()

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

        assert store.delete_blob("artifact-1", 1) is True
        assert store.blob_exists("artifact-1", 1) is False
        assert store.delete_blob("artifact-1", 1) is False

    def test_a_backend_error_on_read_is_not_absent(self, cls, tmp_path: Path):
        """ACCESS_DENIED, a bad region or a timeout used to read as None: an artifact whose
        blob is "gone", a 404, an integrity finding, all silent."""
        denied = OSError("AWS Error ACCESS_DENIED during HeadObject operation")
        store = _arrow_store(cls, _FakeArrowFileSystem(tmp_path, backend_error=denied))

        with pytest.raises(OSError, match="ACCESS_DENIED"):
            store.read_blob("artifact-1", 1)
        with pytest.raises(OSError, match="ACCESS_DENIED"):
            store.open_blob_reader("artifact-1", 1)

    def test_exists_size_and_delete_propagate_a_backend_error(self, cls, tmp_path: Path):
        """``delete_blob`` matters most: the caller removes the metadata row regardless, so a
        False that meant "could not reach the bucket" orphaned the object for good."""
        reset = OSError("connection reset")
        store = _arrow_store(cls, _FakeArrowFileSystem(tmp_path, backend_error=reset))

        for call in (store.blob_exists, store.blob_size, store.delete_blob):
            with pytest.raises(OSError, match="connection reset"):
                call("artifact-1", 1)

    def test_a_delete_that_loses_the_race_reports_absent(self, cls, tmp_path: Path):
        fake_fs = _FakeArrowFileSystem(tmp_path)
        store = _arrow_store(cls, fake_fs)
        store.write_blob("artifact-1", 1, b"x")
        real_delete = fake_fs.delete_file

        def delete_after_the_other_sweep(key):
            (tmp_path / key.replace("/", "_")).unlink()
            real_delete(key)

        fake_fs.delete_file = delete_after_the_other_sweep

        assert store.delete_blob("artifact-1", 1) is False


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

    @pytest.mark.skipif(
        not os.environ.get("VESTIGO_AZURE_CONNECTION_STRING"),
        reason="set VESTIGO_AZURE_CONNECTION_STRING (Azurite) to run the Azure round trip",
    )
    def test_write_and_read_blob_integration(self):
        """Round-trip against an Azure endpoint.

        For Azurite: VESTIGO_AZURE_CONNECTION_STRING=UseDevelopmentStorage=true
        """
        pytest.importorskip("azure.storage.blob")
        from vestigo.blob_store import AzureBlobStore

        store = AzureBlobStore(
            account_name="devstoreaccount1",  # Azurite default
            container_name="test-container",
            prefix="artifacts",
            connection_string=os.environ["VESTIGO_AZURE_CONNECTION_STRING"],
        )

        data = b"test artifact data"
        store.write_blob("artifact-1", 1, data)
        result = store.read_blob("artifact-1", 1)

        assert result == data
