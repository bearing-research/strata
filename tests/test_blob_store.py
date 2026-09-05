"""Tests for blob storage backends."""

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata import blob_store as blob_store_module
from strata.blob_store import (
    BlobStore,
    GCSBlobStore,
    LocalBlobStore,
    S3BlobStore,
    create_blob_store,
)
from strata.config import StrataConfig


class TestLocalBlobStore:
    """Tests for LocalBlobStore."""

    def test_write_and_read_blob(self, tmp_path: Path):
        """Test writing and reading a blob."""
        store = LocalBlobStore(tmp_path / "blobs")
        data = b"test artifact data"

        store.write_blob("artifact-1", 1, data)
        result = store.read_blob("artifact-1", 1)

        assert result == data

    def test_read_nonexistent_blob(self, tmp_path: Path):
        """Test reading a blob that doesn't exist."""
        store = LocalBlobStore(tmp_path / "blobs")

        result = store.read_blob("nonexistent", 1)

        assert result is None

    def test_blob_key_is_case_collision_proof(self, tmp_path: Path):
        """Ids differing only in case must not collapse to one file.

        On a case-insensitive filesystem (macOS/APFS, Windows) ``…_var_Widget``
        and ``…_var_widget`` would otherwise map to the same blob, so a class and
        its same-named instance share an artifact. Asserted at the key level so it
        catches the bug on any host (Linux CI won't reproduce the FS collision).
        """
        store = LocalBlobStore(tmp_path / "blobs")
        k_upper = store._blob_key("nb_x_cell_c_var_Widget", 1)
        k_lower = store._blob_key("nb_x_cell_c_var_widget", 1)
        assert k_upper != k_lower
        # The real bug: they must not collide under case folding.
        assert k_upper.lower() != k_lower.lower()
        # All-lowercase ids keep their stable name (cached blobs survive upgrades).
        assert k_lower == "nb_x_cell_c_var_widget@v=1.arrow"

    def test_case_differing_ids_round_trip_to_distinct_data(self, tmp_path: Path):
        """Writing two case-differing ids and reading them back yields distinct
        data (would fail on a case-insensitive FS before the key fix)."""
        store = LocalBlobStore(tmp_path / "blobs")
        store.write_blob("nb_var_Widget", 1, b"the class")
        store.write_blob("nb_var_widget", 1, b"the instance")
        assert store.read_blob("nb_var_Widget", 1) == b"the class"
        assert store.read_blob("nb_var_widget", 1) == b"the instance"

    def test_blob_exists(self, tmp_path: Path):
        """Test checking if a blob exists."""
        store = LocalBlobStore(tmp_path / "blobs")
        data = b"test data"

        assert not store.blob_exists("artifact-1", 1)

        store.write_blob("artifact-1", 1, data)

        assert store.blob_exists("artifact-1", 1)
        assert not store.blob_exists("artifact-1", 2)

    def test_delete_blob(self, tmp_path: Path):
        """Test deleting a blob."""
        store = LocalBlobStore(tmp_path / "blobs")
        data = b"test data"

        store.write_blob("artifact-1", 1, data)
        assert store.blob_exists("artifact-1", 1)

        result = store.delete_blob("artifact-1", 1)

        assert result is True
        assert not store.blob_exists("artifact-1", 1)

    def test_delete_nonexistent_blob(self, tmp_path: Path):
        """Test deleting a blob that doesn't exist."""
        store = LocalBlobStore(tmp_path / "blobs")

        result = store.delete_blob("nonexistent", 1)

        assert result is False

    def test_multiple_versions(self, tmp_path: Path):
        """Test storing multiple versions of the same artifact."""
        store = LocalBlobStore(tmp_path / "blobs")

        store.write_blob("artifact-1", 1, b"version 1")
        store.write_blob("artifact-1", 2, b"version 2")
        store.write_blob("artifact-1", 3, b"version 3")

        assert store.read_blob("artifact-1", 1) == b"version 1"
        assert store.read_blob("artifact-1", 2) == b"version 2"
        assert store.read_blob("artifact-1", 3) == b"version 3"

    def test_blob_key_format(self, tmp_path: Path):
        """Test the blob key format."""
        store = LocalBlobStore(tmp_path / "blobs")

        key = store._blob_key("abc123", 5)

        assert key == "abc123@v=5.arrow"

    def test_creates_directory(self, tmp_path: Path):
        """Test that the store creates the blobs directory."""
        blobs_dir = tmp_path / "new" / "nested" / "blobs"
        assert not blobs_dir.exists()

        LocalBlobStore(blobs_dir)

        assert blobs_dir.exists()

    def test_atomic_write(self, tmp_path: Path):
        """Test that writes are atomic (no partial files on failure)."""
        store = LocalBlobStore(tmp_path / "blobs")
        data = b"test data" * 1000

        store.write_blob("artifact-1", 1, data)

        # No residual ``*.tmp`` files should remain after a successful write
        residual_tmp = list((tmp_path / "blobs").glob("*.tmp"))
        assert residual_tmp == []

        # The final file should exist with correct content
        assert store.read_blob("artifact-1", 1) == data

    def test_streaming_writer_commits_atomically(self, tmp_path: Path):
        """open_blob_writer should only publish the blob on clean context exit."""
        store = LocalBlobStore(tmp_path / "blobs")

        with store.open_blob_writer("artifact-1", 1) as writer:
            writer.write(b"chunk1")
            writer.write(b"chunk2")
            assert not store.blob_exists("artifact-1", 1)

        assert store.read_blob("artifact-1", 1) == b"chunk1chunk2"

    def test_streaming_writer_discards_on_exception(self, tmp_path: Path):
        """A writer context that exits with an exception must not leave a blob or tmp file."""
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
        """open_blob_reader should yield a file-like streaming chunks."""
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
        """open_blob_reader should return None for missing blobs."""
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
        """blob_size must return length via filesystem metadata."""
        store = LocalBlobStore(tmp_path / "blobs")
        payload = b"x" * 12345
        store.write_blob("artifact-1", 1, payload)

        assert store.blob_size("artifact-1", 1) == len(payload)

    def test_blob_size_missing_returns_none(self, tmp_path: Path):
        """blob_size must return None for missing blobs."""
        store = LocalBlobStore(tmp_path / "blobs")
        assert store.blob_size("absent", 1) is None


class TestStagedLocalWriter:
    """Tests for the shared ``BlobStore._staged_local_writer`` helper.

    The helper backs the atomicity guarantees of S3/GCS/Azure writers —
    staging to a local tempfile, invoking ``commit(path)`` only on clean
    context exit, and discarding the tempfile otherwise.
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
        """``commit`` must not run if the writer context exits via exception."""
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
        """The staged tempfile must be removed even when commit is never invoked."""
        captured_path: list[Path] = []
        import strata.blob_store as blob_store_module

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
        """If commit itself raises, the tempfile still gets removed."""
        captured_path: list[Path] = []
        import strata.blob_store as blob_store_module

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
        """Writer handle must be closed before ``commit`` is invoked.

        On Windows, reopening a file while another handle has write-mode
        open fails with ERROR_SHARING_VIOLATION. The remote backends
        reopen the staged tempfile inside ``commit`` to upload it, so
        the writer must release its handle first.
        """
        observed_state: dict[str, object] = {}

        with BlobStore._staged_local_writer(
            lambda _path: None, prefix="strata_handle_test_"
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
    """Tests for ``BlobStore.publish_blob_from_path``."""

    def test_round_trip_via_local_backend(self, tmp_path: Path):
        """The default implementation publishes a staged file atomically."""
        store = LocalBlobStore(tmp_path / "blobs")
        staging = tmp_path / "incoming.bin"
        payload = b"hello " * 1000
        staging.write_bytes(payload)

        store.publish_blob_from_path("artifact-1", 1, staging)

        assert store.read_blob("artifact-1", 1) == payload
        # Source is not consumed — the caller owns its lifecycle.
        assert staging.exists()

    def test_publish_overwrites_existing_blob(self, tmp_path: Path):
        """Publishing a new file replaces the previously committed blob."""
        store = LocalBlobStore(tmp_path / "blobs")
        store.write_blob("artifact-1", 1, b"v1-bytes")

        staging = tmp_path / "incoming.bin"
        staging.write_bytes(b"v2-bytes")

        store.publish_blob_from_path("artifact-1", 1, staging)

        assert store.read_blob("artifact-1", 1) == b"v2-bytes"


class TestS3BackendAtomicity:
    """Writer atomicity tests for S3BlobStore using a fake PyArrow filesystem."""

    class _FakeOutputStream:
        def __init__(self, sink: Path, record: list[str], *, fail_on_open: bool = False):
            self._sink = sink
            self._record = record
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
            self._handle.write(data)

    class _FakeS3FileSystem:
        def __init__(
            self,
            root: Path,
            *,
            open_fails: bool = False,
            open_raises_on_call: int | None = None,
        ):
            self.root = root
            self._record: list[str] = []
            self.open_fails = open_fails
            self._open_raises_on_call = open_raises_on_call
            self._open_calls = 0

        def open_output_stream(self, key: str):
            self._open_calls += 1
            sink = self.root / key.replace("/", "_")
            if self.open_fails or self._open_calls == self._open_raises_on_call:
                return TestS3BackendAtomicity._FakeOutputStream(
                    sink, self._record, fail_on_open=True
                )
            return TestS3BackendAtomicity._FakeOutputStream(sink, self._record)

    def _build_store_with_fake_fs(self, fake_fs, prefix: str = "artifacts") -> S3BlobStore:
        store = S3BlobStore.__new__(S3BlobStore)
        store.bucket = "fake-bucket"
        store.prefix = prefix
        store._fs = fake_fs
        return store

    def test_writer_commit_produces_remote_object(self, tmp_path: Path):
        """A clean writer context publishes the full payload via the fake fs."""
        fake_fs = self._FakeS3FileSystem(tmp_path)
        store = self._build_store_with_fake_fs(fake_fs)

        with store.open_blob_writer("artifact-1", 1) as writer:
            writer.write(b"part1")
            writer.write(b"part2")

        sink = tmp_path / "fake-bucket_artifacts_artifact-1@v=1.arrow"
        assert sink.read_bytes() == b"part1part2"
        assert fake_fs._record[0].startswith("open:")
        assert fake_fs._record[-1] == "close"

    def test_writer_exception_never_opens_remote_stream(self, tmp_path: Path):
        """An exception inside the writer context must skip the remote upload entirely."""
        fake_fs = self._FakeS3FileSystem(tmp_path)
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
        """If the backend upload fails at open, the local tempfile is still removed."""
        fake_fs = self._FakeS3FileSystem(tmp_path, open_fails=True)
        store = self._build_store_with_fake_fs(fake_fs)

        import strata.blob_store as blob_store_module

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
        assert fake_fs._record == ["open_failed"]

    def test_publish_from_path_skips_double_staging(self, tmp_path: Path, monkeypatch):
        """publish_blob_from_path must upload source directly without mkstemp."""
        fake_fs = self._FakeS3FileSystem(tmp_path)
        store = self._build_store_with_fake_fs(fake_fs)

        import strata.blob_store as blob_store_module

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

        blob_prefix_calls = [p for p in mkstemp_calls if p.startswith("strata_s3_blob_")]
        assert blob_prefix_calls == [], (
            "publish_blob_from_path should not route through the staged-writer tempfile"
        )
        sink = tmp_path / "fake-bucket_artifacts_artifact-1@v=1.arrow"
        assert sink.read_bytes() == payload
        # Source file is not consumed — caller owns cleanup.
        assert source.exists()


class TestS3BlobStore:
    """Tests for S3BlobStore.

    Note: PyArrow's S3FileSystem uses its own C++ AWS SDK which doesn't
    work with moto mock. These tests verify the key generation logic and
    read behavior. Full S3 integration requires actual S3 or LocalStack.
    """

    @pytest.fixture
    def s3_store_mock(self):
        """Create an S3BlobStore with mocked S3 for key tests."""
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws

        with mock_aws():
            conn = boto3.client("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="test-bucket")

            store = S3BlobStore(
                bucket="test-bucket",
                prefix="artifacts",
                region="us-east-1",
            )
            yield store

    def test_s3_key_format(self, s3_store_mock: S3BlobStore):
        """Test the S3 key format includes bucket and prefix."""
        key = s3_store_mock._s3_key("abc123", 5)

        assert key == "test-bucket/artifacts/abc123@v=5.arrow"

    def test_s3_key_without_prefix(self):
        """Test S3 key format without prefix."""
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws

        with mock_aws():
            conn = boto3.client("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="test-bucket")

            store = S3BlobStore(
                bucket="test-bucket",
                prefix="",
                region="us-east-1",
            )

            key = store._s3_key("abc123", 5)
            assert key == "test-bucket/abc123@v=5.arrow"

    def test_blob_key_format(self, s3_store_mock: S3BlobStore):
        """Test the blob key format."""
        key = s3_store_mock._blob_key("abc123", 5)

        assert key == "abc123@v=5.arrow"

    def test_read_nonexistent_blob(self, s3_store_mock: S3BlobStore):
        """Test reading a blob that doesn't exist in S3.

        Note: PyArrow's S3FileSystem doesn't work with moto for writes,
        but read of nonexistent files should return None.
        """
        result = s3_store_mock.read_blob("nonexistent", 1)

        assert result is None

    def test_blob_exists_nonexistent(self, s3_store_mock: S3BlobStore):
        """Test checking if a nonexistent blob exists in S3."""
        assert not s3_store_mock.blob_exists("nonexistent", 1)

    @pytest.mark.skip(reason="Requires actual S3/LocalStack - PyArrow doesn't work with moto")
    def test_write_and_read_blob_integration(self):
        """Test writing and reading a blob from actual S3.

        This test is skipped by default. To run it, start LocalStack and set:
            STRATA_S3_ENDPOINT_URL=http://localhost:4566
            STRATA_S3_REGION=us-east-1
        """
        import os

        endpoint = os.environ.get("STRATA_S3_ENDPOINT_URL")
        if not endpoint:
            pytest.skip("STRATA_S3_ENDPOINT_URL not set")

        store = S3BlobStore(
            bucket="test-bucket",
            prefix="artifacts",
            region="us-east-1",
            endpoint_url=endpoint,
            access_key="test",
            secret_key="test",
        )

        data = b"test artifact data"
        store.write_blob("artifact-1", 1, data)
        result = store.read_blob("artifact-1", 1)

        assert result == data


class TestGCSCredentialResolution:
    """``GOOGLE_APPLICATION_CREDENTIALS`` resolves paths and nothing else.

    The setting is named ``STRATA_GCS_CREDENTIALS_JSON``, so operators paste
    key material into it — and a container deployment usually holds the
    credential as an env var rather than a mounted file. Handing that string
    straight to Google fails at first blob access, well after the deployment
    looks healthy.
    """

    def test_inline_key_material_is_written_to_a_private_file(self, tmp_path):
        from strata.blob_store import _resolve_gcs_credentials

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
        from strata.blob_store import _resolve_gcs_credentials

        key = '{"type": "service_account", "private_key": "k"}'

        first = _resolve_gcs_credentials(key)
        second = _resolve_gcs_credentials(key)
        other = _resolve_gcs_credentials('{"type": "service_account", "private_key": "j"}')

        assert first == second
        assert other != first
        assert stat.S_IMODE(Path(first).stat().st_mode) == 0o600

    def test_a_file_left_world_readable_is_tightened(self):
        # A leftover from a previous run, or a temp dir with a permissive
        # umask: the mode has to be fixed rather than assumed.
        from strata.blob_store import _resolve_gcs_credentials

        key = '{"type": "service_account", "private_key": "loose"}'
        path = Path(_resolve_gcs_credentials(key))
        path.chmod(0o644)

        _resolve_gcs_credentials(key)

        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_a_path_is_passed_through_untouched(self):
        from strata.blob_store import _resolve_gcs_credentials

        assert _resolve_gcs_credentials("/etc/strata/gcs-key.json") == "/etc/strata/gcs-key.json"

    def test_a_json_scalar_is_treated_as_a_path(self):
        # Parsed rather than sniffed for a leading brace, but a bare JSON
        # scalar still parses. A relative filename must not be mistaken for
        # key material just because json.loads accepts it.
        from strata.blob_store import _resolve_gcs_credentials

        assert _resolve_gcs_credentials("123") == "123"


class TestGCSBlobStore:
    """Tests for GCSBlobStore.

    Note: PyArrow's GcsFileSystem uses its own C++ GCS SDK which doesn't
    work with mock libraries. These tests verify the key generation logic.
    Full GCS integration requires actual GCS or fake-gcs-server.
    """

    def test_gcs_key_format(self):
        """Test the GCS key format includes bucket and prefix."""
        # Create store with anonymous access to avoid credential errors
        store = GCSBlobStore(
            bucket="test-bucket",
            prefix="artifacts",
            anonymous=True,
        )

        key = store._gcs_key("abc123", 5)

        assert key == "test-bucket/artifacts/abc123@v=5.arrow"

    def test_gcs_key_without_prefix(self):
        """Test GCS key format without prefix."""
        store = GCSBlobStore(
            bucket="test-bucket",
            prefix="",
            anonymous=True,
        )

        key = store._gcs_key("abc123", 5)
        assert key == "test-bucket/abc123@v=5.arrow"

    def test_blob_key_format(self):
        """Test the blob key format."""
        store = GCSBlobStore(
            bucket="test-bucket",
            prefix="artifacts",
            anonymous=True,
        )

        key = store._blob_key("abc123", 5)

        assert key == "abc123@v=5.arrow"

    def test_read_nonexistent_blob(self):
        """Test reading a blob that doesn't exist in GCS.

        Note: This will fail to connect to GCS but should return None
        due to exception handling.
        """
        store = GCSBlobStore(
            bucket="nonexistent-bucket-xyz123",
            prefix="artifacts",
            anonymous=True,
        )

        # This should return None due to connection/not-found errors
        result = store.read_blob("nonexistent", 1)

        assert result is None

    def test_blob_exists_nonexistent(self):
        """Test checking if a nonexistent blob exists in GCS."""
        store = GCSBlobStore(
            bucket="nonexistent-bucket-xyz123",
            prefix="artifacts",
            anonymous=True,
        )

        # This should return False due to connection/not-found errors
        assert not store.blob_exists("nonexistent", 1)

    @pytest.mark.skip(reason="Requires actual GCS or fake-gcs-server")
    def test_write_and_read_blob_integration(self):
        """Test writing and reading a blob from actual GCS.

        This test is skipped by default. To run it, start fake-gcs-server and set:
            STRATA_GCS_ENDPOINT_OVERRIDE=http://localhost:4443
            STRATA_GCS_ANONYMOUS=true

        Or use actual GCS with GOOGLE_APPLICATION_CREDENTIALS.
        """
        import os

        endpoint = os.environ.get("STRATA_GCS_ENDPOINT_OVERRIDE")
        if not endpoint:
            pytest.skip("STRATA_GCS_ENDPOINT_OVERRIDE not set")

        store = GCSBlobStore(
            bucket="test-bucket",
            prefix="artifacts",
            endpoint_override=endpoint,
            anonymous=True,
        )

        data = b"test artifact data"
        store.write_blob("artifact-1", 1, data)
        result = store.read_blob("artifact-1", 1)

        assert result == data


class TestAzureBlobStore:
    """Tests for AzureBlobStore.

    Note: The Azure SDK doesn't have great mocking support like moto.
    These tests verify the key generation logic and error handling.
    Full Azure integration requires actual Azure Storage or Azurite emulator.
    """

    def test_azure_key_format_with_prefix(self):
        """Test the Azure key format includes prefix."""
        pytest.importorskip("azure.storage.blob")
        from strata.blob_store import AzureBlobStore

        # Create a store using connection string (won't actually connect)
        # We use a fake connection string format for testing key generation
        store = AzureBlobStore(
            account_name="testaccount",
            container_name="test-container",
            prefix="artifacts",
            connection_string="DefaultEndpointsProtocol=https;AccountName=testaccount;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
        )

        key = store._azure_key("abc123", 5)

        assert key == "artifacts/abc123@v=5.arrow"

    def test_azure_key_format_without_prefix(self):
        """Test Azure key format without prefix."""
        pytest.importorskip("azure.storage.blob")
        from strata.blob_store import AzureBlobStore

        store = AzureBlobStore(
            account_name="testaccount",
            container_name="test-container",
            prefix="",
            connection_string="DefaultEndpointsProtocol=https;AccountName=testaccount;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
        )

        key = store._azure_key("abc123", 5)

        assert key == "abc123@v=5.arrow"

    def test_blob_key_format(self):
        """Test the blob key format."""
        pytest.importorskip("azure.storage.blob")
        from strata.blob_store import AzureBlobStore

        store = AzureBlobStore(
            account_name="testaccount",
            container_name="test-container",
            prefix="artifacts",
            connection_string="DefaultEndpointsProtocol=https;AccountName=testaccount;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
        )

        key = store._blob_key("abc123", 5)

        assert key == "abc123@v=5.arrow"

    def test_requires_auth_method(self):
        """Test that Azure store raises error without auth method."""
        pytest.importorskip("azure.storage.blob")
        from strata.blob_store import AzureBlobStore

        with pytest.raises(ValueError, match="requires one of"):
            AzureBlobStore(
                account_name="testaccount",
                container_name="test-container",
                prefix="artifacts",
                # No auth method provided
            )

    @pytest.mark.skip(reason="Requires actual Azure Storage or Azurite emulator")
    def test_write_and_read_blob_integration(self):
        """Test writing and reading a blob from actual Azure Storage.

        This test is skipped by default. To run it, start Azurite and set:
            STRATA_AZURE_CONNECTION_STRING=UseDevelopmentStorage=true

        Or use actual Azure Storage with a connection string.
        """
        import os

        pytest.importorskip("azure.storage.blob")
        from strata.blob_store import AzureBlobStore

        connection_string = os.environ.get("STRATA_AZURE_CONNECTION_STRING")
        if not connection_string:
            pytest.skip("STRATA_AZURE_CONNECTION_STRING not set")

        store = AzureBlobStore(
            account_name="devstoreaccount1",  # Azurite default
            container_name="test-container",
            prefix="artifacts",
            connection_string=connection_string,
        )

        data = b"test artifact data"
        store.write_blob("artifact-1", 1, data)
        result = store.read_blob("artifact-1", 1)

        assert result == data


class TestCreateBlobStore:
    """Tests for the create_blob_store factory function."""

    def test_creates_local_store_by_default(self, tmp_path: Path, monkeypatch):
        """Test that local store is created by default."""
        # Clear any env vars
        monkeypatch.delenv("STRATA_ARTIFACT_BLOB_BACKEND", raising=False)
        monkeypatch.delenv("STRATA_ARTIFACT_S3_BUCKET", raising=False)

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )

        store = create_blob_store(config)

        assert isinstance(store, LocalBlobStore)

    def test_creates_s3_store_from_env(self, tmp_path: Path, monkeypatch):
        """Test that S3 store is created when configured via env."""
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws

        with mock_aws():
            conn = boto3.client("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="my-bucket")

            monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "s3")
            monkeypatch.setenv("STRATA_ARTIFACT_S3_BUCKET", "my-bucket")
            monkeypatch.setenv("STRATA_ARTIFACT_S3_PREFIX", "custom-prefix")

            config = StrataConfig(
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                s3_region="us-east-1",
            )

            store = create_blob_store(config)

            assert isinstance(store, S3BlobStore)
            assert store.bucket == "my-bucket"
            assert store.prefix == "custom-prefix"

    def test_raises_without_s3_bucket(self, tmp_path: Path, monkeypatch):
        """Test that S3 store raises error without bucket."""
        monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "s3")
        monkeypatch.delenv("STRATA_ARTIFACT_S3_BUCKET", raising=False)

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )

        with pytest.raises(ValueError, match="S3 blob backend requires"):
            create_blob_store(config)

    def test_raises_without_artifact_dir(self, monkeypatch):
        """Test that local store raises error without artifact_dir."""
        monkeypatch.delenv("STRATA_ARTIFACT_BLOB_BACKEND", raising=False)

        config = StrataConfig(
            deployment_mode="service",
            artifact_dir=None,
        )

        with pytest.raises(ValueError, match="requires artifact_dir"):
            create_blob_store(config)

    def test_creates_gcs_store_from_env(self, tmp_path: Path, monkeypatch):
        """Test that GCS store is created when configured via env."""
        monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "gcs")
        monkeypatch.setenv("STRATA_ARTIFACT_GCS_BUCKET", "my-gcs-bucket")
        monkeypatch.setenv("STRATA_ARTIFACT_GCS_PREFIX", "custom-prefix")
        monkeypatch.setenv("STRATA_GCS_ANONYMOUS", "true")

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            gcs_anonymous=True,
        )

        store = create_blob_store(config)

        assert isinstance(store, GCSBlobStore)
        assert store.bucket == "my-gcs-bucket"
        assert store.prefix == "custom-prefix"

    def test_raises_without_gcs_bucket(self, tmp_path: Path, monkeypatch):
        """Test that GCS store raises error without bucket."""
        monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "gcs")
        monkeypatch.delenv("STRATA_ARTIFACT_GCS_BUCKET", raising=False)

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )

        with pytest.raises(ValueError, match="GCS blob backend requires"):
            create_blob_store(config)

    def test_creates_azure_store_from_env(self, tmp_path: Path, monkeypatch):
        """Test that Azure store is created when configured via env."""
        pytest.importorskip("azure.storage.blob")
        from strata.blob_store import AzureBlobStore

        monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "azure")
        monkeypatch.setenv("STRATA_ARTIFACT_AZURE_CONTAINER", "my-container")
        monkeypatch.setenv("STRATA_ARTIFACT_AZURE_PREFIX", "custom-prefix")

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            azure_connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
        )

        store = create_blob_store(config)

        assert isinstance(store, AzureBlobStore)
        assert store.container_name == "my-container"
        assert store.prefix == "custom-prefix"

    def test_raises_without_azure_container(self, tmp_path: Path, monkeypatch):
        """Test that Azure store raises error without container."""
        monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "azure")
        monkeypatch.delenv("STRATA_ARTIFACT_AZURE_CONTAINER", raising=False)

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )

        with pytest.raises(ValueError, match="Azure blob backend requires"):
            create_blob_store(config)


class TestConfigCreateBlobStore:
    """Tests for StrataConfig.create_blob_store() method."""

    def test_creates_local_store(self, tmp_path: Path):
        """Test creating local store from config."""
        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="local",
        )

        store = config.create_blob_store()

        assert isinstance(store, LocalBlobStore)

    def test_creates_s3_store(self, tmp_path: Path):
        """Test creating S3 store from config."""
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws

        with mock_aws():
            conn = boto3.client("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="my-bucket")

            config = StrataConfig(
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                artifact_blob_backend="s3",
                artifact_s3_bucket="my-bucket",
                artifact_s3_prefix="my-prefix",
                s3_region="us-east-1",
            )

            store = config.create_blob_store()

            assert isinstance(store, S3BlobStore)
            assert store.bucket == "my-bucket"
            assert store.prefix == "my-prefix"

    def test_raises_without_s3_bucket(self, tmp_path: Path):
        """Test that S3 store raises error without bucket in config."""
        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="s3",
            artifact_s3_bucket=None,
        )

        with pytest.raises(ValueError, match="requires artifact_s3_bucket"):
            config.create_blob_store()

    def test_creates_gcs_store(self, tmp_path: Path):
        """Test creating GCS store from config."""
        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="gcs",
            artifact_gcs_bucket="my-gcs-bucket",
            artifact_gcs_prefix="my-prefix",
            gcs_anonymous=True,
        )

        store = config.create_blob_store()

        assert isinstance(store, GCSBlobStore)
        assert store.bucket == "my-gcs-bucket"
        assert store.prefix == "my-prefix"

    def test_raises_without_gcs_bucket(self, tmp_path: Path):
        """Test that GCS store raises error without bucket in config."""
        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="gcs",
            artifact_gcs_bucket=None,
        )

        with pytest.raises(ValueError, match="requires artifact_gcs_bucket"):
            config.create_blob_store()

    def test_creates_azure_store(self, tmp_path: Path):
        """Test creating Azure store from config."""
        pytest.importorskip("azure.storage.blob")
        from strata.blob_store import AzureBlobStore

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="azure",
            artifact_azure_container="my-container",
            artifact_azure_prefix="my-prefix",
            azure_connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
        )

        store = config.create_blob_store()

        assert isinstance(store, AzureBlobStore)
        assert store.container_name == "my-container"
        assert store.prefix == "my-prefix"

    def test_raises_without_azure_container(self, tmp_path: Path):
        """Test that Azure store raises error without container in config."""
        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="azure",
            artifact_azure_container=None,
        )

        with pytest.raises(ValueError, match="requires artifact_azure_container"):
            config.create_blob_store()


class TestConfiguredBackendIsActuallyWired:
    """The configured blob backend must reach the artifact store.

    ``ArtifactStore`` falls back to ``LocalBlobStore`` whenever ``blob_store``
    is omitted, and every production call site omitted it — so
    ``create_blob_store`` had no caller outside tests and
    ``STRATA_ARTIFACT_BLOB_BACKEND=s3`` silently wrote every artifact to local
    disk. Nothing errored; the bucket stayed empty and blobs vanished with the
    pod.
    """

    def _init(self, config):
        from strata.artifact_store import reset_artifact_store
        from strata.server import _init_configured_artifact_store

        reset_artifact_store()
        try:
            _init_configured_artifact_store(config)
            from strata.artifact_store import get_artifact_store

            return get_artifact_store(config.artifact_dir)
        finally:
            pass

    def test_s3_backend_reaches_the_artifact_store(self, tmp_path: Path):
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws

        from strata.artifact_store import reset_artifact_store

        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="wired-bucket")
            config = StrataConfig(
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                artifact_blob_backend="s3",
                artifact_s3_bucket="wired-bucket",
                s3_region="us-east-1",
            )
            try:
                store = self._init(config)
                assert isinstance(store.blob_store, S3BlobStore)
                assert store.blob_store.bucket == "wired-bucket"
            finally:
                reset_artifact_store()

    def test_local_backend_is_unchanged(self, tmp_path: Path):
        from strata.artifact_store import reset_artifact_store

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )
        assert config.artifact_blob_backend == "local"
        try:
            store = self._init(config)
            assert isinstance(store.blob_store, LocalBlobStore)
        finally:
            reset_artifact_store()

    def test_misconfigured_backend_raises_rather_than_degrading(self, tmp_path: Path):
        """Silently falling back to local disk is what made the
        misconfiguration invisible — and it loses every artifact when the pod
        is replaced. Fail the startup instead."""
        from strata.artifact_store import reset_artifact_store

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="s3",
            artifact_s3_bucket=None,
        )
        try:
            with pytest.raises(ValueError, match="requires artifact_s3_bucket"):
                self._init(config)
        finally:
            reset_artifact_store()


class TestBackendFailuresAreVisible:
    """``blob_exists`` / ``blob_size`` / ``delete_blob`` swallowed every
    exception and returned a confident answer — "absent", "unknown", "nothing
    deleted" — so a transient backend error was indistinguishable from fact.

    ``delete_blob`` is the worst of the three: GC and ``delete_artifact``
    remove the metadata row regardless, so a silent False orphans the object
    with no row left to ever retry it.

    The logger is monkeypatched rather than read through ``caplog``: the
    package configures its own logging, so records do not reliably reach
    pytest's capture handler.
    """

    def _store(self, *, info_error=None, delete_error=None):
        import pyarrow.fs as pafs

        store = S3BlobStore.__new__(S3BlobStore)
        store.bucket = "b"
        store.prefix = "p"

        class _Fs:
            def get_file_info(self, key):
                if info_error is not None:
                    raise info_error
                return SimpleNamespace(type=pafs.FileType.File, size=123)

            def delete_file(self, key):
                if delete_error is not None:
                    raise delete_error

        store._fs = _Fs()
        return store

    def _capture(self, monkeypatch):
        messages: list[str] = []
        monkeypatch.setattr(
            blob_store_module.logger,
            "exception",
            lambda msg, *a, **k: messages.append(msg % a if a else msg),
        )
        return messages

    def test_blob_exists_logs_the_backend_error(self, monkeypatch):
        messages = self._capture(monkeypatch)
        store = self._store(info_error=OSError("connection reset"))

        assert store.blob_exists("a", 1) is False
        assert any("blob_exists failed" in m for m in messages)

    def test_blob_size_logs_the_backend_error(self, monkeypatch):
        messages = self._capture(monkeypatch)
        store = self._store(info_error=OSError("connection reset"))

        assert store.blob_size("a", 1) is None
        assert any("blob_size failed" in m for m in messages)

    def test_delete_blob_logs_the_orphaned_object(self, monkeypatch):
        """The object exists, so the delete is attempted — and its failure is
        what leaves an object no metadata row will ever point at again."""
        messages = self._capture(monkeypatch)
        store = self._store(delete_error=OSError("connection reset"))

        assert store.delete_blob("a", 1) is False
        assert any("orphaned" in m for m in messages)

    def test_a_healthy_backend_logs_nothing(self, monkeypatch):
        messages = self._capture(monkeypatch)
        store = self._store()

        assert store.blob_exists("a", 1) is True
        assert store.blob_size("a", 1) == 123
        assert store.delete_blob("a", 1) is True
        assert messages == []
