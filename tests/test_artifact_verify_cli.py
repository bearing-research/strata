"""Tests for the artifact store: lifecycle, provenance dedup, names, blob I/O, cleanup."""

import json

import pytest
from vestigo.store import (
    ArtifactStore,
    reset_artifact_store,
)


def _ipc_bytes(num_rows: int) -> bytes:
    """A single valid Arrow IPC stream with ``num_rows`` rows."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, pa.schema([("id", pa.int64())])) as writer:
        writer.write_batch(pa.RecordBatch.from_pydict({"id": list(range(num_rows))}))
    return sink.getvalue().to_pybytes()


@pytest.fixture
def artifact_dir(tmp_path):
    """Create a temporary artifact directory."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    yield artifact_dir
    reset_artifact_store()


@pytest.fixture
def store(artifact_dir):
    """An artifact store."""
    return ArtifactStore(artifact_dir)


class TestArtifactVerifyCli:
    """`strata artifact verify` surfaces store inconsistencies."""

    def _run(self, artifact_dir, fmt="human"):
        import argparse

        from strata.artifact.cli import cmd_verify

        args = argparse.Namespace(artifact_dir=str(artifact_dir), format=fmt)
        return cmd_verify(args)

    def test_clean_store_exits_zero(self, store, artifact_dir, capsys):
        store.create_artifact("good", "prov-g")
        store.write_blob("good", 1, _ipc_bytes(5))
        store.finalize_artifact("good", 1, "{}", 5, 100)

        assert self._run(artifact_dir) == 0
        assert "consistent" in capsys.readouterr().out

    def test_problems_exit_one_with_json(self, store, artifact_dir, capsys):
        store.create_artifact("bad", "prov-b")
        store.write_blob("bad", 1, _ipc_bytes(3) + _ipc_bytes(3))
        store.finalize_artifact("bad", 1, "{}", 6, 100)

        assert self._run(artifact_dir, fmt="json") == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["findings"][0]["problem"] == "invalid_stream"

    def test_missing_dir_exits_two(self, tmp_path):
        assert self._run(tmp_path / "nope") == 2

    def test_it_names_the_metadata_and_blobs_it_checks(self, store, artifact_dir, capsys):
        assert self._run(artifact_dir) == 0
        out = capsys.readouterr().out
        assert f"metadata {store.db_path}" in out
        assert f"blobs {store.blob_store.blobs_dir}" in out

    def test_a_remote_store_is_named_by_its_dsn_and_bucket(self, tmp_path):
        """The artifact dir is not what a Postgres + S3 store verifies."""
        from vestigo.blob_store import S3BlobStore
        from vestigo.sql_backend import PostgresDialect

        from strata.artifact.cli import _store_location

        remote = ArtifactStore.__new__(ArtifactStore)
        remote._dialect = PostgresDialect("postgresql://strata:s3cret@db:5432/strata")
        remote.blob_store = S3BlobStore("lake", prefix="artifacts", region="us-east-1")

        metadata, blobs = _store_location(remote)

        assert metadata == "postgresql://strata:***@db:5432/strata"
        assert blobs == "s3://lake/artifacts"
