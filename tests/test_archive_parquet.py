"""A tabular artifact's bundle carries Parquet beside the archived bytes.

Arrow IPC is a transport format. It has a stability promise, but a data
repository indexes Parquet, and a reader in a decade reaches for it with
whatever tool they have. The bundle exists to be opened long after anyone can
ask how — item 30.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

from strata.artifact_cli import cmd_archive
from strata.artifact_store import ArtifactStore

TABLE = pa.table({"id": [1, 2, 3], "value": [0.1, 0.2, 0.3], "name": ["a", "b", "c"]})


def _arrow_bytes(table: pa.Table = TABLE) -> bytes:
    buffer = io.BytesIO()
    with ipc.new_stream(buffer, table.schema) as writer:
        writer.write_table(table)
    return buffer.getvalue()


def _stored(store: ArtifactStore, artifact_id: str, payload: bytes) -> int:
    version = store.create_artifact(artifact_id, hashlib.sha256(payload).hexdigest())
    with store.open_blob_writer(artifact_id, version) as writer:
        writer.write(payload)
    store.finalize_artifact(
        artifact_id, version, schema_json="", row_count=0, byte_size=len(payload)
    )
    return version


def _archive(tmp_path, artifact_id: str, payload: bytes):
    store = ArtifactStore(tmp_path / "store")
    _stored(store, artifact_id, payload)
    dest = tmp_path / "bundle"
    rc = cmd_archive(
        argparse.Namespace(
            ref=artifact_id,
            artifact_dir=str(tmp_path / "store"),
            to=str(dest),
            force=False,
            title="Table 1",
            author=None,
            tenant=None,
            max_depth=10,
        )
    )
    assert rc == 0
    return dest


class TestTabularArtifact:
    def test_the_bundle_carries_parquet(self, tmp_path):
        dest = _archive(tmp_path, "rows", _arrow_bytes())

        assert (dest / "artifact.parquet").exists()

    def test_the_parquet_holds_the_same_rows(self, tmp_path):
        """A second rendering, not a second dataset."""
        dest = _archive(tmp_path, "rows", _arrow_bytes())

        assert pq.read_table(dest / "artifact.parquet").equals(TABLE)

    def test_the_archived_arrow_bytes_are_still_there(self, tmp_path):
        """Parquet is added beside them, not instead of them.

        The manifest's digest covers the archived bytes; replacing them would
        make that digest describe a file the bundle no longer contains.
        """
        dest = _archive(tmp_path, "rows", _arrow_bytes())
        manifest = json.loads((dest / "manifest.json").read_text())

        with open(dest / manifest["content_file"], "rb") as handle:
            assert ipc.open_stream(handle).read_all().equals(TABLE)

    def test_the_manifest_names_its_payload(self, tmp_path):
        """With two files present, an unnamed digest is a claim about nothing."""
        dest = _archive(tmp_path, "rows", _arrow_bytes())
        manifest = json.loads((dest / "manifest.json").read_text())

        named = dest / manifest["content_file"]
        assert named.exists()
        assert named.name != "artifact.parquet"
        assert manifest["content_sha256"] == hashlib.sha256(named.read_bytes()).hexdigest()

    def test_the_manifest_says_which_digest_covers_which_file(self, tmp_path):
        """With one payload "the digest" was unambiguous. With two it is not.

        A digest that does not say what it covers is worse than none in a
        bundle meant to be read when nobody is left to ask.
        """
        dest = _archive(tmp_path, "rows", _arrow_bytes())
        manifest = json.loads((dest / "manifest.json").read_text())

        extra = manifest["additional_files"]
        assert len(extra) == 1
        assert extra[0]["file"] == "artifact.parquet"
        assert extra[0]["content_type"] == "application/vnd.apache.parquet"
        assert (
            extra[0]["sha256"]
            == hashlib.sha256((dest / "artifact.parquet").read_bytes()).hexdigest()
        )

    def test_the_readme_names_it(self, tmp_path):
        """The bundle explains itself to whoever opens it."""
        dest = _archive(tmp_path, "rows", _arrow_bytes())
        readme = (dest / "README.md").read_text()

        assert "artifact.parquet" in readme


class TestNonTabularArtifact:
    @pytest.mark.parametrize("payload", [b"\x89PNG\r\n\x1a\n figure", b"not arrow at all"])
    def test_no_parquet_is_written(self, tmp_path, payload):
        """An image or a pickle has no rows. A zero-row Parquet would be a
        confusing lie about what the bundle holds."""
        dest = _archive(tmp_path, "fig", payload)

        assert not (dest / "artifact.parquet").exists()

    def test_the_manifest_claims_no_extra_files(self, tmp_path):
        dest = _archive(tmp_path, "fig", b"\x89PNG\r\n\x1a\n figure")
        manifest = json.loads((dest / "manifest.json").read_text())

        assert "additional_files" not in manifest
