"""Tests for the ``strata artifact`` inspection CLI (list/show/lineage/pull)."""

from __future__ import annotations

import argparse
import json

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from strata.artifact_cli import cmd_lineage, cmd_list, cmd_pull, cmd_show
from strata.artifact_store import ArtifactStore, TransformSpec, reset_artifact_store


def _ipc_bytes(num_rows: int) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, pa.schema([("id", pa.int64())])) as writer:
        writer.write_batch(pa.RecordBatch.from_pydict({"id": list(range(num_rows))}))
    return sink.getvalue().to_pybytes()


@pytest.fixture
def chain_store(tmp_path):
    """Store with a 3-level chain: model <- features <- scan <- table."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = ArtifactStore(artifact_dir)

    def make(artifact_id, provenance, executor, inputs, input_versions, rows, name=None):
        store.create_artifact(
            artifact_id,
            provenance,
            transform_spec=TransformSpec(executor=executor, params={}, inputs=inputs),
            input_versions=input_versions,
        )
        store.write_blob(artifact_id, 1, _ipc_bytes(rows))
        store.finalize_artifact(artifact_id, 1, "{}", rows, 100)
        if name:
            store.set_name(name, artifact_id, 1)

    make(
        "scan-1",
        "prov-scan",
        "scan@v1",
        ["file:///wh#db.events"],
        {"file:///wh#db.events": "111222333"},
        100,
        name="demo/raw",
    )
    make(
        "feat-1",
        "prov-feat",
        "feature_eng@v1",
        ["strata://artifact/scan-1@v=1"],
        {"strata://artifact/scan-1@v=1": "scan-1@v=1"},
        50,
    )
    make(
        "model-1",
        "prov-model",
        "train@v1",
        ["strata://artifact/feat-1@v=1"],
        {"strata://artifact/feat-1@v=1": "feat-1@v=1"},
        1,
        name="demo/model",
    )

    yield {"dir": str(artifact_dir), "store": store}
    reset_artifact_store()


def _args(**kwargs) -> argparse.Namespace:
    defaults = {"artifact_dir": None, "format": "human"}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestList:
    def test_human_listing(self, chain_store, capsys):
        rc = cmd_list(_args(artifact_dir=chain_store["dir"], state=None, limit=50))
        out = capsys.readouterr().out
        assert rc == 0
        assert "model-1" in out
        assert "demo/model" in out
        assert "scan-1" in out

    def test_json_listing(self, chain_store, capsys):
        rc = cmd_list(_args(artifact_dir=chain_store["dir"], state=None, limit=50, format="json"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        ids = {a["artifact_id"] for a in payload}
        assert {"scan-1", "feat-1", "model-1"} <= ids

    def test_missing_dir_exits_two(self, tmp_path):
        assert cmd_list(_args(artifact_dir=str(tmp_path / "nope"), state=None, limit=5)) == 2


class TestShow:
    def test_show_by_name(self, chain_store, capsys):
        rc = cmd_show(_args(ref="demo/model", artifact_dir=chain_store["dir"]))
        out = capsys.readouterr().out
        assert rc == 0
        assert "model-1@v=1" in out
        assert "train@v1" in out
        assert "demo/model" in out

    def test_show_by_id_version(self, chain_store, capsys):
        rc = cmd_show(_args(ref="feat-1@v=1", artifact_dir=chain_store["dir"]))
        assert rc == 0
        assert "feat-1@v=1" in capsys.readouterr().out

    def test_show_by_bare_id_latest(self, chain_store, capsys):
        rc = cmd_show(_args(ref="scan-1", artifact_dir=chain_store["dir"]))
        assert rc == 0
        assert "scan-1@v=1" in capsys.readouterr().out

    def test_unknown_ref_exits_one(self, chain_store):
        assert cmd_show(_args(ref="ghost", artifact_dir=chain_store["dir"])) == 1


class TestLineage:
    def test_renders_full_chain_to_snapshot(self, chain_store, capsys):
        rc = cmd_lineage(_args(ref="demo/model", artifact_dir=chain_store["dir"], max_depth=10))
        out = capsys.readouterr().out
        assert rc == 0
        # model -> features -> scan -> table @ snapshot, in order
        assert out.index("model-1@v=1") < out.index("feat-1@v=1") < out.index("scan-1@v=1")
        assert "table file:///wh#db.events  @ snapshot 111222333" in out

    def test_json_tree(self, chain_store, capsys):
        rc = cmd_lineage(
            _args(ref="demo/model", artifact_dir=chain_store["dir"], max_depth=10, format="json")
        )
        assert rc == 0
        tree = json.loads(capsys.readouterr().out)
        assert tree["artifact_id"] == "model-1"
        leaf = tree["inputs"][0]["inputs"][0]["inputs"][0]
        assert leaf == {"uri": "file:///wh#db.events", "version": "111222333"}

    def test_max_depth_cuts_recursion(self, chain_store, capsys):
        rc = cmd_lineage(
            _args(ref="demo/model", artifact_dir=chain_store["dir"], max_depth=1, format="json")
        )
        assert rc == 0
        tree = json.loads(capsys.readouterr().out)
        # depth 1: features expanded, scan stays a URI leaf
        feat = tree["inputs"][0]
        assert feat["artifact_id"] == "feat-1"
        assert feat["inputs"][0]["uri"] == "strata://artifact/scan-1@v=1"


class TestPull:
    def test_pull_by_name_to_path(self, chain_store, tmp_path, capsys):
        out_file = tmp_path / "model.arrow"
        rc = cmd_pull(
            _args(ref="demo/model", artifact_dir=chain_store["dir"], to=str(out_file))
        )
        assert rc == 0
        table = ipc.open_stream(out_file.read_bytes()).read_all()
        assert table.num_rows == 1

    def test_pull_unknown_exits_one(self, chain_store, tmp_path):
        rc = cmd_pull(_args(ref="ghost", artifact_dir=chain_store["dir"], to=str(tmp_path / "x")))
        assert rc == 1


class TestTenantAgnosticResolution:
    def test_legacy_default_tenant_name_resolves(self, tmp_path, capsys):
        """A name written under legacy '_default' is still findable by the CLI."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        store = ArtifactStore(artifact_dir)
        store.create_artifact("legacy-1", "prov-x", tenant="_default")
        store.write_blob("legacy-1", 1, _ipc_bytes(2))
        store.finalize_artifact("legacy-1", 1, "{}", 2, 100)
        store.set_name("old/model", "legacy-1", 1, tenant="_default")

        rc = cmd_show(_args(ref="old/model", artifact_dir=str(artifact_dir)))
        out = capsys.readouterr().out
        assert rc == 0
        assert "legacy-1@v=1" in out
        reset_artifact_store()
