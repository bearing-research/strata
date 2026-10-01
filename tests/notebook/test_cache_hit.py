"""Unit tests for the notebook artifact cache."""

from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.notebook.provenance import compute_provenance_hash, compute_source_hash


def test_artifact_manager_initialization(tmp_path):
    mgr = NotebookArtifactManager("test_nb", artifact_dir=tmp_path)
    assert mgr.notebook_id == "test_nb"
    assert mgr.artifact_store is not None


def test_artifact_manager_find_cached_returns_none_for_empty_store(tmp_path):
    mgr = NotebookArtifactManager("test_nb", artifact_dir=tmp_path)
    result = mgr.find_cached("nonexistent_hash")
    assert result is None


def test_artifact_manager_store_and_load(tmp_path):
    mgr = NotebookArtifactManager("test_nb", artifact_dir=tmp_path)

    source_hash = compute_source_hash("x = 1")
    env_hash = compute_source_hash("env")
    prov_hash = compute_provenance_hash([], source_hash, env_hash)

    blob_data = b"test data"
    artifact = mgr.store_cell_output(
        cell_id="cell1",
        variable_name="x",
        blob_data=blob_data,
        content_type="json/object",
        provenance_hash=prov_hash,
    )

    assert artifact is not None
    assert artifact.state == "ready"

    loaded = mgr.load_artifact_data(artifact.id, artifact.version)
    assert loaded == blob_data
