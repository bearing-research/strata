"""Replay ArtifactLifecycle.tla's counterexamples, plus an assumption check.

A TLC trace is a claim about the model. Running the same steps against
the real code shows the code behaves the same way. Each test passes while
the bug exists (asserts the violating outcome). Once a fix lands, invert
the assertion and move the test into ``tests/``.

    uv run pytest formal/ -v
"""

from __future__ import annotations

from pathlib import Path

from strata.artifact_store import ArtifactStore
from strata.types import CacheKey


def _store(tmp: Path) -> ArtifactStore:
    return ArtifactStore(tmp / "artifacts")


def _build(store: ArtifactStore, artifact_id: str, prov: str) -> int:
    """create_artifact -> write_blob -> finalize_artifact."""
    v = store.create_artifact(artifact_id, prov)
    store.blob_store.write_blob(artifact_id, v, b"bytes")
    store.finalize_artifact(artifact_id, v, "", 1, 5)
    return v


def test_cross_id_dedup(tmp_path: Path) -> None:
    """TLA+ config Artifact_CrossIdDedup.

    Trace: two notebook cells with the same provenance (same source,
    inputs and lockfile, e.g. a duplicated notebook). A's output is
    ready. B finalizes, is deduped to A and marked 'failed', and the
    notebook path calls force_finalize_canonical(B). That supersedes A's
    row, and get_latest_version(A), which is how A's downstream cells load
    their input, now returns None.
    """
    store = _store(tmp_path)
    a, b = "nb_A_cell_c1_var_df", "nb_B_cell_c1_var_df"
    _build(store, a, "same-prov")
    vb = store.create_artifact(b, "same-prov")
    store.blob_store.write_blob(b, vb, b"bytes")
    got = store.finalize_artifact(b, vb, "", 1, 5)
    assert got is not None and got.id == a, "expected dedup to A"
    store.force_finalize_canonical(b, vb, "", 1, 5)  # what store_cell_output does
    assert store.get_latest_version(a) is None
    assert store.get_latest_version(b) is not None


def test_projection_fingerprint_collision() -> None:
    """Injectivity check for the row-group cache key (outside the TLA+ model).

    The fingerprint hashes ``",".join(columns)``, so projecting one column
    named ``"a,b"`` and projecting the two columns ``"a"`` and ``"b"`` give
    the same cache key. Iceberg and Parquet both allow a comma in a column
    name.
    """
    fp = CacheKey.compute_projection_fingerprint
    assert fp(["a,b"]) == fp(["a", "b"])
