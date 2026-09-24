"""Replay ArtifactLifecycle.tla's counterexamples, plus two assumption checks.

A TLC trace is a claim about the model. Running the same steps against
the real code shows the code behaves the same way. Each test passes while
the bug exists (asserts the violating outcome). Once a fix lands, invert
the assertion and move the test into ``tests/``.

    uv run pytest formal/ -v
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from strata.artifact_store import ArtifactStore
from strata.filters import Filter, FilterOp
from strata.planner import ReadPlanner
from strata.types import CacheKey


def _store(tmp: Path) -> ArtifactStore:
    return ArtifactStore(tmp / "artifacts")


def _build(store: ArtifactStore, artifact_id: str, prov: str) -> int:
    """create_artifact -> write_blob -> finalize_artifact."""
    v = store.create_artifact(artifact_id, prov)
    store.blob_store.write_blob(artifact_id, v, b"bytes")
    store.finalize_artifact(artifact_id, v, "", 1, 5)
    return v


def test_gc_during_rebuild(tmp_path: Path) -> None:
    """TLA+ config Artifact_GCRebuild.

    Trace: a@v1 ready -> create a@v2 (rebuild starts) -> garbage_collect
    deletes a@v1, because it is no longer MAX(version) -> the rebuild
    fails -> ``a`` has no ready version, so get_latest_version returns None.
    """
    store = _store(tmp_path)
    _build(store, "nb_x_cell_c1_var_df", "p1")
    v2 = store.create_artifact("nb_x_cell_c1_var_df", "p2")
    store.garbage_collect(max_age_days=0)  # "old enough" = anything, as in the model
    store.fail_artifact("nb_x_cell_c1_var_df", v2)
    assert store.get_latest_version("nb_x_cell_c1_var_df") is None


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


def test_nan_ne_pruning(tmp_path: Path) -> None:
    """Assumption check for _should_prune_row_group (outside the TLA+ model).

    Filter.matches_stats is sound *if* [min, max] bounds every value in
    the row group. Parquet writers leave NaN out of min/max, so a row group
    holding [5.0, NaN] has min == max == 5.0. The filter ``value != 5.0``
    then prunes it, and the NaN row is dropped, even though
    ``NaN != 5.0`` is true in Arrow compute, DuckDB and Python.
    """
    path = tmp_path / "nan.parquet"
    pq.write_table(pa.table({"value": [5.0, float("nan")]}), path)
    rg = pq.ParquetFile(path).metadata.row_group(0)
    planner = ReadPlanner.__new__(ReadPlanner)  # the method only uses _convert_stats
    assert planner._should_prune_row_group(rg, [(0, Filter("value", FilterOp.NE, 5.0))])


def test_projection_fingerprint_collision() -> None:
    """Injectivity check for the row-group cache key (outside the TLA+ model).

    The fingerprint hashes ``",".join(columns)``, so projecting one column
    named ``"a,b"`` and projecting the two columns ``"a"`` and ``"b"`` give
    the same cache key. Iceberg and Parquet both allow a comma in a column
    name.
    """
    fp = CacheKey.compute_projection_fingerprint
    assert fp(["a,b"]) == fp(["a", "b"])
