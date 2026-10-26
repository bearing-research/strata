"""The per-store steps Strata completes in a ledger it opens (artifact/upgrades.py)."""

from __future__ import annotations

from vestigo.provenance import TransformSpec
from vestigo.store import ArtifactStore

from strata.artifact.upgrades import (
    MISBOUND_BUILDS_UPGRADE,
    complete_upgrade,
    fail_misbound_builds,
)


def _ready(store, artifact_id, executor, inputs, input_versions=None) -> None:
    spec = TransformSpec(executor=executor, params={"sql": "SELECT 1"}, inputs=inputs)
    version = store.create_artifact(
        artifact_id, f"hash-{artifact_id}", transform_spec=spec, input_versions=input_versions
    )
    store.finalize_artifact(artifact_id, version, "{}", 1, 1)


class TestMisboundBuilds:
    def test_fails_embedded_builds_over_eleven_inputs_and_what_read_them(self, tmp_path):
        """Their bytes bound input10 before input2; anything built over them inherits that."""
        store = ArtifactStore(tmp_path / "s")
        twelve = [f"strata://artifact/in{i}@v=1" for i in range(12)]
        _ready(store, "wide", "local://duckdb_sql@v1", twelve)
        _ready(store, "wide_bare", "duckdb_sql@v1", twelve[:11])
        _ready(
            store,
            "reader",
            "local://duckdb_sql@v1",
            ["strata://name/wide"],
            {"strata://name/wide": "wide@v=1"},
        )
        _ready(store, "narrow", "local://duckdb_sql@v1", twelve[:10])
        _ready(store, "remote", "pandas_script@v1", twelve)

        fail_misbound_builds(store)

        states = {a: store.get_artifact(a, 1).state for a in ("wide", "wide_bare", "reader")}
        assert states == {"wide": "failed", "wide_bare": "failed", "reader": "failed"}
        assert store.find_by_provenance("hash-wide") is None
        assert store.find_by_provenance("hash-reader") is None
        assert store.find_by_provenance("hash-narrow") is not None
        assert store.find_by_provenance("hash-remote") is not None
        assert store.upgrade_applied(MISBOUND_BUILDS_UPGRADE)

    def test_runs_once_per_store(self, tmp_path):
        """A wide build stored after the mark is the new runner's, bound in order."""
        store = ArtifactStore(tmp_path / "s")
        complete_upgrade(store)
        twelve = [f"strata://artifact/in{i}@v=1" for i in range(12)]
        _ready(store, "wide", "local://duckdb_sql@v1", twelve)

        complete_upgrade(store)

        assert store.get_artifact("wide", 1).state == "ready"
