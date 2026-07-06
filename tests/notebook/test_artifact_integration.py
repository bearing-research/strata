"""Tests for NotebookArtifactManager — the notebook/artifact-store bridge.

Focuses on the per-iteration artifact id scheme introduced for loop cells;
regular single-artifact behaviour is exercised implicitly by the executor
and cache-hit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strata.notebook.artifact_integration import NotebookArtifactManager


@pytest.fixture
def manager(tmp_path: Path) -> NotebookArtifactManager:
    return NotebookArtifactManager("nb1", artifact_dir=tmp_path / "artifacts")


class TestCellArtifactId:
    """Canonical artifact id formatting."""

    def test_regular_artifact_id_has_no_iteration_suffix(self, manager):
        assert manager.cell_artifact_id("c1", "state") == "nb_nb1_cell_c1_var_state"

    def test_iteration_artifact_id_has_suffix(self, manager):
        assert manager.cell_artifact_id("c1", "state", 3) == "nb_nb1_cell_c1_var_state@iter=3"

    def test_iteration_zero_gets_suffix(self, manager):
        """iteration=0 is distinct from None — we still want ``@iter=0`` visible."""
        assert manager.cell_artifact_id("c1", "state", 0) == "nb_nb1_cell_c1_var_state@iter=0"


class TestPerIterationArtifacts:
    """Storing and reading per-iteration carry artifacts."""

    def test_store_and_load_iteration_blob(self, manager):
        manager.store_cell_output(
            cell_id="c1",
            variable_name="state",
            blob_data=b"iter-0-bytes",
            content_type="pickle/object",
            provenance_hash="prov-0",
            iteration=0,
        )

        assert manager.load_iteration_blob("c1", "state", 0) == b"iter-0-bytes"

    def test_iterations_are_independent_artifacts(self, manager):
        for k in range(3):
            manager.store_cell_output(
                cell_id="c1",
                variable_name="state",
                blob_data=f"iter-{k}-bytes".encode(),
                content_type="pickle/object",
                provenance_hash=f"prov-{k}",
                iteration=k,
            )

        for k in range(3):
            assert manager.load_iteration_blob("c1", "state", k) == f"iter-{k}-bytes".encode()

    def test_load_missing_iteration_returns_none(self, manager):
        assert manager.load_iteration_blob("c1", "state", 0) is None

    def test_iteration_artifact_does_not_collide_with_regular(self, manager):
        """Storing ``state`` both without and with an iteration suffix must
        produce two distinct artifacts so a cell's one-shot output is never
        overwritten by a loop cell's iteration 0."""
        manager.store_cell_output(
            cell_id="c1",
            variable_name="state",
            blob_data=b"one-shot",
            content_type="pickle/object",
            provenance_hash="prov-one-shot",
        )
        manager.store_cell_output(
            cell_id="c1",
            variable_name="state",
            blob_data=b"iter-0",
            content_type="pickle/object",
            provenance_hash="prov-iter-0",
            iteration=0,
        )

        assert manager.load_iteration_blob("c1", "state", 0) == b"iter-0"

        regular_id = manager.cell_artifact_id("c1", "state")
        regular_latest = manager.artifact_store.get_latest_version(regular_id)
        assert regular_latest is not None
        assert regular_latest.provenance_hash == "prov-one-shot"

    def test_get_iteration_artifact_returns_latest_ready_version(self, manager):
        manager.store_cell_output(
            cell_id="c1",
            variable_name="state",
            blob_data=b"first",
            content_type="pickle/object",
            provenance_hash="prov-first",
            iteration=0,
        )
        artifact = manager.get_iteration_artifact("c1", "state", 0)
        assert artifact is not None
        assert artifact.state == "ready"

    def test_list_iterations_returns_sorted_pairs(self, manager):
        """``list_iterations`` yields ``(k, ArtifactVersion)`` in ascending
        order, regardless of the order artifacts were written in."""
        for k in [2, 0, 5, 1]:
            manager.store_cell_output(
                cell_id="c1",
                variable_name="state",
                blob_data=f"iter-{k}".encode(),
                content_type="pickle/object",
                provenance_hash=f"prov-{k}",
                iteration=k,
            )

        pairs = manager.list_iterations("c1", "state")
        assert [k for k, _ in pairs] == [0, 1, 2, 5]
        for k, artifact in pairs:
            assert artifact.state == "ready"
            assert artifact.id.endswith(f"@iter={k}")

    def test_list_iterations_skips_non_iteration_artifacts(self, manager):
        """A regular ``store_cell_output`` (no iteration) does not appear
        in the iteration list — the id lacks the ``@iter=`` suffix."""
        manager.store_cell_output(
            cell_id="c1",
            variable_name="state",
            blob_data=b"one-shot",
            content_type="pickle/object",
            provenance_hash="prov-one-shot",
        )
        manager.store_cell_output(
            cell_id="c1",
            variable_name="state",
            blob_data=b"iter-0",
            content_type="pickle/object",
            provenance_hash="prov-iter-0",
            iteration=0,
        )

        pairs = manager.list_iterations("c1", "state")
        assert [k for k, _ in pairs] == [0]

    def test_list_iterations_empty_for_unknown_cell(self, manager):
        assert manager.list_iterations("c1", "state") == []

    def test_transform_spec_records_iteration(self, manager):
        """The stored transform_spec should carry the iteration index so
        other subsystems (inspector, diagnostics) can read it back without
        parsing the artifact id."""
        import json as _json

        manager.store_cell_output(
            cell_id="c1",
            variable_name="state",
            blob_data=b"iter-7",
            content_type="pickle/object",
            provenance_hash="prov-7",
            iteration=7,
        )

        artifact_id = manager.cell_artifact_id("c1", "state", 7)
        artifact = manager.artifact_store.get_latest_version(artifact_id)
        assert artifact is not None
        spec = _json.loads(artifact.transform_spec or "{}")
        assert spec.get("params", {}).get("iteration") == "7"


class TestListCellArtifacts:
    """``list_cell_artifacts`` was a NotImplementedError stub; now it works."""

    def test_lists_each_variable_once(self, manager):
        manager.store_cell_output(
            cell_id="c1",
            variable_name="x",
            blob_data=b"x-bytes",
            content_type="pickle/object",
            provenance_hash="prov-x",
        )
        manager.store_cell_output(
            cell_id="c1",
            variable_name="y",
            blob_data=b"y-bytes",
            content_type="pickle/object",
            provenance_hash="prov-y",
        )

        listed = manager.list_cell_artifacts("c1")
        names = {name for name, _ in listed}
        assert names == {"x", "y"}

    def test_excludes_iteration_artifacts(self, manager):
        """Loop-iteration ids carry @iter=k; those belong to list_iterations,
        not the canonical list. Including them here would surface every
        iteration of every variable in cell-level UIs."""
        manager.store_cell_output(
            cell_id="c1",
            variable_name="state",
            blob_data=b"canonical",
            content_type="pickle/object",
            provenance_hash="prov-canonical",
        )
        manager.store_cell_output(
            cell_id="c1",
            variable_name="state",
            blob_data=b"iter-0",
            content_type="pickle/object",
            provenance_hash="prov-iter-0",
            iteration=0,
        )
        manager.store_cell_output(
            cell_id="c1",
            variable_name="state",
            blob_data=b"iter-1",
            content_type="pickle/object",
            provenance_hash="prov-iter-1",
            iteration=1,
        )

        listed = manager.list_cell_artifacts("c1")
        names = [name for name, _ in listed]
        assert names == ["state"]

    def test_other_cells_isolated(self, manager):
        manager.store_cell_output(
            cell_id="c1",
            variable_name="x",
            blob_data=b"c1-x",
            content_type="pickle/object",
            provenance_hash="prov-c1-x",
        )
        manager.store_cell_output(
            cell_id="c2",
            variable_name="y",
            blob_data=b"c2-y",
            content_type="pickle/object",
            provenance_hash="prov-c2-y",
        )

        c1_listing = {name for name, _ in manager.list_cell_artifacts("c1")}
        c2_listing = {name for name, _ in manager.list_cell_artifacts("c2")}
        assert c1_listing == {"x"}
        assert c2_listing == {"y"}


class TestGetArtifactInfo:
    """get_artifact_info used to return content_type='unknown' always; now
    it reads the value from transform_spec.params like get_artifact_preview."""

    def test_content_type_round_trips(self, manager):
        manager.store_cell_output(
            cell_id="c1",
            variable_name="x",
            blob_data=b"json-bytes",
            content_type="json/object",
            provenance_hash="prov-x",
        )
        artifact_id = manager.cell_artifact_id("c1", "x")
        latest = manager.artifact_store.get_latest_version(artifact_id)
        assert latest is not None

        info = manager.get_artifact_info(artifact_id, latest.version)
        assert info is not None
        assert info.content_type == "json/object"

    def test_returns_none_when_artifact_missing(self, manager):
        assert manager.get_artifact_info("nonexistent", 1) is None


class TestPublishedArtifactsDashboard:
    """``GET /v1/notebooks/{id}/artifacts`` (list_notebook_published_artifacts)
    powers the per-cell registry strip: for each cell it surfaces the ready
    registry artifacts stamped ``nb_cell=<id>``, with their names and tags
    (the ``nb_cell`` tag itself hidden)."""

    def _state(self, artifact_dir: Path):
        from types import SimpleNamespace

        return SimpleNamespace(
            config=SimpleNamespace(
                writes_enabled=True,
                server_transforms_enabled=False,
                service_writes_enabled=False,
                artifact_dir=artifact_dir,
            )
        )

    def _session(self, cell_ids: list[str]):
        from types import SimpleNamespace

        cells = [SimpleNamespace(id=cid) for cid in cell_ids]
        return SimpleNamespace(notebook_state=SimpleNamespace(cells=cells))

    def test_groups_named_and_tagged_artifacts_per_cell(self, tmp_path, monkeypatch):
        import asyncio

        import strata.server as server_module
        from strata.artifact_store import get_artifact_store, reset_artifact_store
        from strata.notebook.routes import list_notebook_published_artifacts

        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        reset_artifact_store()
        store = get_artifact_store(artifact_dir)
        try:
            # Cell c1 published a ready, named, tagged model.
            store.create_artifact("model-1", "prov-1")
            store.finalize_artifact("model-1", 1, '{"fields": []}', 3, 64)
            store.set_name("team/model", "model-1", 1)
            store.set_tag("model-1", 1, "nb_cell", "c1")
            store.set_tag("model-1", 1, "stage", "candidate")

            monkeypatch.setattr(server_module, "_state", self._state(artifact_dir))

            # c2 has no published artifacts and must be omitted from the map.
            result = asyncio.run(
                list_notebook_published_artifacts("sess-1", self._session(["c1", "c2"]))
            )

            assert set(result["cells"]) == {"c1"}
            (item,) = result["cells"]["c1"]
            assert item["artifact_id"] == "model-1"
            assert item["version"] == 1
            assert item["uri"] == "strata://artifact/model-1@v=1"
            assert "team/model" in item["names"]
            # The nb_cell stamp is structural plumbing, not surfaced in the strip.
            assert item["tags"] == {"stage": "candidate"}
        finally:
            reset_artifact_store()

    def test_empty_when_store_unreachable_in_service_mode(self, tmp_path, monkeypatch):
        """Service mode 403s the published-tier store; the strip degrades to an
        empty map rather than erroring (graceful until the registry refactor)."""
        import asyncio
        from types import SimpleNamespace

        import strata.server as server_module
        from strata.notebook.routes import list_notebook_published_artifacts

        # writes disabled + no service writes ⇒ _get_artifact_store() raises 403.
        monkeypatch.setattr(
            server_module,
            "_state",
            SimpleNamespace(
                config=SimpleNamespace(
                    writes_enabled=False,
                    server_transforms_enabled=False,
                    service_writes_enabled=False,
                    artifact_dir=tmp_path / "artifacts",
                )
            ),
        )

        result = asyncio.run(list_notebook_published_artifacts("sess-1", self._session(["c1"])))
        assert result == {"cells": {}}


class TestVariantArtifacts:
    """Sweep-v2 fan-out artifact identity (``@variant={name}`` suffix)."""

    def _blob(self, manager, artifact_id):
        latest = manager.artifact_store.get_latest_version(artifact_id)
        if latest is None:
            return None
        return manager.artifact_store.blob_store.read_blob(artifact_id, latest.version)

    def test_variant_artifact_id_has_suffix(self, manager):
        assert (
            manager.cell_artifact_id("c1", "score", variant="logreg")
            == "nb_nb1_cell_c1_var_score@variant=logreg"
        )

    def test_iter_precedes_variant_when_both_given(self, manager):
        # A cell is never both loop and fan-out, but the ordering is pinned.
        assert (
            manager.cell_artifact_id("c1", "score", iteration=2, variant="rf")
            == "nb_nb1_cell_c1_var_score@iter=2@variant=rf"
        )

    def test_variants_are_independent_artifacts(self, manager):
        for name in ("logreg", "rf", "gbm"):
            manager.store_cell_output(
                cell_id="c1",
                variable_name="score",
                blob_data=f"{name}-bytes".encode(),
                content_type="pickle/object",
                provenance_hash=f"prov-{name}",
                variant=name,
            )
        for name in ("logreg", "rf", "gbm"):
            aid = manager.cell_artifact_id("c1", "score", variant=name)
            assert self._blob(manager, aid) == f"{name}-bytes".encode()

    def test_variant_does_not_collide_with_regular(self, manager):
        manager.store_cell_output(
            cell_id="c1",
            variable_name="score",
            blob_data=b"one-shot",
            content_type="pickle/object",
            provenance_hash="prov-one-shot",
        )
        manager.store_cell_output(
            cell_id="c1",
            variable_name="score",
            blob_data=b"logreg-bytes",
            content_type="pickle/object",
            provenance_hash="prov-logreg",
            variant="logreg",
        )
        assert self._blob(manager, manager.cell_artifact_id("c1", "score")) == b"one-shot"
        assert (
            self._blob(manager, manager.cell_artifact_id("c1", "score", variant="logreg"))
            == b"logreg-bytes"
        )

    def test_list_variants_returns_sorted_pairs(self, manager):
        for name in ("rf", "logreg", "gbm"):
            manager.store_cell_output(
                cell_id="c1",
                variable_name="score",
                blob_data=f"{name}".encode(),
                content_type="pickle/object",
                provenance_hash=f"prov-{name}",
                variant=name,
            )
        pairs = manager.list_variants("c1", "score")
        assert [name for name, _ in pairs] == ["gbm", "logreg", "rf"]

    def test_list_variants_skips_regular_artifact(self, manager):
        manager.store_cell_output(
            cell_id="c1",
            variable_name="score",
            blob_data=b"one-shot",
            content_type="pickle/object",
            provenance_hash="prov-one-shot",
        )
        assert manager.list_variants("c1", "score") == []
