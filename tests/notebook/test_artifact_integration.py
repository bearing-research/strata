"""Tests for NotebookArtifactManager, the notebook's bridge to the artifact store.

Focuses on loop cells' per-iteration artifact ids; single-artifact behaviour is covered
by the executor and cache-hit tests.
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
        """iteration=0 is distinct from None: ``@iter=0`` stays visible."""
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
        """``state`` stored without and with an iteration suffix gives two artifacts, so loop
        iteration 0 never overwrites a cell's one-shot output.
        """
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
        """A regular ``store_cell_output`` id lacks ``@iter=``, so it is not listed."""
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
        """transform_spec carries the iteration index, so readers need not parse the id."""
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
        """Loop-iteration ids (@iter=k) belong to list_iterations; listing them here would surface
        every iteration of every variable in cell-level UIs.
        """
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
    """get_artifact_info reads content_type from transform_spec.params, like the preview does."""

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
    """``GET /v1/notebooks/{id}/artifacts`` powers the per-cell registry strip.

    For each cell it lists the ready registry artifacts tagged ``nb_cell=<id>``, with their
    names and tags (the ``nb_cell`` tag itself hidden).
    """

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
        no_reads = SimpleNamespace(list_name_reads=lambda tenant=None: [])
        return SimpleNamespace(
            notebook_state=SimpleNamespace(cells=cells),
            get_artifact_manager=lambda: SimpleNamespace(artifact_store=no_reads),
        )

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
        """Service mode 403s the published-tier store; the strip degrades to an empty map rather
        than erroring.
        """
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
        assert result == {"cells": {}, "readers": {}}


class TestVariantArtifacts:
    """Fan-out artifact identity (``@variant={name}`` suffix)."""

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


class TestLineageRefShape:
    """The recorded ``input_versions`` shape the lineage walk resolves.

    The walk follows an input only when its key is a ``strata://artifact/`` URI and its value
    carries ``@v=``. Which refs a cell writes is tested end to end in
    ``test_e2e_provenance_persistence``.
    """

    def test_lineage_walks_transitively_through_resolved_refs(self, manager):
        from strata.services.artifact import ArtifactService

        def ref(artifact) -> dict[str, str]:
            tail = f"{artifact.id}@v={artifact.version}"
            return {f"strata://artifact/{tail}": tail}

        raw = manager.store_cell_output(
            cell_id="c0",
            variable_name="raw",
            blob_data=b"[0]",
            content_type="json/object",
            provenance_hash="c" * 64,
            input_versions={},
        )
        rows = manager.store_cell_output(
            cell_id="c1",
            variable_name="rows",
            blob_data=b"[1, 2]",
            content_type="json/object",
            provenance_hash="a" * 64,
            input_versions=ref(raw),
        )
        plot = manager.store_cell_output(
            cell_id="c2",
            variable_name="__display__0",
            blob_data=b"PNG",
            content_type="image/png",
            provenance_hash="b" * 64,
            input_versions=ref(rows),
        )

        lineage = ArtifactService().build_lineage(
            manager.artifact_store,
            artifact=plot,
            artifact_id=plot.id,
            version=plot.version,
            tenant_filter=None,
            max_depth=10,
        )

        assert lineage.depth == 2
        assert [n.uri for n in lineage.nodes if n.type == "artifact"] == [
            "strata://artifact/nb_nb1_cell_c2_var___display__0@v=1",
            "strata://artifact/nb_nb1_cell_c1_var_rows@v=1",
            "strata://artifact/nb_nb1_cell_c0_var_raw@v=1",
        ]

    def test_a_shared_store_moves_ready_to_the_newest_writer(self, tmp_path):
        """Why refs are not resolved from provenance hashes.

        The cell id is not in a provenance hash, so an identical cell in another notebook sharing
        the store hashes the same, and its write takes ``ready``. ``find_by_provenance`` would
        then return the other notebook's artifact.
        """
        shared_dir = tmp_path / "shared"
        mine = NotebookArtifactManager("mine", artifact_dir=shared_dir)
        theirs = NotebookArtifactManager("theirs", artifact_dir=shared_dir)
        provenance = "d" * 64

        ours = mine.store_cell_output(
            cell_id="c1",
            variable_name="x",
            blob_data=b"1",
            content_type="json/object",
            provenance_hash=provenance,
            input_versions={},
        )
        theirs.store_cell_output(
            cell_id="zz",
            variable_name="x",
            blob_data=b"1",
            content_type="json/object",
            provenance_hash=provenance,
            input_versions={},
        )

        # Our row is still what our cells read by id, but the hash no longer
        # resolves to it.
        assert mine.artifact_store.get_artifact(ours.id, ours.version).state == "superseded"
        assert mine.artifact_store.find_by_provenance(provenance).id != ours.id


class TestOneRunsOutputsTogether:
    """A cell's outputs from one run become current together, or none of them do.

    Finalized one at a time, a crash between two outputs left one variable from the new run and
    the other from the run before, and a downstream cell read both as current.
    """

    class _Crash(BaseException):
        """Stands in for the process dying: nothing in the store path catches it."""

    @staticmethod
    def _session(tmp_path: Path):
        from strata.notebook.dag import NotebookDag
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook = create_notebook(tmp_path, "together", initialize_environment=False)
        add_cell_to_notebook(notebook, "a", None)
        write_cell(notebook, "a", "model = {}\nmetric = {}\n")
        session = NotebookSession(parse_notebook(notebook), notebook)
        session.dag = NotebookDag(consumed_variables={"a": {"model", "metric"}})
        return session

    @staticmethod
    def _run(executor, output_dir: Path, run: int, *, write_metric: bool = True) -> bool:
        output_dir.mkdir(exist_ok=True)
        for stale in output_dir.iterdir():
            stale.unlink()
        (output_dir / "model.json").write_text(f'{{"run": {run}}}')
        if write_metric:
            (output_dir / "metric.json").write_text(f'{{"run": {run}}}')
        return executor._store_outputs("a", output_dir, "prov" + "0" * 60, [])

    @staticmethod
    def _current(session) -> dict[str, bytes]:
        manager = session.get_artifact_manager()
        current = {}
        for var in ("model", "metric"):
            artifact_id = manager.cell_artifact_id("a", var)
            latest = manager.artifact_store.get_latest_version(artifact_id)
            current[var] = manager.artifact_store.read_blob(artifact_id, latest.version)
        return current

    def test_a_crash_between_two_outputs_leaves_the_earlier_run_current(
        self, tmp_path, monkeypatch
    ):
        from strata.blob_store import LocalBlobStore
        from strata.notebook.executor import CellExecutor

        session = self._session(tmp_path)
        executor = CellExecutor(session)
        assert self._run(executor, tmp_path / "out", 1)

        real_write = LocalBlobStore.write_blob
        writes = []

        def write_then_die(self, artifact_id, version, data):
            writes.append(artifact_id)
            if len(writes) == 2:
                raise TestOneRunsOutputsTogether._Crash
            return real_write(self, artifact_id, version, data)

        monkeypatch.setattr(LocalBlobStore, "write_blob", write_then_die)
        with pytest.raises(self._Crash):
            self._run(executor, tmp_path / "out", 2)
        monkeypatch.undo()

        assert self._current(session) == {"model": b'{"run": 1}', "metric": b'{"run": 1}'}

    def test_a_missing_output_stores_none_of_the_run(self, tmp_path):
        from strata.notebook.executor import CellExecutor

        session = self._session(tmp_path)
        executor = CellExecutor(session)
        assert self._run(executor, tmp_path / "out", 1)

        assert not self._run(executor, tmp_path / "out", 2, write_metric=False)

        assert self._current(session) == {"model": b'{"run": 1}', "metric": b'{"run": 1}'}

    def test_a_complete_run_replaces_both(self, tmp_path):
        from strata.notebook.executor import CellExecutor

        session = self._session(tmp_path)
        executor = CellExecutor(session)
        assert self._run(executor, tmp_path / "out", 1)

        assert self._run(executor, tmp_path / "out", 2)

        assert self._current(session) == {"model": b'{"run": 2}', "metric": b'{"run": 2}'}
