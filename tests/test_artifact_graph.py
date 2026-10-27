"""A cell output names the structure that produced it (ADR 002 §6.3).

The node is the cell's definition, not its run: what a run recorded about itself is
outside the payload, and every output of one run shares the node. Versions from before
the graph get a reconstructed node when the upgraded store is first opened.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vestigo.provenance import TransformSpec
from vestigo.store import ArtifactStore

from strata.artifact.graph import (
    CELL_DEFINITIONS_UPGRADE,
    CELL_DOMAIN,
    RECONSTRUCTED_DOMAIN,
    RECONSTRUCTED_FROM,
    cell_definition,
    definition_payload,
    reconstruct_definitions,
)
from strata.notebook.artifact_integration import NotebookArtifactManager


@pytest.fixture
def manager(tmp_path: Path) -> NotebookArtifactManager:
    return NotebookArtifactManager("nb1", artifact_dir=tmp_path / "artifacts")


def _ref(artifact) -> dict[str, str]:
    tail = f"{artifact.id}@v={artifact.version}"
    return {f"strata://artifact/{tail}": tail}


def _store(manager, cell_id, variable, *, inputs=None, **extra):
    return manager.store_cell_output(
        cell_id=cell_id,
        variable_name=variable,
        blob_data=b"[1]",
        content_type="json/object",
        provenance_hash=f"{cell_id}-{variable}".ljust(64, "0"),
        input_versions=inputs or {},
        source_hash="s" * 64,
        **extra,
    )


class TestPublishedDefinitions:
    def test_a_new_output_carries_the_cells_node(self, manager):
        rows = _store(manager, "c1", "rows")
        store = manager.artifact_store

        node = store.definition_of(rows.id, rows.version)
        assert node is not None
        record = store.graph.get(node)
        assert record is not None
        assert record.domain == CELL_DOMAIN
        assert json.loads(record.payload) == {
            "executor": "notebook/cell@v1",
            "params": {"cell_id": "c1", "source_hash": "s" * 64},
        }
        assert store.graph.inputs(node) == []

    def test_an_upstream_variable_is_an_edge_under_its_name(self, manager):
        rows = _store(manager, "c1", "rows")
        total = _store(manager, "c2", "total", inputs=_ref(rows))
        store = manager.artifact_store

        upstream = store.definition_of(rows.id, rows.version)
        node = store.definition_of(total.id, total.version)
        assert [(p.role, p.position, p.target_node_id) for p in store.graph.inputs(node)] == [
            ("rows", 0, upstream)
        ]
        assert store.graph.ancestors([node]) == [upstream]
        assert store.graph.dependents(upstream) == [node]

    def test_fetches_and_datasets_are_leaves(self, manager):
        inputs = {
            "https://example.com/data.csv": "sha256:abc",
            "strata://name/team/events@v=3": "nb_x@v=1",
        }
        out = _store(manager, "c1", "out", inputs=inputs)
        store = manager.artifact_store

        node = store.definition_of(out.id, out.version)
        ports = {(p.role, p.position): p.target_node_id for p in store.graph.inputs(node)}
        assert set(ports) == {("fetch", 0), ("dataset", 0)}
        fetch = store.graph.get(ports[("fetch", 0)])
        dataset = store.graph.get(ports[("dataset", 0)])
        assert (fetch.domain, fetch.payload) == ("strata/fetch", b"https://example.com/data.csv")
        assert (dataset.domain, dataset.payload) == ("strata/dataset", b"strata://name/team/events")

    def test_a_table_read_is_an_input_the_record_names_and_a_leaf(self, manager):
        """A ``@table`` read is in ``input_versions`` at its snapshot and joins the node
        under ``table``; the leaf is the table, whatever snapshot was read."""
        out = _store(manager, "c1", "out", inputs={"file:///wh#db.events": "7"})
        store = manager.artifact_store

        assert json.loads(out.input_versions or "{}") == {"file:///wh#db.events": "7"}
        node = store.definition_of(out.id, out.version)
        ports = {(p.role, p.position): p.target_node_id for p in store.graph.inputs(node)}
        assert set(ports) == {("table", 0)}
        leaf = store.graph.get(ports[("table", 0)])
        assert (leaf.domain, leaf.payload) == ("strata/table", b"file:///wh#db.events")
        # The same table read again, at another snapshot, is the same leaf.
        again = _store(manager, "c2", "out", inputs={"file:///wh#db.events": "8"})
        node2 = store.definition_of(again.id, again.version)
        assert [p.target_node_id for p in store.graph.inputs(node2)] == [ports[("table", 0)]]

    def test_a_table_leaf_carries_where_the_read_found_it(self, manager):
        """A bare ``db.events`` is one table per default catalog, so the leaf names
        the location the read resolved; a pinned read, never loaded, has none."""
        store = manager.artifact_store
        manager.record_table_locations("c1", {"db.events": "s3://lake-a/wh/db/events"})
        manager.record_table_locations("c2", {"db.events": "s3://lake-b/wh/db/events"})
        leaves = {}
        for cell, snapshot in (("c1", "7"), ("c2", "3"), ("c3", "3")):
            out = _store(manager, cell, "out", inputs={"db.events": snapshot})
            (port,) = store.graph.inputs(store.definition_of(out.id, out.version))
            leaves[cell] = store.graph.get(port.target_node_id)

        assert leaves["c1"].payload == b"db.events\ns3://lake-a/wh/db/events"
        assert leaves["c2"].payload == b"db.events\ns3://lake-b/wh/db/events"
        assert leaves["c3"].payload == b"db.events"
        assert len({leaf.node_id for leaf in leaves.values()}) == 3

    def test_an_upstream_without_a_node_is_an_artifact_leaf(self, manager):
        store = manager.artifact_store
        version = store.create_artifact("imported", "p" * 64, TransformSpec("x@v1", {}, []))
        store.finalize_artifact("imported", version, "{}", 1, 1)
        out = _store(manager, "c1", "out", inputs={f"strata://artifact/imported@v={version}": "x"})

        (port,) = store.graph.inputs(store.definition_of(out.id, out.version))
        leaf = store.graph.get(port.target_node_id)
        assert (port.role, leaf.domain) == ("input", "strata/artifact")
        assert leaf.payload == b"strata://artifact/imported"

    def test_every_output_of_one_run_shares_the_node(self, manager):
        a = _store(manager, "c1", "a")
        b = _store(manager, "c1", "b")
        display = _store(
            manager, "c1", "__display__0", extra_params={"display": "{}", "display_count": "1"}
        )
        store = manager.artifact_store
        node = store.definition_of(a.id, a.version)
        assert store.definition_of(b.id, b.version) == node
        assert store.definition_of(display.id, display.version) == node

    def test_an_upstream_of_another_tenant_is_a_leaf(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts")
        spec = TransformSpec("notebook/cell@v1", {"cell_id": "c"}, [])
        theirs = store.create_artifact(
            "nb_n_cell_c0_var_rows",
            "p" * 64,
            spec,
            tenant="t2",
            definition=cell_definition(store, spec, {}, tenant="t2"),
        )
        uri = f"strata://artifact/nb_n_cell_c0_var_rows@v={theirs}"

        specs = cell_definition(store, spec, {uri: "x"}, tenant="t1")
        mine = store.create_artifact("out", "q" * 64, spec, tenant="t1", definition=specs)

        (port,) = store.graph.inputs(store.definition_of("out", mine, tenant="t1"), tenant="t1")
        leaf = store.graph.get(port.target_node_id, tenant="t1")
        assert (port.role, leaf.domain) == ("rows", "strata/artifact")

    def test_what_a_run_records_about_itself_is_outside_the_node(self, manager):
        store = manager.artifact_store
        first = _store(manager, "c1", "a", build_duration_ms=12.0, build_env="cpython 3.12")
        second = _store(
            manager, "c1", "a", build_duration_ms=900.0, hardware={"gpu": "H100"}, source="x = 1"
        )
        assert store.definition_of(first.id, first.version) == store.definition_of(
            second.id, second.version
        )

    def test_a_different_source_is_a_different_node(self, manager):
        store = manager.artifact_store
        first = _store(manager, "c1", "a")
        second = manager.store_cell_output(
            cell_id="c1",
            variable_name="a",
            blob_data=b"[2]",
            content_type="json/object",
            provenance_hash="q" * 64,
            input_versions={},
            source_hash="t" * 64,
        )
        assert store.definition_of(first.id, first.version) != store.definition_of(
            second.id, second.version
        )

    def test_the_payload_keeps_the_defining_params_only(self):
        spec = TransformSpec(
            "notebook/cell@v1",
            {
                "cell_id": "c1",
                "variable_name": "a",
                "content_type": "json/object",
                "source_hash": "s",
                "env_hash": "e",
                "iteration": "2",
                "fetched_at": "{}",
                "build_duration_ms": "5",
                "source": "x = 1",
                "build_env": "cpython",
                "hardware": "{}",
                "loop_run": "abc",
                "display": "{}",
                "display_count": "2",
                "mount_uri": "s3://b/k",
            },
            [],
        )
        assert json.loads(definition_payload(spec))["params"] == {
            "cell_id": "c1",
            "source_hash": "s",
            "env_hash": "e",
            "iteration": "2",
            "mount_uri": "s3://b/k",
        }


def _old_cell_output(store: ArtifactStore, artifact_id: str, inputs: dict[str, str]) -> int:
    """A row as a store from before the graph holds it: a cell's spec, no node."""
    spec = TransformSpec(
        "notebook/cell@v1",
        {"cell_id": "c", "variable_name": artifact_id, "content_type": "json/object"},
        [],
    )
    version = store.create_artifact(artifact_id, f"prov-{artifact_id}", spec, input_versions=inputs)
    store.finalize_artifact(artifact_id, version, "{}", 1, 1)
    return version


class TestReconstruction:
    def test_old_rows_get_a_reconstructed_node_in_dependency_order(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts")
        raw = _old_cell_output(store, "raw", {})
        rows = _old_cell_output(store, "rows", {f"strata://artifact/raw@v={raw}": "x"})
        total = _old_cell_output(
            store,
            "total",
            {
                f"strata://artifact/rows@v={rows}": "x",
                "https://example.com/d.csv": "sha256:1",
            },
        )

        reconstruct_definitions(store)

        raw_node = store.definition_of("raw", raw)
        rows_node = store.definition_of("rows", rows)
        total_node = store.definition_of("total", total)
        assert store.graph.get(raw_node).domain == RECONSTRUCTED_DOMAIN
        assert [(p.role, p.position, p.target_node_id) for p in store.graph.inputs(rows_node)] == [
            ("input", 0, raw_node)
        ]
        ports = store.graph.inputs(total_node)
        assert [(p.role, p.position) for p in ports] == [("input", 0), ("input", 1)]
        # Positional order is the sorted URI order: the fetch sorts before the artifact.
        assert store.graph.get(ports[0].target_node_id).domain == "strata/fetch"
        assert ports[1].target_node_id == rows_node
        assert store.graph.topological_order([total_node])[-1] == total_node
        assert store.upgrade_applied(CELL_DEFINITIONS_UPGRADE)

    def test_the_step_runs_once(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts")
        reconstruct_definitions(store)
        late = _old_cell_output(store, "late", {})

        reconstruct_definitions(store)

        assert store.definition_of("late", late) is None

    def test_rows_that_are_not_cell_outputs_are_left_alone(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts")
        folder_file = TransformSpec(
            "notebook/cell@v1", {"cell_id": "c", "content_type": "application/octet-stream"}, []
        )
        prompt = TransformSpec("prompt", {"content_type": "text/markdown", "model": "m"}, [])
        v1 = store.create_artifact("file", "p1", folder_file)
        store.finalize_artifact("file", v1, "{}", 1, 1)
        v2 = store.create_artifact("answer", "p2", prompt)
        store.finalize_artifact("answer", v2, "{}", 1, 1)

        reconstruct_definitions(store)

        assert store.definition_of("file", v1) is None
        assert store.definition_of("answer", v2) is None
        assert store.upgrade_applied(CELL_DEFINITIONS_UPGRADE)

    def test_a_fresh_notebook_store_is_marked_on_open(self, manager):
        assert manager.artifact_store.upgrade_applied(CELL_DEFINITIONS_UPGRADE)

    def test_a_rerun_after_the_upgrade_publishes_a_complete_node(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts")
        old = _old_cell_output(store, "nb_nb1_cell_c1_var_a", {})
        reconstruct_definitions(store)
        manager = NotebookArtifactManager("nb1", artifact_dir=tmp_path / "artifacts")

        new = _store(manager, "c1", "a")

        before = store.graph.get(store.definition_of("nb_nb1_cell_c1_var_a", old))
        after = manager.artifact_store.graph.get(store.definition_of(new.id, new.version))
        assert (before.domain, after.domain) == (RECONSTRUCTED_DOMAIN, CELL_DOMAIN)
        assert before.node_id != after.node_id
        # The complete node names the historical one; a second rerun adds nothing.
        (link,) = store.graph.links(after.node_id, RECONSTRUCTED_FROM)
        assert link.target_node_id == before.node_id
        again = _store(manager, "c1", "a")
        assert store.definition_of(again.id, again.version) == after.node_id
        assert store.graph.linked_from(before.node_id) == [link]

    def test_a_rerun_of_a_complete_node_links_nothing(self, manager):
        first = _store(manager, "c1", "a")
        second = manager.store_cell_output(
            cell_id="c1",
            variable_name="a",
            blob_data=b"[2]",
            content_type="json/object",
            provenance_hash="q" * 64,
            input_versions={},
            source_hash="t" * 64,
        )
        store = manager.artifact_store

        assert store.graph.links(store.definition_of(second.id, second.version)) == []
        assert store.definition_of(first.id, first.version) != store.definition_of(
            second.id, second.version
        )
