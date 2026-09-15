"""Writing artifacts into Iceberg tables. Item 27."""

from __future__ import annotations

import io
from types import SimpleNamespace

import httpx
import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog

from strata.artifact_store import ArtifactStore
from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.table_export import EXPORT_TAG, export_artifact


def _ipc(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


@pytest.fixture
def notebook_store(tmp_path):
    return NotebookArtifactManager("nb", artifact_dir=tmp_path / "notebook")


def _version(manager, table: pa.Table, *, content_type="arrow/ipc", tag="1"):
    return manager.store_cell_output(
        cell_id="c1",
        variable_name="features",
        blob_data=_ipc(table) if content_type == "arrow/ipc" else b"[1]",
        content_type=content_type,
        provenance_hash=f"{tag:0>2}" * 32,
        input_versions={},
        source=f"features = v{tag}",
    )


def _catalog(warehouse) -> SqlCatalog:
    return SqlCatalog("strata", uri=f"sqlite:///{warehouse}/catalog.db", warehouse=str(warehouse))


def _config(tmp_path):
    return SimpleNamespace(
        catalog_properties={},
        catalog_name="strata",
        metadata_db=tmp_path / "metadata.db",
        cache_dir=tmp_path / "cache",
        s3_region=None,
        s3_access_key=None,
        s3_secret_key=None,
        s3_endpoint_url=None,
    )


def _written_versions(warehouse) -> list[tuple[str, str]]:
    table = _catalog(warehouse).load_table("taxi.features")
    return sorted(
        (snapshot.summary["strata.artifact_id"], snapshot.summary["strata.version"])
        for snapshot in table.snapshots()
        if snapshot.summary is not None
        and "strata.version" in snapshot.summary
        # An overwrite commits a delete and then an append; the append is
        # the snapshot that holds the version.
        and snapshot.summary.operation.value == "append"
    )


class TestTheWriter:
    def test_versions_become_snapshots_that_name_them(self, tmp_path, notebook_store):
        warehouse = tmp_path / "wh"
        uri = f"{warehouse}#taxi.features"
        store = notebook_store.artifact_store
        first = _version(notebook_store, pa.table({"trip": [1, 2]}), tag="1")
        second = _version(notebook_store, pa.table({"trip": [1, 2, 3]}), tag="2")

        created = export_artifact(store, first, uri, config=_config(tmp_path), promoted_by="ana")
        overwritten = export_artifact(store, second, uri, config=_config(tmp_path))

        assert created.created is True
        assert overwritten.created is False
        assert _written_versions(warehouse) == [(first.id, "1"), (second.id, "2")]
        table = _catalog(warehouse).load_table("taxi.features")
        current = table.current_snapshot()
        assert current.snapshot_id == overwritten.snapshot_id
        assert current.summary["strata.provenance_hash"] == second.provenance_hash
        assert table.scan().to_arrow().column("trip").to_pylist() == [1, 2, 3]
        first_snapshot = next(s for s in table.snapshots() if s.snapshot_id == created.snapshot_id)
        assert first_snapshot.summary["strata.promoted_by"] == "ana"
        assert store.get_tags(second.id, second.version)[EXPORT_TAG] == uri

    def test_a_new_column_evolves_the_table_and_a_changed_type_is_refused(
        self, tmp_path, notebook_store
    ):
        uri = f"{tmp_path / 'wh'}#taxi.features"
        store = notebook_store.artifact_store
        config = _config(tmp_path)
        export_artifact(
            store, _version(notebook_store, pa.table({"trip": [1]}), tag="1"), uri, config=config
        )
        widened = export_artifact(
            store,
            _version(notebook_store, pa.table({"trip": [2], "fare": [9.5]}), tag="2"),
            uri,
            config=config,
        )

        with pytest.raises(ValueError, match="not compatible"):
            export_artifact(
                store,
                _version(notebook_store, pa.table({"trip": ["three"]}), tag="3"),
                uri,
                config=config,
            )

        table = _catalog(tmp_path / "wh").load_table("taxi.features")
        assert [f.name for f in table.schema().fields] == ["trip", "fare"]
        assert table.current_snapshot().snapshot_id == widened.snapshot_id

    def test_an_alias_is_a_tag_on_the_snapshot(self, tmp_path, notebook_store):
        uri = f"{tmp_path / 'wh'}#taxi.features"
        written = export_artifact(
            notebook_store.artifact_store,
            _version(notebook_store, pa.table({"trip": [1]})),
            uri,
            config=_config(tmp_path),
            alias="champion",
        )

        refs = _catalog(tmp_path / "wh").load_table("taxi.features").refs()
        assert refs["champion"].snapshot_id == written.snapshot_id

    def test_what_is_not_a_table_is_refused(self, tmp_path, notebook_store):
        uri = f"{tmp_path / 'wh'}#taxi.features"
        with pytest.raises(ValueError, match="not a table"):
            export_artifact(
                notebook_store.artifact_store,
                _version(notebook_store, pa.table({"x": [1]}), content_type="json/object"),
                uri,
                config=_config(tmp_path),
            )
        tensor = pa.table({"x": [1]}).replace_schema_metadata({"strata.arrow.shape": "tensor"})
        with pytest.raises(ValueError, match="not a table"):
            export_artifact(
                notebook_store.artifact_store,
                _version(notebook_store, tensor, tag="2"),
                uri,
                config=_config(tmp_path),
            )


@pytest.fixture
def team(tmp_path):
    from tests.conftest import run_server_with_context

    with run_server_with_context(tmp_path / "cache", tmp_path / "team", "personal") as ctx:
        yield SimpleNamespace(url=ctx.base_url, dir=tmp_path / "team")


class TestPromotingToATable:
    """Promoting a dataset twice: two snapshots naming the two versions,
    ``champion`` on the second, and a notebook reading the table goes stale
    between them."""

    @staticmethod
    def _promote(notebook_store, artifact, team, warehouse):
        from strata.artifact_transfer import RemoteStore, promote_artifact

        return promote_artifact(
            notebook_store.artifact_store,
            RemoteStore(team.url),
            artifact,
            name="taxi/features",
            alias="champion",
            table=f"{warehouse}#taxi.features",
        )

    @staticmethod
    def _reader(tmp_path, warehouse):
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        nb = create_notebook(tmp_path / "reader", "Reads the table")
        add_cell_to_notebook(nb, "c1", None)
        source = f"# @table features {warehouse}#taxi.features\nsnapshot = features_snapshot"
        write_cell(nb, "c1", source)
        add_cell_to_notebook(nb, "c2", "c1")
        write_cell(nb, "c2", "shown = snapshot")
        return NotebookSession(parse_notebook(nb), nb), source

    async def test_promoting_twice_writes_two_snapshots_and_moves_champion(
        self, tmp_path, notebook_store, team
    ):
        from strata.notebook.executor import CellExecutor

        warehouse = tmp_path / "wh"
        first = _version(notebook_store, pa.table({"trip": [1, 2]}), tag="1")
        promoted = self._promote(notebook_store, first, team, warehouse)
        assert promoted.table_snapshot is not None

        session, source = self._reader(tmp_path, warehouse)
        ran = await CellExecutor(session).execute_cell("c1", source)
        assert ran.success, ran.error
        assert session.compute_staleness()["c1"].status.value == "ready"

        second = _version(notebook_store, pa.table({"trip": [1, 2, 3]}), tag="2")
        again = self._promote(notebook_store, second, team, warehouse)

        assert session.compute_staleness()["c1"].status.value != "ready"
        assert _written_versions(warehouse) == [(first.id, "1"), (second.id, "2")]
        refs = _catalog(warehouse).load_table("taxi.features").refs()
        assert refs["champion"].snapshot_id == again.table_snapshot

    def test_moving_the_alias_back_moves_the_tag(self, tmp_path, notebook_store, team):
        warehouse = tmp_path / "wh"
        first = _version(notebook_store, pa.table({"trip": [1]}), tag="1")
        second = _version(notebook_store, pa.table({"trip": [2]}), tag="2")
        was = self._promote(notebook_store, first, team, warehouse)
        self._promote(notebook_store, second, team, warehouse)

        response = httpx.put(
            f"{team.url}/v1/names/taxi/features/aliases/champion",
            json={"artifact_id": first.id, "version": first.version},
            timeout=30,
        )

        assert response.status_code == 200, response.text
        refs = _catalog(warehouse).load_table("taxi.features").refs()
        assert refs["champion"].snapshot_id == was.table_snapshot

    def test_the_export_route_refuses_what_is_not_a_table(self, tmp_path, team):
        store = ArtifactStore(team.dir)
        manager = NotebookArtifactManager("team", artifact_dir=team.dir)
        rows = _version(manager, pa.table({"x": [1]}), content_type="json/object")
        assert store.get_artifact(rows.id, rows.version) is not None

        response = httpx.post(
            f"{team.url}/v1/artifacts/{rows.id}/v/{rows.version}/export",
            json={"table": f"{tmp_path / 'wh'}#taxi.features"},
            timeout=30,
        )

        assert response.status_code == 400
        assert "not a table" in response.json()["detail"]


def test_a_protected_alias_becomes_a_tag_when_it_is_approved(tmp_path, notebook_store):
    from strata.artifact_transfer import RemoteStore, promote_artifact
    from tests.conftest import run_server_with_context

    warehouse = tmp_path / "wh"
    with run_server_with_context(
        tmp_path / "cache",
        tmp_path / "team",
        "personal",
        registry_protected_aliases=["champion"],
    ) as ctx:
        first = _version(notebook_store, pa.table({"trip": [1]}))
        promoted = promote_artifact(
            notebook_store.artifact_store,
            RemoteStore(ctx.base_url),
            first,
            name="taxi/features",
            alias="champion",
            table=f"{warehouse}#taxi.features",
        )
        assert promoted.alias_pending is True
        assert "champion" not in _catalog(warehouse).load_table("taxi.features").refs()

        approved = httpx.post(
            f"{ctx.base_url}/v1/registry/pending/approve",
            json={"name": "taxi/features", "alias": "champion"},
            timeout=30,
        )

        assert approved.status_code == 200, approved.text
        refs = _catalog(warehouse).load_table("taxi.features").refs()
        assert refs["champion"].snapshot_id == promoted.table_snapshot


def test_the_cli_writes_a_local_artifact_into_a_table(tmp_path, notebook_store, capsys):
    import argparse

    from strata.artifact_cli import cmd_export_table

    artifact = _version(notebook_store, pa.table({"trip": [1, 2]}))
    status = cmd_export_table(
        argparse.Namespace(
            ref=f"{artifact.id}@v={artifact.version}",
            artifact_dir=str(tmp_path / "notebook"),
            table=f"{tmp_path / 'wh'}#taxi.features",
            alias=None,
            by="ana",
            format="human",
            tenant=None,
        )
    )

    assert status == 0, capsys.readouterr().out
    assert "Created" in capsys.readouterr().out
    assert _written_versions(tmp_path / "wh") == [(artifact.id, "1")]
