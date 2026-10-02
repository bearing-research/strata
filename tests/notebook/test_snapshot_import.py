"""A snapshot bundle, turned back into a notebook."""

from __future__ import annotations

import json
import zipfile

import pytest

from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.snapshot import write_committed_files, write_snapshot
from strata.notebook.snapshot_import import NotASnapshotError, import_snapshot
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from tests.conftest import prepared_venv


@pytest.fixture
def ran(tmp_path, capsys):
    """A notebook whose cells have run, so artifacts carry real specs, lineage and provenance."""
    from strata.notebook.cli import run_main

    nb = create_notebook(tmp_path / "src", "Roundtrip", initialize_environment=False)
    prepared_venv(nb)
    add_cell_to_notebook(nb, "rows", None)
    write_cell(nb, "rows", "rows = [1, 2, 3]\n")
    add_cell_to_notebook(nb, "total", "rows")
    write_cell(nb, "total", "total = sum(rows)\n")
    add_cell_to_notebook(nb, "report", "total")
    write_cell(nb, "report", "doubled = total * 2\nprint(doubled)\n")
    # A display-only leaf: it reopens READY only when its cached display output
    # resolves, which needs the runtime entries an import writes back.
    add_cell_to_notebook(nb, "shown", "report")
    write_cell(nb, "shown", "rows[0] + rows[1]\n")
    assert run_main([str(nb), "--no-sync", "--format", "json"]) == 0
    capsys.readouterr()
    return nb


def _export(nb, out, include="all", cells=None):
    session = NotebookSession(parse_notebook(nb), nb)
    with zipfile.ZipFile(out, "w") as archive:
        write_committed_files(session, archive)
        write_snapshot(session, archive, include=include, selected_cells=cells)
    return out


def _statuses(nb) -> dict[str, str]:
    session = NotebookSession(parse_notebook(nb), nb)
    session.compute_staleness()
    return {c.id: str(getattr(c.status, "value", c.status)) for c in session.notebook_state.cells}


class TestAFullSnapshot:
    def test_cells_are_where_they_were_without_running_anything(self, ran, tmp_path):
        """Every cell that produced artifacts has the status it had in the source notebook."""
        bundle = _export(ran, tmp_path / "snap.zip")

        imported = import_snapshot(bundle, tmp_path / "dst")

        before = _statuses(ran)
        after = _statuses(imported.notebook_dir)
        # Non-vacuity first: two all-idle notebooks are "equivalent" and prove nothing.
        # Comparing only cells in the artifact index passes that way, because a print-only
        # leaf has a console artifact but reopens idle regardless.
        ready = {c for c, status in before.items() if status == "ready"}
        assert {"rows", "total", "shown"} <= ready
        assert after == before

    def test_the_id_is_kept_when_nothing_else_has_it(self, ran, tmp_path):
        source_id = parse_notebook(ran).id

        imported = import_snapshot(_export(ran, tmp_path / "snap.zip"), tmp_path / "dst")

        assert imported.notebook_id == source_id
        assert imported.replaced_id is None
        assert imported.by_reference_cells == []


class TestATakenId:
    def test_a_second_copy_gets_its_own_id_and_still_hits(self, ran, tmp_path):
        """A second copy gets its own notebook id and its cells still hit.

        Every artifact id, edge and display URI embedding the old id has to follow,
        or the cells miss despite the bytes being present.
        """
        bundle = _export(ran, tmp_path / "snap.zip")
        source_id = parse_notebook(ran).id

        imported = import_snapshot(bundle, tmp_path / "dst", taken_ids={source_id})

        assert imported.replaced_id == source_id
        assert imported.notebook_id != source_id
        assert parse_notebook(imported.notebook_dir).id == imported.notebook_id
        after = _statuses(imported.notebook_dir)
        assert after == _statuses(ran)
        assert after["rows"] == after["total"] == "ready"

    def test_no_artifact_in_the_copy_still_names_the_old_id(self, ran, tmp_path):
        from strata.artifact_store import ArtifactStore

        bundle = _export(ran, tmp_path / "snap.zip")
        source_id = parse_notebook(ran).id

        imported = import_snapshot(bundle, tmp_path / "dst", taken_ids={source_id})

        store = ArtifactStore(imported.notebook_dir / ".strata" / "artifacts")
        rows = store.list_artifacts(limit=1000)
        assert rows, "nothing was imported"
        for artifact in rows:
            assert source_id not in artifact.id
            assert source_id not in (artifact.input_versions or "")

    def test_lineage_resolves_inside_the_copy(self, ran, tmp_path):
        """An edge still naming the old id would resolve to nothing in the imported store."""
        from strata.artifact_store import ArtifactStore
        from strata.services.artifact import ArtifactService

        bundle = _export(ran, tmp_path / "snap.zip")
        imported = import_snapshot(bundle, tmp_path / "dst", taken_ids={parse_notebook(ran).id})

        store = ArtifactStore(imported.notebook_dir / ".strata" / "artifacts")
        total = next(a for a in store.list_artifacts(limit=1000) if "_cell_total_var_total" in a.id)
        lineage = ArtifactService().build_lineage(
            store,
            artifact=total,
            artifact_id=total.id,
            version=total.version,
            tenant_filter=None,
            max_depth=5,
        )
        artifacts = [n for n in lineage.nodes if n.type == "artifact"]
        assert len(artifacts) >= 2, "the upstream edge was lost"
        for node in artifacts:
            assert store.get_artifact(node.artifact_id, node.version) is not None


class TestASelectedSnapshot:
    def test_carried_cells_hit_and_the_rest_open_idle(self, ran, tmp_path):
        """By-reference cells open IDLE, not STALE: their result is current, just not here."""
        bundle = _export(ran, tmp_path / "snap.zip", include="selected", cells=["rows"])

        imported = import_snapshot(bundle, tmp_path / "dst")

        after = _statuses(imported.notebook_dir)
        assert after["rows"] == "ready"
        assert after["total"] != "ready"
        assert "total" in imported.by_reference_cells
        assert "rows" not in imported.by_reference_cells


class TestRefusals:
    def test_a_zip_with_no_manifest_is_not_a_snapshot(self, tmp_path):
        bogus = tmp_path / "bogus.zip"
        with zipfile.ZipFile(bogus, "w") as archive:
            archive.writestr("notebook.toml", 'notebook_id = "x"\n')

        with pytest.raises(NotASnapshotError, match="no artifacts.json"):
            import_snapshot(bogus, tmp_path / "dst")

    def test_a_format_1_bundle_is_refused_rather_than_imported_wrong(self, ran, tmp_path):
        """Format 1 carried bytes without content types, so its values would load wrong."""
        bundle = _export(ran, tmp_path / "snap.zip")
        old = tmp_path / "old.zip"
        with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(old, "w") as dst:
            for name in src.namelist():
                data = src.read(name)
                if name == "artifacts.json":
                    manifest = json.loads(data)
                    manifest.pop("format_version")
                    manifest.pop("records")
                    data = json.dumps(manifest).encode()
                dst.writestr(name, data)

        with pytest.raises(NotASnapshotError, match="format 1"):
            import_snapshot(old, tmp_path / "dst")

    def test_a_bundle_missing_a_cell_file_is_refused(self, ran, tmp_path):
        bundle = _export(ran, tmp_path / "snap.zip")
        broken = tmp_path / "broken.zip"
        with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(broken, "w") as dst:
            for name in src.namelist():
                if name != "cells/total.py":
                    dst.writestr(name, src.read(name))

        with pytest.raises(NotASnapshotError, match="total.py"):
            import_snapshot(broken, tmp_path / "dst")

    def test_nothing_is_written_when_it_refuses(self, tmp_path):
        bogus = tmp_path / "bogus.zip"
        with zipfile.ZipFile(bogus, "w") as archive:
            archive.writestr("readme.txt", "hi")

        with pytest.raises(NotASnapshotError):
            import_snapshot(bogus, tmp_path / "dst")

        assert not (tmp_path / "dst").exists()

    def test_a_bundle_that_expands_past_the_cap_is_refused_before_extracting(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("strata.notebook.snapshot_import._MAX_UNCOMPRESSED_BYTES", 1024 * 1024)
        bomb = tmp_path / "bomb.zip"
        with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("artifacts.json", "{}")
            archive.writestr("notebook.toml", 'notebook_id = "x"\n')
            archive.writestr("artifacts/big@v=1", b"\0" * (4 * 1024 * 1024))
        assert bomb.stat().st_size < 1024 * 1024

        with pytest.raises(NotASnapshotError, match="import cap"):
            import_snapshot(bomb, tmp_path / "dst")

        assert not (tmp_path / "dst").exists()

    def test_an_occupied_destination_is_refused(self, ran, tmp_path):
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "keep.txt").write_text("mine")

        with pytest.raises(FileExistsError):
            import_snapshot(_export(ran, tmp_path / "snap.zip"), dst)


class TestRunningTheImport:
    def test_a_downstream_cell_computes_from_the_imported_values(self, ran, tmp_path, capsys):
        """Reading a value back is the only proof the imported records matter.

        Staleness compares provenance, so a lost content type still reads READY.
        Editing `report` forces it to run and load `total` from the imported artifact;
        otherwise it would replay its imported console as a cache hit.
        """
        from strata.notebook.cli import run_main

        imported = import_snapshot(_export(ran, tmp_path / "snap.zip"), tmp_path / "dst")
        nb = imported.notebook_dir
        prepared_venv(nb)
        write_cell(nb, "report", "doubled = total * 2\nprint(doubled, 'recomputed')\n")

        assert run_main([str(nb), "--no-sync", "--format", "json"]) == 0
        cells = {c["id"]: c for c in json.loads(capsys.readouterr().out)["cells"]}

        assert cells["rows"]["cache_hit"] is True
        assert cells["total"]["cache_hit"] is True
        assert cells["report"]["cache_hit"] is False, "report must run, or nothing is read back"
        assert cells["report"]["status"] == "ok", cells["report"].get("error")
        assert cells["report"]["stdout"].strip() == "12 recomputed"


class TestArtifactBytes:
    def test_artifact_members_stream_rather_than_load_whole(self, ran, tmp_path, monkeypatch):
        bundle = _export(ran, tmp_path / "snap.zip")
        whole_reads: list[str] = []
        original_read = zipfile.ZipFile.read

        def recording_read(self, name, pwd=None):
            whole_reads.append(name if isinstance(name, str) else name.filename)
            return original_read(self, name, pwd)

        monkeypatch.setattr(zipfile.ZipFile, "read", recording_read)

        imported = import_snapshot(bundle, tmp_path / "dst")

        assert imported.imported_artifacts > 0
        assert "artifacts.json" in whole_reads
        assert [n for n in whole_reads if n.startswith("artifacts/")] == []
        assert _statuses(imported.notebook_dir) == _statuses(ran)


class TestAFailureHalfwayThrough:
    def test_it_leaves_neither_a_half_notebook_nor_its_staging(self, ran, tmp_path):
        """A failed import leaves no half-written notebook, which discovery would list
        and which would block the retry.
        """
        bundle = _export(ran, tmp_path / "snap.zip")
        broken = tmp_path / "broken.zip"
        with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(broken, "w") as dst:
            members = src.namelist()
            dropped = next(n for n in members if n.startswith("artifacts/"))
            for name in members:
                if name != dropped:
                    dst.writestr(name, src.read(name))

        parent = tmp_path / "notebooks"
        with pytest.raises(KeyError):
            import_snapshot(broken, parent / "dst")

        assert not (parent / "dst").exists()
        assert [p.name for p in parent.iterdir()] == []

    def test_the_retry_then_succeeds(self, ran, tmp_path):
        bundle = _export(ran, tmp_path / "snap.zip")
        broken = tmp_path / "broken.zip"
        with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(broken, "w") as dst:
            members = src.namelist()
            dropped = next(n for n in members if n.startswith("artifacts/"))
            for name in members:
                if name != dropped:
                    dst.writestr(name, src.read(name))
        dest = tmp_path / "notebooks" / "dst"
        with pytest.raises(KeyError):
            import_snapshot(broken, dest)

        imported = import_snapshot(bundle, dest)

        assert (imported.notebook_dir / "notebook.toml").exists()


class TestTheCommandLine:
    def _args(self, path, out=None):
        import argparse

        return argparse.Namespace(path=str(path), output_path=out, check_deps=False)

    def test_a_zip_is_imported_as_a_snapshot(self, ran, tmp_path, monkeypatch, capsys):
        """`strata import` tells a `.zip` bundle from `.ipynb` by the file."""
        from strata.notebook.cli import import_main

        monkeypatch.setenv("STRATA_NOTEBOOK_STORAGE_DIR", str(tmp_path / "root"))
        bundle = _export(ran, tmp_path / "demo.snapshot.zip")

        assert import_main(self._args(bundle)) == 0

        dest = tmp_path / "demo"
        assert (dest / "notebook.toml").exists()
        assert "Imported" in capsys.readouterr().out

    def test_an_id_in_use_under_the_storage_root_is_replaced(
        self, ran, tmp_path, monkeypatch, capsys
    ):
        from strata.notebook.cli import import_main

        root = tmp_path / "root"
        monkeypatch.setenv("STRATA_NOTEBOOK_STORAGE_DIR", str(root))
        bundle = _export(ran, tmp_path / "snap.zip")
        assert import_main(self._args(bundle, out=str(root / "first"))) == 0
        capsys.readouterr()

        assert import_main(self._args(bundle, out=str(root / "second"))) == 0

        out = capsys.readouterr().out
        assert "already in use" in out
        assert parse_notebook(root / "first").id != parse_notebook(root / "second").id

    def test_a_zip_that_is_not_a_snapshot_is_a_usage_error(self, tmp_path, monkeypatch, capsys):
        from strata.notebook.cli import import_main

        monkeypatch.setenv("STRATA_NOTEBOOK_STORAGE_DIR", str(tmp_path / "root"))
        bogus = tmp_path / "bogus.zip"
        with zipfile.ZipFile(bogus, "w") as archive:
            archive.writestr("readme.txt", "hi")

        assert import_main(self._args(bogus)) == 2
        assert "not a snapshot" in capsys.readouterr().err


class TestABundleWritesOnlyIntoTheNotebook:
    """Bundle member names are untrusted: ``cells/../../../x`` must not escape the notebook."""

    @staticmethod
    def _bundle_with(tmp_path, member: str, ran):
        good = _export(ran, tmp_path / "snap.zip")
        evil = tmp_path / "evil.zip"
        with zipfile.ZipFile(good) as src, zipfile.ZipFile(evil, "w") as dst:
            for name in src.namelist():
                dst.writestr(name, src.read(name))
            dst.writestr(member, b"pwned")
        return evil

    def test_a_member_that_climbs_out_is_refused(self, ran, tmp_path):
        evil = self._bundle_with(tmp_path, "cells/../../../PWNED.txt", ran)
        target = tmp_path / "dst"

        with pytest.raises(NotASnapshotError, match="cannot write"):
            import_snapshot(evil, target / "nb")

        assert not (tmp_path / "PWNED.txt").exists()
        assert not (target / "PWNED.txt").exists()

    def test_a_member_outside_the_committed_set_is_not_written(self, ran, tmp_path):
        """Only ``cells/`` and the three lock files are written; other members are ignored."""
        evil = self._bundle_with(tmp_path, "sitecustomize.py", ran)
        target = tmp_path / "dst" / "nb"

        imported = import_snapshot(evil, target)

        assert not (imported.notebook_dir / "sitecustomize.py").exists()
        assert not (tmp_path / "sitecustomize.py").exists()

    def test_a_cell_id_that_is_a_path_is_refused(self, ran, tmp_path):
        good = _export(ran, tmp_path / "snap.zip")
        evil = tmp_path / "evil-cell.zip"
        with zipfile.ZipFile(good) as src, zipfile.ZipFile(evil, "w") as dst:
            for name in src.namelist():
                data = src.read(name)
                if name == "artifacts.json":
                    manifest = json.loads(data)
                    manifest.setdefault("cells", {})["../../../escape"] = {
                        "provenance_hash": "x",
                        "source_hash": "y",
                        "env_hash": "z",
                    }
                    data = json.dumps(manifest).encode()
                dst.writestr(name, data)

        with pytest.raises(NotASnapshotError, match="cannot write"):
            import_snapshot(evil, tmp_path / "dst" / "nb")

    def test_an_artifact_id_that_is_a_path_is_refused(self, ran, tmp_path):
        """An artifact id becomes a blob key, so an id that is a path would write outside."""
        good = _export(ran, tmp_path / "snap.zip")
        evil = tmp_path / "evil-artifact.zip"
        escaped = "../../../../victim/pwned"
        with zipfile.ZipFile(good) as src, zipfile.ZipFile(evil, "w") as dst:
            for name in src.namelist():
                data = src.read(name)
                if name == "artifacts.json":
                    manifest = json.loads(data)
                    records = manifest["records"]
                    assert records, "the fixture exports at least one record"
                    ref = next(iter(records))
                    records[ref]["id"] = escaped
                    data = json.dumps(manifest).encode()
                dst.writestr(name, data)

        with pytest.raises(NotASnapshotError, match="cannot write"):
            import_snapshot(evil, tmp_path / "dst" / "nb")

        assert not (tmp_path / "victim").exists()
        assert not (tmp_path.parent / "victim").exists()

    def test_an_ordinary_bundle_still_imports(self, ran, tmp_path):
        bundle = _export(ran, tmp_path / "snap.zip")

        imported = import_snapshot(bundle, tmp_path / "dst")

        assert (imported.notebook_dir / "cells").is_dir()


class TestWidgetSelections:
    """Widget values travel with the snapshot.

    Without them an imported copy falls back to declared defaults and recomputes a
    different scenario than the one the bundle was taken from.
    """

    @pytest.fixture
    def swept(self, tmp_path):
        from strata.notebook.executor import CellExecutor
        from strata.notebook.runtime_state import persist_cell_widget_values

        nb = create_notebook(tmp_path / "src", "Widget Snapshot", initialize_environment=False)
        prepared_venv(nb)
        add_cell_to_notebook(nb, "controls", None, language="widget")
        write_cell(nb, "controls", "alpha = slider(0, 1, default=0.5)\n")

        session = NotebookSession(parse_notebook(nb), nb)
        session._analyze_and_build_dag()
        # The selection a person made, which is not the declared default.
        persist_cell_widget_values(nb, "controls", {"alpha": 0.25})
        return nb, session, CellExecutor(session)

    @pytest.mark.asyncio
    async def test_the_selection_survives_the_round_trip(self, swept, tmp_path):
        from strata.notebook.runtime_state import load_runtime_state

        nb, session, executor = swept
        await executor.execute_cell("controls", session.notebook_state.get_cell("controls").source)

        bundle = _export(nb, tmp_path / "widget.zip")
        imported = import_snapshot(bundle, tmp_path / "dst")

        entry = load_runtime_state(imported.notebook_dir).cells["controls"]
        assert entry.widget_values == {"alpha": 0.25}

        # And the imported notebook reports it, not the 0.5 its source declares (the
        # difference an importer would otherwise silently run).
        opened = NotebookSession(parse_notebook(imported.notebook_dir), imported.notebook_dir)
        payload = opened.serialize_notebook_state()
        controls = next(c for c in payload["cells"] if c["id"] == "controls")
        assert controls["widget"]["values"] == {"alpha": 0.25}
        assert controls["widget"]["descriptors"][0]["default"] == 0.5
