"""A notebook's state at a moment, as one bundle.

The snapshot carries committed files, outputs and provenance together, so it
can be reviewed and can seed a sandbox.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.snapshot import write_snapshot
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.fixture
def session(tmp_path):
    """A two-cell notebook whose first cell has a stored output."""
    nb = create_notebook(tmp_path, "Snapshot", initialize_environment=False)
    add_cell_to_notebook(nb, "rows", None)
    write_cell(nb, "rows", "rows = [1, 2, 3]")
    add_cell_to_notebook(nb, "total", "rows")
    write_cell(nb, "total", "total = sum(rows)")
    # A non-Python cell, because notebook.toml names its file and a bundle that
    # globbed only *.py would describe a cell it does not contain.
    add_cell_to_notebook(nb, "note", "total", language="markdown")
    write_cell(nb, "note", "# A note")

    session = NotebookSession(parse_notebook(nb), nb)
    session.get_artifact_manager().store_cell_output(
        cell_id="rows",
        variable_name="rows",
        blob_data=b"[1, 2, 3]",
        content_type="json/object",
        provenance_hash="a1" * 32,
        input_versions={},
        source="rows = [1, 2, 3]",
    )
    return session


def _bundle(session, **kwargs) -> zipfile.ZipFile:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        write_snapshot(session, archive, **kwargs)
    return zipfile.ZipFile(io.BytesIO(buffer.getvalue()))


def _manifest(bundle) -> dict:
    return json.loads(bundle.read("artifacts.json"))


class TestTheIndex:
    def test_it_names_every_ready_cell_s_artifacts(self, session):
        manifest = _manifest(_bundle(session))

        assert list(manifest["artifacts"]) == ["rows"]
        entry = manifest["artifacts"]["rows"][0]
        assert entry["variable"] == "rows"
        assert entry["provenance_hash"] == "a1" * 32

    def test_every_entry_carries_a_digest(self, session):
        """Digests let two snapshots be compared output by output without reading blobs."""
        manifest = _manifest(_bundle(session))

        assert manifest["artifacts"]["rows"][0]["content_sha256"]

    def test_a_cell_that_produced_nothing_is_absent(self, session):
        """Only variables a downstream cell reads become artifacts, so no entry is normal."""
        assert "total" not in _manifest(_bundle(session))["artifacts"]


class TestIncludeModes:
    def test_all_carries_every_byte(self, session):
        """The form a move between servers uses."""
        bundle = _bundle(session, include="all")

        carried = _manifest(bundle)["carried"]
        assert len(carried) == 1
        assert bundle.read(f"artifacts/{carried[0]}") == b"[1, 2, 3]"

    def test_none_carries_none(self, session):
        bundle = _bundle(session, include="none")

        assert _manifest(bundle)["carried"] == []
        assert not [n for n in bundle.namelist() if n.startswith("artifacts/")]

    def test_selected_carries_the_named_cells_and_describes_the_rest(self, session):
        """A review snapshot: named cells carried, the rest described by reference.

        The index names what was left out so an importer marks those cells stale.
        """
        bundle = _bundle(session, include="selected", selected_cells=["total"])
        manifest = _manifest(bundle)

        assert manifest["carried"] == []
        assert manifest["artifacts"]["rows"], "the omitted cell is still described"

    def test_selected_names_cells_not_artifacts(self, session):
        """Selection is by cell id, not by artifact name."""
        bundle = _bundle(session, include="selected", selected_cells=["rows"])

        assert len(_manifest(bundle)["carried"]) == 1


class TestPerCellState:
    def test_the_console_travels(self, session, tmp_path):
        from strata.notebook.writer import update_cell_console_output

        update_cell_console_output(session.path, "total", "6\n", "")

        bundle = _bundle(session)

        assert json.loads(bundle.read("outputs/total/console.json"))["stdout"] == "6\n"

    def test_a_cell_with_no_console_writes_no_file(self, session):
        bundle = _bundle(session)

        assert "outputs/rows/console.json" not in bundle.namelist()

    def test_provenance_and_timings_come_from_runtime_json(self, session):
        """Provenance and timings live in gitignored ``.strata/``; only snapshots carry them."""
        from strata.notebook.runtime_state import (
            persist_cell_execution_sample,
            persist_cell_provenance,
        )

        persist_cell_provenance(
            session.path,
            "rows",
            last_provenance_hash="a1" * 32,
            last_source_hash="src",
            last_env_hash="env",
        )
        persist_cell_execution_sample(session.path, "rows", duration_ms=1234, cache_hit=False)

        manifest = _manifest(_bundle(session))

        assert manifest["cells"]["rows"]["provenance_hash"] == "a1" * 32
        assert manifest["cells"]["rows"]["execution_samples"][-1]["duration_ms"] == 1234


class TestTheRoute:
    def test_snapshot_is_a_superset_of_the_zip(self, session, monkeypatch):
        """Everything ``fmt=zip`` produces is still in the snapshot."""
        import asyncio

        from strata.notebook.routes import export_notebook

        async def _collect(fmt, **kwargs):
            response = await export_notebook("nb", session, fmt=fmt, **kwargs)
            chunks = [chunk async for chunk in response.body_iterator]
            return b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks)

        def _read(fmt, **kwargs):
            return zipfile.ZipFile(io.BytesIO(asyncio.run(_collect(fmt, **kwargs))))

        plain = set(_read("zip").namelist())
        snapshot = set(_read("snapshot", include="all").namelist())

        assert plain <= snapshot
        assert "artifacts.json" in snapshot
        assert any(n.startswith("artifacts/") for n in snapshot)

    def test_an_unknown_include_is_refused(self, session):
        import asyncio

        from fastapi import HTTPException

        from strata.notebook.routes import export_notebook

        with pytest.raises(HTTPException) as caught:
            asyncio.run(export_notebook("nb", session, fmt="snapshot", include="everything"))

        assert caught.value.status_code == 400


class TestTheCLI:
    def test_it_writes_the_same_bundle_the_route_serves(self, session, tmp_path, capsys):
        """The CLI writes offline, with no server, but from the same implementation as the route."""
        import argparse

        from strata.notebook.cli import export_main

        out = tmp_path / "snap.zip"
        rc = export_main(
            argparse.Namespace(
                path=str(session.path),
                output_format="snapshot",
                output_path=str(out),
                include="all",
                cells=None,
                include_inactive_variants=False,
                no_console=False,
                app_view=False,
                max_output_bytes=None,
            )
        )

        assert rc == 0
        with zipfile.ZipFile(out) as bundle:
            names = set(bundle.namelist())
            manifest = json.loads(bundle.read("artifacts.json"))
        assert "notebook.toml" in names
        assert "cells/rows.py" in names
        assert len(manifest["carried"]) == 1

    def test_a_snapshot_without_an_out_path_is_refused(self, session, capsys):
        """A zip has nothing sensible to put on stdout, so a missing --out is refused."""
        import argparse

        from strata.notebook.cli import export_main

        rc = export_main(
            argparse.Namespace(
                path=str(session.path),
                output_format="snapshot",
                output_path=None,
                include="all",
                cells=None,
                include_inactive_variants=False,
                no_console=False,
                app_view=False,
                max_output_bytes=None,
            )
        )

        assert rc == 2


class TestEveryCellFileTravels:
    def test_a_markdown_cell_s_file_is_in_the_bundle(self, session, tmp_path, capsys):
        """A markdown cell's file must be in the bundle, not only `*.py`.

        A Python-only fixture let the superset test pass while the bundles differed.
        """
        import argparse
        import zipfile as _zipfile

        from strata.notebook.cli import export_main

        out = tmp_path / "snap.zip"
        export_main(
            argparse.Namespace(
                path=str(session.path),
                output_format="snapshot",
                output_path=str(out),
                include="all",
                cells=None,
                include_inactive_variants=False,
                no_console=False,
                app_view=False,
                max_output_bytes=None,
            )
        )

        with _zipfile.ZipFile(out) as bundle:
            assert "cells/note.md" in bundle.namelist()

    def test_the_route_and_the_cli_agree_on_the_members(self, session, tmp_path):
        """The route and the CLI produce the same members."""
        import argparse
        import asyncio
        import zipfile as _zipfile

        from strata.notebook.cli import export_main
        from strata.notebook.routes import export_notebook

        async def _collect():
            response = await export_notebook("nb", session, fmt="snapshot", include="all")
            chunks = [chunk async for chunk in response.body_iterator]
            return b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks)

        from_route = set(_zipfile.ZipFile(io.BytesIO(asyncio.run(_collect()))).namelist())

        out = tmp_path / "snap.zip"
        export_main(
            argparse.Namespace(
                path=str(session.path),
                output_format="snapshot",
                output_path=str(out),
                include="all",
                cells=None,
                include_inactive_variants=False,
                no_console=False,
                app_view=False,
                max_output_bytes=None,
            )
        )
        with _zipfile.ZipFile(out) as bundle:
            from_cli = set(bundle.namelist())

        assert from_route == from_cli
        assert "provenance.json" in from_cli


class TestDisplayOutputs:
    def _with_image(self, session):
        """A PNG display output as a session parsed from disk holds one.

        ``inline_data_url`` is stripped before persistence, so only ``artifact_uri`` is
        left, as after every server restart and in every CLI export.
        """
        from strata.notebook.models import CellOutput

        manager = session.get_artifact_manager()
        stored = manager.store_cell_output(
            cell_id="note",
            variable_name="__display__0",
            blob_data=b"\x89PNG\r\n\x1a\nPRETEND",
            content_type="image/png",
            provenance_hash="b2" * 32,
            input_versions={},
            source="",
        )
        cell = session.notebook_state.get_cell("note")
        cell.display_outputs = [
            CellOutput(
                content_type="image/png",
                bytes=15,
                artifact_uri=f"strata://artifact/{stored.id}@v={stored.version}",
            )
        ]
        return session

    def test_an_image_is_written_as_a_png(self, session):
        """The image lands as `outputs/<cell id>/0.png` holding the bytes, not a JSON stub."""
        bundle = _bundle(self._with_image(session))

        assert "outputs/note/0.png" in bundle.namelist()
        assert bundle.read("outputs/note/0.png").startswith(b"\x89PNG")

    def test_a_table_preview_is_still_described(self, session):
        """Table previews have no useful file format, so they stay JSON."""
        from strata.notebook.models import CellOutput

        cell = session.notebook_state.get_cell("total")
        cell.display_outputs = [CellOutput(content_type="json/object", bytes=2, preview=6)]

        bundle = _bundle(session)

        assert json.loads(bundle.read("outputs/total/0.json"))["preview"] == 6


class TestASelectionThatNamesNothing:
    def test_a_mistyped_cell_id_is_refused(self, session):
        """A mistyped cell id is refused rather than answering 200 with an empty `carried`."""
        import asyncio

        from fastapi import HTTPException

        from strata.notebook.routes import export_notebook

        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                export_notebook("nb", session, fmt="snapshot", include="selected", cells="figur")
            )

        assert caught.value.status_code == 400
        assert "figur" in caught.value.detail

    def test_a_real_cell_that_produced_nothing_is_still_fine(self, session):
        """Naming a real cell with no artifacts is legitimate, not a typo."""
        bundle = _bundle(session, include="selected", selected_cells=["total"])

        assert _manifest(bundle)["carried"] == []


class TestFetches:
    """A preflight flags unpinned fetches from the snapshot alone."""

    def test_every_fetch_is_listed_with_whether_it_is_pinned(self, tmp_path):
        nb = create_notebook(tmp_path, "Fetches", initialize_environment=False)
        pin = "b" * 64
        read = "c" * 64
        add_cell_to_notebook(nb, "loose", None)
        write_cell(nb, "loose", "# @fetch zones https://example.org/zones.csv\nrows = 1")
        add_cell_to_notebook(nb, "tight", "loose")
        write_cell(
            nb,
            "tight",
            f"# @fetch rates https://example.org/rates.csv sha256={pin} refetch=never\nr = 1",
        )
        # What the loose fetch last read, as the cache records it.
        blob = nb / ".strata" / "fetch" / read / "zones.csv"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"zone\n")
        (nb / ".strata" / "fetch" / "index.json").write_text(
            json.dumps({"https://example.org/zones.csv": {"sha256": read, "filename": "zones.csv"}})
        )

        manifest = _manifest(_bundle(NotebookSession(parse_notebook(nb), nb), include="none"))

        assert manifest["fetches"] == [
            {
                "cell_id": "loose",
                "name": "zones",
                "url": "https://example.org/zones.csv",
                "pinned": False,
                "sha256": read,
                "refetch": "stale",
            },
            {
                "cell_id": "tight",
                "name": "rates",
                "url": "https://example.org/rates.csv",
                "pinned": True,
                "sha256": pin,
                "refetch": "never",
            },
        ]


class TestWritingTheBundleOut:
    """An --out path that already holds something is refused, as ``strata artifact archive``
    does."""

    @staticmethod
    def _export(notebook_dir, out, *extra):
        from strata.cli import _build_parser
        from strata.notebook.cli import export_main

        args = _build_parser().parse_args(
            ["export", str(notebook_dir), "--to", "snapshot", "--out", str(out), *extra]
        )
        return export_main(args)

    def test_an_existing_file_is_not_overwritten(self, session, tmp_path):
        out = tmp_path / "precious.csv"
        out.write_text("measurements,1,2,3\n")

        code = self._export(session.path, out)

        assert code == 2
        assert out.read_text() == "measurements,1,2,3\n"

    def test_force_overwrites_it(self, session, tmp_path):
        out = tmp_path / "precious.csv"
        out.write_text("measurements,1,2,3\n")

        code = self._export(session.path, out, "--force")

        assert code == 0
        assert zipfile.is_zipfile(out)

    def test_a_fresh_path_needs_no_flag(self, session, tmp_path):
        out = tmp_path / "snap.zip"

        assert self._export(session.path, out) == 0
        assert zipfile.is_zipfile(out)


class TestTheCommittedSet:
    def test_the_bundle_carries_exactly_the_committed_set(self, session):
        """Cell tests travel; the agent files `strata agent` writes do not."""
        from strata.notebook.layout import committed_paths
        from strata.notebook.snapshot import write_committed_files

        tests_dir = session.path / "cells" / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_rows.py").write_text("def test_rows(): pass\n")
        (session.path / ".mcp.json").write_text("{}\n")
        (session.path / "CLAUDE.md").write_text("# agent\n")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            write_committed_files(session, archive)
        names = set(zipfile.ZipFile(io.BytesIO(buffer.getvalue())).namelist())

        assert "cells/tests/test_rows.py" in names
        assert ".mcp.json" not in names
        assert "CLAUDE.md" not in names
        assert names - {"provenance.json"} == {p.as_posix() for p in committed_paths(session.path)}
