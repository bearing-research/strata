"""A notebook's state at a moment, as one bundle.

``fmt=zip`` carried the committed files and no outputs; the HTML and markdown
exports carried rendered outputs and no machine-readable provenance. Assembling
a reviewable snapshot took three requests and two formats, and the result still
could not seed a sandbox, because nothing in it carried bytes. Item 42.
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
        """What makes a snapshot reviewable rather than merely openable: two
        snapshots of the same notebook can be compared output by output with
        neither side reading a blob."""
        manifest = _manifest(_bundle(session))

        assert manifest["artifacts"]["rows"][0]["content_sha256"]

    def test_a_cell_that_produced_nothing_is_absent(self, session):
        """Only variables a downstream cell reads become artifacts, so a cell
        with no entry is the normal case rather than a gap."""
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
        """A review snapshot: the figure attached, everything else by
        reference. The index still names what was left out, which is how an
        importer knows to mark those cells stale rather than inferring it from
        what it failed to find."""
        bundle = _bundle(session, include="selected", selected_cells=["total"])
        manifest = _manifest(bundle)

        assert manifest["carried"] == []
        assert manifest["artifacts"]["rows"], "the omitted cell is still described"

    def test_selected_names_cells_not_artifacts(self, session):
        """A reviewer thinks "attach the figure cell", not "attach
        nb_…_var___display__0@v=3"."""
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
        """They live in ``.strata/``, which is gitignored and in neither
        existing export — so a snapshot assembled from those two could not say
        what a cell's last run cost or hashed to."""
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
        """One format with a parameter, not two formats that drift: everything
        ``fmt=zip`` produces is still there."""
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
        """Opened offline rather than through a session manager — working on a
        directory with no server running is the CLI's whole point — but the
        members come from one implementation."""
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
        """It is a zip, not text, so there is nothing sensible to put on
        stdout — and a caller who omitted --out expected a file."""
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
        """notebook.toml names `note.md`; a bundle that globbed only `*.py`
        described a cell it did not contain, which no sandbox can be seeded
        from. Found by review — the original fixture was Python-only, so the
        superset test passed while the two bundles genuinely differed."""
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
        """One implementation, asserted rather than claimed. These diverged:
        the route wrote provenance.json and globbed `*.py`, the CLI wrote no
        provenance and globbed everything."""
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
        """A PNG display output as a parsed session actually holds one.

        ``inline_data_url`` is stripped before persistence and the parser
        rebuilds without it, so a session read from disk has only the
        ``artifact_uri`` — which is every CLI export and every server restart.
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
        """Not a JSON stub describing one. The documented layout says
        `outputs/<cell id>/0.png`, and it has to actually be the bytes."""
        bundle = _bundle(self._with_image(session))

        assert "outputs/note/0.png" in bundle.namelist()
        assert bundle.read("outputs/note/0.png").startswith(b"\x89PNG")

    def test_a_table_preview_is_still_described(self, session):
        """There is no format in which double-clicking a row preview means
        anything, so those stay JSON."""
        from strata.notebook.models import CellOutput

        cell = session.notebook_state.get_cell("total")
        cell.display_outputs = [CellOutput(content_type="json/object", bytes=2, preview=6)]

        bundle = _bundle(session)

        assert json.loads(bundle.read("outputs/total/0.json"))["preview"] == 6


class TestASelectionThatNamesNothing:
    def test_a_mistyped_cell_id_is_refused(self, session):
        """It used to answer 200 with an empty `carried`, indistinguishable
        from a selection that legitimately had nothing — and the caller found
        out when the snapshot turned out to be missing the figure."""
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
        """Not every cell has artifacts, and naming one that does not is a
        legitimate request rather than a typo."""
        bundle = _bundle(session, include="selected", selected_cells=["total"])

        assert _manifest(bundle)["carried"] == []
