"""A run reports what it read that its cell does not declare, and the declared reads not."""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import cloudpickle
import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

from strata.notebook import read_audit
from strata.notebook.env import audit_scope
from strata.notebook.executor import CellExecutor
from strata.notebook.models import MountSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.read_audit import (
    CHANGED_DURING_RUN,
    ReadFinding,
    fingerprint,
    merge,
    recording,
    report,
)
from strata.notebook.session import NotebookSession
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    update_notebook_mounts,
    write_cell,
)
from tests.notebook.pool_helpers import warm_again


def _codes(findings: list[ReadFinding]) -> dict[str, list[str]]:
    return {finding["code"]: finding["items"] for finding in findings}


def _stats(findings: list[ReadFinding]) -> dict[str, dict[str, list[int]]]:
    return {f["code"]: f["fingerprints"] for f in findings if "fingerprints" in f}


def _unchanged(findings: list[ReadFinding]) -> dict[str, dict[str, list[int]]]:
    """What the findings' fingerprints should be: each path as it is now."""
    return {
        f["code"]: {path: fingerprint(path) for path in f["items"]}
        for f in findings
        if f["code"] not in ("unlocked_package", "subprocess_run", "audit_failed")
    }


def _report(observed, *, source="", mounts=None, run_dir="/nonexistent-run", scope=None):
    return _codes(
        report(observed, source=source, mounts=mounts or {}, run_dir=run_dir, scope=scope)
    )


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    """A folder no root covers, and a cwd beside it that the run counts as its ``work/``."""
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "a.txt").write_text("a")
    (folder / "b.txt").write_text("b")
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    return folder.resolve()


class TestTheRecord:
    def test_a_read_outside_every_root_is_reported(self, data):
        with recording() as observed:
            (data / "a.txt").read_text()

        assert _report(observed) == {"undeclared_read": [str(data / "a.txt")]}

    def test_a_relative_path_is_where_the_cell_was_when_it_opened_it(self, data):
        with recording() as observed:
            Path("../data/a.txt").read_text()
            os.chdir(data)
            Path("b.txt").read_text()

        assert _report(observed) == {"undeclared_read": [str(data / "a.txt"), str(data / "b.txt")]}

    def test_a_file_the_run_wrote_first_is_its_own(self, data):
        with recording() as observed:
            (data / "new.txt").write_text("mine")
            (data / "new.txt").read_text()
            (data / "a.txt").read_text()
            (data / "a.txt").write_text("read first")

        assert _report(observed) == {"undeclared_read": [str(data / "a.txt")]}

    def test_a_mount_and_the_run_directory_cover_their_files(self, data, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "input.arrow").write_text("x")
        with recording() as observed:
            (data / "a.txt").read_text()
            (run_dir / "input.arrow").read_text()

        mounts = {"data": {"local_path": str(data), "mode": "ro"}}
        assert _report(observed, mounts=mounts, run_dir=str(run_dir)) == {}

    def test_a_link_to_a_mount_is_the_mount(self, data, tmp_path):
        (tmp_path / "link").symlink_to(data, target_is_directory=True)
        with recording() as observed:
            (tmp_path / "link" / "a.txt").read_text()

        assert _report(observed, mounts={"d": {"local_path": str(data)}}) == {}

    def test_a_tools_state_in_the_home_is_not_the_cells_input(self, data, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".matplotlib").mkdir()
        (tmp_path / ".matplotlib" / "fontlist.json").write_text("{}")
        (tmp_path / "notes.txt").write_text("mine")
        with recording() as observed:
            (tmp_path / ".matplotlib" / "fontlist.json").read_text()
            (tmp_path / "notes.txt").read_text()

        assert _report(observed) == {"undeclared_read": [str(tmp_path.resolve() / "notes.txt")]}

    def test_a_file_in_work_another_run_wrote_is_its_own_finding(self, data, tmp_path):
        (tmp_path / "work" / "features.txt").write_text("from another cell")
        with recording() as observed:
            Path("features.txt").read_text()

        work = (tmp_path / "work").resolve()
        assert _report(observed) == {"work_dir_read": [str(work / "features.txt")]}

    def test_a_listing_is_reported_and_an_imports_search_is_not(self, data, monkeypatch):
        monkeypatch.syspath_prepend(str(data))
        with recording() as observed:
            os.listdir(data / "..")
            sorted(data.parent.glob("data/*.txt"))
            os.listdir(data)  # on sys.path: what an import's search lists

        listed = _report(observed)["undeclared_listing"]
        assert listed == [str(data.parent)]

    def test_a_program_started_is_reported(self, data):
        with recording() as observed:
            subprocess.run([sys.executable, "-c", "pass"], check=True)

        assert _report(observed) == {"subprocess_run": [Path(sys.executable).name]}

    def test_a_module_from_a_path_the_cell_added_is_reported_once(self, data, monkeypatch):
        (data / "audit_util_mod.py").write_text("VALUE = 1\n")
        monkeypatch.syspath_prepend(str(data))
        monkeypatch.delitem(sys.modules, "audit_util_mod", raising=False)
        with recording() as observed:
            import audit_util_mod  # noqa: F401

        # Its source and bytecode were read, as an import, not as data.
        assert _report(observed) == {"undeclared_import": [str(data / "audit_util_mod.py")]}

    def test_a_module_already_loaded_is_reported_where_the_source_imports_it(
        self, data, monkeypatch
    ):
        """A batch's second cell imports what the first loaded: no new module, the same input."""
        (data / "audit_seen_mod.py").write_text("VALUE = 1\n")
        monkeypatch.syspath_prepend(str(data))
        monkeypatch.delitem(sys.modules, "audit_seen_mod", raising=False)
        import audit_seen_mod  # noqa: F401

        with recording() as observed:
            pass

        assert _report(observed) == {}
        assert _report(observed, source="import audit_seen_mod\n") == {
            "undeclared_import": [str(data / "audit_seen_mod.py")]
        }

    def test_a_module_from_a_dist_the_lock_does_not_hold(self, tmp_path, monkeypatch):
        site_packages = tmp_path / "site-packages"
        (site_packages / "audit_rogue").mkdir(parents=True)
        (site_packages / "audit_rogue" / "__init__.py").write_text("")
        monkeypatch.syspath_prepend(str(site_packages))
        monkeypatch.delitem(sys.modules, "audit_rogue", raising=False)
        with recording() as observed:
            import audit_rogue  # noqa: F401

        scope = {"site_packages": [str(site_packages)], "unlocked": {"audit_rogue": "rogue"}}
        assert _report(observed, scope=scope) == {"unlocked_package": ["rogue"]}
        # A root the lock keys (a local package's) is declared.
        assert _report(observed, scope={"package_roots": [str(site_packages)]}) == {}

    def test_past_the_cap_the_record_says_it_is_partial(self, data, monkeypatch):
        monkeypatch.setattr(read_audit, "PATH_CAP", 1)
        with recording() as observed:
            (data / "a.txt").read_text()
            (data / "b.txt").read_text()

        (finding,) = report(observed, source="", mounts={}, run_dir="/x", scope=None)
        assert finding == ReadFinding(
            code="undeclared_read", items=[str(data / "a.txt")], total=1, partial=True
        )

    def test_a_loops_iterations_are_one_cells_findings(self):
        def read(*items: str) -> ReadFinding:
            return ReadFinding(
                code="undeclared_read", items=list(items), total=len(items), partial=False
            )

        assert merge([[read("/a")], [read("/a", "/b")], []]) == [read("/a", "/b")]

    def test_a_local_run_fingerprints_each_path_it_names(self, data):
        with recording() as observed:
            (data / "a.txt").read_text()
            os.listdir(data)
            subprocess.run(["true"], check=True)

        local = report(observed, source="", mounts={}, run_dir="/x", scope={})
        remote = report(observed, source="", mounts={}, run_dir="/x", scope=None)

        st = os.stat(data / "a.txt")
        assert _stats(local) == {
            "undeclared_read": {str(data / "a.txt"): [st.st_size, st.st_mtime_ns]},
            "undeclared_listing": {str(data): fingerprint(str(data))},
        }
        # A worker's paths are its own: the server could not stat them again.
        assert _stats(remote) == {}

    def test_a_file_changed_during_the_run_is_changed_already(self, data):
        with recording() as observed:
            (data / "a.txt").read_text()
            (data / "b.txt").read_text()
        # Another process, while the cell still ran.
        (data / "a.txt").write_text("changed by someone else")

        (finding,) = report(observed, source="", mounts={}, run_dir="/x", scope={})

        assert finding["fingerprints"] == {
            str(data / "a.txt"): CHANGED_DURING_RUN,
            str(data / "b.txt"): fingerprint(str(data / "b.txt")),
        }

    @pytest.mark.parametrize("mode", ["w", "r+"])
    def test_a_file_the_run_read_then_changed_is_as_the_run_left_it(self, data, mode):
        with recording() as observed:
            (data / "a.txt").read_text()
            with open(data / "a.txt", mode) as f:
                f.write("the run's own, longer")

        (finding,) = report(observed, source="", mounts={}, run_dir="/x", scope={})

        assert finding["fingerprints"] == {str(data / "a.txt"): fingerprint(str(data / "a.txt"))}

    def test_opening_a_directory_is_not_a_listing(self, data):
        with recording() as observed:
            fd = os.open(data, os.O_RDONLY)
            os.close(fd)

        # Its fingerprint lists it, which the hook must not take for the cell's.
        assert observed.listings == set()

    def test_a_listing_is_its_names(self, data, tmp_path):
        before = fingerprint(str(data))
        # An atomic save: a sibling written, then renamed over the file.
        (data / ".a.txt.tmp").write_text("saved again")
        os.replace(data / ".a.txt.tmp", data / "a.txt")
        assert fingerprint(str(data)) == before

        (data / "c.txt").write_text("c")
        assert fingerprint(str(data)) != before

    def test_a_copy_with_its_times_is_the_same_file(self, data, tmp_path):
        before = fingerprint(str(data / "a.txt"))
        # A new inode (a copied notebook, a restarted container's overlay), same file.
        shutil.copy2(data / "a.txt", tmp_path / "copy.txt")
        os.replace(tmp_path / "copy.txt", data / "a.txt")
        assert fingerprint(str(data / "a.txt")) == before

    def test_a_path_gone_by_the_report_has_no_fingerprint(self, data):
        with recording() as observed:
            (data / "a.txt").read_text()
            (data / "a.txt").unlink()

        (finding,) = report(observed, source="", mounts={}, run_dir="/x", scope={})

        assert finding["items"] == [str(data / "a.txt")]
        assert finding["fingerprints"] == {}

    def test_a_loop_keeps_its_newest_iterations_fingerprint(self):
        def read(*st: int) -> ReadFinding:
            return ReadFinding(
                code="undeclared_read",
                items=["/a"],
                total=1,
                partial=False,
                fingerprints={"/a": list(st)},
            )

        # A later iteration's own write is the cell's, not a change after its result.
        assert merge([[read(1)], [read(2)]]) == [read(2)]
        # A change while any iteration ran stays one.
        assert merge([[read(1)], [read()], [read(2)]]) == [read()]


class TestANativeReader:
    """pyarrow, polars and duckdb open files in native code; their readers record them."""

    @pytest.fixture
    def dataset(self, data) -> Path:
        folder = data / "dataset"
        (folder / "year=2026").mkdir(parents=True)
        for name in ("year=2026/a.parquet", "year=2026/b.parquet"):
            pq.write_table(pa.table({"a": [1]}), folder / name)
        return folder

    @pytest.mark.parametrize(
        "read",
        [
            lambda path: pq.read_table(path),
            lambda path: pq.ParquetFile(path).read(),
            lambda path: pd.read_parquet(path),
            lambda path: pl.read_parquet(path),
            lambda path: pl.scan_parquet(path).collect(),
            lambda path: duckdb.read_parquet(str(path)).fetchall(),
            lambda path: duckdb.connect().read_parquet(str(path)).fetchall(),
            lambda path: pa.memory_map(str(path)).read(),
        ],
        ids=[
            "pq.read_table",
            "pq.ParquetFile",
            "pd.read_parquet",
            "pl.read_parquet",
            "pl.scan_parquet",
            "duckdb.read_parquet",
            "connection.read_parquet",
            "pa.memory_map",
        ],
    )
    def test_its_file_is_read(self, data, read):
        pq.write_table(pa.table({"a": [1]}), data / "t.parquet")
        with recording() as observed:
            read(data / "t.parquet")

        assert _report(observed) == {"undeclared_read": [str(data / "t.parquet")]}

    def test_a_folder_is_each_file_in_it_and_its_listings(self, dataset):
        with recording() as observed:
            pq.read_table(dataset)

        assert _report(observed) == {
            "undeclared_listing": [str(dataset), str(dataset / "year=2026")],
            "undeclared_read": [
                str(dataset / "year=2026" / "a.parquet"),
                str(dataset / "year=2026" / "b.parquet"),
            ],
        }

    def test_a_glob_is_what_it_matches(self, dataset):
        with recording() as observed:
            pl.read_parquet(dataset / "year=2026" / "a*.parquet")

        assert _report(observed) == {
            "undeclared_listing": [str(dataset / "year=2026")],
            "undeclared_read": [str(dataset / "year=2026" / "a.parquet")],
        }

    def test_a_list_and_a_file_uri_are_their_files(self, dataset):
        a, b = sorted((dataset / "year=2026").iterdir())
        with recording() as observed:
            pl.read_parquet([a, f"file://{b}"])

        assert _report(observed) == {"undeclared_read": [str(a), str(b)]}

    def test_a_buffer_and_another_filesystems_path_are_not_files_here(self, data):
        pq.write_table(pa.table({"a": [1]}), data / "t.parquet")
        # The same name in the cwd: read as a local path, it would be this one.
        pq.write_table(pa.table({"a": [1]}), "t.parquet")
        subtree = pafs.SubTreeFileSystem(str(data), pafs.LocalFileSystem())
        body = (data / "t.parquet").read_bytes()
        with recording() as observed:
            pq.read_table(pa.BufferReader(body))
            pq.read_table("t.parquet", filesystem=subtree)

        assert _report(observed) == {}

    def test_a_local_filesystems_path_is_a_file(self, data):
        pq.write_table(pa.table({"a": [1]}), data / "t.parquet")
        with recording() as observed:
            pq.read_table(str(data / "t.parquet"), filesystem=pafs.LocalFileSystem())

        assert _report(observed) == {"undeclared_read": [str(data / "t.parquet")]}

    def test_a_path_in_the_home_is_where_the_reader_expands_it(self, data, monkeypatch):
        monkeypatch.setenv("HOME", str(data))
        pq.write_table(pa.table({"a": [1]}), data / "t.parquet")
        with recording() as observed:
            pl.read_parquet("~/t.parquet")

        assert _report(observed) == {"undeclared_read": [str(data / "t.parquet")]}

    def test_a_map_opened_to_write_is_the_runs_write(self, data):
        (data / "m.bin").write_bytes(b"0000")
        with recording() as observed:
            with pa.memory_map(str(data / "m.bin"), "r+") as mapped:
                mapped.write(b"11")

        # Its fingerprint is then as the run left it, not a change during the run.
        assert observed.written == {str(data / "m.bin")}

    def test_a_reader_imported_during_the_run_is_wrapped(self, data):
        pq.write_table(pa.table({"a": [1]}), data / "t.parquet")
        script = textwrap.dedent(
            f"""\
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location("audit", {read_audit.__file__!r})
            audit = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(audit)
            assert "polars" not in sys.modules
            with audit.recording() as observed:
                import polars as pl
                pl.read_parquet({str(data / "t.parquet")!r})
            print({str(data / "t.parquet")!r} in observed.files)
            """
        )

        out = subprocess.run(
            [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True
        )

        assert out.stdout.strip() == "True"

    def test_an_audit_that_fails_does_not_fail_the_read(self, data, monkeypatch):
        pq.write_table(pa.table({"a": [1]}), data / "t.parquet")

        def broken(self, source, mode):
            raise RuntimeError("audit bug")

        monkeypatch.setattr(read_audit.Observed, "native", broken)
        with recording() as observed:
            table = pq.read_table(data / "t.parquet")

        assert table.num_rows == 1
        assert observed.partial

    def test_a_name_with_pattern_characters_is_that_file(self, data, monkeypatch):
        # Not a glob: pyarrow reads the file of that name, and not the one it would match.
        pq.write_table(pa.table({"a": [1]}), data / "t[1].parquet")
        pq.write_table(pa.table({"a": [1]}), data / "t1.parquet")
        # Nor is the folder a relative pattern is under one: here the cwd, the run's work/.
        under = data / "exp[2]"
        under.mkdir()
        pq.write_table(pa.table({"a": [1]}), under / "u.parquet")
        monkeypatch.chdir(under)
        with recording() as observed:
            pq.read_table(data / "t[1].parquet")
            pl.read_parquet("u*.parquet")

        assert _report(observed) == {
            "undeclared_read": [str(data / "t[1].parquet")],
            "work_dir_read": [str(under / "u.parquet")],
        }

    def test_another_filesystem_passed_by_position_is_not_here(self, data):
        pq.write_table(pa.table({"a": [1]}), data / "t.parquet")
        pq.write_table(pa.table({"a": [1]}), "t.parquet")
        subtree = pafs.SubTreeFileSystem(str(data), pafs.LocalFileSystem())
        with recording() as observed:
            pq.ParquetDataset("t.parquet", subtree).read()
            ds.dataset("t.parquet", None, "parquet", subtree).to_table()

        assert _report(observed) == {}

    @pytest.mark.parametrize(
        "write",
        [
            lambda path: pq.write_table(pa.table({"a": [1]}), path),
            lambda path: pd.DataFrame({"a": [1]}).to_parquet(path),
            lambda path: pl.DataFrame({"a": [1]}).write_parquet(path),
            lambda path: pl.LazyFrame({"a": [1]}).sink_parquet(path),
            lambda path: duckdb.sql("select 1 as a").write_parquet(str(path)),
        ],
        ids=["pq.write_table", "df.to_parquet", "pl.write_parquet", "pl.sink_parquet", "duckdb"],
    )
    def test_a_file_a_writer_made_is_the_runs_own(self, data, write):
        with recording() as observed:
            write(data / "out.parquet")
            pq.read_table(data / "out.parquet")

        assert _report(observed) == {}

    def test_an_ipc_writer_that_writes_at_close_made_its_file(self, data):
        table = pa.table({"a": [1]})
        with recording() as observed:
            with ipc.new_file(str(data / "out.arrow"), table.schema) as writer:
                writer.write_table(table)
            ipc.open_file(str(data / "out.arrow")).read_all()

        assert _report(observed) == {}

    def test_a_writer_adding_to_a_folder_made_only_its_new_file(self, data):
        folder = data / "flat"
        folder.mkdir()
        pq.write_table(pa.table({"a": [1]}), folder / "a.parquet")
        with recording() as observed:
            pq.write_to_dataset(pa.table({"a": [2]}), folder)
            pq.read_table(folder)

        assert len(list(folder.iterdir())) == 2
        assert _report(observed) == {
            "undeclared_listing": [str(folder)],
            "undeclared_read": [str(folder / "a.parquet")],
        }

    def test_a_folders_bookkeeping_is_not_read(self, dataset):
        (dataset / "_SUCCESS").write_text("")
        (dataset / "year=2026" / ".a.parquet.crc").write_text("")
        (dataset / "_delta_log").mkdir()
        (dataset / "_delta_log" / "0.json").write_text("{}")
        with recording() as observed:
            pq.read_table(dataset)

        assert _report(observed) == {
            "undeclared_listing": [str(dataset), str(dataset / "year=2026")],
            "undeclared_read": [
                str(dataset / "year=2026" / "a.parquet"),
                str(dataset / "year=2026" / "b.parquet"),
            ],
        }

    def test_a_record_partial_already_still_takes_a_folder(self, dataset):
        with recording() as observed:
            observed.partial = True  # an earlier listing past the cap, say
            pq.read_table(dataset)

        assert set(observed.files) == {
            str(dataset / "year=2026" / "a.parquet"),
            str(dataset / "year=2026" / "b.parquet"),
        }

    def test_one_call_through_several_readers_records_once(self, dataset, monkeypatch):
        calls = []
        native = read_audit.Observed.native

        def counted(self, source, mode):
            calls.append(source)
            native(self, source, mode)

        monkeypatch.setattr(read_audit.Observed, "native", counted)
        with recording():
            pd.read_parquet(dataset)

        assert calls == [str(dataset)]

    def test_a_polars_line_reader_is_wrapped(self, data):
        with recording() as observed:
            pl.read_lines(data / "a.txt")
            pl.scan_lines(data / "b.txt").collect()

        assert _report(observed) == {"undeclared_read": [str(data / "a.txt"), str(data / "b.txt")]}

    def test_bytes_are_content_not_a_path(self, data):
        # Taken as a path, the bytes would name this file.
        Path("a\n1\n").write_text("")
        with recording() as observed:
            pl.read_csv(b"a\n1\n")

        assert _report(observed) == {}

    def test_another_copy_of_the_module_replaces_the_wrappers_not_nests_them(self, data):
        # The harness loads the file by path; a process running it again holds copies.
        pq.write_table(pa.table({"a": [1]}), data / "t.parquet")
        script = textwrap.dedent(
            f"""\
            import importlib.util, sys
            import pyarrow.parquet as pq

            def load(name):
                spec = importlib.util.spec_from_file_location(name, {read_audit.__file__!r})
                audit = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(audit)
                return audit

            for i in range(3):
                with load(f"audit{{i}}").recording() as observed:
                    pq.read_table({str(data / "t.parquet")!r})
            finders = [f for f in sys.meta_path if type(f).__name__ == "_WrapOnImport"]
            print(len(finders), hasattr(pq.read_table.__wrapped__, "__wrapped__"))
            print({str(data / "t.parquet")!r} in observed.files)
            """
        )

        out = subprocess.run(
            [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True
        )

        assert out.stdout.split() == ["1", "False", "True"]

    @pytest.mark.parametrize(
        "dump", [pickle.dumps, cloudpickle.dumps], ids=["pickle", "cloudpickle"]
    )
    def test_a_wrapped_reader_pickles_as_itself(self, dump):
        with recording():
            for reader in (pq.read_table, pl.DataFrame.write_parquet, pq.ParquetFile.__init__):
                assert reader.__wrapped__
                assert pickle.loads(dump(reader)) is reader


def _lock(notebook_dir: Path, *packages: str) -> None:
    lines = [
        "version = 1",
        "",
        "[[package]]",
        'name = "nb"',
        'version = "0.1.0"',
        'source = { virtual = "." }',
        "dependencies = [" + ", ".join(f'{{ name = "{p}" }}' for p in packages) + "]",
    ]
    for package in packages:
        lines += ["", "[[package]]", f'name = "{package}"', 'version = "1.0"']
    (notebook_dir / "uv.lock").write_text("\n".join(lines) + "\n")


def _install(site_packages: Path, dist: str, *files: str) -> None:
    dist_info = site_packages / f"{dist}-1.0.dist-info"
    dist_info.mkdir(parents=True)
    for file in files:
        (site_packages / file).parent.mkdir(parents=True, exist_ok=True)
        (site_packages / file).write_text("")
    (dist_info / "RECORD").write_text(
        "".join(f"{f},," + "\n" for f in [*files, f"{dist_info.name}/RECORD"])
    )


class TestTheScope:
    def test_a_dist_the_lock_does_not_hold_is_named_by_what_it_installs(self, tmp_path):
        site_packages = tmp_path / ".venv" / "lib" / "python3.14" / "site-packages"
        _lock(tmp_path, "pyarrow")
        _install(site_packages, "pyarrow", "pyarrow/__init__.py")
        _install(site_packages, "rogue", "rogue/__init__.py", "rogue/core.py", "rogue_cli.py")
        _install(site_packages, "ns_part", "ns/part/__init__.py")  # a namespace package
        _install(site_packages, "nb", "nb.pth")  # the notebook's own project

        scope = audit_scope(tmp_path)

        assert scope["site_packages"] == [str(site_packages)]
        assert scope["unlocked"] == {
            "rogue": "rogue",
            "rogue_cli.py": "rogue",
            "ns/part": "ns_part",
        }

    def test_a_local_package_is_where_its_pth_says(self, tmp_path):
        site_packages = tmp_path / "nb" / ".venv" / "lib" / "python3.14" / "site-packages"
        notebook_dir = tmp_path / "nb"
        notebook_dir.mkdir()
        _lock(notebook_dir, "mylib")
        with (notebook_dir / "uv.lock").open("a") as lock:
            lock.write('source = { editable = "../mylib" }\n')
        _install(site_packages, "mylib", "_mylib.pth")
        (site_packages / "_mylib.pth").write_text(str(tmp_path / "mylib" / "src") + "\n")

        assert audit_scope(notebook_dir)["package_roots"] == [str(tmp_path / "mylib" / "src")]

    def test_it_is_kept_until_the_lock_changes(self, tmp_path):
        _lock(tmp_path, "pyarrow")
        (tmp_path / ".venv" / "lib" / "python3.14" / "site-packages").mkdir(parents=True)
        first = audit_scope(tmp_path)
        assert audit_scope(tmp_path) is first

        _lock(tmp_path, "pyarrow", "pandas")
        assert audit_scope(tmp_path) is not first

    def test_without_a_lock_nothing_is_said(self, tmp_path):
        assert audit_scope(tmp_path) == {}


def _session(tmp_path: Path, *cells: tuple[str, str]) -> NotebookSession:
    notebook_dir = create_notebook(tmp_path / "project", "audit")
    after = None
    for cell_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after)
        write_cell(notebook_dir, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


@pytest.fixture
def outside(tmp_path) -> Path:
    folder = tmp_path / "outside"
    folder.mkdir()
    (folder / "events.csv").write_text("a,b\n1,2\n")
    (folder / "util_for_audit.py").write_text("VALUE = 41\n")
    pq.write_table(pa.table({"a": [1, 2]}), folder / "events.parquet")
    return folder.resolve()


def _runs(outside: Path) -> dict[str, str]:
    return {
        "reads": f"text = open({str(outside / 'events.csv')!r}).read()\n",
        "imports": textwrap.dedent(
            f"""\
            import sys
            sys.path.insert(0, {str(outside)!r})
            import util_for_audit
            value = util_for_audit.VALUE + 1
            """
        ),
        "own": (
            "from pathlib import Path\n"
            "Path('mine.txt').write_text('x')\n"
            "back = Path('mine.txt').read_text()\n"
        ),
        # Read in native code, where the hook sees no open: the wrapped readers record it.
        # pyarrow alone: it is the one reader a new notebook's venv (the warm pool's) has.
        "native": textwrap.dedent(
            f"""\
            import pyarrow.csv
            import pyarrow.parquet as pq
            rows = (
                pq.read_table({str(outside / "events.parquet")!r}).num_rows
                + pyarrow.csv.read_csv({str(outside / "events.csv")!r}).num_rows
            )
            """
        ),
    }


def _expected(outside: Path) -> dict[str, dict[str, list[str]]]:
    return {
        "reads": {"undeclared_read": [str(outside / "events.csv")]},
        "imports": {"undeclared_import": [str(outside / "util_for_audit.py")]},
        "own": {},
        "native": {
            "undeclared_read": [str(outside / "events.csv"), str(outside / "events.parquet")]
        },
    }


class TestARun:
    async def test_a_cold_run_reports_what_its_cell_does_not_declare(self, tmp_path, outside):
        runs = _runs(outside)
        session = _session(tmp_path, *runs.items())
        executor = CellExecutor(session)

        for cell_id, source in runs.items():
            result = await executor.execute_cell(cell_id, source)
            assert result.success, result.error
            assert _codes(result.read_audit) == _expected(outside)[cell_id], cell_id
            assert _stats(result.read_audit) == _unchanged(result.read_audit), cell_id

    async def test_an_audit_that_fails_does_not_fail_the_run(self, tmp_path):
        source = textwrap.dedent(
            """\
            import sys, types
            class Lazy(types.ModuleType):
                @property
                def __file__(self):
                    raise RuntimeError("not loaded yet")
            sys.modules["lazy_for_audit"] = Lazy("lazy_for_audit")
            value = 1
            """
        )
        session = _session(tmp_path, ("c", source))

        result = await CellExecutor(session).execute_cell("c", source)

        assert result.success, result.error
        assert _codes(result.read_audit) == {"audit_failed": ["RuntimeError: not loaded yet"]}

    async def test_a_read_through_a_mount_is_declared(self, tmp_path, outside):
        source = "text = (data / 'events.csv').read_text()\n"
        session = _session(tmp_path, ("c", source))
        update_notebook_mounts(session.path, [MountSpec(name="data", uri=str(outside))])
        session = NotebookSession(parse_notebook(session.path), session.path)
        session.refresh_environment_runtime()

        result = await CellExecutor(session).execute_cell("c", source)

        assert result.success, result.error
        assert result.read_audit == []

    async def test_polars_and_duckdb_imported_by_the_cell_are_wrapped(self, tmp_path, outside):
        source = textwrap.dedent(
            f"""\
            import duckdb
            import polars as pl
            rows = (
                pl.scan_parquet({str(outside / "events.parquet")!r}).collect().height
                + len(duckdb.read_csv({str(outside / "events.csv")!r}).fetchall())
            )
            """
        )
        session = _session(tmp_path, ("c", source))

        result = await CellExecutor(session).execute_cell("c", source)

        assert result.success, result.error
        assert _codes(result.read_audit) == {
            "undeclared_read": [str(outside / "events.csv"), str(outside / "events.parquet")]
        }
        assert _stats(result.read_audit) == _unchanged(result.read_audit)

    async def test_a_native_read_through_a_mount_is_declared(self, tmp_path, outside):
        source = "import polars as pl\nrows = pl.read_parquet(data / 'events.parquet').height\n"
        session = _session(tmp_path, ("c", source))
        update_notebook_mounts(session.path, [MountSpec(name="data", uri=str(outside))])
        session = NotebookSession(parse_notebook(session.path), session.path)
        session.refresh_environment_runtime()

        result = await CellExecutor(session).execute_cell("c", source)

        assert result.success, result.error
        assert result.read_audit == []

    async def test_a_dist_the_lock_does_not_hold_is_reported(self, tmp_path):
        notebook_dir = create_notebook(tmp_path, "unlocked")
        site_packages = notebook_dir / ".venv" / "lib" / "python3.14" / "site-packages"
        _lock(notebook_dir, "pyarrow")
        _install(site_packages, "rogue", "rogue_audit_pkg/__init__.py")
        source = f"import sys\nsys.path.insert(0, {str(site_packages)!r})\nimport rogue_audit_pkg\n"
        add_cell_to_notebook(notebook_dir, "c")
        write_cell(notebook_dir, "c", source)
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()

        result = await CellExecutor(session).execute_cell("c", source)

        assert result.success, result.error
        assert _codes(result.read_audit) == {"unlocked_package": ["rogue"]}

    async def test_a_loop_reports_every_iterations_reads(self, tmp_path, outside):
        (outside / "0.txt").write_text("0")
        (outside / "1.txt").write_text("1")
        loop = (
            "# @loop max_iter=2 carry=state\n"
            f"seen = open({str(outside)!r} + f'/{{state}}.txt').read()\n"
            "state = state + 1\n"
        )
        session = _session(tmp_path, ("s", "state = 0\n"), ("l", loop))
        executor = CellExecutor(session)
        await executor.execute_cell("s", "state = 0\n")

        result = await executor.execute_cell("l", loop)

        assert result.success, result.error
        assert _codes(result.read_audit) == {
            "undeclared_read": [str(outside / "0.txt"), str(outside / "1.txt")]
        }
        assert _stats(result.read_audit) == _unchanged(result.read_audit)

    @pytest.mark.integration
    @pytest.mark.warm_pool
    async def test_a_warm_run_reports_what_a_cold_one_does(self, tmp_path, outside, warm_pool):
        runs = _runs(outside)
        session = _session(tmp_path, *runs.items())

        executor = CellExecutor(session, await warm_pool(session))

        for cell_id, source in runs.items():
            await warm_again(executor.pool)
            result = await executor.execute_cell(cell_id, source)
            assert result.success, result.error
            assert result.execution_method == "warm"
            assert _codes(result.read_audit) == _expected(outside)[cell_id], cell_id
            assert _stats(result.read_audit) == _unchanged(result.read_audit), cell_id

    async def test_a_batch_reports_each_read_for_the_cell_that_made_it(self, tmp_path, outside):
        runs = _runs(outside)
        cells = {
            **runs,
            # Writes what the next one reads: an edge the DAG does not have.
            "writes": "from pathlib import Path\nPath('shared.txt').write_text('x')\n",
            "reads_work": "shared = open('shared.txt').read()\n",
            # Imports what "imports" loaded: no new module, the same undeclared input.
            "again": "import util_for_audit\nagain = util_for_audit.VALUE\n",
        }
        session = _session(tmp_path, *cells.items())
        specs = [
            {
                "cell_id": cell_id,
                "source": source,
                "consumed_vars": [],
                "env": {},
                "mount_manifest": {},
                "source_hash": "",
                "env_hash": "",
            }
            for cell_id, source in cells.items()
        ]

        result = await CellExecutor(session).execute_batch(specs)

        work = (session.path / "work").resolve()
        assert {r.cell_id: _codes(r.read_audit) for r in result.cell_results} == {
            **_expected(outside),
            "writes": {},
            "reads_work": {"work_dir_read": [str(work / "shared.txt")]},
            "again": {"undeclared_import": [str(outside / "util_for_audit.py")]},
        }
        for r in result.cell_results:
            assert _stats(r.read_audit) == _unchanged(r.read_audit), r.cell_id
