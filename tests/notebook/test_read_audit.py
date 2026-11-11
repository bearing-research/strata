"""A run reports what it read that its cell does not declare, and the declared reads not."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from strata.notebook import read_audit
from strata.notebook.env import audit_scope
from strata.notebook.executor import CellExecutor
from strata.notebook.models import MountSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.read_audit import ReadFinding, merge, recording, report
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
    }


def _expected(outside: Path) -> dict[str, dict[str, list[str]]]:
    return {
        "reads": {"undeclared_read": [str(outside / "events.csv")]},
        "imports": {"undeclared_import": [str(outside / "util_for_audit.py")]},
        "own": {},
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
