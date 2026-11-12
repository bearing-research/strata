"""A run's undeclared reads and mutations, as the cell shows them: each with its one fix."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.read_audit import ReadFinding, fingerprint
from strata.notebook.run_diagnostics import run_diagnostics
from strata.notebook.runtime_state import load_runtime_state
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _finding(code: str, *items: str, total: int | None = None, partial: bool = False):
    return ReadFinding(code=code, items=list(items), total=total or len(items), partial=partial)


def _messages(
    *findings: ReadFinding, notebook_dir: Path = Path("/nowhere/nb"), taken: tuple[str, ...] = ()
) -> list[str]:
    found = run_diagnostics(list(findings), [], notebook_dir=notebook_dir, taken=taken)
    return [d.message for d in found]


def _mounts(message: str) -> list[str]:
    return re.findall(r"`# @mount (\S+ \S+) ro`", message)


class TestTheMessage:
    def test_a_read_names_the_file_and_the_mount_that_covers_it(self):
        (message,) = _messages(_finding("undeclared_read", "/data/events.csv"))

        assert message == (
            "This cell read `/data/events.csv`, which no mount covers, so its cached result "
            "will not change when the file does. Add `# @mount data /data ro` and read "
            '`data / "events.csv"`.'
        )

    def test_a_folder_beside_the_notebook_is_mounted_by_a_relative_path(self, tmp_path):
        notebook_dir = tmp_path / "nb"
        (message,) = _messages(
            _finding("undeclared_read", str(tmp_path / "shared" / "a.csv")),
            notebook_dir=notebook_dir,
        )

        assert _mounts(message) == ["shared ../shared"]

    def test_a_folder_in_the_notebook_is_a_path_uv_reads_as_one(self, tmp_path):
        notebook_dir = tmp_path / "nb"
        (read,) = _messages(
            _finding("undeclared_read", str(notebook_dir / "data" / "a.csv")),
            notebook_dir=notebook_dir,
        )
        (imported,) = _messages(
            _finding("undeclared_import", str(notebook_dir / "helpers" / "hx.py")),
            notebook_dir=notebook_dir,
        )

        assert _mounts(read) == ["data ./data"]
        assert "`uv add --editable ./helpers`" in imported

    def test_files_share_the_mount_of_their_nearest_common_folder(self):
        (message,) = _messages(
            _finding("undeclared_read", "/data/2024/a.csv", "/data/2025/b.csv", "/other/c.txt")
        )

        assert _mounts(message) == ["data /data", "other /other"]
        assert "read them through those mounts" in message

    def test_the_home_folder_is_never_a_mount(self):
        home = os.path.expanduser("~")
        (message,) = _messages(
            _finding("undeclared_read", f"{home}/notes.txt", f"{home}/papers/a.pdf")
        )

        assert _mounts(message) == [f"papers {home}/papers"]
        assert f"Move the ones directly in `{home}` into a folder, and mount that." in message

    def test_past_five_files_it_counts_the_rest(self):
        files = [f"/data/{n}.csv" for n in range(8)]
        (message,) = _messages(_finding("undeclared_read", *files, total=300, partial=True))

        assert "300 files" in message
        assert "`/data/4.csv` and 295 more" in message
        assert "`/data/5.csv`" not in message
        assert message.endswith(
            "The audit did not record all of this run's reads, so there may be more."
        )

    def test_a_failed_audit_says_only_that(self):
        (message,) = _messages(_finding("audit_failed", "KeyError: 'x'", partial=True))

        assert (
            message
            == "The read audit failed (`KeyError: 'x'`), so this run's reads were not checked."
        )

    def test_a_folder_holding_the_notebook_is_never_a_mount(self, tmp_path):
        # A mount over the notebook keys the cell on every cell's source and notebook.toml.
        notebook_dir = tmp_path / "nb"
        (beside,) = _messages(
            _finding("undeclared_read", str(notebook_dir / "a.csv")), notebook_dir=notebook_dir
        )
        (above,) = _messages(
            _finding(
                "undeclared_read", str(tmp_path / "shared" / "a.csv"), str(tmp_path / "b.yaml")
            ),
            notebook_dir=notebook_dir,
        )

        assert _mounts(beside) == []
        assert "Move it into a folder of its own and mount that." in beside
        assert _mounts(above) == ["shared ../shared"]
        assert f"Move the ones directly in `{tmp_path}` into a folder, and mount that." in above

    def test_a_path_with_a_space_is_mounted_as_a_uri(self, tmp_path):
        notebook_dir = tmp_path / "nb"
        (beside,) = _messages(
            _finding("undeclared_read", str(tmp_path / "My Data" / "a.csv")),
            notebook_dir=notebook_dir,
        )
        (away,) = _messages(_finding("undeclared_read", "/x/My Data/a.csv"))

        assert _mounts(beside) == ["my_data file:../My%20Data"]
        assert _mounts(away) == ["my_data file:///x/My%20Data"]

    def test_a_name_the_cell_uses_is_not_suggested(self):
        (read, listed) = _messages(
            _finding("undeclared_read", "/data/a.csv"),
            _finding("undeclared_listing", "/data"),
            taken=("data",),
        )

        # Not the cell's own name, and the same one for the same folder.
        assert _mounts(read) == ["data_2 /data"]
        assert _mounts(listed) == ["data_2 /data"]

    def test_listing_the_home_folder_names_a_folder_of_its_own(self):
        home = os.path.expanduser("~")
        (message,) = _messages(_finding("undeclared_listing", home))

        assert message == (
            f"This cell listed `{home}`, which no mount covers, so its cached result will not "
            f"change when a file is added or removed. In place of `{home}`, list a folder of "
            "its own, and mount that."
        )

    def test_an_import_names_its_import_root(self, tmp_path):
        notebook_dir = tmp_path / "nb"
        package = tmp_path / "helpers" / "mypkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (in_package,) = _messages(
            _finding("undeclared_import", str(package / "core.py")), notebook_dir=notebook_dir
        )
        (in_notebook,) = _messages(
            _finding("undeclared_import", str(notebook_dir / "hx.py")), notebook_dir=notebook_dir
        )

        assert "`uv add --editable ../helpers`" in in_package
        assert "--editable ." not in in_notebook
        assert f"Move the ones directly in `{notebook_dir}` into a folder" in in_notebook

    def test_each_kind_names_its_own_fix(self, tmp_path):
        found = run_diagnostics(
            [
                _finding("undeclared_listing", "/data/raw"),
                _finding("undeclared_import", "/opt/lib/helpers.py"),
                _finding("unlocked_package", "requests"),
                _finding("work_dir_read", str(tmp_path / "work" / "features.csv")),
                _finding("subprocess_run", "curl"),
            ],
            [],
            notebook_dir=tmp_path,
        )
        by_code = {d.code: d for d in found}

        assert (
            "Add `# @mount raw /data/raw ro` and list `raw`."
            in by_code["undeclared_listing"].message
        )
        assert "`uv add --editable /opt/lib`" in by_code["undeclared_import"].message
        assert "`uv add requests`" in by_code["unlocked_package"].message
        assert "`work/features.csv`" in by_code["work_dir_read"].message
        assert "Return the data from the cell that writes it" in by_code["work_dir_read"].message
        assert by_code["subprocess_run"].severity == "info"
        assert {d.severity for d in found if d.code != "subprocess_run"} == {"warn"}

    def test_a_mutation_warning_is_one_too(self):
        (diagnostic,) = run_diagnostics(
            [],
            [{"var_name": "df", "message": "Cell mutated df", "suggestion": "Return a copy"}],
            notebook_dir=Path("/nb"),
        )

        assert (diagnostic.code, diagnostic.message) == (
            "mutated_input",
            "Cell mutated df. Return a copy",
        )

    def test_a_mutation_every_iteration_makes_is_one(self):
        warning = {"var_name": "df", "message": "Cell mutated df", "suggestion": "Return a copy"}

        found = run_diagnostics([], [warning, dict(warning)], notebook_dir=Path("/nb"))

        assert _codes(found) == ["mutated_input"]


def _session(notebook_dir: Path) -> NotebookSession:
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


@pytest.fixture
def outside(tmp_path) -> Path:
    folder = tmp_path / "outside"
    folder.mkdir()
    (folder / "events.csv").write_text("a,b\n1,2\n")
    return folder.resolve()


def _notebook(tmp_path: Path, *cells: tuple[str, str]) -> Path:
    notebook_dir = create_notebook(tmp_path / "project", "diagnostics")
    after = None
    for cell_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after)
        write_cell(notebook_dir, cell_id, source)
        after = cell_id
    return notebook_dir


def _reads(outside: Path) -> str:
    return f"text = open({str(outside / 'events.csv')!r}).read()\n"


def _codes(diagnostics: list[Any]) -> list[str]:
    return [d["code"] if isinstance(d, dict) else d.code for d in diagnostics]


class TestARun:
    async def test_an_undeclared_read_shows_on_the_cell_and_survives_a_reopen(
        self, tmp_path, outside
    ):
        notebook_dir = _notebook(tmp_path, ("c", _reads(outside)))
        session = _session(notebook_dir)

        result = await CellExecutor(session).execute_cell("c", _reads(outside))

        assert result.success, result.error
        (shown,) = session.notebook_state.get_cell("c").run_diagnostics
        assert shown.code == "undeclared_read"
        assert str(outside / "events.csv") in shown.message
        assert result.run_diagnostics == [shown.model_dump(mode="json")]
        cell = session.notebook_state.get_cell("c")
        events = str(outside / "events.csv")
        assert cell.read_fingerprints == {events: fingerprint(events)}
        # Server-side only: not in what a client gets.
        assert "read_fingerprints" not in cell.serialize()
        reopened = _session(notebook_dir).notebook_state.get_cell("c")
        assert reopened.run_diagnostics == [shown]
        assert reopened.read_fingerprints == cell.read_fingerprints

    async def test_a_rerun_records_the_file_as_it_read_it_again(self, tmp_path, outside):
        notebook_dir = _notebook(tmp_path, ("c", _reads(outside)))
        session = _session(notebook_dir)
        executor = CellExecutor(session)
        first = await executor.execute_cell("c", _reads(outside))
        events = str(outside / "events.csv")
        (outside / "events.csv").write_text("a,b\n1,2\n3,4\n")

        again = await executor.execute_cell_rerun("c", _reads(outside))

        # The same message; the file it read is not the same.
        assert again.run_diagnostics == first.run_diagnostics
        assert session.notebook_state.get_cell("c").read_fingerprints == {
            events: fingerprint(events)
        }
        reopened = _session(notebook_dir).notebook_state.get_cell("c")
        assert reopened.read_fingerprints == {events: fingerprint(events)}

    async def test_a_cache_hit_says_it_again(self, tmp_path, outside):
        notebook_dir = _notebook(tmp_path, ("c", _reads(outside)))
        first = await CellExecutor(_session(notebook_dir)).execute_cell("c", _reads(outside))
        # A fresh session: what the hit shows comes from the run's record, not from memory.
        session = _session(notebook_dir)
        cell = session.notebook_state.get_cell("c")
        recorded = cell.read_fingerprints
        cell.run_diagnostics = []
        cell.read_fingerprints = {}

        hit = await CellExecutor(session).execute_cell("c", _reads(outside))

        assert hit.cache_hit
        assert _codes(hit.run_diagnostics) == ["undeclared_read"]
        assert hit.run_diagnostics == first.run_diagnostics
        assert recorded and cell.read_fingerprints == recorded

    @pytest.mark.parametrize("folder", ["outside", "my data"])
    async def test_the_fix_it_names_clears_it(self, tmp_path, folder):
        outside = tmp_path / folder
        outside.mkdir()
        (outside / "events.csv").write_text("a,b\n1,2\n")
        notebook_dir = _notebook(tmp_path, ("c", _reads(outside)))
        session = _session(notebook_dir)
        first = await CellExecutor(session).execute_cell("c", _reads(outside))
        (mount,) = _mounts(first.run_diagnostics[0]["message"])
        name, path = mount.split()
        fixed = f'# @mount {name} {path} ro\ntext = ({name} / "events.csv").read_text()\n'
        write_cell(notebook_dir, "c", fixed)
        session = _session(notebook_dir)

        result = await CellExecutor(session).execute_cell("c", fixed)

        assert result.success, result.error
        assert result.run_diagnostics == []
        assert session.notebook_state.get_cell("c").run_diagnostics == []
        assert session.notebook_state.get_cell("c").read_fingerprints == {}
        assert load_runtime_state(notebook_dir).cells["c"].run_diagnostics == []
        assert load_runtime_state(notebook_dir).cells["c"].read_fingerprints == {}

    async def test_its_fix_does_not_reuse_a_name_the_cell_defines(self, tmp_path, outside):
        source = f"outside = 1\n{_reads(outside)}"
        notebook_dir = _notebook(tmp_path, ("c", source))

        result = await CellExecutor(_session(notebook_dir)).execute_cell("c", source)

        assert _mounts(result.run_diagnostics[0]["message"]) == [f"outside_2 {outside}"]

    async def test_a_failure_replaces_them_only_with_its_own(self, tmp_path, outside):
        notebook_dir = _notebook(tmp_path, ("c", _reads(outside)))
        session = _session(notebook_dir)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("c", _reads(outside))).success
        before = session.notebook_state.get_cell("c").run_diagnostics

        # Its result still reads the file: they stand.
        failed = await executor.execute_cell("c", "raise ValueError('no')\n")
        assert not failed.success
        assert session.notebook_state.get_cell("c").run_diagnostics == before
        assert load_runtime_state(notebook_dir).cells["c"].run_diagnostics != []

        reads_then_fails = f"{_reads(outside)}import json\njson.loads(text)\n"
        failed = await executor.execute_cell("c", reads_then_fails)
        assert not failed.success
        assert _codes(failed.run_diagnostics) == ["undeclared_read"]

    async def test_a_run_all_shows_each_cells_on_its_frame(self, tmp_path, outside, monkeypatch):
        from strata.notebook import ws

        notebook_dir = _notebook(tmp_path, ("r", _reads(outside)), ("q", "y = 1\n"))
        session = _session(notebook_dir)
        sent: list[dict[str, Any]] = []

        async def record(notebook_id: str, message: dict[str, Any], **kwargs: Any) -> None:
            sent.append(message)

        class _Manager:
            def get_session(self, notebook_id: str) -> NotebookSession:
                return session

        monkeypatch.setattr(ws, "_broadcast_message", record)
        monkeypatch.setattr(ws, "_get_session_manager", lambda: _Manager())
        await ws._execute_run_all(
            None, session, ["r", "q"], ws.NotebookExecutionState(), session.id
        )

        frames = {m["payload"]["cell_id"]: m["payload"] for m in sent if m["type"] == "cell_output"}
        assert _codes(frames["r"]["run_diagnostics"]) == ["undeclared_read"]
        assert frames["q"]["run_diagnostics"] == []
        # Again: a hit inside the batch says it too.
        sent.clear()
        session.notebook_state.get_cell("r").run_diagnostics = []
        await ws._execute_run_all(
            None, session, ["r", "q"], ws.NotebookExecutionState(), session.id
        )
        frames = {m["payload"]["cell_id"]: m["payload"] for m in sent if m["type"] == "cell_output"}
        assert frames["r"]["cache_hit"]
        assert _codes(frames["r"]["run_diagnostics"]) == ["undeclared_read"]
        # The batch stored its record too: a later single-cell hit says it.
        hit = await CellExecutor(session).execute_cell("r", _reads(outside))
        assert hit.cache_hit
        assert _codes(hit.run_diagnostics) == ["undeclared_read"]

    async def test_a_run_all_store_failure_still_shows_them(self, tmp_path, outside, monkeypatch):
        from strata.notebook import ws

        # A consumed top-level lambda is not stored: the batch ends on persist_failed.
        source = f"{_reads(outside)}add = lambda x: x + 1\n"
        notebook_dir = _notebook(tmp_path, ("r", source), ("q", "y = add(1)\n"))
        session = _session(notebook_dir)
        sent: list[dict[str, Any]] = []

        async def record(notebook_id: str, message: dict[str, Any], **kwargs: Any) -> None:
            sent.append(message)

        class _Manager:
            def get_session(self, notebook_id: str) -> NotebookSession:
                return session

        monkeypatch.setattr(ws, "_broadcast_message", record)
        monkeypatch.setattr(ws, "_get_session_manager", lambda: _Manager())
        await ws._execute_run_all(
            None, session, ["r", "q"], ws.NotebookExecutionState(), session.id
        )

        (frame,) = [m["payload"] for m in sent if m["type"] == "cell_error"]
        assert frame["cell_id"] == "r"
        assert _codes(frame["run_diagnostics"]) == ["undeclared_read"]

    async def test_an_each_cell_shows_what_every_instance_read(
        self, tmp_path, outside, monkeypatch
    ):
        (outside / "more.csv").write_text("c\n3\n")
        each = f"# @each name in names\nout = open({str(outside)!r} + '/' + name).read()\n"
        notebook_dir = _notebook(
            tmp_path,
            ("a", 'names = ["events.csv", "more.csv"]\n'),
            ("b", each),
            ("c", "n = len(out)\n"),
        )
        session = _session(notebook_dir)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("a", 'names = ["events.csv", "more.csv"]\n')).success
        from strata.notebook import session as session_module

        writes: list[str] = []
        persist = session_module.persist_cell_run_diagnostics

        def counted(notebook_dir: Path, cell_id: str, *args: Any) -> None:
            writes.append(cell_id)
            persist(notebook_dir, cell_id, *args)

        monkeypatch.setattr(session_module, "persist_cell_run_diagnostics", counted)

        result = await executor.execute_cell("b", each)

        assert result.success, result.error
        # The cell's, once; not each instance's on the way.
        assert writes == ["b"]
        (shown,) = session.notebook_state.get_cell("b").run_diagnostics
        assert shown.code == "undeclared_read"
        assert "2 files" in shown.message
        hit = await CellExecutor(_session(notebook_dir)).execute_cell("b", each)
        assert hit.cache_hit
        assert hit.run_diagnostics == result.run_diagnostics

    async def test_a_loop_cells_hit_says_what_its_iterations_read(self, tmp_path, outside):
        (outside / "0.txt").write_text("0")
        (outside / "1.txt").write_text("1")
        loop = (
            "# @loop max_iter=2 carry=state\n"
            f"seen = open({str(outside)!r} + f'/{{state}}.txt').read()\n"
            "state = state + 1\n"
        )
        notebook_dir = _notebook(tmp_path, ("s", "state = 0\n"), ("l", loop))
        executor = CellExecutor(_session(notebook_dir))
        assert (await executor.execute_cell("s", "state = 0\n")).success
        result = await executor.execute_cell("l", loop)
        assert result.success, result.error
        assert "2 files" in result.run_diagnostics[0]["message"]

        hit = await CellExecutor(_session(notebook_dir)).execute_cell("l", loop)

        assert hit.cache_hit
        assert hit.run_diagnostics == result.run_diagnostics
