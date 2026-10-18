"""A cell runs in its notebook's ``work/``, not beside ``notebook.toml``.

Every harness starts in the notebook's directory and moves into ``work/`` before the cell's
code runs, so a file a cell writes lands there and a later cell reads it back. A path that
is beside the notebook instead gets a hint naming ``../<path>``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from strata.notebook import harness_user
from strata.notebook.executor import CellExecutor
from strata.notebook.harness_user import HarnessUser, UnsafeRunFile, prepare_work_dir
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

WRITE = "from pathlib import Path\nPath('fig.txt').write_text('drawn')\n"
READ = (
    "import os, sys\n"
    "from pathlib import Path\n"
    "here = os.getcwd()\n"
    "back = Path('fig.txt').read_text()\n"
    "on_path = os.getcwd() in sys.path or '' in sys.path\n"
)
MISSING = "open('data.csv').read()\n"


def _session(tmp_path: Path) -> NotebookSession:
    nb = create_notebook(tmp_path, "Work")
    add_cell_to_notebook(nb, "w", None)
    write_cell(nb, "w", WRITE)
    add_cell_to_notebook(nb, "r", "w")
    write_cell(nb, "r", READ)
    add_cell_to_notebook(nb, "m", "r")
    write_cell(nb, "m", MISSING)
    (nb / "data.csv").write_text("a,b\n")
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


def _assert_ran_in_work(session: NotebookSession, read) -> None:
    work = session.path / "work"
    assert read.success, read.error
    assert read.outputs["here"]["preview"] == str(work)
    assert read.outputs["back"]["preview"] == "drawn"
    assert read.outputs["on_path"]["preview"] is False
    assert (work / "fig.txt").read_text() == "drawn"
    assert not (session.path / "fig.txt").exists()


class TestEveryHarnessRunsInWork:
    async def test_a_cold_run(self, tmp_path):
        session = _session(tmp_path)
        executor = CellExecutor(session)

        assert (await executor.execute_cell("w", WRITE)).success
        read = await executor.execute_cell("r", READ)

        assert read.execution_method == "cold"
        _assert_ran_in_work(session, read)

    @pytest.mark.integration
    @pytest.mark.warm_pool
    async def test_a_warm_run(self, tmp_path, warm_pool):
        session = _session(tmp_path)
        assert (await CellExecutor(session).execute_cell("w", WRITE)).success
        executor = CellExecutor(session, await warm_pool(session))

        read = await executor.execute_cell("r", READ)

        assert read.execution_method == "warm"
        _assert_ran_in_work(session, read)

    async def test_a_run_all_batch(self, tmp_path):
        session = _session(tmp_path)
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
            for cell_id, source in (
                ("w", WRITE),
                ("r", READ + "assert here.endswith('work') and back == 'drawn' and not on_path\n"),
            )
        ]

        result = await CellExecutor(session).execute_batch(specs)

        assert result.completed, result.cell_results
        assert (session.path / "work" / "fig.txt").read_text() == "drawn"
        assert not (session.path / "fig.txt").exists()

    async def test_the_inspect_repl(self, tmp_path):
        from strata.notebook.inspect_repl import InspectSession

        session = _session(tmp_path)
        inspect = InspectSession("w")
        try:
            assert await inspect.start(session) == "ready"
            answer = await inspect.evaluate("__import__('os').getcwd()")
        finally:
            await inspect.close()

        assert str(session.path / "work") in str(answer.get("result"))

    async def test_a_worker_run_has_a_work_of_its_own(self, tmp_path, notebook_executor_server):
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        source = "# @worker remote\n" + WRITE + "import os\nhere = os.getcwd()\n"
        session = _session(tmp_path)
        session.notebook_state.workers = [
            WorkerSpec(
                name="remote",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="r",
                config={"url": notebook_executor_server["execute_url"]},
            )
        ]

        result = await CellExecutor(session).execute_cell("w", source)

        assert result.success, result.error
        assert result.execution_method == "executor"
        here = Path(result.outputs["here"]["preview"])
        assert here.name == "work"
        assert here != Path.cwd() and here != session.path / "work"
        assert not (Path.cwd() / "fig.txt").exists()


class TestAPathBesideTheNotebook:
    async def test_a_cold_run_says_read_it_as_dot_dot(self, tmp_path):
        session = _session(tmp_path)

        result = await CellExecutor(session).execute_cell("m", MISSING)

        assert not result.success
        assert "read it as ../data.csv" in result.error

    @pytest.mark.integration
    @pytest.mark.warm_pool
    async def test_a_warm_run_says_so_too(self, tmp_path, warm_pool):
        session = _session(tmp_path)
        executor = CellExecutor(session, await warm_pool(session))

        result = await executor.execute_cell("m", MISSING)

        assert result.execution_method == "warm"
        assert "read it as ../data.csv" in result.error

    async def test_a_batch_cell_says_so_too(self, tmp_path):
        session = _session(tmp_path)
        spec = {
            "cell_id": "m",
            "source": MISSING,
            "consumed_vars": [],
            "env": {},
            "mount_manifest": {},
            "source_hash": "",
            "env_hash": "",
        }

        result = await CellExecutor(session).execute_batch([spec])

        (failed,) = result.cell_results
        assert "read it as ../data.csv" in (failed.error or "")

    async def test_a_path_missing_everywhere_gets_no_hint(self, tmp_path):
        session = _session(tmp_path)
        source = "open('nowhere.csv').read()\n"

        result = await CellExecutor(session).execute_cell("m", source)

        assert not result.success
        assert "../" not in result.error


class TestPreparingWork:
    def test_a_link_is_refused(self, tmp_path):
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "work").symlink_to(tmp_path / "elsewhere")

        with pytest.raises(UnsafeRunFile):
            prepare_work_dir(tmp_path, None)

    def test_only_the_directory_changes_owner(self, tmp_path, monkeypatch):
        """One left by an unisolated run keeps its files' owners: no recursive chown."""
        (tmp_path / "work" / "deep").mkdir(parents=True)
        (tmp_path / "work" / "deep" / "old.txt").write_text("x")
        chowned: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            harness_user.os,
            "chown",
            lambda path, uid, gid, follow_symlinks=True: chowned.append(
                (os.fspath(path), follow_symlinks)
            ),
        )
        user = HarnessUser(name="cells", uid=os.geteuid() + 1, gid=os.getegid() + 1, home="/")

        work = prepare_work_dir(tmp_path, user)

        assert work == tmp_path / "work"
        assert chowned == [(str(tmp_path / "work"), False)]
