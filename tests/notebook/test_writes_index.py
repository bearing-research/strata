"""``.strata/writes/index.json``: the files each cell's last good run wrote through its
read-write mounts, for a host to list what a source was written by.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from strata.auth.principal import principal_context
from strata.notebook import writes_index
from strata.notebook.executor import CellExecutor
from strata.notebook.mounts import MountResolver
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from strata.types import Principal


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _two_files(out: Path) -> str:
    return (
        f"# @mount out {out.as_uri()} rw\n"
        '(out / "a.txt").write_text("aa")\n'
        '(out / "deep").mkdir(exist_ok=True)\n'
        '(out / "deep" / "b.txt").write_text("b")\n'
    )


def _session(tmp_path: Path, source: str) -> NotebookSession:
    nb = create_notebook(tmp_path, "Writes")
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", source)
    add_cell_to_notebook(nb, "c2", "c1")
    write_cell(nb, "c2", "y = 1")
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


async def _run(executor: CellExecutor, session: NotebookSession, source: str | None = None):
    cell = session.notebook_state.get_cell("c1")
    if source is not None and cell.source != source:
        cell.source = source
        session.re_analyze_cell("c1")
    return await executor.execute_cell("c1", cell.source)


class TestALocalRun:
    async def test_both_files_are_in_the_index_with_their_digests(self, tmp_path):
        out = tmp_path / "out"
        session = _session(tmp_path, _two_files(out))

        with principal_context(Principal(id="ada")):
            result = await _run(CellExecutor(session), session)

        assert result.success, result.error
        entry = writes_index.load(session.path)["c1"]
        assert isinstance(entry["ran_at"], int)
        assert entry["principal"] == "ada"
        assert entry["mounts"] == [
            {
                "mount": "out",
                "uri": out.as_uri(),
                "files": [
                    {"path": "a.txt", "size": 2, "sha256": _sha(b"aa")},
                    {"path": "deep/b.txt", "size": 1, "sha256": _sha(b"b")},
                ],
            }
        ]

    async def test_a_rerun_that_writes_one_replaces_the_entry(self, tmp_path):
        out = tmp_path / "out"
        session = _session(tmp_path, _two_files(out))
        executor = CellExecutor(session)
        assert (await _run(executor, session)).success

        one = f'# @mount out {out.as_uri()} rw\n(out / "c.txt").write_text("ccc")\n'
        result = await _run(executor, session, one)

        assert result.success, result.error
        entry = writes_index.load(session.path)["c1"]
        assert entry["principal"] is None
        assert entry["mounts"] == [
            {
                "mount": "out",
                "uri": out.as_uri(),
                "files": [{"path": "c.txt", "size": 3, "sha256": _sha(b"ccc")}],
            }
        ]

    async def test_a_failed_sync_back_leaves_the_entry_as_it_was(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        session = _session(tmp_path, _two_files(out))
        executor = CellExecutor(session)
        assert (await _run(executor, session)).success
        before = writes_index.load(session.path)

        async def fails(self, resolved):
            raise RuntimeError("bucket unreachable")

        monkeypatch.setattr(MountResolver, "sync_back", fails)
        one = f'# @mount out {out.as_uri()} rw\n(out / "c.txt").write_text("ccc")\n'
        result = await _run(executor, session, one)

        assert not result.success
        assert "bucket unreachable" in (result.error or "")
        assert writes_index.load(session.path) == before

    async def test_a_run_that_writes_nothing_drops_the_entry(self, tmp_path):
        out = tmp_path / "out"
        session = _session(tmp_path, _two_files(out))
        executor = CellExecutor(session)
        assert (await _run(executor, session)).success

        result = await _run(executor, session, f"# @mount out {out.as_uri()} rw\nx = 1\n")

        assert result.success, result.error
        assert "c1" not in writes_index.load(session.path)

    async def test_a_deleted_cell_drops_out(self, tmp_path):
        session = _session(tmp_path, _two_files(tmp_path / "out"))
        assert (await _run(CellExecutor(session), session)).success

        session.remove_cell("c1")

        assert writes_index.load(session.path) == {}


class TestAWorkerRun:
    """A worker syncs its own staging copy and reports what it wrote; Python and R agree."""

    @pytest.mark.parametrize("language", ["python", "r"])
    async def test_the_reported_files_are_in_the_index_until_it_reports_none(
        self, tmp_path, monkeypatch, language
    ):
        from strata.notebook.models import CellLanguage, WorkerBackendType, WorkerSpec

        body = "x = 1\n" if language == "python" else "x <- 1\n"
        source = f"# @mount out s3://bucket/out rw\n# @worker remote\n{body}"
        nb = create_notebook(tmp_path, "Worker writes", initialize_environment=False)
        add_cell_to_notebook(nb, "c1", None, language=language)
        write_cell(nb, "c1", source)
        state = parse_notebook(nb)
        state.cells[0].language = CellLanguage(language)
        state.workers = [
            WorkerSpec(
                name="remote",
                backend=WorkerBackendType.EXECUTOR,
                config={"url": "http://worker.invalid/v1/notebook-execute"},
            )
        ]
        session = NotebookSession(state, nb)
        files = [{"path": "a.txt", "size": 2, "sha256": _sha(b"aa")}]

        reports = [[{"mount": "out", "uri": "s3://bucket/out", "files": files}], []]

        async def worker(self, worker, source, input_specs, mount_specs, output_dir, *a, **k):
            result = {"success": True, "variables": {}, "mount_writes": reports.pop(0)}
            return result, output_dir, "executor", {}

        monkeypatch.setattr(CellExecutor, "_dispatch_http_executor", worker)
        executor = CellExecutor(session)

        first = await executor.execute_cell("c1", source)
        recorded = writes_index.load(session.path)
        second = await executor.execute_cell("c1", source)

        assert first.success, first.error
        assert recorded["c1"]["mounts"] == [
            {"mount": "out", "uri": "s3://bucket/out", "files": files}
        ]
        assert second.success, second.error
        assert writes_index.load(session.path) == {}


class TestTheIndex:
    def test_a_damaged_one_is_rewritten_by_the_next_run(self, tmp_path):
        path = writes_index.index_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        entry = {"ran_at": 1, "principal": None, "mounts": []}

        writes_index.record(tmp_path, "c1", entry, ["c1"])

        assert json.loads(path.read_text()) == {"c1": entry}

    def test_a_notebook_that_writes_nothing_has_no_index(self, tmp_path):
        writes_index.record(tmp_path, "c1", None, ["c1"])

        assert not writes_index.index_path(tmp_path).exists()

    def test_a_cell_gone_from_the_notebook_is_dropped_at_the_next_write(self, tmp_path):
        entry = {"ran_at": 1, "principal": None, "mounts": []}
        writes_index.record(tmp_path, "gone", entry, ["gone"])

        writes_index.record(tmp_path, "c1", entry, ["c1"])

        assert json.loads(writes_index.index_path(tmp_path).read_text()) == {"c1": entry}
