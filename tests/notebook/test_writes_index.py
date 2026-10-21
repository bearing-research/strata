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
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    set_variant_mode,
    write_cell,
)
from strata.types import Principal
from tests.notebook.test_table_index import _warehouse, windows


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

    async def test_losing_the_record_does_not_fail_the_run(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        session = _session(tmp_path, _two_files(out))

        def fails(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(writes_index, "record", fails)
        result = await _run(CellExecutor(session), session)

        assert result.success, result.error
        assert (out / "deep" / "b.txt").read_text() == "b"


class TestOtherGoodRuns:
    """A good run that wrote through no read-write mount drops the entry, however it ran."""

    async def test_a_cache_hit_drops_the_entry(self, tmp_path):
        leaf = 'print("hi")\n'
        session = _session(tmp_path, leaf)
        executor = CellExecutor(session)
        assert (await _run(executor, session)).success
        assert (await _run(executor, session, _two_files(tmp_path / "out"))).success
        assert "c1" in writes_index.load(session.path)

        hit = await _run(executor, session, leaf)

        assert hit.success, hit.error
        assert hit.cache_hit is True
        assert "c1" not in writes_index.load(session.path)

    async def test_a_batched_run_drops_the_entry(self, tmp_path):
        from tests.notebook.test_executor_batch import _cell_spec, _populate_consumed_vars

        session = _session(tmp_path, _two_files(tmp_path / "out"))
        executor = CellExecutor(session)
        assert (await _run(executor, session)).success
        cell = session.notebook_state.get_cell("c1")
        cell.source = "x = 2\n"
        session.re_analyze_cell("c1")

        batch = await executor.execute_batch(
            _populate_consumed_vars([_cell_spec("c1", cell.source)], session)
        )

        assert [r.status for r in batch.cell_results] == ["ok"]
        assert "c1" not in writes_index.load(session.path)


def _fanout_session(tmp_path: Path, body: str) -> NotebookSession:
    """A sweep over ``double`` and ``triple``, and ``ev`` run once per variant on ``preds``."""
    nb = create_notebook(tmp_path, "Fanout")
    cells = [
        ("load", "X = [1.0, 2.0, 3.0]\n"),
        ("vdouble", "# @variant model double\npreds = [v * 2 for v in X]\n"),
        ("vtriple", "# @variant model triple\npreds = [v * 3 for v in X]\n"),
        ("ev", body),
    ]
    after = None
    for cell_id, source in cells:
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    set_variant_mode(nb, "model", "sweep")
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


async def _run_ev(executor: CellExecutor, session: NotebookSession, body: str):
    cell = session.notebook_state.get_cell("ev")
    if cell.source != body:
        cell.source = body
        session.re_analyze_cell("ev")
    return await executor.execute_cell("ev", body)


class TestAFanOut:
    """A per_variant cell's entry is what all its variants wrote, once every one succeeded."""

    def _body(self, out: Path, script: str) -> str:
        return f"# @mount out {out.as_uri()} rw\n# @per_variant\nscore = sum(preds)\n{script}"

    async def test_every_variants_files_are_in_the_entry(self, tmp_path):
        out = tmp_path / "out"
        body = self._body(
            out,
            '(out / f"p{score}.txt").write_text("p")\n'
            '(out / "shared.txt").write_text(str(score))\n',
        )
        session = _fanout_session(tmp_path, body)

        result = await _run_ev(CellExecutor(session), session, body)

        assert result.success, result.error
        # Both variants wrote shared.txt: the later one's write is what is on disk.
        assert (out / "shared.txt").read_text() == "18.0"
        assert writes_index.load(session.path)["ev"]["mounts"] == [
            {
                "mount": "out",
                "uri": out.as_uri(),
                "files": [
                    {"path": "p12.0.txt", "size": 1, "sha256": _sha(b"p")},
                    {"path": "p18.0.txt", "size": 1, "sha256": _sha(b"p")},
                    {"path": "shared.txt", "size": 4, "sha256": _sha(b"18.0")},
                ],
            }
        ]

    async def test_a_variant_that_writes_nothing_keeps_the_others_files(self, tmp_path):
        out = tmp_path / "out"
        body = self._body(out, 'if score < 15:\n    (out / f"p{score}.txt").write_text("p")\n')
        session = _fanout_session(tmp_path, body)

        result = await _run_ev(CellExecutor(session), session, body)

        assert result.success, result.error
        (mount,) = writes_index.load(session.path)["ev"]["mounts"]
        assert [f["path"] for f in mount["files"]] == ["p12.0.txt"]

    async def test_a_failed_variant_leaves_the_entry_as_it_was(self, tmp_path):
        out = tmp_path / "out"
        good = self._body(out, '(out / f"p{score}.txt").write_text("p")\n')
        session = _fanout_session(tmp_path, good)
        executor = CellExecutor(session)
        assert (await _run_ev(executor, session, good)).success
        before = writes_index.load(session.path)

        failing = self._body(
            out,
            '(out / f"q{score}.txt").write_text("q")\nif score > 15:\n    raise ValueError("no")\n',
        )
        result = await _run_ev(executor, session, failing)

        assert not result.success
        assert (out / "q12.0.txt").exists()
        assert writes_index.load(session.path) == before

    @windows
    async def test_a_failed_fan_out_leaves_the_reads_entry_its_variants_ran_or_hit(self, tmp_path):
        from strata.notebook import reads_index

        _, uri = _warehouse(tmp_path)
        good = f"# @table events {uri}\n# @per_variant\nscore = sum(preds)\nprint(score)\n"
        session = _fanout_session(tmp_path, good)
        executor = CellExecutor(session)
        first = await _run_ev(executor, session, good)
        assert first.success, first.error
        before = reads_index.load(session.path)
        assert before["ev"]["inputs"]

        # The double variant runs, then is a cache hit; the triple one fails both times.
        failing = good + "if score > 15:\n    raise ValueError('no')\n"
        ran = await _run_ev(executor, session, failing)
        assert not ran.success
        assert reads_index.load(session.path) == before
        hit = await _run_ev(executor, session, failing)
        assert not hit.success
        assert reads_index.load(session.path) == before


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
