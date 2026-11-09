"""A local mount, and each ``@each`` item over one, is keyed by its files' content.

Stat only decides whether a file is read again: a touch costs a read, never a rerun, and
a rewrite that keeps the size and the mtime is still seen. These run real cells.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from vestigo.digests import FileDigests

from strata.notebook import mounts
from strata.notebook.models import CellStatus, MountSpec
from strata.notebook.mounts import (
    STALENESS_READ_BYTES,
    MountResolver,
    hash_local_mount,
    local_digests,
    mount_fingerprint_sync,
    on_loop,
    weigh_mount_sync,
)
from tests.notebook.test_each import READER, Notebook, _logs, _over_files, _touch


@pytest.fixture
def later(monkeypatch):
    """Every digest read 10 s after now: a file just written is past the 2 s rule, so its
    digest is remembered and the next look is the cache's."""
    monkeypatch.setattr(
        mounts, "FileDigests", lambda path: FileDigests(path, clock=lambda: time.time_ns() + 10**10)
    )


def _over_mount(tmp_path: Path) -> tuple[Path, str]:
    raw = tmp_path / "raw"
    raw.mkdir()
    return raw, (
        f"# @mount raw {raw.as_uri()} ro\n"
        + _logs(tmp_path / "ran.log", '"ran"')
        + 'out = (raw / "f.txt").read_text()\n'
    )


class TestALocalMount:
    async def test_a_rewrite_that_keeps_the_size_and_the_mtime_reruns(self, tmp_path, later):
        raw, source = _over_mount(tmp_path)
        _touch(raw / "f.txt", "aaaa")
        nb = Notebook(tmp_path, source, READER.replace("dict(out)", "out"))
        await nb.good("a")
        await nb.good("b")
        assert nb.ran() == ["ran"]
        assert nb.session.compute_staleness()["a"].status == CellStatus.READY

        _touch(raw / "f.txt", "bbbb")  # the same size, the mtime set back

        assert nb.session.compute_staleness()["a"].status != CellStatus.READY
        result = await nb.good("a")
        assert not result.cache_hit
        assert nb.ran() == ["ran"]
        assert "bbbb" in (await nb.good("b", "print(out)\n")).stdout

    async def test_a_touch_or_a_chmod_reruns_nothing(self, tmp_path):
        raw, source = _over_mount(tmp_path)
        _touch(raw / "f.txt", "aaaa")
        nb = Notebook(tmp_path, source, READER.replace("dict(out)", "out"))
        await nb.good("a")
        await nb.good("b")
        nb.ran()

        _touch(raw / "f.txt", "aaaa", tick=5)
        os.chmod(raw / "f.txt", 0o600)

        assert nb.session.compute_staleness()["a"].status == CellStatus.READY
        assert (await nb.good("a")).cache_hit
        assert nb.ran() == []

    async def test_staleness_reads_no_more_than_its_share_and_a_run_reads_the_rest(
        self, tmp_path, monkeypatch
    ):
        raw, source = _over_mount(tmp_path)
        _touch(raw / "f.txt", "aaaa")
        nb = Notebook(tmp_path, source, READER.replace("dict(out)", "out"))
        await nb.good("a")
        await nb.good("b")
        nb.ran()
        # More unread bytes than staleness may read: it reads none of them.
        monkeypatch.setattr(mounts, "STALENESS_READ_BYTES", 3)
        _touch(raw / "f.txt", "aaaa", tick=5)
        read = []
        digest_many = FileDigests.digest_many

        def counted(self, paths, **kwargs):
            paths = list(paths)
            read.extend(paths)
            return digest_many(self, paths, **kwargs)

        monkeypatch.setattr(FileDigests, "digest_many", counted)

        assert nb.session.compute_staleness()["a"].status != CellStatus.READY
        assert read == []
        # A run reads it, and keys the cell as before.
        assert (await nb.good("a")).cache_hit
        assert read

    async def test_a_copy_under_another_uri_differs_only_by_its_name(self, tmp_path):
        first = tmp_path / "first"
        first.mkdir()
        _touch(first / "f.txt", "aaaa")
        second = tmp_path / "second"
        second.mkdir()
        _touch(second / "f.txt", "aaaa", tick=3)
        resolver = MountResolver(cache_dir=tmp_path / "nb" / "mount_cache")

        keys = [
            mount_fingerprint_sync(resolver, MountSpec(name="raw", uri=d.as_uri()))
            for d in (first, second)
        ]

        assert keys[0] != keys[1]
        assert keys[0].rsplit(":", 1)[1] == keys[1].rsplit(":", 1)[1]

    def test_weighing_gives_the_key_staleness_computes(self, tmp_path):
        (tmp_path / "raw").mkdir()
        _touch(tmp_path / "raw" / "f.txt", "aaaa")
        resolver = MountResolver(cache_dir=tmp_path / "nb" / "mount_cache")
        mount = MountSpec(name="raw", uri=(tmp_path / "raw").as_uri())

        ran = weigh_mount_sync(resolver, mount, None)[0]

        assert ran == mount_fingerprint_sync(resolver, mount, read_bytes=STALENESS_READ_BYTES)

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0, reason="needs a file its owner cannot read"
    )
    def test_a_file_that_cannot_be_read_is_keyed_as_such_and_the_rest_by_content(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        _touch(raw / "a.txt", "aaaa")
        _touch(raw / "b.txt", "bbbb")
        (raw / "b.txt").chmod(0)
        try:
            digests = local_digests(raw, tmp_path / "hashes.sqlite")
        finally:
            (raw / "b.txt").chmod(0o600)

        assert digests is not None
        assert digests["b.txt"] == "unreadable"
        assert digests["a.txt"] == local_digests(raw, tmp_path / "hashes.sqlite")["a.txt"]


class TestEachItems:
    async def test_a_touch_reruns_no_item_and_a_rewrite_only_its_own(self, tmp_path, later):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "aaaa")
        _touch(raw / "two.txt", "bbbb")
        nb = Notebook(tmp_path, source, READER + "print(sorted(seen.items()))\n")
        await nb.good("a")
        nb.ran()

        _touch(raw / "one.txt", "aaaa", tick=5)
        assert (await nb.good("a")).cache_hit
        assert nb.ran() == []

        _touch(raw / "two.txt", "cccc")  # the same size, the mtime set back
        await nb.good("a")
        assert nb.ran() == ["two.txt"]
        assert "('two.txt', 'CCCC')" in (await nb.good("b")).stdout


class TestReviewFixes:
    """What a first review of the content key found."""

    @staticmethod
    def _counted(monkeypatch) -> list:
        """The files read from now on."""
        from vestigo import digests

        read: list = []
        read_digest = digests._read_digest

        def counted(path):
            read.append(Path(path))
            return read_digest(path)

        monkeypatch.setattr(digests, "_read_digest", counted)
        return read

    def test_a_mount_not_read_yet_shows_as_checked_not_unavailable(self, tmp_path):
        from strata.notebook import nonce

        (tmp_path / "raw").mkdir()
        _touch(tmp_path / "raw" / "f.txt", "aaaa")
        resolver = MountResolver(cache_dir=tmp_path / "nb" / "mount_cache")
        mount = MountSpec(name="raw", uri=(tmp_path / "raw").as_uri())

        component = mount_fingerprint_sync(resolver, mount, read_bytes=0)

        assert component is not None
        assert not nonce.is_unresolved(component)
        assert nonce.is_unhashed(component)
        assert component != mount_fingerprint_sync(resolver, mount, read_bytes=0)

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0, reason="needs a file its owner cannot read"
    )
    def test_a_file_that_cannot_be_read_does_not_count_against_staleness(self, tmp_path, later):
        raw = tmp_path / "raw"
        raw.mkdir()
        _touch(raw / "a.txt", "aaaa")
        _touch(raw / "big.bin", "b" * 100)
        (raw / "big.bin").chmod(0)
        try:
            ran = local_digests(raw, tmp_path / "hashes.sqlite")
            looked = local_digests(raw, tmp_path / "hashes.sqlite", read_bytes=10)
        finally:
            (raw / "big.bin").chmod(0o600)

        assert looked == ran
        assert ran["big.bin"] == "unreadable"

    def test_a_cache_that_cannot_be_opened_still_remembers(self, tmp_path, monkeypatch, later):
        raw = tmp_path / "raw"
        raw.mkdir()
        _touch(raw / "a.txt", "aaaa")
        unopenable = tmp_path / "not-a-folder"
        unopenable.write_text("")
        read = self._counted(monkeypatch)

        first = local_digests(raw, unopenable / "hashes.sqlite")
        assert read
        read.clear()
        assert local_digests(raw, unopenable / "hashes.sqlite", read_bytes=0) == first
        assert read == []

    def test_a_warm_mount_reads_and_logs_nothing(self, tmp_path, monkeypatch, caplog, later):
        raw = tmp_path / "raw"
        raw.mkdir()
        _touch(raw / "a.txt", "aaaa")
        monkeypatch.setattr(mounts, "_LOG_READ_BYTES", 1)
        local_digests(raw, tmp_path / "hashes.sqlite")
        read = self._counted(monkeypatch)
        caplog.clear()

        with caplog.at_level("INFO", logger="strata.notebook.mounts"):
            local_digests(raw, tmp_path / "hashes.sqlite")

        assert read == []
        assert "Hashing" not in caplog.text

    def test_a_file_that_changed_while_read_gets_a_key_no_run_reproduces(
        self, tmp_path, monkeypatch
    ):
        from vestigo.digests import Digest

        raw = tmp_path / "raw"
        raw.mkdir()
        _touch(raw / "a.txt", "aaaa")
        monkeypatch.setattr(
            FileDigests,
            "digest_many",
            lambda self, paths, **_: [Digest("torn", settled=False) for _ in paths],
        )

        first = local_digests(raw, tmp_path / "hashes.sqlite")
        second = local_digests(raw, tmp_path / "hashes.sqlite")

        assert first["a.txt"] != "torn"
        assert first != second

    def test_a_pinned_mounts_items_are_read_once(self, tmp_path, monkeypatch, later):
        from strata.notebook import each

        raw = tmp_path / "raw"
        raw.mkdir()
        _touch(raw / "a.txt", "aaaa")
        first = each.items_of_mount(raw, None, digests_path=tmp_path / "hashes.sqlite")
        read = self._counted(monkeypatch)
        monkeypatch.setattr(each, "_sha256", lambda f: read.append(f.name) or "uncached")

        again = each.items_of_mount(raw, None, digests_path=tmp_path / "hashes.sqlite")

        assert again == first
        assert read == []


class TestSecondReview:
    """What a second review of the content key found."""

    async def test_weighing_a_cell_before_it_runs_reads_no_cold_mount(self, tmp_path, monkeypatch):
        from strata.notebook import executor as executor_module
        from strata.notebook.executor import CellExecutor

        raw, source = _over_mount(tmp_path)
        _touch(raw / "f.txt", "aaaa")
        nb = Notebook(tmp_path, source, READER.replace("dict(out)", "out"))
        monkeypatch.setattr(executor_module, "STALENESS_READ_BYTES", 0)
        read = TestReviewFixes._counted(monkeypatch)

        await CellExecutor(nb.session).estimate_reads("a", source)

        assert read == []

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0, reason="needs a file its owner cannot read"
    )
    def test_a_pinned_mounts_unreadable_item_is_keyed_as_such(self, tmp_path):
        from strata.notebook import each

        raw = tmp_path / "raw"
        raw.mkdir()
        _touch(raw / "a.txt", "aaaa")
        _touch(raw / "b.txt", "bbbb")
        (raw / "b.txt").chmod(0)
        try:
            items = each.items_of_mount(raw, None, digests_path=tmp_path / "hashes.sqlite")
        finally:
            (raw / "b.txt").chmod(0o600)

        assert {i.key: i.digest for i in items}["b.txt"] == "unreadable"

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0, reason="needs a file its owner cannot read"
    )
    def test_an_unreadable_file_does_not_stop_the_parallel_read(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        raw.mkdir()
        for name in ("a", "b", "c"):
            _touch(raw / f"{name}.txt", name * 4)
        (raw / "b.txt").chmod(0)
        one_by_one = []
        digest = FileDigests.digest
        monkeypatch.setattr(
            FileDigests, "digest", lambda self, p: one_by_one.append(p) or digest(self, p)
        )
        try:
            digests = local_digests(raw, tmp_path / "hashes.sqlite")
        finally:
            (raw / "b.txt").chmod(0o600)

        assert digests is not None and digests["b.txt"] == "unreadable"
        assert one_by_one == []

    def test_a_mount_over_the_notebook_keys_the_same_twice(self, tmp_path, later):
        digests_path = tmp_path / ".strata" / "hashes.sqlite"
        _touch(tmp_path / "a.txt", "aaaa")
        local_digests(tmp_path, digests_path)

        second = local_digests(tmp_path, digests_path)
        third = local_digests(tmp_path, digests_path)

        assert second == third
        assert not any(rel.startswith(".strata/hashes.sqlite") for rel in second)


class TestBackgroundHashing:
    """A mount too large for the staleness check to hash is hashed by a task."""

    async def _cold(self, tmp_path, monkeypatch, *sources: str):
        raw, source = _over_mount(tmp_path)
        _touch(raw / "f.txt", "aaaa")
        nb = Notebook(tmp_path, source, *(sources or (READER.replace("dict(out)", "out"),)))
        for cell_id in nb.ids:
            await nb.good(cell_id)
        nb.ran()
        # A new stat with the same bytes: more to read than staleness may.
        monkeypatch.setattr(mounts, "STALENESS_READ_BYTES", 3)
        _touch(raw / "f.txt", "aaaa", tick=5)
        return nb

    async def test_staleness_reads_nothing_and_a_task_hashes_the_mount_then_it_is_ready(
        self, tmp_path, monkeypatch, later
    ):
        from vestigo import digests

        from strata.notebook import ws
        from strata.notebook.models import CellStaleness, StalenessReason
        from strata.notebook.staleness_explanations import CausalityReason, CausalityType

        nb = await self._cold(tmp_path, monkeypatch)
        reads = []
        read_digest = digests._read_digest
        monkeypatch.setattr(
            digests, "_read_digest", lambda path: reads.append(path) or read_digest(path)
        )
        broadcast = []

        async def record(session, notebook_id, staleness_map):
            broadcast.append({k: v.status for k, v in staleness_map.items()})

        monkeypatch.setattr(ws, "_broadcast_staleness_updates", record)

        staleness = await nb.session.compute_staleness_async()

        assert reads == []
        # Not known to have changed: its own reason, not ``external``.
        assert staleness["a"] == CellStaleness(
            status=CellStatus.STALE, reasons=[StalenessReason.CHECKING]
        )
        chain = nb.session.causality_map["a"]
        assert chain.reason == CausalityReason.CHECKING
        assert [(d.type, d.input_name) for d in chain.details] == [(CausalityType.CHECKING, "raw")]
        jobs = list(nb.session._mount_hashing.values())
        assert len(jobs) == 1

        await asyncio.gather(*jobs)

        assert reads
        assert nb.session.notebook_state.get_cell("a").status == CellStatus.READY
        assert broadcast == [{"a": CellStatus.READY, "b": CellStatus.READY}]
        assert nb.ran() == []

    async def test_one_task_per_mount_however_many_cells_and_checks(
        self, tmp_path, monkeypatch, later
    ):
        raw = tmp_path / "raw"
        second = f"# @mount raw {raw.as_uri()} ro\nalso = (raw / 'f.txt').read_text()\n"
        nb = await self._cold(tmp_path, monkeypatch, second)
        started = []
        hash_mount = nb.session._hash_mount

        def counted(*key):
            started.append(key)
            return hash_mount(*key)

        monkeypatch.setattr(nb.session, "_hash_mount", counted)

        await nb.session.compute_staleness_async()
        await nb.session.compute_staleness_async()

        assert len(started) == 1
        await asyncio.gather(*nb.session._mount_hashing.values())
        assert nb.session._mount_hashing == {}

    async def test_a_closed_session_is_not_recomputed(self, tmp_path, monkeypatch, later):
        from strata.notebook import ws

        nb = await self._cold(tmp_path, monkeypatch)
        refreshed = []

        async def record(session):
            refreshed.append(session)

        monkeypatch.setattr(ws, "refresh_staleness", record)
        await nb.session.compute_staleness_async()
        jobs = list(nb.session._mount_hashing.values())
        nb.session.closed = True

        await asyncio.gather(*jobs)

        assert refreshed == []

    def test_a_run_meanwhile_waits_for_the_hash_and_reads_nothing_again(
        self, tmp_path, monkeypatch, later
    ):
        from vestigo import digests

        raw = tmp_path / "raw"
        raw.mkdir()
        _touch(raw / "f.txt", "aaaa")
        reads = []
        reading = threading.Event()
        release = threading.Event()
        read_digest = digests._read_digest

        def slow(path):
            reads.append(path)
            if len(reads) == 1:
                reading.set()
                release.wait(10)
            return read_digest(path)

        monkeypatch.setattr(digests, "_read_digest", slow)
        cache = tmp_path / "hashes.sqlite"
        found = []
        first = threading.Thread(target=lambda: found.append(local_digests(raw, cache)))
        second = threading.Thread(target=lambda: found.append(local_digests(raw, cache)))
        first.start()
        assert reading.wait(10)
        second.start()
        second.join(0.5)
        # Without the lock it reads the file itself, past the held first read.
        assert second.is_alive()
        release.set()
        first.join(10)
        second.join(10)

        assert len(reads) == 1
        assert found[0] == found[1]

    async def test_a_mount_gone_before_its_hash_is_recomputed_as_changed(
        self, tmp_path, monkeypatch, later
    ):
        import shutil

        from strata.notebook import ws
        from strata.notebook.staleness_explanations import CausalityType

        nb = await self._cold(tmp_path, monkeypatch)
        monkeypatch.setattr(ws, "_broadcast_staleness_updates", lambda *a: asyncio.sleep(0))
        await nb.session.compute_staleness_async()
        jobs = list(nb.session._mount_hashing.values())
        shutil.rmtree(tmp_path / "raw")

        await asyncio.gather(*jobs)

        details = nb.session.causality_map["a"].details
        assert [(d.type, d.input_name) for d in details] == [
            (CausalityType.EXTERNAL_CHANGED, "raw")
        ]

    async def test_staleness_reads_no_mount_its_job_is_reading(self, tmp_path, monkeypatch, later):
        from vestigo import digests

        from strata.notebook.staleness_explanations import CausalityType

        nb = await self._cold(tmp_path, monkeypatch)
        reads = []
        reading = threading.Event()
        release = threading.Event()
        read_digest = digests._read_digest

        def slow(path):
            reads.append(path)
            reading.set()
            release.wait(10)
            return read_digest(path)

        monkeypatch.setattr(digests, "_read_digest", slow)
        await nb.session.compute_staleness_async()
        jobs = list(nb.session._mount_hashing.values())
        await asyncio.to_thread(reading.wait, 10)
        # What is left is within what staleness may read; the job is reading it.
        monkeypatch.setattr(mounts, "STALENESS_READ_BYTES", STALENESS_READ_BYTES)

        await nb.session.compute_staleness_async()

        details = nb.session.causality_map["a"].details
        assert [d.type for d in details] == [CausalityType.CHECKING]
        release.set()
        await asyncio.gather(*jobs)
        assert len(reads) == 1

    def test_a_file_too_fresh_to_remember_is_hashed_again_a_few_times(self, tmp_path, monkeypatch):
        from vestigo import digests

        raw = tmp_path / "raw"
        raw.mkdir()
        _touch(raw / "f.txt", "aaaa")  # its ctime is now: the cache will not keep it
        monkeypatch.setattr(mounts, "STALENESS_READ_BYTES", 3)
        monkeypatch.setattr(mounts, "_SETTLE_SECONDS", 0)
        reads = []
        read_digest = digests._read_digest
        monkeypatch.setattr(
            digests, "_read_digest", lambda path: reads.append(path) or read_digest(path)
        )

        mounts.hash_local_mount(raw, tmp_path / "hashes.sqlite")

        assert len(reads) == mounts._HASH_ATTEMPTS

    async def test_a_hash_that_fails_unexpectedly_ends_its_job(self, tmp_path, monkeypatch):
        from strata.notebook import ws

        nb = await self._cold(tmp_path, monkeypatch)

        def broken(root, digests_path):
            raise ValueError("this FileDigests is closed")

        monkeypatch.setattr(mounts, "hash_local_mount", broken)
        monkeypatch.setattr(ws, "_broadcast_staleness_updates", lambda *a: asyncio.sleep(0))
        await nb.session.compute_staleness_async()
        jobs = list(nb.session._mount_hashing.values())

        await asyncio.wait_for(asyncio.gather(*jobs), 10)

        assert nb.session._mount_hashing == {}

    async def test_a_job_left_on_a_closed_loop_does_not_hold_its_mount(
        self, tmp_path, monkeypatch, later
    ):
        import types

        nb = await self._cold(tmp_path, monkeypatch)
        await nb.session.compute_staleness_async()
        (key,) = nb.session._mount_hashing
        await asyncio.gather(*nb.session._mount_hashing.values())
        closed = asyncio.new_event_loop()
        closed.close()
        nb.session._mount_hashing[key] = types.SimpleNamespace(
            done=lambda: False, get_loop=lambda: closed
        )
        _touch(tmp_path / "raw" / "f.txt", "aaaa", tick=9)

        await nb.session.compute_staleness_async()

        assert nb.session._mount_hashing[key].get_loop() is asyncio.get_running_loop()
        await asyncio.gather(nb.session._mount_hashing[key])

    async def test_no_recompute_while_a_run_is_active(self, tmp_path, monkeypatch):
        from strata.notebook import ws

        nb = await self._cold(tmp_path, monkeypatch)
        monkeypatch.setattr(ws, "notebook_has_active_execution", lambda notebook_id: True)
        recomputed = []
        compute = nb.session.compute_staleness_async

        async def counted(*args, **kwargs):
            recomputed.append(kwargs)
            return await compute(*args, **kwargs)

        await nb.session.compute_staleness_async()
        jobs = list(nb.session._mount_hashing.values())
        monkeypatch.setattr(nb.session, "compute_staleness_async", counted)

        await asyncio.gather(*jobs)

        assert recomputed == []

    async def test_an_offline_handle_starts_no_job(self, tmp_path, monkeypatch):
        from strata.notebook.ops import LocalNotebookOps

        nb = await self._cold(tmp_path, monkeypatch)
        started = []
        monkeypatch.setattr(
            "strata.notebook.session.NotebookSession._hash_mount",
            lambda self, *key: started.append(key) or asyncio.sleep(0),
        )

        LocalNotebookOps(nb.session.path).list_cells()

        assert started == []


class TestHashProgress:
    """A hash of more than a batch reports how far it is, on the cells it is for."""

    @pytest.fixture
    def small_batches(self, monkeypatch):
        monkeypatch.setattr(mounts, "_PROGRESS_BATCH_BYTES", 4)
        monkeypatch.setattr(mounts, "_READ_THREADS", 1)

    def _files(self, tmp_path: Path, count: int = 3) -> Path:
        raw = tmp_path / "raw"
        raw.mkdir()
        for i in range(count):
            _touch(raw / f"{i}.txt", f"{i:04d}", tick=i)
        return raw

    def test_each_batch_is_reported_and_the_last_report_finishes(self, tmp_path, small_batches):
        raw = self._files(tmp_path)
        reports = []

        digests = local_digests(
            raw, tmp_path / "hashes.sqlite", on_progress=lambda *r: reports.append(r)
        )

        assert len(digests) == 3
        assert reports == [(0, 12, False), (4, 12, False), (8, 12, False), (12, 12, True)]

    def test_a_batch_holds_a_file_per_read_thread(self, tmp_path, monkeypatch, small_batches):
        # Fewer would leave the threads ``digest_many`` reads on idle.
        monkeypatch.setattr(mounts, "_READ_THREADS", 2)
        raw = self._files(tmp_path, count=4)
        reports = []

        local_digests(raw, tmp_path / "hashes.sqlite", on_progress=lambda *r: reports.append(r))

        assert reports == [(0, 16, False), (8, 16, False), (16, 16, True)]

    def test_the_tries_of_a_background_hash_finish_once(self, tmp_path, monkeypatch, small_batches):
        raw = self._files(tmp_path)
        monkeypatch.setattr(mounts, "_SETTLE_SECONDS", 0)
        # Never settled, so each try reads the files again.
        monkeypatch.setattr(mounts, "STALENESS_READ_BYTES", 0)
        reports = []

        hash_local_mount(raw, tmp_path / "hashes.sqlite", on_progress=lambda *r: reports.append(r))

        assert [r for r in reports if r[2]] == [(12, 12, True)]
        assert reports[-1] == (12, 12, True)
        assert reports.count((0, 12, False)) == mounts._HASH_ATTEMPTS

    async def test_reports_reach_the_loop_in_order(self):
        loop = asyncio.get_running_loop()
        sent = []

        async def send(bytes_done, total, done):
            await asyncio.sleep(0.01 if bytes_done == 0 else 0)
            sent.append((bytes_done, done))

        report = on_loop(loop, send)
        await asyncio.to_thread(lambda: (report(0, 8, False), report(8, 8, True)))
        while len(sent) < 2:
            await asyncio.sleep(0.01)

        assert sent == [(0, False), (8, True)]

    def test_a_report_after_the_loop_closed_is_dropped(self):
        loop = asyncio.new_event_loop()
        sent = []

        async def send(*report):
            sent.append(report)

        report = on_loop(loop, send)
        loop.close()

        report(0, 8, False)

        assert sent == []

    def test_a_hash_of_one_batch_reports_nothing(self, tmp_path):
        raw = self._files(tmp_path)
        reports = []

        local_digests(raw, tmp_path / "hashes.sqlite", on_progress=lambda *r: reports.append(r))

        assert reports == []

    def test_a_hash_that_fails_still_reports_it_finished(
        self, tmp_path, monkeypatch, small_batches
    ):
        raw = self._files(tmp_path)
        reports = []
        read_batch = mounts._read_batch
        batches = []

        def failing(cache, files):
            batches.append(files)
            if len(batches) == 2:
                raise ValueError("this FileDigests is closed")
            return read_batch(cache, files)

        monkeypatch.setattr(mounts, "_read_batch", failing)

        with pytest.raises(ValueError):
            local_digests(raw, tmp_path / "hashes.sqlite", on_progress=lambda *r: reports.append(r))

        assert reports[-1] == (4, 12, True)

    async def test_a_run_reports_its_hash_on_its_cell(self, tmp_path, small_batches):
        from strata.notebook.executor import CellExecutor

        raw, source = _over_mount(tmp_path)
        _touch(raw / "f.txt", "aaaa")
        _touch(raw / "g.txt", "cccc", tick=1)
        nb = Notebook(tmp_path, source)
        frames = []

        async def record(progress):
            frames.append(progress)

        executor = CellExecutor(nb.session)
        executor.on_mount_progress = record
        result = await executor.execute_cell("a", nb.session.notebook_state.get_cell("a").source)
        assert result.success, result.error
        while not frames or not frames[-1]["done"]:
            await asyncio.sleep(0.01)  # the reports are scheduled from the hashing thread

        assert {(f["cell_id"], f["mount"]) for f in frames} == {("a", "raw")}
        assert frames[0]["bytes_done"] == 0
        assert frames[-1] == {
            "cell_id": "a",
            "mount": "raw",
            "bytes_done": 8,
            "bytes": 8,
            "done": True,
        }

    async def test_the_background_hash_reports_on_every_cell_that_mounts_the_folder(
        self, tmp_path, monkeypatch, later
    ):
        from strata.notebook import ws

        raw, source = _over_mount(tmp_path)
        _touch(raw / "f.txt", "aaaa")
        _touch(raw / "g.txt", "cccc", tick=1)
        second = f"# @mount data {raw.as_uri()} ro\nalso = (data / 'f.txt').read_text()\n"
        nb = Notebook(tmp_path, source, second)
        for cell_id in nb.ids:
            await nb.good(cell_id)
        monkeypatch.setattr(mounts, "STALENESS_READ_BYTES", 3)
        monkeypatch.setattr(mounts, "_PROGRESS_BATCH_BYTES", 4)
        monkeypatch.setattr(mounts, "_READ_THREADS", 1)
        _touch(raw / "f.txt", "aaaa", tick=5)
        _touch(raw / "g.txt", "cccc", tick=6)
        frames = []

        async def record(notebook_id, progress):
            frames.append(progress)

        monkeypatch.setattr(ws, "broadcast_mount_progress", record)
        monkeypatch.setattr(ws, "_broadcast_staleness_updates", lambda *a: asyncio.sleep(0))

        await nb.session.compute_staleness_async()
        await asyncio.gather(*nb.session._mount_hashing.values())
        while sum(f["done"] for f in frames) < 2:
            await asyncio.sleep(0.01)

        assert {(f["cell_id"], f["mount"]) for f in frames} == {("a", "raw"), ("b", "data")}
        assert {f["cell_id"]: f["done"] for f in frames if f["done"]} == {"a": True, "b": True}

    def test_the_cells_shown_are_those_that_mount_the_folder_now(self, tmp_path):
        raw, source = _over_mount(tmp_path)
        _touch(raw / "f.txt", "aaaa")
        twice = (
            f"# @mount data {raw.as_uri()} ro\n# @mount same {raw.as_uri()} ro\n"
            "also = (data / 'f.txt').read_text()\n"
        )
        written = f"# @mount out {raw.as_uri()} rw\nwrote = 1\n"
        nb = Notebook(tmp_path, source, twice, written)
        session = nb.session

        assert sorted(session._cell_mounts_of(raw)) == [("a", "raw"), ("b", "data"), ("b", "same")]

        session.notebook_state.get_cell("a").source = "out = 1\n"

        assert sorted(session._cell_mounts_of(raw)) == [("b", "data"), ("b", "same")]
