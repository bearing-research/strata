"""A local mount, and each ``@each`` item over one, is keyed by its files' content.

Stat only decides whether a file is read again: a touch costs a read, never a rerun, and
a rewrite that keeps the size and the mtime is still seen. These run real cells.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from vestigo.digests import FileDigests

from strata.notebook import mounts
from strata.notebook.models import CellStatus, MountSpec
from strata.notebook.mounts import (
    STALENESS_READ_BYTES,
    MountResolver,
    local_digests,
    mount_fingerprint_sync,
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

    def test_a_mount_not_read_yet_shows_as_changed_not_unavailable(self, tmp_path):
        from strata.notebook import nonce

        (tmp_path / "raw").mkdir()
        _touch(tmp_path / "raw" / "f.txt", "aaaa")
        resolver = MountResolver(cache_dir=tmp_path / "nb" / "mount_cache")
        mount = MountSpec(name="raw", uri=(tmp_path / "raw").as_uri())

        component = mount_fingerprint_sync(resolver, mount, read_bytes=0)

        assert component is not None
        assert not nonce.is_unresolved(component)
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
