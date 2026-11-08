"""The file hash cache: content keys, reused only while the stat holds."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook import hash_cache
from strata.notebook.hash_cache import FileHashes

_LATER_NS = 10_000_000_000


@pytest.fixture
def reads(monkeypatch) -> list[Path]:
    """Every file the cache reads, in order."""
    seen: list[Path] = []
    real = hash_cache._read_digest

    def counted(path: Path) -> str:
        seen.append(path)
        return real(path)

    monkeypatch.setattr(hash_cache, "_read_digest", counted)
    return seen


@pytest.fixture
def stats(monkeypatch) -> dict[Path, SimpleNamespace]:
    """Stats to report in place of the filesystem's, by path: ctime can't be set."""
    fixed: dict[Path, SimpleNamespace] = {}
    real = os.stat

    def stat(path, *args, **kwargs):
        return fixed.get(Path(path)) or real(path, *args, **kwargs)

    monkeypatch.setattr(hash_cache.os, "stat", stat)
    return fixed


def _stat_of(path: Path, **changed: int) -> SimpleNamespace:
    st = os.stat(path)
    fields = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino", "st_dev")
    return SimpleNamespace(**{f: getattr(st, f) for f in fields} | changed)


def _cache(tmp_path: Path, offset_ns: int = _LATER_NS) -> FileHashes:
    """A cache whose clock is *offset_ns* past now, so what it hashes is that old."""
    return FileHashes(
        tmp_path / "state" / "hashes.sqlite", clock=lambda: time.time_ns() + offset_ns
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_a_files_key_is_its_content(tmp_path, reads):
    data = tmp_path / "a.csv"
    data.write_bytes(b"x,y\n1,2\n")
    cache = _cache(tmp_path)

    assert cache.hash_of(data).sha256 == _sha(b"x,y\n1,2\n")
    assert cache.hash_of(data).sha256 == _sha(b"x,y\n1,2\n")
    assert cache.cached(data) == _sha(b"x,y\n1,2\n")
    assert reads == [data]


def test_a_rewrite_that_keeps_size_and_mtime_is_read_again(tmp_path, reads, stats):
    """Only the ctime says so: a write moves it, and nothing can set it back."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    before = _stat_of(data)
    stats[data] = before
    cache = _cache(tmp_path)
    cache.hash_of(data)

    data.write_bytes(b"3,4\n")
    stats[data] = _stat_of(data, st_mtime_ns=before.st_mtime_ns, st_ctime_ns=before.st_ctime_ns + 1)

    assert cache.cached(data) is None
    assert cache.hash_of(data).sha256 == _sha(b"3,4\n")
    assert len(reads) == 2


def test_a_metadata_change_costs_a_read_and_keeps_the_key(tmp_path, reads):
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    cache = _cache(tmp_path)
    first = cache.hash_of(data).sha256

    os.utime(data, ns=(0, 0))
    os.chmod(data, 0o600)

    assert cache.hash_of(data).sha256 == first
    assert len(reads) == 2


def test_a_hash_taken_as_the_file_changed_is_checked_again(tmp_path, stats):
    """The stat cannot see a write in the same tick as the read, so such a hash is not kept."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    stats[data] = _stat_of(data)
    racy = _cache(tmp_path, offset_ns=0)
    racy.hash_of(data)

    data.write_bytes(b"3,4\n")  # in the same tick: every stat field as before

    assert racy.cached(data) is None
    assert racy.hash_of(data).sha256 == _sha(b"3,4\n")


def test_an_old_mtime_does_not_make_a_fresh_write_trusted(tmp_path, stats):
    """An mtime set back (rsync -t, cp -p) says nothing; the ctime is when it changed."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    stats[data] = _stat_of(data, st_mtime_ns=0)
    racy = _cache(tmp_path, offset_ns=0)
    racy.hash_of(data)

    data.write_bytes(b"3,4\n")  # in the same tick, the mtime set back again

    assert racy.hash_of(data).sha256 == _sha(b"3,4\n")


def test_a_hash_taken_well_after_the_change_is_kept(tmp_path, reads, stats):
    """The counterpart: what the racy rule refuses above, it trusts here."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    stats[data] = _stat_of(data)
    cache = _cache(tmp_path)
    cache.hash_of(data)

    data.write_bytes(b"3,4\n")

    assert cache.hash_of(data).sha256 == _sha(b"1,2\n")
    assert reads == [data]


def test_a_file_written_while_it_is_read_is_unsettled_and_not_kept(tmp_path, monkeypatch):
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    real = hash_cache._read_digest
    reads: list[Path] = []

    def written_during(path: Path) -> str:
        reads.append(path)
        digest = real(path)
        if len(reads) == 1:
            with open(path, "ab") as f:
                f.write(b"3,4\n")
        return digest

    monkeypatch.setattr(hash_cache, "_read_digest", written_during)
    cache = _cache(tmp_path)

    first = cache.hash_of(data)
    assert not first.settled
    assert cache.cached(data) is None

    second = cache.hash_of(data)
    assert second.settled
    assert second.sha256 == _sha(b"1,2\n3,4\n")
    assert len(reads) == 2


def test_the_cache_outlives_its_process(tmp_path, reads):
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    cache = _cache(tmp_path)
    cache.hash_of(data)
    cache.close()

    assert _cache(tmp_path).hash_of(data).sha256 == _sha(b"1,2\n")
    assert reads == [data]


def test_prune_forgets_only_the_files_a_walk_of_its_root_did_not_find(tmp_path):
    root = tmp_path / "data"
    sibling = tmp_path / "data2"
    for folder in (root, sibling):
        folder.mkdir()
    kept, gone, nested, beside = (
        root / "kept.csv",
        root / "gone.csv",
        root / "sub" / "gone.csv",
        sibling / "beside.csv",
    )
    nested.parent.mkdir()
    cache = _cache(tmp_path)
    for path in (kept, gone, nested, beside):
        path.write_bytes(path.name.encode())
        cache.hash_of(path)

    assert cache.prune(root, [kept]) == 2

    assert cache.cached(kept) is not None
    assert cache.cached(beside) is not None
    # Still on disk, unchanged: only the forgotten row makes them unknown.
    assert cache.cached(gone) is None
    assert cache.cached(nested) is None


def test_a_missing_file_raises(tmp_path):
    cache = _cache(tmp_path)
    with pytest.raises(FileNotFoundError):
        cache.hash_of(tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError):
        cache.cached(tmp_path / "absent.csv")
