"""The state a server keeps on disk is its own: no other account on the host reads it.

The harness user is such an account, and one serves every tenant, so world-readable
state was every tenant's metadata, publication links and blobs handed to every cell.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

from vestigo.file_modes import PASS_THROUGH_DIR, narrow, private_dir
from vestigo.store import ArtifactStore

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes do not apply")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture(autouse=True)
def default_umask():
    """The umask most servers start under, which made all of this world-readable."""
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def _journals(db: Path) -> list[Path]:
    return [db.with_name(db.name + suffix) for suffix in ("-wal", "-shm")]


def _held_open(db: Path) -> sqlite3.Connection:
    """A connection that keeps the WAL and shared-memory files on disk until closed."""
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE IF NOT EXISTS held (x)")
    conn.commit()
    return conn


class TestTheArtifactStore:
    def test_a_new_store_is_owner_only(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts")
        version = store.create_artifact("fig", "prov")
        with store.open_blob_writer("fig", version) as writer:
            writer.write(b"bytes")
        store.finalize_artifact("fig", version, "{}", row_count=0, byte_size=5)
        held = _held_open(store.db_path)
        try:
            # SQLite gives its journal files the database file's mode.
            assert [_mode(p) for p in _journals(store.db_path)] == [0o600, 0o600]
        finally:
            held.close()

        assert _mode(tmp_path / "artifacts") == 0o700
        assert _mode(store.blobs_dir) == 0o700
        assert _mode(store.db_path) == 0o600
        assert {_mode(p) for p in store.blobs_dir.iterdir()} == {0o600}

    def test_a_store_an_earlier_release_left_wide_is_narrowed(self, tmp_path):
        db = ArtifactStore(tmp_path / "artifacts").db_path
        held = _held_open(db)
        try:
            for path, mode in (
                (tmp_path / "artifacts", 0o755),
                (tmp_path / "artifacts" / "blobs", 0o755),
                (db, 0o644),
                *((journal, 0o644) for journal in _journals(db)),
            ):
                path.chmod(mode)

            ArtifactStore(tmp_path / "artifacts")

            assert [_mode(p) for p in _journals(db)] == [0o600, 0o600]
        finally:
            held.close()
        assert _mode(tmp_path / "artifacts") == 0o700
        assert _mode(tmp_path / "artifacts" / "blobs") == 0o700
        assert _mode(db) == 0o600

    def test_journals_sqlite_deletes_while_the_store_opens_are_skipped(self, tmp_path, monkeypatch):
        """A live server's last connection closing deletes them between any check and the chmod."""
        import vestigo.file_modes

        db = ArtifactStore(tmp_path / "artifacts").db_path
        held = _held_open(db)
        assert all(p.exists() for p in _journals(db))
        real_narrow = vestigo.file_modes.narrow

        def narrow_after_the_other_process_closes(path, mode):
            if path in _journals(db):
                held.close()
            real_narrow(path, mode)

        monkeypatch.setattr(vestigo.file_modes, "narrow", narrow_after_the_other_process_closes)

        store = ArtifactStore(tmp_path / "artifacts")

        assert not any(p.exists() for p in _journals(db))
        assert store.create_artifact("fig", "prov") == 1


class TestNeverLooser:
    def test_a_stricter_umask_stands(self, tmp_path):
        os.umask(0o077)

        assert _mode(private_dir(tmp_path / "strict", PASS_THROUGH_DIR)) == 0o700

    def test_narrowing_never_adds_a_bit(self, tmp_path):
        path = tmp_path / "f"
        path.write_text("x")
        path.chmod(0o400)

        narrow(path, 0o600)

        assert _mode(path) == 0o400
