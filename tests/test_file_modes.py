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

from strata.artifact_store import ArtifactStore
from strata.file_modes import PASS_THROUGH_DIR, narrow, private_dir

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


class TestTheApiKeyCli:
    def test_a_key_minted_before_the_first_start_leaves_the_store_owner_only(
        self, tmp_path, capsys
    ):
        import argparse

        from strata.api_key_cli import cmd_create

        args = argparse.Namespace(
            principal="svc",
            tenant=None,
            scopes=None,
            description=None,
            expires_in_days=None,
            artifact_dir=str(tmp_path / "artifacts"),
            dsn=None,
        )
        assert cmd_create(args) == 0
        capsys.readouterr()

        assert _mode(tmp_path / "artifacts") == 0o700
        assert _mode(tmp_path / "artifacts" / "artifacts.sqlite") == 0o600


class TestTheCache:
    def test_building_a_config_leaves_the_cache_dir_mode_alone(self, tmp_path):
        """Tests and tools build configs all the time; only a running cache narrows the dir."""
        from strata.config import StrataConfig

        cache = tmp_path / "cache"
        cache.mkdir(mode=0o755)

        StrataConfig(cache_dir=cache, artifact_dir=tmp_path / "a")

        assert _mode(cache) == 0o755

    def test_server_startup_makes_the_cache_dir_owner_only(self, tmp_path):
        from strata.config import StrataConfig
        from strata.server import ServerState

        cache = tmp_path / "cache"
        cache.mkdir(mode=0o755)
        state = ServerState(StrataConfig(cache_dir=cache, artifact_dir=tmp_path / "a"))
        try:
            assert _mode(cache) == 0o700
        finally:
            state._planning_executor.shutdown()
            state._fetch_executor.shutdown()


class TestANotebooksRuntimeState:
    @pytest.fixture
    def session(self, tmp_path, monkeypatch):
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import create_notebook

        monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)
        nb = create_notebook(tmp_path, "modes", initialize_environment=False)
        return NotebookSession(parse_notebook(nb), nb)

    def test_the_server_only_parts_are_owner_only(self, session):
        from strata.notebook.writer import update_cell_console_output

        update_cell_console_output(session.path, "c1", "printed", "")
        session._persist_environment_job_history()
        state = session.path / ".strata"

        assert _mode(state / "artifacts") == 0o700
        assert _mode(state / "artifacts" / "artifacts.sqlite") == 0o600
        assert _mode(state / "console") == 0o700
        assert _mode(state / "environment_jobs.json") == 0o600

    def test_the_harness_user_can_still_pass_through(self, session):
        """Its per-run directories, fetched bytes and mounted inputs live in ``.strata/``."""
        assert _mode(session.path / ".strata") == PASS_THROUGH_DIR


class TestTheTenantStorageRoot:
    def test_it_can_be_passed_through_but_not_listed(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from strata.notebook import routes

        monkeypatch.setattr(routes, "_get_notebook_storage_root", lambda: tmp_path / "nbs")
        monkeypatch.setattr(routes, "_caller_tenant_dir", lambda: "acme")

        root = routes._get_caller_storage_root(SimpleNamespace())

        assert root == tmp_path / "nbs" / "acme"
        assert _mode(root) == PASS_THROUGH_DIR


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
