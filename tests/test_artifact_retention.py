"""Retention, Strata's side: the settings, the sweep through the server and the command, and
the notebook's own pruning.

The store's rules (use, minted ids, the byte cap, folders, what a kept result needs) are tested
in vestigo.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

import pytest
from vestigo.store import ArtifactStore


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def _row(store: ArtifactStore, artifact_id: str, version: int) -> sqlite3.Row:
    conn = sqlite3.connect(store.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT last_used_at, minted, created_at FROM artifact_versions "
            "WHERE id = ? AND version = ?",
            (artifact_id, version),
        ).fetchone()
    finally:
        conn.close()


def _set(store: ArtifactStore, artifact_id: str, version: int, **columns) -> None:
    conn = sqlite3.connect(store.db_path)
    try:
        assignments = ", ".join(f"{name} = ?" for name in columns)
        conn.execute(
            f"UPDATE artifact_versions SET {assignments} WHERE id = ? AND version = ?",
            (*columns.values(), artifact_id, version),
        )
        conn.commit()
    finally:
        conn.close()


def _ready(
    store: ArtifactStore,
    artifact_id: str | None = None,
    *,
    size: int = 100,
    inputs: dict[str, str] | None = None,
) -> tuple[str, int]:
    """A ready version; with no id, one the store minted, like a materialize miss."""
    minted = artifact_id is None
    artifact_id = artifact_id or str(uuid.uuid4())
    version = store.create_artifact(
        artifact_id, f"prov-{uuid.uuid4()}", input_versions=inputs, minted=minted
    )
    store.write_blob(artifact_id, version, b"x" * size)
    store.finalize_artifact(artifact_id, version, "{}", 1, size)
    return artifact_id, version


def _last_used(store: ArtifactStore, key: tuple[str, int], seconds_ago: float) -> None:
    """Make *key* look created long ago and last used *seconds_ago*."""
    _set(store, *key, created_at=0.0, last_used_at=time.time() - seconds_ago)


def _exists(store: ArtifactStore, key: tuple[str, int]) -> bool:
    return store.get_artifact(*key) is not None


DAY = 86400.0


class TestTheSettings:
    def _config(self, tmp_path: Path, **overrides):
        from strata.config import StrataConfig

        return StrataConfig(
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            **overrides,
        )

    def test_personal_mode_is_bounded_unless_told_otherwise(self, tmp_path):
        config = self._config(tmp_path, deployment_mode="personal")

        assert config.artifact_gc_interval_seconds == 3600
        assert config.artifact_gc_policy() == {
            "max_idle_days": 30.0,
            "max_bytes": 20 * 1024**3,
            "min_idle_seconds": 3600.0,
        }

    def test_service_mode_sweeps_only_when_an_operator_says_so(self, tmp_path):
        config = self._config(
            tmp_path, deployment_mode="service", auth_mode="trusted_proxy", proxy_token="t"
        )

        assert config.artifact_gc_interval_seconds is None
        assert config.artifact_gc_policy()["max_bytes"] is None

    def test_zero_turns_each_limit_off(self, tmp_path):
        config = self._config(
            tmp_path,
            deployment_mode="personal",
            artifact_gc_interval_seconds=0,
            artifact_gc_max_bytes=0,
            artifact_gc_max_idle_days=0,
        )

        assert not config.artifact_gc_interval_seconds
        assert config.artifact_gc_policy()["max_bytes"] is None
        assert config.artifact_gc_policy()["max_idle_days"] is None

    def test_every_setting_reads_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRATA_ARTIFACT_GC_INTERVAL_SECONDS", "600")
        monkeypatch.setenv("STRATA_ARTIFACT_GC_MAX_BYTES", "1000")
        monkeypatch.setenv("STRATA_ARTIFACT_GC_MAX_IDLE_DAYS", "2")
        monkeypatch.setenv("STRATA_ARTIFACT_GC_MIN_IDLE_SECONDS", "5")

        config = self._config(tmp_path, deployment_mode="personal")

        assert config.artifact_gc_interval_seconds == 600
        assert config.artifact_gc_policy() == {
            "max_idle_days": 2.0,
            "max_bytes": 1000,
            "min_idle_seconds": 5.0,
        }


class TestThroughTheServer:
    @pytest.fixture
    def server(self, tmp_path):
        from tests.conftest import run_server_with_context

        cache_dir = tmp_path / "cache"
        artifact_dir = tmp_path / "artifacts"
        cache_dir.mkdir()
        artifact_dir.mkdir()
        with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
            yield ctx.base_url, ArtifactStore(artifact_dir)

    @staticmethod
    def _put(base_url: str, path: str, metadata: dict) -> str:
        import json

        import httpx
        import pyarrow as pa

        from tests.conftest import table_to_ipc_bytes

        files = {
            "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
            "data": (
                "data.arrow",
                table_to_ipc_bytes(pa.table({"id": [1]})),
                "application/octet-stream",
            ),
        }
        response = httpx.put(f"{base_url}{path}", files=files, timeout=30.0)
        assert response.status_code == 200, response.text
        return response.json()["artifact_uri"]

    @staticmethod
    def _key(uri: str) -> tuple[str, int]:
        artifact_id, version = uri.removeprefix("strata://artifact/").split("@v=")
        return artifact_id, int(version)

    def test_a_collected_result_is_rebuilt_under_a_new_id(self, server, temp_warehouse):
        """A collected id's version is never reissued: the next miss mints a new id."""
        from strata_client.client import StrataClient

        base_url, store = server
        scan = {"executor": "scan@v1", "params": {"columns": ["id"]}}
        client = StrataClient(base_url=base_url)
        try:
            first = client.materialize(inputs=[temp_warehouse["table_uri"]], transform=scan)
            rows = first.to_table().num_rows
            assert _row(store, first.artifact_id, first.version)["minted"] == 1

            swept = client.garbage_collect(max_idle_days=0, min_idle_seconds=0)
            assert swept["deleted_count"] >= 1
            assert store.get_artifact(first.artifact_id, first.version) is None

            again = client.materialize(inputs=[temp_warehouse["table_uri"]], transform=scan)
            assert again.cache_hit is False
            assert again.artifact_id != first.artifact_id
            assert again.to_table().num_rows == rows
        finally:
            client.close()

    def test_a_result_used_only_as_an_input_stays_while_its_downstream_is_asked_for(
        self, server, temp_warehouse
    ):
        """A dashboard always hits the downstream and never reads the scan, so input use must count
        or the scan idles out.
        """
        from strata_client.client import StrataClient

        base_url, store = server
        scan_spec = {"executor": "scan@v1", "params": {"columns": ["id", "value"]}}
        sql_spec = {"executor": "duckdb_sql@v1", "params": {"sql": "SELECT id FROM input0"}}
        client = StrataClient(base_url=base_url)
        try:
            scan = client.materialize(inputs=[temp_warehouse["table_uri"]], transform=scan_spec)
            scan.to_table()
            client.materialize(inputs=[scan.uri], transform=sql_spec).to_table()
            conn = sqlite3.connect(store.db_path)
            conn.execute(
                "UPDATE artifact_versions SET created_at = ?, last_used_at = ?",
                (time.time() - 40 * DAY, time.time() - 40 * DAY),
            )
            conn.commit()
            conn.close()

            assert client.materialize(inputs=[scan.uri], transform=sql_spec).cache_hit
            client.garbage_collect()  # the configured retention, as the sweep runs it

            assert _exists(store, (scan.artifact_id, scan.version))
            assert client.materialize(inputs=[scan.uri], transform=sql_spec).cache_hit
        finally:
            client.close()

    def test_a_transform_result_is_minted(self, server, temp_warehouse):
        from strata_client.client import StrataClient

        base_url, store = server
        client = StrataClient(base_url=base_url)
        try:
            result = client.materialize(
                inputs=[temp_warehouse["table_uri"]],
                transform={"executor": "duckdb_sql@v1", "params": {"sql": "SELECT * FROM input0"}},
            )
            assert _row(store, result.artifact_id, result.version)["minted"] == 1
        finally:
            client.close()

    def test_a_put_result_is_minted(self, server):
        base_url, store = server
        uri = self._put(
            base_url,
            "/v1/artifacts",
            {"inputs": [], "transform": {"executor": "local@v1", "params": {}}},
        )

        assert _row(store, *self._key(uri))["minted"] == 1

    def test_a_cell_output_that_names_its_id_is_not(self, server):
        """The team-cache write: the notebook names the id and reads back its latest version."""
        base_url, store = server
        named = self._put(
            base_url,
            f"/v1/artifacts/by-provenance/{'a' * 64}",
            {"content_type": "arrow/ipc", "artifact_id": "nb_abc_cell_def_var_df"},
        )
        unnamed = self._put(
            base_url,
            f"/v1/artifacts/by-provenance/{'b' * 64}",
            {"content_type": "arrow/ipc"},
        )

        assert _row(store, *self._key(named))["minted"] == 0
        assert _row(store, *self._key(unnamed))["minted"] == 1

    def test_a_bare_sweep_request_uses_the_configured_retention(self, server):
        """Nothing just written goes: the configured one-hour floor applies."""
        import httpx

        from tests.conftest import LIVE_SERVER_TIMEOUT

        base_url, store = server
        key = _ready(store)
        _last_used(store, key, 60)
        idle = _ready(store)
        _last_used(store, idle, 7200)

        dry = httpx.post(
            f"{base_url}/v1/artifacts/gc",
            params={"dry_run": True, "max_idle_days": 0},
            timeout=LIVE_SERVER_TIMEOUT,
        )

        assert dry.status_code == 200, dry.text
        assert [(c["artifact_id"], c["version"]) for c in dry.json()["collected"]] == [idle]
        assert _exists(store, key)


class TestTheCommand:
    @staticmethod
    def _run(capsys, *argv: str) -> tuple[int, dict]:
        from strata import cli

        code = cli.main(["artifact", "gc", "--format", "json", *argv])
        return code, json.loads(capsys.readouterr().out)

    def test_a_dry_run_reports_and_keeps(self, store, capsys):
        idle = _ready(store)
        _last_used(store, idle, 40 * DAY)

        code, report = self._run(
            capsys, "--artifact-dir", str(store.artifact_dir), "--max-idle-days", "30", "--dry-run"
        )

        assert code == 0
        assert [(c["artifact_id"], c["version"]) for c in report["collected"]] == [idle]
        assert _exists(store, idle)

    def test_a_size_limit_collects_the_least_recently_used_down_to_it(self, store, capsys):
        keys = [_ready(store, size=1024) for _ in range(3)]
        for hours_ago, key in zip((3, 2, 1), keys, strict=True):
            _last_used(store, key, hours_ago * 3600)

        # 3 KiB over a 2 KiB cap: down to 80% of it, so the two least recent go.
        # The configured idle limit (30 days) takes nothing here.
        code, report = self._run(
            capsys,
            "--artifact-dir",
            str(store.artifact_dir),
            "--max-bytes",
            "2K",
            "--min-idle-seconds",
            "0",
        )

        assert code == 0
        assert report["deleted_count"] == 2
        assert [_exists(store, key) for key in keys] == [False, False, True]

    @pytest.mark.parametrize(
        ("text", "size"),
        [
            ("500", 500),
            ("2K", 2048),
            ("20G", 20 * 1024**3),
            ("1.5M", 1572864),
            ("20GiB", 20 * 1024**3),
        ],
    )
    def test_sizes_read_as_powers_of_1024(self, text, size):
        from strata.artifact.cli import _parse_size

        assert _parse_size(text) == size

    def test_a_size_that_is_not_one_is_refused(self, store, capsys):
        from strata import cli

        code = cli.main(
            ["artifact", "gc", "--artifact-dir", str(store.artifact_dir), "--max-bytes", "lots"]
        )

        assert code == 2
        assert "not a size" in capsys.readouterr().err


def _values(
    store: ArtifactStore, artifact_id: str, count: int, *, fail: tuple[int, ...] = ()
) -> None:
    """*count* successive values of one cell output; the ones in *fail* failed."""
    for version in range(1, count + 1):
        store.create_artifact(artifact_id, f"prov-{artifact_id}-{version}")
        if version in fail:
            store.fail_artifact(artifact_id, version)
            continue
        store.write_blob(artifact_id, version, b"x")
        store.finalize_artifact(artifact_id, version, "{}", 1, 1)


class TestNotebookStores:
    OUTPUT = "nb_abc_cell_def_var_df"

    @staticmethod
    def _notebook_with_history(tmp_path: Path) -> tuple[Path, ArtifactStore]:
        from strata.notebook.writer import create_notebook

        notebook = create_notebook(tmp_path, "history", initialize_environment=False)
        store = ArtifactStore(notebook / ".strata" / "artifacts")
        _values(store, TestNotebookStores.OUTPUT, 5)
        return notebook, store

    @staticmethod
    def _open(notebook: Path, monkeypatch, config) -> None:
        import threading

        from strata.notebook import harness_user
        from strata.notebook.session import SessionManager

        monkeypatch.setattr(harness_user, "running_server_config", lambda: config)
        SessionManager().open_notebook(notebook, skip_initial_venv_sync=True)
        for thread in threading.enumerate():
            if thread.name == "notebook-artifact-prune":
                thread.join(timeout=30)

    def test_the_server_prunes_a_notebook_it_opens(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        notebook, store = self._notebook_with_history(tmp_path)
        config = SimpleNamespace(
            notebook_keep_superseded_versions=2, artifact_gc_min_idle_seconds=0
        )

        self._open(notebook, monkeypatch, config)

        assert [_exists(store, (self.OUTPUT, v)) for v in range(1, 6)] == [
            False,
            False,
            True,
            True,
            True,
        ]

    def test_reopening_a_notebook_that_is_already_open_prunes_it_too(self, tmp_path, monkeypatch):
        """A personal server reuses the session, which would otherwise never prune again."""
        import threading
        from types import SimpleNamespace

        from strata.notebook import harness_user
        from strata.notebook.session import SessionManager

        notebook, store = self._notebook_with_history(tmp_path)
        config = SimpleNamespace(
            notebook_keep_superseded_versions=2, artifact_gc_min_idle_seconds=0
        )
        monkeypatch.setattr(harness_user, "running_server_config", lambda: config)
        monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)
        manager = SessionManager()

        def open_and_wait_for_the_prune():
            session = manager.open_notebook(notebook, reuse_existing=True)
            for thread in threading.enumerate():
                if thread.name == "notebook-artifact-prune":
                    thread.join(timeout=30)
            return session

        first = open_and_wait_for_the_prune()
        # Two more edits of the same output while the session stays open.
        for version in (6, 7):
            store.create_artifact(self.OUTPUT, f"prov-{self.OUTPUT}-{version}")
            store.write_blob(self.OUTPUT, version, b"x")
            store.finalize_artifact(self.OUTPUT, version, "{}", 1, 1)
        second = open_and_wait_for_the_prune()

        assert second is first
        assert [_exists(store, (self.OUTPUT, v)) for v in range(1, 8)] == [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
        ]

    def test_opening_drops_a_removed_cells_idle_values(self, tmp_path, monkeypatch):
        """The store keeps every id's current value, so a removed cell's stayed for good."""
        from types import SimpleNamespace

        from strata.notebook.parser import parse_notebook
        from strata.notebook.writer import add_cell_to_notebook, create_notebook

        notebook = create_notebook(tmp_path, "removed", initialize_environment=False)
        add_cell_to_notebook(notebook, "live0001")
        notebook_id = parse_notebook(notebook).id
        store = ArtifactStore(notebook / ".strata" / "artifacts")
        removed = _ready(store, f"nb_{notebook_id}_cell_gone0001_var_df")
        live = _ready(store, f"nb_{notebook_id}_cell_live0001_var_df")
        for key in (removed, live):
            _last_used(store, key, 31 * DAY)
        config = SimpleNamespace(
            notebook_keep_superseded_versions=2, artifact_gc_min_idle_seconds=0
        )

        self._open(notebook, monkeypatch, config)

        assert not _exists(store, removed)
        assert _exists(store, live)

    def test_reopening_sees_a_cell_that_came_back_on_disk(self, tmp_path, monkeypatch):
        """A reused session prunes before it reloads; its cell list is the old one."""
        import threading
        from types import SimpleNamespace

        from strata.notebook import harness_user
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import SessionManager
        from strata.notebook.writer import add_cell_to_notebook, create_notebook

        notebook = create_notebook(tmp_path, "back", initialize_environment=False)
        notebook_id = parse_notebook(notebook).id
        config = SimpleNamespace(
            notebook_keep_superseded_versions=2, artifact_gc_min_idle_seconds=0
        )
        monkeypatch.setattr(harness_user, "running_server_config", lambda: config)
        monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)
        manager = SessionManager()

        def open_and_wait_for_the_prune():
            manager.open_notebook(notebook, reuse_existing=True)
            for thread in threading.enumerate():
                if thread.name == "notebook-artifact-prune":
                    thread.join(timeout=30)

        open_and_wait_for_the_prune()
        # A checkout brings the cell back while the session stays open.
        add_cell_to_notebook(notebook, "back0001")
        store = ArtifactStore(notebook / ".strata" / "artifacts")
        value = _ready(store, f"nb_{notebook_id}_cell_back0001_var_df")
        _last_used(store, value, 31 * DAY)

        open_and_wait_for_the_prune()

        assert _exists(store, value)

    @pytest.mark.parametrize("config", [None, "off"], ids=["no server", "set to 0"])
    def test_without_a_server_or_with_pruning_off_nothing_changes(
        self, tmp_path, monkeypatch, config
    ):
        from types import SimpleNamespace

        notebook, store = self._notebook_with_history(tmp_path)
        if config == "off":
            config = SimpleNamespace(
                notebook_keep_superseded_versions=0, artifact_gc_min_idle_seconds=0
            )

        self._open(notebook, monkeypatch, config)

        assert all(_exists(store, (self.OUTPUT, v)) for v in range(1, 6))

    def test_a_write_a_crash_left_stops_holding_what_it_read(self, tmp_path):
        """A crash between create and finalize leaves a ``building`` row, a root that would
        keep every input behind it for good.
        """
        from strata.notebook.artifact_integration import NotebookArtifactManager

        manager = NotebookArtifactManager("nb", artifact_dir=tmp_path / "artifacts")
        store = manager.artifact_store
        up = "nb_nb_cell_up_var_x"
        _values(store, up, 1)
        ref = f"{up}@v=1"
        crashed = store.create_artifact(
            "nb_nb_cell_dn_var_y", "prov-dn", input_versions={f"strata://artifact/{ref}": ref}
        )
        _set(store, "nb_nb_cell_dn_var_y", crashed, created_at=time.time() - 2 * 3600)
        for version in range(2, 6):
            store.create_artifact(up, f"prov-{up}-{version}")
            store.write_blob(up, version, b"x")
            store.finalize_artifact(up, version, "{}", 1, 1)

        manager.prune(keep_superseded=1, min_idle_seconds=0)

        assert store.get_artifact("nb_nb_cell_dn_var_y", crashed).state == "failed"
        assert not _exists(store, (up, 1))

    def test_the_setting_reads_from_the_environment(self, tmp_path, monkeypatch):
        from strata.config import StrataConfig

        monkeypatch.setenv("STRATA_NOTEBOOK_KEEP_SUPERSEDED_VERSIONS", "5")

        config = StrataConfig(cache_dir=tmp_path / "cache", artifact_dir=tmp_path / "a")

        assert config.notebook_keep_superseded_versions == 5

    def test_a_quiesced_notebook_is_not_pruned(self, tmp_path):
        """A copy in progress must see the store as it was."""
        from strata.notebook import quiesce
        from strata.notebook.artifact_integration import NotebookArtifactManager

        notebook = (tmp_path / "nb").resolve()
        manager = NotebookArtifactManager("nb", artifact_dir=notebook / ".strata" / "artifacts")
        _values(manager.artifact_store, "nb_nb_cell_c_var_x", 5)
        quiesce.settle(quiesce.begin(notebook, max_hold_seconds=60))
        try:
            with pytest.raises(quiesce.NotebookQuiesced):
                manager.prune(1, 0)
        finally:
            quiesce.release(notebook)

        assert all(_exists(manager.artifact_store, ("nb_nb_cell_c_var_x", v)) for v in range(1, 6))


async def test_the_first_sweep_runs_soon_after_startup(monkeypatch):
    """A laptop server restarted more often than hourly must still sweep."""
    import asyncio
    from types import SimpleNamespace

    import strata.server as server_module

    assert server_module._FIRST_ARTIFACT_GC_DELAY_SECONDS <= 60
    monkeypatch.setattr(server_module, "_FIRST_ARTIFACT_GC_DELAY_SECONDS", 0.0)
    swept = asyncio.Event()

    def garbage_collect(**_policy):
        swept.set()
        return {"deleted_count": 0, "deleted_bytes": 0}

    task = asyncio.create_task(
        server_module._artifact_gc_loop(SimpleNamespace(garbage_collect=garbage_collect), 3600, {})
    )
    try:
        await asyncio.wait_for(swept.wait(), timeout=30)
    finally:
        task.cancel()
