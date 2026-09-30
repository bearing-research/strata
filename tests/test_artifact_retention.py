"""Retention: the artifact store stays bounded without anyone deleting by hand.

A personal store grew by every distinct ``materialize`` forever. The sweep that
existed could not help: every result is version 1 of an id the store minted for
it, so "spare the latest version of an id" spared all of them. Retention now
keys on use rather than creation, knows which ids the store minted, and can
hold a store under a byte cap.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

import pytest

from strata.artifact_store import _LATEST_SCHEMA_VERSION, ArtifactStore


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
    """A ready version. With no id, one the store minted, as a materialize miss."""
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


class TestUse:
    def test_a_provenance_hit_is_a_use(self, store):
        key = _ready(store)
        provenance = store.get_artifact(*key).provenance_hash
        assert _row(store, *key)["last_used_at"] is None

        store.find_by_provenance(provenance)

        assert _row(store, *key)["last_used_at"] == pytest.approx(time.time(), abs=60)

    @pytest.mark.parametrize("read", ["read_blob", "open_blob_reader"])
    def test_reading_the_bytes_is_a_use(self, store, read):
        key = _ready(store)

        result = getattr(store, read)(*key)
        if read == "open_blob_reader":
            with result as reader:
                reader.read()

        assert _row(store, *key)["last_used_at"] == pytest.approx(time.time(), abs=60)

    def test_a_second_use_within_the_hour_writes_nothing(self, store):
        """Every read an UPDATE would make reads writes for no gain."""
        key = _ready(store)
        earlier = time.time() - 120
        _set(store, *key, last_used_at=earlier)

        store.read_blob(*key)

        assert _row(store, *key)["last_used_at"] == earlier

    def test_hashing_the_bytes_is_not_a_use(self, store):
        """Finalize and verify hash every version; neither is anyone using it."""
        key = _ready(store)  # finalize computed the digest

        store.blob_digest(*key)
        store.verify_artifacts()

        assert _row(store, *key)["last_used_at"] is None

    def test_a_store_that_cannot_take_the_write_still_serves_the_read(self, store, monkeypatch):
        key = _ready(store)
        real = store._get_connection

        class Refusing:
            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *args):
                if sql.startswith("UPDATE artifact_versions SET last_used_at"):
                    raise sqlite3.OperationalError("attempt to write a readonly database")
                return self._conn.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        monkeypatch.setattr(store, "_get_connection", lambda: Refusing(real()))

        assert store.read_blob(*key) == b"x" * 100
        assert _row(store, *key)["last_used_at"] is None


class TestMinted:
    def test_a_later_version_of_a_minted_id_stays_minted(self, store):
        """A refresh rebuild reuses the id it was handed and does not say so."""
        artifact_id, _ = _ready(store)
        store.create_artifact(artifact_id, "prov-refresh")

        assert _row(store, artifact_id, 2)["minted"] == 1

    def test_a_named_id_is_not_minted(self, store):
        key = _ready(store, "nb_abc_cell_def_var_model")

        assert _row(store, *key)["minted"] == 0

    def test_the_migration_marks_the_ids_the_store_made_up(self, tmp_path):
        """Rows from before the column: every id the store minted is a uuid4,
        and the ids callers choose are not."""
        store = ArtifactStore(tmp_path / "old")
        minted = _ready(store)
        named = _ready(store, "nb_abc_cell_def_var_model")
        conn = sqlite3.connect(store.db_path)
        conn.execute("UPDATE schema_version SET version = ?", (_LATEST_SCHEMA_VERSION - 1,))
        conn.execute("ALTER TABLE artifact_versions DROP COLUMN minted")
        conn.execute("ALTER TABLE artifact_versions DROP COLUMN last_used_at")
        conn.commit()
        conn.close()

        reopened = ArtifactStore(tmp_path / "old")

        assert _row(reopened, *minted)["minted"] == 1
        assert _row(reopened, *named)["minted"] == 0
        assert _row(reopened, *minted)["last_used_at"] is None


class TestWhatIsCollected:
    def test_an_idle_minted_result_goes_and_a_notebook_output_stays(self, store):
        """A notebook output's latest version is the cell's value; a minted
        id's is only a result nobody asked for again."""
        minted = _ready(store)
        notebook = _ready(store, "nb_abc_cell_def_var_model")
        for key in (minted, notebook):
            _last_used(store, key, 40 * DAY)

        result = store.garbage_collect(max_idle_days=30)

        assert result["deleted_count"] == 1
        assert not _exists(store, minted)
        assert _exists(store, notebook)

    def test_use_not_creation_decides(self, store):
        """The scan a dashboard hits every day is as old as an abandoned one."""
        used = _ready(store)
        abandoned = _ready(store)
        _last_used(store, used, 1 * DAY)
        _last_used(store, abandoned, 40 * DAY)

        store.garbage_collect(max_idle_days=30)

        assert _exists(store, used)
        assert not _exists(store, abandoned)

    def test_never_what_was_just_used(self, store):
        just_used = _ready(store)
        a_while_ago = _ready(store)
        _last_used(store, just_used, 60)
        _last_used(store, a_while_ago, 7200)

        store.garbage_collect(max_idle_days=0, min_idle_seconds=3600)

        assert _exists(store, just_used)
        assert not _exists(store, a_while_ago)

    def test_over_the_cap_the_least_recently_used_go_first_down_to_80_percent(self, store):
        keys = [_ready(store, size=100) for _ in range(5)]
        for hours_ago, key in zip((5, 4, 3, 2, 1), keys, strict=True):
            _last_used(store, key, hours_ago * 3600)

        # 500 bytes over a 400 cap: down to 320, so the two least recent go.
        result = store.garbage_collect(max_bytes=400)

        assert result["store_bytes"] == 500
        assert result["deleted_count"] == 2
        assert [_exists(store, key) for key in keys] == [False, False, True, True, True]

    def test_under_the_cap_nothing_goes(self, store):
        keys = [_ready(store, size=100) for _ in range(3)]
        for key in keys:
            _last_used(store, key, 40 * DAY)

        assert store.garbage_collect(max_bytes=1000)["deleted_count"] == 0

    def test_with_no_limit_nothing_goes(self, store):
        key = _ready(store)
        _last_used(store, key, 400 * DAY)

        assert store.garbage_collect()["deleted_count"] == 0
        assert _exists(store, key)

    def test_what_a_running_build_reads_is_kept(self, store):
        source = _ready(store)
        _last_used(store, source, 40 * DAY)
        ref = f"{source[0]}@v={source[1]}"
        store.create_artifact(
            str(uuid.uuid4()), "prov-building", input_versions={f"strata://artifact/{ref}": ref}
        )

        store.garbage_collect(max_idle_days=30)

        assert _exists(store, source)

    def test_an_id_never_loses_its_top_while_keeping_a_lower_version(self, store):
        """``create_artifact`` numbers versions MAX + 1: a top collected under
        a kept version would be reissued, and a URI somebody holds for it
        would serve other bytes."""
        artifact_id, _ = _ready(store)
        store.create_artifact(artifact_id, "prov-v2")
        store.write_blob(artifact_id, 2, b"y" * 100)
        store.finalize_artifact(artifact_id, 2, "{}", 1, 100)
        _last_used(store, (artifact_id, 1), 60)  # v1 kept: just used
        _last_used(store, (artifact_id, 2), 40 * DAY)

        store.garbage_collect(max_idle_days=30, min_idle_seconds=3600)

        assert _exists(store, (artifact_id, 2))

        _last_used(store, (artifact_id, 1), 40 * DAY)
        store.garbage_collect(max_idle_days=30, min_idle_seconds=3600)

        assert not _exists(store, (artifact_id, 1))
        assert not _exists(store, (artifact_id, 2))

    def test_a_dry_run_names_what_would_go_and_deletes_nothing(self, store):
        idle = _ready(store)
        _last_used(store, idle, 40 * DAY)

        result = store.garbage_collect(max_idle_days=30, dry_run=True)

        assert result["dry_run"] is True
        assert [(c["artifact_id"], c["version"]) for c in result["collected"]] == [idle]
        assert result["deleted_bytes"] == 100
        assert _exists(store, idle)


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
        config = self._config(tmp_path, deployment_mode="service")

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
        """The store never reissues a collected id's version number: the next
        miss mints another id rather than writing version 1 of the old one."""
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
        """The team-cache write: the notebook names the id and reads it back
        as its latest version."""
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

        base_url, store = server
        key = _ready(store)
        _last_used(store, key, 60)

        dry = httpx.post(
            f"{base_url}/v1/artifacts/gc", params={"dry_run": True, "max_idle_days": 0}
        )

        assert dry.status_code == 200, dry.text
        assert dry.json()["collected"] == []
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
        from strata.artifact_cli import _parse_size

        assert _parse_size(text) == size

    def test_a_size_that_is_not_one_is_refused(self, store, capsys):
        from strata import cli

        code = cli.main(
            ["artifact", "gc", "--artifact-dir", str(store.artifact_dir), "--max-bytes", "lots"]
        )

        assert code == 2
        assert "not a size" in capsys.readouterr().err
