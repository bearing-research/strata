"""The server-managed worker registry lives in the artifact metadata store.

An admin change survives a restart, and every node sharing the store sees it; otherwise the
catalogue silently reverts, or a second replica refuses cells the first one accepts.
"""

from __future__ import annotations

import json
import logging

import pytest

from strata.artifact_store import ArtifactStore, get_artifact_store
from strata.notebook.models import WorkerBackendType, WorkerSpec
from strata.notebook.workers import (
    ManagedWorkerRecord,
    create_server_managed_worker_record,
    delete_server_managed_worker_record,
    get_server_managed_worker_records,
    import_worker_registry_file,
    replace_server_managed_worker_records,
    set_server_managed_worker_enabled,
    update_server_managed_worker_record,
)


def _worker(name: str, url: str = "http://gpu.internal:9000") -> WorkerSpec:
    return WorkerSpec(
        name=name,
        backend=WorkerBackendType.EXECUTOR,
        runtime_id=f"{name}-runtime",
        config={"url": url},
    )


def _names() -> list[str]:
    return [record.worker.name for record in get_server_managed_worker_records()]


@pytest.fixture
def server(tmp_path):
    """A server state with an artifact store to persist into."""
    from tests.conftest import run_server_with_context

    with run_server_with_context(tmp_path / "cache", tmp_path / "artifacts", "personal") as ctx:
        yield ctx


def _configure(*names: str) -> None:
    from strata.server import get_state

    get_state().config.transforms_config["notebook_workers"] = [
        {"name": name, "backend": "executor", "config": {"url": "http://x:1"}} for name in names
    ]


class TestPersistence:
    def test_an_admin_change_is_written_to_the_store(self, server):
        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("gpu-a100"), True)])

        entries = get_artifact_store().notebook_worker_entries()

        assert entries is not None and [e["name"] for e in entries] == ["gpu-a100"]
        assert not (server.config.artifact_dir / "notebook_workers.json").exists()

    def test_the_store_wins_over_the_configured_table(self, server):
        """The config table is rewritten afterwards, so this cannot pass by reading it back."""
        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("from-admin"), True)])
        _configure("from-config")

        assert _names() == ["from-admin"]

    def test_an_empty_registry_is_not_a_missing_one(self, server):
        """Deleting every worker is a decision; falling back would resurrect removed types."""
        _configure("from-config")
        replace_server_managed_worker_records([])

        assert get_artifact_store().notebook_worker_entries() == []
        assert get_server_managed_worker_records() == []

    def test_an_unset_registry_falls_back_to_the_configured_table(self, server):
        _configure("from-config")

        assert get_artifact_store().notebook_worker_entries() is None
        assert _names() == ["from-config"]

    def test_the_first_change_starts_from_the_configured_table(self, server):
        """Adding one worker to a configured registry must not drop the configured ones."""
        _configure("from-config")

        create_server_managed_worker_record(ManagedWorkerRecord(_worker("gpu-a100"), True))

        assert _names() == ["from-config", "gpu-a100"]

    def test_a_configured_table_naming_a_worker_twice_still_takes_a_change(self, server):
        # The store keys rows by name, so writing both entries failed the change with a 500.
        _configure("gpu", "cpu", "gpu")

        create_server_managed_worker_record(ManagedWorkerRecord(_worker("gpu-a100"), True))

        assert _names() == ["cpu", "gpu", "gpu-a100"]


class TestRowOperations:
    def test_create_update_enable_and_delete(self, server):
        create_server_managed_worker_record(ManagedWorkerRecord(_worker("a"), True))
        create_server_managed_worker_record(ManagedWorkerRecord(_worker("b"), True))
        with pytest.raises(ValueError):
            create_server_managed_worker_record(ManagedWorkerRecord(_worker("a"), True))

        # A rename keeps its place in the catalogue.
        update_server_managed_worker_record("a", ManagedWorkerRecord(_worker("a2"), True))
        assert _names() == ["a2", "b"]

        set_server_managed_worker_enabled("b", False)
        assert [r.enabled for r in get_server_managed_worker_records()] == [True, False]

        delete_server_managed_worker_record("a2")
        assert _names() == ["b"]
        with pytest.raises(KeyError):
            delete_server_managed_worker_record("a2")

    def test_a_refused_change_writes_nothing(self, server):
        create_server_managed_worker_record(ManagedWorkerRecord(_worker("a"), True))

        with pytest.raises(KeyError):
            set_server_managed_worker_enabled("missing", False)

        assert _names() == ["a"]


class TestAfterARestart:
    """What dispatch sees after a restart, not just what the admin routes report."""

    def test_the_catalogue_reflects_the_persisted_registry(self, server):
        from strata.notebook.models import NotebookState
        from strata.notebook.workers import build_worker_catalog
        from strata.server import get_state

        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("gpu-a100"), True)])

        # The restart: the process comes back with the configured table, and
        # the registry is whatever the store holds.
        _configure("from-config")
        get_state().config.deployment_mode = "service"

        names = {entry["name"] for entry in build_worker_catalog(NotebookState(id="nb", name="nb"))}

        assert "gpu-a100" in names, "dispatch must see what the admin routes persisted"
        assert "from-config" not in names


class TestSharedAcrossNodes:
    def test_another_store_on_the_same_database_sees_the_change(self, server):
        """A second node opens its own store on the same metadata; no per-node copy."""
        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("box"), True)])

        other_node = ArtifactStore(server.config.artifact_dir)
        entries = other_node.notebook_worker_entries()

        assert entries is not None and [e["name"] for e in entries] == ["box"]


class TestImportFromTheFile:
    """Earlier releases kept the registry in ``notebook_workers.json`` beside the store."""

    def _write(self, artifact_dir, *names: str):
        path = artifact_dir / "notebook_workers.json"
        path.write_text(
            json.dumps(
                [
                    {**_worker(name).model_dump(mode="json"), "enabled": name != "off"}
                    for name in names
                ]
            )
        )
        return path

    def test_the_file_is_imported_once_and_renamed(self, tmp_path, caplog):
        store = ArtifactStore(tmp_path / "artifacts")
        path = self._write(tmp_path / "artifacts", "box", "off")

        with caplog.at_level(logging.WARNING, logger="strata.notebook.workers"):
            import_worker_registry_file(tmp_path / "artifacts", store)

        entries = store.notebook_worker_entries()
        assert entries is not None
        assert [(e["name"], e["enabled"]) for e in entries] == [("box", True), ("off", False)]
        assert not path.exists()
        assert path.with_name("notebook_workers.json.migrated").exists()
        assert "Imported 2 notebook workers" in caplog.text

        # An admin change after the import is not undone by starting again.
        store.update_notebook_workers(lambda current: [])
        import_worker_registry_file(tmp_path / "artifacts", store)
        assert store.notebook_worker_entries() == []

    def test_a_stale_file_does_not_overwrite_a_registry_already_in_the_store(
        self, tmp_path, caplog
    ):
        """A second node starting with an old copy must not clobber what the first set."""
        store = ArtifactStore(tmp_path / "artifacts")
        store.update_notebook_workers(
            lambda current: [{**_worker("current").model_dump(mode="json"), "enabled": True}]
        )
        path = self._write(tmp_path / "artifacts", "stale")

        with caplog.at_level(logging.WARNING, logger="strata.notebook.workers"):
            import_worker_registry_file(tmp_path / "artifacts", store)

        entries = store.notebook_worker_entries()
        assert entries is not None and [e["name"] for e in entries] == ["current"]
        assert path.with_name("notebook_workers.json.migrated").exists()
        assert "already holds a notebook worker registry" in caplog.text

    def test_a_file_naming_a_worker_twice_imports_its_last_entry(self, tmp_path, caplog):
        # A 0.8.0 admin change could write such a file, and the import then failed startup.
        store = ArtifactStore(tmp_path / "artifacts")
        path = tmp_path / "artifacts" / "notebook_workers.json"
        path.write_text(
            json.dumps(
                [
                    _worker("gpu", "http://old:1").model_dump(mode="json"),
                    _worker("cpu").model_dump(mode="json"),
                    _worker("gpu", "http://new:1").model_dump(mode="json"),
                ]
            )
        )

        with caplog.at_level(logging.WARNING, logger="strata.notebook.workers"):
            import_worker_registry_file(tmp_path / "artifacts", store)

        entries = store.notebook_worker_entries()
        assert entries is not None
        assert [(e["name"], e["config"]["url"]) for e in entries] == [
            ("cpu", "http://gpu.internal:9000"),
            ("gpu", "http://new:1"),
        ]
        assert "worker(s) gpu more than once" in caplog.text

    def test_an_earlier_migrated_copy_is_never_replaced(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts")
        artifact_dir = tmp_path / "artifacts"
        self._write(artifact_dir, "imported")
        import_worker_registry_file(artifact_dir, store)
        first = (artifact_dir / "notebook_workers.json.migrated").read_text()

        self._write(artifact_dir, "second")
        import_worker_registry_file(artifact_dir, store)
        self._write(artifact_dir, "third")
        import_worker_registry_file(artifact_dir, store)

        assert (artifact_dir / "notebook_workers.json.migrated").read_text() == first
        assert "second" in (artifact_dir / "notebook_workers.json.migrated.1").read_text()
        assert "third" in (artifact_dir / "notebook_workers.json.migrated.2").read_text()

    def test_an_unreadable_file_is_left_in_place(self, tmp_path, caplog):
        store = ArtifactStore(tmp_path / "artifacts")
        path = tmp_path / "artifacts" / "notebook_workers.json"
        path.write_text("{ this is not json")

        with caplog.at_level(logging.ERROR, logger="strata.notebook.workers"):
            import_worker_registry_file(tmp_path / "artifacts", store)

        assert path.exists()
        assert store.notebook_worker_entries() is None
        assert "was not imported" in caplog.text

    def test_server_startup_imports_the_file(self, tmp_path):
        from tests.conftest import run_server_with_context

        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        self._write(artifact_dir, "box")

        with run_server_with_context(tmp_path / "cache", artifact_dir, "personal"):
            assert _names() == ["box"]
        assert (artifact_dir / "notebook_workers.json.migrated").exists()


class TestHealthCachePruning:
    def test_entries_for_removed_workers_are_dropped(self, server, monkeypatch):
        from strata.notebook import workers as workers_mod

        # The live key survives pruning, so a sentinel left in the real cache
        # would be read as a health record by a later test.
        monkeypatch.setattr(workers_mod, "_worker_health_cache", {})
        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("gpu-a100"), True)])
        live_url = workers_mod._health_url_for_worker(_worker("gpu-a100"))
        retired_url = "http://retired.internal:9000/health"
        workers_mod._worker_health_cache[live_url] = object()
        workers_mod._worker_health_cache[retired_url] = object()

        workers_mod.prune_worker_health_cache()

        assert live_url in workers_mod._worker_health_cache
        assert retired_url not in workers_mod._worker_health_cache


class TestAStoreErrorFailsClosed:
    """Falling back to personal rules on a read error would let a service-mode notebook's own
    ``[[workers]]`` entry run cells and be edited."""

    def test_a_notebook_worker_stays_refused_and_uneditable(self, server, monkeypatch):
        from strata.notebook.models import NotebookState
        from strata.notebook.workers import (
            notebook_worker_definitions_editable,
            resolve_worker_spec,
        )
        from strata.server import get_state

        get_state().config.deployment_mode = "service"
        notebook = NotebookState(id="nb", name="nb", workers=[_worker("evil", "http://attacker:9")])
        assert resolve_worker_spec(notebook, "evil") is None

        def _outage(self):
            raise RuntimeError("pool timeout")

        monkeypatch.setattr(ArtifactStore, "notebook_worker_entries", _outage)

        with pytest.raises(RuntimeError, match="pool timeout"):
            resolve_worker_spec(notebook, "evil")
        with pytest.raises(RuntimeError, match="pool timeout"):
            notebook_worker_definitions_editable(notebook)
