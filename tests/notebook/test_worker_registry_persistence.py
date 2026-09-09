"""An admin change to the worker registry survives a restart.

The admin routes replaced an in-memory dict and nothing wrote it anywhere, so
adding a machine type through the API worked until the next restart and then
silently reverted — which for a fleet manager means a catalogue that quietly
disagrees with what the server will dispatch to.
"""

from __future__ import annotations

import json

import pytest

from strata.notebook.models import WorkerBackendType, WorkerSpec
from strata.notebook.workers import (
    ManagedWorkerRecord,
    get_server_managed_worker_records,
    load_persisted_managed_worker_records,
    managed_worker_registry_path,
    replace_server_managed_worker_records,
)


def _worker(name: str, url: str = "http://gpu.internal:9000") -> WorkerSpec:
    return WorkerSpec(
        name=name,
        backend=WorkerBackendType.EXECUTOR,
        runtime_id=f"{name}-runtime",
        config={"url": url},
    )


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A service-mode server state with an artifact dir to persist into."""
    from tests.conftest import run_server_with_context

    with run_server_with_context(tmp_path / "cache", tmp_path / "artifacts", "personal") as ctx:
        yield ctx


class TestPersistence:
    def test_an_admin_change_is_written_to_disk(self, server):
        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("gpu-a100"), True)])

        path = managed_worker_registry_path()
        assert path is not None and path.exists()
        assert json.loads(path.read_text())[0]["name"] == "gpu-a100"

    def test_it_is_read_back(self, server):
        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("gpu-a100"), True)])

        records = load_persisted_managed_worker_records()

        assert records is not None
        assert [r.worker.name for r in records] == ["gpu-a100"]

    def test_the_file_wins_over_the_configured_table(self, server):
        """It is the later statement: an admin said so after the config did.

        The config table is rewritten afterwards, so this cannot pass by
        reading back the in-memory value the mutation also set — which is what
        it would do if nothing were persisted at all.
        """
        from strata.server import get_state

        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("from-admin"), True)])
        get_state().config.transforms_config["notebook_workers"] = [
            {"name": "from-config", "backend": "executor", "config": {"url": "http://x:1"}}
        ]

        assert [r.worker.name for r in get_server_managed_worker_records()] == ["from-admin"]

    def test_an_empty_registry_is_not_a_missing_one(self, server):
        """Deleting every worker is a decision, not an absence.

        Falling back to the config table here would resurrect the machine
        types an operator just removed.
        """
        from strata.server import get_state

        get_state().config.transforms_config["notebook_workers"] = [
            {"name": "from-config", "backend": "executor", "config": {"url": "http://x:1"}}
        ]
        replace_server_managed_worker_records([])

        assert load_persisted_managed_worker_records() == []
        assert get_server_managed_worker_records() == []

    def test_no_file_means_fall_back_to_the_configured_table(self, server):
        from strata.server import get_state

        get_state().config.transforms_config["notebook_workers"] = [
            {"name": "from-config", "backend": "executor", "config": {"url": "http://x:1"}}
        ]

        assert load_persisted_managed_worker_records() is None
        assert [r.worker.name for r in get_server_managed_worker_records()] == ["from-config"]


class TestAfterARestart:
    """What dispatch sees, not just what the admin routes report.

    Every other test here runs in one process, where the mutation sets both
    the file and the in-memory config table — so they agree for the wrong
    reason. A restart is the case that matters: the file holds the admin's
    registry and the config table holds whatever was configured. If those two
    are read by different code paths, the admin UI shows one catalogue while
    cells dispatch to another.
    """

    def test_the_catalogue_reflects_the_persisted_registry(self, server, tmp_path):
        from strata.notebook.models import NotebookState
        from strata.notebook.workers import build_worker_catalog
        from strata.server import get_state

        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("gpu-a100"), True)])

        # The restart: the process comes back with the configured table, and
        # the registry is whatever is on disk.
        get_state().config.transforms_config["notebook_workers"] = [
            {"name": "from-config", "backend": "executor", "config": {"url": "http://x:1"}}
        ]
        get_state().config.deployment_mode = "service"

        names = {entry["name"] for entry in build_worker_catalog(NotebookState(id="nb", name="nb"))}

        assert "gpu-a100" in names, "dispatch must see what the admin routes persisted"
        assert "from-config" not in names


class TestDurability:
    def test_a_corrupt_file_does_not_stop_the_server_reading_a_registry(self, server):
        """A bad file must not be fatal, and must not be silent either."""
        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("gpu-a100"), True)])
        path = managed_worker_registry_path()
        path.write_text("{ this is not json")

        assert load_persisted_managed_worker_records() is None

    def test_the_write_is_atomic(self, server):
        """Rewritten on every mutation; a truncated file at boot is a server
        that starts with no workers and no obvious reason why."""
        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("a"), True)])
        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("b"), True)])

        path = managed_worker_registry_path()
        leftovers = list(path.parent.glob(f"{path.name}.*"))

        assert leftovers == [], f"temp files left behind: {leftovers}"
        assert json.loads(path.read_text())[0]["name"] == "b"


class TestHealthCachePruning:
    def test_entries_for_removed_workers_are_dropped(self, server):
        from strata.notebook import workers as workers_mod

        # The cache is module-global and other tests populate it, so this
        # asserts on the two keys it owns rather than on a total.
        replace_server_managed_worker_records([ManagedWorkerRecord(_worker("gpu-a100"), True)])
        live_url = workers_mod._health_url_for_worker(_worker("gpu-a100"))
        retired_url = "http://retired.internal:9000/health"
        workers_mod._worker_health_cache[live_url] = object()
        workers_mod._worker_health_cache[retired_url] = object()

        workers_mod.prune_worker_health_cache()

        assert live_url in workers_mod._worker_health_cache
        assert retired_url not in workers_mod._worker_health_cache
