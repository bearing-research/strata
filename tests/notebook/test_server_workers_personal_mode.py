"""A personal server can offer a managed catalogue of machine types.

Without it, every ``notebook.toml`` needs the same ``[[workers]]`` block, which
shows up in git diffs and drifts when the catalogue changes.
"""

from __future__ import annotations

import pytest

from strata.notebook.models import NotebookState, WorkerBackendType, WorkerSpec
from strata.notebook.workers import (
    ManagedWorkerRecord,
    build_worker_catalog,
    notebook_worker_definitions_editable,
    replace_server_managed_worker_records,
    resolve_worker_spec,
)


def _spec(name: str, url: str) -> WorkerSpec:
    return WorkerSpec(
        name=name,
        backend=WorkerBackendType.EXECUTOR,
        runtime_id=f"{name}-runtime",
        config={"url": url},
    )


@pytest.fixture
def personal_server(tmp_path):
    from tests.conftest import run_server_with_context

    with run_server_with_context(tmp_path / "cache", tmp_path / "artifacts", "personal") as ctx:
        yield ctx


@pytest.fixture
def notebook():
    return NotebookState(id="nb", name="nb")


class TestMergedCatalogue:
    def test_a_server_worker_is_offered_to_a_notebook_with_none(self, personal_server, notebook):
        """No ``[[workers]]`` block is needed in notebook.toml."""
        replace_server_managed_worker_records(
            [ManagedWorkerRecord(_spec("gpu-a100", "http://gpu.internal:9000"), True)]
        )

        entry = next(e for e in build_worker_catalog(notebook) if e["name"] == "gpu-a100")

        assert entry["source"] == "server"
        assert entry["allowed"] is True

    def test_it_is_dispatchable(self, personal_server, notebook):
        """Offering a worker in the panel and refusing to run on it would be worse than hiding
        it."""
        replace_server_managed_worker_records(
            [ManagedWorkerRecord(_spec("gpu-a100", "http://gpu.internal:9000"), True)]
        )

        resolved = resolve_worker_spec(notebook, "gpu-a100")

        assert resolved is not None
        assert resolved.config.url == "http://gpu.internal:9000"

    def test_a_disabled_server_worker_is_not_dispatchable(self, personal_server, notebook):
        replace_server_managed_worker_records(
            [ManagedWorkerRecord(_spec("retired", "http://old.internal:9000"), False)]
        )

        assert resolve_worker_spec(notebook, "retired") is None


class TestTheTokenIsNotListed:
    """A managed worker's literal token stays on the server; the catalogue omits it."""

    def _register(self):
        spec = _spec("gpu-a100", "http://gpu.internal:9000")
        spec.config.token = "LITERAL-TOKEN"
        replace_server_managed_worker_records([ManagedWorkerRecord(spec, True)])

    def test_the_notebook_catalogue(self, personal_server, notebook):
        self._register()

        entry = next(e for e in build_worker_catalog(notebook) if e["name"] == "gpu-a100")

        assert entry["config"] == {"url": "http://gpu.internal:9000"}
        assert resolve_worker_spec(notebook, "gpu-a100").config.token == "LITERAL-TOKEN"

    async def test_the_server_catalogue(self, personal_server, monkeypatch):
        from strata.notebook import workers

        async def healthy(worker, **_kwargs):
            return workers.WorkerHealthSnapshot(checked_at=0.0, health="healthy")

        monkeypatch.setattr(workers, "probe_worker_health", healthy)
        self._register()

        catalog = await workers.build_server_worker_catalog_with_health(
            workers.get_server_managed_worker_records()
        )

        entry = next(e for e in catalog if e["name"] == "gpu-a100")
        assert entry["config"] == {"url": "http://gpu.internal:9000"}


class TestNotebookWins:
    """A name the notebook defines beats the server's, in both places.

    The catalogue (first-writer-wins over a list) and dispatch (last-writer-wins
    over a dict) resolve collisions independently, so each is checked; otherwise
    the panel could show one URL while the cell runs against another.
    """

    def _collide(self, notebook):
        notebook.workers = [_spec("gpu-a100", "http://mine.local:9000")]
        replace_server_managed_worker_records(
            [ManagedWorkerRecord(_spec("gpu-a100", "http://server.internal:9000"), True)]
        )

    def test_dispatch_uses_the_notebook_entry(self, personal_server, notebook):
        self._collide(notebook)

        assert resolve_worker_spec(notebook, "gpu-a100").config.url == "http://mine.local:9000"

    def test_the_catalogue_shows_the_notebook_entry(self, personal_server, notebook):
        self._collide(notebook)

        entries = [e for e in build_worker_catalog(notebook) if e["name"] == "gpu-a100"]

        assert len(entries) == 1, "one name, one entry"
        assert entries[0]["source"] == "notebook"
        assert entries[0]["config"]["url"] == "http://mine.local:9000"


class TestUnchanged:
    def test_notebook_definitions_stay_editable(self, personal_server, notebook):
        """A merged server catalogue leaves the notebook's own ``[[workers]]`` editable."""
        replace_server_managed_worker_records(
            [ManagedWorkerRecord(_spec("gpu-a100", "http://gpu.internal:9000"), True)]
        )

        assert notebook_worker_definitions_editable(notebook) is True

    def test_local_is_still_there(self, personal_server, notebook):
        replace_server_managed_worker_records(
            [ManagedWorkerRecord(_spec("gpu-a100", "http://gpu.internal:9000"), True)]
        )

        assert any(e["name"] == "local" for e in build_worker_catalog(notebook))

    def test_no_registry_means_the_notebook_alone(self, personal_server, notebook):
        notebook.workers = [_spec("mine", "http://mine.local:9000")]

        names = {e["name"] for e in build_worker_catalog(notebook)}

        assert names == {"local", "mine"}
