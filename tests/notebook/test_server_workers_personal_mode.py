"""A personal server can offer a managed catalogue of machine types.

Before, the server-managed registry was consulted only in service mode. A
platform that manages machine types for its users had to write the same
``[[workers]]`` block into every ``notebook.toml`` — where it shows up in git
diffs and drifts the moment the catalogue changes. Item 20.
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
        """The point: no `[[workers]]` block needed in notebook.toml."""
        replace_server_managed_worker_records(
            [ManagedWorkerRecord(_spec("gpu-a100", "http://gpu.internal:9000"), True)]
        )

        entry = next(e for e in build_worker_catalog(notebook) if e["name"] == "gpu-a100")

        assert entry["source"] == "server"
        assert entry["allowed"] is True

    def test_it_is_dispatchable(self, personal_server, notebook):
        """Showing it in the panel and refusing to run on it would be worse
        than not offering it."""
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


class TestNotebookWins:
    """A name the notebook defines beats the server's, in both places.

    The catalogue and dispatch resolve collisions independently — one is
    first-writer-wins over an ordered list, the other last-writer-wins over a
    dict — so they have to be checked separately or they will disagree
    silently: the panel showing one URL while the cell runs against another.
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
        """Merging a server catalogue does not make a personal notebook
        read-only — its own `[[workers]]` are still its to edit."""
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
