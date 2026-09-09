"""The registry dashboard describes the store the cells actually write to.

With ``notebook_remote_store_url`` set, a cell's ``strata.put(name=...)`` and
every promotion land in the team's store. The dashboard read the local one, so
it showed an empty registry on exactly the deployment where the registry is the
point. Item 22.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi import HTTPException

from strata.artifact_store import ArtifactStore


@pytest.fixture
def team_dir(tmp_path):
    return tmp_path / "team"


@pytest.fixture
def team_store(tmp_path, team_dir):
    """A server standing in for the organization's store.

    The local routes are called directly rather than over HTTP, so this server
    is only ever the *target*. Pointing a server's remote-store URL at itself
    would make every registry read recurse, which is not a shape worth
    building a test around.
    """
    from tests.conftest import run_server_with_context

    with run_server_with_context(tmp_path / "cache", team_dir, "personal") as ctx:
        yield ctx.base_url


@pytest.fixture
def team_registry(team_dir, team_store):
    """A named artifact in the team's store, stamped as a cell's output."""
    store = ArtifactStore(team_dir)
    store.create_artifact("shared-model", "aa" * 32)
    store.finalize_artifact("shared-model", 1, '{"fields": []}', 3, 64)
    store.set_name("taxi/model", "shared-model", 1)
    store.set_tag("shared-model", 1, "nb_cell", "c1")
    store.set_tag("shared-model", 1, "stage", "candidate")
    return store


@pytest.fixture
def pointed_at_team(team_store, monkeypatch):
    """Make the local registry routes forward to the team store.

    Only from this thread. In a real deployment the notebook server and the
    team store are two processes, and only the first has a remote-store URL;
    in one process they would share ``remote_registry`` and the team store
    would forward every request back to itself. The thread is what separates
    them here: the routes under test are called directly from the test thread,
    the team store answers on uvicorn's.
    """
    import strata.api.remote_registry as remote
    import strata.api.routers.registry as registry_router

    target = (team_store.rstrip("/"), {})
    caller = threading.get_ident()

    def _target_for_the_caller():
        return target if threading.get_ident() == caller else None

    monkeypatch.setattr(registry_router, "remote_registry", _target_for_the_caller)
    monkeypatch.setattr(remote, "remote_registry", _target_for_the_caller)
    return target


def _local_store(tmp_path):
    """A local store holding something the team's does not, so a route that
    read the wrong one would be visibly reading the wrong one."""
    store = ArtifactStore(tmp_path / "local")
    store.create_artifact("private-scratch", "bb" * 32)
    store.finalize_artifact("private-scratch", 1, '{"fields": []}', 1, 8)
    store.set_name("scratch/private", "private-scratch", 1)
    return store


class TestRegistryTab:
    def test_the_summary_lists_the_teams_names(self, tmp_path, team_registry, pointed_at_team):
        from strata.api.routers.registry import registry_summary

        body = asyncio.run(registry_summary(_local_store(tmp_path), None))

        assert [row["name"] for row in body["names"]] == ["taxi/model"]

    def test_the_audit_is_the_teams(self, tmp_path, team_registry, pointed_at_team):
        from strata.api.routers.registry import registry_audit

        body = asyncio.run(registry_audit(_local_store(tmp_path), None))

        assert any(entry.get("name") == "taxi/model" for entry in body["entries"])

    def test_pending_changes_come_from_the_team_store(
        self, tmp_path, team_dir, team_registry, pointed_at_team
    ):
        from strata.api.routers.registry import registry_pending

        team_registry.request_alias_change(
            "taxi/model", "champion", "set", artifact_id="shared-model", version=1, actor="alice"
        )

        body = asyncio.run(registry_pending(_local_store(tmp_path), None))

        assert [p["name"] for p in body["pending"]] == ["taxi/model"]

    def test_approving_moves_the_alias_in_the_team_store(
        self, tmp_path, team_dir, team_registry, pointed_at_team
    ):
        """The decision has to land where the alias lives — approving into the
        local store would report success and change nothing anyone reads."""
        from strata.api.routers.registry import PendingDecisionRequest, approve_pending

        team_registry.request_alias_change(
            "taxi/model", "champion", "set", artifact_id="shared-model", version=1, actor="alice"
        )

        result = asyncio.run(
            approve_pending(
                PendingDecisionRequest(name="taxi/model", alias="champion"),
                (None, _local_store(tmp_path)),
            )
        )

        assert result["status"] == "approved"
        assert ArtifactStore(team_dir).list_pending_changes() == []

    def test_a_store_that_cannot_be_reached_is_a_bad_gateway(self, tmp_path, monkeypatch):
        """Not a 500, and not a silently empty registry: an empty dashboard
        reads as "nobody has published anything", which is a different fact."""
        import strata.api.routers.registry as registry_router

        monkeypatch.setattr(registry_router, "remote_registry", lambda: ("http://127.0.0.1:1", {}))

        with pytest.raises(HTTPException) as caught:
            asyncio.run(registry_router.registry_summary(_local_store(tmp_path), None))

        assert caught.value.status_code == 502

    def test_without_a_team_store_the_local_one_answers(self, tmp_path):
        from strata.api.routers.registry import registry_summary

        body = asyncio.run(registry_summary(_local_store(tmp_path), None))

        assert [row["name"] for row in body["names"]] == ["scratch/private"]


class TestPerCellStrip:
    def _session(self, cell_ids):
        from types import SimpleNamespace

        return SimpleNamespace(
            notebook_state=SimpleNamespace(cells=[SimpleNamespace(id=c) for c in cell_ids])
        )

    def test_the_strip_shows_what_the_cell_published_to_the_team(
        self, tmp_path, team_registry, pointed_at_team, monkeypatch
    ):
        import strata.server as server_module
        from strata.notebook.routes import list_notebook_published_artifacts

        # A local store with its own answer for the same cell, so reading the
        # wrong one is visible rather than indistinguishable.
        local = ArtifactStore(tmp_path / "local")
        local.create_artifact("local-only", "cc" * 32)
        local.finalize_artifact("local-only", 1, '{"fields": []}', 1, 8)
        local.set_name("scratch/local", "local-only", 1)
        local.set_tag("local-only", 1, "nb_cell", "c1")
        # Only on this thread, for the same reason the fixture above is: the
        # team store serves on uvicorn's and must keep its own store.
        real = server_module._get_artifact_store
        caller = threading.get_ident()
        monkeypatch.setattr(
            server_module,
            "_get_artifact_store",
            lambda **kw: local if threading.get_ident() == caller else real(**kw),
        )

        body = asyncio.run(list_notebook_published_artifacts("nb", self._session(["c1", "c2"])))

        assert list(body["cells"]) == ["c1"]
        entry = body["cells"]["c1"][0]
        assert entry["names"] == ["taxi/model"]
        assert entry["tags"] == {"stage": "candidate"}
        # The stamp is how it was found, not something to show back.
        assert "nb_cell" not in entry["tags"]

    def test_a_whole_notebook_costs_one_request(self, team_registry, pointed_at_team, monkeypatch):
        """Not one per cell. A strip that asked per cell would put a round trip
        on every cell of a long notebook every time the panel refreshed."""
        import strata.api.remote_registry as remote
        from strata.notebook.routes import list_notebook_published_artifacts

        real = remote.forward
        calls = []

        async def _counting(*args, **kwargs):
            calls.append(kwargs.get("params"))
            return await real(*args, **kwargs)

        monkeypatch.setattr(remote, "forward", _counting)
        asyncio.run(
            list_notebook_published_artifacts("nb", self._session([f"c{i}" for i in range(12)]))
        )

        assert len(calls) == 1

    def test_a_stamp_from_a_deleted_cell_is_not_shown(self, team_registry, pointed_at_team):
        """The store keeps the artifact; this notebook has nowhere to put it."""
        from strata.notebook.routes import list_notebook_published_artifacts

        body = asyncio.run(list_notebook_published_artifacts("nb", self._session(["c9"])))

        assert body["cells"] == {}


class TestTargetResolution:
    """``remote_registry()`` — which store the routes above will ask."""

    def _config(self, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(config=SimpleNamespace(**kwargs))

    def test_none_without_a_team_store(self, monkeypatch):
        """The ordinary single-machine case: the local store *is* the registry
        the cells write to, so there is nothing to forward."""
        import strata.server as server_module
        from strata.api.remote_registry import remote_registry

        monkeypatch.setattr(server_module, "_state", self._config(notebook_remote_store_url=None))

        assert remote_registry() is None

    def test_the_url_and_its_auth_travel_together(self, monkeypatch):
        """The headers are why this is server-side. Handing them to the page so
        it could call the team store itself would put the proxy token in every
        user's devtools."""
        import strata.server as server_module
        from strata.api.remote_registry import remote_registry

        monkeypatch.setattr(
            server_module,
            "_state",
            self._config(
                notebook_remote_store_url="http://store.example/",
                notebook_remote_store_headers={"X-Strata-Principal": "alice"},
            ),
        )

        assert remote_registry() == (
            "http://store.example",
            {"X-Strata-Principal": "alice"},
        )


def test_a_remote_store_url_naming_this_server_is_refused():
    """It would make every registry read forward to itself and recurse.

    Refused at startup rather than at the first Registry tab, which is where
    it would otherwise surface — as a hang, with nothing naming the setting.
    """
    from pydantic import ValidationError

    from strata.config import StrataConfig

    with pytest.raises(ValidationError, match="is this server"):
        StrataConfig(host="127.0.0.1", port=8765, notebook_remote_store_url="http://127.0.0.1:8765")


class TestArtifactsByTag:
    """The store read behind the strip, on either machine."""

    def _store(self, tmp_path):
        store = ArtifactStore(tmp_path / "s")
        store.create_artifact("done", "d1" * 32)
        store.finalize_artifact("done", 1, '{"fields": []}', 1, 8)
        store.set_tag("done", 1, "nb_cell", "c1")
        # Stamped, then overtaken: a second version of the same id with the
        # same provenance supersedes v1, and the tag stays on v1.
        v1 = store.create_artifact("gone", "d2" * 32)
        store.finalize_artifact("gone", v1, '{"fields": []}', 1, 8)
        store.set_tag("gone", v1, "nb_cell", "c1")
        v2 = store.create_artifact("gone", "d2" * 32)
        store.finalize_artifact("gone", v2, '{"fields": []}', 1, 8)
        return store

    def test_a_version_that_has_been_overtaken_is_not_shown(self, tmp_path):
        """A tag stays on the version it was set on, so a strip that showed
        every stamped row would keep offering a superseded one as current.
        Longstanding strip behaviour, restated here because the read moved."""
        from strata.services.registry import registry_service

        rows = registry_service.artifacts_by_tag(self._store(tmp_path), "nb_cell", tenant=None)

        assert [r["artifact_id"] for r in rows] == ["done"]

    def test_one_key_lookup_reports_which_cell_each_row_came_from(self, tmp_path):
        """What lets the strip ask once for a whole notebook."""
        from strata.services.registry import registry_service

        store = self._store(tmp_path)
        store.create_artifact("other", "d3" * 32)
        store.finalize_artifact("other", 1, '{"fields": []}', 1, 8)
        store.set_tag("other", 1, "nb_cell", "c2")

        rows = registry_service.artifacts_by_tag(store, "nb_cell", tenant=None)

        assert {r["artifact_id"]: r["tag_value"] for r in rows} == {"done": "c1", "other": "c2"}

    def test_asking_for_one_value_narrows_to_it(self, tmp_path):
        from strata.services.registry import registry_service

        rows = registry_service.artifacts_by_tag(
            self._store(tmp_path), "nb_cell", "c9", tenant=None
        )

        assert rows == []
