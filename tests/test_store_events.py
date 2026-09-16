"""What changed on a store, in order, for something following it. Item 8.

Publishing and withdrawing wrote ``artifact_publications`` and nothing else,
and the registry audit could only be read newest first, so a platform that
wanted to know what happened re-read everything and diffed.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from strata.api.dependencies import current_principal, read_store
from strata.api.routers.registry import router
from strata.artifact_store import ArtifactStore
from strata.types import Principal


def _ready(store: ArtifactStore, artifact_id: str, tenant: str | None = None) -> None:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, pa.schema([("id", pa.int64())])) as writer:
        writer.write_batch(pa.RecordBatch.from_pydict({"id": [1]}))
    store.create_artifact(artifact_id, f"prov-{artifact_id}", tenant=tenant)
    store.write_blob(artifact_id, 1, sink.getvalue().to_pybytes())
    store.finalize_artifact(artifact_id, 1, "{}", 1, 10)


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


class TestTheSequence:
    def test_publish_withdraw_and_registry_moves_each_append_one_event(self, store):
        _ready(store, "fig")
        store.set_name("paper/fig", "fig", 1)
        publication = store.publish_artifact("fig", 1, published_by="ana")
        store.set_tag("fig", 1, "reviewed", "yes")
        assert store.revoke_publication(publication.token, actor="ben")

        events = store.read_events()

        assert [e["action"] for e in events] == ["name_set", "publish", "tag_set", "withdraw"]
        assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)
        publish, withdraw = events[1], events[3]
        assert (publish["artifact_id"], publish["to_version"]) == ("fig", 1)
        assert (publish["key"], publish["value"], publish["actor"]) == (
            "token",
            publication.token,
            "ana",
        )
        assert (withdraw["value"], withdraw["actor"]) == (publication.token, "ben")

    def test_what_changes_nothing_is_not_an_event(self, store):
        """Publishing a published version returns its grant; revoking a
        revoked one fails. Neither happened, so neither is recorded."""
        _ready(store, "fig")
        first = store.publish_artifact("fig", 1)
        again = store.publish_artifact("fig", 1)
        assert again.token == first.token
        assert store.revoke_publication(first.token)
        assert not store.revoke_publication(first.token)
        assert not store.revoke_publication("no-such-token")

        assert [e["action"] for e in store.read_events()] == ["publish", "withdraw"]

    def test_a_follower_paging_by_since_sees_each_event_once(self, store):
        for index in range(5):
            _ready(store, f"a{index}")
            store.set_name(f"n/{index}", f"a{index}", 1)

        seen, since = [], 0
        page = store.read_events(since=since, limit=2)
        while page:
            seen.extend(e["seq"] for e in page)
            since = page[-1]["seq"]
            # Something lands between two pages.
            if len(seen) == 2:
                _ready(store, "late")
                store.set_name("n/late", "late", 1)
            page = store.read_events(since=since, limit=2)

        assert len(seen) == 6
        assert len(set(seen)) == 6

    def test_a_tenant_reads_only_its_own(self, store):
        _ready(store, "acme-fig", tenant="acme")
        _ready(store, "globex-fig", tenant="globex")
        store.publish_artifact("acme-fig", 1, tenant="acme")
        store.publish_artifact("globex-fig", 1, tenant="globex")

        assert [e["artifact_id"] for e in store.read_events(tenant="acme")] == ["acme-fig"]
        assert {e["tenant"] for e in store.read_events()} == {"acme", "globex"}


class TestTheRoute:
    @staticmethod
    def _client(store: ArtifactStore, principal: Principal | None) -> TestClient:
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[read_store] = lambda: store
        app.dependency_overrides[current_principal] = lambda: principal
        return TestClient(app)

    def _seed(self, store):
        _ready(store, "acme-fig", tenant="acme")
        _ready(store, "globex-fig", tenant="globex")
        store.publish_artifact("acme-fig", 1, tenant="acme")
        store.publish_artifact("globex-fig", 1, tenant="globex")
        store.set_name("acme/fig", "acme-fig", 1, tenant="acme")

    def test_a_principal_follows_its_tenant_from_next(self, store):
        self._seed(store)
        client = self._client(store, Principal(id="ana", tenant="acme"))

        first = client.get("/v1/events", params={"limit": 1}).json()
        second = client.get("/v1/events", params={"since": first["next"]}).json()
        third = client.get("/v1/events", params={"since": second["next"]}).json()

        assert [e["action"] for e in first["events"]] == ["publish"]
        assert [e["action"] for e in second["events"]] == ["name_set"]
        assert all(e["tenant"] == "acme" for e in first["events"] + second["events"])
        # Caught up: nothing new, and the cursor stays where it was.
        assert third == {"events": [], "next": second["next"]}

    def test_admin_sees_every_tenant(self, store):
        self._seed(store)
        client = self._client(store, Principal(id="ops", scopes=frozenset({"admin:*"})))

        events = client.get("/v1/events").json()["events"]

        assert {e["tenant"] for e in events} == {"acme", "globex"}
