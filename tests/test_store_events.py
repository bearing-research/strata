"""What changed on a store, in order, for something following it.

Without an event log a follower re-reads everything and diffs.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vestigo.store import ArtifactStore

from strata.api.dependencies import current_principal, read_store
from strata.api.routers.registry import router
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

    def test_the_audit_limit_is_bounded(self, store):
        """``limit=-1`` read the whole audit on SQLite and was a 500 on Postgres."""
        self._seed(store)
        client = self._client(store, None)

        assert client.get("/v1/registry/audit", params={"limit": -1}).status_code == 422
        assert client.get("/v1/registry/audit", params={"limit": 1001}).status_code == 422
        assert len(client.get("/v1/registry/audit", params={"limit": 1}).json()["entries"]) == 1

    def test_admin_summary_lists_every_tenants_names(self, store):
        self._seed(store)
        client = self._client(store, Principal(id="ops", scopes=frozenset({"admin:*"})))

        names = client.get("/v1/registry/summary").json()["names"]

        assert [n["name"] for n in names] == ["acme/fig"]

    def test_a_tenants_tag_lookup_names_its_artifacts(self, store):
        self._seed(store)
        store.set_tag("acme-fig", 1, "nb_cell", "c1", tenant="acme")
        client = self._client(store, Principal(id="ana", tenant="acme"))

        found = client.get("/v1/registry/artifacts", params={"tag_key": "nb_cell"}).json()

        assert [a["names"] for a in found["artifacts"]] == [["acme/fig"]]
