"""What changed on a store, in order, for something following it.

Without an event log a follower re-reads everything and diffs.
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
        # The id, never the token: the feed is read by followers, not only by the owner.
        assert (publish["key"], publish["value"], publish["actor"]) == (
            "token",
            publication.id,
            "ana",
        )
        assert (withdraw["value"], withdraw["actor"]) == (publication.id, "ben")

    def test_a_credit_change_is_an_event(self, store):
        """A DOI or author list arriving later is news to a follower mirroring the page."""
        _ready(store, "fig")
        publication = store.publish_artifact("fig", 1)
        store.update_publication_credits(
            publication.token, external_ids=[{"scheme": "doi", "value": "10.1/x"}], actor="ana"
        )

        credit = store.read_events()[-1]

        assert credit["action"] == "credit"
        assert (credit["artifact_id"], credit["to_version"]) == ("fig", 1)
        assert (credit["key"], credit["value"], credit["actor"]) == (
            "token",
            publication.id,
            "ana",
        )

    def test_a_credit_change_to_an_unknown_token_is_not_an_event(self, store):
        assert store.update_publication_credits("nope", authors=[]) is None
        assert store.read_events() == []

    def test_what_changes_nothing_is_not_an_event(self, store):
        """Republishing returns the existing grant and re-revoking fails; neither is recorded."""
        _ready(store, "fig")
        first = store.publish_artifact("fig", 1)
        again = store.publish_artifact("fig", 1)
        assert again.id == first.id
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


class TestProtectedAliasEvents:
    """A protected move is queued, then approved or rejected; each step is news."""

    @staticmethod
    def _alias_events(store, **kwargs):
        return [
            (e["action"], e["actor"], e["artifact_id"], e["to_version"])
            for e in store.read_events(**kwargs)
            if e["alias"] == "champion"
        ]

    def test_a_request_then_approval_appends_request_approval_and_move(self, store):
        _ready(store, "old")
        _ready(store, "new")
        store.set_alias("team/model", "champion", "old", 1)
        store.request_alias_change(
            "team/model", "champion", "set", artifact_id="new", version=1, actor="ana"
        )
        store.approve_alias_change("team/model", "champion", actor="ben")

        events = [e for e in store.read_events() if e["alias"] == "champion"]

        assert [(e["action"], e["actor"]) for e in events] == [
            ("alias_set", None),
            ("alias_request_set", "ana"),
            ("alias_approved", "ben"),
            ("alias_set", "ben"),
        ]
        request, approved, moved = events[1:]
        assert (request["artifact_id"], request["to_version"]) == ("new", 1)
        assert (approved["artifact_id"], approved["to_version"]) == ("new", 1)
        assert (moved["from_artifact_id"], moved["artifact_id"]) == ("old", "new")

    def test_a_request_then_rejection_appends_request_and_rejection_only(self, store):
        _ready(store, "new")
        store.request_alias_change(
            "team/model", "champion", "set", artifact_id="new", version=1, actor="ana"
        )
        store.reject_alias_change("team/model", "champion", actor="ben")

        assert self._alias_events(store) == [
            ("alias_request_set", "ana", "new", 1),
            ("alias_rejected", "ben", "new", 1),
        ]
        assert store.resolve_alias("team/model", "champion") is None

    def test_an_approved_delete_appends_request_approval_and_delete(self, store):
        _ready(store, "fig")
        store.set_alias("team/model", "champion", "fig", 1)
        store.request_alias_change("team/model", "champion", "delete", actor="ana")
        store.approve_alias_change("team/model", "champion", actor="ben")

        events = [e for e in store.read_events() if e["alias"] == "champion"][1:]

        assert [(e["action"], e["actor"]) for e in events] == [
            ("alias_request_delete", "ana"),
            ("alias_approved", "ben"),
            ("alias_delete", "ben"),
        ]
        assert (events[-1]["from_artifact_id"], events[-1]["from_version"]) == ("fig", 1)

    def test_a_refused_or_empty_decision_is_not_an_event(self, store):
        """Nothing changed, so a follower must see nothing."""
        _ready(store, "fig")
        _ready(store, "gone")
        store.set_alias("team/model", "champion", "fig", 1)
        baseline = len(store.read_events())

        # Already the live pointer: nothing queued.
        assert not store.request_alias_change(
            "team/model", "champion", "set", artifact_id="fig", version=1, actor="ana"
        )
        with pytest.raises(ValueError, match="No pending change"):
            store.approve_alias_change("team/model", "champion", actor="ben")
        with pytest.raises(ValueError, match="No pending change"):
            store.reject_alias_change("team/model", "champion", actor="ben")
        assert len(store.read_events()) == baseline

        store.request_alias_change(
            "team/model", "champion", "set", artifact_id="gone", version=1, actor="ana"
        )
        queued = len(store.read_events())
        with pytest.raises(ValueError, match="Separation of duty"):
            store.approve_alias_change(
                "team/model", "champion", actor="ana", require_distinct_approver=True
            )
        assert store.delete_artifact("gone", 1)
        after_delete = len(store.read_events())
        with pytest.raises(ValueError, match="no longer available"):
            store.approve_alias_change("team/model", "champion", actor="ben")

        assert queued == baseline + 1
        assert len(store.read_events()) == after_delete
        assert store.list_pending_changes()[0]["artifact_id"] == "gone"

    def test_a_tenant_follows_only_its_own_protected_moves(self, store):
        _ready(store, "acme-fig", tenant="acme")
        _ready(store, "globex-fig", tenant="globex")
        for tenant in ("acme", "globex"):
            store.request_alias_change(
                "team/model",
                "champion",
                "set",
                artifact_id=f"{tenant}-fig",
                version=1,
                tenant=tenant,
                actor="ana",
            )
            store.approve_alias_change("team/model", "champion", tenant=tenant, actor="ben")

        acme = self._alias_events(store, tenant="acme")

        assert [a for a, *_ in acme] == ["alias_request_set", "alias_approved", "alias_set"]
        assert {artifact for _, _, artifact, _ in acme} == {"acme-fig"}


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
