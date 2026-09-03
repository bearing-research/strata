"""Lookup by provenance hash — the primitive a shared store exists for.

Every other artifact read starts from an id somebody already holds. This one
starts from a hash, which is the only identifier two people compute
independently and arrive at the same value for. Without it a "team cache" has
no join key: a colleague's copy of the identical computation lives at an
artifact id nobody else would guess.

So the tests here are about the two things that make it safe to expose:

* a miss is an ordinary answer (404), not a failure, and
* the hash is a *lookup key*, not a capability — knowing team-a's hash must not
  read team-a's bytes from team-b.

The tenant test deliberately uses the **same** computation on both sides, so
the isolation being asserted is scoping and not merely two different hashes.
"""

import json

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from strata_client.client import StrataClient

from strata.artifact_store import ArtifactStore
from tests.conftest import run_server_with_context, table_to_ipc_bytes

PROXY_TOKEN = "by-provenance-token"
# Well-formed but never computed: the shape passes the route's pattern, so a
# 404 here proves the *store* found nothing rather than the router rejecting it.
ABSENT_HASH = "0" * 64


def _headers(tenant: str, principal: str, scopes: str | None = None) -> dict:
    headers = {
        "X-Strata-Proxy-Token": PROXY_TOKEN,
        "X-Strata-Principal": principal,
        "X-Tenant-ID": tenant,
    }
    if scopes:
        headers["X-Strata-Scopes"] = scopes
    return headers


def _publish(base_url: str, table: pa.Table, headers: dict | None = None) -> str:
    """Persist a locally computed table, returning its artifact URI."""
    metadata = {
        "inputs": [],
        "transform": {"executor": "researcher_local@v1", "params": {}},
    }
    files = {
        "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        "data": ("data.arrow", table_to_ipc_bytes(table), "application/vnd.apache.arrow.stream"),
    }
    response = httpx.put(
        f"{base_url}/v1/artifacts", files=files, headers=headers or {}, timeout=30.0
    )
    assert response.status_code == 200, response.text
    return response.json()["artifact_uri"]


def _ref(artifact_uri: str) -> tuple[str, int]:
    artifact_id, version = artifact_uri.removeprefix("strata://artifact/").split("@v=")
    return artifact_id, int(version)


def _provenance_of(artifact_dir, artifact_uri: str) -> str:
    """Read a published artifact's provenance hash off disk.

    Nothing in the HTTP surface hands a publisher its own provenance hash back
    — ``PutArtifactResponse`` carries the URI, not the key — so the test opens
    the store the server just wrote to. That reads real state rather than
    recomputing the hash with a copy of the server's formula, which would pass
    even if both were wrong together.
    """
    artifact_id, version = _ref(artifact_uri)
    stored = ArtifactStore(artifact_dir).get_artifact(artifact_id, version)
    assert stored is not None, f"{artifact_uri} was not written to {artifact_dir}"
    return stored.provenance_hash


@pytest.fixture
def personal_server(tmp_path):
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()
    with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
        yield {"base_url": ctx.base_url, "artifact_dir": artifact_dir}


@pytest.fixture
def team_server(tmp_path):
    """A shared store as it would actually be deployed: service mode, behind a
    trusted proxy, multi-tenant, with authenticated write-back enabled."""
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()
    with run_server_with_context(
        cache_dir,
        artifact_dir,
        "service",
        auth_mode="trusted_proxy",
        proxy_token=PROXY_TOKEN,
        multi_tenant_enabled=True,
        service_writes_enabled=True,
        hide_forbidden_as_not_found=True,
    ) as ctx:
        yield {"base_url": ctx.base_url, "artifact_dir": artifact_dir}


def test_a_stored_result_is_findable_by_its_provenance_hash(personal_server):
    base_url = personal_server["base_url"]
    uri = _publish(base_url, pa.table({"id": [1, 2, 3]}))
    artifact_id, version = _ref(uri)
    provenance = _provenance_of(personal_server["artifact_dir"], uri)

    found = httpx.get(f"{base_url}/v1/artifacts/by-provenance/{provenance}")

    assert found.status_code == 200, found.text
    body = found.json()
    # The id is returned, not supplied — that is the whole point. A caller with
    # only a hash learns where the result lives.
    assert (body["artifact_id"], body["version"]) == (artifact_id, version)
    assert body["state"] == "ready"
    assert body["row_count"] == 3


def test_a_hash_nobody_computed_is_a_miss_not_an_error(personal_server):
    """404 is the ordinary answer. Callers branch on it every time they run a
    cell, so it must not be an exceptional path."""
    response = httpx.get(f"{personal_server['base_url']}/v1/artifacts/by-provenance/{ABSENT_HASH}")

    assert response.status_code == 404


@pytest.mark.parametrize(
    "bad_hash",
    [
        "not-a-hash",
        "ABC" * 21 + "D",  # right length, uppercase — digests are lowercase hex
        "0" * 63,  # one short
        "0" * 65,  # one long
    ],
)
def test_a_malformed_hash_is_rejected_before_the_store(personal_server, bad_hash):
    """The lookup key reaches SQLite, so its shape is the route's business.
    Rejecting at the boundary keeps "what can be asked of the store" a property
    of the route rather than of every handler that might grow one."""
    response = httpx.get(f"{personal_server['base_url']}/v1/artifacts/by-provenance/{bad_hash}")

    assert response.status_code == 422


def test_the_match_carries_enough_to_fetch_the_bytes(team_server):
    """The team hit, end to end: alice computes, bob has only the hash.

    Bob holds no write scope and never saw alice's artifact id — everything he
    needs comes back from the lookup.
    """
    base_url = team_server["base_url"]
    dataset = pa.table({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    uri = _publish(base_url, dataset, _headers("team-a", "alice", scopes="artifacts:write"))
    provenance = _provenance_of(team_server["artifact_dir"], uri)

    bob = _headers("team-a", "bob")
    found = httpx.get(f"{base_url}/v1/artifacts/by-provenance/{provenance}", headers=bob)
    assert found.status_code == 200, found.text
    # Attribution: an artifact that appears with no author is indistinguishable
    # from a bug, so the store says who computed it.
    assert found.json()["principal"] == "alice"

    artifact_id, version = found.json()["artifact_id"], found.json()["version"]
    data = httpx.get(f"{base_url}/v1/artifacts/{artifact_id}/v/{version}/data", headers=bob)
    assert data.status_code == 200
    assert ipc.open_stream(data.content).read_all().equals(dataset)


def test_another_teams_identical_computation_is_invisible(team_server):
    """Both teams run the *same* computation, so both arrive at the same hash.

    That is the case worth testing: isolation here cannot be an accident of
    two different keys. Team-b must miss on team-a's result and then, once it
    computes the same thing itself, hit its *own*.
    """
    base_url = team_server["base_url"]
    artifact_dir = team_server["artifact_dir"]
    dataset = pa.table({"id": [1, 2, 3]})

    a_uri = _publish(base_url, dataset, _headers("team-a", "alice", scopes="artifacts:write"))
    provenance = _provenance_of(artifact_dir, a_uri)

    carol = _headers("team-b", "carol", scopes="artifacts:write")
    assert (
        httpx.get(f"{base_url}/v1/artifacts/by-provenance/{provenance}", headers=carol).status_code
        == 404
    )

    b_uri = _publish(base_url, dataset, carol)
    assert _provenance_of(artifact_dir, b_uri) == provenance, (
        "the two teams must share a hash for this test to mean anything"
    )
    assert b_uri != a_uri

    hit = httpx.get(f"{base_url}/v1/artifacts/by-provenance/{provenance}", headers=carol)
    assert hit.status_code == 200
    assert hit.json()["artifact_id"] == _ref(b_uri)[0]
    assert hit.json()["principal"] == "carol"


def _publish_by_provenance(
    base_url: str,
    provenance: str,
    blob: bytes,
    *,
    content_type: str = "pickle/object",
    headers: dict | None = None,
) -> httpx.Response:
    return httpx.put(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}",
        files={
            "metadata": (
                "metadata.json",
                json.dumps({"content_type": content_type, "variable_name": "model"}),
                "application/json",
            ),
            "data": ("data.bin", blob, "application/octet-stream"),
        },
        headers=headers or {},
        timeout=30.0,
    )


def test_a_caller_computed_key_round_trips_with_opaque_bytes(personal_server):
    """The write half. Bytes that are not Arrow — a pickle here — must survive,
    because a cell variable is Arrow, JSON, or a pickle and only the notebook's
    serializer knows which."""
    base_url = personal_server["base_url"]
    provenance = "c" * 64
    blob = b"\x80\x05\x95not-arrow-at-all"

    stored = _publish_by_provenance(base_url, provenance, blob)
    assert stored.status_code == 200, stored.text
    assert stored.json()["hit"] is False

    found = httpx.get(f"{base_url}/v1/artifacts/by-provenance/{provenance}")
    assert found.status_code == 200
    assert found.json()["content_type"] == "pickle/object"

    artifact_id, version = found.json()["artifact_id"], found.json()["version"]
    data = httpx.get(f"{base_url}/v1/artifacts/{artifact_id}/v/{version}/data")
    assert data.status_code == 200
    assert data.content == blob


def test_the_first_writer_of_a_key_wins(personal_server):
    """A shared cache key must not be reassignable by whoever writes last.

    This is most of what makes accepting a caller-computed key tolerable: it
    turns "poison the team's cache" into "race to be first", and a team that
    shares a cache already runs each other's code.

    The *outcome* is also guaranteed one layer down — ``finalize_artifact``
    collapses a duplicate provenance whatever the route does. What the route's
    own check adds is that the loser never reaches the disk at all: no blob
    written, no failed row left behind. The version count is what asserts that
    part, so it is not merely re-testing the store.
    """
    base_url = personal_server["base_url"]
    provenance = "d" * 64

    first = _publish_by_provenance(base_url, provenance, b"the original")
    assert first.json()["hit"] is False

    second = _publish_by_provenance(base_url, provenance, b"a replacement")
    assert second.status_code == 200
    assert second.json()["hit"] is True
    assert second.json()["artifact_uri"] == first.json()["artifact_uri"]

    found = httpx.get(f"{base_url}/v1/artifacts/by-provenance/{provenance}").json()
    data = httpx.get(f"{base_url}/v1/artifacts/{found['artifact_id']}/v/{found['version']}/data")
    assert data.content == b"the original"

    stored = ArtifactStore(personal_server["artifact_dir"])
    versions = [
        artifact
        for artifact in stored.list_artifacts(limit=100)
        if artifact.provenance_hash == provenance
    ]
    assert len(versions) == 1, "the rejected publish left a row behind"


def test_publishing_needs_the_write_scope(team_server):
    """Reading the team cache and contributing to it are different rights."""
    base_url = team_server["base_url"]
    provenance = "e" * 64

    denied = _publish_by_provenance(base_url, provenance, b"x", headers=_headers("team-a", "bob"))
    assert denied.status_code == 403

    allowed = _publish_by_provenance(
        base_url,
        provenance,
        b"x",
        headers=_headers("team-a", "alice", scopes="artifacts:write"),
    )
    assert allowed.status_code == 200, allowed.text


def test_a_published_key_is_only_visible_to_its_own_team(team_server):
    """The same guarantee as the read side, on the path that creates the data."""
    base_url = team_server["base_url"]
    provenance = "f" * 64

    _publish_by_provenance(
        base_url,
        provenance,
        b"team-a only",
        headers=_headers("team-a", "alice", scopes="artifacts:write"),
    )

    carol = _headers("team-b", "carol")
    assert (
        httpx.get(f"{base_url}/v1/artifacts/by-provenance/{provenance}", headers=carol).status_code
        == 404
    )


def test_a_publish_without_a_content_type_is_refused(personal_server):
    """The reader has no other source for it, so an artifact stored without one
    is one nobody can decode — better refused at the door than discovered on a
    pull months later."""
    response = httpx.put(
        f"{personal_server['base_url']}/v1/artifacts/by-provenance/{'a' * 64}",
        files={
            "metadata": ("metadata.json", json.dumps({"variable_name": "x"}), "application/json"),
            "data": ("data.bin", b"bytes", "application/octet-stream"),
        },
        timeout=30.0,
    )
    assert response.status_code == 400
    assert "content_type" in response.text


def test_the_client_publishes_and_finds_its_own_key(personal_server):
    with StrataClient(base_url=personal_server["base_url"]) as client:
        provenance = "9" * 64
        stored = client.put_by_provenance(
            provenance, b"opaque", content_type="json/object", variable_name="v"
        )
        assert stored["hit"] is False

        found = client.find_by_provenance(provenance)
        assert found is not None
        assert found["content_type"] == "json/object"


def test_an_admin_hits_on_what_it_just_published(team_server):
    """``admin:*`` widens reads by id; it must not *narrow* this one.

    ``CurrentTenant`` yields None for an admin meaning "do not filter", while
    ``find_by_provenance(tenant=None)`` means the tenantless namespace
    specifically. Passing one into the other made an admin miss on results it
    had published seconds earlier and silently recompute them.
    """
    base_url = team_server["base_url"]
    admin = _headers("team-a", "root", scopes="admin:*")
    uri = _publish(base_url, pa.table({"id": [1, 2, 3]}), admin)
    provenance = _provenance_of(team_server["artifact_dir"], uri)

    # By id the admin can already read it — so a miss below is this route's
    # scoping being wrong, not the artifact being unreachable.
    artifact_id, version = _ref(uri)
    assert (
        httpx.get(f"{base_url}/v1/artifacts/{artifact_id}/v/{version}", headers=admin).status_code
        == 200
    )

    found = httpx.get(f"{base_url}/v1/artifacts/by-provenance/{provenance}", headers=admin)
    assert found.status_code == 200, found.text
    assert found.json()["artifact_id"] == artifact_id


def test_a_miss_is_marked_so_it_cannot_be_confused_with_a_broken_store(personal_server):
    """A 404 alone is ambiguous: an old server 404s the unknown path, and a
    gateway with no artifact store 404s with its own message. Both would read
    as "nobody computed it" and recompute forever, looking healthy throughout.
    """
    base_url = personal_server["base_url"]

    miss = httpx.get(f"{base_url}/v1/artifacts/by-provenance/{ABSENT_HASH}")
    assert miss.status_code == 404
    assert miss.headers.get("X-Strata-Provenance-Miss") == "1"

    # The stand-in for every other 404 the same client can receive: a path
    # this server does not serve, exactly as an older deployment would answer.
    unknown_route = httpx.get(f"{base_url}/v1/artifacts/by-provenance-typo/{ABSENT_HASH}")
    assert unknown_route.status_code == 404
    assert "X-Strata-Provenance-Miss" not in unknown_route.headers


def test_the_client_raises_rather_than_reporting_a_miss_it_cannot_verify():
    """An *unmarked* 404 must surface, not become None.

    This is what an older store answers for a route it does not serve, and
    what a service-mode gateway with no artifact store answers with its own
    message. Reporting "no result" for a question that was never answered is
    the failure mode that recomputes forever without ever looking wrong.
    """
    unmarked_404 = httpx.MockTransport(
        lambda request: httpx.Response(404, json={"detail": "Not Found"})
    )
    with StrataClient.from_transport(unmarked_404) as client:
        with pytest.raises(httpx.HTTPStatusError):
            client.find_by_provenance(ABSENT_HASH)

    marked_404 = httpx.MockTransport(
        lambda request: httpx.Response(
            404,
            json={"detail": "No artifact has been computed for that provenance hash"},
            headers={"X-Strata-Provenance-Miss": "1"},
        )
    )
    with StrataClient.from_transport(marked_404) as client:
        assert client.find_by_provenance(ABSENT_HASH) is None


async def test_the_async_client_answers_the_same_way(personal_server):
    """The executor's lookup sits inside an already-async cell run, so the
    sync client would block the event loop once per cell."""
    from strata_client.client import AsyncStrataClient

    uri = _publish(personal_server["base_url"], pa.table({"id": [1]}))
    provenance = _provenance_of(personal_server["artifact_dir"], uri)

    async with AsyncStrataClient(base_url=personal_server["base_url"]) as client:
        assert await client.find_by_provenance(ABSENT_HASH) is None

        found = await client.find_by_provenance(provenance)
        assert found is not None
        assert found["artifact_id"] == _ref(uri)[0]


def test_the_client_returns_none_for_a_miss(personal_server):
    """A miss is a value, not an exception — the caller is on the hot path of
    "should I run this?" and would otherwise wrap every call in try/except."""
    with StrataClient(base_url=personal_server["base_url"]) as client:
        uri = _publish(personal_server["base_url"], pa.table({"id": [1]}))
        provenance = _provenance_of(personal_server["artifact_dir"], uri)

        assert client.find_by_provenance(ABSENT_HASH) is None

        found = client.find_by_provenance(provenance)
        assert found is not None
        assert found["artifact_id"] == _ref(uri)[0]
        assert found["provenance_hash"] == provenance
