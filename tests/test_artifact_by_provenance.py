"""Lookup by provenance hash: the join key a shared store needs.

A hash is the only identifier two people compute independently and agree on. A miss is an ordinary
answer (404), and the hash is a lookup key, not a capability: team-b must not read team-a's bytes.
The tenant test uses the same computation on both sides, so the isolation is scoping, not two
different hashes.
"""

import json

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from strata_client.client import StrataClient

from strata.artifact_store import ArtifactStore
from tests.conftest import LIVE_SERVER_TIMEOUT, run_server_with_context, table_to_ipc_bytes

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

    ``PutArtifactResponse`` carries the URI, not the key. Reading real state avoids recomputing with
    a copy of the server's formula, which would pass even if both were wrong.
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
    """A shared store as deployed: service mode, trusted proxy, multi-tenant, write-back on."""
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

    found = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}", timeout=LIVE_SERVER_TIMEOUT
    )

    assert found.status_code == 200, found.text
    body = found.json()
    # The id is returned, not supplied: a caller with only a hash learns where the
    # result lives.
    assert (body["artifact_id"], body["version"]) == (artifact_id, version)
    assert body["state"] == "ready"
    assert body["row_count"] == 3


def test_a_hash_nobody_computed_is_a_miss_not_an_error(personal_server):
    """404 is the ordinary answer; callers branch on it every cell run."""
    response = httpx.get(
        f"{personal_server['base_url']}/v1/artifacts/by-provenance/{ABSENT_HASH}",
        timeout=LIVE_SERVER_TIMEOUT,
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "bad_hash",
    [
        "not-a-hash",
        "ABC" * 21 + "D",  # right length, uppercase; digests are lowercase hex
        "0" * 63,  # one short
        "0" * 65,  # one long
    ],
)
def test_a_malformed_hash_is_rejected_before_the_store(personal_server, bad_hash):
    """The key reaches SQLite, so the route validates its shape at the boundary."""
    response = httpx.get(
        f"{personal_server['base_url']}/v1/artifacts/by-provenance/{bad_hash}",
        timeout=LIVE_SERVER_TIMEOUT,
    )

    assert response.status_code == 422


def test_the_match_carries_enough_to_fetch_the_bytes(team_server):
    """The team hit end to end: bob has only the hash, no write scope and no artifact id."""
    base_url = team_server["base_url"]
    dataset = pa.table({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    uri = _publish(base_url, dataset, _headers("team-a", "alice", scopes="artifacts:write"))
    provenance = _provenance_of(team_server["artifact_dir"], uri)

    bob = _headers("team-a", "bob")
    found = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}",
        headers=bob,
        timeout=LIVE_SERVER_TIMEOUT,
    )
    assert found.status_code == 200, found.text
    # Attribution: an artifact that appears with no author is indistinguishable
    # from a bug, so the store says who computed it.
    assert found.json()["principal"] == "alice"

    artifact_id, version = found.json()["artifact_id"], found.json()["version"]
    data = httpx.get(
        f"{base_url}/v1/artifacts/{artifact_id}/v/{version}/data",
        headers=bob,
        timeout=LIVE_SERVER_TIMEOUT,
    )
    assert data.status_code == 200
    assert ipc.open_stream(data.content).read_all().equals(dataset)


def test_another_teams_identical_computation_is_invisible(team_server):
    """Both teams run the same computation and get the same hash, so isolation is not an accident of
    different keys. Team-b misses on team-a's result, then hits its own.
    """
    base_url = team_server["base_url"]
    artifact_dir = team_server["artifact_dir"]
    dataset = pa.table({"id": [1, 2, 3]})

    a_uri = _publish(base_url, dataset, _headers("team-a", "alice", scopes="artifacts:write"))
    provenance = _provenance_of(artifact_dir, a_uri)

    carol = _headers("team-b", "carol", scopes="artifacts:write")
    assert (
        httpx.get(
            f"{base_url}/v1/artifacts/by-provenance/{provenance}",
            headers=carol,
            timeout=LIVE_SERVER_TIMEOUT,
        ).status_code
        == 404
    )

    b_uri = _publish(base_url, dataset, carol)
    assert _provenance_of(artifact_dir, b_uri) == provenance, (
        "the two teams must share a hash for this test to mean anything"
    )
    assert b_uri != a_uri

    hit = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}",
        headers=carol,
        timeout=LIVE_SERVER_TIMEOUT,
    )
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


@pytest.mark.parametrize("finished", [True, False])
def test_a_named_id_is_held_by_its_tenant_from_its_first_upload(team_server, finished):
    """Another team cannot append a version to it, even while its first upload is in flight."""
    store = ArtifactStore(team_server["artifact_dir"])
    version = store.create_artifact("nb_x", "a" * 64, tenant="team-a")
    if finished:
        store.write_blob("nb_x", version, b"payload")
        store.finalize_artifact("nb_x", version, "", row_count=None, byte_size=7)

    response = httpx.put(
        f"{team_server['base_url']}/v1/artifacts/by-provenance/{'b' * 64}",
        files={
            "metadata": (
                "metadata.json",
                json.dumps({"content_type": "pickle/object", "artifact_id": "nb_x"}),
                "application/json",
            ),
            "data": ("data.bin", b"other", "application/octet-stream"),
        },
        headers=_headers("team-b", "carol", scopes="artifacts:write"),
        timeout=30.0,
    )

    assert response.status_code == 409, response.text
    assert store.get_artifact("nb_x", version + 1) is None


def test_another_teams_upload_landing_mid_request_still_holds_the_id(team_server, monkeypatch):
    """Team-b's first version of the id commits after team-a's route found the id unused."""
    other = ArtifactStore(team_server["artifact_dir"])
    create = ArtifactStore.create_artifact

    def team_b_lands_first(self, artifact_id, *args, **kwargs):
        if artifact_id == "nb_race" and kwargs.get("tenant") == "team-a":
            create(other, "nb_race", "a" * 64, tenant="team-b")
        return create(self, artifact_id, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "create_artifact", team_b_lands_first)

    response = httpx.put(
        f"{team_server['base_url']}/v1/artifacts/by-provenance/{'b' * 64}",
        files={
            "metadata": (
                "metadata.json",
                json.dumps({"content_type": "pickle/object", "artifact_id": "nb_race"}),
                "application/json",
            ),
            "data": ("data.bin", b"other", "application/octet-stream"),
        },
        headers=_headers("team-a", "alice", scopes="artifacts:write"),
        timeout=30.0,
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "nb_race already exists under another tenant"
    assert other.get_artifact("nb_race", 1).tenant == "team-b"
    assert other.get_artifact("nb_race", 2) is None


def test_a_caller_computed_key_round_trips_with_opaque_bytes(personal_server):
    """Non-Arrow bytes (a pickle) must survive; only the notebook's serializer knows the format."""
    base_url = personal_server["base_url"]
    provenance = "c" * 64
    blob = b"\x80\x05\x95not-arrow-at-all"

    stored = _publish_by_provenance(base_url, provenance, blob)
    assert stored.status_code == 200, stored.text
    assert stored.json()["hit"] is False

    found = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}", timeout=LIVE_SERVER_TIMEOUT
    )
    assert found.status_code == 200
    assert found.json()["content_type"] == "pickle/object"

    artifact_id, version = found.json()["artifact_id"], found.json()["version"]
    data = httpx.get(
        f"{base_url}/v1/artifacts/{artifact_id}/v/{version}/data", timeout=LIVE_SERVER_TIMEOUT
    )
    assert data.status_code == 200
    assert data.content == blob


def test_an_arrow_result_published_without_a_row_count_verifies_clean(personal_server):
    """Neither the notebook nor the client sends a row count; recording 0 made verify flag
    every Arrow result a team store holds."""
    base_url = personal_server["base_url"]
    blob = table_to_ipc_bytes(pa.table({"x": [1, 2, 3]}))
    stored = _publish_by_provenance(base_url, "4" * 64, blob, content_type="arrow/ipc")
    assert stored.status_code == 200, stored.text

    store = ArtifactStore(personal_server["artifact_dir"])
    artifact_id, version = _ref(stored.json()["artifact_uri"])
    assert store.get_artifact(artifact_id, version).row_count is None
    assert store.verify_artifacts() == []


def test_a_malformed_row_count_is_refused(personal_server):
    response = httpx.put(
        f"{personal_server['base_url']}/v1/artifacts/by-provenance/{'3' * 64}",
        files={
            "metadata": (
                "metadata.json",
                json.dumps({"content_type": "arrow/ipc", "row_count": "3"}),
                "application/json",
            ),
            "data": ("data.bin", b"bytes", "application/octet-stream"),
        },
        timeout=30.0,
    )
    assert response.status_code == 400
    assert "row_count" in response.text


def test_the_first_writer_of_a_key_wins(personal_server):
    """A shared cache key is not reassignable by whoever writes last.

    This turns "poison the team's cache" into "race to be first". ``finalize_artifact`` collapses
    duplicates anyway; the version count asserts the route's check keeps the loser off disk.
    """
    base_url = personal_server["base_url"]
    provenance = "d" * 64

    first = _publish_by_provenance(base_url, provenance, b"the original")
    assert first.json()["hit"] is False

    second = _publish_by_provenance(base_url, provenance, b"a replacement")
    assert second.status_code == 200
    assert second.json()["hit"] is True
    assert second.json()["artifact_uri"] == first.json()["artifact_uri"]

    found = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}", timeout=LIVE_SERVER_TIMEOUT
    ).json()
    data = httpx.get(
        f"{base_url}/v1/artifacts/{found['artifact_id']}/v/{found['version']}/data",
        timeout=LIVE_SERVER_TIMEOUT,
    )
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
    """The read side's tenant guarantee, on the path that creates the data."""
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
        httpx.get(
            f"{base_url}/v1/artifacts/by-provenance/{provenance}",
            headers=carol,
            timeout=LIVE_SERVER_TIMEOUT,
        ).status_code
        == 404
    )


def test_a_publish_without_a_content_type_is_refused(personal_server):
    """Readers have no other source for the content type; without it the artifact is undecodable."""
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
    """``admin:*`` widens reads by id; it must not narrow this one.

    ``CurrentTenant`` is None for an admin ("do not filter"), but
    ``find_by_provenance(tenant=None)`` means the tenantless namespace, so an admin missed on its
    own results.
    """
    base_url = team_server["base_url"]
    admin = _headers("team-a", "root", scopes="admin:*")
    uri = _publish(base_url, pa.table({"id": [1, 2, 3]}), admin)
    provenance = _provenance_of(team_server["artifact_dir"], uri)

    # The admin can already read it by id, so a miss below is this route's scoping
    # being wrong, not the artifact being unreachable.
    artifact_id, version = _ref(uri)
    assert (
        httpx.get(
            f"{base_url}/v1/artifacts/{artifact_id}/v/{version}",
            headers=admin,
            timeout=LIVE_SERVER_TIMEOUT,
        ).status_code
        == 200
    )

    found = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}",
        headers=admin,
        timeout=LIVE_SERVER_TIMEOUT,
    )
    assert found.status_code == 200, found.text
    assert found.json()["artifact_id"] == artifact_id


def test_a_miss_is_marked_so_it_cannot_be_confused_with_a_broken_store(personal_server):
    """An old server or a gateway without a store also 404s; an unmarked 404 would read as a miss
    and recompute forever.
    """
    base_url = personal_server["base_url"]

    miss = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance/{ABSENT_HASH}", timeout=LIVE_SERVER_TIMEOUT
    )
    assert miss.status_code == 404
    assert miss.headers.get("X-Strata-Provenance-Miss") == "1"

    # The stand-in for every other 404 the same client can receive: a path
    # this server does not serve, exactly as an older deployment would answer.
    unknown_route = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance-typo/{ABSENT_HASH}", timeout=LIVE_SERVER_TIMEOUT
    )
    assert unknown_route.status_code == 404
    assert "X-Strata-Provenance-Miss" not in unknown_route.headers


def test_the_client_raises_rather_than_reporting_a_miss_it_cannot_verify():
    """An unmarked 404 must raise, not become None, or an unanswered question recomputes forever."""
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
    """The executor's lookup runs inside an async cell run, where the sync client would block the
    loop.
    """
    from strata_client.client import AsyncStrataClient

    uri = _publish(personal_server["base_url"], pa.table({"id": [1]}))
    provenance = _provenance_of(personal_server["artifact_dir"], uri)

    async with AsyncStrataClient(base_url=personal_server["base_url"]) as client:
        assert await client.find_by_provenance(ABSENT_HASH) is None

        found = await client.find_by_provenance(provenance)
        assert found is not None
        assert found["artifact_id"] == _ref(uri)[0]


def test_the_client_returns_none_for_a_miss(personal_server):
    """A miss is a value, not an exception: the caller is on the hot "should I run this?" path."""
    with StrataClient(base_url=personal_server["base_url"]) as client:
        uri = _publish(personal_server["base_url"], pa.table({"id": [1]}))
        provenance = _provenance_of(personal_server["artifact_dir"], uri)

        assert client.find_by_provenance(ABSENT_HASH) is None

        found = client.find_by_provenance(provenance)
        assert found is not None
        assert found["artifact_id"] == _ref(uri)[0]
        assert found["provenance_hash"] == provenance


def test_the_build_environment_travels_with_the_result(personal_server):
    """The provenance key covers the lockfile, not the platform, so the store records and returns
    where a result was produced.
    """
    base_url = personal_server["base_url"]
    provenance = "b" * 64
    response = httpx.put(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}",
        files={
            "metadata": (
                "metadata.json",
                json.dumps(
                    {
                        "content_type": "arrow/ipc",
                        "build_env": "cpython-3.12-linux-x86_64",
                    }
                ),
                "application/json",
            ),
            "data": ("data.bin", b"bytes", "application/octet-stream"),
        },
        timeout=30.0,
    )
    assert response.status_code == 200, response.text

    found = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}", timeout=LIVE_SERVER_TIMEOUT
    ).json()
    assert found["build_env"] == "cpython-3.12-linux-x86_64"
    # Distinct fields, not one read twice: a shared param reader that ignored its key
    # argument would fail here.
    assert found["content_type"] == "arrow/ipc"


def test_an_artifact_stored_without_a_platform_reports_an_empty_one(personal_server):
    """Older artifacts and core transforms have none; a plausible default would be a fabricated
    claim.
    """
    base_url = personal_server["base_url"]
    provenance = "7" * 64
    _publish_by_provenance(base_url, provenance, b"x")

    found = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}", timeout=LIVE_SERVER_TIMEOUT
    ).json()
    assert found["build_env"] == ""


def test_the_environment_identity_round_trips(personal_server):
    """Both halves: which package set (env_hash) and on what (build_env)."""
    base_url = personal_server["base_url"]
    provenance = "5" * 64
    response = httpx.put(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}",
        files={
            "metadata": (
                "metadata.json",
                json.dumps(
                    {
                        "content_type": "arrow/ipc",
                        "build_env": "cpython-3.13-linux-aarch64",
                        "env_hash": "e" * 64,
                    }
                ),
                "application/json",
            ),
            "data": ("data.bin", b"bytes", "application/octet-stream"),
        },
        timeout=30.0,
    )
    assert response.status_code == 200, response.text

    found = httpx.get(
        f"{base_url}/v1/artifacts/by-provenance/{provenance}", timeout=LIVE_SERVER_TIMEOUT
    ).json()
    assert found["env_hash"] == "e" * 64
    assert found["build_env"] == "cpython-3.13-linux-aarch64"
