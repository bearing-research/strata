"""The team cache tier: a colleague's result instead of your recomputation.

The claim being tested is narrow and load-bearing: a pull must leave the local
store in *exactly* the state a local run would have. The executor's cache-hit
check does not trust ``find_by_provenance`` on its own — it re-reads each
consumed variable's canonical artifact and compares its provenance — so a pull
that writes anything less specific than that is a pull that still misses.

The other half is that none of this can break a cell. A store that is
unreachable, refusing, or missing one variable has to end in "run it locally",
never in an exception.
"""

from __future__ import annotations

import httpx
import pytest

from strata.artifact_store import ArtifactStore, TransformSpec
from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.notebook.provenance import derive_subkey
from strata.notebook.team_store import TeamStore, pull_cell_outputs
from tests.conftest import run_server_with_context

NOTEBOOK_ID = "nbteam"
CELL_ID = "c1"
CELL_PROVENANCE = "a" * 64


@pytest.fixture
def team_store_server(tmp_path):
    """A shared store, and the directory its artifacts land in."""
    cache_dir = tmp_path / "shared-cache"
    artifact_dir = tmp_path / "shared-artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()
    with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
        yield {"base_url": ctx.base_url, "artifact_dir": artifact_dir}


@pytest.fixture
def local_manager(tmp_path):
    """This machine's own notebook artifact store — empty."""
    return NotebookArtifactManager(NOTEBOOK_ID, artifact_dir=tmp_path / "local-artifacts")


def seed_team_result(
    artifact_dir,
    *,
    variable: str,
    blob: bytes,
    content_type: str = "json/object",
    principal: str | None = "alice",
    cell_provenance: str = CELL_PROVENANCE,
) -> str:
    """Put a teammate's result in the shared store, keyed by provenance.

    Written directly rather than over HTTP because ``PUT /v1/artifacts``
    computes its own provenance hash from inputs+transform and offers no way
    to supply the notebook's. Storing bytes under a caller-supplied key is the
    *next* slice; this one is about reading them back, so the seeding here is
    deliberately the same shape that slice will produce over the wire.

    The artifact id is deliberately unlike anything this notebook would
    construct — it belongs to a different notebook, which is the whole point:
    the hash is the join key, not the id.
    """
    artifact_id = f"nb_someone_elses_notebook_cell_zz_var_{variable}"
    provenance = derive_subkey(cell_provenance, variable)
    store = ArtifactStore(artifact_dir)
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance,
        transform_spec=TransformSpec(
            executor="notebook/cell@v1",
            params={"content_type": content_type, "variable_name": variable},
            inputs=[],
        ),
        principal=principal,
    )
    store.blob_store.write_blob(artifact_id, version, blob)
    store.finalize_artifact(
        artifact_id=artifact_id,
        version=version,
        schema_json="",
        row_count=0,
        byte_size=len(blob),
    )
    return artifact_id


def canonical_provenance(manager: NotebookArtifactManager, variable: str) -> str | None:
    """What the executor's cache-hit check reads: the local canonical artifact."""
    stored = manager.artifact_store.get_latest_version(
        manager.cell_artifact_id(CELL_ID, variable),
    )
    return stored.provenance_hash if stored else None


async def test_a_pull_lands_where_the_cache_check_looks(team_store_server, local_manager):
    """Not just "an artifact exists" — the exact canonical id and provenance
    the executor re-reads before it will call a hit."""
    seed_team_result(team_store_server["artifact_dir"], variable="model", blob=b'{"trees": 200}')
    seed_team_result(team_store_server["artifact_dir"], variable="scaler", blob=b'{"mean": 0}')

    store = TeamStore(team_store_server["base_url"])
    try:
        pull = await pull_cell_outputs(
            store,
            local_manager,
            cell_id=CELL_ID,
            provenance_hash=CELL_PROVENANCE,
            consumed_vars={"model", "scaler"},
        )
    finally:
        await store.aclose()

    assert pull is not None
    assert pull.variables == ("model", "scaler")
    assert pull.principal == "alice"

    for variable in ("model", "scaler"):
        assert canonical_provenance(local_manager, variable) == derive_subkey(
            CELL_PROVENANCE, variable
        )

    # And the bytes survived the round trip, not just the metadata.
    stored_id = local_manager.cell_artifact_id(CELL_ID, "model")
    latest = local_manager.artifact_store.get_latest_version(stored_id)
    assert latest is not None
    assert local_manager.load_artifact_data(stored_id, latest.version) == b'{"trees": 200}'


async def test_one_missing_variable_is_a_miss_and_writes_nothing(team_store_server, local_manager):
    """The cache check needs every consumed variable, so a partial pull is a
    miss that also littered the local store. Fetch all, then write."""
    seed_team_result(team_store_server["artifact_dir"], variable="model", blob=b'{"trees": 200}')

    store = TeamStore(team_store_server["base_url"])
    try:
        pull = await pull_cell_outputs(
            store,
            local_manager,
            cell_id=CELL_ID,
            provenance_hash=CELL_PROVENANCE,
            consumed_vars={"model", "scaler"},
        )
    finally:
        await store.aclose()

    assert pull is None
    assert canonical_provenance(local_manager, "model") is None
    assert canonical_provenance(local_manager, "scaler") is None


async def test_an_unreachable_store_is_a_miss_not_an_error(local_manager):
    """A store that is down is a store you recompute past. Raising here would
    turn a shared-cache outage into every teammate's notebook breaking."""
    unreachable = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ConnectError("no route to host"))
        )
    )
    store = TeamStore("http://store.invalid", client=unreachable)
    try:
        pull = await pull_cell_outputs(
            store,
            local_manager,
            cell_id=CELL_ID,
            provenance_hash=CELL_PROVENANCE,
            consumed_vars={"model"},
        )
    finally:
        await unreachable.aclose()

    assert pull is None
    assert canonical_provenance(local_manager, "model") is None


async def test_a_refusing_store_is_loud_while_an_empty_one_is_quiet(monkeypatch):
    """Both end in "run it locally", so the log is the only place the
    difference can live — and the difference matters: an expired token is a
    permanent unexplained slowdown, an empty cache is Tuesday.

    The logger is monkeypatched rather than read through ``caplog``: the
    package configures its own logging with ``propagate=False``, so records
    never reach pytest's capture handler (same reason as
    ``tests/test_blob_store.py``).
    """

    class _Recorder:
        def __init__(self):
            self.warnings: list[str] = []

        def warning(self, msg, *args):
            self.warnings.append(msg % args if args else msg)

        def debug(self, msg, *args):
            return None

        def info(self, msg, *args):
            return None

    recorder = _Recorder()
    monkeypatch.setattr("strata.notebook.team_store.logger", recorder)

    def respond(request: httpx.Request) -> httpx.Response:
        if "denied" in str(request.url):
            return httpx.Response(403, json={"detail": "nope"})
        return httpx.Response(
            404,
            json={"detail": "nothing here"},
            headers={"X-Strata-Provenance-Miss": "1"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    store = TeamStore("http://store.example", client=client)
    try:
        assert await store.fetch("b" * 64) is None
        assert recorder.warnings == [], "an ordinary empty cache must not warn"

        assert await store.fetch("denied" + "b" * 58) is None
        assert any("refused" in message for message in recorder.warnings)
    finally:
        await client.aclose()


async def test_a_teammates_result_is_served_instead_of_running_the_cell(
    tmp_path, team_store_server, monkeypatch
):
    """The product claim, end to end.

    Two notebooks that never met run the same cell. The first computes it and
    its result reaches the shared store; the second is served that result and
    does not run — no subprocess, no seconds. It works because the notebook's
    provenance key contains no notebook id and no cell id, so two people arrive
    at the same hash independently.

    The seeding is the first notebook's *real* stored bytes, copied into the
    shared store under the same provenance. That keeps this test about the pull
    and not about a hand-built blob that happens to deserialize.
    """
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    upstream_source = "import time\nvalue = sum(range(1000))"
    downstream_source = "doubled = value * 2"

    def build(name: str):
        notebook_dir = create_notebook(tmp_path / name, name)
        add_cell_to_notebook(notebook_dir, "up", None)
        write_cell(notebook_dir, "up", upstream_source)
        add_cell_to_notebook(notebook_dir, "down", "up")
        write_cell(notebook_dir, "down", downstream_source)
        return notebook_dir, NotebookSession(parse_notebook(notebook_dir), notebook_dir)

    # --- Alice runs it for real ---
    alice_dir, alice = build("alice")
    alice_result = await CellExecutor(alice).execute_cell("up", upstream_source)
    assert alice_result.success, alice_result.error
    assert alice_result.cache_hit is False

    alice_store = alice.get_artifact_manager()
    alice_artifact_id = alice_store.cell_artifact_id("up", "value")
    alice_artifact = alice_store.artifact_store.get_latest_version(alice_artifact_id)
    assert alice_artifact is not None
    alice_blob = alice_store.load_artifact_data(alice_artifact_id, alice_artifact.version)

    # --- Her result reaches the shared store, keyed by provenance ---
    # (Publishing it from the notebook is the next slice; what matters here is
    # that the bytes and the key are the real ones she produced.)
    shared = ArtifactStore(team_store_server["artifact_dir"])
    shared_id = "nb_alice_cell_up_var_value"
    version = shared.create_artifact(
        artifact_id=shared_id,
        provenance_hash=alice_artifact.provenance_hash,
        transform_spec=TransformSpec(
            executor="notebook/cell@v1",
            params={"content_type": "json/object", "variable_name": "value"},
            inputs=[],
        ),
        principal="alice",
    )
    shared.blob_store.write_blob(shared_id, version, alice_blob)
    shared.finalize_artifact(
        artifact_id=shared_id,
        version=version,
        schema_json="",
        row_count=0,
        byte_size=len(alice_blob),
    )

    # --- Bob, on a cold notebook, points at the shared store ---
    bob_dir, bob = build("bob")
    team_config = StrataConfig(
        cache_dir=tmp_path / "bob-cache",
        notebook_remote_store_url=team_store_server["base_url"],
        notebook_team_cache_enabled=True,
    )
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: team_config)

    bob_executor = CellExecutor(bob)
    bob_result = await bob_executor.execute_cell("up", upstream_source)

    assert bob_result.success, bob_result.error
    assert bob_result.cache_hit is True
    # Attribution, not just speed: a result that appears with no author is
    # indistinguishable from a bug.
    assert bob_result.team_cache_principal == "alice"

    # And it is a real local artifact afterwards, so Bob's downstream cell
    # resolves `value` without touching the network again.
    bob_store = bob.get_artifact_manager()
    bob_artifact_id = bob_store.cell_artifact_id("up", "value")
    bob_artifact = bob_store.artifact_store.get_latest_version(bob_artifact_id)
    assert bob_artifact is not None
    assert bob_artifact.provenance_hash == alice_artifact.provenance_hash
    assert bob_store.load_artifact_data(bob_artifact_id, bob_artifact.version) == alice_blob


async def test_the_pull_is_off_unless_it_is_switched_on(tmp_path, monkeypatch):
    """A configured remote store is for publishing. Sourcing results from one
    is the separate opt-in, and with it off nothing reaches the network."""
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    notebook_dir = create_notebook(tmp_path / "solo", "solo")
    add_cell_to_notebook(notebook_dir, "up", None)
    write_cell(notebook_dir, "up", "x = 1")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)

    monkeypatch.setattr(
        CellExecutor,
        "_lake_config",
        lambda self: StrataConfig(
            cache_dir=tmp_path / "cache",
            notebook_remote_store_url="http://store.invalid",
        ),
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the team store must not be consulted when the pull is off")

    monkeypatch.setattr("strata.notebook.executor.pull_cell_outputs", refuse)

    assert (
        await CellExecutor(session)._pull_from_team_store(
            cell_id="up",
            provenance_hash=CELL_PROVENANCE,
            consumed_vars={"x"},
            source_hash="",
            env_hash="",
        )
        is None
    )


async def test_a_cell_with_no_downstream_consumers_is_not_pulled(local_manager):
    """Nothing to pull: a leaf cell stores no artifacts, and its console and
    display outputs live in the session's runtime state instead."""
    store = TeamStore("http://store.example", client=httpx.AsyncClient())
    pull = await pull_cell_outputs(
        store,
        local_manager,
        cell_id=CELL_ID,
        provenance_hash=CELL_PROVENANCE,
        consumed_vars=set(),
    )
    assert pull is None
