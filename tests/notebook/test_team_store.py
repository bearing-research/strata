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


async def test_a_result_alice_never_published_by_hand_reaches_bob(
    tmp_path, team_store_server, monkeypatch
):
    """The whole loop, with nothing seeded.

    Alice runs a cell; the push happens because she has the team cache on, not
    because the test placed anything anywhere. Bob then runs the same cell on a
    cold notebook and is served her result. This is the claim the roadmap's
    Phase 1 asks about — a colleague's expensive preprocessing is your instant
    result — with no step performed by the test that a real user would not get
    for free.
    """
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    upstream_source = "value = sum(range(5000))"

    def build(name: str):
        notebook_dir = create_notebook(tmp_path / name, name)
        add_cell_to_notebook(notebook_dir, "up", None)
        write_cell(notebook_dir, "up", upstream_source)
        add_cell_to_notebook(notebook_dir, "down", "up")
        write_cell(notebook_dir, "down", "doubled = value * 2")
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        # Opening a notebook syncs its environment. Publishing requires that
        # attestation — an unsynced session cannot say the venv matches
        # uv.lock — so a test that constructs a session directly has to do
        # what opening one does.
        session.ensure_venv_synced()
        return session

    team_config = StrataConfig(
        cache_dir=tmp_path / "shared-config-cache",
        notebook_remote_store_url=team_store_server["base_url"],
        notebook_team_cache_enabled=True,
    )
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: team_config)

    alice = build("alice")
    alice_result = await CellExecutor(alice).execute_cell("up", upstream_source)
    assert alice_result.success, alice_result.error
    assert alice_result.cache_hit is False
    # Nobody had computed it, so she had nothing to be served.
    assert alice_result.team_cache_principal is None

    bob = build("bob")
    bob_result = await CellExecutor(bob).execute_cell("up", upstream_source)

    assert bob_result.success, bob_result.error
    assert bob_result.cache_hit is True, (
        "Bob recomputed a cell Alice had already published to the shared store"
    )

    bob_store = bob.get_artifact_manager()
    bob_artifact_id = bob_store.cell_artifact_id("up", "value")
    bob_artifact = bob_store.artifact_store.get_latest_version(bob_artifact_id)
    assert bob_artifact is not None

    alice_store = alice.get_artifact_manager()
    alice_artifact_id = alice_store.cell_artifact_id("up", "value")
    alice_artifact = alice_store.artifact_store.get_latest_version(alice_artifact_id)
    assert alice_artifact is not None
    assert bob_artifact.provenance_hash == alice_artifact.provenance_hash
    assert bob_store.load_artifact_data(
        bob_artifact_id, bob_artifact.version
    ) == alice_store.load_artifact_data(alice_artifact_id, alice_artifact.version)


async def test_a_store_that_refuses_a_publish_does_not_fail_the_cell(tmp_path, monkeypatch):
    """The cell already succeeded. A read-only member, an expired token, or a
    store that is simply down must cost the *next* person a recomputation and
    this one nothing at all."""
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    source = "value = 7"
    notebook_dir = create_notebook(tmp_path / "solo", "solo")
    add_cell_to_notebook(notebook_dir, "up", None)
    write_cell(notebook_dir, "up", source)
    add_cell_to_notebook(notebook_dir, "down", "up")
    write_cell(notebook_dir, "down", "doubled = value * 2")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)

    monkeypatch.setattr(
        CellExecutor,
        "_lake_config",
        lambda self: StrataConfig(
            cache_dir=tmp_path / "cache",
            # Nothing listens here, so the publish attempt is a real failure
            # rather than a mocked one.
            notebook_remote_store_url="http://127.0.0.1:1",
            notebook_team_cache_enabled=True,
        ),
    )

    result = await CellExecutor(session).execute_cell("up", source)

    assert result.success, result.error
    stored = session.get_artifact_manager().artifact_store.get_latest_version(
        session.get_artifact_manager().cell_artifact_id("up", "value")
    )
    assert stored is not None, "the local artifact must survive a failed publish"


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


async def test_a_pulled_result_says_where_it_was_computed(tmp_path, team_store_server, monkeypatch):
    """The honesty half of the team cache.

    The provenance key covers the lockfile, not the platform: `uv.lock`
    resolves to different wheels on macOS-arm64 and Linux-x86_64, so a hit can
    legitimately cross machines. That is deliberate — hashing the platform
    would drop cross-machine hit rate to roughly zero and delete the feature in
    order to protect it — but sharing across platforms while recording *nothing*
    is not defensible. So the producer records what ran it, and the pull says
    so.
    """
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.harness import build_env_identity
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    source = "value = sum(range(3000))"

    def build(name: str):
        notebook_dir = create_notebook(tmp_path / name, name)
        add_cell_to_notebook(notebook_dir, "up", None)
        write_cell(notebook_dir, "up", source)
        add_cell_to_notebook(notebook_dir, "down", "up")
        write_cell(notebook_dir, "down", "doubled = value * 2")
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        # Opening a notebook syncs its environment. Publishing requires that
        # attestation — an unsynced session cannot say the venv matches
        # uv.lock — so a test that constructs a session directly has to do
        # what opening one does.
        session.ensure_venv_synced()
        return session

    monkeypatch.setattr(
        CellExecutor,
        "_lake_config",
        lambda self: StrataConfig(
            cache_dir=tmp_path / "cache",
            notebook_remote_store_url=team_store_server["base_url"],
            notebook_team_cache_enabled=True,
        ),
    )

    alice = build("alice")
    assert (await CellExecutor(alice).execute_cell("up", source)).success

    bob = build("bob")
    hit = await CellExecutor(bob).execute_cell("up", source)

    assert hit.cache_hit is True
    # The venv here shims to the dev interpreter, so the harness reports this
    # process's identity — which is what makes the expected value knowable.
    assert hit.team_cache_build_env == build_env_identity()
    assert hit.team_cache_build_env != ""


async def test_a_pulled_result_keeps_the_publishers_platform(team_store_server, local_manager):
    """Preserved, not restamped.

    Rewriting it with the puller's own identity would convert a record of
    where the result came from into a claim that this machine produced it —
    and the next person to pull from *this* store would inherit the lie.
    """
    import json as json_module

    foreign = "cpython-3.11-linux-s390x"
    artifact_id = "nb_someone_else_cell_zz_var_model"
    provenance = derive_subkey(CELL_PROVENANCE, "model")
    shared = ArtifactStore(team_store_server["artifact_dir"])
    version = shared.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance,
        transform_spec=TransformSpec(
            executor="notebook/cell@v1",
            params={"content_type": "json/object", "build_env": foreign},
            inputs=[],
        ),
        principal="alice",
    )
    shared.blob_store.write_blob(artifact_id, version, b'{"ok": 1}')
    shared.finalize_artifact(
        artifact_id=artifact_id, version=version, schema_json="", row_count=0, byte_size=9
    )

    store = TeamStore(team_store_server["base_url"])
    try:
        pull = await pull_cell_outputs(
            store,
            local_manager,
            cell_id=CELL_ID,
            provenance_hash=CELL_PROVENANCE,
            consumed_vars={"model"},
        )
    finally:
        await store.aclose()

    assert pull is not None
    assert pull.build_env == foreign

    stored = local_manager.artifact_store.get_latest_version(
        local_manager.cell_artifact_id(CELL_ID, "model")
    )
    assert stored is not None
    params = json_module.loads(stored.transform_spec)["params"]
    assert params["build_env"] == foreign


async def test_a_team_hit_is_priced_by_the_run_it_replaced(
    tmp_path, team_store_server, monkeypatch
):
    """The number that makes the shared store legible, end to end.

    Alice runs the cell; what it cost her travels with the bytes. Bob, who has
    never run it, is told what he skipped. Without the publisher's duration
    riding along there is nothing to report: his own history has no comparable
    run, so the savings estimate would credit zero for exactly the case the
    shared store exists to create.
    """
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    source = "value = sum(range(4000))"

    def build(name: str):
        notebook_dir = create_notebook(tmp_path / name, name)
        add_cell_to_notebook(notebook_dir, "up", None)
        write_cell(notebook_dir, "up", source)
        add_cell_to_notebook(notebook_dir, "down", "up")
        write_cell(notebook_dir, "down", "doubled = value * 2")
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        # Opening a notebook syncs its environment. Publishing requires that
        # attestation — an unsynced session cannot say the venv matches
        # uv.lock — so a test that constructs a session directly has to do
        # what opening one does.
        session.ensure_venv_synced()
        return session

    monkeypatch.setattr(
        CellExecutor,
        "_lake_config",
        lambda self: StrataConfig(
            cache_dir=tmp_path / "cache",
            notebook_remote_store_url=team_store_server["base_url"],
            notebook_team_cache_enabled=True,
        ),
    )

    alice = build("alice")
    alice_run = await CellExecutor(alice).execute_cell("up", source)
    assert alice_run.success, alice_run.error

    bob = build("bob")
    hit = await CellExecutor(bob).execute_cell("up", source)

    assert hit.cache_hit is True
    assert hit.team_cache_saved_ms > 0, "a team hit that reports no saving is not legible"
    # It is *her* run being reported, not his instant one. No timing threshold
    # here — only that the number came from the run that actually happened.
    assert hit.team_cache_saved_ms == int(alice_run.duration_ms)

    # And it reaches the profiling summary as team savings, not just total.
    bob.record_execution(
        "up",
        hit.duration_ms,
        hit.cache_hit,
        from_team=hit.from_team_cache,
        team_principal=hit.team_cache_principal,
        team_saved_ms=hit.team_cache_saved_ms,
    )
    summary = bob.get_profiling_summary()
    assert summary["team_cache_savings_ms"] == int(alice_run.duration_ms)
    assert summary["team_cache_hits"] == 1


async def test_a_failed_environment_sync_does_not_publish(tmp_path, team_store_server, monkeypatch):
    """The poisoning vector this gate exists for.

    A failed ``uv sync`` keeps the previous venv and leaves the sync state
    ``ready`` — deliberately, so a transient network failure does not lock
    someone out of their own notebook. But provenance is computed from
    ``uv.lock`` on disk, which has moved on. Every artifact produced from then
    on is stamped with an environment it was not built in.

    Locally that is the owner's problem. Published to a shared store it is
    permanent and everyone's: first-writer-wins means the stale-environment
    result becomes the answer the whole team gets.

    The cell must still run, and its result must still be stored locally.
    Only the publish is refused.
    """
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    source = "value = sum(range(1000))"
    notebook_dir = create_notebook(tmp_path / "broken", "broken")
    add_cell_to_notebook(notebook_dir, "up", None)
    write_cell(notebook_dir, "up", source)
    add_cell_to_notebook(notebook_dir, "down", "up")
    write_cell(notebook_dir, "down", "doubled = value * 2")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.ensure_venv_synced()
    assert session.environment_attestation_error() is None

    # The lockfile moves on and the re-sync fails: the venv still holds the old
    # environment while provenance now describes the new one.
    (notebook_dir / "uv.lock").write_text('version = 1\n[[package]]\nname = "new"\n')
    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda *a, **k: False)
    session.ensure_venv_synced()

    assert session.environment_sync_state == "ready", (
        "a failed sync deliberately stays usable — that is why the publish "
        "needs its own gate rather than relying on the sync state"
    )
    assert session.environment_attestation_error() is not None

    monkeypatch.setattr(
        CellExecutor,
        "_lake_config",
        lambda self: StrataConfig(
            cache_dir=tmp_path / "cache",
            notebook_remote_store_url=team_store_server["base_url"],
            notebook_team_cache_enabled=True,
        ),
    )
    result = await CellExecutor(session).execute_cell("up", source)

    # The cell ran and its result is local.
    assert result.success, result.error
    manager = session.get_artifact_manager()
    stored = manager.artifact_store.get_latest_version(manager.cell_artifact_id("up", "value"))
    assert stored is not None

    # But nothing reached the team.
    shared = ArtifactStore(team_store_server["artifact_dir"])
    assert shared.find_by_provenance(stored.provenance_hash) is None, (
        "an artifact built in a stale environment was published to the team"
    )


async def test_publishing_resumes_once_the_environment_is_synced(
    tmp_path, team_store_server, monkeypatch
):
    """The gate must be a gate, not a latch — a fixed environment publishes."""
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    source = "value = sum(range(1200))"
    notebook_dir = create_notebook(tmp_path / "recovered", "recovered")
    add_cell_to_notebook(notebook_dir, "up", None)
    write_cell(notebook_dir, "up", source)
    add_cell_to_notebook(notebook_dir, "down", "up")
    write_cell(notebook_dir, "down", "doubled = value * 2")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)

    (notebook_dir / "uv.lock").write_text('version = 1\n[[package]]\nname = "new"\n')
    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda *a, **k: False)
    session.ensure_venv_synced()
    assert session.environment_attestation_error() is not None

    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda *a, **k: True)
    session.ensure_venv_synced()
    assert session.environment_attestation_error() is None

    monkeypatch.setattr(
        CellExecutor,
        "_lake_config",
        lambda self: StrataConfig(
            cache_dir=tmp_path / "cache",
            notebook_remote_store_url=team_store_server["base_url"],
            notebook_team_cache_enabled=True,
        ),
    )
    result = await CellExecutor(session).execute_cell("up", source)
    assert result.success, result.error

    manager = session.get_artifact_manager()
    stored = manager.artifact_store.get_latest_version(manager.cell_artifact_id("up", "value"))
    assert stored is not None
    shared = ArtifactStore(team_store_server["artifact_dir"])
    assert shared.find_by_provenance(stored.provenance_hash) is not None


def _synced_notebook(tmp_path, name: str):
    """A notebook in the state opening one leaves it: synced and attested."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    notebook_dir = create_notebook(tmp_path / name, name)
    add_cell_to_notebook(notebook_dir, "up", None)
    write_cell(notebook_dir, "up", "value = 1")
    add_cell_to_notebook(notebook_dir, "down", "up")
    write_cell(notebook_dir, "down", "doubled = value * 2")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.ensure_venv_synced()
    assert session.environment_attestation_error() is None
    return session


def test_an_environment_metadata_refresh_keeps_the_attestation(tmp_path):
    """The metadata snapshot is rebuilt wholesale on every sync, dependency
    change, and environment job. It describes what is *declared*; the
    attestation describes what was *installed*. Rebuilding the record without
    carrying the attestation forward revoked it every time — so clicking "Sync
    environment" once turned publishing off permanently, on a healthy
    environment, with only a per-cell warning to show for it.
    """
    from strata.notebook.writer import update_environment_metadata

    session = _synced_notebook(tmp_path, "refreshed")
    update_environment_metadata(session.path)

    assert session.environment_attestation_error() is None


def test_reopening_a_session_does_not_launder_a_failed_sync(tmp_path, monkeypatch):
    """``refresh_environment_runtime`` runs on the session-reuse path — every
    reopen of an already-open notebook — where nothing is installed.

    Attesting there meant a failed sync could be laundered by reloading the
    browser tab: the new lockfile would be recorded as realized with nothing
    installed, and the next cell would publish a stale-environment result.
    """
    session = _synced_notebook(tmp_path, "reopened")

    (session.path / "uv.lock").write_text('version = 1\n[[package]]\nname = "new"\n')
    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda *a, **k: False)
    session.ensure_venv_synced()
    assert session.environment_attestation_error() is not None

    # The reopen path. Nothing was installed, so nothing may be attested.
    session.refresh_environment_runtime()

    assert session.environment_attestation_error() is not None, (
        "reopening the notebook laundered a failed sync into an attestation"
    )


def test_a_directly_constructed_session_can_still_publish(tmp_path):
    """The CLI, the MCP ops layer, and the agent scratchpad all build sessions
    directly and never call a sync, leaving ``interpreter_source`` at
    ``unknown``. Treating that as a failure switched publishing off for that
    entire surface — with nothing to diagnose it but a warning about uv.lock.

    ``unknown`` means unprobed, not broken. Only a *known* system-python
    fallback disqualifies.
    """
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    notebook_dir = create_notebook(tmp_path / "cli", "cli")
    add_cell_to_notebook(notebook_dir, "up", None)
    write_cell(notebook_dir, "up", "value = 1")

    fresh = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    assert fresh.environment_interpreter_source == "unknown"
    assert fresh.environment_attestation_error() is None


def test_a_stale_r_library_does_not_publish(tmp_path):
    """The Python attestation covers ``compute_lockfile_hash``, which folds
    ``renv.lock`` in too — so on an R notebook it can be satisfied by a
    ``uv sync`` that ran before a ``renv::restore()`` that failed, leaving the
    R library stale while the gate says everything is fine.
    """
    session = _synced_notebook(tmp_path, "rnotebook")

    # An R lockfile appears and was never restored. The Python side re-syncs
    # happily — its own install genuinely succeeded.
    (session.path / "renv.lock").write_text('{"Packages": {"jsonlite": {"Version": "1.8.0"}}}')
    session.ensure_venv_synced()

    error = session.environment_attestation_error()
    assert error is not None and "R library" in error


def test_both_execution_paths_report_the_same_build_environment():
    """A warm-pool cell and a cold-harness cell land in the same shared cache.

    ``harness.py`` and ``pool_worker.py`` both run inside the notebook venv and
    neither can ``import strata``, so each carries its own copy of the identity
    function. The duplication is fine; a divergence is not — the values are
    compared as strings by anyone reading the store, and the two paths are
    interchangeable from the user's point of view.

    The pool worker originally had no copy at all. Every test exercised the
    cold path, so it went unnoticed until a live team store showed warm-pool
    artifacts published with no platform recorded.
    """
    from strata.notebook.harness import build_env_identity as harness_identity
    from strata.notebook.pool_worker import build_env_identity as pool_identity

    assert pool_identity() == harness_identity()
    assert harness_identity().count("-") >= 3, "expected impl-version-platform-machine"


async def test_a_store_that_disagrees_about_the_environment_is_not_believed(
    team_store_server, local_manager, monkeypatch
):
    """An honest pull cannot disagree — ``env_hash`` is part of the provenance
    key, so a matching provenance implies a matching env_hash.

    If one disagrees anyway, importing it pushes the wrongness somewhere it
    cannot be read as a problem: ``causality._get_stored_hash`` prefers the
    stored env_hash over the cell's own when explaining staleness, so the cell
    would report "the environment changed" forever with nothing pointing at
    why. Keep the local value and say so out loud instead.
    """
    import json as json_module

    artifact_id = "nb_liar_cell_zz_var_model"
    provenance = derive_subkey(CELL_PROVENANCE, "model")
    shared = ArtifactStore(team_store_server["artifact_dir"])
    version = shared.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance,
        transform_spec=TransformSpec(
            executor="notebook/cell@v1",
            params={"content_type": "json/object", "env_hash": "9" * 64},
            inputs=[],
        ),
        principal="alice",
    )
    shared.blob_store.write_blob(artifact_id, version, b'{"ok": 1}')
    shared.finalize_artifact(
        artifact_id=artifact_id, version=version, schema_json="", row_count=0, byte_size=9
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        "strata.notebook.team_store.logger.warning",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )

    store = TeamStore(team_store_server["base_url"])
    try:
        pull = await pull_cell_outputs(
            store,
            local_manager,
            cell_id=CELL_ID,
            provenance_hash=CELL_PROVENANCE,
            consumed_vars={"model"},
            env_hash="1" * 64,
        )
    finally:
        await store.aclose()

    assert pull is not None, "a disagreement must not turn a usable result into a miss"
    assert any("cannot both be right" in w for w in warnings)

    stored = local_manager.artifact_store.get_latest_version(
        local_manager.cell_artifact_id(CELL_ID, "model")
    )
    assert stored is not None
    params = json_module.loads(stored.transform_spec)["params"]
    assert params["env_hash"] == "1" * 64, "the store's disputed value was imported"
