"""The team cache tier: a colleague's result instead of your recomputation.

A pull must leave the local store in exactly the state a local run would: the cache-hit check
re-reads each consumed variable's canonical artifact and compares its provenance, so anything less
specific still misses. And a store that is unreachable, refusing, or missing a variable must end in
"run it locally", never an exception.
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
    """This machine's own notebook artifact store, empty."""
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

    Written directly because ``PUT /v1/artifacts`` computes its own provenance hash. The artifact id
    belongs to a different notebook on purpose: the hash is the join key, not the id.
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
    """The local canonical artifact's provenance, which the cache-hit check reads."""
    stored = manager.artifact_store.get_latest_version(
        manager.cell_artifact_id(CELL_ID, variable),
    )
    return stored.provenance_hash if stored else None


async def test_a_pull_lands_where_the_cache_check_looks(team_store_server, local_manager):
    """Checks the exact canonical id and provenance the executor re-reads, not just that an artifact
    exists.
    """
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
    """The cache check needs every consumed variable, so a partial pull must write nothing."""
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
    """Raising would turn a shared-cache outage into every teammate's notebook breaking."""
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
    """Both end in a local run, so only the log can show the difference: an expired token is a
    permanent unexplained slowdown, an empty cache is normal.

    The logger is monkeypatched because the package logs with ``propagate=False``, so ``caplog``
    never sees the records.
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
    """End to end: two unrelated notebooks run the same cell and the second is served the first's
    result without running.

    It works because the provenance key holds no notebook or cell id. The seed is the first
    notebook's real stored bytes, so this tests the pull, not a hand-built blob.
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
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        return notebook_dir, session

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
    # (Put there directly; what matters is that the bytes and key are the ones she produced.)
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
    # She shared it on purpose: it arrived as part of a promotion.
    shared.set_tag(shared_id, version, "nb_promotion", "taxi/value")

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
    assert bob_result.team_cache_promotion == "taxi/value"

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
    """The whole loop with nothing seeded: Alice's run pushes because team cache is on, and Bob's
    cold notebook is served her result.
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
        # Opening a notebook syncs its environment and publishing requires that
        # attestation, so a session constructed directly must sync too.
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


def _project_lock(project: str) -> str:
    """The ``uv.lock`` uv writes for a ``strata new`` notebook: its own project is the root."""
    return (
        'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n\n'
        f'[[package]]\nname = "{project}"\nversion = "0.1.0"\nsource = {{ virtual = "." }}\n'
        'dependencies = [\n    { name = "pyarrow" },\n]\n\n'
        '[package.metadata]\nrequires-dist = [{ name = "pyarrow", specifier = ">=18" }]\n\n'
        '[[package]]\nname = "pyarrow"\nversion = "21.0.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        'sdist = { url = "https://x/pyarrow-21.0.0.tar.gz", hash = "sha256:00" }\n'
    )


async def test_notebooks_named_apart_with_one_lock_share_a_result(
    tmp_path, team_store_server, monkeypatch
):
    """Two notebooks whose ``uv.lock`` differs only in their own project's name get one
    provenance, so the second is served the first's result.
    """
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    upstream_source = "value = sum(range(5000))"

    def build(name: str):
        notebook_dir = create_notebook(tmp_path / name, name)
        (notebook_dir / "uv.lock").write_text(_project_lock(name))
        add_cell_to_notebook(notebook_dir, "up", None)
        write_cell(notebook_dir, "up", upstream_source)
        add_cell_to_notebook(notebook_dir, "down", "up")
        write_cell(notebook_dir, "down", "doubled = value * 2")
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.ensure_venv_synced()
        return session

    team_config = StrataConfig(
        cache_dir=tmp_path / "shared-config-cache",
        notebook_remote_store_url=team_store_server["base_url"],
        notebook_team_cache_enabled=True,
    )
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: team_config)

    nb0 = build("nb0")
    first = await CellExecutor(nb0).execute_cell("up", upstream_source)
    assert first.success, first.error
    assert first.cache_hit is False

    nb1 = build("nb1")
    assert (nb0.path / "uv.lock").read_text() != (nb1.path / "uv.lock").read_text()
    second = await CellExecutor(nb1).execute_cell("up", upstream_source)

    assert second.success, second.error
    assert second.cache_hit is True, "the project name kept two identical environments apart"
    first_store, second_store = (session.get_artifact_manager() for session in (nb0, nb1))
    first_artifact = first_store.artifact_store.get_latest_version(
        first_store.cell_artifact_id("up", "value")
    )
    second_artifact = second_store.artifact_store.get_latest_version(
        second_store.cell_artifact_id("up", "value")
    )
    assert first_artifact is not None and second_artifact is not None
    assert first_artifact.provenance_hash == second_artifact.provenance_hash


async def test_run_all_both_contributes_to_the_team_and_is_served_by_it(
    tmp_path, team_store_server, monkeypatch
):
    """The same loop driven through Run All, which must both offer its results and look before
    computing.
    """
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    upstream_source = "value = sum(range(5000))"
    downstream_source = "doubled = value * 2"

    def build(name: str):
        notebook_dir = create_notebook(tmp_path / name, name)
        add_cell_to_notebook(notebook_dir, "up", None)
        write_cell(notebook_dir, "up", upstream_source)
        add_cell_to_notebook(notebook_dir, "down", "up")
        write_cell(notebook_dir, "down", downstream_source)
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.ensure_venv_synced()
        return session

    def specs(session):
        dag = session.dag
        return [
            {
                "cell_id": cell_id,
                "source": source,
                "consumed_vars": sorted(dag.consumed_variables.get(cell_id, set()) if dag else ()),
                "references": [],
                "env": {},
                "mount_manifest": {},
                "table_manifest": {},
                "source_hash": "",
                "env_hash": "",
            }
            for cell_id, source in (("up", upstream_source), ("down", downstream_source))
        ]

    team_config = StrataConfig(
        cache_dir=tmp_path / "shared-config-cache",
        notebook_remote_store_url=team_store_server["base_url"],
        notebook_team_cache_enabled=True,
    )
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: team_config)

    alice = build("alice")
    alice_run = await CellExecutor(alice).execute_batch(specs(alice))
    assert alice_run.completed, alice_run.end_reason
    assert {r.cell_id: r.cache_hit for r in alice_run.cell_results}["up"] is False

    bob = build("bob")
    bob_run = await CellExecutor(bob).execute_batch(specs(bob))

    assert bob_run.completed, bob_run.end_reason
    assert {r.cell_id: r.cache_hit for r in bob_run.cell_results}["up"] is True, (
        "Run All recomputed a cell Alice's Run All had already published"
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


async def test_a_store_that_refuses_a_publish_does_not_fail_the_cell(tmp_path, monkeypatch):
    """A read-only member, expired token or down store costs the next person a recompute, never this
    cell.
    """
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
    session.refresh_environment_runtime()

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
    """A configured remote store is for publishing; pulling is a separate opt-in, and with it off
    nothing reaches the network.
    """
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
            source="",
            env_hash="",
            input_versions={},
        )
        is None
    )


async def test_a_cell_with_no_downstream_consumers_is_not_pulled(local_manager):
    """A leaf cell stores no artifacts; its console and display live in runtime state."""
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
    """The provenance key covers the lockfile, not the platform, so a hit can cross machines.

    That is deliberate (hashing the platform would kill cross-machine hits), so the producer records
    what ran it and the pull reports it.
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
        # Opening a notebook syncs its environment and publishing requires that
        # attestation, so a session constructed directly must sync too.
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
    # The venv shims to the dev interpreter, so the harness reports this process's
    # identity, which makes the expected value knowable.
    assert hit.team_cache_build_env == build_env_identity()
    assert hit.team_cache_build_env != ""


async def test_a_pulled_result_keeps_the_publishers_platform(team_store_server, local_manager):
    """Preserved, not restamped: the puller's identity would claim this machine produced it, and the
    next pull from this store would inherit that.
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
    """The publisher's duration travels with the bytes so Bob is told what he skipped.

    His own history has no comparable run, so without it the savings estimate would be zero.
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
        # Opening a notebook syncs its environment and publishing requires that
        # attestation, so a session constructed directly must sync too.
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
    # It is *her* run being reported, not his instant one: the number must come from
    # the run that actually happened.
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
    """A failed ``uv sync`` keeps the old venv and leaves sync state ``ready``, while provenance
    follows the new ``uv.lock``.

    Published, such a result would become the team's answer under first-writer-wins. The cell still
    runs and stores locally; only the publish is refused.
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
    """The gate must be a gate, not a latch: a fixed environment publishes."""
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
    """The metadata snapshot (what is declared) is rebuilt on every sync; it must carry the
    attestation (what was installed) forward, or one "Sync environment" turns publishing off for
    good.
    """
    from strata.notebook.writer import update_environment_metadata

    session = _synced_notebook(tmp_path, "refreshed")
    update_environment_metadata(session.path)

    assert session.environment_attestation_error() is None


def test_reopening_a_session_does_not_launder_a_failed_sync(tmp_path, monkeypatch):
    """``refresh_environment_runtime`` runs on every reopen, where nothing is installed.

    Attesting there would let a browser reload launder a failed sync and publish a stale result.
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
    """CLI, MCP and scratchpad sessions never sync, leaving ``interpreter_source`` ``unknown``.

    ``unknown`` means unprobed, not broken; only a known system-python fallback disqualifies.
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
    """The Python attestation folds ``renv.lock`` in, so a ``uv sync`` before a failed
    ``renv::restore()`` would otherwise pass the gate with a stale R library.
    """
    session = _synced_notebook(tmp_path, "rnotebook")

    # An R lockfile appears and was never restored. The Python side re-syncs happily,
    # since its own install genuinely succeeded.
    (session.path / "renv.lock").write_text('{"Packages": {"jsonlite": {"Version": "1.8.0"}}}')
    session.ensure_venv_synced()

    error = session.environment_attestation_error()
    assert error is not None and "R library" in error


def test_both_execution_paths_report_the_same_build_environment():
    """A warm-pool cell and a cold-harness cell land in the same shared cache.

    ``harness.py`` and ``pool_worker.py`` cannot ``import strata``, so each carries its own copy of
    the identity function; the copies must agree.
    """
    from strata.notebook.harness import build_env_identity as harness_identity
    from strata.notebook.pool_worker import build_env_identity as pool_identity

    assert pool_identity() == harness_identity()
    assert harness_identity().count("-") >= 3, "expected impl-version-platform-machine"


async def test_a_store_that_disagrees_about_the_environment_is_not_believed(
    team_store_server, local_manager, monkeypatch
):
    """An honest pull cannot disagree: ``env_hash`` is part of the provenance key.

    If one does, importing it would make ``causality._get_stored_hash`` report "the environment
    changed" forever with no cause shown, so the local value is kept and a warning logged.
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


async def test_a_pulled_result_keeps_its_author_in_the_local_store(
    team_store_server, local_manager
):
    """The lineage view reads the local store, so the pulled principal must be persisted there."""
    seed_team_result(
        team_store_server["artifact_dir"], variable="model", blob=b"{}", principal="alice@lab"
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

    assert pull is not None and pull.principal == "alice@lab"
    stored = local_manager.artifact_store.get_latest_version(
        local_manager.cell_artifact_id(CELL_ID, "model")
    )
    assert stored is not None
    assert stored.principal == "alice@lab", "the author was surfaced but not persisted"


async def test_a_disputed_env_hash_on_any_variable_is_caught(
    team_store_server, local_manager, monkeypatch
):
    """Stamping every variable from the alphabetically first one would silently launder a bad value
    on the rest.
    """
    import json as json_module

    shared = ArtifactStore(team_store_server["artifact_dir"])
    for variable, stored_env in (("alpha", "1" * 64), ("zeta", "9" * 64)):
        artifact_id = f"nb_other_cell_zz_var_{variable}"
        version = shared.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=derive_subkey(CELL_PROVENANCE, variable),
            transform_spec=TransformSpec(
                executor="notebook/cell@v1",
                params={"content_type": "json/object", "env_hash": stored_env},
                inputs=[],
            ),
        )
        shared.blob_store.write_blob(artifact_id, version, b"{}")
        shared.finalize_artifact(
            artifact_id=artifact_id, version=version, schema_json="", row_count=0, byte_size=2
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
            consumed_vars={"alpha", "zeta"},
            env_hash="1" * 64,  # agrees with alpha, not with zeta
        )
    finally:
        await store.aclose()

    assert pull is not None
    assert any("cannot both be right" in w for w in warnings), (
        "a bad env_hash on a later-sorting variable went unreported"
    )
    for variable in ("alpha", "zeta"):
        stored = local_manager.artifact_store.get_latest_version(
            local_manager.cell_artifact_id(CELL_ID, variable)
        )
        params = json_module.loads(stored.transform_spec)["params"]
        assert params["env_hash"] == "1" * 64


async def test_under_the_promoted_policy_a_cell_run_offers_nothing(
    tmp_path, team_store_server, monkeypatch
):
    """Under ``promoted`` a cell run offers nothing, so Bob misses.

    On a personal server, offering every intermediate would share everything a researcher computes.
    Pulls are unchanged, so Bob is checked for a miss.
    """
    from strata.artifact_store import ArtifactStore
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    upstream_source = "value = sum(range(4321))"

    def build(name: str):
        notebook_dir = create_notebook(tmp_path / name, name)
        add_cell_to_notebook(notebook_dir, "up", None)
        write_cell(notebook_dir, "up", upstream_source)
        # A downstream cell, because only *consumed* variables are offered:
        # without one there is nothing to push and the test would pass with
        # the policy gate removed entirely.
        add_cell_to_notebook(notebook_dir, "down", "up")
        write_cell(notebook_dir, "down", "doubled = value * 2")
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.ensure_venv_synced()
        return session

    team_config = StrataConfig(
        cache_dir=tmp_path / "promoted-cache",
        notebook_remote_store_url=team_store_server["base_url"],
        notebook_team_cache_enabled=True,
        notebook_team_cache_publish="promoted",
    )
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: team_config)

    alice = build("alice")
    alice_result = await CellExecutor(alice).execute_cell("up", upstream_source)
    assert alice_result.success, alice_result.error

    # Nothing of hers is in the shared store.
    shared = ArtifactStore(team_store_server["artifact_dir"])
    assert shared.stats()["total_versions"] == 0

    # So Bob runs it himself.
    bob = build("bob")
    bob_result = await CellExecutor(bob).execute_cell("up", upstream_source)

    assert bob_result.success, bob_result.error
    assert bob_result.cache_hit is False
    assert bob_result.team_cache_principal is None


async def test_each_callers_offered_results_carry_that_callers_principal(tmp_path, monkeypatch):
    """A shared server offers results for several members; each must arrive as the member who ran
    the cell, not as the server.
    """
    import asyncio
    import http.server
    import threading

    from strata.auth import set_principal
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
    from strata.types import PROVENANCE_MISS_HEADER, Principal

    offered: list[dict[str, str]] = []

    class Store(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(404)
            self.send_header(PROVENANCE_MISS_HEADER, "1")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_PUT(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            offered.append(dict(self.headers))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            return None

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Store)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def build(name: str, source: str):
        notebook_dir = create_notebook(tmp_path / name, name)
        add_cell_to_notebook(notebook_dir, "up", None)
        write_cell(notebook_dir, "up", source)
        add_cell_to_notebook(notebook_dir, "down", "up")
        write_cell(notebook_dir, "down", "doubled = value * 2")
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.ensure_venv_synced()
        return session

    def configure(forward: bool):
        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            notebook_remote_store_url=f"http://127.0.0.1:{server.server_address[1]}",
            notebook_remote_store_headers={"X-Strata-Principal": "server:7"},
            notebook_team_cache_enabled=True,
            notebook_remote_store_forward_principal=forward,
        )
        monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)

    async def run_as(principal_id: str, session, source: str):
        # A task of its own, as each WebSocket's execution is: the principal is
        # the context the auth layer set for that caller.
        async def body():
            set_principal(Principal(id=principal_id, tenant="acme"))
            return await CellExecutor(session).execute_cell("up", source)

        return await asyncio.create_task(body())

    try:
        configure(forward=True)
        ana = await run_as("ana", build("ana", "value = 11"), "value = 11")
        ben = await run_as("ben", build("ben", "value = 22"), "value = 22")
        assert ana.success and ben.success, (ana.error, ben.error)
        forwarded = [h["X-Strata-Principal"] for h in offered]

        offered.clear()
        configure(forward=False)
        cam = await run_as("cam", build("cam", "value = 33"), "value = 33")
        assert cam.success, cam.error
        fixed = [h["X-Strata-Principal"] for h in offered]
    finally:
        server.shutdown()

    assert forwarded == ["ana", "ben"]
    assert fixed == ["server:7"]


async def test_an_output_overtaken_by_a_duplicate_offers_its_canonical_bytes(local_manager):
    """A version dedup overtook holds no bytes of its own; the offer reads them through it."""
    from strata.notebook.team_store import publish_cell_outputs

    store = local_manager.artifact_store
    spec = TransformSpec(executor="notebook/cell@v1", params={}, inputs=[])
    for artifact_id in ("canonical", local_manager.cell_artifact_id(CELL_ID, "x")):
        version = store.create_artifact(artifact_id, provenance_hash="p", transform_spec=spec)
        store.write_blob(artifact_id, version, b"bytes")
        store.finalize_artifact(artifact_id, version, "", 0, 5)
    offered: list[bytes] = []

    class Recording:
        async def publish(self, provenance_hash, blob, **_):
            offered.append(blob)
            return True

    published = await publish_cell_outputs(
        Recording(), local_manager, cell_id=CELL_ID, consumed_vars={"x"}
    )
    assert published == 1
    assert offered == [b"bytes"]
