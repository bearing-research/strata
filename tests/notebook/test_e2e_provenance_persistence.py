"""E2E invariant: every successful execution path must persist
``last_provenance_hash`` / ``last_source_hash`` / ``last_env_hash``
to ``.strata/runtime.json``.

The executor has three branches that end in ``record_successful_execution_provenance``:

1. **cold** — cache miss, fresh harness run
2. **cached** — provenance hit on a prior artifact, no subprocess
3. **loop** — loop-cell path that emits per-iteration artifacts

If any branch stops calling ``persist_cell_provenance``, reopened
notebooks silently lose the ability to classify cells as READY/STALE
without a re-execution. This test covers all three.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.runtime_state import load_runtime_state
from strata.notebook.session import SessionManager
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

pytestmark = pytest.mark.integration


def _provenance(nb_dir: Path, cell_id: str) -> dict[str, str]:
    entry = load_runtime_state(nb_dir).cells.get(cell_id)
    if entry is None:
        return {k: "" for k in ("last_provenance_hash", "last_source_hash", "last_env_hash")}
    return {
        "last_provenance_hash": entry.last_provenance_hash or "",
        "last_source_hash": entry.last_source_hash or "",
        "last_env_hash": entry.last_env_hash or "",
    }


@pytest.mark.asyncio
async def test_cold_and_cached_paths_persist_provenance(tmp_path: Path):
    """A second run of the same cell hits the cache branch — provenance
    must still be persisted so a subsequent reopen sees it."""
    nb_dir = create_notebook(tmp_path, "prov_persist_cache")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1")
    # Downstream consumer forces c1's output to be stored as an artifact,
    # which is the precondition for the cache-hit branch on re-run.
    add_cell_to_notebook(nb_dir, "c2", after_cell_id="c1")
    write_cell(nb_dir, "c2", "y = x + 1")

    session = SessionManager().open_notebook(nb_dir)
    session.ensure_venv_synced()
    executor = CellExecutor(session)

    first = await executor.execute_cell("c1", "x = 1")
    assert first.success
    assert first.execution_method == "cold"
    cold_prov = _provenance(nb_dir, "c1")
    assert cold_prov["last_provenance_hash"], "cold path must persist provenance"
    assert cold_prov["last_source_hash"]
    assert cold_prov["last_env_hash"]

    second = await executor.execute_cell("c1", "x = 1")
    assert second.success
    assert second.execution_method == "cached", (
        f"expected cache hit on rerun, got {second.execution_method}"
    )
    cached_prov = _provenance(nb_dir, "c1")
    assert cached_prov == cold_prov, (
        f"cache-hit branch must re-persist provenance; cold={cold_prov} cached={cached_prov}"
    )


@pytest.mark.asyncio
async def test_loop_path_persists_provenance(tmp_path: Path):
    """Loop cells take a separate execution branch that must also call
    ``record_successful_execution_provenance`` on success."""
    nb_dir = create_notebook(tmp_path, "prov_persist_loop")
    add_cell_to_notebook(nb_dir, "seed")
    write_cell(nb_dir, "seed", "state = {'n': 0}")
    add_cell_to_notebook(nb_dir, "loop", after_cell_id="seed")
    loop_src = "# @loop max_iter=3 carry=state\nstate = {'n': state['n'] + 1}\n"
    write_cell(nb_dir, "loop", loop_src)

    session = SessionManager().open_notebook(nb_dir)
    session.ensure_venv_synced()
    executor = CellExecutor(session)

    await executor.execute_cell("seed", "state = {'n': 0}")
    result = await executor.execute_cell("loop", loop_src)
    assert result.success, result.error
    assert result.execution_method == "loop"

    loop_prov = _provenance(nb_dir, "loop")
    assert loop_prov["last_provenance_hash"], "loop path must persist provenance"
    assert loop_prov["last_source_hash"]
    assert loop_prov["last_env_hash"]


@pytest.mark.asyncio
async def test_loop_cell_stores_non_carry_consumed_variables(tmp_path: Path):
    """A loop cell that defines a second variable consumed downstream must
    materialize it — only the carry used to be stored, so the downstream
    cell ran with the name unbound (code-review finding)."""
    nb_dir = create_notebook(tmp_path, "prov_loop_extra_var")
    add_cell_to_notebook(nb_dir, "seed")
    write_cell(nb_dir, "seed", "state = {'n': 0}")
    add_cell_to_notebook(nb_dir, "loop", after_cell_id="seed")
    loop_src = (
        "# @loop max_iter=3 carry=state\nstate = {'n': state['n'] + 1}\nsummary = state['n'] * 10\n"
    )
    write_cell(nb_dir, "loop", loop_src)
    add_cell_to_notebook(nb_dir, "use", after_cell_id="loop")
    use_src = "final = summary + 1"
    write_cell(nb_dir, "use", use_src)

    session = SessionManager().open_notebook(nb_dir)
    session.ensure_venv_synced()
    executor = CellExecutor(session)

    await executor.execute_cell("seed", "state = {'n': 0}")
    loop_result = await executor.execute_cell("loop", loop_src)
    assert loop_result.success, loop_result.error
    assert "summary" in loop_result.outputs

    # The canonical artifact for the non-carry consumed variable exists and
    # holds the FINAL iteration's value (3 iterations → 30).
    import json as _json

    artifact_mgr = session.get_artifact_manager()
    canonical_id = f"nb_{session.notebook_state.id}_cell_loop_var_summary"
    latest = artifact_mgr.artifact_store.get_latest_version(canonical_id)
    assert latest is not None
    assert _json.loads(artifact_mgr.load_artifact_data(canonical_id, latest.version)) == 30

    # And the downstream consumer resolves it and runs successfully
    # (previously `summary` was never materialized → NameError).
    use_result = await executor.execute_cell("use", use_src)
    assert use_result.success, use_result.error


@pytest.mark.asyncio
async def test_executed_cells_record_lineage_the_graph_walk_can_resolve(tmp_path: Path):
    """A three-cell chain must produce a lineage graph three artifacts deep.

    ``input_versions`` is what both the lineage API and ``strata artifact
    lineage`` walk, and both resolve an input only when its key is a
    ``strata://artifact/`` URI. The executor recorded raw provenance digests
    instead, so a notebook artifact's ancestry stopped one hop out at a node
    the reader could not identify — the entire graph a published artifact
    would show.

    Asserted through ``build_lineage`` against a real execution rather than on
    the recorded dict: the format only matters insofar as the walk consumes
    it, and only the executor can prove the refs are recorded where the
    resolution actually happens.
    """
    from strata.services.artifact import ArtifactService

    nb_dir = create_notebook(tmp_path, "lineage_chain")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "rows = [1, 2, 3]")
    add_cell_to_notebook(nb_dir, "c2", after_cell_id="c1")
    write_cell(nb_dir, "c2", "doubled = [r * 2 for r in rows]")
    add_cell_to_notebook(nb_dir, "c3", after_cell_id="c2")
    write_cell(nb_dir, "c3", "total = sum(doubled)")
    # Only *consumed* variables are stored as artifacts, so the chain needs a
    # cell downstream of c3 or ``total`` never reaches the store and the walk
    # has nothing to start from.
    add_cell_to_notebook(nb_dir, "c4", after_cell_id="c3")
    write_cell(nb_dir, "c4", "scaled = total * 10")

    session = SessionManager().open_notebook(nb_dir)
    session.ensure_venv_synced()
    executor = CellExecutor(session)

    for cell_id, source in (
        ("c1", "rows = [1, 2, 3]"),
        ("c2", "doubled = [r * 2 for r in rows]"),
        ("c3", "total = sum(doubled)"),
        ("c4", "scaled = total * 10"),
    ):
        result = await executor.execute_cell(cell_id, source)
        assert result.success, f"{cell_id}: {result.error}"

    manager = session.get_artifact_manager()
    leaf = manager.artifact_store.get_latest_version(manager.cell_artifact_id("c3", "total"))
    assert leaf is not None, "the last cell's output was never stored"

    lineage = ArtifactService().build_lineage(
        manager.artifact_store,
        artifact=leaf,
        artifact_id=leaf.id,
        version=leaf.version,
        tenant_filter=None,
        max_depth=10,
    )

    def uri_for(cell_id: str, var: str) -> str:
        return f"strata://artifact/{manager.cell_artifact_id(cell_id, var)}@v=1"

    resolved = [n.uri for n in lineage.nodes if n.type == "artifact"]
    assert resolved == [
        uri_for("c3", "total"),
        uri_for("c2", "doubled"),
        uri_for("c1", "rows"),
    ], f"lineage did not walk the chain: {resolved}"
    assert lineage.depth == 2


@pytest.mark.asyncio
async def test_stored_artifact_carries_the_source_that_produced_it(tmp_path: Path):
    """The artifact records the executed source, and keeps it after an edit.

    A reader outside the notebook has no ``cells/{id}.py`` to check a digest
    against, so ``source_hash`` alone explains nothing to them. What makes the
    recorded text trustworthy is *when* it is captured: the cell can be edited
    after the run, and reading the source back at publish time would pair a
    cached artifact with code that did not produce it. So the edit below is
    the point of the test, not decoration.
    """
    import json

    nb_dir = create_notebook(tmp_path, "source_capture")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "rows = [1, 2, 3]")
    add_cell_to_notebook(nb_dir, "c2", after_cell_id="c1")
    write_cell(nb_dir, "c2", "doubled = [r * 2 for r in rows]")

    session = SessionManager().open_notebook(nb_dir)
    session.ensure_venv_synced()
    executor = CellExecutor(session)

    assert (await executor.execute_cell("c1", "rows = [1, 2, 3]")).success
    assert (await executor.execute_cell("c2", "doubled = [r * 2 for r in rows]")).success

    manager = session.get_artifact_manager()

    def stored_source(cell_id: str, var: str) -> str:
        artifact = manager.artifact_store.get_latest_version(manager.cell_artifact_id(cell_id, var))
        assert artifact is not None, f"{cell_id}.{var} was never stored"
        return json.loads(artifact.transform_spec)["params"].get("source", "")

    assert stored_source("c1", "rows") == "rows = [1, 2, 3]"

    # Edit the cell without re-running it. The artifact still describes the
    # run that happened, not the source now sitting on disk.
    write_cell(nb_dir, "c1", "rows = [9, 9, 9]")
    assert stored_source("c1", "rows") == "rows = [1, 2, 3]"
