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
