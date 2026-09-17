"""What a cell is given must not depend on where it runs.

``_dispatch_execution`` hands the local and embedded backends
``mutation_defines`` and ``tables``; the HTTP branch did not take them, so a
cell that mutates a value in place recaptured it locally and silently did not
on a worker -- under the same provenance hash, which is a wrong answer served
from cache forever -- and an ``@table`` name was simply undefined there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from strata.notebook.executor import CellExecutor
from strata.notebook.models import WorkerBackendType, WorkerSpec

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX worker execution")


def _session(tmp_path: Path, name: str):
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import create_notebook

    notebook_dir = create_notebook(tmp_path, name, initialize_environment=False)
    return NotebookSession(parse_notebook(notebook_dir), notebook_dir)


@pytest.mark.parametrize("transport", ["direct", "signed"])
@pytest.mark.asyncio
async def test_a_table_reaches_the_cell_on_a_worker(
    tmp_path, transport, notebook_executor_server, notebook_build_server
):
    """``@table`` resolves to a uri and a snapshot id the server already
    folded into the provenance hash. Dropped on the way, the cell ran with a
    correct key and no variable: NameError on a worker, fine locally."""
    session = _session(tmp_path, f"tables-{transport}")
    worker = WorkerSpec(
        name="w",
        backend=WorkerBackendType.EXECUTOR,
        config={
            "url": notebook_executor_server["execute_url"],
            "transport": transport,
            "strata_url": notebook_build_server["base_url"],
        },
    )
    output_dir = tmp_path / f"out-{transport}"
    output_dir.mkdir()
    source = (
        "assert trips == 'lake://db.trips', trips\n"
        "assert trips_snapshot == 7, trips_snapshot\n"
        "checked = True\n"
    )

    result, _, method, _ = await CellExecutor(session)._dispatch_http_executor(
        worker,
        source,
        {},
        [],
        output_dir,
        {},
        60.0,
        tables={"trips": {"uri": "lake://db.trips", "snapshot_id": 7}},
    )

    assert method == "executor"
    assert result["success"] is True, result.get("error")


def test_a_worker_puts_both_into_the_harness_manifest(monkeypatch, tmp_path):
    """The worker's own half: whatever it is sent has to reach the manifest
    the harness reads, or forwarding them changed nothing."""
    from strata.notebook import remote_executor

    monkeypatch.delenv("STRATA_WORKER_TOKEN", raising=False)
    seen: dict[str, object] = {}

    async def _capture(harness_path, manifest_path, *args, **kwargs):
        seen.update(json.loads(Path(manifest_path).read_text()))
        raise RuntimeError("stop here: the manifest is what this test is about")

    monkeypatch.setattr(remote_executor, "_run_harness", _capture)
    client = TestClient(remote_executor.create_notebook_executor_app())
    metadata = {
        "protocol_version": "v1",
        "build_id": "b1",
        "transform": {
            "ref": "notebook_cell@v1",
            "code_hash": "x",
            "params": {
                "source": "pass\n",
                "timeout_seconds": 30.0,
                "mounts": [],
                "env": {},
                "mutation_defines": ["frame"],
                "tables": {"trips": {"uri": "lake://db.trips", "snapshot_id": 7}},
            },
        },
        "inputs": [],
    }

    client.post(
        "/v1/execute",
        files={"metadata": ("metadata.json", json.dumps(metadata).encode(), "application/json")},
    )

    assert seen.get("mutation_defines") == ["frame"], (
        "the worker dropped the list of variables the cell mutates in place, "
        "so the harness stores none of them"
    )
    assert seen.get("tables") == {"trips": {"uri": "lake://db.trips", "snapshot_id": 7}}
