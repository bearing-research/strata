"""R cells on remote workers. Item 46.

A worker runs an R cell's ``harness.R`` under its own ``Rscript``, and says so
in ``/health``. The first tests need no R: a worker without ``Rscript`` refuses
an R cell by name, and a stand-in ``Rscript`` shows which harness it is given.
The rest run R: an R cell with a worker runs there, over both transports, is a
cache hit locally afterwards, and a downstream Python cell reads its data frame.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.models import WorkerBackendType, WorkerSpec
from strata.notebook.remote_executor import (
    NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
    create_notebook_executor_app,
)
from tests.notebook.conftest import skip_if_no_r, skip_if_no_r_arrow

_REAL_WHICH = shutil.which


async def _execute(language: str) -> httpx.Response:
    metadata = {
        "protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": "out <- 1\n",
        "language": language,
        "inputs": {},
        "mounts": [],
        "env": {},
    }
    transport = httpx.ASGITransport(app=create_notebook_executor_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        return await client.post(
            "/v1/notebook-execute",
            files={
                "metadata": ("metadata.json", json.dumps(metadata).encode(), "application/json")
            },
        )


async def _languages() -> list[str]:
    transport = httpx.ASGITransport(app=create_notebook_executor_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        health = (await client.get("/health")).json()
    return health["capabilities"]["features"]["languages"]


@pytest.mark.asyncio
async def test_a_worker_without_rscript_refuses_an_r_cell(monkeypatch):
    monkeypatch.setattr(
        shutil, "which", lambda name: None if name == "Rscript" else _REAL_WHICH(name)
    )

    response = await _execute("r")

    assert response.status_code == 500
    assert response.json()["error"] == "Rscript is not installed on this worker"
    assert await _languages() == ["python"]


def _stand_in_rscript(tmp_path: Path, monkeypatch) -> Path:
    """An ``Rscript`` first on PATH that records what it was asked to run and
    answers as ``harness.R`` does. Returns the file its argv is written to."""
    calls = tmp_path / "calls.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    rscript = bin_dir / "Rscript"
    rscript.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, pathlib\n"
        f"pathlib.Path({str(calls)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "manifest = pathlib.Path(sys.argv[2])\n"
        "out = pathlib.Path(json.loads(manifest.read_text())['output_dir'])\n"
        "(out / 'out.json').write_text('1')\n"
        "(manifest.parent / 'harness-result.json').write_text(json.dumps({\n"
        "    'success': True, 'stdout': '', 'stderr': '', 'mutation_warnings': [],\n"
        "    'variables': {'out': {'content_type': 'json/object', 'file': 'out.json',\n"
        "                          'preview': 1}}}))\n"
    )
    rscript.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}")
    return calls


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in Rscript is a script")
@pytest.mark.asyncio
async def test_a_worker_runs_an_r_cell_with_harness_r(tmp_path, monkeypatch):
    calls = _stand_in_rscript(tmp_path, monkeypatch)

    response = await _execute("r")

    assert response.status_code == 200, response.text
    argv = json.loads(calls.read_text())
    assert Path(argv[0]).parts[-3:] == ("languages", "r", "harness.R")
    assert "r" in await _languages()


@pytest.mark.asyncio
async def test_an_unknown_language_is_refused(tmp_path):
    response = await _execute("julia")

    assert response.status_code == 400
    assert response.json()["error"] == "unsupported cell language 'julia'"


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in Rscript is a script")
@pytest.mark.parametrize("transport", ["direct", "signed"])
@pytest.mark.asyncio
async def test_the_server_tells_the_worker_the_cell_is_r(
    tmp_path, monkeypatch, transport, notebook_executor_server, notebook_build_server
):
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import create_notebook

    calls = _stand_in_rscript(tmp_path, monkeypatch)
    notebook_dir = create_notebook(tmp_path, "r-dispatch", initialize_environment=False)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    worker = WorkerSpec(
        name="r-worker",
        backend=WorkerBackendType.EXECUTOR,
        config={
            "url": notebook_executor_server["execute_url"],
            "transport": transport,
            "strata_url": notebook_build_server["base_url"],
        },
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    result, _, method, _ = await CellExecutor(session)._dispatch_http_executor(
        worker, "out <- 1\n", {}, [], output_dir, {}, 60.0, language="r"
    )

    assert method == "executor"
    assert result["success"] is True
    assert Path(json.loads(calls.read_text())[0]).name == "harness.R"


def _on_worker(session, cell_id: str, config: dict[str, str]) -> None:
    session.notebook_state.workers = [
        WorkerSpec(name="r-worker", backend=WorkerBackendType.EXECUTOR, config=config)
    ]
    session.notebook_state.get_cell(cell_id).worker = "r-worker"


_PY_C1 = "import pandas as pd\ndf = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})\n"
_R_C2 = "df_r <- df\ndf_r$z <- df_r$x + df_r$y\n"
_PY_C3 = "total = int(df_r['z'].sum())\n"


@skip_if_no_r
@skip_if_no_r_arrow
@pytest.mark.asyncio
async def test_an_r_cell_runs_on_its_worker_and_python_reads_its_data_frame(
    r_notebook, notebook_executor_server
):
    _, session = r_notebook(
        cells=[
            ("c1", None, _PY_C1, "python"),
            ("c2", "c1", _R_C2, "r"),
            ("c3", "c2", _PY_C3, "python"),
        ]
    )
    _on_worker(session, "c2", {"url": notebook_executor_server["execute_url"]})
    executor = CellExecutor(session)

    assert (await executor.execute_cell("c1", _PY_C1)).success

    ran = await executor.execute_cell("c2", _R_C2)
    assert ran.success is True, ran.error
    assert ran.execution_method == "executor"
    assert ran.remote_worker == "r-worker"
    assert ran.outputs["df_r"]["content_type"] == "arrow/ipc"

    again = await executor.execute_cell("c2", _R_C2)
    assert again.success is True, again.error
    assert again.cache_hit is True

    downstream = await executor.execute_cell("c3", _PY_C3)
    assert downstream.success is True, downstream.error
    assert downstream.outputs["total"]["preview"] == 66


@skip_if_no_r
@skip_if_no_r_arrow
@pytest.mark.asyncio
async def test_an_r_cell_runs_on_a_signed_transport_worker(
    r_notebook, notebook_executor_server, notebook_build_server
):
    config = {
        "url": notebook_executor_server["execute_url"],
        "transport": "signed",
        "strata_url": notebook_build_server["base_url"],
    }
    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {"name": "r-worker", "backend": "executor", "config": config}
    ]
    r_c1 = 'model <- structure(list(coef = 1.5), class = "fit_model")\n'
    r_c2 = "coef <- model$coef * 2\n"
    _, session = r_notebook(cells=[("c1", None, r_c1, "r"), ("c2", "c1", r_c2, "r")])
    _on_worker(session, "c1", config)
    # The build server runs in service mode, which refuses cells on its own
    # host, so the downstream cell reads the RDS on the worker too.
    session.notebook_state.get_cell("c2").worker = "r-worker"
    executor = CellExecutor(session)

    ran = await executor.execute_cell("c1", r_c1)

    assert ran.success is True, ran.error
    assert ran.remote_transport == "signed"
    assert ran.remote_build_state == "ready"
    # An R-only value comes back as the RDS bytes it is locally, and goes out
    # to a worker again as an input.
    assert ran.outputs["model"]["content_type"] == "application/x-r-rds"
    downstream = await executor.execute_cell("c2", r_c2)
    assert downstream.success is True, downstream.error
    assert downstream.outputs["coef"]["preview"] == "3"
