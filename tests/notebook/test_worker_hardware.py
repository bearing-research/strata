"""The hardware a worker runs on, from the worker. Item 53.

A caller that wanted to know which accelerator, driver or CUDA version a
machine has used to submit a job that ran ``nvidia-smi``. A fake
``nvidia-smi`` on ``PATH`` stands in for the driver here, so the parsing and
the subprocess call are the real ones.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from strata.notebook import hardware
from strata.notebook.hardware import probe_hardware

_SMI = """#!/bin/sh
if [ "$1" = "--query-gpu=name,memory.total,driver_version" ]; then
  echo "NVIDIA A100-SXM4-80GB, 81920, 535.104.05"
  echo "NVIDIA A100-SXM4-80GB, 81920, 535.104.05"
  exit 0
fi
cat <<'EOT'
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
+---------------------------------------------------------------------------------------+
EOT
"""

A100 = {"name": "NVIDIA A100-SXM4-80GB", "memory_mb": 81920, "driver": "535.104.05"}


def _on_path(tmp_path: Path, monkeypatch, script: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text(script)
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    hardware.hardware_report.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_report():
    hardware.hardware_report.cache_clear()
    yield
    hardware.hardware_report.cache_clear()


class TestTheProbe:
    def test_a_gpu_machine_reports_each_accelerator_and_the_cuda_version(
        self, tmp_path, monkeypatch
    ):
        _on_path(tmp_path, monkeypatch, _SMI)

        report = probe_hardware()

        assert report["accelerators"] == [A100, A100]
        assert report["cuda"] == "12.2"
        assert report["cpus"] >= 1
        assert report["memory_mb"] > 0

    def test_without_the_driver_there_is_no_accelerator_claim(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))  # nowhere with an nvidia-smi

        report = probe_hardware()

        assert "accelerators" not in report
        assert "cuda" not in report

    def test_a_failing_driver_is_unknown_not_empty(self, tmp_path, monkeypatch):
        _on_path(tmp_path, monkeypatch, "#!/bin/sh\necho 'NVIDIA-SMI has failed' >&2\nexit 9\n")

        report = probe_hardware()

        assert "accelerators" not in report
        assert "cuda" not in report


def test_health_and_a_cell_run_report_the_same_hardware(
    tmp_path, monkeypatch, notebook_executor_server
):
    """The worker answers ``/health`` with the accelerator, and the artifact a
    cell run on it produces records the same values."""
    import asyncio

    import httpx

    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    _on_path(tmp_path, monkeypatch, _SMI)
    health = httpx.get(f"{notebook_executor_server['base_url']}/health").json()
    assert health["hardware"]["accelerators"] == [A100, A100]
    assert health["hardware"]["cuda"] == "12.2"

    nb = create_notebook(tmp_path / "nb", "Hardware")
    add_cell_to_notebook(nb, "up", None)
    write_cell(nb, "up", "value = 1")
    add_cell_to_notebook(nb, "down", "up")
    write_cell(nb, "down", "doubled = value * 2")
    session = NotebookSession(parse_notebook(nb), nb)
    session.notebook_state.workers = [
        WorkerSpec(
            name="gpu",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="gpu",
            config={"url": notebook_executor_server["execute_url"]},
        )
    ]
    session.notebook_state.worker = "gpu"

    result = asyncio.run(CellExecutor(session).execute_cell("up", "value = 1"))

    assert result.success, result.error
    assert result.execution_method == "executor"
    manager = session.get_artifact_manager()
    ((_, artifact),) = manager.list_cell_artifacts("up")
    params = json.loads(artifact.transform_spec)["params"]
    assert json.loads(params["hardware"]) == health["hardware"]
