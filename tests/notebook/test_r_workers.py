"""R cells on remote workers.

The first tests need no R: a worker without ``Rscript`` refuses an R cell by
name, and a stand-in ``Rscript`` shows which harness it gets and how the
notebook's ``renv.lock`` is restored. The rest run R on a worker over both
transports and read the result from Python.
"""

from __future__ import annotations

import http.server
import io
import json
import shutil
import sys
import tarfile
import threading
from pathlib import Path

import httpx
import pytest

from strata.notebook import worker_env
from strata.notebook.env import renv_lock_key
from strata.notebook.executor import CellExecutor
from strata.notebook.models import WorkerBackendType, WorkerSpec
from strata.notebook.remote_executor import (
    NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
    create_notebook_executor_app,
)
from tests.conftest import prepared_venv
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
    """An ``Rscript`` first on PATH that answers like ``harness.R`` and records its argv.

    Returns the file the argv is written to.
    """
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
    prepared_venv(notebook_dir)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
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


# --- The notebook's renv.lock on a worker ---

_LOCK = '{"R": {"Version": "4.4.0"}, "Packages": {"jsonlite": {"Version": "2.0.0"}}}\n'
_BUILD = "R-4.4.0 aarch64-apple-darwin20"
_RENV_FIXTURE = Path(__file__).parent / "fixtures" / "renv_jsonlite"
posix_stand_in = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in is a script")


def _r_environment(lock: str = _LOCK) -> dict[str, str]:
    return {"key": renv_lock_key(lock), "lockfile": lock}


def _stand_in_rscript_with_renv(tmp_path: Path, monkeypatch, *, renv: bool = True) -> Path:
    """An ``Rscript`` first on PATH with renv: it answers the build probe, restores a lock by
    writing each package's DESCRIPTION into the library it is given, and runs a cell as
    ``harness.R`` would.

    Returns its log: a JSON line per restore and per cell, the cell's with its ``R_LIBS``.
    """
    log = tmp_path / "rscript.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    rscript = bin_dir / "Rscript"
    rscript.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, pathlib\n"
        f"log = pathlib.Path({str(log)!r})\n"
        "args = [a for a in sys.argv[1:] if a != '--vanilla']\n"
        "if args[0] == '-e':\n"
        "    if 'R.version' in args[1]:\n"
        f"        print({_BUILD!r}, end='')\n"
        "    elif 'requireNamespace' in args[1]:\n"
        f"        sys.exit(0 if {renv!r} else 1)\n"
        "    elif 'renv::restore' in args[1]:\n"
        "        library = pathlib.Path(os.environ['STRATA_R_LIBRARY'])\n"
        "        for name in json.loads(pathlib.Path('renv.lock').read_text())['Packages']:\n"
        "            (library / name).mkdir(parents=True, exist_ok=True)\n"
        "            (library / name / 'DESCRIPTION').write_text('Package: ' + name)\n"
        "        with log.open('a') as f:\n"
        "            f.write(json.dumps({'restore': str(library)}) + '\\n')\n"
        "    else:\n"
        "        sys.exit(2)\n"
        "    sys.exit(0)\n"
        "with log.open('a') as f:\n"
        "    f.write(json.dumps({'cell': args[0], 'R_LIBS': os.environ.get('R_LIBS')}) + '\\n')\n"
        "manifest = pathlib.Path(args[1])\n"
        "out = pathlib.Path(json.loads(manifest.read_text())['output_dir'])\n"
        "(out / 'out.json').write_text('1')\n"
        "(manifest.parent / 'harness-result.json').write_text(json.dumps({\n"
        "    'success': True, 'stdout': '', 'stderr': '', 'mutation_warnings': [],\n"
        "    'variables': {'out': {'content_type': 'json/object', 'file': 'out.json',\n"
        "                          'preview': 1}}}))\n"
    )
    rscript.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}")
    monkeypatch.setenv(worker_env.ENV_ROOT_VAR, str(tmp_path / "envs"))
    return log


def _calls(log: Path, kind: str) -> list[dict[str, str]]:
    if not log.exists():
        return []
    return [entry for entry in map(json.loads, log.read_text().splitlines()) if kind in entry]


async def _execute_r(environment: dict[str, str]) -> httpx.Response:
    metadata = {
        "protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": "out <- 1\n",
        "language": "r",
        "environment": environment,
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


async def _features() -> dict:
    transport = httpx.ASGITransport(app=create_notebook_executor_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        return (await client.get("/health")).json()["capabilities"]["features"]


@posix_stand_in
class TestAWorkerRestoresTheRLock:
    async def test_once_per_lock_and_every_cell_runs_against_that_library(
        self, tmp_path, monkeypatch
    ):
        log = _stand_in_rscript_with_renv(tmp_path, monkeypatch)

        first = await _execute_r(_r_environment())
        second = await _execute_r(_r_environment())

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        restores = _calls(log, "restore")
        assert len(restores) == 1, "the same lock is not restored twice"
        library = Path(restores[0]["restore"])
        assert library.parent == tmp_path / "envs" / "r"
        cells = _calls(log, "cell")
        assert [cell["R_LIBS"] for cell in cells] == [str(library)] * 2, (
            "the cell runs with the restored library first on its path"
        )

    async def test_another_lock_gets_its_own_library(self, tmp_path, monkeypatch):
        log = _stand_in_rscript_with_renv(tmp_path, monkeypatch)
        other = _LOCK.replace("}}}", '}, "ggplot2": {"Version": "3.5.0"}}}')

        assert (await _execute_r(_r_environment())).status_code == 200
        assert (await _execute_r(_r_environment(other))).status_code == 200

        libraries = [Path(entry["restore"]) for entry in _calls(log, "restore")]
        assert len(set(libraries)) == 2
        assert (libraries[1] / "ggplot2" / "DESCRIPTION").exists()

    async def test_a_lock_that_does_not_match_its_key_is_refused(self, tmp_path, monkeypatch):
        log = _stand_in_rscript_with_renv(tmp_path, monkeypatch)
        environment = _r_environment()
        environment["lockfile"] = _LOCK.replace("2.0.0", "1.8.0")

        response = await _execute_r(environment)

        assert response.status_code == 500
        assert "does not match environment.key" in response.json()["error"]
        assert _calls(log, "restore") == [] and _calls(log, "cell") == []

    async def test_health_says_whether_renv_can_restore(self, tmp_path, monkeypatch):
        _stand_in_rscript_with_renv(tmp_path, monkeypatch, renv=False)
        assert (await _features())["locked_r_environments"] is False, (
            "a worker whose R lacks renv claimed it could restore a lock"
        )

        _stand_in_rscript_with_renv(tmp_path / "with", monkeypatch)
        assert (await _features())["locked_r_environments"] is True


async def test_a_worker_without_rscript_does_not_offer_r_locks(monkeypatch):
    monkeypatch.setattr(
        shutil, "which", lambda name: None if name == "Rscript" else _REAL_WHICH(name)
    )

    assert (await _features())["locked_r_environments"] is False


def _registry(status: int, body: bytes = b""):
    served: list[str] = []

    class Registry(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            served.append(self.path)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return None

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Registry)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, served


def _archive(*packages: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in packages:
            description = f"Package: {name}\n".encode()
            info = tarfile.TarInfo(f"{name}/DESCRIPTION")
            info.size = len(description)
            tar.addfile(info, io.BytesIO(description))
    return buffer.getvalue()


@posix_stand_in
class TestTheRRegistry:
    async def _prepare(self, tmp_path, monkeypatch, status: int, body: bytes = b""):
        log = _stand_in_rscript_with_renv(tmp_path, monkeypatch)
        server, served = _registry(status, body)
        monkeypatch.setenv(
            worker_env.REGISTRY_VAR, f"http://127.0.0.1:{server.server_address[1]}/envs"
        )
        rscript = shutil.which("Rscript")
        assert rscript is not None
        try:
            prepared = await worker_env.ensure_r_library(_r_environment(), rscript)
            again = await worker_env.ensure_r_library(_r_environment(), rscript)
        finally:
            server.shutdown()
        return log, served, prepared, again

    async def test_supplies_the_library_instead_of_a_restore(self, tmp_path, monkeypatch):
        log, served, prepared, again = await self._prepare(
            tmp_path, monkeypatch, 200, _archive("jsonlite")
        )

        key = renv_lock_key(_LOCK)
        assert served == [f"/envs/r/{key}/R-4.4.0/aarch64-apple-darwin20"], (
            "the fetch path names the R version and platform the library was built for"
        )
        assert _calls(log, "restore") == []
        assert (prepared.library / "jsonlite" / "DESCRIPTION").exists()
        assert prepared.installed is True
        assert again.installed is False

    async def test_a_miss_is_restored_locally(self, tmp_path, monkeypatch):
        log, served, prepared, again = await self._prepare(tmp_path, monkeypatch, 404)

        assert len(served) == 1
        assert [Path(entry["restore"]) for entry in _calls(log, "restore")] == [prepared.library]
        assert again.installed is False, "a restored library is reused, not refetched"

    async def test_a_failing_registry_is_an_error_not_a_restore(self, tmp_path, monkeypatch):
        with pytest.raises(worker_env.WorkerEnvironmentError, match="could not fetch"):
            await self._prepare(tmp_path, monkeypatch, 500)

        assert _calls(tmp_path / "rscript.log", "restore") == []

    async def test_an_archive_short_of_the_lock_is_not_kept(self, tmp_path, monkeypatch):
        with pytest.raises(worker_env.WorkerEnvironmentError, match="lacks jsonlite"):
            await self._prepare(tmp_path, monkeypatch, 200, _archive("cli"))

        root = tmp_path / "envs" / "r"
        assert not list(root.glob(f"*/{worker_env.COMPLETE_MARKER}")), (
            "a library without the lock's packages was marked complete"
        )


@posix_stand_in
@pytest.mark.parametrize("transport", ["direct", "signed"])
async def test_the_server_sends_the_renv_lock_to_a_worker_that_restores_it(
    tmp_path, monkeypatch, transport, notebook_executor_server, notebook_build_server
):
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import create_notebook

    log = _stand_in_rscript_with_renv(tmp_path, monkeypatch)
    notebook_dir = create_notebook(tmp_path, "r-locked", initialize_environment=False)
    (notebook_dir / "renv.lock").write_text(_LOCK)
    prepared_venv(notebook_dir)
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

    result, _, _, _ = await CellExecutor(session)._dispatch_http_executor(
        worker, "out <- 1\n", {}, [], output_dir, {}, 60.0, language="r"
    )

    assert result["success"] is True
    restores = _calls(log, "restore")
    assert len(restores) == 1
    assert _calls(log, "cell")[0]["R_LIBS"] == restores[0]["restore"]


class TestWhichWorkerGetsTheRLock:
    @staticmethod
    def _notebook(tmp_path: Path, lock: str | None = _LOCK):
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import create_notebook

        notebook_dir = create_notebook(tmp_path, "r-lock", initialize_environment=False)
        if lock is not None:
            (notebook_dir / "renv.lock").write_text(lock)
        return CellExecutor(NotebookSession(parse_notebook(notebook_dir), notebook_dir))

    @staticmethod
    def _worker(url: str) -> WorkerSpec:
        return WorkerSpec(name="w", backend=WorkerBackendType.EXECUTOR, config={"url": url})

    async def test_only_one_that_advertises_r_locks(self, tmp_path, monkeypatch):
        from strata.notebook import workers

        offered: dict[str, bool] = {}

        async def _advertises(worker, feature):
            return offered.get(feature, False)

        monkeypatch.setattr(workers, "worker_advertises", _advertises)
        executor = self._notebook(tmp_path)
        worker = self._worker("http://worker/v1/execute")

        offered["locked_environments"] = True
        assert await executor._locked_environment(worker, "r") is None, (
            "a worker that builds uv locks was sent an renv.lock it would ignore"
        )
        offered["locked_r_environments"] = True
        assert await executor._locked_environment(worker, "r") == _r_environment()

    async def test_a_worker_that_cannot_be_asked_is_refused(self, tmp_path):
        executor = self._notebook(tmp_path)

        with pytest.raises(RuntimeError, match="could not be asked"):
            await executor._locked_environment(self._worker("http://127.0.0.1:1/v1/execute"), "r")

    async def test_a_notebook_without_an_renv_lock_does_not_care(self, tmp_path):
        executor = self._notebook(tmp_path, lock=None)

        assert (
            await executor._locked_environment(self._worker("http://127.0.0.1:1/v1/execute"), "r")
            is None
        )


@skip_if_no_r
@pytest.mark.skipif(
    _REAL_WHICH("Rscript") is None or not worker_env.renv_available(_REAL_WHICH("Rscript")),
    reason="needs R with renv in its library",
)
async def test_a_real_lock_is_restored_on_the_worker_once(
    tmp_path, monkeypatch, notebook_executor_server
):
    from strata.notebook.models import CellLanguage
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    monkeypatch.setenv(worker_env.ENV_ROOT_VAR, str(tmp_path / "envs"))
    restores: list[Path] = []
    real_restore = worker_env._restore_r

    def counted(rscript: str, lockfile: str, library: Path) -> None:
        restores.append(library)
        real_restore(rscript, lockfile, library)

    monkeypatch.setattr(worker_env, "_restore_r", counted)
    source = "library(jsonlite)\nout <- as.character(toJSON(list(ok = TRUE), auto_unbox = TRUE))\n"
    notebook_dir = create_notebook(tmp_path, "r-real", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1", None, language="r")
    write_cell(notebook_dir, "c1", source)
    shutil.copy(_RENV_FIXTURE / "renv.lock", notebook_dir / "renv.lock")
    prepared_venv(notebook_dir)
    state = parse_notebook(notebook_dir)
    state.cells[0].language = CellLanguage.R
    session = NotebookSession(state, notebook_dir)
    _on_worker(session, "c1", {"url": notebook_executor_server["execute_url"]})

    ran = await CellExecutor(session).execute_cell("c1", source)
    again = await CellExecutor(session).execute_cell_force("c1", source)

    assert ran.success is True, ran.error
    assert ran.outputs["out"]["preview"] == '{"ok":true}'
    assert again.success is True, again.error
    assert len(restores) == 1, "the same lock is not restored twice"
    assert (restores[0] / "jsonlite" / "DESCRIPTION").exists()
