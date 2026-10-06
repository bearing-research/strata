"""Workers running a cell in the notebook's locked environment."""

from __future__ import annotations

import http.server
import io
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import threading
import tomllib
from pathlib import Path

import pytest
import tomli_w

from strata.notebook import worker_env
from strata.notebook.env import uv_lock_key

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX worker environments")


def _notebook(tmp_path: Path) -> Path:
    """A notebook whose lock includes a local package the worker image lacks."""
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    package = tmp_path / "tinydep"
    (package / "tinydep").mkdir(parents=True)
    (package / "tinydep" / "__init__.py").write_text("ANSWER = 42\n")
    (package / "pyproject.toml").write_text('[project]\nname = "tinydep"\nversion = "0.1.0"\n')
    nb = create_notebook(tmp_path, "locked", initialize_environment=False)
    pyproject = tomllib.loads((nb / "pyproject.toml").read_text())
    pyproject["project"]["dependencies"].append("tinydep")
    # Absolute, so the lock names a place the worker can read too.
    pyproject["tool"]["uv"]["sources"] = {"tinydep": {"path": str(package)}}
    (nb / "pyproject.toml").write_text(tomli_w.dumps(pyproject))
    subprocess.run(["uv", "lock"], cwd=nb, check=True, capture_output=True)
    add_cell_to_notebook(nb, "c1", None)
    add_cell_to_notebook(nb, "c2", "c1")
    write_cell(nb, "c2", "shown = (answer, where)")
    return nb


def _session(nb: Path, worker_url: str, transport: str = "direct", strata_url: str = ""):
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    session = NotebookSession(parse_notebook(nb), nb)
    session.notebook_state.workers = [
        WorkerSpec(
            name="remote",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="r",
            config={
                "url": worker_url,
                "transport": transport,
                **({"strata_url": strata_url} if strata_url else {}),
            },
        )
    ]
    session.notebook_state.worker = "remote"
    return session


@pytest.mark.locked_environments
@pytest.mark.parametrize("transport", ["direct", "signed"])
async def test_a_cell_runs_in_the_notebooks_lock_and_a_second_dispatch_installs_nothing(
    tmp_path, monkeypatch, notebook_executor_server, notebook_personal_server, transport
):
    from strata.notebook.executor import CellExecutor
    from strata.notebook.writer import write_cell

    monkeypatch.setenv(worker_env.ENV_ROOT_VAR, str(tmp_path / "envs"))
    installs = []
    install = worker_env._install
    monkeypatch.setattr(
        worker_env, "_install", lambda *args: installs.append(args[2]) or install(*args)
    )
    nb = _notebook(tmp_path)
    source = "import sys, tinydep\nanswer = tinydep.ANSWER\nwhere = sys.executable"
    write_cell(nb, "c1", source)
    session = _session(
        nb,
        notebook_executor_server["execute_url"],
        transport,
        notebook_personal_server["base_url"] if transport == "signed" else "",
    )

    first = await CellExecutor(session).execute_cell("c1", source)
    assert first.success, first.error
    assert first.outputs["answer"]["preview"] == 42
    assert first.outputs["where"]["preview"].startswith(str(tmp_path / "envs"))
    assert len(installs) == 1

    again = await CellExecutor(session).execute_cell_force("c1", source)
    assert again.success, again.error
    assert again.outputs["where"]["preview"] == first.outputs["where"]["preview"]
    assert len(installs) == 1, "the same lock is not installed twice"

    manager = session.get_artifact_manager()
    artifacts = dict(manager.list_cell_artifacts("c1"))
    params = json.loads(artifacts["answer"].transform_spec)["params"]
    assert params["build_env"].startswith("cpython-"), "the worker's harness reports its build"


@pytest.mark.locked_environments
async def test_only_a_worker_that_advertises_it_is_sent_the_lock(tmp_path):
    from strata.notebook import workers
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps({"capabilities": {"features": {"cancel": True}}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return None

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    nb = _notebook(tmp_path)
    (nb / "uv.lock").write_text("version = 1\n")
    try:
        older = WorkerSpec(
            name="older",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="r",
            config={"url": f"http://127.0.0.1:{server.server_address[1]}/v1/execute"},
        )
        executor = CellExecutor(_session(nb, older.config.url))

        assert await workers.worker_advertises(older, "cancel") is True
        assert await workers.worker_advertises(older, "locked_environments") is False
        assert await executor._locked_environment(older) is None
    finally:
        server.shutdown()


@pytest.mark.locked_environments
class TestAWorkerThatCannotBeAsked:
    """A failed probe is not "no": treating it so ran the cell on the worker image while provenance
    recorded the lock's hash.
    """

    @staticmethod
    def _worker(url: str):
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        return WorkerSpec(
            name="unreachable",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="r",
            config={"url": url},
        )

    @pytest.mark.asyncio
    async def test_an_unreachable_worker_answers_nothing(self, tmp_path):
        from strata.notebook import workers

        # Nothing is listening on this port.
        worker = self._worker("http://127.0.0.1:1/v1/execute")

        assert await workers.worker_advertises(worker, "locked_environments") is None

    @pytest.mark.asyncio
    async def test_a_failed_probe_is_not_cached_as_an_answer(self, tmp_path):
        """One timed-out probe must not decide every cell dispatched in the next minute."""
        from strata.notebook import workers

        worker = self._worker("http://127.0.0.1:1/v1/execute")
        await workers.worker_advertises(worker, "locked_environments")

        health = workers._health_url_for_worker(worker)
        assert health not in workers._advertised_features

    @staticmethod
    def _serving(status: int, body: bytes = b"{}"):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return None

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    @pytest.mark.asyncio
    async def test_a_worker_with_no_health_route_has_answered(self, tmp_path):
        """A 404 is a live worker older than the health document; it keeps pre-feature behaviour."""
        from strata.notebook import workers

        server = self._serving(404)
        try:
            worker = self._worker(f"http://127.0.0.1:{server.server_address[1]}/v1/execute")

            assert await workers.worker_advertises(worker, "locked_environments") is False
        finally:
            server.shutdown()

    @pytest.mark.asyncio
    async def test_a_worker_that_is_up_and_unwell_has_not(self, tmp_path):
        """A 503 while starting is not an answer about its features."""
        from strata.notebook import workers

        server = self._serving(503)
        try:
            worker = self._worker(f"http://127.0.0.1:{server.server_address[1]}/v1/execute")

            assert await workers.worker_advertises(worker, "locked_environments") is None
        finally:
            server.shutdown()

    def test_the_probe_outlasts_the_worker_it_asks(self):
        """On a GPU box the first ``/health`` shells out to nvidia-smi with its own timeout; a
        shorter probe would refuse the opening cell on a freshly started machine.
        """
        from strata.notebook.hardware import _NVIDIA_SMI_TIMEOUT_SECONDS
        from strata.notebook.workers import _PROBE_TIMEOUT_SECONDS

        assert _PROBE_TIMEOUT_SECONDS > _NVIDIA_SMI_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_a_locked_notebook_refuses_rather_than_guess(self, tmp_path):
        from strata.notebook.executor import CellExecutor

        nb = _notebook(tmp_path)
        (nb / "uv.lock").write_text("version = 1\n")
        worker = self._worker("http://127.0.0.1:1/v1/execute")
        executor = CellExecutor(_session(nb, worker.config.url))

        with pytest.raises(RuntimeError, match="could not be asked"):
            await executor._locked_environment(worker)

    @pytest.mark.asyncio
    async def test_a_notebook_without_a_lock_does_not_care(self, tmp_path):
        """With no lock, an unanswered probe costs nothing and must not fail the cell."""
        from strata.notebook.executor import CellExecutor
        from strata.notebook.writer import add_cell_to_notebook, create_notebook

        nb = create_notebook(tmp_path / "plain", "unlocked", initialize_environment=False)
        add_cell_to_notebook(nb, "c1", None)
        (nb / "uv.lock").unlink(missing_ok=True)
        worker = self._worker("http://127.0.0.1:1/v1/execute")
        executor = CellExecutor(_session(nb, worker.config.url))

        assert await executor._locked_environment(worker) is None


class TestTheWorkerSide:
    def _spec(self, lock: str = "version = 1\n") -> dict[str, str]:
        return {
            "key": uv_lock_key(lock),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "lockfile": lock,
            "pyproject": '[project]\nname = "x"\nversion = "0"\n',
        }

    async def test_a_lock_that_does_not_match_its_key_is_refused(self):
        spec = self._spec()
        spec["lockfile"] = "version = 2\n"

        with pytest.raises(worker_env.WorkerEnvironmentError, match="does not match"):
            await worker_env.ensure_environment(spec)

    async def test_a_lock_that_is_not_toml_is_refused(self):
        spec = self._spec()
        spec["lockfile"] = "version = \n"

        with pytest.raises(worker_env.WorkerEnvironmentError, match="not TOML"):
            await worker_env.ensure_environment(spec)

    async def test_a_registry_supplies_the_environment_instead_of_an_install(
        self, tmp_path, monkeypatch
    ):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            python = f"{sys.executable}".encode()
            info = tarfile.TarInfo("bin/python")
            info.size = len(python)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(python))
        body = archive.getvalue()
        spec = self._spec()
        # This interpreter, so the expected build below is the one the worker probes.
        spec["python"] = sys.executable
        served = []

        class Registry(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                served.append(self.path)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return None

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Registry)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        monkeypatch.setenv(worker_env.ENV_ROOT_VAR, str(tmp_path / "envs"))
        monkeypatch.setenv(
            worker_env.REGISTRY_VAR, f"http://127.0.0.1:{server.server_address[1]}/envs"
        )
        monkeypatch.setattr(worker_env, "_install", lambda *a: pytest.fail("installed"))
        try:
            prepared = await worker_env.ensure_environment(spec)
            again = await worker_env.ensure_environment(spec)
        finally:
            server.shutdown()

        build = f"{sys.implementation.name}-{platform.python_version()}{sys.abiflags}"
        assert served == [f"/envs/{spec['key']}/{build}/{sysconfig.get_platform()}"], (
            "the fetch path names the interpreter build and platform the archive was built for"
        )
        assert prepared.installed is True
        assert again.installed is False
        assert prepared.python.exists()

    @staticmethod
    def _registry(status: int):
        served: list[str] = []

        class Registry(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                served.append(self.path)
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                return None

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Registry)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, served

    async def test_a_registry_miss_is_built_locally(self, tmp_path, monkeypatch):
        nb = _notebook(tmp_path)
        lock = (nb / "uv.lock").read_text()
        spec = {
            "key": uv_lock_key(lock),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "lockfile": lock,
            "pyproject": (nb / "pyproject.toml").read_text(),
        }
        server, served = self._registry(404)
        monkeypatch.setenv(worker_env.ENV_ROOT_VAR, str(tmp_path / "envs"))
        monkeypatch.setenv(
            worker_env.REGISTRY_VAR, f"http://127.0.0.1:{server.server_address[1]}/envs"
        )
        try:
            prepared = await worker_env.ensure_environment(spec)
            again = await worker_env.ensure_environment(spec)
        finally:
            server.shutdown()

        assert len(served) == 1 and served[0].startswith(f"/envs/{spec['key']}/")
        assert prepared.installed is True
        assert again.installed is False, "a locally built environment is reused, not refetched"
        answer = subprocess.run(
            [str(prepared.python), "-c", "import tinydep; print(tinydep.ANSWER)"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert answer.stdout.strip() == "42"

    async def test_a_failing_registry_is_an_error_not_a_build(self, tmp_path, monkeypatch):
        server, served = self._registry(500)
        monkeypatch.setenv(worker_env.ENV_ROOT_VAR, str(tmp_path / "envs"))
        monkeypatch.setenv(
            worker_env.REGISTRY_VAR, f"http://127.0.0.1:{server.server_address[1]}/envs"
        )
        monkeypatch.setattr(worker_env, "_install", lambda *a: pytest.fail("installed"))
        try:
            with pytest.raises(worker_env.WorkerEnvironmentError, match="could not fetch"):
                await worker_env.ensure_environment(self._spec())
        finally:
            server.shutdown()

        assert len(served) == 1

    async def test_a_registry_serving_a_broken_archive_is_an_environment_error(
        self, tmp_path, monkeypatch
    ):
        """The route answers only WorkerEnvironmentError with the locked-environment body."""
        body = b"\x1f\x8b not a gzip member"

        class Registry(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return None

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Registry)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        monkeypatch.setenv(worker_env.ENV_ROOT_VAR, str(tmp_path / "envs"))
        monkeypatch.setenv(
            worker_env.REGISTRY_VAR, f"http://127.0.0.1:{server.server_address[1]}/envs"
        )
        monkeypatch.setattr(
            worker_env, "_interpreter", lambda python: (Path(sys.executable), "cpython-3 posix")
        )
        monkeypatch.setattr(worker_env, "_install", lambda *a: pytest.fail("installed"))
        try:
            with pytest.raises(worker_env.WorkerEnvironmentError, match="could not unpack"):
                await worker_env.ensure_environment(self._spec())
        finally:
            server.shutdown()

    @pytest.mark.parametrize(
        ("step", "message"),
        [
            (lambda tmp: worker_env._interpreter("3.13"), "finding Python 3.13"),
            (
                lambda tmp: worker_env._install(
                    {"pyproject": "", "lockfile": ""}, Path(sys.executable), tmp / "env"
                ),
                "uv sync",
            ),
            (lambda tmp: worker_env._r_build("Rscript"), "R its version"),
            (lambda tmp: worker_env._restore_r("Rscript", "{}", tmp / "lib"), "renv::restore"),
        ],
        ids=["interpreter", "install", "r_build", "restore_r"],
    )
    def test_a_subprocess_timeout_names_its_step(self, tmp_path, monkeypatch, step, message):
        def _timeout(args, **kwargs):
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))

        monkeypatch.setattr(worker_env.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(worker_env.subprocess, "run", _timeout)

        with pytest.raises(worker_env.WorkerEnvironmentError, match=f"{message}.*timed out"):
            step(tmp_path)


class TestAnEnvironmentIsCompleteWhenItRuns:
    """The marker lets a worker reuse a directory without installing, so an archive with no
    interpreter where the worker looks must not be marked complete.
    """

    @pytest.mark.asyncio
    async def test_an_archive_without_an_interpreter_is_not_kept(self, tmp_path, monkeypatch):
        import tarfile

        from strata.notebook.worker_env import (
            COMPLETE_MARKER,
            WorkerEnvironmentError,
            ensure_environment,
            env_root,
        )

        empty = tmp_path / "env"
        (empty / "lib").mkdir(parents=True)
        archive = tmp_path / "env.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(empty, arcname=".")
        served = tmp_path / "served"
        served.mkdir()
        monkeypatch.setenv("STRATA_WORKER_ENV_ROOT", str(tmp_path / "worker-envs"))

        def _fetch(url, env_dir):
            with tarfile.open(archive) as tar:
                tar.extractall(env_dir, filter="data")
            return True

        monkeypatch.setattr("strata.notebook.worker_env._fetch", _fetch)
        monkeypatch.setenv("STRATA_WORKER_ENV_REGISTRY_URL", "http://registry")
        lock = "version = 1\n"
        spec = {
            "key": uv_lock_key(lock),
            "python": "",
            "lockfile": lock,
            "pyproject": "[project]\nname = 'nb'\nversion = '0'\n",
        }

        with pytest.raises(WorkerEnvironmentError, match="no interpreter"):
            await ensure_environment(spec)

        built = [p for p in env_root().iterdir() if p.is_dir()]
        assert built, "the directory it unpacked into is still there to be replaced"
        assert not any((p / COMPLETE_MARKER).exists() for p in built), (
            "an environment with no interpreter was marked complete"
        )

        # A registry that has been fixed is fetched again rather than skipped.
        fetched: list[str] = []

        def _good_fetch(url, env_dir):
            fetched.append(url)
            (env_dir / "bin").mkdir(parents=True, exist_ok=True)
            (env_dir / "bin" / "python").write_text("#!/bin/sh\n")
            return True

        monkeypatch.setattr("strata.notebook.worker_env._fetch", _good_fetch)
        prepared = await ensure_environment(spec)

        assert len(fetched) == 1 and fetched[0].startswith(f"http://registry/{spec['key']}/")
        assert prepared.installed is True


class TestAWorkerAnswersForTheMachineItIsOn:
    """The server believes a worker's answer about building locked environments, so the answer must
    reflect the machine: an image without uv must not claim it.
    """

    def _features(self, monkeypatch, uv_path):
        from fastapi.testclient import TestClient

        from strata.notebook.remote_executor import create_notebook_executor_app

        monkeypatch.delenv("STRATA_WORKER_TOKEN", raising=False)
        real_which = shutil.which
        monkeypatch.setattr(
            "strata.notebook.remote_executor.shutil.which",
            lambda name, *a, **kw: uv_path if name == "uv" else real_which(name, *a, **kw),
        )
        client = TestClient(create_notebook_executor_app())
        return client.get("/health").json()["capabilities"]["features"]

    def test_without_uv_it_says_so(self, monkeypatch):
        features = self._features(monkeypatch, None)

        assert features["locked_environments"] is False, (
            "a worker with no uv claimed it could build a locked environment; "
            "the server would send one and every python cell would fail"
        )

    def test_with_uv_it_offers_the_feature(self, monkeypatch):
        features = self._features(monkeypatch, "/usr/local/bin/uv")

        assert features["locked_environments"] is True
