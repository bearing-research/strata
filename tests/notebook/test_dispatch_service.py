"""The dispatch service against fakes: no worker, no server and no store behind it."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from vestigo.provenance import derive_subkey

from strata.notebook import output_dirs
from strata.notebook.cell_paths import dispatch
from strata.notebook.cell_paths.context import RemoteExecutionError
from strata.notebook.cell_paths.dispatch import (
    DispatchContext,
    DispatchRoutes,
    EmbeddedContext,
    await_accepted_job,
    broadcast_remote_phase,
    cancel_remote_execution,
    dispatch_embedded,
    dispatch_execution,
    dispatch_http,
    dispatch_signed,
    execute_manifest_url,
    extract_remote_error,
    locked_environment,
    stage_signed_transport_inputs,
)
from strata.notebook.executor import CellExecutor
from strata.notebook.models import MountMode, MountSpec, WorkerBackendType
from strata.notebook.parser import parse_notebook
from strata.notebook.remote_bundle import pack_notebook_output_bundle
from strata.notebook.remote_executor import NOTEBOOK_EXECUTOR_PROTOCOL_VERSION
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook
from strata.types import EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION


def _worker(url="http://worker:9/v1/execute", transport="direct", **config):
    fields = {
        "url": url,
        "transport": transport,
        "token": None,
        "token_env": None,
        "strata_url": None,
    }
    return SimpleNamespace(
        name="w",
        backend=WorkerBackendType.EXECUTOR,
        config=SimpleNamespace(**{**fields, **config}),
    )


def _context(**overrides):
    phases = []

    async def phase(cell_id, worker_spec, name, *, queue=(None, None)):
        phases.append((cell_id, name, queue))

    fields = {
        "origin": "notebook",
        "credential_resolver": lambda: SimpleNamespace(
            ensure_allowed=lambda _name: None,
            ensure_options_allowed=lambda _scheme, _options: None,
            ensure_writable=lambda _scheme, _credential: None,
        ),
        "note_worker_mount_bytes": lambda *_args: None,
        "timeout_message": lambda cell_id, seconds: f"{cell_id} timed out after {seconds}",
        "broadcast_remote_phase": phase,
        "lake_config": lambda: SimpleNamespace(worker_provisioning_timeout_seconds=50.0),
        "received_output_files": {},
    }
    context = DispatchContext(**{**fields, **overrides})
    return context, phases


def _host(tmp_path=Path("notebook")):
    return SimpleNamespace(
        id="session",
        path=tmp_path,
        notebook_state=SimpleNamespace(id="nb"),
        get_artifact_manager=lambda: "artifact manager",
    )


def _bundle(tmp_path, **result):
    """The bytes of a real output bundle."""
    path = tmp_path / "answer.tar"
    (tmp_path / "packed").mkdir(exist_ok=True)
    pack_notebook_output_bundle(
        path, {"success": True, "variables": {}, **result}, tmp_path / "packed"
    )
    return path.read_bytes()


@pytest.fixture
def http(monkeypatch):
    """Real httpx over a transport that answers from a list and records what it was asked."""
    seen = SimpleNamespace(requests=[], answers=[], timeouts=[])

    def answer(request):
        seen.requests.append(request)
        reply = seen.answers.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    real = httpx.AsyncClient

    def client(**kwargs):
        seen.timeouts.append(kwargs.get("timeout"))
        return real(transport=httpx.MockTransport(answer), **kwargs)

    monkeypatch.setattr(dispatch.httpx, "AsyncClient", client)
    return seen


def _metadata(request):
    """The JSON part of a direct transport's multipart body."""
    body = request.content
    match = re.search(rb'name="metadata".*?\r\n\r\n(.*?)\r\n--', body, re.S)
    assert match is not None
    return json.loads(match.group(1))


class TestRoutes:
    """Which backend a run goes to, and what it hears."""

    @staticmethod
    def _routes():
        seen = []

        def route(name):
            async def call(*args, **kwargs):
                seen.append((name, args, kwargs))
                return {"from": name}, Path("out"), name, {}

            return call

        return DispatchRoutes(route("local"), route("embedded"), route("http")), seen

    @staticmethod
    async def _dispatch(routes, worker, **overrides):
        return await dispatch_execution(
            _host(),
            routes,
            worker,
            "x = 1\n",
            {"i": {}},
            [],
            Path("out"),
            Path("result"),
            Path("venv"),
            {"K": "V"},
            7.0,
            **{
                "remote_build_id": "b1",
                "mutation_defines": ["d"],
                "tables": {"t": {}},
                "cell_id": "c",
                "cell_provenance_hash": "key",
                "local_run": "run",
                "restore": {"r": 1},
                "resolved_mounts": {"m": 1},
                **overrides,
            },
        )

    async def test_a_local_worker_runs_here(self):
        routes, seen = self._routes()

        await self._dispatch(routes, SimpleNamespace(backend=WorkerBackendType.LOCAL))

        name, args, kwargs = seen[0]
        assert name == "local"
        assert args == (
            "x = 1\n",
            {"i": {}},
            [],
            Path("out"),
            Path("venv"),
            {"K": "V"},
            7.0,
        )
        assert kwargs == {
            "mutation_defines": ["d"],
            "tables": {"t": {}},
            "cell_id": "c",
            "local_run": "run",
            "restore": {"r": 1},
            "resolved_mounts": {"m": 1},
        }

    async def test_an_embedded_worker_gets_the_result_directory(self):
        routes, seen = self._routes()

        await self._dispatch(routes, _worker("embedded://local"))

        name, args, kwargs = seen[0]
        assert name == "embedded"
        assert args[4] == Path("result")
        assert kwargs == {
            "mutation_defines": ["d"],
            "tables": {"t": {}},
            "cell_id": "c",
            "resolved_mounts": {"m": 1},
        }

    async def test_an_http_worker_gets_the_build_and_the_cells_key(self):
        routes, seen = self._routes()

        await self._dispatch(routes, _worker())

        name, args, kwargs = seen[0]
        assert name == "http"
        assert args[0].name == "w"
        assert kwargs == {
            "remote_build_id": "b1",
            "mutation_defines": ["d"],
            "tables": {"t": {}},
            "cell_id": "c",
            "cell_provenance_hash": "key",
        }

    async def test_any_other_backend_is_refused(self):
        routes, seen = self._routes()
        worker = SimpleNamespace(
            backend=WorkerBackendType.EXECUTOR, config=SimpleNamespace(url="ftp://x")
        )

        with pytest.raises(RuntimeError, match="Unsupported worker backend"):
            await self._dispatch(routes, worker)

        assert seen == []


class TestEmbedded:
    """A bundle-based run on this machine."""

    @staticmethod
    def _context(seen, result):
        @contextmanager
        def run_token(_seconds):
            yield "token"

        def write_manifest(*args, **kwargs):
            seen.manifest = (args, kwargs)
            return Path("run/manifest.json")

        async def run_harness(*args, **kwargs):
            seen.harness = (args, kwargs)
            return result

        async def prepare_mounts(specs, cell_id):
            seen.prepared = (specs, cell_id)
            return {"prepared": 1}

        return EmbeddedContext(prepare_mounts, run_token, write_manifest, run_harness)

    @staticmethod
    async def _run(context, tmp_path, resolved_mounts=None):
        (tmp_path / "out").mkdir()
        return await dispatch_embedded(
            context,
            "x = 1\n",
            {},
            ["spec"],
            tmp_path / "out",
            tmp_path / "result",
            Path("venv"),
            {"K": "V"},
            7.0,
            mutation_defines=["d"],
            tables={"t": {}},
            cell_id="c",
            resolved_mounts=resolved_mounts,
        )

    async def test_it_prepares_mounts_only_when_the_caller_has_not(self, tmp_path):
        seen = SimpleNamespace(prepared=None)
        context = self._context(seen, {"success": True, "variables": {}})

        _result, _output, _method, mounts = await self._run(context, tmp_path, {"given": 1})

        assert seen.prepared is None
        assert mounts == {"given": 1}

    async def test_it_prepares_them_otherwise(self, tmp_path):
        seen = SimpleNamespace(prepared=None)
        context = self._context(seen, {"success": True, "variables": {}})

        _result, _output, _method, mounts = await self._run(context, tmp_path)

        assert seen.prepared == (["spec"], "c")
        assert mounts == {"prepared": 1}

    async def test_the_result_is_unpacked_where_the_cell_cannot_reach(self, tmp_path):
        seen = SimpleNamespace(prepared=None)
        context = self._context(seen, {"success": True, "variables": {}, "stdout": "printed"})

        result, output, method, _mounts = await self._run(context, tmp_path)

        assert output == tmp_path / "result"
        assert method == "executor"
        assert result["stdout"] == "printed"

    async def test_the_run_gets_the_manifest_and_its_token(self, tmp_path):
        seen = SimpleNamespace(prepared=None)
        context = self._context(seen, {"success": True, "variables": {}})

        await self._run(context, tmp_path)

        assert seen.manifest[1] == {
            "mutation_defines": ["d"],
            "tables": {"t": {}},
            "cell_id": "c",
            "run_token": "token",
        }
        assert seen.harness == (
            (Path("run/manifest.json"), Path("venv"), 7.0),
            {"run_token": "token"},
        )


class TestLockedEnvironment:
    SPEC = {"lock": "contents"}

    @pytest.fixture
    def advertised(self, monkeypatch):
        seen = SimpleNamespace(answer=True, asked=[])

        async def worker_advertises(_worker, feature):
            seen.asked.append(feature)
            return seen.answer

        monkeypatch.setattr("strata.notebook.workers.worker_advertises", worker_advertises)
        monkeypatch.setattr(
            "strata.notebook.worker_env.environment_spec",
            lambda path, minor: {"python": str(path), **self.SPEC},
        )
        monkeypatch.setattr(
            "strata.notebook.worker_env.r_environment_spec",
            lambda path: {"r": str(path), **self.SPEC},
        )
        return seen

    async def test_python_asks_about_uv_locks(self, advertised):
        spec = await locked_environment(_host(), _worker())

        assert spec["python"] == "notebook"
        assert advertised.asked == ["locked_environments"]

    async def test_r_asks_about_renv_locks(self, advertised):
        spec = await locked_environment(_host(), _worker(), "r")

        assert spec["r"] == "notebook"
        assert advertised.asked == ["locked_r_environments"]

    async def test_another_language_has_no_lock(self, advertised):
        assert await locked_environment(_host(), _worker(), "julia") is None
        assert advertised.asked == []

    async def test_no_lock_asks_nothing(self, advertised, monkeypatch):
        monkeypatch.setattr("strata.notebook.worker_env.environment_spec", lambda *_args: None)

        assert await locked_environment(_host(), _worker()) is None
        assert advertised.asked == []

    async def test_a_worker_without_the_feature_uses_its_own(self, advertised):
        advertised.answer = False

        assert await locked_environment(_host(), _worker()) is None

    async def test_a_worker_that_cannot_be_asked_is_refused(self, advertised):
        advertised.answer = None

        with pytest.raises(RuntimeError, match="could not be asked"):
            await locked_environment(_host(), _worker())


class TestDirect:
    """The direct transport: multipart up, a bundle back."""

    @pytest.fixture
    def features(self, monkeypatch):
        seen = SimpleNamespace(answers={}, env=None)

        async def worker_advertises(_worker, feature):
            return seen.answers.get(feature, True)

        monkeypatch.setattr("strata.notebook.workers.worker_advertises", worker_advertises)

        async def locked(_host, _worker, _language):
            return seen.env

        monkeypatch.setattr(dispatch, "locked_environment", locked)
        return seen

    @staticmethod
    async def _run(tmp_path, *, worker=None, context=None, input_specs=None, mounts=(), **kwargs):
        context = context or _context()[0]
        (tmp_path / "out").mkdir(exist_ok=True)
        return await dispatch_http(
            _host(tmp_path),
            context,
            worker or _worker(),
            kwargs.pop("source", "x = 1\n"),
            input_specs or {},
            list(mounts),
            tmp_path / "out",
            {"K": "V"},
            7.0,
            **{"cell_id": "c", **kwargs},
        )

    @staticmethod
    def _bundle_reply(tmp_path, **headers):
        return httpx.Response(200, content=_bundle(tmp_path, stdout="printed"), headers=headers)

    # Refusals before anything is sent

    async def test_an_older_worker_is_refused_output_dirs(self, http, features, tmp_path):
        features.answers["output_dirs"] = False

        with pytest.raises(RuntimeError, match="does not return # @output_dir"):
            await self._run(tmp_path, source="# @output_dir out\nx = 1\n")

        assert http.requests == []

    async def test_a_worker_that_cannot_say_is_refused_output_dirs(self, http, features, tmp_path):
        features.answers["output_dirs"] = None

        with pytest.raises(RuntimeError, match="could not be asked whether it returns"):
            await self._run(tmp_path, source="# @output_dir out\nx = 1\n")

    @pytest.mark.parametrize(
        ("answer", "message"),
        [(False, "cannot take 'f'"), (None, "could not be asked whether it takes")],
    )
    async def test_a_folder_input_needs_a_worker_that_takes_one(
        self, http, features, tmp_path, answer, message
    ):
        features.answers["folder_inputs"] = answer
        specs = {"f": {"content_type": output_dirs.CONTENT_TYPE, "file": "f.dir"}}

        with pytest.raises(RuntimeError, match=message):
            await self._run(tmp_path, input_specs=specs)

        assert http.requests == []

    async def test_a_folder_input_travels_as_a_tar(self, http, features, tmp_path, monkeypatch):
        specs = {"f": {"content_type": output_dirs.CONTENT_TYPE, "file": "f.dir"}}
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "f.tar").write_text("tar")
        monkeypatch.setattr(
            dispatch,
            "_pack_folder_inputs",
            lambda _specs, _dir: {"f": {"content_type": "tar", "file": "f.tar"}},
        )
        http.answers.append(self._bundle_reply(tmp_path))

        await self._run(tmp_path, input_specs=specs)

        assert _metadata(http.requests[0])["inputs"][0]["file"] == "f.tar"

    @pytest.mark.parametrize("uri", ["file:///data", "/data", "../data"])
    async def test_a_file_mount_cannot_go_to_a_worker(self, http, features, tmp_path, uri):
        mount = MountSpec(name="m", uri=uri)

        with pytest.raises(RuntimeError, match="mount 'm' is a local path"):
            await self._run(tmp_path, mounts=[mount])

    async def test_credentials_are_checked_before_the_names_leave(self, http, features, tmp_path):
        seen = []
        resolver = SimpleNamespace(
            ensure_allowed=seen.append,
            ensure_options_allowed=lambda scheme, options: seen.append((scheme, options)),
            ensure_writable=lambda scheme, credential: seen.append(("rw", scheme, credential)),
        )
        context, _phases = _context(credential_resolver=lambda: resolver)
        mount = MountSpec(
            name="m",
            uri="s3://bucket/key",
            credential="cred",
            mode=MountMode.READ_WRITE,
            options={"anon": "true"},
        )
        http.answers.append(self._bundle_reply(tmp_path))

        await self._run(tmp_path, context=context, mounts=[mount])

        assert seen == ["cred", ("s3", {"anon": "true"}), ("rw", "s3", "cred")]

    async def test_a_worker_with_no_url_is_refused(self, http, features, tmp_path):
        with pytest.raises(RuntimeError, match="is missing config.url"):
            await self._run(tmp_path, worker=_worker(url=""))

    # What is sent

    async def test_the_request_names_the_run(self, http, features, tmp_path):
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "x.pickle").write_text("value")
        http.answers.append(self._bundle_reply(tmp_path))
        context, _phases = _context(origin="run")
        mount = MountSpec(name="m", uri="s3://bucket/key")

        await self._run(
            tmp_path,
            context=context,
            input_specs={"x": {"content_type": "pickle/object", "file": "x.pickle"}},
            mounts=[mount],
            mutation_defines=["d"],
            tables={"t": {}},
        )

        request = http.requests[0]
        metadata = _metadata(request)
        assert str(request.url) == "http://worker:9/v1/execute"
        assert request.headers[EXECUTOR_PROTOCOL_HEADER] == EXECUTOR_PROTOCOL_VERSION
        assert metadata["origin"] == "run"
        # Cached transport hashes key off this exact form.
        assert metadata["provenance_hash"] == derive_subkey("x = 1\n", "['x']")
        assert metadata["inputs"][0] == {
            "name": "x",
            "format": "pickle/object",
            "uri": None,
            "byte_size": 5,
            "file": "x.pickle",
        }
        params = metadata["transform"]["params"]
        assert params["source"] == "x = 1\n"
        assert params["timeout_seconds"] == 7.0
        assert params["env"] == {"K": "V"}
        assert params["mounts"][0]["name"] == "m"
        assert params["mutation_defines"] == ["d"]
        assert params["tables"] == {"t": {}}
        assert "language" not in params
        assert "environment" not in params
        assert b'filename="x.pickle"' in request.content
        assert b"value" in request.content
        assert "authorization" not in request.headers

    async def test_a_token_authorizes_it(self, http, features, tmp_path):
        http.answers.append(self._bundle_reply(tmp_path))

        await self._run(tmp_path, worker=_worker(token="secret"))

        assert http.requests[0].headers["Authorization"] == "Bearer secret"

    async def test_an_r_cell_names_its_language_and_lock(self, http, features, tmp_path):
        features.env = {"lock": "renv"}
        http.answers.append(self._bundle_reply(tmp_path))

        await self._run(tmp_path, language="r")

        params = _metadata(http.requests[0])["transform"]["params"]
        assert params["language"] == "r"
        assert params["environment"] == {"lock": "renv"}

    async def test_injected_values_ride_along(self, http, features, tmp_path):
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "mod.json").write_text("module")
        (tmp_path / "out" / "inj.pickle").write_text("injected")
        http.answers.append(self._bundle_reply(tmp_path))
        injected = {"v": {"file": "inj.pickle", "content_type": "pickle/object"}}

        await self._run(
            tmp_path,
            input_specs={
                "mod": {"content_type": "module/cell", "file": "mod.json", "injected": injected}
            },
        )

        request = http.requests[0]
        assert _metadata(request)["inputs"][0]["injected"] == injected
        assert b'filename="inj.pickle"' in request.content

    async def test_the_request_may_outlast_the_cell(self, http, features, tmp_path):
        http.answers.append(self._bundle_reply(tmp_path))

        await self._run(tmp_path)

        assert http.timeouts == [30.0]

    # What comes back

    async def test_a_bundle_is_the_result(self, http, features, tmp_path):
        noted = []
        context, _phases = _context(note_worker_mount_bytes=lambda *args: noted.append(args))
        http.answers.append(self._bundle_reply(tmp_path))

        result, output, method, mounts = await self._run(tmp_path, context=context)

        assert result["stdout"] == "printed"
        assert output == tmp_path / "out" / "_executor_result"
        assert method == "executor"
        assert mounts == {}
        assert noted == [("c", result, [])]

    @pytest.mark.parametrize(
        ("reply", "code", "message"),
        [
            (httpx.Response(408), "TIMEOUT", "c timed out after 7.0"),
            (httpx.Response(500, json={"detail": "boom"}), "EXECUTOR_HTTP_ERROR", "500: boom"),
            (
                httpx.Response(200, headers={EXECUTOR_PROTOCOL_HEADER: "other"}),
                "PROTOCOL_ERROR",
                "unsupported protocol version 'other'",
            ),
            (
                httpx.Response(200, headers={"X-Strata-Notebook-Executor-Protocol": "other"}),
                "PROTOCOL_ERROR",
                "unsupported notebook protocol version 'other'",
            ),
            (httpx.TimeoutException("slow"), "TIMEOUT", "c timed out after 7.0"),
            (httpx.ConnectError("down"), "REQUEST_FAILED", "request failed for worker 'w': down"),
        ],
    )
    async def test_a_bad_answer_is_a_remote_failure(
        self, http, features, tmp_path, reply, code, message
    ):
        http.answers.append(reply)

        with pytest.raises(RemoteExecutionError, match=message) as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == code

    async def test_an_error_is_read_off_the_stream(self, http, features, tmp_path):
        http.answers.append(httpx.Response(500, stream=httpx.ByteStream(b'{"detail": "boom"}')))

        with pytest.raises(RemoteExecutionError, match="500: boom"):
            await self._run(tmp_path)

    async def test_matching_protocols_are_accepted(self, http, features, tmp_path):
        http.answers.append(
            self._bundle_reply(
                tmp_path,
                **{
                    EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION,
                    "X-Strata-Notebook-Executor-Protocol": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
                },
            )
        )

        result, _output, _method, _mounts = await self._run(tmp_path)

        assert result["success"] is True

    async def test_a_cancel_reclaims_the_machine(self, http, features, tmp_path, monkeypatch):
        cancelled = []

        async def cancel(url, build_id, token):
            cancelled.append((url, token))

        monkeypatch.setattr(dispatch, "cancel_remote_execution", cancel)
        http.answers.append(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await self._run(tmp_path, worker=_worker(token="secret"))

        assert cancelled == [("http://worker:9/v1/execute", "secret")]

    @pytest.mark.parametrize("lost", [httpx.TimeoutException("slow"), httpx.ConnectError("down")])
    async def test_a_lost_request_asks_the_worker_to_stop(
        self, http, features, tmp_path, monkeypatch, lost
    ):
        cancelled = []

        async def cancel(url, build_id, token):
            cancelled.append((url, build_id, token))

        monkeypatch.setattr(dispatch, "cancel_remote_execution", cancel)
        http.answers.append(lost)

        with pytest.raises(RemoteExecutionError):
            await self._run(tmp_path, worker=_worker(token="secret"))

        sent = _metadata(http.requests[0])
        assert cancelled == [("http://worker:9/v1/execute", sent["build_id"], "secret")]

    async def test_a_missing_input_closes_the_files_opened_before_it(
        self, http, features, tmp_path, monkeypatch
    ):
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "x.pickle").write_text("value")
        opened = []

        def recording_open(path, mode):
            handle = open(path, mode)
            opened.append(handle)
            return handle

        monkeypatch.setattr(dispatch, "open", recording_open, raising=False)

        with pytest.raises(FileNotFoundError):
            await self._run(
                tmp_path,
                input_specs={
                    "x": {"file": "x.pickle", "injected": {"y": {"file": "gone.pickle"}}},
                },
            )

        assert [handle.closed for handle in opened] == [True]
        assert http.requests == []

    async def test_input_files_are_closed_whatever_happens(self, http, features, tmp_path):
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "x.pickle").write_text("value")
        http.answers.append(httpx.Response(500, json={"detail": "boom"}))
        opened = []
        real_open = open

        def recording_open(*args, **kwargs):
            handle = real_open(*args, **kwargs)
            opened.append(handle)
            return handle

        dispatch.open = recording_open  # type: ignore[attr-defined]
        try:
            with pytest.raises(RemoteExecutionError):
                await self._run(
                    tmp_path,
                    input_specs={"x": {"content_type": "pickle/object", "file": "x.pickle"}},
                )
        finally:
            del dispatch.open  # type: ignore[attr-defined]

        assert opened
        assert all(handle.closed for handle in opened)

    # A worker that accepts first

    async def test_an_accepted_job_is_followed_and_its_bundle_received(
        self, http, features, tmp_path
    ):
        context, phases = _context()
        http.answers.extend(
            [
                httpx.Response(202, json={"job_url": "/jobs/1"}),
                httpx.Response(
                    200,
                    content=_bundle(tmp_path, stdout="printed"),
                    headers={"content-type": "application/x-tar"},
                ),
            ]
        )

        result, _output, _method, _mounts = await self._run(tmp_path, context=context)

        assert result["stdout"] == "printed"
        assert str(http.requests[1].url) == "http://worker:9/jobs/1"
        assert phases[0] == ("c", "starting", (None, None))

    async def test_a_finished_job_with_no_bundle_is_a_protocol_error(
        self, http, features, tmp_path
    ):
        http.answers.extend(
            [
                httpx.Response(202, json={"job_url": "/jobs/1"}),
                httpx.Response(200, json={"state": "finished"}),
            ]
        )

        with pytest.raises(RemoteExecutionError, match="without answering") as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "PROTOCOL_ERROR"

    async def test_a_failed_job_reports_its_error(self, http, features, tmp_path):
        http.answers.extend(
            [
                httpx.Response(202, json={"job_url": "/jobs/1"}),
                httpx.Response(200, json={"state": "failed", "error": "ran out of memory"}),
            ]
        )

        with pytest.raises(RemoteExecutionError, match="502: ran out of memory"):
            await self._run(tmp_path)

    async def test_a_signed_worker_takes_the_other_road(
        self, http, features, tmp_path, monkeypatch
    ):
        noted = []
        context, _phases = _context(note_worker_mount_bytes=lambda *args: noted.append(args))
        calls = []

        async def signed(host, ctx, worker, *args, **kwargs):
            calls.append(kwargs)
            return {"stdout": "signed"}, Path("x"), "executor", {}

        monkeypatch.setattr(dispatch, "dispatch_signed", signed)

        result, _output, _method, _mounts = await self._run(
            tmp_path,
            context=context,
            worker=_worker(transport="signed"),
            remote_build_id="b1",
            mutation_defines=["d"],
            tables={"t": {}},
            cell_provenance_hash="key",
            language="r",
        )

        assert calls == [
            {
                "build_id": "b1",
                "mutation_defines": ["d"],
                "tables": {"t": {}},
                "cell_id": "c",
                "cell_provenance_hash": "key",
                "language": "r",
            }
        ]
        assert result["stdout"] == "signed"
        assert noted == [("c", result, [])]
        assert http.requests == []


class TestAwaitAcceptedJob:
    """Following a job a worker took with 202."""

    SUBMIT = "http://worker:9/v1/execute"

    @pytest.fixture
    def clock(self, monkeypatch):
        now = SimpleNamespace(value=0.0, step=0.0)

        def tick():
            now.value += now.step
            return now.value

        monkeypatch.setattr(dispatch, "_monotonic", tick)
        monkeypatch.setattr(dispatch, "_JOB_POLL_SECONDS", 0)
        return now

    @staticmethod
    async def _await(accepted=None, *, context=None, receive_bundle=None):
        cancelled = []

        async def cancel():
            cancelled.append(True)

        context = context or _context()[0]
        reply = accepted or httpx.Response(202, json={"job_url": "/jobs/1"})
        try:
            return (
                await await_accepted_job(
                    context,
                    reply,
                    submit_url=TestAwaitAcceptedJob.SUBMIT,
                    headers={"Authorization": "Bearer t"},
                    timeout_seconds=7.0,
                    cancel=cancel,
                    worker_spec=_worker(),
                    cell_id="c",
                    receive_bundle=receive_bundle,
                ),
                cancelled,
            )
        except RemoteExecutionError as exc:
            exc.cancelled = cancelled  # ty: ignore[unresolved-attribute]
            raise

    @pytest.mark.parametrize(
        "accepted",
        [
            httpx.Response(202, text="not json"),
            httpx.Response(202, json=["a list"]),
            httpx.Response(202, json={"job_url": ""}),
            httpx.Response(202, json={"job_url": 3}),
        ],
    )
    async def test_an_acceptance_with_no_job_url_stops_the_job(self, http, clock, accepted):
        with pytest.raises(RemoteExecutionError, match="without a job_url") as raised:
            await self._await(accepted)

        assert raised.value.remote_error_code == "PROTOCOL_ERROR"
        assert raised.value.remote_build_state == "failed"
        assert raised.value.cancelled == [True]
        assert http.requests == []

    @pytest.mark.parametrize(
        "reply",
        [
            httpx.Response(200, text="<html>gateway</html>"),
            httpx.Response(200, json=["a list"]),
            httpx.Response(200, json={"state": "finished", "status_code": "ok"}),
            httpx.Response(200, json={"state": "failed", "status_code": ["502"]}),
        ],
    )
    async def test_a_reply_that_is_not_a_job_stops_the_job(self, http, clock, reply):
        http.answers.append(reply)

        with pytest.raises(RemoteExecutionError, match="cannot read") as raised:
            await self._await()

        assert raised.value.remote_error_code == "JOB_STATUS_FAILED"
        assert raised.value.remote_build_state == "failed"
        assert raised.value.cancelled == [True]

    async def test_a_job_on_another_host_is_never_polled(self, http, clock):
        accepted = httpx.Response(202, json={"job_url": "http://evil:9/jobs/1"})

        with pytest.raises(RemoteExecutionError, match="on another") as raised:
            await self._await(accepted)

        assert raised.value.cancelled == [True]
        assert http.requests == []

    async def test_polling_keeps_the_workers_headers(self, http, clock):
        http.answers.append(httpx.Response(200, json={"state": "finished", "result": {"ok": 1}}))

        reply, _cancelled = await self._await()

        assert http.requests[0].headers["Authorization"] == "Bearer t"
        assert str(http.requests[0].url) == "http://worker:9/jobs/1"
        assert reply.status_code == 200
        assert reply.json() == {"ok": 1}

    async def test_a_finished_jobs_status_code_is_kept(self, http, clock):
        http.answers.append(
            httpx.Response(200, json={"state": "finished", "status_code": 500, "error": "boom"})
        )

        reply, _cancelled = await self._await()

        assert reply.status_code == 500
        assert reply.json() == {"detail": "boom"}

    async def test_a_failed_job_with_nothing_said_is_a_bad_gateway(self, http, clock):
        http.answers.append(httpx.Response(200, json={"state": "failed"}))

        reply, _cancelled = await self._await()

        assert reply.status_code == 502
        assert reply.json() == {"detail": "job failed"}

    async def test_a_failed_poll_stops_the_job(self, http, clock):
        http.answers.append(httpx.ConnectError("down"))

        with pytest.raises(RemoteExecutionError, match="status request failed: down") as raised:
            await self._await()

        assert raised.value.remote_error_code == "JOB_STATUS_FAILED"
        assert raised.value.remote_build_state == "failed"
        assert raised.value.cancelled == [True]

    async def test_a_refused_poll_stops_the_job(self, http, clock):
        http.answers.append(httpx.Response(404, json={"detail": "gone"}))

        with pytest.raises(RemoteExecutionError, match="returned 404: gone") as raised:
            await self._await()

        assert raised.value.remote_error_code == "JOB_STATUS_FAILED"
        assert raised.value.remote_build_state == "failed"
        assert raised.value.cancelled == [True]

    async def test_phases_follow_the_job(self, http, clock):
        context, phases = _context()
        http.answers.extend(
            [
                httpx.Response(200, json={"state": "queued", "position": 2, "detail": "in line"}),
                httpx.Response(200, json={"state": "queued", "position": 2, "detail": "in line"}),
                httpx.Response(200, json={"state": "running"}),
                httpx.Response(200, json={"state": "running"}),
                httpx.Response(200, json={"state": "running", "detail": "machine 3"}),
                httpx.Response(200, json={"state": "finished"}),
            ]
        )

        await self._await(context=context)

        assert phases == [
            ("c", "starting", (None, None)),
            ("c", "starting", (2, "in line")),
            ("c", "running", (None, None)),
            ("c", "running", (None, "machine 3")),
        ]

    async def test_the_cells_timeout_starts_when_it_runs(self, http, clock):
        # Two seconds a call: 100 s of provisioning at a 50 s limit would have failed already
        # if it counted, and 7 s of running does.
        clock.step = 2.0
        http.answers.extend([httpx.Response(200, json={"state": "running"}) for _ in range(6)])

        with pytest.raises(RemoteExecutionError, match="c timed out after 7.0") as raised:
            await self._await()

        assert raised.value.remote_error_code == "TIMEOUT"
        assert raised.value.remote_build_state == "failed"
        assert raised.value.cancelled == [True]

    async def test_provisioning_does_not_count_against_the_cell(self, http, clock):
        clock.step = 2.0
        http.answers.extend(
            [httpx.Response(200, json={"state": "starting"}) for _ in range(6)]
            + [
                httpx.Response(200, json={"state": "running"}),
                httpx.Response(200, json={"state": "finished"}),
            ]
        )

        reply, cancelled = await self._await()

        assert reply.status_code == 200
        assert cancelled == []

    async def test_news_from_a_running_job_does_not_restart_its_timeout(self, http, clock):
        clock.step = 2.0
        http.answers.extend(
            httpx.Response(200, json={"state": "running", "detail": f"step {i}"}) for i in range(9)
        )

        with pytest.raises(RemoteExecutionError) as raised:
            await self._await()

        assert raised.value.remote_error_code == "TIMEOUT"

    async def test_it_waits_between_polls(self, http, clock, monkeypatch):
        waits = []
        real_sleep = asyncio.sleep

        async def sleep(seconds):
            waits.append(seconds)
            await real_sleep(0)

        monkeypatch.setattr(dispatch, "_JOB_POLL_SECONDS", 0.25)
        monkeypatch.setattr(dispatch.asyncio, "sleep", sleep)
        http.answers.extend(
            [
                httpx.Response(200, json={"state": "queued"}),
                httpx.Response(200, json={"state": "finished"}),
            ]
        )

        await self._await()

        assert waits.count(0.25) == 1

    async def test_provisioning_has_its_own_limit(self, http, clock):
        clock.step = 30.0
        http.answers.extend([httpx.Response(200, json={"state": "starting"}) for _ in range(3)])

        with pytest.raises(
            RemoteExecutionError, match="within 50s .last state: starting"
        ) as raised:
            await self._await()

        assert raised.value.remote_error_code == "PROVISIONING_TIMEOUT"
        assert raised.value.remote_build_state == "failed"
        assert raised.value.cancelled == [True]

    async def test_a_tar_answer_goes_to_the_receiver(self, http, clock):
        received = []

        async def receive(reply):
            received.append(await reply.aread())

        http.answers.append(
            httpx.Response(200, content=b"tar", headers={"content-type": "application/x-tar"})
        )

        reply, _cancelled = await self._await(receive_bundle=receive)

        assert received == [b"tar"]
        assert reply.status_code == 200

    async def test_a_refused_tar_is_not_a_bundle(self, http, clock):
        received = []

        async def receive(reply):
            received.append(reply)

        http.answers.append(
            httpx.Response(500, content=b"tar", headers={"content-type": "application/x-tar"})
        )

        with pytest.raises(RemoteExecutionError) as raised:
            await self._await(receive_bundle=receive)

        assert raised.value.remote_error_code == "JOB_STATUS_FAILED"
        assert received == []

    async def test_a_tar_answer_is_not_a_status_without_a_receiver(self, http, clock):
        http.answers.append(
            httpx.Response(
                200,
                json={"state": "finished", "result": {"ok": 1}},
                headers={"content-type": "application/x-tar"},
            )
        )

        reply, _cancelled = await self._await()

        assert reply.json() == {"ok": 1}


class TestBackgroundCalls:
    """The small calls around a run, whose failures must not end it."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("http://w:9", "http://w:9/v1/execute-manifest"),
            ("http://w:9/", "http://w:9/v1/execute-manifest"),
            ("http://w:9/v1/execute", "http://w:9/v1/execute-manifest"),
            ("http://w:9/v1/notebook-execute", "http://w:9/v1/execute-manifest"),
            ("http://w:9/pool/m1/?x=1#f", "http://w:9/pool/m1/v1/execute-manifest"),
        ],
    )
    def test_the_manifest_endpoint(self, url, expected):
        assert execute_manifest_url(url) == expected

    async def test_a_cancel_carries_the_workers_token(self, http):
        http.answers.append(httpx.Response(200))

        await cancel_remote_execution("http://w:9/v1/execute", "b1", "secret")

        request = http.requests[0]
        assert request.method == "POST"
        assert str(request.url) == "http://w:9/v1/executions/b1/cancel"
        assert request.headers["Authorization"] == "Bearer secret"

    async def test_a_cancel_without_a_token_sends_none(self, http):
        http.answers.append(httpx.Response(200))

        await cancel_remote_execution("http://w:9/v1/execute", "b1", None)

        assert "authorization" not in http.requests[0].headers

    @pytest.mark.parametrize("answer", [httpx.Response(500), httpx.ConnectError("down")])
    async def test_a_failed_cancel_is_swallowed(self, http, answer):
        http.answers.append(answer)

        await cancel_remote_execution("http://w:9/v1/execute", "b1", None)

    @staticmethod
    def _ws(monkeypatch, *, fails=False):
        sent = []

        async def broadcast(session_id, message):
            if fails:
                raise RuntimeError("no sockets")
            sent.append((session_id, message))

        monkeypatch.setattr("strata.notebook.ws._broadcast_message", broadcast)
        return sent

    async def test_a_phase_goes_to_the_sessions_sockets(self, monkeypatch):
        sent = self._ws(monkeypatch)

        await broadcast_remote_phase(_host(), "c", _worker(), "running", queue=(2, "in line"))

        session_id, message = sent[0]
        payload = message["payload"]
        assert session_id == "session"
        assert payload["cell_id"] == "c"
        assert payload["remote_worker"] == "w"
        assert payload["remote_build_state"] == "running"
        assert payload["remote_queue_position"] == 2
        assert payload["remote_queue_detail"] == "in line"

    async def test_a_phase_with_no_cell_says_nothing(self, monkeypatch):
        built = []
        monkeypatch.setattr(
            "strata.notebook.ws_payloads.cell_status_payload",
            lambda *args, **kwargs: built.append(args) or {},
        )
        sent = self._ws(monkeypatch)

        await broadcast_remote_phase(_host(), None, _worker(), "running")

        assert built == []
        assert sent == []

    async def test_a_failed_phase_does_not_fail_the_run(self, monkeypatch):
        self._ws(monkeypatch, fails=True)

        await broadcast_remote_phase(_host(), "c", _worker(), "running")

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            (httpx.Response(500, json={"detail": "said"}), "said"),
            (httpx.Response(500, json={"error": "said"}), "said"),
            (httpx.Response(500, json={"detail": "", "error": "said"}), "said"),
            (httpx.Response(500, json={"detail": 3}), "Unknown remote executor error"),
            (httpx.Response(500, json=["a list"]), "Unknown remote executor error"),
            (httpx.Response(500, text=" plain "), "plain"),
            (httpx.Response(500, text=""), "Unknown remote executor error"),
        ],
    )
    def test_what_a_failure_said(self, reply, expected):
        assert extract_remote_error(reply) == expected

    def test_a_token_comes_from_the_env_first(self, monkeypatch):
        monkeypatch.setenv("W_TOKEN", " from-env ")

        worker = _worker(token_env="W_TOKEN", token="literal")

        assert dispatch._resolve_worker_token(worker) == "from-env"

    def test_then_the_literal(self, monkeypatch):
        monkeypatch.delenv("W_TOKEN", raising=False)

        worker = _worker(token_env="W_TOKEN", token=" literal ")

        assert dispatch._resolve_worker_token(worker) == "literal"

    def test_then_the_runtime_registry(self, monkeypatch):
        monkeypatch.setattr(
            "strata.notebook.worker_secrets.get_runtime_worker_token",
            lambda name: f"registered for {name}",
        )

        assert dispatch._resolve_worker_token(_worker()) == "registered for w"

    def test_nothing_without_a_config(self):
        assert dispatch._resolve_worker_token(SimpleNamespace()) is None

    def test_harness_params_are_added_only_when_present(self):
        params = {}

        dispatch._add_harness_params(params, [], {})
        assert params == {}

        dispatch._add_harness_params(params, ["d"], {"t": {}})
        assert params == {"mutation_defines": ["d"], "tables": {"t": {}}}

    def test_a_folder_input_is_packed_and_the_others_are_left(self, tmp_path):
        (tmp_path / "f.dir").mkdir()
        (tmp_path / "f.dir" / "a.txt").write_text("a")
        stem = "f"
        specs = {
            "f": {"content_type": output_dirs.CONTENT_TYPE, "file": stem + output_dirs.EXTENSION},
            "x": {"content_type": "pickle/object", "file": "x.pickle"},
        }

        packed = dispatch._pack_folder_inputs(specs, tmp_path)

        assert packed["f"]["content_type"] == output_dirs.TAR_CONTENT_TYPE
        assert (tmp_path / packed["f"]["file"]).exists()
        assert packed["x"] == specs["x"]
        assert specs["f"]["content_type"] == output_dirs.CONTENT_TYPE


class TestStaging:
    """Upstream blobs as artifacts a worker can fetch by signed URL."""

    @staticmethod
    def _store():
        seen = SimpleNamespace(created=[], published=[], finalized=[])

        def create_artifact(**kwargs):
            seen.created.append(kwargs)
            return 3

        def finalize_artifact(artifact_id, version, **kwargs):
            seen.finalized.append((artifact_id, kwargs))
            return SimpleNamespace(id=artifact_id, version=version)

        store = SimpleNamespace(
            create_artifact=create_artifact,
            publish_blob_from_path=lambda *args: seen.published.append(args),
            finalize_artifact=finalize_artifact,
        )
        return store, seen

    @staticmethod
    def _stage(store, tmp_path, specs, tenant="t1"):
        return stage_signed_transport_inputs(
            _host(),
            artifact_store=store,
            build_id="b1",
            input_specs=specs,
            output_dir=tmp_path,
            tenant_id=tenant,
            principal_id="p1",
        )

    def test_an_input_becomes_a_finalized_artifact(self, tmp_path):
        (tmp_path / "x.pickle").write_bytes(b"value")
        store, seen = self._store()

        specs, artifacts = self._stage(
            store,
            tmp_path,
            {"x": {"file": "x.pickle", "content_type": "json/object", "uri": "strata://up"}},
        )

        assert specs["x"]["content_type"] == "json/object"
        assert specs["x"]["uri"].endswith("_x@v=3")
        assert "file" not in specs["x"]
        created = seen.created[0]
        assert created["tenant"] == "t1"
        assert created["principal"] == "p1"
        assert created["input_versions"] == {"strata://up": "strata://up"}
        assert created["transform_spec"].inputs == ["strata://up"]
        assert created["transform_spec"].params["build_id"] == "b1"
        assert seen.published[0][2] == tmp_path / "x.pickle"
        finalized = seen.finalized[0][1]
        assert finalized["byte_size"] == 5
        assert finalized["content_sha256"] == hashlib.sha256(b"value").hexdigest()
        assert artifacts == [(created["artifact_id"], 3)]

    def test_two_tenants_never_share_an_id(self, tmp_path):
        (tmp_path / "x.pickle").write_bytes(b"value")
        store, seen = self._store()
        spec = {"x": {"file": "x.pickle", "uri": "strata://up"}}

        self._stage(store, tmp_path, spec, tenant="t1")
        self._stage(store, tmp_path, spec, tenant="t2")

        first, second = (created["artifact_id"] for created in seen.created)
        assert first != second

    def test_the_same_bytes_from_two_places_have_two_keys(self, tmp_path):
        (tmp_path / "x.pickle").write_bytes(b"value")
        store, seen = self._store()

        self._stage(store, tmp_path, {"x": {"file": "x.pickle", "uri": "strata://a"}})
        self._stage(store, tmp_path, {"x": {"file": "x.pickle", "uri": "strata://b"}})

        first, second = (created["provenance_hash"] for created in seen.created)
        assert first != second

    def test_a_key_follows_the_bytes(self, tmp_path):
        store, seen = self._store()
        path = tmp_path / "x.pickle"

        path.write_bytes(b"one")
        self._stage(store, tmp_path, {"x": {"file": "x.pickle", "uri": "strata://a"}})
        path.write_bytes(b"two")
        self._stage(store, tmp_path, {"x": {"file": "x.pickle", "uri": "strata://a"}})

        first, second = (created["provenance_hash"] for created in seen.created)
        assert first != second

    def test_a_fetched_file_keeps_its_name(self, tmp_path):
        (tmp_path / "__fetch_0_data.csv").write_bytes(b"bytes")
        store, _seen = self._store()

        specs, _artifacts = self._stage(
            store,
            tmp_path,
            {"f": {"file": "__fetch_0_data.csv", "content_type": "file/path"}},
        )

        assert specs["f"]["file"] == "__fetch_0_data.csv"

    def test_injected_values_are_staged_too(self, tmp_path):
        (tmp_path / "mod.json").write_bytes(b"module")
        (tmp_path / "inj.pickle").write_bytes(b"injected")
        store, _seen = self._store()

        specs, artifacts = self._stage(
            store,
            tmp_path,
            {
                "mod": {
                    "file": "mod.json",
                    "content_type": "module/cell",
                    "injected": {"v": {"file": "inj.pickle", "content_type": "json/object"}},
                }
            },
        )

        injected = specs["mod"]["injected"]["v"]
        assert injected["content_type"] == "json/object"
        assert "_mod__inj__v@v=3" in injected["uri"]
        assert len(artifacts) == 2

    def test_an_input_with_no_file_is_refused(self, tmp_path):
        store, _seen = self._store()

        with pytest.raises(RuntimeError, match="missing a local input file for x"):
            self._stage(store, tmp_path, {"x": {"file": " "}})

    def test_a_missing_file_is_refused(self, tmp_path):
        store, _seen = self._store()

        with pytest.raises(RuntimeError, match="could not find input file 'gone'"):
            self._stage(store, tmp_path, {"x": {"file": "gone"}})

    def test_a_failed_finalize_is_refused(self, tmp_path):
        (tmp_path / "x.pickle").write_bytes(b"value")
        store, _seen = self._store()
        store.finalize_artifact = lambda *_args, **_kwargs: None

        with pytest.raises(RuntimeError, match="Failed to finalize"):
            self._stage(store, tmp_path, {"x": {"file": "x.pickle"}})


class TestSigned:
    """The build plus signed-URL road: stores and signer faked, real bundle bytes."""

    @pytest.fixture
    def signed(self, monkeypatch, tmp_path):
        seen = SimpleNamespace(
            failed=[],
            failed_artifacts=[],
            created=[],
            builds=[],
            started=[],
            manifest=None,
            relayed=[],
            receiving=[],
            receivers=[],
            config=SimpleNamespace(
                server_transforms_enabled=False,
                writes_enabled=True,
                artifact_dir=tmp_path / "store",
                server_url="http://server",
                artifact_presigned_urls=False,
                max_transform_output_bytes=99,
                signed_url_expiry_seconds=1.0,
                worker_provisioning_timeout_seconds=100.0,
                node_advertised_url=None,
            ),
            build=SimpleNamespace(
                state="ready",
                artifact_id="art",
                version=1,
                error_code=None,
                error_message=None,
            ),
            blob=_bundle(tmp_path, stdout="printed"),
            principal=SimpleNamespace(id="p1", tenant="t1"),
            received={"out": {"a.txt": {}}},
        )

        def generate_build_manifest(**kwargs):
            seen.manifest = kwargs
            return SimpleNamespace(to_dict=lambda: {"signed": True})

        state = SimpleNamespace(
            config=seen.config,
            url_signer=SimpleNamespace(generate_build_manifest=generate_build_manifest),
        )
        monkeypatch.setattr("strata.server.get_state", lambda: state)
        monkeypatch.setattr("strata.auth.principal.get_principal", lambda: seen.principal)

        def create_artifact(**kwargs):
            seen.created.append(kwargs)
            return 4

        def fail_artifact(artifact_id, version, tenant=None):
            seen.failed_artifacts.append((artifact_id, version, tenant))

        def open_blob_reader(_id, _version):
            if seen.blob is None:
                return None
            return io.BytesIO(seen.blob)

        seen.artifact_store = SimpleNamespace(
            dialect="sqlite",
            blob_store="blobs",
            get_artifact="get artifact",
            create_artifact=create_artifact,
            fail_artifact=fail_artifact,
            open_blob_reader=open_blob_reader,
        )

        def fail_build(build_id, message, code):
            seen.failed.append((message, code))

        seen.build_store = SimpleNamespace(
            fail_build=fail_build,
            create_build=lambda **kwargs: seen.builds.append(kwargs),
            start_build=seen.started.append,
            get_build=lambda _id: seen.build,
        )
        monkeypatch.setattr(dispatch, "get_artifact_store", lambda _dir: seen.artifact_store)
        monkeypatch.setattr(dispatch, "get_build_store", lambda *_a, **_k: seen.build_store)
        monkeypatch.setattr(dispatch, "current_trace_context", lambda: {"traceparent": "tp"})
        monkeypatch.setattr(
            dispatch,
            "stage_signed_transport_inputs",
            lambda host, **kwargs: (
                {"x": {"uri": "strata://staged@v=1", "content_type": "json/object"}},
                [("staged", 1)],
            ),
        )

        @asynccontextmanager
        async def relaying(build_id, session_id, cell_id, shared_store=None):
            seen.relayed.append((build_id, session_id, cell_id, shared_store))
            yield

        @contextmanager
        def receiving(build_id, receiver):
            seen.receiving.append((build_id, receiver))
            yield receiver

        def receiver(artifact_mgr, cell_id, upload_dir, folders):
            seen.receivers.append((artifact_mgr, cell_id, upload_dir, folders))
            return SimpleNamespace(received=lambda: seen.received)

        monkeypatch.setattr(dispatch.console_relay, "relaying", relaying)
        monkeypatch.setattr(dispatch.output_file_relay, "receiving", receiving)
        monkeypatch.setattr(dispatch.output_file_relay, "Receiver", receiver)
        return seen

    @staticmethod
    async def _run(tmp_path, *, context=None, worker=None, **kwargs):
        context = context or _context()[0]
        (tmp_path / "out").mkdir(exist_ok=True)
        return await dispatch_signed(
            _host(tmp_path),
            context,
            worker or _worker(transport="signed"),
            kwargs.pop("source", "x = 1\n"),
            {"x": {"file": "x.pickle"}},
            [],
            tmp_path / "out",
            {"SECRET": "value"},
            7.0,
            **{"build_id": "b1", "cell_id": "c", **kwargs},
        )

    # Refusals

    async def test_it_needs_writes_or_transforms(self, signed, tmp_path):
        signed.config.writes_enabled = False

        with pytest.raises(RuntimeError, match="requires personal-mode writes"):
            await self._run(tmp_path)

    async def test_transforms_are_enough(self, signed, http, tmp_path):
        signed.config.writes_enabled = False
        signed.config.server_transforms_enabled = True
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path)

        assert http.requests

    async def test_it_needs_an_artifact_directory(self, signed, tmp_path):
        signed.config.artifact_dir = None

        with pytest.raises(RuntimeError, match="Artifact store is not configured"):
            await self._run(tmp_path)

    async def test_it_needs_a_build_store(self, signed, tmp_path, monkeypatch):
        monkeypatch.setattr(dispatch, "get_build_store", lambda *_a, **_k: None)

        with pytest.raises(RuntimeError, match="Build store is not initialized"):
            await self._run(tmp_path)

    async def test_it_needs_a_url(self, signed, tmp_path):
        with pytest.raises(RuntimeError, match="is missing config.url"):
            await self._run(tmp_path, worker=_worker(url="", transport="signed"))

    @pytest.fixture
    def stopped(self, monkeypatch):
        calls = []

        async def cancel(url, build_id, token):
            calls.append((url, build_id, token))

        monkeypatch.setattr(dispatch, "cancel_remote_execution", cancel)
        return calls

    async def test_a_cancel_before_the_request_tells_the_worker_nothing(
        self, signed, http, stopped, tmp_path
    ):
        loop = asyncio.get_running_loop()
        real = signed.build_store.create_build

        def create_build(**kwargs):
            real(**kwargs)
            loop.call_soon_threadsafe(running.cancel)

        signed.build_store.create_build = create_build
        running = asyncio.create_task(self._run(tmp_path, worker=_worker(transport="signed")))

        with pytest.raises(asyncio.CancelledError):
            await running

        assert signed.failed == [("Notebook manifest execution cancelled", "CANCELLED")]
        assert stopped == []
        assert http.requests == []

    async def test_a_setup_failure_before_the_request_tells_the_worker_nothing(
        self, signed, http, stopped, tmp_path
    ):
        def create_build(**kwargs):
            raise OSError("store down")

        signed.build_store.create_build = create_build

        with pytest.raises(RemoteExecutionError, match="setup failed") as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "SETUP_FAILED"
        assert stopped == []

    @pytest.mark.parametrize("lost", [httpx.TimeoutException("slow"), httpx.ConnectError("down")])
    async def test_a_lost_request_asks_the_worker_to_stop(
        self, signed, http, stopped, tmp_path, lost
    ):
        http.answers.append(lost)

        with pytest.raises(RemoteExecutionError):
            await self._run(tmp_path, worker=_worker(transport="signed", token="secret"))

        assert stopped == [("http://worker:9/v1/execute", "b1", "secret")]

    async def test_an_error_after_the_request_asks_the_worker_to_stop(
        self, signed, http, stopped, tmp_path, monkeypatch
    ):
        async def broken(*_args, **_kwargs):
            raise ValueError("broken")

        monkeypatch.setattr(dispatch, "await_accepted_job", broken)
        http.answers.append(httpx.Response(202, json={"job_url": "/jobs/1"}))

        with pytest.raises(RemoteExecutionError, match="setup failed"):
            await self._run(tmp_path, worker=_worker(transport="signed", token="secret"))

        assert stopped == [("http://worker:9/v1/execute", "b1", "secret")]

    async def test_a_cancel_while_recording_a_failure_keeps_that_failure(
        self, signed, http, tmp_path
    ):
        loop = asyncio.get_running_loop()
        http.answers.append(httpx.Response(500, json={"detail": "boom"}))

        def fail_build(build_id, message, code):
            signed.failed.append((message, code))
            loop.call_soon_threadsafe(running.cancel)

        signed.build_store.fail_build = fail_build
        running = asyncio.create_task(self._run(tmp_path))

        with pytest.raises(asyncio.CancelledError):
            await running

        assert [code for _message, code in signed.failed] == ["EXECUTOR_HTTP_ERROR"]

    # What is recorded and what is sent

    async def test_a_build_is_recorded_without_the_envs_values(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path, language="r", mutation_defines=["d"], tables={"t": {}})

        recorded = signed.created[0]["transform_spec"].params
        assert recorded["env"]["names"] == ["SECRET"]
        assert "value" not in json.dumps(recorded["env"])
        assert recorded["language"] == "r"
        assert recorded["mutation_defines"] == ["d"]
        assert recorded["tables"] == {"t": {}}
        assert signed.builds[0]["params"]["env"] == recorded["env"]

        sent = signed.manifest["metadata"]["params"]
        assert sent["env"] == {"SECRET": "value"}
        assert sent["language"] == "r"
        assert sent["input_specs"]["x"]["uri"] == "strata://staged@v=1"

    async def test_a_python_cell_names_no_language(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path)

        assert "language" not in signed.manifest["metadata"]["params"]

    async def test_the_lock_goes_to_the_worker(self, signed, http, tmp_path, monkeypatch):
        async def locked(_host, _worker, language):
            return {"lock": language}

        monkeypatch.setattr(dispatch, "locked_environment", locked)
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path, language="r")

        assert signed.manifest["metadata"]["params"]["environment"] == {"lock": "r"}

    async def test_the_artifact_and_build_belong_to_the_run(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path)

        created = signed.created[0]
        assert created["artifact_id"] == "nb_remote_nb_b1"
        assert created["tenant"] == "t1"
        assert created["principal"] == "p1"
        assert created["input_versions"] == {"strata://staged@v=1": "strata://staged@v=1"}
        assert created["transform_spec"].inputs == ["strata://staged@v=1"]
        build = signed.builds[0]
        assert build["build_id"] == "b1"
        assert build["version"] == 4
        assert build["tenant_id"] == "t1"
        assert build["principal_id"] == "p1"
        assert build["input_uris"] == ["strata://staged@v=1"]
        assert build["dispatch"] == "external"
        assert signed.started == ["b1"]

    async def test_a_build_id_is_made_when_none_is_given(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path, build_id=None)

        assert signed.builds[0]["build_id"].startswith("nbbuild-")

    async def test_a_key_changes_with_what_runs(self, signed, http, tmp_path):
        http.answers.extend([httpx.Response(200), httpx.Response(200)])

        await self._run(tmp_path)
        await self._run(tmp_path, source="x = 2\n")

        first, second = (created["provenance_hash"] for created in signed.created)
        assert first != second

    async def test_the_manifest_says_who_and_what_it_is_for(self, signed, http, tmp_path):
        context, _phases = _context(origin="run")
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path, context=context, cell_provenance_hash="key")

        manifest = signed.manifest
        metadata = manifest["metadata"]
        assert manifest["base_url"] == "http://server"
        assert manifest["build_id"] == "b1"
        assert manifest["input_artifacts"] == [("staged", 1)]
        assert manifest["max_output_bytes"] == 99
        assert manifest["blob_store"] is None
        assert manifest["version_record"] == "get artifact"
        assert manifest["url_expiry_seconds"] == 407.0
        assert metadata["principal"] == "p1"
        assert metadata["tenant"] == "t1"
        assert metadata["notebook_id"] == "nb"
        assert metadata["cell_id"] == "c"
        assert metadata["cell_provenance_hash"] == "key"
        assert metadata["origin"] == "run"
        assert metadata["version"] == 4
        assert metadata["traceparent"] == "tp"

    async def test_a_worker_may_name_its_own_server_url(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path, worker=_worker(transport="signed", strata_url="http://public"))

        assert signed.manifest["base_url"] == "http://public"

    async def test_presigned_urls_are_offered_when_set(self, signed, http, tmp_path):
        signed.config.artifact_presigned_urls = True
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path)

        assert signed.manifest["blob_store"] == "blobs"

    async def test_an_expiry_setting_is_a_floor(self, signed, http, tmp_path):
        signed.config.signed_url_expiry_seconds = 10000.0
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path)

        assert signed.manifest["url_expiry_seconds"] == 10000.0

    async def test_the_manifest_is_posted_with_the_token(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path, worker=_worker(transport="signed", token="secret"))

        request = http.requests[0]
        assert str(request.url) == "http://worker:9/v1/execute-manifest"
        assert json.loads(request.content) == {"signed": True}
        assert request.headers["Authorization"] == "Bearer secret"
        assert request.headers["traceparent"] == "tp"
        assert http.timeouts == [30.0]

    # Relays

    async def test_the_console_is_relayed_for_a_cell(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path)

        assert signed.relayed == [("b1", "session", "c", None)]

    async def test_a_second_node_shares_the_console_through_the_store(self, signed, http, tmp_path):
        signed.config.node_advertised_url = "http://node"
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path)

        assert signed.relayed[0][3] is signed.build_store

    async def test_no_cell_relays_nothing(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path, cell_id=None)

        assert signed.relayed == []
        assert signed.receiving == []

    async def test_uploads_are_received_only_for_declared_folders(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path)

        assert signed.receiving == []

    async def test_uploads_are_kept_whatever_the_worker_answers(self, signed, http, tmp_path):
        context, _phases = _context(received_output_files={"c": "from an earlier run"})
        http.answers.append(httpx.ConnectError("down"))

        with pytest.raises(RemoteExecutionError):
            await self._run(tmp_path, context=context, source="# @output_dir out\nx = 1\n")

        assert signed.receiving[0][0] == "b1"
        assert signed.receivers[0] == (
            "artifact manager",
            "c",
            tmp_path / ".strata" / "uploads",
            frozenset({"out"}),
        )
        assert context.received_output_files["c"] == signed.received

    async def test_the_last_runs_uploads_do_not_survive_a_run_that_never_sends(
        self, signed, http, tmp_path
    ):
        context, _phases = _context(received_output_files={"c": "from an earlier run"})
        http.answers.append(httpx.Response(200))

        await self._run(tmp_path, context=context, cell_id=None)

        assert context.received_output_files == {"c": "from an earlier run"}

    async def test_a_run_that_fails_before_it_sends_keeps_no_old_uploads(
        self, signed, tmp_path, monkeypatch
    ):
        @asynccontextmanager
        async def relaying(*_args, **_kwargs):
            raise OSError("no relay")
            yield

        monkeypatch.setattr(dispatch.console_relay, "relaying", relaying)
        context, _phases = _context(received_output_files={"c": "from an earlier run"})

        with pytest.raises(RemoteExecutionError, match="no relay"):
            await self._run(tmp_path, context=context)

        assert context.received_output_files == {}

    # An accepted job

    async def test_an_accepted_job_is_followed(self, signed, http, tmp_path):
        context, phases = _context()
        http.answers.extend(
            [
                httpx.Response(202, json={"job_url": "/jobs/1"}),
                httpx.Response(200, json={"state": "finished", "result": {}}),
            ]
        )

        result, _output, _method, _mounts = await self._run(tmp_path, context=context)

        assert result["stdout"] == "printed"
        assert str(http.requests[1].url) == "http://worker:9/jobs/1"
        assert phases[0] == ("c", "starting", (None, None))

    # Failures while sending

    @pytest.mark.parametrize(
        ("answer", "code", "recorded", "message"),
        [
            (
                httpx.TimeoutException("slow"),
                "TIMEOUT",
                "Notebook manifest execution timed out",
                "c timed out after 7.0",
            ),
            (
                httpx.ConnectError("down"),
                "REQUEST_FAILED",
                "Remote executor request failed for worker 'w': down",
                "Remote executor request failed for worker 'w': down",
            ),
            (
                ValueError("odd"),
                "SETUP_FAILED",
                "odd",
                "Remote executor setup failed for worker 'w': odd",
            ),
        ],
    )
    async def test_a_failed_send_fails_the_build(
        self, signed, http, tmp_path, answer, code, recorded, message
    ):
        http.answers.append(answer)

        with pytest.raises(RemoteExecutionError) as raised:
            await self._run(tmp_path)

        assert str(raised.value) == message
        assert raised.value.remote_error_code == code
        assert raised.value.remote_build_state == "failed"
        assert signed.failed == [(recorded, code)]
        assert signed.failed_artifacts == [("nb_remote_nb_b1", 4, "t1")]

    async def test_a_remote_failure_keeps_its_own_code(self, signed, http, tmp_path, monkeypatch):
        async def lost(*_args, **_kwargs):
            raise RemoteExecutionError("lost it", remote_error_code="E1")

        monkeypatch.setattr(dispatch, "await_accepted_job", lost)
        http.answers.append(httpx.Response(202, json={"job_url": "/jobs/1"}))

        with pytest.raises(RemoteExecutionError, match="lost it"):
            await self._run(tmp_path)

        assert signed.failed == [("lost it", "E1")]

    async def test_a_remote_failure_with_no_code_is_an_executor_error(
        self, signed, http, tmp_path, monkeypatch
    ):
        async def lost(*_args, **_kwargs):
            raise RemoteExecutionError("lost it")

        monkeypatch.setattr(dispatch, "await_accepted_job", lost)
        http.answers.append(httpx.Response(202, json={"job_url": "/jobs/1"}))

        with pytest.raises(RemoteExecutionError):
            await self._run(tmp_path)

        assert signed.failed == [("lost it", "EXECUTOR_ERROR")]

    async def test_a_cancel_fails_the_build_and_tells_the_worker(
        self, signed, http, tmp_path, monkeypatch
    ):
        cancelled = []

        async def cancel(url, build_id, token):
            cancelled.append((url, build_id, token))

        monkeypatch.setattr(dispatch, "cancel_remote_execution", cancel)
        http.answers.append(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await self._run(tmp_path, worker=_worker(transport="signed", token="secret"))

        assert signed.failed == [("Notebook manifest execution cancelled", "CANCELLED")]
        assert cancelled == [("http://worker:9/v1/execute", "b1", "secret")]

    async def test_a_failure_to_record_a_failure_does_not_hide_it(self, signed, http, tmp_path):
        def fail_build(*_args):
            raise OSError("store gone")

        signed.build_store.fail_build = fail_build
        signed.artifact_store.fail_artifact = fail_build
        http.answers.append(httpx.ConnectError("down"))

        with pytest.raises(RemoteExecutionError, match="request failed"):
            await self._run(tmp_path)

    # The answer

    async def test_a_ready_build_is_the_result(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(200))

        result, output, method, mounts = await self._run(tmp_path)

        assert result["stdout"] == "printed"
        assert output == tmp_path / "out" / "_executor_result"
        assert method == "executor"
        assert mounts == {}
        assert signed.failed == []

    async def test_a_408_is_the_cells_timeout(self, signed, http, tmp_path):
        http.answers.append(httpx.Response(408))

        with pytest.raises(RemoteExecutionError, match="c timed out after 7.0") as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "TIMEOUT"
        assert raised.value.remote_build_state == "failed"
        assert signed.failed == [("Notebook manifest execution timed out", "TIMEOUT")]

    async def test_a_refusal_fails_the_build(self, signed, http, tmp_path):
        signed.build = SimpleNamespace(state="running", error_code=None, error_message=None)
        http.answers.append(httpx.Response(500, json={"detail": "boom"}))

        with pytest.raises(RemoteExecutionError, match="returned 500: boom") as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "EXECUTOR_HTTP_ERROR"
        assert raised.value.remote_build_state == "running"
        assert signed.failed == [("Remote executor 'w' returned 500: boom", "EXECUTOR_HTTP_ERROR")]

    async def test_a_failed_finalize_is_named(self, signed, http, tmp_path):
        signed.build = None
        http.answers.append(
            httpx.Response(500, json={"detail": "Failed to finalize notebook bundle build b1"})
        )

        with pytest.raises(RemoteExecutionError) as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "FINALIZE_FAILED"
        assert raised.value.remote_build_state == "failed"

    async def test_the_failure_a_build_recorded_is_kept(self, signed, http, tmp_path):
        signed.build = SimpleNamespace(
            state="failed", error_code="OWN_CODE", error_message="its own words"
        )
        http.answers.append(httpx.Response(500, json={"detail": "boom"}))

        with pytest.raises(RemoteExecutionError, match="its own words") as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "OWN_CODE"
        assert raised.value.remote_build_state == "failed"

    async def test_a_build_that_is_not_ready_says_why(self, signed, http, tmp_path):
        signed.build = SimpleNamespace(
            state="running", error_code="OWN_CODE", error_message="its own words"
        )
        http.answers.append(httpx.Response(200))

        with pytest.raises(RemoteExecutionError, match="its own words") as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "OWN_CODE"
        assert raised.value.remote_build_state == "running"

    async def test_a_build_that_is_gone_is_unknown(self, signed, http, tmp_path):
        signed.build = None
        http.answers.append(httpx.Response(200))

        with pytest.raises(RemoteExecutionError, match="did not complete") as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "BUILD_FAILED"
        assert raised.value.remote_build_state == "unknown"

    async def test_a_ready_build_with_no_blob_fails(self, signed, http, tmp_path):
        signed.blob = None
        http.answers.append(httpx.Response(200))

        with pytest.raises(RemoteExecutionError, match="without a stored bundle") as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "MISSING_OUTPUT_BLOB"
        assert raised.value.remote_build_state == "failed"
        assert signed.failed[0][1] == "MISSING_OUTPUT_BLOB"

    async def test_a_bad_bundle_fails(self, signed, http, tmp_path):
        signed.blob = b"not a tar"
        http.answers.append(httpx.Response(200))

        with pytest.raises(RemoteExecutionError, match="invalid output bundle") as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "INVALID_NOTEBOOK_BUNDLE"
        assert raised.value.remote_build_state == "failed"
        assert signed.failed[0][1] == "INVALID_NOTEBOOK_BUNDLE"

    async def test_anything_else_is_an_executor_error(self, signed, http, tmp_path, monkeypatch):
        def unpack(*_args):
            raise ValueError("odd")

        monkeypatch.setattr(dispatch, "unpack_notebook_output_bundle", unpack)
        http.answers.append(httpx.Response(200))

        with pytest.raises(RemoteExecutionError, match="odd") as raised:
            await self._run(tmp_path)

        assert raised.value.remote_error_code == "EXECUTOR_ERROR"
        assert raised.value.remote_build_state == "failed"
        assert signed.failed == [("odd", "EXECUTOR_ERROR")]

    async def test_a_cancel_after_the_answer_fails_the_build(
        self, signed, http, tmp_path, monkeypatch
    ):
        def unpack(*_args):
            raise asyncio.CancelledError()

        monkeypatch.setattr(dispatch, "unpack_notebook_output_bundle", unpack)
        http.answers.append(httpx.Response(200))

        with pytest.raises(asyncio.CancelledError):
            await self._run(tmp_path)

        assert signed.failed == [("Notebook manifest execution cancelled", "CANCELLED")]


async def test_the_executor_hands_the_service_its_own_state(tmp_path):
    nb = create_notebook(tmp_path / "nb", "Dispatch")
    add_cell_to_notebook(nb, "c1", None)
    executor = CellExecutor(NotebookSession(parse_notebook(nb), nb))
    executor.origin = "run"

    context = executor._dispatch_context()

    assert context.origin == "run"
    assert context.received_output_files is executor._received_output_files


class TestTheExecutorsMethods:
    """A test replaces these on the class, so the service must reach the current ones."""

    @staticmethod
    def _executor(tmp_path):
        nb = create_notebook(tmp_path / "nb", "Dispatch")
        add_cell_to_notebook(nb, "c1", None)
        return CellExecutor(NotebookSession(parse_notebook(nb), nb))

    @pytest.mark.parametrize(
        ("method", "worker"),
        [
            ("_dispatch_local", SimpleNamespace(backend=WorkerBackendType.LOCAL)),
            ("_dispatch_embedded_executor", _worker("embedded://local")),
            ("_dispatch_http_executor", _worker()),
        ],
    )
    async def test_a_run_goes_to_the_method_for_its_backend(
        self, tmp_path, monkeypatch, method, worker
    ):
        called = []

        async def replaced(self, *args, **kwargs):
            called.append(method)
            return {}, Path("out"), method, {}

        monkeypatch.setattr(CellExecutor, method, replaced)

        await self._executor(tmp_path)._dispatch_execution(
            worker,
            "x = 1\n",
            {},
            [],
            Path("out"),
            Path("result"),
            Path("venv"),
            {},
            7.0,
        )

        assert called == [method]

    async def test_a_cells_language_reaches_the_worker(self, tmp_path, monkeypatch):
        seen = {}

        async def replaced(_host, _context, *_args, **kwargs):
            seen.update(kwargs)
            return {}, Path("out"), "executor", {}

        monkeypatch.setattr("strata.notebook.executor.dispatch_http", replaced)

        await self._executor(tmp_path)._dispatch_http_executor(
            _worker(), "x = 1\n", {}, [], Path("out"), {}, 7.0, language="r"
        )

        assert seen["language"] == "r"
