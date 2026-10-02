"""HTTP executor app for notebook cell execution."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import hmac
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask

from strata.blob_store import BLOB_STREAM_CHUNK_BYTES
from strata.notebook.credentials import CredentialResolver
from strata.notebook.hardware import hardware_report
from strata.notebook.models import MountSpec
from strata.notebook.mounts import MountResolver, parse_mount_uri
from strata.notebook.remote_bundle import pack_notebook_output_bundle
from strata.tracing import trace_span_from
from strata.types import EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION
from strata.url_safety import guarded_async_transport, host_is_allowlisted, url_safety_problem

logger = logging.getLogger(__name__)

# Small enough that a chatty cell surfaces promptly, large enough that
# per-line printing isn't one HTTP request per line.
_LOG_READ_CHUNK_BYTES = 8192

NOTEBOOK_EXECUTOR_PROTOCOL_VERSION = "notebook-cell-v1"
NOTEBOOK_EXECUTOR_TRANSFORM_REF = "notebook_cell@v1"
NOTEBOOK_EXECUTOR_MANIFEST_VERSION = "notebook-build-manifest@v1"

# --- Signed-URL manifest defenses ---
#
# The worker can't verify a signed URL's signature, so a compromised or buggy
# orchestrator could point it at internal services (SSRF) or unbounded streams
# (OOM). These checks don't depend on the orchestrator behaving.

# Per-input download cap. Override via STRATA_WORKER_MAX_INPUT_BYTES.
_DEFAULT_MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
# Matches the server's streamed reads and writes.
_INPUT_CHUNK_BYTES = 1024 * 1024


def _input_path(output_dir: Path, file_name: str) -> Path:
    """Path for a request-named input, refused unless directly in the run directory.

    The name is already reduced to its last component, and ``..`` is one.
    """
    # normpath + startswith rather than Path.resolve: the form static analysis
    # recognises as a path check.
    root = os.path.realpath(output_dir)
    target = os.path.normpath(os.path.join(root, file_name))
    if not target.startswith(root + os.sep) or os.path.dirname(target) != root:
        raise HTTPException(
            status_code=400, detail=f"Input file name {file_name!r} is not a plain file name"
        )
    return Path(target)


def _max_input_bytes() -> int:
    raw = os.environ.get("STRATA_WORKER_MAX_INPUT_BYTES")
    if not raw:
        return _DEFAULT_MAX_INPUT_BYTES
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_MAX_INPUT_BYTES
    return parsed if parsed > 0 else _DEFAULT_MAX_INPUT_BYTES


def _allow_local_hosts() -> bool:
    """Whether to bypass the host-IP SSRF check (``STRATA_WORKER_ALLOW_LOCAL_HOSTS=1``).

    For tests and local development only; production workers leave it unset.
    """
    return os.environ.get("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _allowed_hosts() -> tuple[str, ...]:
    """Hostnames and suffixes that pass regardless of the address they resolve to.

    ``STRATA_WORKER_ALLOWED_HOSTS``, comma-separated. A leading dot is a suffix
    (``.internal`` matches ``build.internal``); anything else must match exactly.
    """
    raw = os.environ.get("STRATA_WORKER_ALLOWED_HOSTS", "")
    return tuple(entry.strip().lower() for entry in raw.split(",") if entry.strip())


def _host_is_allowlisted(host: str) -> bool:
    """Whether *host* is named in ``STRATA_WORKER_ALLOWED_HOSTS``."""
    return host_is_allowlisted(host, _allowed_hosts())


def _assert_url_safe(url: str, field: str) -> None:
    """Reject manifest URLs that are scheme- or host-unsafe (see ``url_safety``)."""
    problem = url_safety_problem(
        url, f"Manifest {field}", allowed_hosts=_allowed_hosts(), allow_local=_allow_local_hosts()
    )
    if problem is not None:
        raise HTTPException(status_code=400, detail=problem)


def _guarded_transport() -> httpx.AsyncHTTPTransport | None:
    """A transport that connects manifest URLs only to addresses the guard passed.

    ``_assert_url_safe`` checks a URL when the manifest arrives; this holds the
    later connection to the same rule (see ``url_safety``).
    """
    return guarded_async_transport(allowed_hosts=_allowed_hosts(), allow_local=_allow_local_hosts())


# Console is advisory: a cell that outruns the link keeps its own speed.
_LOG_QUEUE_CHUNKS = 64


async def _post_log_chunk(client: httpx.AsyncClient, log_url: str, stream: str, text: str) -> None:
    """Forward one console chunk; never let doing so slow or fail the cell.

    Console is advisory and the bundle is the record, so this has a short timeout
    and swallows everything.
    """
    separator = "&" if "?" in log_url else "?"
    try:
        await client.post(
            f"{log_url}{separator}stream={stream}",
            content=text.encode("utf-8"),
        )
    except Exception:
        logger.debug("Could not forward %s chunk for the running cell", stream, exc_info=True)


async def _drain(proc: Any, log_url: str | None) -> tuple[bytes, bytes]:
    """Read both pipes to completion, forwarding chunks as they arrive when asked to.

    Used instead of ``communicate()``, which returns only at exit. Both pipes are
    read concurrently so a full buffer on one cannot block the process. Chunks per
    stream are posted in order (the notebook cannot reorder them) from their own
    task over one connection with a bounded queue, so a cell that prints faster
    than the link runs at its own speed and the oldest waiting chunks are dropped.
    """
    queue: asyncio.Queue[tuple[str, str]] | None = None
    forwarder: asyncio.Task[None] | None = None

    async def _forward() -> None:
        assert queue is not None and log_url is not None
        async with httpx.AsyncClient(timeout=5.0, transport=_guarded_transport()) as client:
            while True:
                item = await queue.get()
                try:
                    stream, text = item
                    await _post_log_chunk(client, log_url, stream, text)
                finally:
                    queue.task_done()

    async def _pump(reader: Any, stream: str) -> bytes:
        collected: list[bytes] = []
        while True:
            chunk = await reader.read(_LOG_READ_CHUNK_BYTES)
            if not chunk:
                break
            collected.append(chunk)
            if queue is not None:
                if queue.full():
                    # Drop the new chunk, not the oldest: what was shown stays a prefix, so the
                    # final report can send exactly the missing part.
                    continue
                queue.put_nowait((stream, chunk.decode("utf-8", errors="replace")))
        return b"".join(collected)

    if log_url:
        queue = asyncio.Queue(maxsize=_LOG_QUEUE_CHUNKS)
        forwarder = asyncio.create_task(_forward())
    try:
        stdout, stderr = await asyncio.gather(
            _pump(proc.stdout, "stdout"),
            _pump(proc.stderr, "stderr"),
        )
        await proc.wait()
        if queue is not None:
            # Give what is still queued a moment to land.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(queue.join(), timeout=10.0)
    finally:
        if forwarder is not None:
            forwarder.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await forwarder
    return stdout, stderr


async def _run_harness(
    harness_path: Path,
    manifest_path: Path,
    timeout_seconds: float,
    *,
    in_flight: dict[str, Any] | None = None,
    build_id: str | None = None,
    log_url: str | None = None,
    env: dict[str, str] | None = None,
    interpreter: Path | None = None,
) -> dict[str, Any]:
    """Run the notebook harness with one manifest file.

    With *interpreter*, the harness runs under it rather than the worker's own
    Python: the notebook's locked environment (``worker_env``), or ``Rscript`` for
    ``harness.R``. With *build_id* and *in_flight*, the process is registered so the
    cancel route can reach it. With *log_url*, output is also forwarded live to the
    dispatching server; the bundle is the same either way.
    """
    from strata.notebook.process_tree import (
        subprocess_kwargs_for_new_group,
        terminate_subprocess_tree,
    )

    proc = await asyncio.create_subprocess_exec(
        str(interpreter) if interpreter is not None else sys.executable,
        str(harness_path),
        str(manifest_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        **subprocess_kwargs_for_new_group(),
    )
    if in_flight is not None and build_id:
        in_flight[build_id] = proc
    try:
        _stdout, stderr = await asyncio.wait_for(
            _drain(proc, log_url),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        await terminate_subprocess_tree(proc)
        raise TimeoutError()
    finally:
        if in_flight is not None and build_id:
            in_flight.pop(build_id, None)

    result_path = manifest_path.parent / "harness-result.json"
    if not result_path.exists():
        raise RuntimeError(f"Harness did not produce harness-result.json: {stderr.decode()}")

    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


# A cell holds a slot while it runs, so this is a polling interval, not a promise.
RETRY_AFTER_SECONDS = 5


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


# Never handed to a cell: the bearer token (a cell holding it could dispatch
# as the server) and the credentials mounts and connections resolve against.
_WORKER_SECRETS = (
    "STRATA_WORKER_TOKEN",
    "STRATA_NOTEBOOK_CREDENTIALS",
    "STRATA_NOTEBOOK_MOUNT_CREDENTIALS",
    "STRATA_PROXY_TOKEN",
    "STRATA_NOTEBOOK_REMOTE_STORE_HEADERS",
)


_CAPTURED_SECRETS: dict[str, str] = {}


def _scrub_environ_value(name: str) -> None:
    """Zero the bytes of ``name``'s value in libc's environment block.

    ``/proc/<pid>/environ`` reads the process's original environment from memory,
    and unsetting a variable leaves those bytes in place.
    """
    if sys.platform == "win32":
        return
    try:
        getenv = ctypes.CDLL(None).getenv
    except (OSError, AttributeError) as exc:
        logger.warning("Could not scrub worker secrets from the environment block: %s", exc)
        return
    getenv.argtypes = [ctypes.c_char_p]
    getenv.restype = ctypes.c_void_p
    address = getenv(name.encode())
    if address:
        ctypes.memset(address, 0, len(ctypes.string_at(address)))


def capture_worker_secrets() -> None:
    """Take the worker's secrets out of the process environment, into memory.

    A cell runs as a child under the same uid and could read them from
    ``/proc/<ppid>/environ``, so each value is zeroed in the environment block
    before it is unset. Called by the worker entry point, so an in-process app in a
    test still reads the environment.
    """
    for name in _WORKER_SECRETS:
        value = os.environ.get(name)
        if value is None:
            continue
        _CAPTURED_SECRETS[name] = value
        _scrub_environ_value(name)
        os.environ.pop(name)


def worker_secret(name: str) -> str:
    """One of the worker's secrets, wherever it is now."""
    return _CAPTURED_SECRETS.get(name) or os.environ.get(name, "") or ""


def _cell_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a cell's harness runs with on a worker.

    ``STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST`` narrows it as on the server, and the
    worker's own secrets are dropped whatever it says.
    """
    from strata.notebook.harness_env import harness_env

    allowlist = _configured_allowlist()
    env = harness_env(allowlist, extra) if allowlist else {**os.environ, **(extra or {})}
    for name in _WORKER_SECRETS:
        env.pop(name, None)
    return env


def _configured_allowlist() -> list[str]:
    """``STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST``, in either form the server takes.

    A comma-separated list or a JSON array; reading only one form would silently
    narrow a cell's environment to nothing.
    """
    raw = (os.environ.get("STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST") or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return []
        return [str(entry).strip() for entry in parsed if str(entry).strip()]
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _worker_credentials() -> CredentialResolver:
    """Named credentials from the worker's environment.

    Read from ``STRATA_NOTEBOOK_CREDENTIALS`` and ``STRATA_NOTEBOOK_MOUNT_CREDENTIALS``
    directly: a worker is not a Strata server and has no server config.
    """
    return CredentialResolver(
        json.loads(worker_secret("STRATA_NOTEBOOK_CREDENTIALS") or "{}"),
        scheme_defaults=json.loads(worker_secret("STRATA_NOTEBOOK_MOUNT_CREDENTIALS") or "{}"),
    )


def create_notebook_executor_app(
    max_concurrent: int | None = None,
    gpu_slots: int | None = None,
) -> FastAPI:
    """Create a standalone notebook executor HTTP app.

    With ``STRATA_WORKER_TOKEN`` set, the ``/v1/*`` endpoints require
    ``Authorization: Bearer <token>``; ``/health`` stays open for platform probes.
    Unset means no auth.

    Args:
        max_concurrent: Executions run at once; one more gets 503 with
            ``Retry-After``. ``None`` is unlimited. Enforced here so a caller that
            skips the dispatcher cannot overcommit the machine. Falls back to
            ``STRATA_WORKER_MAX_CONCURRENT``.
        gpu_slots: GPUs handed out one per execution via ``CUDA_VISIBLE_DEVICES``,
            overriding the caller's value so concurrent cells never share a GPU.
            Falls back to ``STRATA_WORKER_GPU_SLOTS``.
    """
    started_at = time.time()
    active_executions = 0
    if max_concurrent is None:
        max_concurrent = _positive_int_env("STRATA_WORKER_MAX_CONCURRENT")
    if gpu_slots is None:
        gpu_slots = _positive_int_env("STRATA_WORKER_GPU_SLOTS")
    free_gpus: list[int] = list(range(gpu_slots or 0))

    def _admit() -> int | None:
        """Reserve a slot for one execution, or refuse; returns its GPU, if any.

        No ``await`` between the check and the reservation, so two requests arriving
        together cannot both take the last slot.
        """
        nonlocal active_executions
        if max_concurrent is not None and active_executions >= max_concurrent:
            raise HTTPException(
                status_code=503,
                detail=f"Worker is running {active_executions} of {max_concurrent} executions",
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )
        gpu: int | None = None
        if gpu_slots:
            if not free_gpus:
                raise HTTPException(
                    status_code=503,
                    detail=f"All {gpu_slots} GPU slots are in use",
                    headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
                )
            gpu = free_gpus.pop(0)
        active_executions += 1
        return gpu

    def _release(gpu: int | None) -> None:
        nonlocal active_executions
        active_executions -= 1
        if gpu is not None:
            free_gpus.append(gpu)
            free_gpus.sort()

    # build_id -> harness process, for cancel. Per app so two workers in one
    # test process can't cancel each other's runs.
    in_flight: dict[str, Any] = {}

    # ---- Bearer-token gate ----
    expected_token = worker_secret("STRATA_WORKER_TOKEN").strip() or None

    async def require_worker_token(http_request: Request) -> None:
        if expected_token is None:
            return
        header = http_request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or malformed Authorization header (expected Bearer token)",
            )
        presented = header[len("Bearer ") :]
        # Constant-time, so timing doesn't leak token length. Bytes, because
        # ``compare_digest`` raises TypeError on non-ASCII str.
        if not hmac.compare_digest(presented.encode(), expected_token.encode()):
            raise HTTPException(status_code=401, detail="Invalid worker token")

    def _input_extension(content_type: str) -> str:
        return {
            "arrow/ipc": ".arrow",
            "json/object": ".json",
            "pickle/object": ".pickle",
            "module/import": ".module.json",
            "module/cell": ".cell_module.json",
            "module/cell-instance": ".cell_instance.pickle",
        }.get(content_type, ".bin")

    def _response_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except Exception:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("error")
            if detail:
                return str(detail)

        text = response.text.strip()
        return text or f"HTTP {response.status_code}"

    async def _execute_to_bundle(
        *,
        source: str,
        timeout_seconds: float,
        raw_inputs: dict[str, dict[str, Any]],
        raw_mounts: list[dict[str, Any]],
        runtime_env: dict[str, str],
        write_input: Any,
        build_id: str | None = None,
        log_url: str | None = None,
        trace_carrier: dict[str, Any] | None = None,
        notebook_id: str | None = None,
        cell_id: str | None = None,
        environment: Any = None,
        language: str = "python",
        mutation_defines: list[str] | None = None,
        tables: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[Path, Path] | JSONResponse:
        """Execute a cell and pack outputs into a bundle file.

        On success returns ``(bundle_path, tmpdir)`` and the caller deletes ``tmpdir``
        once the bundle is consumed. On failure returns a ``JSONResponse`` with the
        tmpdir already removed.
        """
        if not isinstance(raw_inputs, dict):
            raise HTTPException(status_code=400, detail="inputs must be an object")
        if not isinstance(raw_mounts, list):
            raise HTTPException(status_code=400, detail="mounts must be a list")

        mount_specs = [MountSpec(**mount) for mount in raw_mounts]
        for mount in mount_specs:
            scheme, _path = parse_mount_uri(mount.uri)
            if scheme == "file":
                raise HTTPException(
                    status_code=400,
                    detail=(f"Remote execution does not support file:// mount '{mount.name}'"),
                )

        # Before any download, so a refused request costs nothing; held until the harness exits.
        gpu = _admit()
        try:
            # Parented to the dispatcher's span (the server's, or a pool's) when it sent context.
            with trace_span_from(
                "worker.execute",
                trace_carrier,
                build_id=build_id,
                notebook_id=notebook_id,
                cell_id=cell_id,
            ):
                return await _stage_and_run(
                    source=source,
                    timeout_seconds=timeout_seconds,
                    raw_inputs=raw_inputs,
                    mount_specs=mount_specs,
                    runtime_env=runtime_env,
                    write_input=write_input,
                    build_id=build_id,
                    log_url=log_url,
                    gpu=gpu,
                    environment=environment,
                    language=language,
                    mutation_defines=mutation_defines,
                    tables=tables,
                )
        finally:
            _release(gpu)

    async def _stage_and_run(
        *,
        source: str,
        timeout_seconds: float,
        raw_inputs: dict[str, dict[str, Any]],
        mount_specs: list[MountSpec],
        runtime_env: dict[str, str],
        write_input: Any,
        build_id: str | None,
        log_url: str | None,
        gpu: int | None,
        environment: Any = None,
        language: str = "python",
        mutation_defines: list[str] | None = None,
        tables: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[Path, Path] | JSONResponse:
        if gpu is not None:
            # Set in both: the manifest env reaches the cell's own CUDA init, the process
            # env covers what the harness imports first.
            runtime_env = {**runtime_env, "CUDA_VISIBLE_DEVICES": str(gpu)}
        tmpdir = Path(tempfile.mkdtemp(prefix="strata_notebook_executor_"))
        try:
            output_dir = tmpdir

            inputs: dict[str, dict[str, Any]] = {}
            for var_name, spec in raw_inputs.items():
                if not isinstance(spec, dict):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input spec for {var_name} must be an object",
                    )
                content_type = str(spec.get("content_type", "pickle/object"))
                requested_file_name = Path(str(spec.get("file", ""))).name
                file_name = requested_file_name or f"{var_name}{_input_extension(content_type)}"
                await write_input(var_name, file_name, spec, _input_path(output_dir, file_name))
                inputs[var_name] = {
                    "content_type": content_type,
                    "file": file_name,
                }
                # Injected values a module/cell export's defs close over; the harness hydrates them.
                injected_map = spec.get("injected")
                if isinstance(injected_map, dict):
                    resolved_injected: dict[str, dict[str, str]] = {}
                    for inj_index, inj_key in enumerate(injected_map):
                        inj_spec = injected_map[inj_key]
                        if not isinstance(inj_spec, dict):
                            continue
                        inj_name = str(inj_key)
                        inj_ct = str(inj_spec.get("content_type", "pickle/object"))
                        # The client's name is only a lookup key; the on-disk name is built locally
                        # so no request value reaches a filesystem path.
                        inj_lookup = Path(str(inj_spec.get("file", ""))).name
                        safe_file = f"__inj_{len(inputs)}_{inj_index}{_input_extension(inj_ct)}"
                        await write_input(inj_name, inj_lookup, inj_spec, output_dir / safe_file)
                        resolved_injected[inj_name] = {"content_type": inj_ct, "file": safe_file}
                    if resolved_injected:
                        inputs[var_name]["injected"] = resolved_injected

            mount_resolver = MountResolver(
                cache_dir=output_dir / "mount_cache",
                # Resolved against the worker's own config; only the name travels.
                credential_resolver=_worker_credentials(),
            )
            resolved_mounts = await mount_resolver.prepare_mounts(mount_specs)
            manifest_mounts = {
                name: {
                    "uri": rm.spec.uri,
                    "mode": rm.spec.mode.value,
                    "local_path": str(rm.local_path),
                }
                for name, rm in resolved_mounts.items()
            }

            manifest = {
                "source": source,
                "inputs": inputs,
                "output_dir": str(output_dir),
                "mounts": manifest_mounts,
                "env": runtime_env,
                # Same as a local harness manifest: without them the worker skipped mutation
                # recapture and left @table names undefined.
                "mutation_defines": list(mutation_defines or []),
                "tables": tables or {},
                # Only when a reader forwards it (``_drain`` posts chunks to log_url);
                # otherwise the bundle carries the console at the end.
                "stream_console": log_url is not None,
            }
            manifest_path = output_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            harness_path = Path(__file__).parent / "harness.py"
            interpreter: Path | None = None
            prepared = None
            if language == "r":
                # R cells run under the worker's Rscript; the Python lock doesn't apply.
                rscript = shutil.which("Rscript")
                if rscript is None:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    return JSONResponse(
                        status_code=500,
                        content={
                            "success": False,
                            "error": "Rscript is not installed on this worker",
                        },
                    )
                harness_path = Path(__file__).parent / "languages" / "r" / "harness.R"
                interpreter = Path(rscript)
            elif language != "python":
                shutil.rmtree(tmpdir, ignore_errors=True)
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": f"unsupported cell language {language!r}"},
                )
            elif environment is not None:
                from strata.notebook.worker_env import WorkerEnvironmentError, ensure_environment

                try:
                    prepared = await ensure_environment(environment)
                except WorkerEnvironmentError as exc:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "error": f"locked environment: {exc}"},
                    )
                interpreter = prepared.python
            try:
                result = await _run_harness(
                    harness_path,
                    manifest_path,
                    timeout_seconds,
                    in_flight=in_flight,
                    build_id=build_id,
                    log_url=log_url,
                    env=_cell_env({"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else None),
                    interpreter=interpreter,
                )
                if result.get("success", False):
                    await mount_resolver.sync_back(resolved_mounts)
            except TimeoutError:
                shutil.rmtree(tmpdir, ignore_errors=True)
                # Lazy: keeps the heavy executor module out of worker load on a rare path.
                from strata.notebook.executor import cell_timeout_message

                return JSONResponse(
                    status_code=408,
                    content={
                        "success": False,
                        "error": cell_timeout_message(timeout_seconds),
                    },
                )
            except Exception as exc:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": str(exc)},
                )

            bundle_path = output_dir / "notebook-output-bundle.tar"
            # Records the machine that computed it, not only what was asked for.
            result = {**result, "hardware": await asyncio.to_thread(hardware_report)}
            if prepared is not None:
                result["environment"] = {"key": prepared.key, "installed": prepared.installed}
            pack_notebook_output_bundle(bundle_path, result, output_dir)
            return bundle_path, tmpdir
        except BaseException:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

    async def _run_notebook_execution(
        *,
        source: str,
        timeout_seconds: float,
        raw_inputs: dict[str, dict[str, Any]],
        raw_mounts: list[dict[str, Any]],
        runtime_env: dict[str, str],
        form: Any,
        build_id: str | None = None,
        log_url: str | None = None,
        trace_carrier: dict[str, Any] | None = None,
        environment: Any = None,
        language: str = "python",
        mutation_defines: list[str] | None = None,
        tables: dict[str, dict[str, Any]] | None = None,
    ) -> Response:
        async def _write_uploaded_input(
            var_name: str,
            requested_file_name: str,
            _spec: dict[str, Any],
            target: Path,
        ) -> None:
            upload = form.get(var_name) or form.get(requested_file_name)
            if upload is None or isinstance(upload, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing uploaded input file: {var_name}",
                )
            # Chunked: the form parser already spooled the part to disk.
            with open(target, "wb") as out:
                while chunk := await upload.read(_INPUT_CHUNK_BYTES):
                    out.write(chunk)

        result = await _execute_to_bundle(
            source=source,
            timeout_seconds=timeout_seconds,
            raw_inputs=raw_inputs,
            raw_mounts=raw_mounts,
            runtime_env=runtime_env,
            write_input=_write_uploaded_input,
            build_id=build_id,
            log_url=log_url,
            trace_carrier=trace_carrier,
            environment=environment,
            language=language,
            mutation_defines=mutation_defines,
            tables=tables,
        )
        if isinstance(result, JSONResponse):
            return result
        bundle_path, tmpdir = result
        return FileResponse(
            path=bundle_path,
            media_type="application/x-tar",
            headers={
                "X-Strata-Notebook-Executor-Protocol": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
                EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION,
            },
            background=BackgroundTask(shutil.rmtree, tmpdir, True),
        )

    app = FastAPI(
        title="Strata Notebook Executor",
        description="Reference notebook executor for remote notebook workers",
        version="1.0.0",
    )
    # Exposed for the cancel route's end-to-end tests and for operators.
    app.state.in_flight = in_flight

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "capabilities": {
                "protocol_versions": [EXECUTOR_PROTOCOL_VERSION],
                "transform_refs": [NOTEBOOK_EXECUTOR_TRANSFORM_REF],
                "features": {
                    "notebook_protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
                    "output_format": "notebook-output-bundle@v1",
                    "pull_model": True,
                    "cancel": True,
                    # Building the locked env is ``uv sync --frozen``, so probe for uv: claimed
                    # unconditionally, a pip-installed image would fail every Python cell with 500.
                    "locked_environments": shutil.which("uv") is not None,
                    # R needs Rscript with jsonlite and arrow in its library.
                    "languages": ["python", "r"] if shutil.which("Rscript") else ["python"],
                },
            },
            "version": "1.0.0",
            "uptime_seconds": max(0.0, time.time() - started_at),
            "active_executions": active_executions,
            # So a caller can plan rather than discover the limit by 503. ``None`` = unlimited.
            "max_concurrent": max_concurrent,
            "gpu_slots": gpu_slots,
            "free_gpu_slots": len(free_gpus) if gpu_slots else None,
            # Lets a caller check a machine against its sold class without a job.
            # Missing fields mean unknown.
            "hardware": await asyncio.to_thread(hardware_report),
        }

    @app.post("/v1/executions/{build_id}/cancel", dependencies=[Depends(require_worker_token)])
    async def cancel_execution(build_id: str) -> dict[str, Any]:
        """Stop the harness running *build_id*, if it is still running.

        A cancelled build's result would be refused anyway, so this frees the machine.
        ``cancelled: false`` is a normal answer: the execution may have just finished,
        and a caller treating it as a failure would retire a healthy worker.
        """
        proc = in_flight.get(build_id)
        if proc is None:
            return {"build_id": build_id, "cancelled": False}

        from strata.notebook.process_tree import terminate_subprocess_tree

        # The tree: spawned DataLoader workers or pools would otherwise hold the GPU.
        await terminate_subprocess_tree(proc)
        return {"build_id": build_id, "cancelled": True}

    @app.post("/v1/notebook-execute", dependencies=[Depends(require_worker_token)])
    async def execute(http_request: Request) -> Response:
        form = await http_request.form()
        metadata_file = form.get("metadata")
        if metadata_file is None or isinstance(metadata_file, str):
            raise HTTPException(status_code=400, detail="Missing metadata")

        try:
            metadata = json.loads((await metadata_file.read()).decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid metadata: {exc}")

        protocol_version = metadata.get("protocol_version")
        if protocol_version != NOTEBOOK_EXECUTOR_PROTOCOL_VERSION:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported protocol version: {protocol_version}. "
                    f"Expected: {NOTEBOOK_EXECUTOR_PROTOCOL_VERSION}"
                ),
            )

        source = str(metadata.get("source", ""))
        timeout_seconds = float(metadata.get("timeout_seconds", 30.0))
        raw_inputs = metadata.get("inputs", {})
        raw_mounts = metadata.get("mounts", [])
        runtime_env = metadata.get("env", {})

        return await _run_notebook_execution(
            source=source,
            timeout_seconds=timeout_seconds,
            raw_inputs=raw_inputs,
            raw_mounts=raw_mounts,
            runtime_env=runtime_env,
            form=form,
            build_id=str(metadata.get("build_id") or "") or None,
            trace_carrier=dict(http_request.headers),
            mutation_defines=metadata.get("mutation_defines"),
            tables=metadata.get("tables"),
            environment=metadata.get("environment"),
            language=str(metadata.get("language") or "python"),
        )

    @app.post("/v1/execute", dependencies=[Depends(require_worker_token)])
    async def execute_protocol_v1(http_request: Request) -> Response:
        """Execute notebook cells using the standard executor v1 metadata envelope."""
        form = await http_request.form()
        metadata_file = form.get("metadata")
        if metadata_file is None or isinstance(metadata_file, str):
            raise HTTPException(status_code=400, detail="Missing metadata")

        try:
            metadata = json.loads((await metadata_file.read()).decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid metadata: {exc}")

        protocol_version = metadata.get("protocol_version", EXECUTOR_PROTOCOL_VERSION)
        if protocol_version != EXECUTOR_PROTOCOL_VERSION:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported protocol version: {protocol_version}. "
                    f"Expected: {EXECUTOR_PROTOCOL_VERSION}"
                ),
            )

        transform = metadata.get("transform", {})
        transform_ref = str(transform.get("ref", ""))
        if transform_ref != NOTEBOOK_EXECUTOR_TRANSFORM_REF:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported transform: {transform_ref}. "
                    f"Expected: {NOTEBOOK_EXECUTOR_TRANSFORM_REF}"
                ),
            )

        params = transform.get("params", {})
        source = str(params.get("source", ""))
        timeout_seconds = float(params.get("timeout_seconds", 30.0))
        raw_mounts = params.get("mounts", [])
        runtime_env = params.get("env", {})
        input_descriptors = metadata.get("inputs", [])

        if not isinstance(input_descriptors, list):
            raise HTTPException(status_code=400, detail="inputs must be a list")

        raw_inputs: dict[str, dict[str, Any]] = {}
        for descriptor in input_descriptors:
            if not isinstance(descriptor, dict):
                raise HTTPException(status_code=400, detail="input descriptor must be an object")
            name = str(descriptor.get("name", "")).strip()
            if not name:
                raise HTTPException(status_code=400, detail="input descriptor missing name")
            content_type = str(descriptor.get("format", "pickle/object"))
            # The recorded filename is case-safe (uppercase names get a hash suffix).
            # Older clients omit it, and their {name}{ext} matched the upload name.
            recorded_file = str(descriptor.get("file", "")).strip()
            raw_inputs[name] = {
                "content_type": content_type,
                "file": recorded_file or f"{name}{_input_extension(content_type)}",
            }
            if isinstance(descriptor.get("injected"), dict):
                raw_inputs[name]["injected"] = descriptor["injected"]

        return await _run_notebook_execution(
            source=source,
            timeout_seconds=timeout_seconds,
            raw_inputs=raw_inputs,
            raw_mounts=raw_mounts,
            runtime_env=runtime_env,
            form=form,
            build_id=str(metadata.get("build_id") or "") or None,
            trace_carrier=dict(http_request.headers),
            environment=params.get("environment"),
            language=str(params.get("language") or "python"),
            mutation_defines=params.get("mutation_defines"),
            tables=params.get("tables"),
        )

    @app.post("/v1/execute-manifest", dependencies=[Depends(require_worker_token)])
    async def execute_manifest(http_request: Request) -> Response:
        """Execute a notebook build from a signed manifest."""
        try:
            manifest = await http_request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid manifest payload: {exc}")

        if not isinstance(manifest, dict):
            raise HTTPException(status_code=400, detail="Manifest payload must be an object")

        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="Manifest metadata must be an object")

        executor_ref = str(metadata.get("executor_ref", ""))
        if executor_ref != NOTEBOOK_EXECUTOR_TRANSFORM_REF:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported executor ref: {executor_ref}. "
                    f"Expected: {NOTEBOOK_EXECUTOR_TRANSFORM_REF}"
                ),
            )

        params = metadata.get("params", {})
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="Manifest params must be an object")

        raw_inputs = params.get("input_specs", {})
        raw_mounts = params.get("mounts", [])
        runtime_env = params.get("env", {})
        source = str(params.get("source", ""))
        timeout_seconds = float(params.get("timeout_seconds", 30.0))

        input_urls = manifest.get("inputs", [])
        if not isinstance(input_urls, list):
            raise HTTPException(status_code=400, detail="Manifest inputs must be a list")

        input_url_by_uri: dict[str, str] = {}
        for item in input_urls:
            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Manifest input entry must be an object",
                )
            artifact_id = str(item.get("artifact_id", "")).strip()
            version = item.get("version")
            url = str(item.get("url", "")).strip()
            if not artifact_id or not url or not isinstance(version, int):
                raise HTTPException(status_code=400, detail="Manifest input entry is incomplete")
            _assert_url_safe(url, f"input[{artifact_id}@v={version}]")
            input_url_by_uri[f"strata://artifact/{artifact_id}@v={version}"] = url

        output = manifest.get("output", {})
        if not isinstance(output, dict):
            raise HTTPException(status_code=400, detail="Manifest output must be an object")
        upload_url = str(output.get("url", "")).strip()
        finalize_url = str(manifest.get("finalize_url", "")).strip()
        if not upload_url or not finalize_url:
            raise HTTPException(status_code=400, detail="Manifest is missing upload/finalize URLs")
        _assert_url_safe(upload_url, "output.url")
        _assert_url_safe(finalize_url, "finalize_url")

        # Optional (older servers omit it). SSRF-checked like every manifest URL,
        # since the worker POSTs to it.
        log_url = str(manifest.get("log_url", "")).strip() or None
        if log_url:
            _assert_url_safe(log_url, "log_url")

        async def _download_input(
            var_name: str,
            _requested_file_name: str,
            spec: dict[str, Any],
            target: Path,
        ) -> None:
            input_uri = str(spec.get("uri", "")).strip()
            if not input_uri:
                raise HTTPException(
                    status_code=400,
                    detail=f"Manifest input spec for {var_name} is missing uri",
                )
            download_url = input_url_by_uri.get(input_uri)
            if download_url is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Manifest does not include a signed URL for {input_uri}",
                )
            # Streamed with the cap checked as bytes land, so inputs are bounded by disk,
            # not memory. Content-Length, when present, rejects up front.
            max_bytes = _max_input_bytes()
            async with httpx.AsyncClient(
                timeout=max(timeout_seconds, 30.0), transport=_guarded_transport()
            ) as client:
                async with client.stream("GET", download_url) as response:
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                f"Failed to download notebook input {input_uri}: "
                                f"{response.status_code}"
                            ),
                        )
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            declared_bytes = int(declared)
                        except ValueError:
                            declared_bytes = -1
                        if declared_bytes > max_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"Notebook input {input_uri} declared "
                                    f"{declared_bytes} bytes, exceeds {max_bytes}-byte cap"
                                ),
                            )
                    written = 0
                    with open(target, "wb") as out:
                        async for chunk in response.aiter_bytes(_INPUT_CHUNK_BYTES):
                            written += len(chunk)
                            if written > max_bytes:
                                raise HTTPException(
                                    status_code=413,
                                    detail=(
                                        f"Notebook input {input_uri} exceeds "
                                        f"{max_bytes}-byte cap during download"
                                    ),
                                )
                            out.write(chunk)

        bundle_result = await _execute_to_bundle(
            source=source,
            timeout_seconds=timeout_seconds,
            raw_inputs=raw_inputs,
            raw_mounts=raw_mounts,
            runtime_env=runtime_env,
            write_input=_download_input,
            build_id=str(metadata.get("build_id") or "") or None,
            log_url=log_url,
            # Headers first: a pool in between forwards its own, nearer span. The
            # manifest's copy is the server's, for dispatchers that send only the body.
            trace_carrier=(
                dict(http_request.headers) if "traceparent" in http_request.headers else metadata
            ),
            notebook_id=str(metadata.get("notebook_id") or "") or None,
            cell_id=str(metadata.get("cell_id") or "") or None,
            environment=params.get("environment"),
            language=str(params.get("language") or "python"),
            mutation_defines=params.get("mutation_defines"),
            tables=params.get("tables"),
        )
        if isinstance(bundle_result, JSONResponse):
            return bundle_result

        bundle_path, tmpdir = bundle_result
        try:
            byte_size = bundle_path.stat().st_size

            async def _stream_bundle_body() -> AsyncIterator[bytes]:
                with open(bundle_path, "rb") as f:
                    while chunk := f.read(BLOB_STREAM_CHUNK_BYTES):
                        yield chunk

            upload_fields = output.get("fields")
            try:
                async with httpx.AsyncClient(
                    timeout=max(timeout_seconds, 30.0), transport=_guarded_transport()
                ) as client:
                    if isinstance(upload_fields, dict):
                        # Presigned upload straight to the object store, bypassing the server.
                        with open(bundle_path, "rb") as bundle_file:
                            upload_response = await client.post(
                                upload_url,
                                data={str(k): str(v) for k, v in upload_fields.items()},
                                files={"file": ("bundle.tar", bundle_file, "application/x-tar")},
                            )
                    else:
                        upload_response = await client.post(
                            upload_url,
                            content=_stream_bundle_body(),
                            headers={
                                "Content-Type": "application/x-tar",
                                "Content-Length": str(byte_size),
                            },
                        )
                    if upload_response.status_code not in (200, 201, 204):
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                "Failed to upload notebook bundle output: "
                                f"{upload_response.status_code} "
                                f"({_response_error_detail(upload_response)})"
                            ),
                        )

                    finalize_response = await client.post(
                        finalize_url,
                        json={"output_format": "notebook-output-bundle@v1"},
                    )
            except httpx.TimeoutException as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Notebook bundle transfer timed out: {exc}",
                ) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Notebook bundle transfer failed: {exc}",
                ) from exc

            if finalize_response.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Failed to finalize notebook bundle build: "
                        f"{finalize_response.status_code} "
                        f"({_response_error_detail(finalize_response)})"
                    ),
                )

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "build_id": manifest.get("build_id"),
                    "byte_size": byte_size,
                    "protocol_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
                    "finalize": finalize_response.json(),
                },
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @app.post("/execute", dependencies=[Depends(require_worker_token)])
    async def execute_pool_contract(http_request: Request) -> Response:
        """The worker-pool contract path: an alias of ``/v1/execute-manifest``.

        ``strata-pool`` posts the job payload verbatim with no content type, which rules
        out the multipart ``/v1/*`` endpoints. Sharing one handler keeps a pooled
        manifest and a pushed one from diverging in what they accept.
        """
        return await execute_manifest(http_request)

    return app


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point: run the notebook executor HTTP app.

    Used by ``python -m strata.notebook.remote_executor --port 9000`` and by
    single-process deployment images.
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="strata-worker",
        description=(
            "Run a Strata notebook worker — an HTTP endpoint that accepts "
            "cells and returns their outputs. Cells run in the Python "
            "environment this process was started in, so install your "
            "workload dependencies (pandas, torch, datafusion, ...) before "
            "launching."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000, help="Bind port (default: 9000)")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Uvicorn log level",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=(
            "Executions to run at once; more are refused with 503 and Retry-After "
            "(default: unlimited, or STRATA_WORKER_MAX_CONCURRENT)"
        ),
    )
    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=None,
        help=(
            "GPUs to hand out one per execution, via CUDA_VISIBLE_DEVICES set by the "
            "worker (default: none, or STRATA_WORKER_GPU_SLOTS)"
        ),
    )
    args = parser.parse_args(argv)
    for flag, value in (("--max-concurrent", args.max_concurrent), ("--gpu-slots", args.gpu_slots)):
        if value is not None and value < 1:
            parser.error(f"{flag} must be a positive integer")

    # Before any cell can spawn: a harness under this uid can read
    # /proc/<ppid>/environ, so secrets move from the environment into memory.
    capture_worker_secrets()

    # Cells run arbitrary code, so a non-loopback bind without a token lets
    # anyone who reaches the port run code as this user. Make that loud.
    if (
        args.host not in ("127.0.0.1", "localhost", "::1")
        and not worker_secret("STRATA_WORKER_TOKEN").strip()
    ):
        logger.warning(
            "strata-worker is binding %s WITHOUT authentication - anyone who can "
            "reach port %d can execute code on this machine. Set "
            "STRATA_WORKER_TOKEN (see docs/notebook/workers.md) or bind "
            "--host 127.0.0.1.",
            args.host,
            args.port,
        )

    app = create_notebook_executor_app(max_concurrent=args.max_concurrent, gpu_slots=args.gpu_slots)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
