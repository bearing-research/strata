"""Running a cell elsewhere: picking the backend, then embedded, HTTP or signed transport."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse, urlsplit, urlunparse

import anyio.to_thread
import httpx
from vestigo.blob_store import BLOB_STREAM_CHUNK_BYTES
from vestigo.build_store import get_build_store
from vestigo.provenance import TransformSpec as ArtifactTransformSpec
from vestigo.provenance import derive_subkey
from vestigo.store import get_artifact_store

from strata.notebook import console_relay, output_dirs, output_file_relay
from strata.notebook.annotations import parse_annotations
from strata.notebook.cell_paths.context import RemoteExecutionError
from strata.notebook.credentials import CredentialResolver
from strata.notebook.local_runs import LocalRun
from strata.notebook.models import CellLanguage, MountSpec, WorkerBackendType
from strata.notebook.mounts import ResolvedMount, ensure_writable, parse_mount_uri
from strata.notebook.provenance import compute_source_hash
from strata.notebook.remote_bundle import (
    pack_notebook_output_bundle,
    read_notebook_output_bundle_manifest_path,
    unpack_notebook_output_bundle,
)
from strata.notebook.remote_executor import (
    NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
    NOTEBOOK_EXECUTOR_TRANSFORM_REF,
)
from strata.notebook.serializer import ContentType
from strata.notebook.workers import (
    SIGNED_TRANSPORTS,
    is_embedded_executor_worker,
    is_http_executor_worker,
    worker_transport,
)
from strata.observability.tracing import current_trace_context, trace_span
from strata.types import EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION

if TYPE_CHECKING:
    from strata.notebook.executor_host import ExecutorHost

logger = logging.getLogger(__name__)

# Module-level so a test can make polling immediate.
_JOB_POLL_SECONDS = 1.0

# Patchable for the same reason.
_monotonic = time.monotonic

# A dispatcher's sentence goes on a badge; anything longer is cut.
_QUEUE_DETAIL_MAX_CHARS = 200


@dataclass(frozen=True)
class DispatchRoutes:
    """The backends a run can go to; each is a method tests replace on the executor."""

    local: Callable[..., Awaitable[tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]]]
    embedded: Callable[..., Awaitable[tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]]]
    http: Callable[..., Awaitable[tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]]]


@dataclass(frozen=True)
class EmbeddedContext:
    """What a bundle-based run on this machine asks of the executor."""

    prepare_mounts: Callable[..., Awaitable[dict[str, ResolvedMount]]]
    run_token: Callable[[float], Any]
    write_manifest: Callable[..., Path]
    run_harness: Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class DispatchContext:
    """What a run on another machine asks of the executor."""

    origin: str
    credential_resolver: Callable[[], CredentialResolver]
    note_worker_mount_bytes: Callable[..., None]
    timeout_message: Callable[[str | None, float], str]
    broadcast_remote_phase: Callable[..., Awaitable[None]]
    lake_config: Callable[[], Any]
    # The executor's own dict, by cell id.
    received_output_files: dict[str, Any]


def _resolve_worker_token(worker_spec: Any) -> str | None:
    """Look up the shared-secret token for an HTTP executor worker.

    In order: the env var named by ``config.token_env`` (keeps secrets out of
    notebook.toml), the literal ``config.token``, then the runtime registry
    (SSH-provisioned workers). ``None`` means the worker is reached without auth.
    """
    config = getattr(worker_spec, "config", None)
    if config is None:
        return None
    token_env = str(getattr(config, "token_env", None) or "").strip()
    if token_env:
        value = os.environ.get(token_env, "").strip()
        if value:
            return value
    literal = str(getattr(config, "token", None) or "").strip()
    if literal:
        return literal
    # A dynamically provisioned worker (SSH tunnel) keeps its token in the server process,
    # never in notebook.toml, so look it up by name in the runtime registry.
    name = str(getattr(worker_spec, "name", None) or "").strip()
    if name:
        from strata.notebook.worker_secrets import get_runtime_worker_token

        runtime_token = get_runtime_worker_token(name)
        if runtime_token:
            return runtime_token
    return None


def _add_harness_params(
    params: dict[str, Any],
    mutation_defines: list[str] | None,
    tables: dict[str, dict[str, Any]] | None,
) -> None:
    """Add ``mutation_defines`` and ``tables`` to a worker's harness params.

    Without ``mutation_defines`` the harness sees an unchanged ``id()`` for an
    in-place mutation and stores nothing. ``tables`` carries each ``@table``'s
    uri and resolved snapshot, so the worker needs no catalog. Added only when
    present, so other cells send the same bytes as before.
    """
    if mutation_defines:
        params["mutation_defines"] = list(mutation_defines)
    if tables:
        params["tables"] = tables


def _pack_folder_inputs(
    input_specs: dict[str, dict[str, Any]], output_dir: Path
) -> dict[str, dict[str, Any]]:
    """Each ``# @output_dir`` input as a tar of its rebuilt folder, the form a worker takes."""
    packed = dict(input_specs)
    for name, spec in input_specs.items():
        if spec.get("content_type") != output_dirs.CONTENT_TYPE:
            continue
        stem = str(spec["file"]).removesuffix(output_dirs.EXTENSION)
        file_name = stem + output_dirs.TAR_EXTENSION
        output_dirs.pack(output_dir / f"{stem}.dir", output_dir / file_name)
        packed[name] = {**spec, "content_type": output_dirs.TAR_CONTENT_TYPE, "file": file_name}
    return packed


def _queue_report(job: dict[str, Any]) -> tuple[int | None, str | None]:
    """A job's optional ``(position, detail)``; whatever is malformed is dropped."""
    position = job.get("position")
    if isinstance(position, bool) or not isinstance(position, int) or position < 1:
        position = None
    detail = job.get("detail")
    if not isinstance(detail, str) or not detail.strip():
        detail = None
    else:
        detail = detail.strip()[:_QUEUE_DETAIL_MAX_CHARS]
    return position, detail


async def dispatch_execution(
    host: ExecutorHost,
    routes: DispatchRoutes,
    worker_spec: Any,
    source: str,
    input_specs: dict[str, dict[str, str]],
    mount_specs: list[MountSpec],
    output_dir: Path,
    result_dir: Path,
    venv_path: Path | None,
    runtime_env: dict[str, str],
    timeout_seconds: float,
    remote_build_id: str | None = None,
    mutation_defines: list[str] | None = None,
    tables: dict[str, dict[str, Any]] | None = None,
    cell_id: str | None = None,
    cell_provenance_hash: str | None = None,
    local_run: LocalRun | None = None,
    restore: dict[str, Any] | None = None,
    resolved_mounts: dict[str, ResolvedMount] | None = None,
) -> tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]:
    """Dispatch one cell execution through the selected worker backend.

    *result_dir* is an empty server-only directory an embedded worker's result is
    unpacked into. *local_run* is the local path's execution, whose checkpoint
    events it releases; *restore* is the manifest entry naming what the cell
    starts from. *resolved_mounts* are the mounts a local or embedded run was keyed
    on, already prepared.
    """
    if worker_spec.backend == WorkerBackendType.LOCAL:
        return await routes.local(
            source,
            input_specs,
            mount_specs,
            output_dir,
            venv_path,
            runtime_env,
            timeout_seconds,
            mutation_defines=mutation_defines,
            tables=tables,
            cell_id=cell_id,
            local_run=local_run,
            restore=restore,
            resolved_mounts=resolved_mounts,
        )

    if is_embedded_executor_worker(worker_spec):
        return await routes.embedded(
            source,
            input_specs,
            mount_specs,
            output_dir,
            result_dir,
            venv_path,
            runtime_env,
            timeout_seconds,
            mutation_defines=mutation_defines,
            tables=tables,
            cell_id=cell_id,
            resolved_mounts=resolved_mounts,
        )

    if is_http_executor_worker(worker_spec):
        # Parent span for the remote side: its context travels in headers and the manifest,
        # so worker (and pool) spans join this trace.
        with trace_span(
            "notebook.dispatch",
            worker=worker_spec.name,
            build_id=remote_build_id,
            notebook_id=host.notebook_state.id,
            cell_id=cell_id,
        ):
            return await routes.http(
                worker_spec,
                source,
                input_specs,
                mount_specs,
                output_dir,
                runtime_env,
                timeout_seconds,
                remote_build_id=remote_build_id,
                mutation_defines=mutation_defines,
                tables=tables,
                cell_id=cell_id,
                cell_provenance_hash=cell_provenance_hash,
            )

    raise RuntimeError(f"Unsupported worker backend: {worker_spec.backend.value}")


async def dispatch_embedded(
    context: EmbeddedContext,
    source: str,
    input_specs: dict[str, dict[str, str]],
    mount_specs: list[MountSpec],
    output_dir: Path,
    result_dir: Path,
    venv_path: Path | None,
    runtime_env: dict[str, str],
    timeout_seconds: float,
    mutation_defines: list[str] | None = None,
    tables: dict[str, dict[str, Any]] | None = None,
    cell_id: str | None = None,
    resolved_mounts: dict[str, ResolvedMount] | None = None,
) -> tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]:
    """Run the bundle-based executor path locally for supported executor workers.

    The result is unpacked into *result_dir*, a server-only directory: output_dir
    still holds the folders' files, and the cell can redirect anything under it.
    *resolved_mounts* are prepared here when the caller has not.
    """
    if resolved_mounts is None:
        resolved_mounts = await context.prepare_mounts(mount_specs, cell_id)
    with context.run_token(timeout_seconds) as run_token:
        manifest_path = context.write_manifest(
            source,
            input_specs,
            output_dir,
            runtime_env,
            resolved_mounts,
            mutation_defines=mutation_defines,
            tables=tables,
            cell_id=cell_id,
            run_token=run_token,
        )
        result = await context.run_harness(
            manifest_path, venv_path, timeout_seconds, run_token=run_token
        )

    # A process the cell left running still owns output_dir and can swap any path under
    # it for a link: tar into an unnamed file, unpack where the cell cannot reach.
    with tempfile.TemporaryFile() as bundle:
        pack_notebook_output_bundle(bundle, result, output_dir)
        bundle.seek(0)
        unpacked_result = unpack_notebook_output_bundle(bundle, result_dir)
    return unpacked_result, result_dir, "executor", resolved_mounts


async def locked_environment(
    host: ExecutorHost, worker_spec: Any, language: str = "python"
) -> dict[str, str] | None:
    """The notebook's lock for *language*, for a worker that runs cells in it.

    Only a worker whose ``/health`` advertises ``locked_environments`` (for
    ``uv.lock``) or ``locked_r_environments`` (for ``renv.lock``) gets it; one
    that answers without it uses its own environment. A worker that cannot be
    asked is refused for a locked notebook: otherwise the result would be
    cached under the lock's hash for an environment it never ran in.
    """
    from strata.notebook.python_versions import read_requested_python_minor
    from strata.notebook.worker_env import environment_spec, r_environment_spec
    from strata.notebook.workers import worker_advertises

    if language == "r":
        spec = r_environment_spec(host.path)
        feature = "locked_r_environments"
    elif language == "python":
        spec = environment_spec(host.path, read_requested_python_minor(host.path))
        feature = "locked_environments"
    else:
        return None
    if spec is None:
        # No lock to run in, so the worker's features change nothing.
        return None
    advertised = await worker_advertises(worker_spec, feature)
    if advertised is None:
        raise RuntimeError(
            f"worker {worker_spec.name!r} could not be asked whether it runs cells in "
            f"a locked environment, and this notebook has one. Running the cell anyway "
            f"would record it as the lock's result without it having been used."
        )
    return spec if advertised else None


async def dispatch_http(
    host: ExecutorHost,
    context: DispatchContext,
    worker_spec: Any,
    source: str,
    input_specs: dict[str, dict[str, Any]],
    mount_specs: list[MountSpec],
    output_dir: Path,
    runtime_env: dict[str, str],
    timeout_seconds: float,
    remote_build_id: str | None = None,
    mutation_defines: list[str] | None = None,
    tables: dict[str, dict[str, Any]] | None = None,
    cell_id: str | None = None,
    cell_provenance_hash: str | None = None,
    language: str = "python",
) -> tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]:
    """Run a cell through an external notebook executor over HTTP."""
    from strata.notebook.workers import worker_advertises

    # Before anything runs: an older worker would not make the folder, so a long run
    # would end in a NameError with nothing kept.
    if parse_annotations(source, CellLanguage.PYTHON).output_dirs:
        advertised = await worker_advertises(worker_spec, "output_dirs")
        if advertised is None:
            raise RuntimeError(
                f"worker {worker_spec.name!r} could not be asked whether it returns "
                "# @output_dir folders"
            )
        if not advertised:
            raise RuntimeError(
                f"worker {worker_spec.name!r} does not return # @output_dir folders; "
                "upgrade strata on it"
            )
    folders = sorted(
        name
        for name, spec in input_specs.items()
        if spec.get("content_type") == output_dirs.CONTENT_TYPE
    )
    if folders:
        advertised = await worker_advertises(worker_spec, "folder_inputs")
        if advertised is None:
            raise RuntimeError(
                f"worker {worker_spec.name!r} could not be asked whether it takes "
                "# @output_dir folders as inputs"
            )
        if not advertised:
            raise RuntimeError(
                f"worker {worker_spec.name!r} cannot take {folders[0]!r}, an "
                "# @output_dir folder, as an input; upgrade strata on it"
            )
        input_specs = await anyio.to_thread.run_sync(_pack_folder_inputs, input_specs, output_dir)
    for mount in mount_specs:
        if mount.uri.startswith("file://"):
            raise RuntimeError(
                f"Remote executor workers do not support file:// mounts: '{mount.name}'"
            )
    # The worker resolves names from its own registry, so this notebook's allowance is
    # checked here, before the names leave.
    credentials = context.credential_resolver()
    for mount in mount_specs:
        if mount.credential:
            credentials.ensure_allowed(mount.credential)
        credentials.ensure_options_allowed(parse_mount_uri(mount.uri)[0], mount.options)
    ensure_writable(credentials, mount_specs)

    executor_url = str(worker_spec.config.url or "").strip()
    if not executor_url:
        raise RuntimeError(f"Executor worker '{worker_spec.name}' is missing config.url")

    worker_token = _resolve_worker_token(worker_spec)
    transport = str(worker_spec.config.transport or "direct").strip().lower()
    if transport in SIGNED_TRANSPORTS:
        signed = await dispatch_signed(
            host,
            context,
            worker_spec,
            source,
            input_specs,
            mount_specs,
            output_dir,
            runtime_env,
            timeout_seconds,
            build_id=remote_build_id,
            mutation_defines=mutation_defines,
            tables=tables,
            cell_id=cell_id,
            cell_provenance_hash=cell_provenance_hash,
            language=language,
        )
        context.note_worker_mount_bytes(cell_id, signed[0], mount_specs)
        return signed

    metadata_inputs: list[dict[str, Any]] = []
    for var_name, spec in sorted(input_specs.items()):
        entry: dict[str, Any] = {
            "name": var_name,
            "format": str(spec.get("content_type", "pickle/object")),
            "uri": None,
            "byte_size": (output_dir / str(spec["file"])).stat().st_size,
            # The on-disk filename: re-deriving ``{var_name}{ext}`` misses names that got a
            # case-safety hash suffix.
            "file": str(spec["file"]),
        }
        # A module/cell export carries injected values its defs close over; the worker needs
        # the sub-spec to hydrate them.
        if spec.get("injected"):
            entry["injected"] = spec["injected"]
        metadata_inputs.append(entry)

    metadata = {
        "protocol_version": EXECUTOR_PROTOCOL_VERSION,
        "build_id": f"notebook-{uuid.uuid4().hex[:12]}",
        "tenant": None,
        "principal": None,
        "origin": context.origin,
        # Names the cell's frames in a traceback. Beside ``params``, which feed the hash.
        "cell_id": cell_id,
        # ``str(sorted(...))`` byte format is load-bearing: cached transport hashes key off it.
        "provenance_hash": derive_subkey(source, str(sorted(input_specs))),
        "transform": {
            "ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "code_hash": compute_source_hash(source),
            "params": {
                "source": source,
                "timeout_seconds": timeout_seconds,
                "mounts": [mount.model_dump(mode="json") for mount in mount_specs],
                "env": runtime_env,
            },
        },
        "inputs": metadata_inputs,
    }
    # Only non-Python cells name their language, so Python payloads and hashes are
    # unchanged.
    if language != "python":
        metadata["transform"]["params"]["language"] = language
    # Sent only when present, so a cell with neither sends the same bytes. Without them a
    # worker would skip input rebinding and mutation recapture that a local run does, under
    # the same provenance hash.
    _add_harness_params(metadata["transform"]["params"], mutation_defines, tables)
    environment = await locked_environment(host, worker_spec, language)
    if environment is not None:
        metadata["transform"]["params"]["environment"] = environment

    files: list[tuple[str, tuple[str, Any, str]]] = [
        (
            "metadata",
            (
                "metadata.json",
                json.dumps(metadata).encode("utf-8"),
                "application/json",
            ),
        )
    ]
    input_file_handles: list[Any] = []

    def _add_file(file_name: str) -> None:
        handle = open(output_dir / file_name, "rb")
        input_file_handles.append(handle)
        files.append((file_name, (file_name, handle, "application/octet-stream")))

    timeout = max(timeout_seconds + 5.0, 30.0)
    headers = {
        EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION,
        **current_trace_context(),
    }
    if worker_token:
        headers["Authorization"] = f"Bearer {worker_token}"
    bundle_path = output_dir / "notebook-output-bundle.tar"
    build_id = str(metadata["build_id"])

    async def _receive_bundle(response: httpx.Response) -> None:
        if response.status_code == 408:
            raise RemoteExecutionError(
                context.timeout_message(cell_id, timeout_seconds),
                remote_error_code="TIMEOUT",
            )
        if response.status_code != 200:
            await response.aread()
            detail = extract_remote_error(response)
            raise RemoteExecutionError(
                f"Remote executor '{worker_spec.name}' returned {response.status_code}: {detail}",
                remote_error_code="EXECUTOR_HTTP_ERROR",
            )

        protocol = response.headers.get(EXECUTOR_PROTOCOL_HEADER)
        if protocol and protocol != EXECUTOR_PROTOCOL_VERSION:
            raise RemoteExecutionError(
                f"Remote executor '{worker_spec.name}' returned unsupported "
                f"protocol version {protocol!r}",
                remote_error_code="PROTOCOL_ERROR",
            )
        notebook_protocol = response.headers.get("X-Strata-Notebook-Executor-Protocol")
        if notebook_protocol and notebook_protocol != NOTEBOOK_EXECUTOR_PROTOCOL_VERSION:
            raise RemoteExecutionError(
                f"Remote executor '{worker_spec.name}' returned unsupported "
                f"notebook protocol version {notebook_protocol!r}",
                remote_error_code="PROTOCOL_ERROR",
            )

        with open(bundle_path, "wb") as f:
            async for chunk in response.aiter_bytes():
                f.write(chunk)

    try:
        for spec in input_specs.values():
            _add_file(str(spec["file"]))
            # Ship the injected blobs alongside the module descriptor.
            for inj in (spec.get("injected") or {}).values():
                _add_file(str(inj["file"]))

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    executor_url,
                    files=files,
                    headers=headers,
                ) as response:
                    if response.status_code == 202:
                        await response.aread()
                    else:
                        await _receive_bundle(response)
            if response.status_code == 202:
                # A dispatcher booting a machine: its wait does not count against the cell.
                finished = await await_accepted_job(
                    context,
                    response,
                    submit_url=executor_url,
                    headers=headers,
                    timeout_seconds=timeout_seconds,
                    cancel=lambda: cancel_remote_execution(executor_url, build_id, worker_token),
                    worker_spec=worker_spec,
                    cell_id=cell_id,
                    receive_bundle=_receive_bundle,
                )
                if not bundle_path.exists():
                    if finished.status_code == 200:
                        raise RemoteExecutionError(
                            f"Remote executor '{worker_spec.name}' finished the job "
                            "without answering the output bundle",
                            remote_error_code="PROTOCOL_ERROR",
                        )
                    await _receive_bundle(finished)
        except asyncio.CancelledError:
            # No build row to mark failed here, so cancelling only reclaims the machine, which
            # would otherwise finish the cell for a gone caller. Shielded: runs inside the
            # propagating cancellation.
            await asyncio.shield(cancel_remote_execution(executor_url, build_id, worker_token))
            raise
        except httpx.TimeoutException as exc:
            # The worker may never see this connection drop, and would finish the cell for nobody.
            await cancel_remote_execution(executor_url, build_id, worker_token)
            raise RemoteExecutionError(
                context.timeout_message(cell_id, timeout_seconds),
                remote_error_code="TIMEOUT",
            ) from exc
        except httpx.HTTPError as exc:
            await cancel_remote_execution(executor_url, build_id, worker_token)
            raise RemoteExecutionError(
                f"Remote executor request failed for worker '{worker_spec.name}': {exc}",
                remote_error_code="REQUEST_FAILED",
            ) from exc
    finally:
        for handle in input_file_handles:
            handle.close()

    unpacked_dir = output_dir / "_executor_result"
    unpacked_result = unpack_notebook_output_bundle(bundle_path, unpacked_dir)
    context.note_worker_mount_bytes(cell_id, unpacked_result, mount_specs)
    return unpacked_result, unpacked_dir, "executor", {}


async def dispatch_signed(
    host: ExecutorHost,
    context: DispatchContext,
    worker_spec: Any,
    source: str,
    input_specs: dict[str, dict[str, str]],
    mount_specs: list[MountSpec],
    output_dir: Path,
    runtime_env: dict[str, str],
    timeout_seconds: float,
    build_id: str | None = None,
    mutation_defines: list[str] | None = None,
    tables: dict[str, dict[str, Any]] | None = None,
    cell_id: str | None = None,
    cell_provenance_hash: str | None = None,
    language: str = "python",
) -> tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]:
    """Run a cell through the core build + signed-URL transport path."""
    from strata.auth.principal import get_principal
    from strata.server import get_state

    state = get_state()
    if not (state.config.server_transforms_enabled or state.config.writes_enabled):
        raise RuntimeError(
            "Signed notebook executor transport requires "
            "personal-mode writes or server-mode transforms to be enabled. "
            "For local testing, restart Strata with "
            "STRATA_DEPLOYMENT_MODE=personal."
        )

    artifact_dir = state.config.artifact_dir
    if artifact_dir is None:
        raise RuntimeError("Artifact store is not configured for signed notebook transport")

    artifact_store = get_artifact_store(artifact_dir)
    build_store = get_build_store(
        artifact_dir / "artifacts.sqlite",
        dialect=artifact_store.dialect if artifact_store else None,
    )
    if artifact_store is None or build_store is None:
        raise RuntimeError("Build store is not initialized")

    executor_url = str(worker_spec.config.url or "").strip()
    if not executor_url:
        raise RuntimeError(f"Executor worker '{worker_spec.name}' is missing config.url")

    worker_token = _resolve_worker_token(worker_spec)
    base_url = str(worker_spec.config.strata_url or "").strip() or state.config.server_url
    principal = get_principal()
    tenant_id = principal.tenant if principal is not None else None
    principal_id = principal.id if principal is not None else None

    build_id = build_id or f"nbbuild-{uuid.uuid4().hex[:12]}"
    artifact_id = f"nb_remote_{host.notebook_state.id}_{build_id}"
    trace_context = current_trace_context()
    artifact_version: int | None = None
    failure_recorded = False
    submitted = False

    def _mark_failed(message: str, error_code: str) -> None:
        nonlocal failure_recorded
        if failure_recorded:
            return
        failure_recorded = True
        try:
            build_store.fail_build(build_id, message, error_code)
        except Exception:
            logger.exception(
                "Failed to mark notebook build %s as failed (%s)",
                build_id,
                error_code,
            )
        if artifact_version is not None:
            try:
                artifact_store.fail_artifact(artifact_id, artifact_version, tenant=tenant_id)
            except Exception:
                logger.exception(
                    "Failed to mark notebook artifact %s@v=%s as failed",
                    artifact_id,
                    artifact_version,
                )

    async def _fail(message: str, error_code: str) -> None:
        await anyio.to_thread.run_sync(_mark_failed, message, error_code)

    async def _stop_worker() -> None:
        """Ask the worker to stop, if it was ever asked to start."""
        if submitted:
            await cancel_remote_execution(executor_url, build_id, worker_token)

    staged_input_specs, input_artifacts = await anyio.to_thread.run_sync(
        functools.partial(
            stage_signed_transport_inputs,
            host,
            artifact_store=artifact_store,
            build_id=build_id,
            input_specs=input_specs,
            output_dir=output_dir,
            tenant_id=tenant_id,
            principal_id=principal_id,
        )
    )
    input_uris = sorted(
        {str(spec["uri"]) for spec in staged_input_specs.values() if spec.get("uri")}
        | {
            str(inj["uri"])
            for spec in staged_input_specs.values()
            for inj in (spec.get("injected") or {}).values()
            if isinstance(inj, dict) and inj.get("uri")
        }
    )

    build_params = {
        "source": source,
        "timeout_seconds": timeout_seconds,
        "mounts": [mount.model_dump(mode="json") for mount in mount_specs],
        "env": runtime_env,
        "input_specs": staged_input_specs,
        "output_format": "notebook-output-bundle@v1",
    }
    if language != "python":
        build_params["language"] = language
    # As the v1 path. In ``params``, because they change what the cell computes.
    _add_harness_params(build_params, mutation_defines, tables)
    environment = await locked_environment(host, worker_spec, language)
    if environment is not None:
        build_params["environment"] = environment
    # The env holds secrets and the artifact and build rows are readable by the tenant,
    # so what is stored names the keys and digests the values; only the manifest the
    # worker receives carries them.
    recorded_params = {
        **build_params,
        "env": {
            "names": sorted(runtime_env),
            "sha256": hashlib.sha256(
                json.dumps(runtime_env, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
    }
    transport_provenance = hashlib.sha256(
        json.dumps(
            {
                "executor": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
                "executor_url": executor_url,
                "inputs": [
                    {
                        "name": name,
                        "uri": str(spec.get("uri", "")),
                        "content_type": str(spec.get("content_type", "pickle/object")),
                    }
                    for name, spec in sorted(input_specs.items())
                ],
                "params": recorded_params,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    transform_spec = ArtifactTransformSpec(
        executor=NOTEBOOK_EXECUTOR_TRANSFORM_REF,
        params=recorded_params,
        inputs=input_uris,
    )

    def _start_build() -> None:
        nonlocal artifact_version
        artifact_version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=transport_provenance,
            transform_spec=transform_spec,
            input_versions={uri: uri for uri in input_uris},
            tenant=tenant_id,
            principal=principal_id,
        )
        build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=artifact_version,
            executor_ref=NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            executor_url=executor_url,
            tenant_id=tenant_id,
            principal_id=principal_id,
            input_uris=input_uris,
            params=recorded_params,
            dispatch="external",
        )
        build_store.start_build(build_id)

    try:
        await anyio.to_thread.run_sync(_start_build)

        # Presigning can call the cloud (role credentials, IAM signBlob, a delegation key).
        manifest = (
            await asyncio.to_thread(
                state.url_signer.generate_build_manifest,
                base_url=base_url,
                build_id=build_id,
                metadata={
                    "build_id": build_id,
                    "artifact_id": artifact_id,
                    "version": artifact_version,
                    "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
                    "params": build_params,
                    # Who and what this dispatch is for, so a dispatcher can attribute the job
                    # and dedupe submissions without a GET /v1/builds. Kept out of ``params``,
                    # which feeds transport provenance: identity must not change what is
                    # cached. ``cell_provenance_hash`` is the cell's own key, not the transport
                    # hash.
                    "principal": principal_id,
                    "tenant": tenant_id,
                    "notebook_id": host.notebook_state.id,
                    "cell_id": cell_id,
                    "cell_provenance_hash": cell_provenance_hash,
                    "origin": context.origin,
                    # W3C trace context, for a worker reached without the headers (a
                    # dispatcher that forwards only the body).
                    **trace_context,
                },
                input_artifacts=input_artifacts,
                max_output_bytes=state.config.max_transform_output_bytes,
                blob_store=(
                    artifact_store.blob_store if state.config.artifact_presigned_urls else None
                ),
                version_record=artifact_store.get_artifact,
                # The worker uploads and finalizes after provisioning plus the whole
                # cell run (and 5 minutes for the upload), so the setting is a floor.
                url_expiry_seconds=max(
                    state.config.signed_url_expiry_seconds,
                    timeout_seconds + state.config.worker_provisioning_timeout_seconds + 300.0,
                ),
            )
        ).to_dict()

        manifest_execute_url = execute_manifest_url(executor_url)
        headers = {**trace_context}
        if worker_token:
            headers["Authorization"] = f"Bearer {worker_token}"
        # Before the worker can send a chunk; keyed by the session id, as the sockets are.
        relay = (
            console_relay.relaying(
                build_id,
                host.id,
                cell_id,
                shared_store=build_store if state.config.node_advertised_url else None,
            )
            if cell_id
            else nullcontext()
        )
        # Files the worker uploads while the cell runs are stored as they arrive, so a
        # worker lost later still leaves them.
        declared = frozenset(parse_annotations(source, CellLanguage.PYTHON).output_dirs)
        receiver = output_file_relay.Receiver(
            host.get_artifact_manager(),
            cell_id or "",
            upload_dir=host.path / ".strata" / "uploads",
            folders=declared,
        )
        receiving = (
            output_file_relay.receiving(build_id, receiver)
            if cell_id and declared
            else nullcontext()
        )
        context.received_output_files.pop(cell_id or "", None)
        async with relay:
            with receiving:
                try:
                    submitted = True
                    async with httpx.AsyncClient(
                        timeout=max(timeout_seconds + 10.0, 30.0)
                    ) as client:
                        response = await client.post(
                            manifest_execute_url, json=manifest, headers=headers
                        )
                    if response.status_code == 202:
                        response = await await_accepted_job(
                            context,
                            response,
                            submit_url=manifest_execute_url,
                            headers=headers,
                            timeout_seconds=timeout_seconds,
                            cancel=lambda: cancel_remote_execution(
                                executor_url, build_id, worker_token
                            ),
                            worker_spec=worker_spec,
                            cell_id=cell_id,
                        )
                finally:
                    if cell_id:
                        context.received_output_files[cell_id] = receiver.received()
    except RemoteExecutionError as exc:
        await _fail(str(exc), exc.remote_error_code or "EXECUTOR_ERROR")
        raise
    except asyncio.CancelledError:
        # Shielded: inside a cancellation an unshielded await is cancelled before the build
        # is failed or the worker hears.
        await asyncio.shield(_fail("Notebook manifest execution cancelled", "CANCELLED"))
        await asyncio.shield(_stop_worker())
        raise
    except httpx.TimeoutException as exc:
        await _fail("Notebook manifest execution timed out", "TIMEOUT")
        await _stop_worker()
        raise RemoteExecutionError(
            context.timeout_message(cell_id, timeout_seconds),
            remote_build_state="failed",
            remote_error_code="TIMEOUT",
        ) from exc
    except httpx.HTTPError as exc:
        await _fail(
            f"Remote executor request failed for worker '{worker_spec.name}': {exc}",
            "REQUEST_FAILED",
        )
        await _stop_worker()
        raise RemoteExecutionError(
            f"Remote executor request failed for worker '{worker_spec.name}': {exc}",
            remote_build_state="failed",
            remote_error_code="REQUEST_FAILED",
        ) from exc
    except Exception as exc:
        await _fail(str(exc), "SETUP_FAILED")
        await _stop_worker()
        raise RemoteExecutionError(
            f"Remote executor setup failed for worker '{worker_spec.name}': {exc}",
            remote_build_state="failed",
            remote_error_code="SETUP_FAILED",
        ) from exc

    def _executor_error(status_code: int, detail: str) -> RemoteExecutionError:
        """Fail the build for a non-200 answer, keeping the failure the build recorded."""
        build = build_store.get_build(build_id)
        inferred_error_code = (
            "FINALIZE_FAILED"
            if "Failed to finalize notebook bundle build" in detail
            else "EXECUTOR_HTTP_ERROR"
        )
        error_code = (
            build.error_code
            if build is not None and build.state == "failed" and build.error_code
            else inferred_error_code
        )
        error_message = (
            build.error_message
            if build is not None and build.state == "failed" and build.error_message
            else f"Remote executor '{worker_spec.name}' returned {status_code}: {detail}"
        )
        _mark_failed(
            error_message,
            error_code,
        )
        refreshed_build = build_store.get_build(build_id)
        return RemoteExecutionError(
            error_message,
            remote_build_state=(refreshed_build.state if refreshed_build is not None else "failed"),
            remote_error_code=error_code,
        )

    def _fetch_bundle(bundle_path: Path) -> None:
        """Copy the finalized result bundle, possibly from object storage, to a local file."""
        build = build_store.get_build(build_id)
        if build is None or build.state != "ready":
            build_error_message = build.error_message if build is not None else None
            build_error_code = build.error_code if build is not None and build.error_code else None
            raise RemoteExecutionError(
                build_error_message or f"Notebook build {build_id} did not complete successfully",
                remote_build_state=build.state if build is not None else "unknown",
                remote_error_code=build_error_code or "BUILD_FAILED",
            )

        reader_cm = artifact_store.open_blob_reader(build.artifact_id, build.version)
        if reader_cm is None:
            _mark_failed(
                f"Notebook build {build_id} completed without a stored bundle artifact",
                "MISSING_OUTPUT_BLOB",
            )
            raise RemoteExecutionError(
                f"Notebook build {build_id} completed without a stored bundle artifact",
                remote_build_state="failed",
                remote_error_code="MISSING_OUTPUT_BLOB",
            )

        with reader_cm as blob_reader, open(bundle_path, "wb") as dst:
            while True:
                chunk = blob_reader.read(BLOB_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)

    try:
        if response.status_code == 408:
            await _fail("Notebook manifest execution timed out", "TIMEOUT")
            raise RemoteExecutionError(
                context.timeout_message(cell_id, timeout_seconds),
                remote_build_state="failed",
                remote_error_code="TIMEOUT",
            )
        if response.status_code != 200:
            detail = extract_remote_error(response)
            raise await anyio.to_thread.run_sync(_executor_error, response.status_code, detail)

        bundle_path = output_dir / "notebook-output-bundle.tar"
        await anyio.to_thread.run_sync(_fetch_bundle, bundle_path)

        try:
            read_notebook_output_bundle_manifest_path(bundle_path)
        except Exception as exc:
            await _fail(
                f"Notebook build {build_id} produced an invalid output bundle: {exc}",
                "INVALID_NOTEBOOK_BUNDLE",
            )
            raise RemoteExecutionError(
                f"Notebook build {build_id} produced an invalid output bundle: {exc}",
                remote_build_state="failed",
                remote_error_code="INVALID_NOTEBOOK_BUNDLE",
            ) from exc

        unpacked_dir = output_dir / "_executor_result"
        unpacked_result = unpack_notebook_output_bundle(bundle_path, unpacked_dir)
        return unpacked_result, unpacked_dir, "executor", {}
    except asyncio.CancelledError:
        await asyncio.shield(_fail("Notebook manifest execution cancelled", "CANCELLED"))
        raise
    except RemoteExecutionError:
        raise
    except Exception as exc:
        await _fail(str(exc), "EXECUTOR_ERROR")
        raise RemoteExecutionError(
            str(exc),
            remote_build_state="failed",
            remote_error_code="EXECUTOR_ERROR",
        ) from exc


def cancel_url(executor_url: str, build_id: str) -> str:
    """Map an executor base URL to the cancel endpoint for one execution."""
    parsed = urlparse(executor_url)
    path = parsed.path or ""
    for suffix in ("/v1/execute-manifest", "/v1/notebook-execute", "/v1/execute"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    base = path.rstrip("/")
    return urlunparse(
        parsed._replace(
            path=f"{base}/v1/executions/{build_id}/cancel",
            params="",
            query="",
            fragment="",
        )
    )


async def cancel_remote_execution(
    executor_url: str, build_id: str, worker_token: str | None
) -> None:
    """Ask a worker to stop an execution we have stopped waiting for.

    The result could never land (the build is already failed), but this
    frees the machine. Every failure is logged and swallowed, since raising
    during cancellation would replace the reason the cell stopped.
    """
    headers = {"Authorization": f"Bearer {worker_token}"} if worker_token else None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(cancel_url(executor_url, build_id), headers=headers)
        if response.status_code >= 400:
            logger.info(
                "Worker refused cancel for build %s: HTTP %d",
                build_id,
                response.status_code,
            )
    except Exception as exc:
        logger.info("Could not cancel build %s on the worker: %s", build_id, exc)


async def await_accepted_job(
    context: DispatchContext,
    accepted: httpx.Response,
    *,
    submit_url: str,
    headers: dict[str, str],
    timeout_seconds: float,
    cancel: Callable[[], Awaitable[None]],
    worker_spec: Any,
    cell_id: str | None,
    receive_bundle: Callable[[httpx.Response], Awaitable[None]] | None = None,
) -> httpx.Response:
    """Follow a job a worker accepted with 202 until it finishes.

    Queueing, booting and env pulls count against
    ``worker_provisioning_timeout_seconds``; the cell's timeout starts when
    the job reports ``running``. Either deadline cancels the job, as does a
    poll that fails or cannot be read. The job URL
    answers ``{"state": ...}`` (``queued``/``provisioning``/``starting``,
    ``running``, then ``finished``/``failed`` with ``status_code`` and
    ``result`` or ``error``); that response is handled as a synchronous one.
    With ``receive_bundle`` (the direct transport), a finished job's URL
    answers the output bundle itself as ``application/x-tar``; it is
    streamed to ``receive_bundle`` and that reply returned.
    """
    try:
        body = accepted.json()
    except ValueError:
        body = {}
    job_url = body.get("job_url") if isinstance(body, dict) else None
    if not isinstance(job_url, str) or not job_url:
        await cancel()
        raise RemoteExecutionError(
            f"Remote executor '{worker_spec.name}' accepted the job without a job_url",
            remote_build_state="failed",
            remote_error_code="PROTOCOL_ERROR",
        )
    job_url = urljoin(submit_url, job_url)
    if urlsplit(job_url)[:2] != urlsplit(submit_url)[:2]:
        # The job lives on the worker the request went to; polling an absolute URL elsewhere
        # would send the worker's token to a host of its choosing. Stop the job first.
        await cancel()
        raise RemoteExecutionError(
            f"Remote executor '{worker_spec.name}' answered with a job_url on another "
            f"host ({job_url})",
            remote_build_state="failed",
            remote_error_code="PROTOCOL_ERROR",
        )
    provisioning_limit = float(
        getattr(context.lake_config(), "worker_provisioning_timeout_seconds", 600.0)
    )

    await context.broadcast_remote_phase(cell_id, worker_spec, "starting")
    started = _monotonic()
    running_since: float | None = None
    queue_shown: tuple[int | None, str | None] = (None, None)
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                async with client.stream("GET", job_url, headers=headers) as reply:
                    if (
                        receive_bundle is not None
                        and reply.status_code == 200
                        and reply.headers.get("content-type", "").startswith("application/x-tar")
                    ):
                        await receive_bundle(reply)
                        return reply
                    await reply.aread()
            except httpx.HTTPError as exc:
                # The job outlives this request; a failed poll must stop it or the machine runs
                # on for a build already marked failed.
                await cancel()
                raise RemoteExecutionError(
                    f"Remote executor '{worker_spec.name}' job status request failed: {exc}",
                    remote_build_state="failed",
                    remote_error_code="JOB_STATUS_FAILED",
                ) from exc
            if reply.status_code != 200:
                await cancel()
                raise RemoteExecutionError(
                    f"Remote executor '{worker_spec.name}' job status returned "
                    f"{reply.status_code}: {extract_remote_error(reply)}",
                    remote_build_state="failed",
                    remote_error_code="JOB_STATUS_FAILED",
                )
            try:
                job = reply.json()
                state = str(job.get("state", ""))
                status_code = int(job.get("status_code") or (200 if state == "finished" else 502))
            except (ValueError, TypeError, AttributeError) as exc:
                await cancel()
                raise RemoteExecutionError(
                    f"Remote executor '{worker_spec.name}' answered its job status with "
                    "something Strata cannot read",
                    remote_build_state="failed",
                    remote_error_code="JOB_STATUS_FAILED",
                ) from exc
            if state in ("finished", "failed"):
                content = job.get("result")
                if content is None:
                    content = {"detail": job.get("error") or f"job {state}"}
                return httpx.Response(status_code, json=content)

            now = _monotonic()
            if state == "running":
                # A running job has no place in line, but may say where it runs.
                shown = (None, _queue_report(job)[1])
                if running_since is None or shown != queue_shown:
                    if running_since is None:
                        running_since = now
                    queue_shown = shown
                    await context.broadcast_remote_phase(
                        cell_id, worker_spec, "running", queue=shown
                    )
                if now - running_since > timeout_seconds:
                    await cancel()
                    raise RemoteExecutionError(
                        context.timeout_message(cell_id, timeout_seconds),
                        remote_build_state="failed",
                        remote_error_code="TIMEOUT",
                    )
            else:
                queue = _queue_report(job)
                if queue != queue_shown:
                    queue_shown = queue
                    await context.broadcast_remote_phase(
                        cell_id, worker_spec, "starting", queue=queue
                    )
                if now - started > provisioning_limit:
                    await cancel()
                    raise RemoteExecutionError(
                        f"Remote executor '{worker_spec.name}' did not start the job within "
                        f"{provisioning_limit:g}s (last state: {state or 'unknown'}); set "
                        "STRATA_WORKER_PROVISIONING_TIMEOUT_SECONDS, or keep a machine warm",
                        remote_build_state="failed",
                        remote_error_code="PROVISIONING_TIMEOUT",
                    )
            await asyncio.sleep(_JOB_POLL_SECONDS)


async def broadcast_remote_phase(
    host: ExecutorHost,
    cell_id: str | None,
    worker_spec: Any,
    phase: str,
    *,
    queue: tuple[int | None, str | None] = (None, None),
) -> None:
    """Move the cell's badge between ``starting`` (provisioning) and ``running``.

    *queue* is the job's ``(position, detail)``, for the badge.
    Best effort: a failed broadcast must not fail the run.
    """
    if not cell_id:
        return
    try:
        from strata.notebook.protocol import MessageType
        from strata.notebook.ws import _broadcast_message, _make_message, next_notebook_sequence
        from strata.notebook.ws_payloads import cell_status_payload

        # Sockets are keyed by the session id, not the notebook.toml id.
        session_id = host.id
        await _broadcast_message(
            session_id,
            _make_message(
                MessageType.CELL_STATUS,
                next_notebook_sequence(session_id),
                cell_status_payload(
                    cell_id,
                    "running",
                    remote_worker=worker_spec.name,
                    remote_transport=worker_transport(worker_spec),
                    remote_build_state=phase,
                    remote_queue_position=queue[0],
                    remote_queue_detail=queue[1],
                ),
            ),
        )
    except Exception:
        logger.debug("remote phase broadcast failed", exc_info=True)


def execute_manifest_url(executor_url: str) -> str:
    """Map an executor base URL to the notebook manifest execution endpoint."""
    parsed = urlparse(executor_url)
    path = parsed.path or ""
    if path.endswith("/v1/execute"):
        path = path[: -len("/v1/execute")] + "/v1/execute-manifest"
    elif path.endswith("/v1/notebook-execute"):
        path = path[: -len("/v1/notebook-execute")] + "/v1/execute-manifest"
    elif not path or path == "/":
        path = "/v1/execute-manifest"
    else:
        path = f"{path.rstrip('/')}/v1/execute-manifest"
    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def stage_signed_transport_inputs(
    host: ExecutorHost,
    *,
    artifact_store: Any,
    build_id: str,
    input_specs: dict[str, dict[str, Any]],
    output_dir: Path,
    tenant_id: str | None,
    principal_id: str | None,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, int]]]:
    """Stage notebook upstream blobs into the service artifact store for signed transport."""
    staged_specs: dict[str, dict[str, Any]] = {}
    input_artifacts: list[tuple[str, int]] = []

    def _stage_blob(var_name: str, file_name: str, content_type: str, source_uri: str) -> str:
        """Stage one local blob as an artifact; return its ``strata://`` URI."""
        input_path = output_dir / file_name
        if not input_path.exists():
            raise RuntimeError(
                f"Signed notebook executor transport could not find input file {file_name!r}"
            )
        # Hashed in chunks and published from the path: a folder's tar can be gigabytes.
        byte_hash = hashlib.sha256()
        with open(input_path, "rb") as f:
            while chunk := f.read(1 << 20):
                byte_hash.update(chunk)
        byte_size = input_path.stat().st_size
        source_token = source_uri or f"local:{file_name}"
        # The tenant is in the id: the store refuses an id another tenant holds, and two
        # tenants' notebooks can share a notebook id and input URIs.
        tenant_source = f"{tenant_id or ''}\0{source_token}"
        source_hash = hashlib.sha256(tenant_source.encode()).hexdigest()[:16]
        artifact_id = f"nb_remote_input_{host.notebook_state.id}_{source_hash}_{var_name}"
        provenance_hash = hashlib.sha256(
            json.dumps(
                {
                    "source": source_token,
                    "content_type": content_type,
                    "byte_hash": byte_hash.hexdigest(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        transform_spec = ArtifactTransformSpec(
            executor="notebook_input_stage@v1",
            params={
                "content_type": content_type,
                "source_uri": source_uri,
                "build_id": build_id,
            },
            inputs=[source_uri] if source_uri else [],
        )
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=provenance_hash,
            transform_spec=transform_spec,
            input_versions={source_uri: source_uri} if source_uri else {},
            tenant=tenant_id,
            principal=principal_id,
        )
        artifact_store.publish_blob_from_path(artifact_id, version, input_path)
        finalized = artifact_store.finalize_artifact(
            artifact_id,
            version,
            schema_json=json.dumps({"content_type": content_type}),
            row_count=None,
            byte_size=byte_size,
            content_sha256=byte_hash.hexdigest(),
        )
        if finalized is None:
            raise RuntimeError(f"Failed to finalize staged notebook input artifact for {var_name}")
        input_artifacts.append((finalized.id, finalized.version))
        return f"strata://artifact/{finalized.id}@v={finalized.version}"

    for var_name, spec in sorted(input_specs.items()):
        file_name = str(spec.get("file", "")).strip()
        if not file_name:
            raise RuntimeError(
                f"Signed notebook executor transport is missing a local input file for {var_name}"
            )
        content_type = str(spec.get("content_type", "pickle/object"))
        source_uri = str(spec.get("uri", "")).strip()
        staged_uri = _stage_blob(var_name, file_name, content_type, source_uri)
        staged_specs[var_name] = {"uri": staged_uri, "content_type": content_type}
        if content_type == ContentType.FILE_PATH:
            # Otherwise the worker names it ``<var>.bin``, and a cell may go by the extension.
            staged_specs[var_name]["file"] = file_name

        # Stage each injected value so the worker can fetch it by signed URL and hydrate the
        # module.
        injected = spec.get("injected")
        if isinstance(injected, dict):
            staged_injected: dict[str, dict[str, str]] = {}
            for inj_name, inj_spec in injected.items():
                inj_ct = str(inj_spec.get("content_type", "pickle/object"))
                inj_uri = _stage_blob(
                    f"{var_name}__inj__{inj_name}", str(inj_spec["file"]), inj_ct, ""
                )
                staged_injected[inj_name] = {"uri": inj_uri, "content_type": inj_ct}
            staged_specs[var_name]["injected"] = staged_injected

    return staged_specs, input_artifacts


def extract_remote_error(response: httpx.Response) -> str:
    """Extract the most useful error message from a remote executor response."""
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or "Unknown remote executor error"

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        error = payload.get("error")
        if isinstance(error, str) and error:
            return error
    return "Unknown remote executor error"
