"""Run All's batch path: a sequence of cells in one harness subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from strata.notebook.cell_paths.context import (
    _ARTIFACT_EXT_BY_CONTENT_TYPE,
    STORE_FAILED,
    OutputNotSaved,
    cell_timeout_message,
)
from strata.notebook.cell_paths.context import artifact_content_type as _artifact_content_type
from strata.notebook.cell_paths.context import no_interpreter_message as _no_interpreter_message
from strata.notebook.cell_paths.module_export import write_module_export_outputs
from strata.notebook.env import moved_input
from strata.notebook.execution_result import hit_console
from strata.notebook.harness_user import (
    LocalExecutionRefused,
    UnsafeRunFile,
    hand_over,
    identity_env,
    resolve_harness_user,
    spawn_kwargs,
    write_run_file,
)
from strata.notebook.immutability import MutationWarning
from strata.notebook.lifeline import lifeline_handoff
from strata.notebook.models import CellOutput
from strata.notebook.mounts import moved_key, moved_line, moved_parts
from strata.notebook.process_tree import (
    SUBPROCESS_LINE_LIMIT,
    subprocess_kwargs_for_new_group,
    terminate_subprocess_tree,
)
from strata.notebook.provenance import safe_filename_stem
from strata.notebook.read_audit import ReadFinding
from strata.notebook.run_scope import RunScope
from strata.notebook.table_writes import TableWriteError
from strata.notebook.team_store import TeamPull

if TYPE_CHECKING:
    from strata.notebook.executor import _CellProvenance
    from strata.notebook.executor_host import ExecutorHost

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BatchCellResult:
    """Per-cell outcome inside a batched run-all execution."""

    cell_id: str
    status: str  # "ok" | "cache_hit" | "cell_error" | "persist_failed" | "not_run"
    error: str | None = None
    traceback: str | None = None
    traceback_full: str | None = None
    stdout: str = ""
    stderr: str = ""
    # Enough payload for ws._execute_run_all to fan out cell_output frames without going
    # back to the harness or the store.
    outputs: dict[str, Any] = field(default_factory=dict)
    display_outputs: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False
    mutation_warnings: list[MutationWarning] = field(default_factory=list)
    read_audit: list[ReadFinding] = field(default_factory=list)
    # None when the cell's time is unknown (it never started, or was killed): not 0,
    # which would price its later cache hits at nothing.
    duration_ms: float | None = None
    # Set on a hit the team store served; ws prices the hit from it.
    team_pull: TeamPull | None = None
    replaces_console: bool = False
    hit_run_ms: int = 0


@dataclass(kw_only=True)
class BatchExecutionResult:
    """Outcome of one ``CellExecutor.execute_batch``: which cells still need single-cell runs."""

    cell_results: list[BatchCellResult]
    completed: bool  # True if batch_end with reason=complete
    failed_cell_id: str | None = None
    end_reason: str = "complete"  # "complete" | "cell_error" | "persist_failed" | "subprocess_died"


@dataclass(frozen=True)
class BatchContext:
    """What a batch's run asks of the executor, and nothing else."""

    harness_path: Path
    harness_env: Callable[..., dict[str, str] | None]
    harness_fields: Callable[..., dict[str, Any]]
    run_token: Callable[..., Any]
    compute_provenance: Callable[..., Awaitable[_CellProvenance]]
    canonical_output: Callable[..., Any]
    input_refs: Callable[[str], dict[str, str]]
    moved: Callable[[str], tuple[dict[str, int | None], dict[str, int]]]
    promote_reverted_outputs: Callable[..., int | None]
    restore_gone_tables: Callable[[str, str], Awaitable[str | None]]
    promote_reverted_leaf: Callable[..., Any]
    pull_from_team_store: Callable[..., Awaitable[Any]]
    push_to_team_store: Callable[..., Awaitable[None]]
    store_outputs: Callable[..., bool]
    store_console_outputs: Callable[..., Any]
    store_display_outputs: Callable[..., Any]
    write_input_file: Callable[..., bool]


async def _drain_stream(stream: asyncio.StreamReader) -> None:
    """Read and discard ``stream`` until EOF, so chatty native code cannot fill the pipe."""
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return


def _refused_batch(cell_specs: list[dict[str, Any]], error: str) -> BatchExecutionResult:
    """A batch result for cells this session would not start, each with *error*."""
    return BatchExecutionResult(
        cell_results=[
            BatchCellResult(cell_id=spec["cell_id"], status="cell_error", error=error)
            for spec in cell_specs
        ],
        completed=False,
        failed_cell_id=cell_specs[0]["cell_id"] if cell_specs else None,
        end_reason="cell_error",
    )


async def run_batch(
    host: ExecutorHost,
    context: BatchContext,
    cell_specs: list[dict[str, Any]],
    *,
    use_cache: bool,
    batch_timeout_seconds: float,
    cell_timeout_seconds: float,
    on_cell_event: Callable[[BatchCellResult], Awaitable[None]] | None = None,
    on_cell_start: Callable[[str], Awaitable[None]] | None = None,
    provenance: Mapping[str, _CellProvenance] | None = None,
) -> BatchExecutionResult:
    """Run a sequence of cells in one harness subprocess (see ``execute_batch``).

    Per-cell stdout/stderr attribution is not implemented.
    """
    # As in single-cell: no interpreter, no run.
    venv_python = host.venv_python
    if venv_python is None:
        return _refused_batch(cell_specs, _no_interpreter_message(host))

    try:
        harness_user = resolve_harness_user()
    except LocalExecutionRefused as exc:
        # The run-all dispatcher doesn't batch on a refusing host (single-cell still serves
        # cache hits), so only a direct batch call gets here.
        return _refused_batch(cell_specs, str(exc))

    batch_tmpdir = Path(
        tempfile.mkdtemp(
            prefix=f"strata_batch_{uuid.uuid4().hex[:8]}_",
            dir=str(host.path / ".strata"),
        )
    )
    manifest_path = batch_tmpdir / "batch_manifest.json"

    # Pipe pairs: frame_r/w (harness -> parent), resp_r/w (parent -> harness).
    frame_r, frame_w = os.pipe()
    resp_r, resp_w = os.pipe()

    cell_results: dict[str, BatchCellResult] = {}
    for spec in cell_specs:
        cell_results[spec["cell_id"]] = BatchCellResult(
            cell_id=spec["cell_id"],
            status="not_run",
        )

    completed = False
    failed_cell_id: str | None = None

    with ExitStack() as token_scope:
        try:
            # Leaving the scope by any exception, cancellation included, stops the
            # harness's process tree and its drains before the files below go: none may
            # leave the user's code running, unreaped.
            async with RunScope("Batch execution") as scope:
                upstream_inputs = batch_resolve_upstream_inputs(
                    host, context, cell_specs, batch_tmpdir
                )
                # One token for the batch: Run All acts as whoever started it, and each
                # cell's scans count under it by cell id. Minted after the inputs are
                # staged, since the batch's timeout runs from the spawn.
                run_token = token_scope.enter_context(context.run_token(batch_timeout_seconds))
                manifest = {
                    "cells": cell_specs,
                    "upstream_inputs": upstream_inputs,
                    **context.harness_fields(),
                }
                manifest["run_token_on_stdin"] = True
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                lifeline_fds, lifeline_vars = lifeline_handoff()
                env = identity_env(
                    context.harness_env(
                        {
                            "STRATA_BATCH_FRAME_FD": str(frame_w),
                            "STRATA_BATCH_RESP_FD": str(resp_r),
                            "STRATA_BATCH_OUTPUT_DIR": str(batch_tmpdir),
                            **lifeline_vars,
                        }
                    ),
                    harness_user,
                )
                hand_over(batch_tmpdir, harness_user)

                # New process group so a timeout SIGKILL reaches every descendant. Mirrors
                # _run_harness.
                proc = scope.own(
                    await asyncio.create_subprocess_exec(
                        str(venv_python),
                        str(context.harness_path),
                        "--batch",
                        str(manifest_path),
                        env=env,
                        pass_fds=(frame_w, resp_r, *lifeline_fds),
                        stdin=asyncio.subprocess.PIPE if run_token is not None else None,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=str(host.path),
                        **spawn_kwargs(harness_user),
                        **subprocess_kwargs_for_new_group(),
                    )
                )

                if run_token is not None:
                    assert proc.stdin is not None
                    proc.stdin.write(f"{run_token}\n".encode())
                    proc.stdin.close()

                # Close the parent's copies of harness-side fds so EOF detection works on exit.
                os.close(frame_w)
                os.close(resp_r)
                frame_w = -1  # mark already-closed for finally
                resp_r = -1

                # Drain stdout/stderr concurrently to prevent pipe-buffer deadlock (no per-cell
                # attribution).
                assert proc.stdout is not None
                assert proc.stderr is not None
                stdout_task = scope.start(_drain_stream(proc.stdout))
                stderr_task = scope.start(_drain_stream(proc.stderr))

                # The default 64 KiB limit would make readline() raise on any frame embedding a big
                # stdout capture or display payload, aborting the batch.
                loop = asyncio.get_running_loop()
                frame_reader = asyncio.StreamReader(limit=SUBPROCESS_LINE_LIMIT, loop=loop)
                frame_protocol = asyncio.StreamReaderProtocol(frame_reader, loop=loop)
                frame_file = os.fdopen(frame_r, "rb")
                frame_r = -1  # ownership transferred to file
                await loop.connect_read_pipe(lambda: frame_protocol, frame_file)

                # Sync os.write: the kernel pipe buffer absorbs our small JSON responses.
                resp_w_fd = resp_w
                resp_w = -1  # ownership transferred

                def send_response(payload: dict) -> None:
                    line = (json.dumps(payload) + "\n").encode("utf-8")
                    os.write(resp_w_fd, line)

                # Per-cell watchdog, integrated into the service loop's readline via wait_for (no
                # separate task). On timeout the loop records the cell and SIGKILLs the harness;
                # readline then returns "" and the loop exits. Cells with explicit timeouts aren't
                # batchable, so the default applies to every cell.
                watchdog_state: dict[str, Any] = {
                    "active_cell_id": None,
                    "active_started_at": None,
                    "active_timeout": cell_timeout_seconds,
                    "timed_out": False,
                    "timeout_cell_id": None,
                    "proc": proc,
                }

                try:
                    end_reason, failed_cell_id = await asyncio.wait_for(
                        batch_service_loop(
                            host,
                            context,
                            frame_reader,
                            send_response,
                            cell_results,
                            batch_tmpdir,
                            # Sources frozen at partition time: ``cell.source`` can change under a
                            # running batch, and hashing it would file outputs under the edit's key.
                            executed_sources={
                                str(spec.get("cell_id", "")): str(spec.get("source", ""))
                                for spec in cell_specs
                            },
                            provenance=dict(provenance or {}),
                            use_cache=use_cache,
                            on_cell_event=on_cell_event,
                            on_cell_start=on_cell_start,
                            watchdog_state=watchdog_state,
                        ),
                        timeout=batch_timeout_seconds,
                    )
                    completed = end_reason == "complete"
                except TimeoutError:
                    # SIGTERM the whole tree; proc.kill() alone would leak user-spawned children.
                    await terminate_subprocess_tree(proc)
                    end_reason = "subprocess_died"
                    completed = False
                finally:
                    # Close the response fd so a still-live harness sees EOF.
                    try:
                        os.close(resp_w_fd)
                    except OSError:
                        pass

                # A watchdog kill ends the loop as "subprocess_died"; override with the timeout
                # details so the dispatcher sees which cell hung.
                if watchdog_state["timed_out"] and watchdog_state["timeout_cell_id"]:
                    timed_out_id = watchdog_state["timeout_cell_id"]
                    timed_out_result = BatchCellResult(
                        cell_id=timed_out_id,
                        status="cell_error",
                        error=cell_timeout_message(watchdog_state["active_timeout"]),
                    )
                    cell_results[timed_out_id] = timed_out_result
                    if on_cell_event is not None:
                        await on_cell_event(timed_out_result)
                    end_reason = "cell_timeout"
                    failed_cell_id = timed_out_id
                    completed = False

                await proc.wait()
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        finally:
            for fd in (frame_w, resp_r, frame_r, resp_w):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            shutil.rmtree(batch_tmpdir, ignore_errors=True)

    return BatchExecutionResult(
        cell_results=list(cell_results.values()),
        completed=completed,
        failed_cell_id=failed_cell_id,
        end_reason=end_reason,
    )


async def batch_service_loop(
    host: ExecutorHost,
    context: BatchContext,
    frame_reader: asyncio.StreamReader,
    send_response: Callable[[dict], None],
    cell_results: dict[str, BatchCellResult],
    batch_tmpdir: Path,
    *,
    executed_sources: dict[str, str],
    provenance: dict[str, _CellProvenance],
    use_cache: bool,
    on_cell_event: Callable[[BatchCellResult], Awaitable[None]] | None = None,
    on_cell_start: Callable[[str], Awaitable[None]] | None = None,
    watchdog_state: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Read frames and service requests until ``batch_end``.

    Returns ``(end_reason, failed_cell_id)``. ``on_cell_event`` is awaited
    after each ``BatchCellResult`` is recorded, ``on_cell_start`` on each
    ``cell_start``.
    """
    active_cell_id: str | None = None
    end_reason = "subprocess_died"
    failed_cell_id: str | None = None
    team_pulls: dict[str, TeamPull] = {}
    # Parent-side fields of a hit's console record, by cell: what the harness does not echo.
    hit_consoles: dict[str, dict[str, Any]] = {}

    async def _record(cell_id: str, result: BatchCellResult) -> None:
        cell_results[cell_id] = result
        if on_cell_event is not None:
            await on_cell_event(result)

    async def _serviced[T](request: Awaitable[T]) -> T:
        """Await the parent's answer to a harness request off the cell's clock.

        The time is the parent's (cache lookup, store write), so the window
        start moves forward by it; otherwise a slow cache check under load
        times out a cell whose code has not started.
        """
        began = time.time()
        try:
            return await request
        finally:
            if watchdog_state is not None and watchdog_state.get("active_started_at"):
                watchdog_state["active_started_at"] += time.time() - began

    async def _abort_oversized_frame() -> None:
        # A frame exceeded even SUBPROCESS_LINE_LIMIT. Terminate the harness tree (or it
        # leaks) and let the loop exit as subprocess_died.
        logger.error(
            "Batch harness emitted a frame over the %d-byte line limit; aborting the batch",
            SUBPROCESS_LINE_LIMIT,
        )
        if watchdog_state is not None:
            try:
                await terminate_subprocess_tree(watchdog_state["proc"])
            except Exception:
                logger.exception("Failed terminating batch harness after oversized frame")

    while True:
        # Cap readline at the active cell's remaining timeout; on expiry, SIGKILL the
        # harness, record cell_error and exit.
        if watchdog_state is not None and watchdog_state.get("active_started_at"):
            # Set whenever active_started_at is; cast for the type checker.
            started = cast(float, watchdog_state["active_started_at"])
            timeout = cast(float, watchdog_state["active_timeout"])
            remaining = timeout - (time.time() - started)
            if remaining <= 0:
                watchdog_state["timed_out"] = True
                watchdog_state["timeout_cell_id"] = watchdog_state["active_cell_id"]
                try:
                    await terminate_subprocess_tree(watchdog_state["proc"])
                except Exception:
                    logger.exception("Failed terminating batch harness after a timeout")
                break
            try:
                line = await asyncio.wait_for(frame_reader.readline(), timeout=remaining)
            except TimeoutError:
                watchdog_state["timed_out"] = True
                watchdog_state["timeout_cell_id"] = watchdog_state["active_cell_id"]
                try:
                    await terminate_subprocess_tree(watchdog_state["proc"])
                except Exception:
                    logger.exception("Failed terminating batch harness after a timeout")
                break
            except ValueError:
                await _abort_oversized_frame()
                break
        else:
            try:
                line = await frame_reader.readline()
            except ValueError:
                await _abort_oversized_frame()
                break
        if not line:
            # Pipe closed without batch_end: the subprocess died.
            break
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Batch harness emitted unparseable frame: %r", line[:200])
            continue

        ftype = frame.get("type")
        payload = frame.get("payload") or {}

        if ftype == "cell_start":
            active_cell_id = payload.get("cell_id")
            # Before the watchdog window opens: the announcement is the parent's time.
            if on_cell_start is not None and active_cell_id:
                await on_cell_start(active_cell_id)
            # Watchdog window: from cell_start until a completion frame (cache_hit output,
            # cell_error, or persist).
            if watchdog_state is not None and active_cell_id:
                watchdog_state["active_cell_id"] = active_cell_id
                watchdog_state["active_started_at"] = time.time()
        elif ftype == "cache_check":
            response = await _serviced(
                batch_service_cache_check(
                    host,
                    context,
                    payload.get("cell_id", ""),
                    batch_tmpdir,
                    executed_sources=executed_sources,
                    provenance=provenance,
                    use_cache=use_cache,
                )
            )
            # Parent-side only: the harness has no use for it and it isn't JSON.
            team_pull = response.pop("team_pull", None)
            if team_pull is not None:
                team_pulls[payload.get("cell_id", "")] = team_pull
            hit_consoles[payload.get("cell_id", "")] = {
                "replaces_console": response.pop("replaces_console", False),
                "hit_run_ms": response.pop("hit_run_ms", 0),
                "read_audit": response.pop("read_audit", []),
                "mutation_warnings": response.pop("mutation_warnings", []),
            }
            # Answered once here instead of at each cache-check return; the cell's ambient
            # client needs it to name what it promotes.
            response["input_uris"] = upstream_artifact_uris(
                host, context, payload.get("cell_id", "")
            )
            send_response(response)
        elif ftype == "persist":
            response = await _serviced(
                batch_service_persist(
                    host,
                    context,
                    payload,
                    batch_tmpdir,
                    executed_sources=executed_sources,
                    provenance=provenance,
                )
            )
            cell_id_pl = payload["cell_id"]
            if response.get("ok"):
                # What was stored, as single-cell does: the harness's metadata names files with no
                # artifact behind them, and a failed display store is an empty list here.
                display_outputs = response["display_outputs"]
                await _record(
                    cell_id_pl,
                    BatchCellResult(
                        cell_id=cell_id_pl,
                        status="ok",
                        stdout=payload.get("stdout", ""),
                        stderr=payload.get("stderr", ""),
                        outputs=payload.get("outputs") or {},
                        display_outputs=display_outputs,
                        mutation_warnings=payload.get("mutation_warnings") or [],
                        read_audit=payload.get("read_audit") or [],
                        duration_ms=payload.get("duration_ms"),
                    ),
                )
            else:
                # Persist rejected: the harness sees persist_err and ends with
                # reason="persist_failed". Record it so the dispatcher doesn't report "not_run".
                await _record(
                    cell_id_pl,
                    BatchCellResult(
                        cell_id=cell_id_pl,
                        status="persist_failed",
                        error=response.get("error"),
                        stdout=payload.get("stdout", ""),
                        stderr=payload.get("stderr", ""),
                        mutation_warnings=payload.get("mutation_warnings") or [],
                        read_audit=payload.get("read_audit") or [],
                        duration_ms=payload.get("duration_ms"),
                    ),
                )
            send_response(response)
            # Close the watchdog window so idle time between cells doesn't trip it.
            if watchdog_state is not None:
                watchdog_state["active_cell_id"] = None
                watchdog_state["active_started_at"] = None
        elif ftype == "cell_output":
            if payload.get("cache_hit") and active_cell_id:
                await _record(
                    active_cell_id,
                    BatchCellResult(
                        cell_id=active_cell_id,
                        status="cache_hit",
                        cache_hit=True,
                        stdout=payload.get("stdout", ""),
                        stderr=payload.get("stderr", ""),
                        outputs=payload.get("outputs") or {},
                        display_outputs=payload.get("display_outputs") or [],
                        duration_ms=payload.get("duration_ms"),
                        team_pull=team_pulls.get(active_cell_id),
                        **hit_consoles[active_cell_id],
                    ),
                )
            if watchdog_state is not None:
                watchdog_state["active_cell_id"] = None
                watchdog_state["active_started_at"] = None
        elif ftype == "cell_error":
            cell_id = payload.get("cell_id", "")
            await _record(
                cell_id,
                BatchCellResult(
                    cell_id=cell_id,
                    status="cell_error",
                    error=payload.get("error"),
                    traceback=payload.get("traceback"),
                    traceback_full=payload.get("traceback_full"),
                    stdout=payload.get("stdout", ""),
                    stderr=payload.get("stderr", ""),
                    duration_ms=payload.get("duration_ms"),
                ),
            )
            if watchdog_state is not None:
                watchdog_state["active_cell_id"] = None
                watchdog_state["active_started_at"] = None
        elif ftype == "batch_end":
            end_reason = payload.get("reason", "complete")
            failed_cell_id = payload.get("failed_cell_id")
            break

    return end_reason, failed_cell_id


def batch_resolve_upstream_inputs(
    host: ExecutorHost,
    context: BatchContext,
    cell_specs: list[dict[str, Any]],
    batch_tmpdir: Path,
) -> dict[str, dict[str, str]]:
    """Materialize artifacts of cells upstream of the batch into batch_tmpdir.

    The harness seeds its namespace from these. Variables produced inside
    the batch must not be included.
    """
    batch_cell_ids = {spec["cell_id"] for spec in cell_specs}
    inputs: dict[str, dict[str, str]] = {}
    upstream_dir = batch_tmpdir / "__upstream__"
    upstream_dir.mkdir(parents=True, exist_ok=True)

    for spec in cell_specs:
        cell_id = spec["cell_id"]
        cell = host.notebook_state.get_cell(cell_id)
        if cell is None:
            continue
        for upstream_id in cell.upstream_ids:
            if upstream_id in batch_cell_ids:
                continue
            upstream_cell = host.notebook_state.get_cell(upstream_id)
            if upstream_cell is None:
                continue
            wired = host.wired_variables(cell_id, upstream_id)
            for var_name, uri in upstream_cell.artifact_uris.items():
                if var_name in inputs or var_name not in wired:
                    continue
                spec_dict = materialize_artifact_to_dir(host, context, uri, upstream_dir, var_name)
                if spec_dict is not None:
                    inputs[var_name] = spec_dict

    # Paths are relative to batch_tmpdir (the harness deserializes with
    # output_dir=batch_tmpdir).
    return {
        name: {
            "content_type": spec["content_type"],
            "file": f"__upstream__/{spec['file']}",
        }
        for name, spec in inputs.items()
    }


def materialize_artifact_to_dir(
    host: ExecutorHost,
    context: BatchContext,
    uri: str,
    target_dir: Path,
    var_name: str,
) -> dict[str, str] | None:
    """Write an artifact's blob to ``target_dir/<var><ext>``; return the spec or None."""
    artifact_id, version = host.parse_artifact_uri(uri)
    artifact_mgr = host.get_artifact_manager()
    art = artifact_mgr.artifact_store.get_artifact(artifact_id, version)
    if art is None:
        return None
    content_type = _artifact_content_type(art)
    ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".bin")
    file_name = f"{safe_filename_stem(var_name)}{ext}"
    if not context.write_input_file(target_dir / file_name, artifact_id, version, content_type):
        return None
    return {"content_type": content_type, "file": file_name}


def upstream_artifact_uris(
    host: ExecutorHost, context: BatchContext, cell_id: str
) -> dict[str, dict[str, str]]:
    """The inputs this cell reads, as the manifest's ``{variable: {"uri": ...}}``.

    ``strata.promote`` needs inputs named as the cell names them. A batch has
    no per-cell manifest, but in-batch upstreams have persisted by the time a
    cell checks its cache, so the map is built here.
    """
    cell = host.notebook_state.get_cell(cell_id)
    if cell is None:
        return {}
    inputs: dict[str, dict[str, str]] = {}
    for upstream_id in cell.upstream_ids:
        upstream = host.notebook_state.get_cell(upstream_id)
        if upstream is None:
            continue
        wired = host.wired_variables(cell_id, upstream_id)
        for var_name, uri in upstream.artifact_uris.items():
            if var_name in wired:
                inputs.setdefault(var_name, {"uri": uri})
    return inputs


def materialize_batch_cache_hit(
    host: ExecutorHost,
    context: BatchContext,
    cell: Any,
    cell_id: str,
    provenance_hash: str,
    consumed_vars: set[str] | list[str],
    cell_output_dir: Path,
) -> dict[str, dict[str, str]] | None:
    """Write every consumed variable's cached bytes where the harness reads.

    ``None`` as soon as one is missing or does not match this cell's hash: a
    partial hit is a miss. Separate so it can be retried after a team pull.
    """
    artifact_mgr = host.get_artifact_manager()
    cached_outputs: dict[str, dict[str, str]] = {}
    for var_name in consumed_vars:
        canonical_art = context.canonical_output(cell_id, var_name, provenance_hash)
        if canonical_art is None:
            return None
        blob = artifact_mgr.artifact_store.read_blob(canonical_art.id, canonical_art.version)
        if blob is None:
            return None
        content_type = _artifact_content_type(canonical_art)
        ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".bin")
        file_name = f"{safe_filename_stem(var_name)}{ext}"
        write_run_file(cell_output_dir, file_name, blob)
        cached_outputs[var_name] = {
            "content_type": content_type,
            "file": file_name,
        }
        # Populate artifact_uris, as single-cell does, so later batch cells key on them.
        uri = f"strata://artifact/{canonical_art.id}@v={canonical_art.version}"
        cell.artifact_uris[var_name] = uri
        cell.artifact_uri = uri
    return cached_outputs


async def batch_service_cache_check(
    host: ExecutorHost,
    context: BatchContext,
    cell_id: str,
    batch_tmpdir: Path,
    *,
    executed_sources: dict[str, str],
    provenance: dict[str, _CellProvenance],
    use_cache: bool,
) -> dict[str, Any]:
    """Service a ``cache_check`` request from the batch harness.

    On a hit, writes cached blobs to ``batch_tmpdir/<cell_id>/{var}{ext}`` and
    sets ``cell.artifact_uris`` and ``display_outputs`` as single-cell does. The
    provenance checked replaces the cell's entry in *provenance*, for its persist.
    """
    cell = host.notebook_state.get_cell(cell_id)
    if cell is None:
        return {"cache_hit": False, "provenance_hash": ""}

    # The partition's source, not the session's: a mid-batch edit must not decide whether
    # this run is a hit.
    source = executed_sources.get(cell_id, cell.source)
    base = provenance.get(cell_id)
    try:
        # Over the inputs as they are now: the cells before it in the batch persisted.
        prov = (
            base.rekeyed()
            if base is not None
            else await context.compute_provenance(cell_id, source)
        )
    except Exception as exc:
        logger.warning("Batch cache_check provenance failed for %s: %s", cell_id, exc)
        return {"cache_hit": False, "provenance_hash": ""}
    provenance[cell_id] = prov

    provenance_hash = prov.provenance_hash
    # ``# @nocache`` marks an effect the artifact doesn't capture (a write, a POST, a
    # clock read); Run All must not serve it from cache either.
    if not use_cache or prov.annotations.nocache:
        return {"cache_hit": False, "provenance_hash": provenance_hash}

    cell_output_dir = batch_tmpdir / cell_id
    cell_output_dir.mkdir(parents=True, exist_ok=True)
    artifact_mgr = host.get_artifact_manager()
    consumed_vars = (
        host.dag.consumed_variables.get(cell_id, set()) if host.dag is not None else set()
    )

    # Hydrate cached displays the way single-cell does
    # (session.resolve_cached_display_outputs): rich models when every artifact exists
    # with matching provenance, else []. Their blobs then go to the per-cell tmpdir.
    existing_display_outputs = cell.display_outputs or (
        [cell.display_output] if cell.display_output is not None else []
    )
    # A revert finds its outputs under an older version, as a single run does; promoted
    # first so everything below reads them as latest.
    if consumed_vars:
        context.promote_reverted_outputs(
            cell_id,
            provenance_hash,
            consumed_vars,
            prov.annotations.table_writes,
            len(existing_display_outputs),
        )
    elif host.resolve_cached_console(cell_id, provenance_hash) is None:
        context.promote_reverted_leaf(cell_id, provenance_hash, len(existing_display_outputs))
    cached_display_models = host.resolve_cached_display_outputs(
        cell_id, provenance_hash, existing_display_outputs
    )
    cached_displays: list[dict[str, Any]] = []
    for index, cached in enumerate(cached_display_models):
        display_artifact_id = artifact_mgr.cell_artifact_id(cell_id, f"__display__{index}")
        display_art = artifact_mgr.artifact_store.get_latest_version(display_artifact_id)
        if display_art is None:
            continue  # Should not happen: the helper validated existence.
        blob = artifact_mgr.artifact_store.read_blob(display_art.id, display_art.version)
        if blob is None:
            continue
        content_type = _artifact_content_type(display_art)
        ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".bin")
        file_name = f"__display__{index}{ext}"
        write_run_file(cell_output_dir, file_name, blob)
        meta = cached.model_dump()
        meta["file"] = file_name
        cached_displays.append(meta)

    # A leaf's record is its console (empty when it printed nothing), as in single-cell;
    # without one, or any cached display, there is nothing to serve, so run it.
    cached_console = host.resolve_cached_console(cell_id, provenance_hash)
    if not consumed_vars and not cached_displays and cached_console is None:
        return {"cache_hit": False, "provenance_hash": provenance_hash}

    cached_outputs = materialize_batch_cache_hit(
        host, context, cell, cell_id, provenance_hash, consumed_vars, cell_output_dir
    )
    team_pull: TeamPull | None = None
    if cached_outputs is None and consumed_vars:
        # Team cache tier, in single-run order: only after a local miss. A pull writes each
        # variable under its canonical local id, so the same probe decides afterwards.
        team_pull = await context.pull_from_team_store(
            cell_id=cell_id,
            provenance_hash=provenance_hash,
            consumed_vars=set(consumed_vars),
            source_hash=prov.source_hash,
            source=source,
            env_hash=prov.env_hash,
            input_versions=context.input_refs(cell_id),
        )
        if team_pull is not None:
            cached_outputs = materialize_batch_cache_hit(
                host, context, cell, cell_id, provenance_hash, consumed_vars, cell_output_dir
            )
    if cached_outputs is None:
        return {"cache_hit": False, "provenance_hash": provenance_hash}
    refusal = await context.restore_gone_tables(cell_id, source)
    if refusal is not None:
        # The harness fails the cell: running it would append to a table still there.
        return {"cache_hit": False, "provenance_hash": provenance_hash, "error": refusal}

    cell.cache_hit = True
    if cached_display_models:
        cell.display_outputs = list(cached_display_models)
        cell.display_output = cached_display_models[-1]
    # Before the record below moves the cell's last key on to this one.
    console = hit_console(
        cached_console, shows_this_key=cell.last_provenance_hash == provenance_hash
    )
    # As a single-cell hit does, so the table index names the reads behind these outputs.
    host.record_successful_execution_provenance(
        cell_id,
        provenance_hash,
        prov.source_hash,
        prov.env_hash,
        table_reads=prov.table_reads,
        reads=prov.reads(*context.moved(cell_id)),
        key_parts=prov.key_parts,
    )

    return {
        "cache_hit": True,
        "provenance_hash": provenance_hash,
        "cached_outputs": cached_outputs,
        "cached_displays": cached_displays,
        "stdout": console.get("stdout", ""),
        "stderr": console.get("stderr", ""),
        "replaces_console": console["replaces_console"],
        "hit_run_ms": console.get("hit_run_ms", 0),
        "read_audit": console.get("read_audit", []),
        "mutation_warnings": console.get("mutation_warnings", []),
        "team_pull": team_pull,
    }


async def batch_service_persist(
    host: ExecutorHost,
    context: BatchContext,
    payload: dict[str, Any],
    batch_tmpdir: Path,
    *,
    executed_sources: dict[str, str],
    provenance: dict[str, _CellProvenance],
) -> dict[str, Any]:
    """Service a ``persist`` request from the batch harness.

    Uses the single-cell chain (``write_module_export_outputs``,
    ``_store_outputs``, ``_store_display_outputs``).
    """
    cell_id = payload.get("cell_id", "")
    cell = host.notebook_state.get_cell(cell_id)
    if cell is None:
        return {"ok": False, "error": f"cell {cell_id} not found"}

    cell_output_dir = batch_tmpdir / cell_id
    if not cell_output_dir.exists():
        return {"ok": False, "error": f"output dir missing for {cell_id}"}

    # Everything recorded describes the run, so read the source that ran: hashing an edit
    # made mid-batch would file outputs under the wrong key (green on reopen, wrong bytes).
    executed_source = executed_sources.get(cell_id, cell.source)
    # The key the cache check looked up: resolving again could see a table that moved
    # since the run's snapshot was pinned.
    prov = provenance.get(cell_id)
    if prov is None:
        return {"ok": False, "error": f"no cache check for {cell_id}"}

    provenance_hash = prov.provenance_hash
    source_hash = prov.source_hash
    env_hash = prov.env_hash
    input_hashes = prov.input_hashes

    # Module export writes synthetic .cell_module.json / .cell_instance.pickle for
    # top-level defs/classes; _store_outputs picks them up. An error string means a
    # consumed def/class can't be exported safely: refuse to persist and signal the
    # harness so the cell errors, as single-cell does.
    try:
        module_export_error = write_module_export_outputs(
            host,
            cell_id,
            executed_source,
            cell_output_dir,
            provenance_hash,
            {},  # outputs dict; module export reads from AST, not this
        )
    except Exception as exc:
        return {"ok": False, "error": f"module export raised: {exc}"}
    if module_export_error:
        return {"ok": False, "error": f"module export rejected: {module_export_error}"}

    # A mount or local package that moved while the cell ran: its outputs are kept for
    # this run only.
    moved = await asyncio.to_thread(moved_input, prov.prepared, prov.local_packages)
    if moved is not None:
        # The cell's result reads its console from this payload.
        payload["stderr"] = payload.get("stderr", "") + moved_line(moved)
    # Under a key no check makes: the cell stays stale and no hit replays its displays.
    kept_hash = provenance_hash if moved is None else moved_key(provenance_hash)
    try:
        stored_ok = context.store_outputs(
            cell_id,
            cell_output_dir,
            provenance_hash,
            input_hashes,
            source_hash=source_hash,
            source=executed_source,
            env_hash=env_hash,
            stochastic=prov.annotations.stochastic or moved is not None,
            run_outputs=payload.get("outputs"),
        )
    except (UnsafeRunFile, TableWriteError, OutputNotSaved) as exc:
        return {"ok": False, "error": str(exc)}

    if not stored_ok:
        return {"ok": False, "error": STORE_FAILED}

    # Carry the post-persist metadata (with artifact_uri) in the ack so the dispatcher
    # broadcasts the URI-bearing version, as single-cell does.
    display_outputs_meta = payload.get("display_outputs") or []
    persisted_displays: list[dict[str, Any]] = []
    if display_outputs_meta:
        try:
            persisted_displays = context.store_display_outputs(
                cell_id,
                cell_output_dir,
                kept_hash,
                input_hashes,
                display_outputs_meta,
                source_hash=source_hash,
                source=executed_source,
                env_hash=env_hash,
            )
        except Exception as exc:
            logger.warning("Batch display persist failed for %s: %s", cell_id, exc)

    # Stored as in single-cell. Rerun All stores it too, or the next Run All replays an
    # older run's; only ``# @nocache`` stores none.
    if not prov.annotations.nocache and moved is None:
        context.store_console_outputs(
            cell_id,
            provenance_hash,
            payload.get("stdout", ""),
            payload.get("stderr", ""),
            input_hashes,
            source_hash=source_hash,
            source=executed_source,
            env_hash=env_hash,
            display_count=len(persisted_displays),
            duration_ms=float(payload.get("duration_ms") or 0.0),
            read_audit=payload.get("read_audit") or [],
            mutation_warnings=payload.get("mutation_warnings") or [],
        )

    # Offer it to the team, as a single run does. Inert unless configured, and it
    # swallows its own failures: a shared-cache problem must not fail a finished cell.
    await context.push_to_team_store(cell_id=cell_id)

    # Record provenance + execution as single-cell does.
    try:
        host.record_successful_execution_provenance(
            cell_id,
            kept_hash,
            source_hash,
            env_hash,
            table_reads=prov.table_reads,
            reads=prov.reads(*context.moved(cell_id)),
            key_parts=prov.key_parts if moved is None else moved_parts(prov.key_parts, moved),
        )
    except Exception:
        logger.exception("Batch could not record the run of cell %s", cell_id)

    # So a later cache check finds rich CellOutput objects. The dispatcher does this too,
    # but direct execute_batch calls (tests, REPL) need it here.
    if persisted_displays:
        cell.display_outputs = [CellOutput(**d) for d in persisted_displays]
        cell.display_output = cell.display_outputs[-1]
    # Persist to runtime.json, as single-cell does, or the display is lost on reopen.
    host.persist_display_outputs(cell_id, persisted_displays or None)

    uri = cell.artifact_uri or ""
    return {"ok": True, "uri": uri, "display_outputs": persisted_displays}
