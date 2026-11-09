"""R cells: provenance, upstreams, cache check, ``Rscript`` harness or a worker, persist."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio.to_thread

from strata.notebook.cell_paths.context import (
    OutputNotSaved,
    RemoteExecutionError,
    _mount_sync_failure_result,
    _refused_result,
    _store_failure_result,
    _worker_mount_writes,
    refused_run,
)
from strata.notebook.execution_result import CellExecutionResult, hit_console
from strata.notebook.harness_user import (
    LocalExecutionRefused,
    UnsafeRunFile,
    hand_over,
    identity_env,
    read_run_file,
    resolve_harness_user,
    spawn_kwargs,
)
from strata.notebook.lifeline import lifeline_command, lifeline_handoff
from strata.notebook.models import MountMode, MountSpec, WorkerBackendType
from strata.notebook.mounts import moved_file, moved_key, moved_line, moved_parts
from strata.notebook.table_writes import TableWriteError, write_back_failed
from strata.notebook.team_store import TeamPull
from strata.notebook.workers import is_http_executor_worker, resolve_worker_spec, worker_transport
from strata.observability.tracing import trace_span

if TYPE_CHECKING:
    from strata.notebook.executor import _CellProvenance
    from strata.notebook.executor_host import ExecutorHost
    from strata.notebook.mounts import MountWrite, ResolvedMount

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RContext:
    """What an R cell's run asks of the executor, and nothing else."""

    materialize_upstreams: Callable[[str], Awaitable[None]]
    compute_provenance: Callable[[str, str], Awaitable[_CellProvenance]]
    remote_execution_metadata: Callable[..., dict[str, str]]
    find_or_pull_outputs: Callable[..., Awaitable[tuple[Any, TeamPull | None]]]
    validate_output_hit: Callable[..., tuple[Any, list[Any]]]
    promote_reverted_outputs: Callable[..., int | None]
    moved: Callable[[str], tuple[dict[str, int | None], dict[str, int]]]
    table_write_refusal: Callable[[str], str | None]
    restore_gone_tables: Callable[[str, str], Awaitable[None]]
    load_input_blobs: Callable[[str, Path], Any]
    dispatch_http_executor: Callable[..., Awaitable[tuple[Any, Path, str, Any]]]
    prepare_mounts: Callable[..., Awaitable[dict[str, ResolvedMount]]]
    write_manifest: Callable[..., Path]
    run_r_harness: Callable[[Path, float], Awaitable[dict[str, Any]]]
    parse_result: Callable[..., CellExecutionResult]
    collect_output_dirs_off_loop: Callable[..., Awaitable[Any]]
    keep_failed_output_dirs: Callable[..., None]
    store_outputs: Callable[..., bool]
    push_to_team_store: Callable[..., Awaitable[None]]
    store_result_displays: Callable[..., None]
    sync_back_mounts: Callable[..., Awaitable[list[MountWrite]]]
    store_console_outputs: Callable[..., Any]
    store_mount_writes: Callable[..., None]
    record_good_run: Callable[..., None]
    timeout_message: Callable[[str, float], str]
    # Cross-path state, the same dicts the executor holds, by cell id. A worker's uploads
    # land in the first as they arrive; a run clears the cell's entries as it starts.
    received_output_files: dict[str, Any]
    folder_listings: dict[str, Any]


async def run_r_cell(
    host: ExecutorHost,
    context: RContext,
    cell_id: str,
    source: str,
    timeout_seconds: float,
    start_time: float,
    *,
    materialize_upstreams: bool,
    use_cache: bool,
) -> CellExecutionResult:
    """R cell pipeline: provenance, upstreams, cache check, ``Rscript`` harness, persist.

    Runs locally (warm pool, then cold ``Rscript``) or on an HTTP executor
    worker. Cache, mounts and storage are shared with Python: they key off
    provenance hashes and file extensions, which the R harness produces identically.
    """
    remote_metadata: dict[str, str] = {}
    exec_result: CellExecutionResult | None = None
    # Nothing a worker uploaded for an earlier run belongs to this one.
    context.received_output_files.pop(cell_id, None)
    context.folder_listings.pop(cell_id, None)
    try:
        cell = host.notebook_state.get_cell(cell_id)

        if materialize_upstreams:
            await context.materialize_upstreams(cell_id)

        prov = await context.compute_provenance(cell_id, source)
        source_hash = prov.source_hash
        runtime_env = prov.runtime_env
        env_hash = prov.env_hash
        input_hashes = prov.input_hashes
        provenance_hash = prov.provenance_hash
        if prov.annotations.datasets:
            return refused_run(
                host,
                CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error="@dataset is not supported on R cells; read the dataset in a "
                    "Python cell upstream",
                    execution_method="error",
                ),
            )
        # harness.R binds no table names, so the cell would run without what it declared.
        if prov.annotations.tables:
            return refused_run(
                host,
                CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error="@table is not supported on R cells; read the table in a "
                    "Python cell upstream",
                    execution_method="error",
                ),
            )
        # As in Python: an unchecked fetch fails the run, and each fetched file arrives as a
        # read-only mount that harness.R binds to its name as a path string.
        if prov.fetch_error is not None:
            return refused_run(
                host,
                CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=prov.fetch_error,
                    error_code=prov.fetch_error_code,
                    execution_method="error",
                ),
            )
        worker_spec = resolve_worker_spec(host.notebook_state, prov.effective_worker)
        remote = (
            worker_spec
            if worker_spec is not None and is_http_executor_worker(worker_spec)
            else None
        )
        if (
            worker_spec is not None
            and remote is None
            and worker_spec.backend != WorkerBackendType.LOCAL
        ):
            return refused_run(
                host,
                CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=f"R cells run locally or on an executor worker; worker "
                    f"'{worker_spec.name}' is neither",
                    execution_method="error",
                ),
            )
        if remote is not None and prov.fetched:
            return refused_run(
                host,
                CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error="@fetch on an R cell is read on this machine; run the cell "
                    "locally or fetch in a Python cell upstream",
                    execution_method="error",
                ),
            )
        remote_metadata = context.remote_execution_metadata(worker_spec)
        mount_specs = [
            *prov.mount_specs,
            *(
                MountSpec(name=name, uri=path.resolve().as_uri(), mode=MountMode.READ_ONLY)
                for name, path in prov.fetched.items()
            ),
        ]

        if prov.has_rw_mount:
            use_cache = False
        if prov.annotations.nocache:
            use_cache = False

        logger.info(
            "execute_r_cell %s: source_hash=%s env_hash=%s provenance=%s",
            cell_id,
            source_hash[:12],
            env_hash[:12],
            provenance_hash[:12],
        )

        artifact_mgr = host.get_artifact_manager()
        consumed_vars = host.dag.consumed_variables.get(cell_id, set()) if host.dag else set()

        cached_artifact = None
        if cell is not None:
            current_display_outputs = cell.display_outputs or (
                [cell.display_output] if cell.display_output is not None else []
            )
        else:
            current_display_outputs = []
        # Same gate as Python: ungated replay makes ``# @nocache`` and force/rerun no-ops
        # for a leaf cell.
        cached_display_outputs = (
            host.resolve_cached_display_outputs(
                cell_id,
                provenance_hash,
                current_display_outputs,
            )
            if (cell is not None and use_cache)
            else []
        )
        team_pull: TeamPull | None = None
        if use_cache:
            cached_artifact, team_pull = await context.find_or_pull_outputs(
                cell_id,
                provenance_hash,
                consumed_vars,
                source_hash=source_hash,
                source=source,
                env_hash=env_hash,
            )

        notebook_id = host.notebook_state.id
        if use_cache and cached_artifact is not None and consumed_vars:
            cached_artifact, cached_display_outputs = context.validate_output_hit(
                cell,
                cell_id,
                provenance_hash,
                consumed_vars,
                prov.annotations.table_writes,
                cached_artifact,
                current_display_outputs,
                cached_display_outputs,
            )
        # A leaf's displays, promoted as a variable's above. An R leaf with only a console
        # is not cached, so it has nothing to promote.
        elif (
            use_cache
            and cell is not None
            and not consumed_vars
            and not cached_display_outputs
            and context.promote_reverted_outputs(
                cell_id, provenance_hash, [], [], len(current_display_outputs)
            )
        ):
            cached_display_outputs = host.resolve_cached_display_outputs(
                cell_id, provenance_hash, current_display_outputs
            )

        if cached_artifact is not None or (not consumed_vars and cached_display_outputs):
            if cached_artifact is not None:
                try:
                    await context.restore_gone_tables(cell_id, source)
                except TableWriteError as exc:
                    return refused_run(
                        host,
                        CellExecutionResult(
                            cell_id=cell_id,
                            success=False,
                            error=write_back_failed(exc),
                            execution_method="error",
                        ),
                    )
            # After the promotion above, which carries the console.
            cached_console = host.resolve_cached_console(cell_id, provenance_hash)
            duration_ms = (time.time() - start_time) * 1000
            if cell:
                cell.cache_hit = True
                cell.display_outputs = list(cached_display_outputs)
                cell.display_output = cached_display_outputs[-1] if cached_display_outputs else None
                for var_name in consumed_vars:
                    canonical_id = artifact_mgr.cell_artifact_id(cell_id, var_name)
                    canonical_art = artifact_mgr.artifact_store.get_latest_version(
                        canonical_id,
                    )
                    if canonical_art:
                        uri = f"strata://artifact/{canonical_art.id}@v={canonical_art.version}"
                        cell.artifact_uris[var_name] = uri
                        cell.artifact_uri = uri
            cached_result = CellExecutionResult(
                cell_id=cell_id,
                success=True,
                outputs={},
                **hit_console(
                    cached_console,
                    shows_this_key=(
                        cell is not None and cell.last_provenance_hash == provenance_hash
                    ),
                ),
                display_outputs=[output.model_dump() for output in cached_display_outputs],
                display_output=(
                    cached_display_outputs[-1].model_dump() if cached_display_outputs else None
                ),
                duration_ms=duration_ms,
                cache_hit=True,
                artifact_uri=(
                    f"strata://artifact/{cached_artifact.id}@v={cached_artifact.version}"
                    if cached_artifact is not None
                    else (
                        cached_display_outputs[-1].artifact_uri if cached_display_outputs else None
                    )
                ),
                execution_method="cached",
                team_cache_principal=team_pull.principal if team_pull else None,
                team_cache_build_env=team_pull.build_env if team_pull else "",
                team_cache_saved_ms=team_pull.saved_ms if team_pull else 0,
                team_cache_promotion=team_pull.promotion if team_pull else None,
                from_team_cache=team_pull is not None,
            ).apply_remote_metadata(**remote_metadata)
            host.record_successful_execution_provenance(
                cell_id,
                provenance_hash,
                source_hash,
                env_hash,
                table_reads=prov.table_reads,
                reads=prov.reads(*context.moved(cell_id)),
                key_parts=prov.key_parts,
            )
            host.apply_execution_result_metadata(cell_id, cached_result)
            return cached_result

        refusal = context.table_write_refusal(source)
        if refusal is not None:
            return refused_run(
                host,
                CellExecutionResult(
                    cell_id=cell_id, success=False, error=refusal, execution_method="error"
                ),
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Off the loop: an input folder can hold gigabytes.
            input_specs = await anyio.to_thread.run_sync(
                context.load_input_blobs, cell_id, output_dir
            )
            result_output_dir = output_dir
            if remote is not None:
                remote_build_id = (
                    f"nbbuild-{uuid.uuid4().hex[:12]}"
                    if worker_transport(remote) == "signed"
                    else None
                )
                remote_metadata = context.remote_execution_metadata(
                    remote, remote_build_id=remote_build_id
                )
                with trace_span(
                    "notebook.dispatch",
                    worker=remote.name,
                    build_id=remote_build_id,
                    notebook_id=notebook_id,
                    cell_id=cell_id,
                ):
                    (
                        result,
                        result_output_dir,
                        execution_method,
                        resolved_mounts,
                    ) = await context.dispatch_http_executor(
                        remote,
                        source,
                        input_specs,
                        mount_specs,
                        output_dir,
                        runtime_env,
                        timeout_seconds,
                        remote_build_id=remote_build_id,
                        cell_id=cell_id,
                        cell_provenance_hash=provenance_hash,
                        language="r",
                    )
                if remote_build_id and remote_metadata.get("remote_transport") == "signed":
                    remote_metadata["remote_build_state"] = "ready"
            else:
                resolved_mounts = await context.prepare_mounts(mount_specs, cell_id)
                # Keyed on the listing it reads, which the cache check's may predate.
                prov = prov.over_mounts(resolved_mounts)
                provenance_hash = prov.provenance_hash
                manifest_path = context.write_manifest(
                    source,
                    input_specs,
                    output_dir,
                    runtime_env,
                    resolved_mounts,
                )

                # Warm R pool first (prepaid Rscript startup + renv activation), cold harness as
                # fallback; mirrors _dispatch_local.
                result = None
                execution_method = "cold"
                r_pool = host.r_warm_pool
                if r_pool is not None:
                    from strata.notebook.pool import PooledCellExecutor

                    pool_result = await PooledCellExecutor.execute_with_pool(
                        r_pool,
                        manifest_path,
                        host.path,
                        timeout_seconds,
                    )
                    if pool_result is not None:
                        result = pool_result
                        execution_method = "warm"
                if result is None:
                    result = await context.run_r_harness(manifest_path, timeout_seconds)

            duration_ms = (time.time() - start_time) * 1000
            exec_result = context.parse_result(
                cell_id, result, duration_ms, execution_method
            ).apply_remote_metadata(**remote_metadata)
            collected = await context.collect_output_dirs_off_loop(
                cell_id, source, result_output_dir
            )
            if not exec_result.success:
                context.keep_failed_output_dirs(
                    cell_id, source, result_output_dir, provenance_hash, env_hash, collected
                )

            # A mount that moved while the cell ran: its outputs are kept for this run only.
            moved = (
                await asyncio.to_thread(moved_file, prov.prepared)
                if exec_result.success and prov.prepared
                else None
            )
            if moved is not None:
                exec_result = replace(
                    exec_result, stderr=exec_result.stderr + moved_line(moved), moved_input=moved
                )
            # Under a key no check makes: the cell stays stale and no hit replays its displays.
            kept_hash = provenance_hash if moved is None else moved_key(provenance_hash)

            if exec_result.success:
                store_error = None
                try:
                    stored_ok = context.store_outputs(
                        cell_id,
                        result_output_dir,
                        provenance_hash,
                        input_hashes,
                        source_hash=source_hash,
                        source=source,
                        env_hash=env_hash,
                        # Reported by whatever ran the cell; empty for a local run.
                        build_env=str(result.get("build_env") or ""),
                        hardware=result.get("hardware") or {},
                        collected=collected,
                        stochastic=moved is not None,
                        run_outputs=exec_result.outputs,
                    )
                except (UnsafeRunFile, TableWriteError, OutputNotSaved) as exc:
                    stored_ok, store_error = False, str(exc)
                if not stored_ok:
                    logger.error(
                        "R cell %s executed OK but artifact storage failed.",
                        cell_id,
                    )
                    exec_result = _store_failure_result(
                        exec_result, store_error
                    ).apply_remote_metadata(**remote_metadata)

                if stored_ok:
                    await context.push_to_team_store(cell_id=cell_id)

                if exec_result.success:
                    context.store_result_displays(
                        exec_result,
                        result_output_dir,
                        kept_hash,
                        input_hashes,
                        source_hash=source_hash,
                        source=source,
                        env_hash=env_hash,
                    )
                    # So a hit replays this run's. An R leaf with only a console isn't
                    # cached, and a record would read it as ready.
                    if (
                        not prov.has_rw_mount
                        and not prov.annotations.nocache
                        and moved is None
                        and (consumed_vars or exec_result.display_outputs)
                    ):
                        context.store_console_outputs(
                            cell_id,
                            provenance_hash,
                            exec_result.stdout,
                            exec_result.stderr,
                            input_hashes,
                            source_hash=source_hash,
                            source=source,
                            env_hash=env_hash,
                            display_count=len(exec_result.display_outputs),
                            duration_ms=exec_result.duration_ms,
                        )

                # A worker syncs its own and reports what was written in its result.
                mount_writes = _worker_mount_writes(result, mount_specs)
                if exec_result.success and resolved_mounts:
                    try:
                        mount_writes = await context.sync_back_mounts(resolved_mounts)
                    except Exception as exc:
                        logger.exception(
                            "Failed to sync-back RW mounts for R cell %s",
                            cell_id,
                        )
                        exec_result = _mount_sync_failure_result(
                            exec_result, exc
                        ).apply_remote_metadata(**remote_metadata)
                if exec_result.success:
                    context.store_mount_writes(
                        cell_id,
                        provenance_hash,
                        mount_writes,
                        source_hash=source_hash,
                        source=source,
                        env_hash=env_hash,
                    )
                    context.record_good_run(
                        cell_id,
                        kept_hash,
                        source_hash,
                        env_hash,
                        prov
                        if moved is None
                        else replace(prov, key_parts=moved_parts(prov.key_parts, moved)),
                        mount_writes,
                    )

            host.persist_display_outputs(
                cell_id,
                exec_result.display_outputs if exec_result.success else None,
            )
            host.apply_execution_result_metadata(cell_id, exec_result)
            return exec_result

    except asyncio.CancelledError:
        if context.received_output_files.get(cell_id):
            # Cancelled mid-run: keep what the worker already uploaded, as on a loss.
            context.keep_failed_output_dirs(cell_id, source, None, provenance_hash, env_hash)
        raise
    except RemoteExecutionError as e:
        duration_ms = (time.time() - start_time) * 1000
        if context.received_output_files.get(cell_id):
            # No result came back, so what the worker uploaded is all there is.
            context.keep_failed_output_dirs(cell_id, source, None, provenance_hash, env_hash)
        error_result = CellExecutionResult(
            cell_id=cell_id,
            success=False,
            duration_ms=duration_ms,
            error=str(e),
        ).apply_remote_metadata(
            **remote_metadata,
            remote_build_state=e.remote_build_state,
            remote_error_code=e.remote_error_code,
        )
        host.persist_display_outputs(cell_id, None)
        host.apply_execution_result_metadata(cell_id, error_result)
        return error_result
    except TimeoutError:
        duration_ms = (time.time() - start_time) * 1000
        if context.received_output_files.get(cell_id):
            # No result came back, so what the worker uploaded is all there is.
            context.keep_failed_output_dirs(cell_id, source, None, provenance_hash, env_hash)
        timeout_result = CellExecutionResult(
            cell_id=cell_id,
            success=False,
            duration_ms=duration_ms,
            error=context.timeout_message(cell_id, timeout_seconds),
        ).apply_remote_metadata(**remote_metadata)
        host.persist_display_outputs(cell_id, None)
        host.apply_execution_result_metadata(cell_id, timeout_result)
        return timeout_result
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        if context.received_output_files.get(cell_id):
            # No result came back, so what the worker uploaded is all there is.
            context.keep_failed_output_dirs(cell_id, source, None, provenance_hash, env_hash)
        # A step after the run failed: what the run printed still stands.
        error_result = CellExecutionResult(
            cell_id=cell_id,
            success=False,
            stdout=exec_result.stdout if exec_result is not None else "",
            stderr=exec_result.stderr if exec_result is not None else "",
            duration_ms=duration_ms,
            error=f"R execution failed: {e}",
            mutation_warnings=exec_result.mutation_warnings if exec_result is not None else [],
        ).apply_remote_metadata(**remote_metadata)
        host.persist_display_outputs(cell_id, None)
        host.apply_execution_result_metadata(cell_id, error_result)
        return error_result


async def run_r_harness(
    notebook_path: Path,
    r_harness_path: Path,
    harness_env: Callable[[dict[str, str] | None], dict[str, str] | None],
    manifest_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run the R harness via ``Rscript``, mirroring ``_run_harness``.

    ``cwd`` is the notebook directory so ``.Rprofile`` activates the
    project's renv library.
    """
    try:
        harness_user = resolve_harness_user()
    except LocalExecutionRefused as exc:
        return _refused_result(exc)
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {
            "success": False,
            "error": (
                "Rscript not found on PATH. Install R "
                "(https://cran.r-project.org/) and reopen the notebook."
            ),
            "stderr": "",
            "stdout": "",
            "variables": {},
        }

    cmd = lifeline_command([rscript, str(r_harness_path), str(manifest_path)])

    from strata.notebook.process_tree import (
        subprocess_kwargs_for_new_group,
        terminate_subprocess_tree,
    )

    hand_over(manifest_path.parent, harness_user)
    lifeline_fds, lifeline_vars = lifeline_handoff()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(notebook_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=identity_env(harness_env(lifeline_vars), harness_user),
        pass_fds=lifeline_fds,
        **spawn_kwargs(harness_user),
        **subprocess_kwargs_for_new_group(),
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.CancelledError:
        logger.info(
            "R cell execution cancelled; terminating Rscript pid=%s",
            proc.pid,
        )
        try:
            await asyncio.shield(terminate_subprocess_tree(proc))
        except Exception:
            logger.exception(
                "Failed to terminate cancelled Rscript subprocess tree pid=%s",
                proc.pid,
            )
        raise
    except TimeoutError:
        await terminate_subprocess_tree(proc)
        raise TimeoutError()

    result_path = manifest_path.parent / "harness-result.json"
    if not result_path.exists():
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        return {
            "success": False,
            "error": (stderr_text.strip() or "Rscript exited without producing a result manifest"),
            "stderr": stderr_text,
            "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
            "variables": {},
        }
    return json.loads(read_run_file(result_path.parent, result_path.name))
