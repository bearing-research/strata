"""Loop cells: one harness run per iteration, the carry threaded between them."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio.to_thread
from vestigo.provenance import derive_subkey

from strata.notebook import output_dirs
from strata.notebook.annotations import LoopAnnotation, parse_annotations
from strata.notebook.cell_paths.context import artifact_content_type, effective_worker_name
from strata.notebook.env import moved_input
from strata.notebook.execution_result import CellExecutionResult, hit_console
from strata.notebook.harness_user import (
    LocalExecutionRefused,
    hand_over,
    read_run_file,
    resolve_harness_user,
)
from strata.notebook.immutability import MutationWarning
from strata.notebook.models import CellLanguage, MountSpec, TableWriteSpec
from strata.notebook.mounts import moved_key, moved_line, moved_parts
from strata.notebook.table_writes import StagedOutput, TableWriteError, write_back_failed

if TYPE_CHECKING:
    from strata.notebook.executor import _CellProvenance
    from strata.notebook.executor_host import ExecutorHost

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopContext:
    """What a loop cell asks of the executor, and nothing else."""

    materialize_upstreams_or_failure: Callable[
        [str, float, str], Awaitable[CellExecutionResult | None]
    ]
    resolve_cell_mount_specs: Callable[[str, str], list[MountSpec]]
    fingerprint_mounts: Callable[[list[MountSpec]], Awaitable[tuple[list[str], bool]]]
    resolve_runtime_env: Callable[[str, dict[str, str]], dict[str, str]]
    compute_provenance: Callable[..., Awaitable[_CellProvenance]]
    manifest_tables: Callable[..., dict[str, dict[str, Any]]]
    moved: Callable[[str], tuple[dict[str, int | None], dict[str, int]]]
    promote_reverted_outputs: Callable[..., int | None]
    canonical_output: Callable[[str, str, str], Any]
    table_write_refusal: Callable[[str], str | None]
    restore_gone_tables: Callable[[str, str], Awaitable[None]]
    prepare_mounts: Callable[..., Awaitable[dict[str, Any]]]
    load_input_blobs: Callable[..., dict[str, Any]]
    keep_failed_output_dirs: Callable[..., None]
    write_manifest: Callable[..., Path]
    run_token: Callable[[float], AbstractContextManager[str]]
    run_harness: Callable[..., Awaitable[dict[str, Any]]]
    input_refs: Callable[[str], dict[str, str]]
    write_cell_tables: Callable[..., None]
    store_display_outputs: Callable[..., list[dict[str, Any]]]
    store_console_outputs: Callable[..., None]
    # Cross-path state: what each cell's run put in its folders, by cell and folder name.
    folder_listings: dict[str, dict[str, Any]]
    # Read as the run starts; the WebSocket handler sets it when it makes the executor.
    on_iteration_complete: Callable[[dict[str, Any]], Awaitable[None]] | None


_LOOP_CONTENT_TYPE_EXT = {
    "arrow/ipc": ".arrow",
    "json/object": ".json",
    "pickle/object": ".pickle",
    "module/import": ".module.json",
    "module/cell": ".cell_module.json",
    "module/cell-instance": ".cell_instance.pickle",
}


@contextmanager
def _loop_folders(names: list[str]) -> Generator[Path | None]:
    """Where a loop's ``# @output_dir`` folders live, outside every iteration's run directory.

    Each iteration sees what the earlier ones wrote; the sweep holds off until the loop
    ends, since each iteration stores files ahead of the cell's manifest.
    """
    if not names:
        yield None
        return
    with output_dirs.assembling(), tempfile.TemporaryDirectory(prefix="strata_loop_dirs_") as root:
        yield Path(root)


def _loop_run_token(artifact: Any) -> str:
    """The token of the loop run that stored an ``@iter=k`` artifact."""
    return str(artifact.params.get("loop_run", ""))


def _cached_loop_result(
    host: ExecutorHost,
    context: LoopContext,
    cell_id: str,
    loop: LoopAnnotation,
    cell_provenance: str,
    start_time: float,
    table_writes: list[TableWriteSpec],
) -> CellExecutionResult | None:
    """The loop's own result from a previous identical run, or None.

    A hit needs the carry and every consumed variable to be the cell's canonical
    rows under this provenance: a duplicate's row is not this cell's result, and
    an output read only since the last run was never stored.
    """
    artifact_mgr = host.get_artifact_manager()
    if artifact_mgr.find_cached(derive_subkey(cell_provenance, loop.carry)) is None:
        return None
    consumed_vars = host.dag.consumed_variables.get(cell_id, set()) if host.dag else set()
    cell = host.notebook_state.get_cell(cell_id)
    # A revert finds its outputs under an older version, as a single run does.
    if (
        context.promote_reverted_outputs(
            cell_id,
            cell_provenance,
            {loop.carry, *consumed_vars},
            table_writes,
            len(cell.display_outputs) if cell is not None else 0,
        )
        is None
    ):
        return None
    uris: dict[str, str] = {}
    for var_name in sorted({loop.carry, *consumed_vars}):
        canonical = context.canonical_output(cell_id, var_name, cell_provenance)
        if canonical is None:
            return None
        uris[var_name] = f"strata://artifact/{canonical.id}@v={canonical.version}"
    uri = uris[loop.carry]
    if cell is not None:
        cell.cache_hit = True
    _set_loop_artifact_uris(host, cell_id, uri, consumed_vars, uris)
    cached_displays = host.resolve_cached_display_outputs(
        cell_id, cell_provenance, cell.display_outputs if cell is not None else []
    )
    result = CellExecutionResult(
        cell_id=cell_id,
        success=True,
        outputs={},
        **hit_console(
            host.resolve_cached_console(cell_id, cell_provenance),
            shows_this_key=cell is not None and cell.last_provenance_hash == cell_provenance,
        ),
        display_outputs=[output.model_dump() for output in cached_displays],
        duration_ms=(time.time() - start_time) * 1000,
        cache_hit=True,
        artifact_uri=uri,
        execution_method="cached",
    )
    return result


def _set_loop_artifact_uris(
    host: ExecutorHost,
    cell_id: str,
    carry_uri: str,
    consumed_vars: set[str],
    uris: dict[str, str],
) -> None:
    """Record a loop's outputs on the cell the way ``_store_outputs`` does: consumed only."""
    cell = host.notebook_state.get_cell(cell_id)
    if cell is None:
        return
    for var_name, uri in uris.items():
        if var_name in consumed_vars:
            cell.artifact_uris[var_name] = uri
    cell.artifact_uri = carry_uri


@dataclass
class _LoopRun:
    """What a failure anywhere in the loop still reports."""

    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)
    mutation_warnings: list[MutationWarning] = field(default_factory=list)
    # Until the outputs are current; after that a folder is not a failed run's to keep.
    keep_failed: Callable[[], None] | None = None


async def run_loop_cell(
    host: ExecutorHost,
    context: LoopContext,
    cell_id: str,
    source: str,
    loop: LoopAnnotation,
    timeout_seconds: float,
    start_time: float,
    *,
    materialize_upstreams: bool,
    use_cache: bool = True,
    folders: Path | None = None,
) -> CellExecutionResult:
    """Execute a loop cell by running the body up to ``loop.max_iter`` times.

    Local worker only, no rw mounts. Each iteration is a fresh harness
    subprocess; the carry passes in as a named input and out as a named
    output, stored with an ``@iter=k`` suffix. ``# @output_dir`` folders live
    under *folders* for the whole loop, each stored after every iteration too.
    """
    run = _LoopRun()
    try:
        return await _run_loop(
            host,
            context,
            run,
            cell_id,
            source,
            loop,
            timeout_seconds,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
            folders=folders,
        )
    except Exception as exc:
        logger.exception("Loop cell %s failed", cell_id)
        if run.keep_failed is not None:
            run.keep_failed()
        return CellExecutionResult(
            cell_id=cell_id,
            success=False,
            error=f"Loop execution failed: {exc}",
            stdout="".join(run.stdout),
            stderr="".join(run.stderr),
            duration_ms=(time.time() - start_time) * 1000,
            execution_method="loop",
            mutation_warnings=run.mutation_warnings,
        )


async def _run_loop(
    host: ExecutorHost,
    context: LoopContext,
    run: _LoopRun,
    cell_id: str,
    source: str,
    loop: LoopAnnotation,
    timeout_seconds: float,
    start_time: float,
    *,
    materialize_upstreams: bool,
    use_cache: bool,
    folders: Path | None,
) -> CellExecutionResult:
    annotations = parse_annotations(source, CellLanguage.PYTHON)
    effective_worker = effective_worker_name(host.notebook_state, cell_id, annotations.worker)
    if effective_worker != "local":
        return CellExecutionResult(
            cell_id=cell_id,
            success=False,
            error=(
                f"A loop cell runs on the notebook's own server, not on worker "
                f"'{effective_worker}'. Add `# @worker local` to run the loop here, or "
                f"move the work that needs '{effective_worker}' into an ordinary cell."
            ),
            execution_method="loop",
        )
    if materialize_upstreams:
        failure = await context.materialize_upstreams_or_failure(cell_id, start_time, "loop")
        if failure is not None:
            return failure

    if annotations.datasets:
        return CellExecutionResult(
            cell_id=cell_id,
            success=False,
            error=(
                "@dataset is not supported on loop cells; read the dataset in an "
                "upstream cell and pass what the loop needs from it"
            ),
            execution_method="loop",
        )

    if annotations.fetches:
        # Each iteration is its own harness run and the fetch is checked only after the loop,
        # so nothing would inject the name and the recorded digest could differ from the bytes
        # read.
        return CellExecutionResult(
            cell_id=cell_id,
            success=False,
            error=(
                "@fetch is not supported on loop cells; fetch in an upstream "
                "cell and pass what the loop needs from it"
            ),
            execution_method="loop",
        )

    mount_specs = context.resolve_cell_mount_specs(cell_id, source)
    _, has_rw_mount = await context.fingerprint_mounts(mount_specs)
    if has_rw_mount:
        return CellExecutionResult(
            cell_id=cell_id,
            success=False,
            error=(
                "Loop cells do not support rw mounts, which would make "
                "per-iteration caching incorrect. Use an ro mount or "
                "move the side-effect to a non-loop cell."
            ),
            execution_method="loop",
        )

    runtime_env = context.resolve_runtime_env(cell_id, annotations.env)

    # Computed up front so the loop is skipped on a cache hit. The loop dispatch precedes
    # ``execute_cell``'s cache check, and ``_materialize_upstreams`` calls ``execute_cell``
    # on every upstream assuming it caches, so otherwise any downstream run re-runs the loop.
    prov = await context.compute_provenance(
        cell_id,
        source,
        annotations=annotations,
        mount_specs=mount_specs,
    )
    cell_provenance = prov.provenance_hash
    env_hash = prov.env_hash
    source_hash = prov.source_hash
    # Every iteration reads the snapshots the provenance names, as a single run does.
    try:
        loop_tables = context.manifest_tables(annotations.tables, prov.table_snapshots)
    except RuntimeError as exc:
        return CellExecutionResult(
            cell_id=cell_id,
            success=False,
            error=str(exc),
            execution_method="loop",
        )
    if use_cache:
        cached = _cached_loop_result(
            host, context, cell_id, loop, cell_provenance, start_time, annotations.table_writes
        )
        if cached is not None:
            try:
                await context.restore_gone_tables(cell_id, source)
            except TableWriteError as exc:
                cell = host.notebook_state.get_cell(cell_id)
                if cell is not None:
                    cell.cache_hit = False
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=write_back_failed(exc),
                    execution_method="loop",
                )
            from strata.notebook.runtime_state import persist_cell_loop_run

            # A hit serves an earlier run, not the one last started; let the steps on
            # disk speak for themselves again.
            persist_cell_loop_run(host.path, cell_id, None)
            host.record_successful_execution_provenance(
                cell_id,
                cell_provenance,
                source_hash,
                env_hash,
                table_reads=prov.table_reads,
                reads=prov.reads(*context.moved(cell_id)),
                key_parts=prov.key_parts,
            )
            return cached

    refusal = context.table_write_refusal(source)
    if refusal is not None:
        return CellExecutionResult(
            cell_id=cell_id, success=False, error=refusal, execution_method="loop"
        )

    # Prepared once, and the run keyed on it: every iteration reads this listing, which
    # the one the cache check used may have moved past.
    resolved_mounts = await context.prepare_mounts(mount_specs, cell_id)
    prov = prov.over_mounts(resolved_mounts)
    cell_provenance = prov.provenance_hash
    carry_var_provenance = derive_subkey(cell_provenance, loop.carry)

    try:
        carry, carry_content_type = _resolve_loop_seed(host, cell_id, loop)
        if folders is not None:
            _seed_loop_folders(host, loop, annotations.output_dirs, folders)
    except ValueError as exc:
        return CellExecutionResult(
            cell_id=cell_id,
            success=False,
            error=str(exc),
            execution_method="loop",
        )

    artifact_mgr = host.get_artifact_manager()

    # Downstream-consumed variables beyond the carry must be persisted too, or a later
    # cell reading e.g. ``final_metrics`` finds the name unbound.
    consumed_vars = host.dag.consumed_variables.get(cell_id, set()) if host.dag else set()
    folder_names = list(annotations.output_dirs)
    extra_consumed = sorted(consumed_vars - {loop.carry, *folder_names})
    # var -> (blob, content_type), captured each iteration (the tmpdir dies with the
    # ``with`` block); the last capture is the final state.
    extra_blobs: dict[str, tuple[bytes, str]] = {}
    # An in-place update (``state["i"] += 1``) keeps the carry's id(), so the harness
    # serializes it only when it is listed as mutated.
    loop_cell = host.notebook_state.get_cell(cell_id)
    mutation_defines = sorted(
        {loop.carry, *(loop_cell.mutation_defines if loop_cell is not None else [])}
    )

    final_artifact_uri: str | None = None
    final_result: dict[str, Any] | None = None
    loop_run = uuid.uuid4().hex
    from strata.notebook.runtime_state import persist_cell_loop_run

    persist_cell_loop_run(host.path, cell_id, loop_run)
    combined_stdout = run.stdout
    combined_stderr = run.stderr
    all_mutation_warnings = run.mutation_warnings
    final_display_files: dict[str, bytes] = {}
    context.folder_listings.pop(cell_id, None)

    folder_mounts: dict[str, dict[str, str]] | None = None
    folder_seen: dict[str, dict[str, tuple[tuple[int, ...], dict[str, Any]]]] = {}
    folder_manifests: dict[str, bytes] = {}
    if folders is not None:
        # Made once, before any iteration: a process one leaves behind could put a link
        # where a later mkdir would follow it.
        folder_mounts = output_dirs.mount_entries(source, folders)
        try:
            harness_user = resolve_harness_user()
        except LocalExecutionRefused:
            harness_user = None  # the run is refused where the cell would start
        hand_over(folders, harness_user)
        folder_seen = {name: {} for name in folder_names}

    def keep_failed() -> None:
        if folders is not None:
            context.keep_failed_output_dirs(cell_id, source, folders, cell_provenance, env_hash)

    run.keep_failed = keep_failed

    def no_carry_file(error: str) -> CellExecutionResult:
        keep_failed()
        return CellExecutionResult(
            cell_id=cell_id,
            success=False,
            error=error,
            stdout="".join(combined_stdout),
            stderr="".join(combined_stderr),
            duration_ms=(time.time() - start_time) * 1000,
            execution_method="loop",
            mutation_warnings=all_mutation_warnings,
        )

    prev_carry_hash: str | None = None
    for k in range(loop.max_iter):
        with tempfile.TemporaryDirectory(prefix=f"strata_loop_iter_{k}_") as tmpdir:
            output_dir = Path(tmpdir)

            # This writes the upstream seed for the carry too; it is overwritten below so iter k
            # sees iter k-1's output.
            input_specs = context.load_input_blobs(cell_id, output_dir)

            ext = _LOOP_CONTENT_TYPE_EXT.get(carry_content_type, ".pickle")
            carry_file = f"{loop.carry}{ext}"
            prev_carry_hash = await anyio.to_thread.run_sync(
                _write_carry, artifact_mgr, carry, output_dir / carry_file, prev_carry_hash
            )
            input_specs[loop.carry] = {
                "content_type": carry_content_type,
                "file": carry_file,
            }

            loop_config: dict[str, Any] | None = (
                {"until_expr": loop.until_expr, "iteration": k}
                if loop.until_expr is not None
                else {"iteration": k}
            )
            with context.run_token(timeout_seconds) as run_token:
                manifest_path = context.write_manifest(
                    source,
                    input_specs,
                    output_dir,
                    runtime_env,
                    resolved_mounts,
                    mutation_defines=mutation_defines,
                    loop_config=loop_config,
                    tables=loop_tables,
                    cell_id=cell_id,
                    folder_mounts=folder_mounts,
                    run_token=run_token,
                )

                venv_path = host.venv_python
                try:
                    result = await context.run_harness(
                        manifest_path, venv_path, timeout_seconds, run_token=run_token
                    )
                except TimeoutError:
                    keep_failed()
                    duration_ms = (time.time() - start_time) * 1000
                    return CellExecutionResult(
                        cell_id=cell_id,
                        success=False,
                        error=(
                            f"Loop cell iter {k} timed out after "
                            f"{timeout_seconds}s (per-iteration timeout)."
                        ),
                        stdout="".join(combined_stdout),
                        stderr="".join(combined_stderr),
                        duration_ms=duration_ms,
                        execution_method="loop",
                        mutation_warnings=all_mutation_warnings,
                    )

            combined_stdout.append(result.get("stdout", ""))
            combined_stderr.append(result.get("stderr", ""))
            all_mutation_warnings.extend(result.get("mutation_warnings", []))
            final_result = result

            if not result.get("success", False):
                traceback_text = result.get("traceback") or None
                # ``str()`` of a bare ``assert`` is empty; the traceback's last line is not.
                error_msg = result.get("error") or (
                    traceback_text.strip().splitlines()[-1] if traceback_text else "Unknown error"
                )
                keep_failed()
                duration_ms = (time.time() - start_time) * 1000
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=f"Loop cell iter {k} failed: {error_msg}",
                    # What a reader sees in place of ``error``, so it says where too.
                    traceback=(
                        f"Loop cell iter {k} failed:\n{traceback_text}" if traceback_text else None
                    ),
                    traceback_full=result.get("traceback_full") or None,
                    stdout="".join(combined_stdout),
                    stderr="".join(combined_stderr),
                    duration_ms=duration_ms,
                    execution_method="loop",
                    mutation_warnings=all_mutation_warnings,
                )

            loop_state = result.get("loop") or {}
            if loop_state.get("error"):
                # A predicate that cannot be evaluated would otherwise run every iteration and
                # report success.
                keep_failed()
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=f"Loop cell iter {k}: {loop_state['error']}",
                    stdout="".join(combined_stdout),
                    stderr="".join(combined_stderr),
                    duration_ms=(time.time() - start_time) * 1000,
                    execution_method="loop",
                    mutation_warnings=all_mutation_warnings,
                )

            carry_meta = result.get("variables", {}).get(loop.carry)
            if not isinstance(carry_meta, dict) or carry_meta.get("content_type") == "error":
                duration_ms = (time.time() - start_time) * 1000
                detail = (
                    carry_meta.get("error")
                    if isinstance(carry_meta, dict)
                    else "carry variable missing from cell outputs"
                )
                keep_failed()
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=(
                        f"Loop cell iter {k} did not produce carry "
                        f"variable '{loop.carry}': {detail}. The cell "
                        f"body must rebind `{loop.carry}` every "
                        f"iteration."
                    ),
                    stdout="".join(combined_stdout),
                    stderr="".join(combined_stderr),
                    duration_ms=duration_ms,
                    execution_method="loop",
                    mutation_warnings=all_mutation_warnings,
                )

            new_content_type = str(carry_meta.get("content_type", "pickle/object"))
            new_carry_file = carry_meta.get("file")
            if not isinstance(new_carry_file, str):
                return no_carry_file(
                    f"Loop cell iter {k} produced carry metadata without a 'file' entry: "
                    f"{carry_meta}"
                )
            if not (output_dir / new_carry_file).exists():
                return no_carry_file(
                    f"Loop cell iter {k} carry file not produced by harness: {new_carry_file}"
                )
            new_carry_blob = read_run_file(output_dir, new_carry_file)

            # Capture while the iteration tmpdir is still alive.
            for extra_var in extra_consumed:
                extra_meta = result.get("variables", {}).get(extra_var)
                if not isinstance(extra_meta, dict) or extra_meta.get("content_type") == "error":
                    continue
                extra_file = extra_meta.get("file")
                if not isinstance(extra_file, str):
                    continue
                extra_path = output_dir / extra_file
                if extra_path.exists():
                    extra_blobs[extra_var] = (
                        read_run_file(output_dir, extra_file),
                        str(extra_meta.get("content_type", "pickle/object")),
                    )

            if loop_state.get("until_reached") or k == loop.max_iter - 1:
                # The final state is the cell's result, so its displays are the cell's.
                final_display_files = {
                    name: read_run_file(output_dir, name)
                    for display in result.get("displays") or []
                    if isinstance(display, dict)
                    and isinstance(name := display.get("file"), str)
                    and name
                    and (output_dir / name).exists()
                }

            # Chains through the previous iteration's carry bytes so identical chains are
            # detectable.
            iter_provenance = derive_subkey(source_hash, prev_carry_hash, f"iter={k}")

            artifact = artifact_mgr.store_cell_output(
                cell_id=cell_id,
                variable_name=loop.carry,
                blob_data=new_carry_blob,
                content_type=new_content_type,
                provenance_hash=iter_provenance,
                source_hash=source_hash,
                source=source,
                iteration=k,
                extra_params={"loop_run": loop_run},
            )
            prev_carry_hash = hashlib.sha256(new_carry_blob).hexdigest()
            # The next iteration copies it from the store: a checkpoint can be gigabytes.
            del new_carry_blob
            final_artifact_uri = f"strata://artifact/{artifact.id}@v={artifact.version}"
            if folders is not None:
                for name in folder_names:
                    folder_manifests[name] = output_dirs.collect(
                        artifact_mgr,
                        cell_id,
                        folders,
                        name,
                        seen=folder_seen[name],
                    )
                    artifact_mgr.store_cell_output(
                        cell_id=cell_id,
                        variable_name=name,
                        blob_data=folder_manifests[name],
                        content_type=output_dirs.CONTENT_TYPE,
                        provenance_hash=derive_subkey(iter_provenance, name),
                        source_hash=source_hash,
                        source=source,
                        iteration=k,
                        extra_params={"loop_run": loop_run},
                    )

            carry = (artifact.id, artifact.version)
            carry_content_type = new_content_type

            iter_duration_ms = (time.time() - start_time) * 1000
            if context.on_iteration_complete is not None:
                try:
                    await context.on_iteration_complete(
                        {
                            "cell_id": cell_id,
                            "iteration": k,
                            "max_iter": loop.max_iter,
                            "artifact_uri": final_artifact_uri,
                            "content_type": new_content_type,
                            "until_reached": bool(loop_state.get("until_reached")),
                            "duration_ms": int(iter_duration_ms),
                        }
                    )
                except Exception:
                    logger.exception(
                        "on_iteration_complete callback failed for cell %s iter %d",
                        cell_id,
                        k,
                    )
            if loop_state.get("until_reached"):
                break

    # Store the final carry under the canonical (non-iter) id so downstream cells resolve
    # it via _load_input_blobs. Its provenance must use the non-loop per-variable scheme
    # ``sha256(prov:var_name)`` that ``compute_staleness`` recomputes on re-open, or the
    # loop cell always looks stale. ``cell_provenance``, ``env_hash`` and
    # ``carry_var_provenance`` were computed before the loop.

    # A mount or local package that moved while the loop ran: its outputs are kept for this
    # run only, under
    # a key of its own, and the cell is recorded under one no check makes.
    moved = await anyio.to_thread.run_sync(moved_input, prov.prepared, prov.local_packages)
    output_provenance = cell_provenance
    recorded_provenance, recorded_parts = cell_provenance, prov.key_parts
    if moved is not None:
        combined_stderr.append(moved_line(moved))
        output_provenance = derive_subkey(cell_provenance, f"run={uuid.uuid4().hex}")
        carry_var_provenance = derive_subkey(output_provenance, loop.carry)
        recorded_provenance = moved_key(cell_provenance)
        recorded_parts = moved_parts(prov.key_parts, moved)

    # Environment identity for lineage and the team cache; the cost is the whole loop,
    # which is what a teammate skipping it saves.
    loop_build_env = str((final_result or {}).get("build_env") or "")
    loop_duration_ms = (time.time() - start_time) * 1000

    # Record inputs so a training loop's output has a lineage graph.
    loop_input_versions = context.input_refs(cell_id)

    # Staged, then finalized with the extra outputs below so all become current together.
    carry_blob = artifact_mgr.load_artifact_data(*carry)
    staged = [
        artifact_mgr.stage_cell_output(
            cell_id=cell_id,
            variable_name=loop.carry,
            blob_data=carry_blob,
            content_type=carry_content_type,
            provenance_hash=carry_var_provenance,
            input_versions=loop_input_versions,
            source_hash=source_hash,
            source=source,
            env_hash=env_hash,
            build_env=loop_build_env,
            build_duration_ms=loop_duration_ms,
        )
    ]
    table_outputs = {
        loop.carry: StagedOutput(
            artifact_id=staged[0].artifact_id,
            version=staged[0].version,
            blob=carry_blob,
            content_type=carry_content_type,
            provenance_hash=carry_var_provenance,
        )
    }

    # Same canonical ids and per-variable provenance as non-loop outputs.
    extra_outputs: dict[str, Any] = {}
    for extra_var in extra_consumed:
        captured = extra_blobs.get(extra_var)
        if captured is None:
            logger.error(
                "Loop cell %s defines '%s' (consumed downstream) but the final "
                "iteration did not produce it, so downstream cells will miss it.",
                cell_id,
                extra_var,
            )
            continue
        extra_blob, extra_content_type = captured
        staged.append(
            artifact_mgr.stage_cell_output(
                cell_id=cell_id,
                variable_name=extra_var,
                blob_data=extra_blob,
                content_type=extra_content_type,
                provenance_hash=derive_subkey(output_provenance, extra_var),
                input_versions=loop_input_versions,
                source_hash=source_hash,
                source=source,
                env_hash=env_hash,
                build_env=loop_build_env,
                build_duration_ms=loop_duration_ms,
            )
        )
        table_outputs[extra_var] = StagedOutput(
            artifact_id=staged[-1].artifact_id,
            version=staged[-1].version,
            blob=extra_blob,
            content_type=extra_content_type,
            provenance_hash=derive_subkey(output_provenance, extra_var),
        )
        extra_outputs[extra_var] = {
            "content_type": extra_content_type,
            "file": (f"{extra_var}{_LOOP_CONTENT_TYPE_EXT.get(extra_content_type, '.pickle')}"),
        }

    # The last iteration's folders are the cell's, kept even when no cell reads them.
    for name in folder_names:
        staged.append(
            artifact_mgr.stage_cell_output(
                cell_id=cell_id,
                variable_name=name,
                blob_data=folder_manifests[name],
                content_type=output_dirs.CONTENT_TYPE,
                provenance_hash=derive_subkey(output_provenance, name),
                input_versions=loop_input_versions,
                source_hash=source_hash,
                source=source,
                env_hash=env_hash,
                build_env=loop_build_env,
                build_duration_ms=loop_duration_ms,
            )
        )
        context.folder_listings.setdefault(cell_id, {})[name] = output_dirs.listing(
            name, folder_manifests[name]
        )

    # The final values, before they become current, as for any cell: a failed write
    # keeps none of them, so the next run is not a cache hit and writes again.
    if annotations.table_writes:
        try:
            context.write_cell_tables(annotations.table_writes, table_outputs)
        except TableWriteError as exc:
            artifact_mgr.discard_cell_outputs(staged)
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                error=str(exc),
                stdout="".join(combined_stdout),
                stderr="".join(combined_stderr),
                duration_ms=(time.time() - start_time) * 1000,
                execution_method="loop",
                mutation_warnings=all_mutation_warnings,
            )

    stored = artifact_mgr.finalize_cell_outputs(staged)
    run.keep_failed = None
    canonical_artifact = stored[0]
    canonical_uri = f"strata://artifact/{canonical_artifact.id}@v={canonical_artifact.version}"
    # Downstream provenance hashes these, as for any other cell's consumed outputs.
    _set_loop_artifact_uris(
        host,
        cell_id,
        canonical_uri,
        consumed_vars,
        {
            name: f"strata://artifact/{artifact.id}@v={artifact.version}"
            for name, artifact in zip(
                [loop.carry, *extra_outputs, *folder_names], stored, strict=True
            )
        },
    )

    # Lets ``compute_staleness`` hit the "uncached ready" path for leaf loop cells.
    host.record_successful_execution_provenance(
        cell_id,
        recorded_provenance,
        source_hash,
        env_hash,
        table_reads=prov.table_reads,
        reads=prov.reads(*context.moved(cell_id)),
        key_parts=recorded_parts,
    )
    raw_displays = final_result.get("displays") if final_result else None
    display_outputs = (
        [d for d in raw_displays if isinstance(d, dict)] if isinstance(raw_displays, list) else []
    )
    if display_outputs or context.folder_listings.get(cell_id):
        # The iteration's run dir is gone; stage its display files to store them as a
        # Python cell's are, after the outputs so no display outlives a failed finalize.
        with tempfile.TemporaryDirectory(prefix="strata_loop_displays_") as display_dir:
            for name, blob in final_display_files.items():
                (Path(display_dir) / name).write_bytes(blob)
            display_outputs = context.store_display_outputs(
                cell_id,
                Path(display_dir),
                recorded_provenance,
                prov.input_hashes,
                display_outputs,
                source_hash=source_hash,
                source=source,
                env_hash=env_hash,
            )
    # So a hit replays this run's. A leaf's is its record of the run: a cold open reads it ready.
    if not annotations.nocache and moved is None:
        context.store_console_outputs(
            cell_id,
            cell_provenance,
            "".join(combined_stdout),
            "".join(combined_stderr),
            prov.input_hashes,
            source_hash=source_hash,
            source=source,
            env_hash=env_hash,
            display_count=len(display_outputs),
            duration_ms=loop_duration_ms,
        )

    duration_ms = loop_duration_ms

    return CellExecutionResult(
        cell_id=cell_id,
        success=True,
        stdout="".join(combined_stdout),
        stderr="".join(combined_stderr),
        outputs={
            loop.carry: {
                "content_type": carry_content_type,
                "file": (
                    f"{loop.carry}{_LOOP_CONTENT_TYPE_EXT.get(carry_content_type, '.pickle')}"
                ),
            },
            **extra_outputs,
        },
        display_outputs=display_outputs,
        duration_ms=duration_ms,
        artifact_uri=canonical_uri,
        execution_method="loop",
        mutation_warnings=all_mutation_warnings,
        moved_input=moved,
    )


def _seed_loop_folders(
    host: ExecutorHost, loop: LoopAnnotation, names: list[str], folders: Path
) -> None:
    """Start a resumed loop's folders as ``start_from``'s iteration left them.

    Only a folder the seed carry's own run stored: one from another run would not
    match the carry. Raises ``ValueError`` if that folder's manifest is gone.
    """
    if loop.start_from_cell is None or loop.start_from_iter is None:
        return
    artifact_mgr = host.get_artifact_manager()
    carry = artifact_mgr.get_iteration_artifact(
        loop.start_from_cell, loop.carry, loop.start_from_iter
    )
    if carry is None:
        return
    for name in names:
        kept = artifact_mgr.get_iteration_artifact(loop.start_from_cell, name, loop.start_from_iter)
        if kept is None or _loop_run_token(kept) != _loop_run_token(carry):
            continue
        manifest = artifact_mgr.artifact_store.read_blob(kept.id, kept.version)
        if manifest is None:
            raise ValueError(f"Loop seed folder missing for {kept.id}@v={kept.version}.")
        output_dirs.materialize(
            artifact_mgr.artifact_store,
            kept.id,
            manifest,
            folders / output_dirs.OUTPUT_DIRS / name,
            tenant=kept.tenant,
        )


def _write_carry(manager: Any, carry: tuple[str, int], path: Path, digest: str | None) -> str:
    """Copy the carry where the harness reads it, returning *digest*, or its bytes' if unknown.

    Only the seed's is unknown; reading the carry each iteration would cost what a clone saved.
    """
    # The seed may be here already, and a clone cannot replace a file.
    path.unlink(missing_ok=True)
    manager.copy_ready_blob(*carry, path)
    if digest is not None:
        return digest
    with path.open("rb") as written:
        return hashlib.file_digest(written, "sha256").hexdigest()


def _resolve_loop_seed(
    host: ExecutorHost,
    cell_id: str,
    loop: LoopAnnotation,
) -> tuple[tuple[str, int], str]:
    """Resolve the iter-0 carry seed as ``((artifact_id, version), content_type)``.

    From ``# @loop start_from=<cell>@iter=<k>``, else the latest artifact of
    the upstream cell defining the carry. Raises ``ValueError`` if neither exists.
    """
    artifact_mgr = host.get_artifact_manager()

    if loop.start_from_cell is not None and loop.start_from_iter is not None:
        artifact_id = artifact_mgr.cell_artifact_id(
            loop.start_from_cell, loop.carry, loop.start_from_iter
        )
        artifact = artifact_mgr.artifact_store.get_latest_version(artifact_id)
        if artifact is None or artifact.state not in ("ready", "superseded"):
            raise ValueError(
                f"Loop seed artifact not found for "
                f"start_from={loop.start_from_cell}@iter={loop.start_from_iter}. "
                f"Run that cell through iteration {loop.start_from_iter} first."
            )
        # Every run rewrites @iter=0, so a step whose run token differs from it is
        # left over from an older, longer run. A run that failed at iteration 0
        # rewrote nothing; the token recorded at its start catches that.
        from strata.notebook.runtime_state import load_runtime_state

        first = artifact_mgr.get_iteration_artifact(loop.start_from_cell, loop.carry, 0)
        started = load_runtime_state(host.path).cells.get(loop.start_from_cell)
        latest_run = started.loop_run if started is not None else None
        if (
            first is None
            or _loop_run_token(first) != _loop_run_token(artifact)
            or (latest_run is not None and latest_run != _loop_run_token(artifact))
        ):
            raise ValueError(
                f"Loop seed start_from={loop.start_from_cell}@iter={loop.start_from_iter} "
                f"is left over from an older run of cell {loop.start_from_cell}: its "
                f"latest run did not reach iteration {loop.start_from_iter}. Run that "
                f"cell through iteration {loop.start_from_iter} first."
            )
        if not artifact_mgr.artifact_store.blob_exists(artifact_id, artifact.version):
            raise ValueError(f"Loop seed blob missing for {artifact_id}@v={artifact.version}.")
        return (artifact_id, artifact.version), artifact_content_type(artifact)

    cell = host.notebook_state.get_cell(cell_id)
    if cell is None:
        raise ValueError(f"Loop cell {cell_id!r} not found in notebook state.")

    notebook_id = host.notebook_state.id
    for upstream_id in cell.upstream_ids:
        upstream_cell = host.notebook_state.get_cell(upstream_id)
        if upstream_cell is None or loop.carry not in host.wired_variables(cell_id, upstream_id):
            continue
        upstream_artifact_id = f"nb_{notebook_id}_cell_{upstream_id}_var_{loop.carry}"
        artifact = artifact_mgr.artifact_store.get_latest_version(upstream_artifact_id)
        if artifact is None or artifact.state not in ("ready", "superseded"):
            continue
        if not artifact_mgr.artifact_store.blob_exists(upstream_artifact_id, artifact.version):
            raise ValueError(
                f"Loop seed blob missing for {upstream_artifact_id}@v={artifact.version}."
            )
        return (upstream_artifact_id, artifact.version), artifact_content_type(artifact)

    raise ValueError(
        f"Cannot resolve loop carry '{loop.carry}': no upstream cell "
        f"defines it and no @loop start_from annotation is set. "
        f"Define `{loop.carry}` in an upstream cell or add "
        f"`# @loop start_from=<cell>@iter=<k>`."
    )
