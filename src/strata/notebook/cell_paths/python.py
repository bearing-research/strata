"""Python cells: provenance, upstreams, cache check, the harness or a worker, persist."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio.to_thread
from vestigo import checkpoint_files
from vestigo.provenance import derive_subkey

from strata.notebook.annotations import parse_annotations
from strata.notebook.cell_paths.context import (
    _ARTIFACT_EXT_BY_CONTENT_TYPE,
    OutputNotSaved,
    RemoteExecutionError,
    _mount_sync_failure_result,
    _refused_result,
    _store_failure_result,
    _worker_mount_writes,
    refused_run,
)
from strata.notebook.cell_paths.context import no_interpreter_message as _no_interpreter_message
from strata.notebook.cell_paths.module_export import write_module_export_outputs
from strata.notebook.dependencies import UV_NOT_FOUND_MESSAGE, resolve_uv, uv_env
from strata.notebook.execution_result import CellExecutionResult, hit_console
from strata.notebook.harness_user import (
    HarnessUser,
    LocalExecutionRefused,
    UnsafeRunFile,
    hand_over,
    identity_env,
    read_run_file,
    resolve_harness_user,
    spawn_kwargs,
)
from strata.notebook.lifeline import lifeline_handoff
from strata.notebook.local_runs import (
    EVENTS_DRAIN_SECONDS,
    RESTORE_DIR,
    LocalRun,
    LocalRuns,
    RunFenced,
    RunRefused,
    relay_events,
)
from strata.notebook.models import MountMode, MountSpec, WorkerBackendType
from strata.notebook.mounts import MountWrite, ResolvedMount
from strata.notebook.provenance import safe_filename_stem
from strata.notebook.run_channel import CHECKPOINTS_DIR, EVENTS_FD_ENV
from strata.notebook.serializer import ContentType
from strata.notebook.table_writes import TableWriteError, write_back_failed
from strata.notebook.team_store import TeamPull
from strata.notebook.workers import (
    is_embedded_executor_worker,
    is_http_executor_worker,
    resolve_worker_spec,
    worker_transport,
)

if TYPE_CHECKING:
    from strata.notebook.executor import _CellProvenance
    from strata.notebook.executor_host import ExecutorHost

logger = logging.getLogger(__name__)

_OUTPUT_VARIABLE = re.compile(r"_var_([^@]+)")


@dataclass(frozen=True)
class PythonContext:
    """What a Python cell's run asks of the executor, and nothing else."""

    execute_each_cell: Callable[..., Awaitable[CellExecutionResult]]
    execute_fanout_cell: Callable[..., Awaitable[CellExecutionResult]]
    fanout_variants: Callable[[str], Any]
    materialize_upstreams: Callable[[str], Awaitable[None]]
    compute_provenance: Callable[[str, str], Awaitable[_CellProvenance]]
    remote_execution_metadata: Callable[..., dict[str, str]]
    manifest_tables: Callable[..., dict[str, dict[str, Any]]]
    find_or_pull_outputs: Callable[..., Awaitable[tuple[Any, TeamPull | None]]]
    validate_output_hit: Callable[..., tuple[Any, list[Any]]]
    promote_reverted_leaf: Callable[..., Any]
    moved: Callable[[str], tuple[dict[str, int | None], dict[str, int]]]
    table_write_refusal: Callable[[str], str | None]
    restore_gone_tables: Callable[[str, str], Awaitable[None]]
    load_input_blobs: Callable[..., Any]
    add_dataset_inputs: Callable[..., None]
    prepare_mounts: Callable[..., Awaitable[dict[str, ResolvedMount]]]
    local_runs: Callable[[], LocalRuns]
    input_refs: Callable[..., dict[str, str]]
    restore_manifest: Callable[[LocalRun, Path], dict[str, Any] | None]
    dispatch_execution: Callable[
        ..., Awaitable[tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]]
    ]
    parse_result: Callable[..., CellExecutionResult]
    collect_output_dirs_off_loop: Callable[..., Awaitable[Any]]
    keep_failed_output_dirs: Callable[..., None]
    store_outputs: Callable[..., bool]
    close_open_run: Callable[[str], Awaitable[None]]
    push_to_team_store: Callable[..., Awaitable[None]]
    store_result_displays: Callable[..., None]
    store_console_outputs: Callable[..., Any]
    store_mount_writes: Callable[..., None]
    record_good_run: Callable[..., None]
    timeout_message: Callable[[str, float], str]
    sync_back_mounts: Callable[..., Awaitable[list[MountWrite]]]
    # Cross-path state, the same dicts the executor holds, by cell id.
    received_output_files: dict[str, Any]
    folder_listings: dict[str, Any]
    each_items: dict[str, dict[str, Any]]
    each_listed: dict[str, dict[str, ResolvedMount]]
    branch_inits: dict[str, str]
    open_runs: dict[str, tuple[LocalRun, asyncio.Task[None]]]
    variant_writes: dict[str, list[MountWrite]]


@dataclass(frozen=True)
class LocalDispatchContext:
    """What a run on this machine asks of the executor."""

    prepare_mounts: Callable[..., Awaitable[dict[str, ResolvedMount]]]
    local_runs: Callable[[], LocalRuns]
    run_token: Callable[[float], Any]
    write_manifest: Callable[..., Path]
    run_harness: Callable[..., Awaitable[dict[str, Any]]]
    pool: Any


def _add_fetch_inputs(
    input_specs: dict[str, dict[str, Any]], fetched: dict[str, Path], output_dir: Path
) -> None:
    """Put each ``@fetch``'s bytes in *output_dir* as an input a worker receives.

    It rides every transport like an upstream value; ``file/path`` tells the
    harness to inject the path rather than load it. Files keep the URL's file
    name behind an index, since two fetch names can differ only in case.
    """
    for index, (name, path) in enumerate(sorted(fetched.items())):
        file_name = f"__fetch_{index}_{path.name}"
        target = output_dir / file_name
        try:
            os.link(path, target)
        except OSError:
            shutil.copyfile(path, target)
        input_specs[name] = {"content_type": ContentType.FILE_PATH.value, "file": file_name}


def _fenced_result(exc: RunFenced) -> dict[str, Any]:
    """A harness result for a run whose attempt lost its build mid-cell."""
    return {"success": False, "error": str(exc), "stderr": "", "stdout": "", "variables": {}}


def python_harness_command(
    harness_path: Path,
    notebook_path: Path,
    manifest_path: Path,
    venv_python: Path,
    harness_user: HarnessUser | None,
) -> list[str] | None:
    """The argv that starts the cold Python harness; ``None`` without uv.

    A separate method so a test can skip ``uv run`` without replacing the
    spawn's environment, user and refusal logic.
    """
    if harness_user is not None:
        # Straight to the venv interpreter: ``uv run`` wants a writable uv cache a separate
        # user would need arranged, and the venv is already synced.
        return [str(venv_python), str(harness_path), str(manifest_path)]
    uv = resolve_uv()
    if uv is None:
        return None
    return [
        uv,
        "run",
        "--directory",
        str(notebook_path),
        "python",
        str(harness_path),
        str(manifest_path),
    ]


def restore_manifest(
    host: ExecutorHost,
    write_ready_input_file: Callable[..., Any],
    run: LocalRun,
    output_dir: Path,
) -> dict[str, Any] | None:
    """The manifest's ``restore`` entry: the release's parts, written as input files.

    A checkpoint of files (``vestigo.checkpoint_files``, whoever wrote it) is written
    out as a folder under its own paths. A resumed run continues its build's
    checkpoint sequence; a branch starts its own.
    """
    release_id = run.restores_from
    if release_id is None:
        return None
    artifact_mgr = host.get_artifact_manager()
    bundle = artifact_mgr.release_store.bundle(release_id)
    if bundle is None or bundle.release.state != "accepted":
        raise RuntimeError(f"Release {release_id} is not an accepted release of this notebook")
    (output_dir / RESTORE_DIR).mkdir(exist_ok=True)
    restore: dict[str, Any] = {
        "release_id": release_id,
        "sequence": bundle.release.sequence if run.resumed_from else 0,
    }
    files = checkpoint_files.files_of(bundle.outputs)
    if files:
        folder = f"{RESTORE_DIR}/{CHECKPOINTS_DIR}"
        for rel, version in files.items():
            target = output_dir / folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            artifact_mgr.copy_ready_blob(version.id, version.version, target)
        restore["folder"] = folder
    parts: dict[str, dict[str, str]] = {}
    for version in bundle.outputs:
        if checkpoint_files.FILE in version.params or version.params.get(checkpoint_files.MANIFEST):
            continue
        match = _OUTPUT_VARIABLE.search(version.id)
        if match is None:
            continue
        name = match.group(1)
        content_type = version.params.get("content_type") or "pickle/object"
        ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".pickle")
        file_name = f"{RESTORE_DIR}/{safe_filename_stem(name)}{ext}"
        write_ready_input_file(output_dir / file_name, version.id, version.version, content_type)
        parts[name] = {
            "content_type": content_type,
            "file": file_name,
            "uri": f"strata://artifact/{version.id}@v={version.version}",
        }
    return {**restore, "parts": parts}


async def run_python_cell(
    host: ExecutorHost,
    context: PythonContext,
    cell_id: str,
    source: str,
    timeout_seconds: float,
    start_time: float,
    *,
    materialize_upstreams: bool,
    use_cache: bool,
    fanout_variant: str | None = None,
    cell_provenance: _CellProvenance | None = None,
) -> CellExecutionResult:
    """Python cell pipeline: provenance, upstreams, cache check, harness, persist.

    A top-level call on a ``# @per_variant`` cell redirects to
    :meth:`_execute_fanout_cell`, which calls back once per variant with
    ``fanout_variant`` set; provenance, cache, inputs and artifacts are then
    scoped to that variant.
    """
    remote_metadata: dict[str, str] = {}
    exec_result: CellExecutionResult | None = None
    # Unset in the frame that dispatches a fan-out: each instance keeps what its worker
    # uploaded under its own key, so that frame has nothing to keep.
    provenance_hash: str | None = None
    env_hash = ""
    # Nothing a worker uploaded for an earlier run belongs to this one.
    context.received_output_files.pop(cell_id, None)
    context.folder_listings.pop(cell_id, None)
    try:
        # Re-fetch the cell: the dispatch wrapper already reset ``cache_hit``, but the rest of
        # the pipeline needs ``cell`` in scope.
        cell = host.notebook_state.get_cell(cell_id)

        # A @per_variant cell fans out: the orchestrator re-enters this pipeline once per
        # variant with fanout_variant set.
        if fanout_variant is None:
            if host.dag is not None and cell_id in host.dag.each_cells:
                return await context.execute_each_cell(
                    cell_id,
                    source,
                    timeout_seconds,
                    start_time,
                    materialize_upstreams=materialize_upstreams,
                    use_cache=use_cache,
                )
            variant_names = context.fanout_variants(cell_id)
            if variant_names is not None:
                return await context.execute_fanout_cell(
                    cell_id,
                    source,
                    timeout_seconds,
                    start_time,
                    materialize_upstreams=materialize_upstreams,
                    use_cache=use_cache,
                    variant_names=variant_names,
                )

        # ① Materialize every upstream whose artifact is missing (recursive; each upstream
        # miss executes its own upstreams).
        if materialize_upstreams:
            await context.materialize_upstreams(cell_id)

        # ② Provenance. Every cell-kind path uses the same helper so the hash agrees with
        # ``compute_staleness`` on re-open.
        # A fan-out computes the cell's provenance once (``cell_provenance``); each
        # instance keys on the inputs it reads.
        prov = cell_provenance or await context.compute_provenance(cell_id, source)
        each_item = context.each_items.get(cell_id, {}).get(fanout_variant or "")
        if fanout_variant is not None:
            prov = prov.for_instance(each_item.instance if each_item else fanout_variant)
        source_hash = prov.source_hash
        runtime_env = prov.runtime_env
        effective_worker = prov.effective_worker
        env_hash = prov.env_hash
        input_hashes = prov.input_hashes
        mount_specs = prov.mount_specs
        mount_fingerprints = prov.mount_fingerprints
        init = context.branch_inits.get(cell_id)

        def run_key(prov: _CellProvenance) -> str:
            # An instance's key is scoped to its variant, so each caches independently.
            key = prov.provenance_hash
            # A branch is a new computation: its initial state is bound, never defined.
            if init is not None:
                key = derive_subkey(key, f"init={init}")
            return key

        provenance_hash = run_key(prov)

        # RW mounts have side effects: not cacheable. A rerun bypasses the cache but still
        # records a cacheable cell's console, or the next plain run replays an older one.
        cacheable = not prov.has_rw_mount and not prov.annotations.nocache
        if not cacheable:
            use_cache = False

        worker_spec = resolve_worker_spec(
            host.notebook_state,
            effective_worker,
        )
        remote_metadata = context.remote_execution_metadata(worker_spec)

        logger.info(
            "execute_cell %s: source_hash=%s env_hash=%s "
            "input_hashes=%s mount_fps=%s table_fps=%s provenance=%s",
            cell_id,
            source_hash[:12],
            env_hash[:12],
            [h[:12] for h in input_hashes],
            [fp[:20] for fp in mount_fingerprints],
            [fp[:40] for fp in prov.table_fingerprints],
            provenance_hash[:12],
        )

        # A fetch that could not be checked, or whose bytes differ from its pin, fails the
        # cell before it runs: a cache hit would claim unverified bytes had not moved.
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
        # Likewise a dataset the registry could not resolve or hand over.
        if prov.dataset_error is not None:
            return refused_run(
                host,
                CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=prov.dataset_error,
                    execution_method="error",
                ),
            )
        # Locally each fetch is a read-only mount of its cached bytes. A remote worker can't
        # see this machine's paths, so there the bytes travel as inputs (``_add_fetch_inputs``).
        fetches_as_inputs = bool(prov.fetched) and is_http_executor_worker(worker_spec)
        if not fetches_as_inputs:
            mount_specs = [
                *mount_specs,
                *(
                    MountSpec(name=name, uri=path.resolve().as_uri(), mode=MountMode.READ_ONLY)
                    for name, path in prov.fetched.items()
                ),
            ]

        # Declared lake tables must resolve to snapshots before the cell runs (injection
        # needs them).
        try:
            manifest_tables = context.manifest_tables(prov.annotations.tables, prov.table_snapshots)
        except RuntimeError as e:
            return refused_run(
                host,
                CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=str(e),
                    execution_method="error",
                ),
            )

        # ③ Cache check for THIS cell.
        artifact_mgr = host.get_artifact_manager()
        consumed_vars = host.dag.consumed_variables.get(cell_id, set()) if host.dag else set()

        cached_artifact = None
        if cell is not None:
            current_display_outputs = cell.display_outputs or (
                [cell.display_output] if cell.display_output is not None else []
            )
        else:
            current_display_outputs = []
        # Gated on ``use_cache`` like the artifact lookup: replaying a leaf's display is a
        # cache hit, so ``# @nocache``, rw mounts, force and rerun would otherwise skip the
        # effect they exist to re-trigger.
        cached_display_outputs = (
            host.resolve_cached_display_outputs(
                cell_id,
                provenance_hash,
                current_display_outputs,
            )
            if (cell is not None and use_cache)
            else []
        )
        # A leaf cell has no artifact, but its stdout/stderr are stored by provenance; replay
        # them so an unchanged re-run is instant.
        cached_console = (
            host.resolve_cached_console(cell_id, provenance_hash, variant=fanout_variant)
            if (cell is not None and use_cache)
            else None
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
                variant=fanout_variant,
            )

        if use_cache and cached_artifact is not None and consumed_vars:
            cached_artifact, cached_display_outputs = context.validate_output_hit(
                cell,
                cell_id,
                provenance_hash,
                consumed_vars,
                # An @each instance writes no table: its cell writes once, over them all.
                [] if each_item else prov.annotations.table_writes,
                cached_artifact,
                current_display_outputs,
                cached_display_outputs,
                fanout_variant,
            )
            # The hit may have just promoted it.
            cached_console = host.resolve_cached_console(
                cell_id, provenance_hash, variant=fanout_variant
            )

        # A leaf's record is its console and displays; a revert finds them under an older
        # version, promoted like a consumed variable's above. Its displays alone may match:
        # the run it reverts over could have made none.
        if (
            use_cache
            and cell is not None
            and not consumed_vars
            and cached_console is None
            and context.promote_reverted_leaf(
                cell_id, provenance_hash, len(current_display_outputs)
            )
        ):
            cached_display_outputs = host.resolve_cached_display_outputs(
                cell_id, provenance_hash, current_display_outputs
            )
            cached_console = host.resolve_cached_console(cell_id, provenance_hash)

        logger.info(
            "execute_cell %s: consumed_vars=%s use_cache=%s cache_hit=%s",
            cell_id,
            consumed_vars,
            use_cache,
            cached_artifact is not None
            or bool(cached_display_outputs)
            or cached_console is not None,
        )

        if cached_artifact is not None or (
            not consumed_vars and (cached_display_outputs or cached_console)
        ):
            # A fan-out writes back for its instances, once they all have run.
            if cached_artifact is not None and fanout_variant is None:
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
            if remote_metadata.get("remote_transport") == "signed":
                remote_metadata.setdefault("remote_build_state", "ready")
            duration_ms = (time.time() - start_time) * 1000
            if cell:
                cell.cache_hit = True
                cell.display_outputs = list(cached_display_outputs)
                cell.display_output = cached_display_outputs[-1] if cached_display_outputs else None
                for var_name in consumed_vars:
                    canonical_id = artifact_mgr.cell_artifact_id(
                        cell_id, var_name, variant=fanout_variant
                    )
                    canonical_art = artifact_mgr.artifact_store.get_latest_version(
                        canonical_id,
                    )
                    if canonical_art:
                        uri = f"strata://artifact/{canonical_art.id}@v={canonical_art.version}"
                        cell.artifact_uris[var_name] = uri
                        cell.artifact_uri = uri  # backward compat
            cached_result = CellExecutionResult(
                cell_id=cell_id,
                success=True,
                outputs={},
                # A fan-out says this of the cell, once it has every instance's.
                **hit_console(
                    cached_console,
                    shows_this_key=fanout_variant is not None
                    or (cell is not None and cell.last_provenance_hash == provenance_hash),
                ),
                display_outputs=[output.model_dump() for output in cached_display_outputs],
                display_output=(
                    cached_display_outputs[-1].model_dump() if cached_display_outputs else None
                ),
                duration_ms=duration_ms,
                cache_hit=True,
                artifact_uri=(
                    (f"strata://artifact/{cached_artifact.id}@v={cached_artifact.version}")
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
            # A variant's hit is recorded by its fan-out, once every variant succeeded.
            if fanout_variant is None:
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

        # ④ Cache miss: execute the cell.
        refusal = context.table_write_refusal(source)
        if refusal is not None:
            return refused_run(
                host,
                CellExecutionResult(
                    cell_id=cell_id, success=False, error=refusal, execution_method="error"
                ),
            )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.TemporaryDirectory() as result_tmpdir,
        ):
            output_dir = Path(tmpdir)
            remote_build_id = (
                f"nbbuild-{uuid.uuid4().hex[:12]}"
                if worker_spec is not None and worker_transport(worker_spec) == "signed"
                else None
            )
            remote_metadata = context.remote_execution_metadata(
                worker_spec,
                remote_build_id=remote_build_id,
            )

            # Force may skip upstream materialization, so missing inputs are allowed to surface
            # at execution time. Off the loop: an input folder can hold gigabytes.
            input_specs = await anyio.to_thread.run_sync(
                functools.partial(
                    context.load_input_blobs,
                    cell_id,
                    output_dir,
                    fanout_variant=fanout_variant,
                )
            )
            if fetches_as_inputs:
                _add_fetch_inputs(input_specs, prov.fetched, output_dir)
            await anyio.to_thread.run_sync(
                context.add_dataset_inputs, input_specs, prov.datasets, output_dir
            )

            venv_path = host.venv_python

            # Mounts this server prepares are prepared before the run is keyed: it reads
            # this listing, which the one its key came from may have moved past.
            prepared_mounts: dict[str, ResolvedMount] | None = None
            if worker_spec is not None and (
                worker_spec.backend == WorkerBackendType.LOCAL
                or is_embedded_executor_worker(worker_spec)
            ):
                # An item's mount was listed for the run, not again per instance.
                listed = context.each_listed.get(cell_id, {}) if each_item else {}
                prepared_mounts = {
                    **listed,
                    **await context.prepare_mounts(
                        [m for m in mount_specs if m.name not in listed], cell_id
                    ),
                }
                prov = prov.over_mounts(prepared_mounts)
                provenance_hash = run_key(prov)
                if each_item is not None and each_item.path is not None:
                    # A file binds as a one-file mount: a Path, read-only, in every harness.
                    each_spec = parse_annotations(source).each
                    assert each_spec is not None  # it has an item
                    prepared_mounts[each_spec["name"]] = ResolvedMount(
                        spec=MountSpec(name=each_spec["name"], uri=each_item.path.as_uri()),
                        local_path=each_item.path,
                        fingerprint=each_item.digest,
                    )

            # A local run is an execution of the ledger: a build with an attempt whose
            # lease this process renews, open until its outputs are stored.
            local_run: LocalRun | None = None
            if worker_spec is not None and worker_spec.backend == WorkerBackendType.LOCAL:
                try:
                    local_run = await anyio.to_thread.run_sync(
                        functools.partial(
                            context.local_runs().open,
                            cell_id=cell_id,
                            computation_key=provenance_hash,
                            variant=fanout_variant,
                            input_versions=context.input_refs(cell_id, fanout_variant),
                            timeout_seconds=timeout_seconds,
                            resume=use_cache,
                            init=init,
                        )
                    )
                except RunRefused as exc:
                    return refused_run(
                        host,
                        CellExecutionResult(
                            cell_id=cell_id,
                            success=False,
                            error=str(exc),
                            execution_method="error",
                        ),
                    )
                context.open_runs[cell_id] = (
                    local_run,
                    context.local_runs().start_renewing(local_run),
                )
                restore = await anyio.to_thread.run_sync(
                    context.restore_manifest, local_run, output_dir
                )
            else:
                restore = None

            (
                result,
                result_output_dir,
                execution_method,
                resolved_mounts,
            ) = await context.dispatch_execution(
                worker_spec,
                source,
                input_specs,
                mount_specs,
                output_dir,
                Path(result_tmpdir),
                venv_path,
                runtime_env,
                timeout_seconds,
                remote_build_id=remote_build_id,
                mutation_defines=list(getattr(cell, "mutation_defines", []) or []),
                tables=manifest_tables,
                cell_id=cell_id,
                cell_provenance_hash=provenance_hash,
                local_run=local_run,
                restore=restore,
                resolved_mounts=prepared_mounts,
            )
            if remote_build_id and remote_metadata.get("remote_transport") == "signed":
                remote_metadata["remote_build_state"] = "ready"

            duration_ms = (time.time() - start_time) * 1000
            exec_result = context.parse_result(
                cell_id,
                result,
                duration_ms,
                execution_method,
            ).apply_remote_metadata(**remote_metadata)
            collected = await context.collect_output_dirs_off_loop(
                cell_id, source, result_output_dir
            )
            if not exec_result.success:
                context.keep_failed_output_dirs(
                    cell_id,
                    source,
                    result_output_dir,
                    provenance_hash,
                    env_hash,
                    collected,
                    variant=fanout_variant,
                )

            # ⑤ Store artifacts for consumed variables.
            if exec_result.success:
                module_export_error = write_module_export_outputs(
                    host,
                    cell_id,
                    source,
                    result_output_dir,
                    provenance_hash,
                    exec_result.outputs,
                )
                if module_export_error is not None:
                    exec_result = CellExecutionResult(
                        cell_id=cell_id,
                        success=False,
                        stdout=exec_result.stdout,
                        stderr=exec_result.stderr,
                        outputs=exec_result.outputs,
                        duration_ms=exec_result.duration_ms,
                        error=module_export_error,
                        execution_method=exec_result.execution_method,
                        mutation_warnings=exec_result.mutation_warnings,
                    ).apply_remote_metadata(**remote_metadata)

            if exec_result.success:
                store_error: str | None = None
                try:
                    stored_ok = context.store_outputs(
                        cell_id,
                        result_output_dir,
                        provenance_hash,
                        input_hashes,
                        source_hash=source_hash,
                        source=source,
                        env_hash=env_hash,
                        variant=fanout_variant,
                        # Reported by whatever ran the cell (venv or remote worker); this
                        # process may not be on that machine.
                        build_env=str(result.get("build_env") or ""),
                        hardware=result.get("hardware") or {},
                        # What a teammate is told they saved; they never ran the cell, so it
                        # travels with the bytes.
                        build_duration_ms=exec_result.duration_ms,
                        collected=collected,
                        local_run=local_run,
                        stochastic=prov.annotations.stochastic,
                        run_outputs=exec_result.outputs,
                    )
                except (UnsafeRunFile, TableWriteError, OutputNotSaved) as exc:
                    stored_ok, store_error = False, str(exc)
                await context.close_open_run(cell_id)
                if not stored_ok:
                    logger.error(
                        "Cell %s executed OK but artifact storage failed.",
                        cell_id,
                    )
                    exec_result = _store_failure_result(
                        exec_result, store_error
                    ).apply_remote_metadata(**remote_metadata)

                if stored_ok:
                    # Only once artifacts are on disk: the publish reads them back, so a pull
                    # reproduces exactly what a local run stored.
                    await context.push_to_team_store(cell_id=cell_id, variant=fanout_variant)

                if exec_result.success:
                    context.store_result_displays(
                        exec_result,
                        result_output_dir,
                        provenance_hash,
                        input_hashes,
                        source_hash=source_hash,
                        source=source,
                        env_hash=env_hash,
                    )
                    # Cache the cell's console by provenance so a hit replays this run's.
                    if cacheable:
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
                            variant=fanout_variant,
                        )

                # ⑥ Sync back read-write mounts. A worker syncs its own and reports what
                # was written in its result.
                mount_writes = _worker_mount_writes(result, mount_specs)
                if exec_result.success and resolved_mounts:
                    try:
                        mount_writes = await context.sync_back_mounts(resolved_mounts)
                    except Exception as exc:
                        logger.exception(
                            "Failed to sync-back RW mounts for cell %s",
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
                    if fanout_variant is not None:
                        # Recorded by the fan-out, once every variant has succeeded.
                        context.variant_writes.setdefault(cell_id, []).extend(mount_writes)
                    else:
                        context.record_good_run(
                            cell_id, provenance_hash, source_hash, env_hash, prov, mount_writes
                        )

            host.persist_display_outputs(
                cell_id,
                exec_result.display_outputs if exec_result.success else None,
            )
            host.apply_execution_result_metadata(cell_id, exec_result)
            return exec_result

    except asyncio.CancelledError:
        if provenance_hash is not None and context.received_output_files.get(cell_id):
            # Cancelled mid-run: keep what the worker already uploaded, as on a loss.
            context.keep_failed_output_dirs(
                cell_id, source, None, provenance_hash, env_hash, variant=fanout_variant
            )
        raise
    except RemoteExecutionError as e:
        duration_ms = (time.time() - start_time) * 1000
        if provenance_hash is not None and context.received_output_files.get(cell_id):
            # No result came back, so what the worker uploaded is all there is.
            context.keep_failed_output_dirs(
                cell_id, source, None, provenance_hash, env_hash, variant=fanout_variant
            )
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
        if provenance_hash is not None and context.received_output_files.get(cell_id):
            # No result came back, so what the worker uploaded is all there is.
            context.keep_failed_output_dirs(
                cell_id, source, None, provenance_hash, env_hash, variant=fanout_variant
            )
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
        if provenance_hash is not None and context.received_output_files.get(cell_id):
            # No result came back, so what the worker uploaded is all there is.
            context.keep_failed_output_dirs(
                cell_id, source, None, provenance_hash, env_hash, variant=fanout_variant
            )
        # A step after the run failed: what the run printed still stands.
        error_result = CellExecutionResult(
            cell_id=cell_id,
            success=False,
            stdout=exec_result.stdout if exec_result is not None else "",
            stderr=exec_result.stderr if exec_result is not None else "",
            duration_ms=duration_ms,
            error=f"Execution failed: {e}",
            mutation_warnings=exec_result.mutation_warnings if exec_result is not None else [],
        ).apply_remote_metadata(**remote_metadata)
        host.persist_display_outputs(cell_id, None)
        host.apply_execution_result_metadata(cell_id, error_result)
        return error_result


async def dispatch_local(
    host: ExecutorHost,
    context: LocalDispatchContext,
    source: str,
    input_specs: dict[str, dict[str, str]],
    mount_specs: list[MountSpec],
    output_dir: Path,
    venv_path: Path | None,
    runtime_env: dict[str, str],
    timeout_seconds: float,
    mutation_defines: list[str] | None = None,
    tables: dict[str, dict[str, Any]] | None = None,
    cell_id: str | None = None,
    local_run: LocalRun | None = None,
    restore: dict[str, Any] | None = None,
    resolved_mounts: dict[str, ResolvedMount] | None = None,
) -> tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]:
    """Run the direct local execution path, on *resolved_mounts* if already prepared."""
    result = None
    execution_method = "cold"
    if resolved_mounts is None:
        resolved_mounts = await context.prepare_mounts(mount_specs, cell_id)
    on_event = (
        context.local_runs().event_handler(local_run, output_dir) if local_run is not None else None
    )
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
            restore=restore,
        )

        # Without an interpreter nothing runs, not even in a pre-started warm process:
        # _run_harness refuses below.
        if context.pool is not None and venv_path is not None:
            from strata.notebook.pool import PooledCellExecutor

            try:
                pool_result = await PooledCellExecutor.execute_with_pool(
                    context.pool,
                    manifest_path,
                    host.path,
                    timeout_seconds,
                    run_token=run_token,
                    on_event=on_event,
                )
            except RunFenced as exc:
                return _fenced_result(exc), output_dir, "warm", resolved_mounts
            if pool_result is not None:
                result = pool_result
                execution_method = "warm"
                logger.debug(
                    "Executed cell %s with warm process",
                    manifest_path.parent.name,
                )

    if result is None:
        # A token of its own: a warm process that died late in the cell spent the first
        # one's time, and anything it left running keeps no live token.
        with context.run_token(timeout_seconds) as run_token:
            result = await context.run_harness(
                manifest_path,
                venv_path,
                timeout_seconds,
                run_token=run_token,
                on_event=on_event,
            )

    return result, output_dir, execution_method, resolved_mounts


async def run_harness(
    host: ExecutorHost,
    harness_command: Callable[..., list[str] | None],
    harness_env: Callable[[dict[str, str] | None], dict[str, str] | None],
    manifest_path: Path,
    venv_python: Path | None,
    timeout_seconds: float,
    *,
    run_token: str | None = None,
    on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Run the harness via uv, or as the harness user; no-op if ``venv_python`` is None.

    A *run_token* goes to the harness on its stdin. With *on_event*, the harness
    gets a pipe for its event lines (``run_channel``), relayed there in order.
    """
    if venv_python is None:
        return {
            "success": False,
            "error": _no_interpreter_message(host),
            "stderr": "",
            "stdout": "",
            "variables": {},
        }
    try:
        harness_user = resolve_harness_user()
    except LocalExecutionRefused as exc:
        return _refused_result(exc)
    cmd = harness_command(manifest_path, venv_python, harness_user)
    if cmd is None:
        # Otherwise every cell dies with a bare ``[Errno 2] ... 'uv'`` (common headless, where
        # ~/.local/bin isn't on PATH). Mirrors the Rscript guard in _run_r_harness.
        return {
            "success": False,
            "error": UV_NOT_FOUND_MESSAGE,
            "stderr": "",
            "stdout": "",
            "variables": {},
        }

    # New process group so cancel can signal the whole descendant tree (DataLoader
    # workers, multiprocessing pools); ``proc.kill()`` alone leaks children.

    from strata.notebook.process_tree import (
        subprocess_kwargs_for_new_group,
        terminate_subprocess_tree,
    )

    hand_over(manifest_path.parent, harness_user)
    lifeline_fds, lifeline_vars = lifeline_handoff()
    # The event pipe: the harness inherits the write end and keeps it from its
    # children, so EOF on the read end is the harness exiting. None on Windows,
    # which passes no descriptors.
    events_fds: tuple[int, ...] = ()
    events_vars: dict[str, str] = {}
    read_fd = -1
    if on_event is not None and sys.platform != "win32":
        read_fd, write_fd = os.pipe()
        events_fds = (write_fd,)
        events_vars = {EVENTS_FD_ENV: str(write_fd)}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(host.path),
            stdin=asyncio.subprocess.PIPE if run_token is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # ``uv run`` must find the notebook's .venv, not the server's environment.
            env=uv_env(identity_env(harness_env({**lifeline_vars, **events_vars}), harness_user)),
            pass_fds=(*lifeline_fds, *events_fds),
            **spawn_kwargs(harness_user),
            **subprocess_kwargs_for_new_group(),
        )
    except BaseException:
        if read_fd >= 0:
            os.close(read_fd)
        raise
    finally:
        for fd in events_fds:
            os.close(fd)
    relay = (
        asyncio.create_task(relay_events(read_fd, on_event, proc))
        if events_fds and on_event is not None
        else None
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(None if run_token is None else f"{run_token}\n".encode()),
            timeout=timeout_seconds,
        )
    except asyncio.CancelledError:
        logger.info(
            "Cell execution cancelled; terminating harness subprocess tree pid=%s",
            proc.pid,
        )
        if relay is not None:
            relay.cancel()
        try:
            await asyncio.shield(terminate_subprocess_tree(proc))
        except Exception:
            logger.exception(
                "Failed to terminate cancelled harness subprocess tree pid=%s",
                proc.pid,
            )
        raise
    except TimeoutError:
        if relay is not None:
            relay.cancel()
        await terminate_subprocess_tree(proc)
        raise TimeoutError()
    # The harness has exited, so every line it wrote is in the pipe; EOF follows
    # unless a child of the cell still holds the write end, which the harness kept
    # from them, so a short wait is the guard, not the mechanism.
    fenced = None
    if relay is not None:
        try:
            fenced = await asyncio.wait_for(relay, timeout=EVENTS_DRAIN_SECONDS)
        except TimeoutError:
            logger.warning(
                "Event pipe of cell run pid=%s still open after exit; stopped reading",
                proc.pid,
            )
    if fenced is not None:
        return _fenced_result(fenced)

    # Separate from manifest.json and user files like result.json. Absent means the
    # harness crashed before its finally block (typically an import error): surface
    # stderr instead of "Unknown error".
    result_path = manifest_path.parent / "harness-result.json"
    if not result_path.exists():
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        return {
            "success": False,
            "error": (stderr_text.strip() or "Harness exited without producing a result manifest"),
            "stderr": stderr_text,
            "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
            "variables": {},
        }
    return json.loads(read_run_file(result_path.parent, result_path.name))
