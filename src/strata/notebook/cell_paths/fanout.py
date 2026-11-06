"""``# @per_variant`` and ``# @each``: one Python run per instance, then the cell's result."""

from __future__ import annotations

import functools
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio
from vestigo.provenance import derive_subkey
from vestigo.store import ArtifactVersion, StagedVersion

from strata.notebook import each, output_dirs
from strata.notebook.annotations import parse_annotations
from strata.notebook.cell_paths.context import _writes_entry
from strata.notebook.execution_result import CellExecutionResult, hit_console
from strata.notebook.models import MountMode, TableWriteSpec, WorkerBackendType
from strata.notebook.mounts import MountWrite, parse_mount_uri
from strata.notebook.table_writes import StagedOutput, TableWriteError, write_back_failed
from strata.notebook.workers import resolve_worker_spec

if TYPE_CHECKING:
    from strata.notebook.executor import _CellProvenance
    from strata.notebook.executor_host import ExecutorHost
    from strata.notebook.mounts import ResolvedMount

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FanoutContext:
    """What running a cell's instances asks of the executor, and nothing else."""

    materialize_upstreams: Callable[[str], Awaitable[None]]
    compute_provenance: Callable[..., Awaitable[_CellProvenance]]
    # An instance is a Python run under the cell's base key.
    execute_python_cell: Callable[..., Awaitable[CellExecutionResult]]
    moved: Callable[[str], tuple[dict[str, int | None], dict[str, int]]]
    promote_reverted_outputs: Callable[..., int | None]
    gone_tables: Callable[[str], Awaitable[list[TableWriteSpec]]]
    # (cell id, specs, variant): a hit's stored outputs into those tables.
    write_back: Callable[..., None]
    store_console_outputs: Callable[..., None]
    prepare_mounts: Callable[..., Awaitable[dict[str, ResolvedMount]]]
    input_refs: Callable[[str], dict[str, str]]
    write_cell_tables: Callable[..., None]
    # Cross-path state the Python path reads or adds to while an instance runs, by cell id.
    variant_writes: dict[str, list[MountWrite]]
    each_items: dict[str, dict[str, each.EachItem]]
    each_listed: dict[str, dict[str, ResolvedMount]]
    branch_inits: dict[str, str]
    # Read as the run starts; the WebSocket handler and the CLI set it before they run a cell.
    on_variant_complete: Callable[[dict[str, Any]], Awaitable[None]] | None


def _each_files(spec: dict[str, str] | None) -> dict[str, str] | None:
    """What of its mount an ``@each`` reads, as ``only``: the source's glob by its name."""
    if spec is None or not spec.get("glob"):
        return None
    return {spec["source"]: spec["glob"]}


def _merge_variant_writes(writes: list[MountWrite]) -> list[MountWrite]:
    """A fan-out's writes, one per mount; a path two variants wrote keeps the later's record."""
    merged: dict[str, MountWrite] = {}
    for write in writes:
        seen = merged.setdefault(
            write.mount, MountWrite(mount=write.mount, uri=write.uri, files=[])
        )
        files = {f.path: f for f in [*seen.files, *write.files]}
        seen.files = sorted(files.values(), key=lambda f: f.path)
    return list(merged.values())


async def run_fanout_cell(
    host: ExecutorHost,
    context: FanoutContext,
    cell_id: str,
    source: str,
    timeout_seconds: float,
    start_time: float,
    *,
    materialize_upstreams: bool,
    use_cache: bool,
    variant_names: tuple[str, ...],
) -> CellExecutionResult:
    """Run a ``# @per_variant`` cell once per variant of its fan-out group.

    Each instance binds that variant's scalar and stores ``@variant={name}``
    artifacts, which a downstream consumer collapses back to a
    ``{variant: value}`` dict. Succeeds iff every variant succeeds.
    Upstreams are materialised once, not per variant.
    """
    # Each variant writes the same table: an overwrite or sync would keep only the last's.
    replacing = [
        spec
        for spec in parse_annotations(source).table_writes
        if spec.mode in ("overwrite", "sync")
    ]
    if replacing:
        return CellExecutionResult(
            cell_id=cell_id,
            success=False,
            error=(
                f"@write_table {replacing[0].table} {replacing[0].mode} is not supported on "
                "a @per_variant cell: each variant would remove the rows of the one before "
                "it. Append, merge on a column that tells the variants apart, or write from "
                "a cell that reads the {variant: value} dict"
            ),
            execution_method="fanout",
        )
    if materialize_upstreams:
        await context.materialize_upstreams(cell_id)
    # Mounts, tables, fetches and the env are the cell's, not a variant's: resolved once,
    # so every instance sees the same snapshot and a mount is listed once.
    base = await context.compute_provenance(cell_id, source)
    # Before any variant runs: the first to run brings the table back for the rest.
    gone = await context.gone_tables(source) if use_cache else []

    results = await _run_fanout_instances(
        host,
        context,
        cell_id,
        source,
        timeout_seconds,
        start_time,
        use_cache=use_cache,
        base=base,
        instances=[(name, name) for name in variant_names],
    )
    # A variant that ran wrote its rows. One that hit writes its stored ones, even when
    # another failed: the next run finds the table there and would never write them.
    error = None
    if gone:
        try:
            for name, result in results:
                if result.success and result.cache_hit:
                    await anyio.to_thread.run_sync(context.write_back, cell_id, gone, name)
        except TableWriteError as exc:
            error = write_back_failed(exc)
    return _finish_fanout(
        host, context, cell_id, results, base, start_time, error=error, shows_displays=True
    )


async def _run_fanout_instances(
    host: ExecutorHost,
    context: FanoutContext,
    cell_id: str,
    source: str,
    timeout_seconds: float,
    start_time: float,
    *,
    use_cache: bool,
    base: _CellProvenance,
    instances: list[tuple[str, str]],
) -> list[tuple[str, CellExecutionResult]]:
    """Run each ``(instance name, what to call it)`` in sequence; report as each completes."""
    results: list[tuple[str, CellExecutionResult]] = []
    context.variant_writes[cell_id] = []
    total = len(instances)
    for index, (name, label) in enumerate(instances):
        variant_start = time.time()
        res = await context.execute_python_cell(
            cell_id,
            source,
            timeout_seconds,
            start_time,
            materialize_upstreams=False,
            use_cache=use_cache,
            fanout_variant=name,
            cell_provenance=base,
        )
        results.append((label, res))
        if context.on_variant_complete is not None:
            try:
                await context.on_variant_complete(
                    {
                        "cell_id": cell_id,
                        "variant": label,
                        "index": index,
                        "total": total,
                        "success": res.success,
                        "duration_ms": int((time.time() - variant_start) * 1000),
                        "error": res.error,
                    }
                )
            except Exception:
                logger.exception(
                    "on_variant_complete callback failed for cell %s variant %s",
                    cell_id,
                    label,
                )
    return results


def _finish_fanout(
    host: ExecutorHost,
    context: FanoutContext,
    cell_id: str,
    results: list[tuple[str, CellExecutionResult]],
    base: _CellProvenance,
    start_time: float,
    *,
    error: str | None = None,
    instance_noun: str = "Fan-out variant",
    shows_displays: bool = False,
) -> CellExecutionResult:
    """The cell's result from its instances' (or *error*); record a good run's base key.

    With *shows_displays*, a good run shows every instance's displays in turn, each
    saying which it is.
    """
    writes = _merge_variant_writes(context.variant_writes.pop(cell_id, []))
    duration_ms = (time.time() - start_time) * 1000
    failures = [(name, r) for name, r in results if not r.success]
    stdout = "\n".join(f"[variant={name}]\n{r.stdout}" for name, r in results if r.stdout)
    stderr = "\n".join(f"[variant={name}]\n{r.stderr}" for name, r in results if r.stderr)
    failure_traceback = None
    failure_traceback_full = None
    if failures:
        first_name, first = failures[0]
        error = f"{instance_noun} '{first_name}' failed: {first.error}"
        # What a reader sees in place of ``error``, so it says which too.
        if first.traceback:
            failure_traceback = f"{instance_noun} '{first_name}' failed:\n{first.traceback}"
        failure_traceback_full = first.traceback_full
    if error is not None:
        aggregate = CellExecutionResult(
            cell_id=cell_id,
            success=False,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            error=error,
            traceback=failure_traceback,
            traceback_full=failure_traceback_full,
            execution_method="fanout",
        )
    else:
        # Cached only when no instance ran: one recompute makes the cell a run.
        cache_hit = bool(results) and all(r.cache_hit for _, r in results)
        cell = host.notebook_state.get_cell(cell_id)
        displays = (
            [{**output, "variant": name} for name, r in results for output in r.display_outputs]
            if shows_displays
            else []
        )
        aggregate = CellExecutionResult(
            cell_id=cell_id,
            success=True,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            display_outputs=displays,
            display_output=displays[-1] if displays else None,
            cache_hit=cache_hit,
            execution_method="fanout",
            # Before the record below moves the cell's last key on to this one.
            replaces_console=cache_hit
            and (
                all(r.replaces_console for _, r in results)
                or cell is None
                or cell.last_provenance_hash != base.provenance_hash
            ),
            # Instances run in turn, so their costs add. One unpriced leaves the sum unknown.
            hit_run_ms=(
                sum(r.hit_run_ms for _, r in results)
                if cache_hit and all(r.hit_run_ms for _, r in results)
                else 0
            ),
        )
        # Record the BASE provenance: compute_staleness recomputes the base hash, so a
        # variant-scoped value would leave the fan-out cell perpetually stale on re-open.
        try:
            host.record_successful_execution_provenance(
                cell_id,
                base.provenance_hash,
                base.source_hash,
                base.env_hash,
                table_reads=base.table_reads,
                reads=base.reads(*context.moved(cell_id)),
                key_parts=base.key_parts,
                writes=_writes_entry(writes),
            )
        except Exception:
            logger.exception("Failed to record fan-out base provenance for cell %s", cell_id)
    # After every instance's, which each showed only its own.
    host.persist_display_outputs(cell_id, aggregate.display_outputs if aggregate.success else None)
    host.apply_execution_result_metadata(cell_id, aggregate)
    return aggregate


async def run_each_cell(
    host: ExecutorHost,
    context: FanoutContext,
    cell_id: str,
    source: str,
    timeout_seconds: float,
    start_time: float,
    *,
    materialize_upstreams: bool,
    use_cache: bool,
) -> CellExecutionResult:
    """Run a ``# @each`` cell once per item, then store one manifest per output.

    Nothing changed is one probe of the cell's key: no item is listed. Otherwise each
    item keys on itself, so only changed and new ones run, and a removed one leaves
    by not being in the manifest. A failed instance stores no manifest.
    """

    def refused(error: str) -> CellExecutionResult:
        return CellExecutionResult(
            cell_id=cell_id, success=False, error=error, execution_method="fanout"
        )

    if materialize_upstreams:
        await context.materialize_upstreams(cell_id)
    # A release is one instance's, and the manifest's key does not carry a branch.
    if cell_id in context.branch_inits:
        return refused("Branching an @each cell from a release is not supported yet")
    annotations = parse_annotations(source)
    assert annotations.each is not None  # the cell is in the DAG's each_cells
    base = await context.compute_provenance(cell_id, source)
    worker = resolve_worker_spec(host.notebook_state, base.effective_worker)
    if worker is None or worker.backend != WorkerBackendType.LOCAL:
        return refused("@each runs on the local worker only")
    if base.has_rw_mount or annotations.nocache:
        use_cache = False

    assert host.dag is not None
    consumed = host.dag.consumed_variables.get(cell_id, set())
    if (
        use_cache
        and consumed
        and context.promote_reverted_outputs(
            cell_id, base.provenance_hash, consumed, annotations.table_writes, 0
        )
        is not None
    ):
        gone = await context.gone_tables(source)
        if gone:
            try:
                await anyio.to_thread.run_sync(_each_write_back, host, context, cell_id, gone)
            except (TableWriteError, each.EachError) as exc:
                return refused(write_back_failed(exc))
        return _each_cached(host, context, cell_id, consumed, base, start_time)

    # Each instance names itself the cell's output; a run with no manifest puts them back.
    cell = host.notebook_state.get_cell(cell_id)
    last_good = (dict(cell.artifact_uris), cell.artifact_uri) if cell is not None else None
    with output_dirs.assembling():
        try:
            try:
                items, base = await _list_each_items(host, context, cell_id, annotations.each, base)
            except each.EachError as exc:
                return refused(str(exc))
            if not items and any(w.mode == "sync" for w in annotations.table_writes):
                return refused(
                    f"@each over `{annotations.each['source']}` found no items, and syncing "
                    "that would delete every row; nothing was written"
                )
            context.each_items[cell_id] = {each.instance_name(item.key): item for item in items}
            results = await _run_fanout_instances(
                host,
                context,
                cell_id,
                source,
                timeout_seconds,
                start_time,
                use_cache=use_cache,
                base=base,
                instances=[(each.instance_name(item.key), item.key) for item in items],
            )
        finally:
            context.each_items.pop(cell_id, None)
            context.each_listed.pop(cell_id, None)
        error = None
        if all(result.success for _, result in results):
            displays = [
                (key, *host.parse_artifact_uri(output["artifact_uri"]))
                for key, result in results
                for output in result.display_outputs
                if output.get("artifact_uri")
            ]
            try:
                await anyio.to_thread.run_sync(
                    _store_each_manifests,
                    host,
                    context,
                    cell_id,
                    source,
                    items,
                    base,
                    sorted(consumed),
                    annotations.table_writes,
                    displays,
                )
            except (TableWriteError, each.EachError) as exc:
                error = str(exc)
        else:
            error = "an item failed"
    if error is not None and cell is not None and last_good is not None:
        cell.artifact_uris, cell.artifact_uri = last_good
    finished = _finish_fanout(
        host,
        context,
        cell_id,
        results,
        base,
        start_time,
        error=error,
        instance_noun="Item",
        shows_displays=True,
    )
    # The one probe that serves a later hit runs no instance, so it reads their console here.
    if finished.success and not finished.cache_hit and consumed and not annotations.nocache:
        context.store_console_outputs(
            cell_id,
            base.provenance_hash,
            finished.stdout,
            finished.stderr,
            base.input_hashes,
            source_hash=base.source_hash,
            source=source,
            env_hash=base.env_hash,
            duration_ms=finished.duration_ms,
        )
    return finished


def _each_write_back(
    host: ExecutorHost, context: FanoutContext, cell_id: str, specs: list[TableWriteSpec]
) -> None:
    """Write every item's stored rows, as its manifest names them, into *specs*' tables.

    Raises:
        TableWriteError: as for a run's write.
        each.EachError: a manifest or an item's value is not what a write needs.
    """
    manager = host.get_artifact_manager()
    store = manager.artifact_store
    outputs: dict[str, StagedOutput] = {}
    for spec in specs:
        manifest = store.get_latest_version(manager.cell_artifact_id(cell_id, spec.variable))
        if manifest is None:
            continue  # The write names it as not produced.
        versions = []
        for key, artifact_id, version in each.manifest_refs(
            manifest.id, manager.load_artifact_data(manifest.id, manifest.version)
        ):
            item = store.get_artifact(artifact_id, version)
            if item is None:
                raise each.EachError(f"item {key!r} is {artifact_id}@v={version}, which is gone")
            versions.append(item)
        outputs[spec.variable] = StagedOutput(
            artifact_id=manifest.id,
            version=manifest.version,
            blob=each.rows_of(manager, spec.variable, versions),
            content_type="arrow/ipc",
            provenance_hash=manifest.provenance_hash,
        )
    context.write_cell_tables(specs, outputs)


def _each_cached(
    host: ExecutorHost,
    context: FanoutContext,
    cell_id: str,
    consumed: set[str],
    base: _CellProvenance,
    start_time: float,
) -> CellExecutionResult:
    """A ``# @each`` cell whose key found its manifests: nothing to do."""
    manager = host.get_artifact_manager()
    cell = host.notebook_state.get_cell(cell_id)
    console = hit_console(
        host.resolve_cached_console(cell_id, base.provenance_hash),
        shows_this_key=cell is not None and cell.last_provenance_hash == base.provenance_hash,
    )
    manifests = []
    for var_name in sorted(consumed):
        manifest = manager.artifact_store.get_latest_version(
            manager.cell_artifact_id(cell_id, var_name)
        )
        if manifest is not None:
            manifests.append(manifest)
            if cell is not None:
                uri = f"strata://artifact/{manifest.id}@v={manifest.version}"
                cell.artifact_uris[var_name] = uri
                cell.artifact_uri = uri
    if cell is not None:
        cell.cache_hit = True
    # Every manifest names the same displays.
    displays = _each_displays(host, manifests[0]) if manifests else []
    host.record_successful_execution_provenance(
        cell_id,
        base.provenance_hash,
        base.source_hash,
        base.env_hash,
        table_reads=base.table_reads,
        reads=base.reads(*context.moved(cell_id)),
        key_parts=base.key_parts,
    )
    result = CellExecutionResult(
        cell_id=cell_id,
        success=True,
        **console,
        display_outputs=displays,
        display_output=displays[-1] if displays else None,
        duration_ms=(time.time() - start_time) * 1000,
        cache_hit=True,
        execution_method="cached",
    )
    host.persist_display_outputs(cell_id, displays)
    host.apply_execution_result_metadata(cell_id, result)
    return result


def _each_displays(host: ExecutorHost, manifest: ArtifactVersion) -> list[dict[str, Any]]:
    """What the items a manifest names displayed, each naming its item."""
    manager = host.get_artifact_manager()
    store = manager.artifact_store
    shown = []
    for key, artifact_id, version in each.manifest_displays(
        manager.load_artifact_data(manifest.id, manifest.version)
    ):
        artifact = store.get_artifact(artifact_id, version)
        if artifact is None:
            continue
        for output in host.stored_display_outputs([artifact]):
            shown.append({**output.model_dump(), "variant": key})
    return shown


async def _list_each_items(
    host: ExecutorHost,
    context: FanoutContext,
    cell_id: str,
    spec: dict[str, str],
    base: _CellProvenance,
) -> tuple[list[each.EachItem], _CellProvenance]:
    """The items of *spec*'s source now, and *base* over the listing they came from.

    Raises:
        each.EachError: the source cannot be iterated.
    """
    source = spec["source"]
    mount = next((m for m in base.mount_specs if m.name == source), None)
    if mount is not None:
        if "key" in spec:
            raise each.EachError(f"@each over the mount `{source}` takes glob=, not key=")
        if mount.mode != MountMode.READ_ONLY:
            raise each.EachError(f"@each over `{source}`: the mount must be read-only")
        prepared = await context.prepare_mounts(base.mount_specs, cell_id, only=_each_files(spec))
        # A read-write mount gets its own staging per instance.
        context.each_listed[cell_id] = {
            m.name: prepared[m.name] for m in base.mount_specs if m.mode == MountMode.READ_ONLY
        }
        scheme, _ = parse_mount_uri(mount.uri)
        items = await anyio.to_thread.run_sync(
            functools.partial(
                each.items_of_mount,
                prepared[source].local_path,
                spec.get("glob"),
                by_content=scheme != "file",
                versions=prepared[source].file_versions,
            )
        )
        return items, base.over_mounts(prepared)

    if "glob" in spec:
        raise each.EachError(f"@each over the variable `{source}` takes key=, not glob=")
    assert host.dag is not None
    producer = host.dag.read_producer(cell_id, source)
    if not isinstance(producer, str):
        raise each.EachError(f"@each over `{source}`: it is not a mount or a variable")
    items = await anyio.to_thread.run_sync(_each_items_of_variable, host, producer, spec)
    return items, base


def _each_items_of_variable(
    host: ExecutorHost, producer: str, spec: dict[str, str]
) -> list[each.EachItem]:
    """The items of the variable *producer* stored, without unpickling anything."""
    source = spec["source"]
    manager = host.get_artifact_manager()
    artifact = manager.artifact_store.get_latest_version(manager.cell_artifact_id(producer, source))
    if artifact is None:
        raise each.EachError(f"@each over `{source}`: its cell has not stored it")
    blob = manager.load_artifact_data(artifact.id, artifact.version)
    content_type = artifact.params.get("content_type")
    if content_type == "json/object":
        return each.items_of_json(json.loads(blob), source)
    if content_type == "arrow/ipc":
        from strata.artifact.table_export import arrow_table_from_ipc

        try:
            table = arrow_table_from_ipc(blob, f"{artifact.id}@v={artifact.version}")
        except ValueError as exc:
            raise each.EachError(f"@each over `{source}`: {exc}") from exc
        return each.items_of_table(table, spec.get("key"), source)
    raise each.EachError(
        f"@each over `{source}`: it is {content_type}; expected a dict, a list or a table"
    )


def _store_each_manifests(
    host: ExecutorHost,
    context: FanoutContext,
    cell_id: str,
    source: str,
    items: list[each.EachItem],
    base: _CellProvenance,
    variables: list[str],
    table_writes: list[TableWriteSpec],
    displays: list[tuple[str, str, int]],
) -> None:
    """One manifest per output naming each item's version and *displays*, then the table
    writes, then all of them current: a write that fails keeps none, as for any other run."""
    manager = host.get_artifact_manager()
    store = manager.artifact_store
    input_versions = context.input_refs(cell_id)
    written = {write.variable for write in table_writes}
    staged: list[StagedVersion] = []
    outputs: dict[str, StagedOutput] = {}
    try:
        for var_name in variables:
            versions = []
            for item in items:
                version = store.get_latest_version(
                    manager.cell_artifact_id(
                        cell_id, var_name, variant=each.instance_name(item.key)
                    )
                )
                if version is None:
                    raise each.EachError(f"@each item {item.key!r} did not produce `{var_name}`")
                versions.append((item.key, version))
            provenance = derive_subkey(base.provenance_hash, var_name)
            staged.append(
                manager.stage_cell_output(
                    cell_id=cell_id,
                    variable_name=var_name,
                    blob_data=each.encode_manifest(
                        manager.cell_artifact_id(cell_id, var_name),
                        [(k, v.id, v.version) for k, v in versions],
                        displays,
                    ),
                    content_type=each.CONTENT_TYPE,
                    provenance_hash=provenance,
                    input_versions=input_versions,
                    source_hash=base.source_hash,
                    source=source,
                    env_hash=base.env_hash,
                    computation_key=base.provenance_hash,
                )
            )
            if var_name in written:
                # One commit over every item's rows, cached items' too.
                outputs[var_name] = StagedOutput(
                    artifact_id=staged[-1].artifact_id,
                    version=staged[-1].version,
                    blob=each.rows_of(manager, var_name, [v for _, v in versions]),
                    content_type="arrow/ipc",
                    provenance_hash=provenance,
                )
        if table_writes:
            context.write_cell_tables(table_writes, outputs)
        stored = manager.finalize_cell_outputs(staged)
    except Exception:
        manager.discard_cell_outputs(staged)
        raise
    cell = host.notebook_state.get_cell(cell_id)
    if cell is not None:
        for version in stored:
            uri = f"strata://artifact/{version.id}@v={version.version}"
            cell.artifact_uris[version.params["variable_name"]] = uri
            cell.artifact_uri = uri
