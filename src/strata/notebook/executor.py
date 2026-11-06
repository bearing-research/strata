"""Cell executor: materialize a notebook cell.

``execute_cell`` recursively materializes upstream inputs, computes provenance
(``sha256(sorted_input_hashes + source_hash + env_hash)``), returns a cache hit
if one exists, and otherwise runs the harness and stores every consumed variable.
It works on any cell; the cascade planner only previews what will run.

This is deliberately a separate pipeline from Core's ``materialize`` SDK
(``POST /v1/materialize``): cells are in-process, multi-output and
source-as-transform. The two share only the artifact store
(``find_by_provenance`` / ``put``) and ``vestigo.provenance.derive_subkey``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import tempfile as tempfile
import time
from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import anyio.to_thread
import httpx
from vestigo.provenance import derive_subkey, run_output_key

# Stdlib json, not orjson: executor.py is imported via ``strata.config`` and must load
# with core deps only (orjson is in the ``[notebook]`` extra, absent from the Docker
# image). Batch frames are small dicts.
from vestigo.store import ArtifactVersion, StagedVersion

from strata.notebook import each, output_dirs
from strata.notebook.analyzer import imported_names
from strata.notebook.annotation_validation import blocking_diagnostic
from strata.notebook.annotations import CellAnnotations, LoopAnnotation, parse_annotations
from strata.notebook.cell_paths.batch import (
    BatchCellResult,
    BatchContext,
    BatchExecutionResult,
    run_batch,
)
from strata.notebook.cell_paths.batch import _drain_stream as _drain_stream
from strata.notebook.cell_paths.batch import _refused_batch as _refused_batch
from strata.notebook.cell_paths.cell_tests import (
    CellTestsContext,
    run_cell_tests,
)
from strata.notebook.cell_paths.cell_tests import (
    _merged_test_runs as _merged_test_runs,
)
from strata.notebook.cell_paths.cell_tests import (
    _variant_test_run as _variant_test_run,
)
from strata.notebook.cell_paths.context import (
    _ARTIFACT_EXT_BY_CONTENT_TYPE,
    DEFAULT_CELL_TIMEOUT_SECONDS,
    PathContext,
    cell_timeout_message,
    output_not_saved,
)
from strata.notebook.cell_paths.context import RemoteExecutionError as RemoteExecutionError
from strata.notebook.cell_paths.context import (
    _mount_sync_failure_result as _mount_sync_failure_result,
)
from strata.notebook.cell_paths.context import _refused_result as _refused_result
from strata.notebook.cell_paths.context import _store_failure_result as _store_failure_result
from strata.notebook.cell_paths.context import _worker_mount_writes as _worker_mount_writes
from strata.notebook.cell_paths.context import _writes_entry as _writes_entry
from strata.notebook.cell_paths.context import (
    artifact_content_type as _artifact_content_type,  # noqa: F401
)
from strata.notebook.cell_paths.context import effective_worker_name as effective_worker_name
from strata.notebook.cell_paths.context import (
    no_interpreter_message as _no_interpreter_message,  # noqa: F401
)
from strata.notebook.cell_paths.dispatch import _JOB_POLL_SECONDS as _JOB_POLL_SECONDS
from strata.notebook.cell_paths.dispatch import _QUEUE_DETAIL_MAX_CHARS as _QUEUE_DETAIL_MAX_CHARS
from strata.notebook.cell_paths.dispatch import (
    DispatchContext,
    DispatchRoutes,
    EmbeddedContext,
    await_accepted_job,
    broadcast_remote_phase,
    dispatch_embedded,
    dispatch_execution,
    dispatch_http,
    locked_environment,
    stage_signed_transport_inputs,
)
from strata.notebook.cell_paths.dispatch import _add_harness_params as _add_harness_params
from strata.notebook.cell_paths.dispatch import _monotonic as _monotonic
from strata.notebook.cell_paths.dispatch import _pack_folder_inputs as _pack_folder_inputs
from strata.notebook.cell_paths.dispatch import _queue_report as _queue_report
from strata.notebook.cell_paths.dispatch import _resolve_worker_token as _resolve_worker_token
from strata.notebook.cell_paths.fanout import FanoutContext, run_each_cell, run_fanout_cell
from strata.notebook.cell_paths.fanout import _each_files as _each_files
from strata.notebook.cell_paths.fanout import _merge_variant_writes as _merge_variant_writes
from strata.notebook.cell_paths.loop import LoopContext, run_loop_cell
from strata.notebook.cell_paths.loop import _loop_folders as _loop_folders
from strata.notebook.cell_paths.loop import _loop_run_token as _loop_run_token
from strata.notebook.cell_paths.prompt import run_prompt_cell
from strata.notebook.cell_paths.python import _OUTPUT_VARIABLE as _OUTPUT_VARIABLE
from strata.notebook.cell_paths.python import (
    LocalDispatchContext,
    PythonContext,
    dispatch_local,
    python_harness_command,
    restore_manifest,
    run_harness,
    run_python_cell,
)
from strata.notebook.cell_paths.python import _add_fetch_inputs as _add_fetch_inputs
from strata.notebook.cell_paths.python import _fenced_result as _fenced_result
from strata.notebook.cell_paths.r import RContext, run_r_cell, run_r_harness
from strata.notebook.cell_paths.sql import run_sql_cell
from strata.notebook.cell_paths.widget import run_widget_cell
from strata.notebook.credentials import CredentialResolver
from strata.notebook.dag import SweepProducer
from strata.notebook.env import compute_execution_env_hash, narrow_env_for_provenance
from strata.notebook.execution_result import CellExecutionResult
from strata.notebook.graph import analysis_with, mounts_of
from strata.notebook.harness_user import (
    HarnessUser,
    LocalExecutionRefused,
    UnsafeRunFile,
    hand_over,
    let_through,
    prepare_work_dir,
    read_run_file,
    resolve_harness_user,
)
from strata.notebook.keys import cell_key
from strata.notebook.layout import WORK_DIR
from strata.notebook.local_runs import (
    LocalRun,
    LocalRuns,
)
from strata.notebook.models import (
    CellLanguage,
    CellStatus,
    CellTestResult,
    DatasetSpec,
    FetchSpec,
    InputWeight,
    MountMode,
    MountSpec,
    TableSpec,
    TableWriteSpec,
    WorkerBackendType,
)
from strata.notebook.mount_cache import MountCache
from strata.notebook.mounts import (
    MountCredentials,
    MountResolver,
    MountWrite,
    ResolvedMount,
    ensure_writable,
    listing_entries,
    mount_component,
    parse_mount_uri,
    resolve_cell_mounts,
    weigh_mount_sync,
)
from strata.notebook.provenance import (
    compute_source_hash,
    safe_filename_stem,
)
from strata.notebook.quiesce import NotebookQuiesced
from strata.notebook.read_confirm import (
    ReadConfirmation,
    ReadEstimate,
    confirmation,
    estimate,
    input_fingerprint,
)
from strata.notebook.read_confirm import asks as read_confirm_asks
from strata.notebook.table_writes import (
    StagedOutput,
    TableWriteError,
    authorize_table_writes,
    gone_tables,
    write_cell_tables,
)
from strata.notebook.team_store import (
    TeamPull,
    TeamStore,
    publish_cell_outputs,
    pull_cell_outputs,
)
from strata.notebook.workers import (
    get_worker_execution_error,
    is_embedded_executor_worker,
    one_policy_read,
    read_worker_policy,
    resolve_worker_spec,
    worker_cached_bytes,
    worker_runtime_identity,
    worker_supports_notebook_execution,
    worker_transport,
)
from strata.notebook.writer import drop_blanked_secrets

if TYPE_CHECKING:
    import cellgraph as cg

    from strata.config import StrataConfig
    from strata.notebook.datasets import DatasetInput
    from strata.notebook.executor_host import ExecutorHost
    from strata.notebook.pool import WarmProcessPool

logger = logging.getLogger(__name__)

# Well-known module → PyPI package name mappings where they differ.
_MODULE_TO_PACKAGE: dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "dateutil": "python-dateutil",
    "jose": "python-jose",
    "dotenv": "python-dotenv",
    "gi": "pygobject",
}


def _detect_missing_module(error: str, stderr: str) -> tuple[str, str] | None:
    """Detect a missing package in a cell failure: ``(language, package)`` or None.

    ``language`` is ``"python"`` or ``"r"`` so the caller picks the installer.
    Python matches ``No module named 'pkg'`` and maps the top-level module to its
    PyPI name via ``_MODULE_TO_PACKAGE`` (cv2 to opencv-python). R matches
    ``there is no package called 'pkg'``; the name is the CRAN name.
    """
    import re

    combined = f"{error}\n{stderr}"

    # Python: ``No module named 'pkg'`` (full ``ModuleNotFoundError`` or harness form).
    py_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", combined)
    if py_match:
        module = py_match.group(1).split(".")[0]
        return ("python", _MODULE_TO_PACKAGE.get(module, module))

    # R: ``there is no package called 'pkg'``; some locales emit curly quotes.
    r_match = re.search(
        r"there is no package called [‘'\"]([^’'\"]+)[’'\"]",
        combined,
    )
    if r_match:
        return ("r", r_match.group(1))

    return None


@dataclass(kw_only=True, frozen=True)
class _CellProvenance:
    """Inputs and result of the standard cell-provenance computation.

    Every cell kind (default, prompt, sql, loop) must produce the same hash for
    the same inputs, since ``compute_staleness`` recomputes it on reopen. The
    intermediates are kept so callers can reuse them.
    """

    # What it is keyed by: the cell (an instance of it, for a fan-out) in an analysis.
    session: ExecutorHost
    analysis: cg.Analysis
    cell_id: str
    annotations: CellAnnotations
    source_hash: str
    runtime_env: dict[str, str]
    effective_worker: str
    runtime_identity: str | None
    env_hash: str
    input_hashes: list[str]
    mount_specs: list[MountSpec]
    mount_fingerprints: list[str]
    has_rw_mount: bool
    table_fingerprints: list[str]
    table_snapshots: dict[str, int | None]
    provenance_hash: str
    # What the key was built from, recorded with a good run to explain a later change.
    key_parts: dict[str, Any] | None = None
    # ``@fetch`` inputs: what each name resolved to, and why any did not.
    fetch_fingerprints: list[str] = field(default_factory=list)
    fetched: dict[str, Path] = field(default_factory=dict)
    fetch_error: str | None = None
    fetch_error_code: str | None = None
    # ``@dataset`` inputs: the version each name resolved to (already in the notebook's
    # store), and why any did not.
    dataset_fingerprints: list[str] = field(default_factory=list)
    datasets: dict[str, DatasetInput] = field(default_factory=dict)
    dataset_error: str | None = None
    # The ``@table`` reads, as ``.strata/tables/index.json`` entries (see table_index.py).
    table_reads: list[dict[str, Any]] = field(default_factory=list)
    # What each declared input weighs (Amber item 89), for the badge and the reads index.
    input_weights: list[InputWeight] = field(default_factory=list)
    # Whether a large read was confirmed (Amber item 90); ``None`` with no threshold set.
    read_confirmation: ReadConfirmation | None = None
    # The fan-out instance this keys, whose key carries its ``variant=`` or ``each=`` subkey.
    instance: str | cg.Item | None = None

    def for_instance(self, instance: str | cg.Item) -> _CellProvenance:
        """The same provenance for one fan-out instance, over the inputs it reads."""
        return self._rehashed(instance=instance)

    def rekeyed(self) -> _CellProvenance:
        """The same provenance over what the upstreams publish now."""
        return self._rehashed()

    def over_mounts(self, prepared: Mapping[str, ResolvedMount]) -> _CellProvenance:
        """The same provenance over the mounts as a run prepared them, the listing it reads.

        The cell was keyed on an earlier listing, which the mount may have moved past.
        """
        fingerprints = [
            mount_component(spec, fingerprint)
            for spec in sorted(self.mount_specs, key=lambda item: item.name)
            if (fingerprint := prepared[spec.name].fingerprint) is not None
        ]
        if fingerprints == self.mount_fingerprints:
            return self
        return self._rehashed(mount_fingerprints=fingerprints)

    def _rehashed(self, **changes: Any) -> _CellProvenance:
        changed = replace(self, **changes)
        key = cell_key(
            changed.session,
            changed.analysis,
            changed.cell_id,
            mounts=changed.mount_fingerprints,
            tables=changed.table_fingerprints,
            fetches=changed.fetch_fingerprints,
            datasets=changed.dataset_fingerprints,
            env_hash=changed.env_hash,
            instance=changed.instance,
        )
        return replace(
            changed,
            provenance_hash=key.provenance_hash,
            input_hashes=key.upstream,
            key_parts=key.parts,
        )

    def reads(
        self, mounts_moved: dict[str, int | None], tables_moved: dict[str, int]
    ) -> dict[str, Any] | None:
        """The reads index entry for a good run: what each input moved, if known."""
        if not self.input_weights:
            return None
        inputs = []
        for weight in self.input_weights:
            moved: int | None = None
            if weight.kind == "mount":
                moved = mounts_moved.get(weight.name, 0)
            elif weight.kind == "table":
                moved = tables_moved.get(weight.uri)
            elif weight.kind == "fetch" and weight.bytes is not None:
                moved = weight.bytes - (weight.cached_bytes or 0)
            inputs.append(
                {
                    **weight.model_dump(include={"kind", "name", "uri", "credential"}),
                    "bytes_moved": moved,
                }
            )
        entry: dict[str, Any] = {
            "machine": self.effective_worker,
            "ran_at": int(time.time() * 1000),
            "inputs": inputs,
        }
        if self.read_confirmation is not None:
            entry["confirmation"] = self.read_confirmation
        return entry


def _team_cache_publish_policy(config: Any) -> str:
    """What the team cache offers outward: ``all``, ``promoted`` or ``off``.

    Read defensively: ``_lake_config`` may return an older config object, and a
    missing setting means the pre-existing behaviour.
    """
    return str(getattr(config, "notebook_team_cache_publish", "all") or "all")


def _worker_mount_bytes(result: Any, mount_specs: list[MountSpec]) -> dict[str, int | None]:
    """What a remote worker reports its read-only remote mounts downloaded.

    ``None`` for a mount it does not report (a worker older than the report).
    """
    raw = result.get("mount_bytes") if isinstance(result, dict) else None
    reported = raw if isinstance(raw, dict) else {}
    moved: dict[str, int | None] = {}
    for mount in mount_specs:
        if mount.mode == MountMode.READ_WRITE or parse_mount_uri(mount.uri)[0] == "file":
            continue
        value = reported.get(mount.name)
        ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        moved[mount.name] = value if ok else None
    return moved


class CellExecutor:
    """Materialize notebook cells (cache-or-build per cell).

    ``execute_cell`` ensures upstream artifacts exist (recursively on a miss),
    checks this cell's cache, and executes and stores on a miss.
    """

    # Outside the server, the config ``_lake_config`` loaded for this executor.
    _loaded_config: StrataConfig | None = None

    def __init__(
        self,
        session: ExecutorHost,
        pool: WarmProcessPool | None = None,
        mount_credentials: MountCredentials | None = None,
    ):
        self.session = session
        self.harness_path = Path(__file__).parent / "harness.py"
        self.r_harness_path = Path(__file__).parent / "languages" / "r" / "harness.R"
        self.pool = pool
        # DAG cycle guard for recursive materialization. Per-instance is correct: each
        # top-level call makes a fresh CellExecutor.
        self._materializing: set[str] = set()
        # Local runs in flight, by cell: the build each holds and the task renewing its
        # lease (``local_runs``). Closed when the run's outputs are stored, or on exit.
        self._open_runs: dict[str, tuple[LocalRun, asyncio.Task[None]]] = {}
        # ``init=`` of a pending branch run, by cell (design note 10 §4).
        self._branch_inits: dict[str, str] = {}
        # Upstreams this run settled, with their results, so the caller can tell a watching
        # client what became of them. Two kinds: each failure along a broken chain, and a
        # cell that previously errored and now ran clean (nothing else would retract the
        # error). Innermost first, so what broke is announced before what it broke.
        self.upstream_results: dict[str, CellExecutionResult] = {}
        # Each cell's result within the open multi-cell run (see ``one_run``). ``None``
        # outside a run, so standalone executions are unchanged.
        self._run_scope: dict[str, CellExecutionResult] | None = None
        # Fetch digests (``{url: "sha256:<hex>"}``) captured when provenance was computed, not
        # re-read at store time: a staleness check in between can record newer bytes than the
        # run used.
        self._fetch_refs: dict[str, dict[str, str]] = {}
        # Each table a cell's last preparation read, by URI, to the snapshot it resolved.
        self._table_refs: dict[str, dict[str, str]] = {}
        # When each of those bytes was downloaded, ``{url: unix time}``, for the record.
        self._fetch_times: dict[str, dict[str, float]] = {}
        # Same for ``@dataset``: ``{strata://name/<reference>: <id>@v=<n>}``.
        self._dataset_refs: dict[str, dict[str, str]] = {}
        # Bytes each read-only remote mount downloaded on this server during a cell's run,
        # ``{mount name: bytes}`` (``None``: a worker did not say), for the reads index.
        # A mount missing from it moved nothing: the run did not prepare it. Reset as a run starts
        # (``forget_moved``), not as provenance is made: a batch prepares mounts
        # first, and a fan-out cell makes its provenance again after its variants ran.
        self._mounts_moved: dict[str, dict[str, int | None]] = {}
        # Bytes each cell's scans through this server read from storage during its run,
        # ``{cell id: {table uri: bytes}}``, counted by the server under the run's token.
        # A table missing from it was not scanned here. Reset with the mounts.
        self._tables_moved: dict[str, dict[str, int]] = {}
        # ``# @output_dir`` files a worker uploaded during a cell's last remote run, by cell:
        # ``{folder: {path: entry}}``. Taken by the run's staging, success or failure.
        self._received_output_files: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        # Each ``# @output_dir`` folder's listing from a cell's run, by cell then folder,
        # shown ahead of the run's displays when it succeeds.
        self._folder_listings: dict[str, dict[str, str]] = {}
        # The items of a running ``# @each`` cell, by cell then instance name.
        self._each_items: dict[str, dict[str, each.EachItem]] = {}
        # The mounts its listing came from, held for the run: its items' files are in them.
        self._each_listed: dict[str, dict[str, ResolvedMount]] = {}
        # What each variant of a running ``# @per_variant`` cell wrote through its mounts,
        # recorded together once every variant has succeeded.
        self._variant_writes: dict[str, list[MountWrite]] = {}
        # Also store the variables of a cell nothing downstream reads. Off by default: only
        # consumed variables are kept, and a leaf's result can be large.
        self.store_leaf_outputs = False
        # What a remote dispatch reports as ``metadata.origin``: "notebook" for an open
        # notebook, "run" for a headless ``strata run``, so a dispatcher can queue
        # interactive cells first.
        self.origin: Literal["notebook", "run"] = "notebook"
        self._mount_resolver = MountResolver(
            cache_dir=session.path / ".strata" / "mount_cache",
            credentials=mount_credentials,
        )
        # Fired after each loop iteration. Set by the WS handler for live progress; unset for
        # REST / CLI.
        self.on_iteration_complete: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        # Fired per prompt-cell streaming event (CELL_OUTPUT_DELTA). Same wiring.
        self.on_prompt_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        # Fired after each @per_variant variant completes (CELL_VARIANT_PROGRESS). Same wiring.
        self.on_variant_complete: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float = DEFAULT_CELL_TIMEOUT_SECONDS,
        *,
        init: str | None = None,
    ) -> CellExecutionResult:
        """Materialise a cell: ensure inputs, check cache, execute, store.

        ``init`` branches: the cell starts from that accepted release's parts
        (``strata.restore()``), as a new execution of the same definition.
        """
        return await self._execute_cell(
            cell_id,
            source,
            timeout_seconds,
            materialize_upstreams=True,
            use_cache=True,
            init=init,
        )

    async def execute_cell_force(
        self, cell_id: str, source: str, timeout_seconds: float = DEFAULT_CELL_TIMEOUT_SECONDS
    ) -> CellExecutionResult:
        """Execute a cell against whatever upstream artifacts exist now ("Run this only").

        Skips upstream materialization and the target's cache lookup.
        """
        return await self._execute_cell(
            cell_id,
            source,
            timeout_seconds,
            materialize_upstreams=False,
            use_cache=False,
        )

    async def execute_cell_rerun(
        self, cell_id: str, source: str, timeout_seconds: float = DEFAULT_CELL_TIMEOUT_SECONDS
    ) -> CellExecutionResult:
        """Re-execute a cell, bypassing only its own cache; upstreams materialize normally."""
        return await self._execute_cell(
            cell_id,
            source,
            timeout_seconds,
            materialize_upstreams=True,
            use_cache=False,
        )

    def _local_runs(self) -> LocalRuns:
        return LocalRuns(self.session.get_artifact_manager())

    async def _close_open_run(self, cell_id: str) -> None:
        """End the cell's local run, if one is open: stop renewing, settle its build."""
        opened = self._open_runs.pop(cell_id, None)
        if opened is not None:
            await self._local_runs().close(opened[0], opened[1])

    async def run_cell_tests(self, cell_id: str, test_source: str) -> CellTestResult:
        """Run a Python cell's unit tests and persist the result."""
        return await run_cell_tests(
            self.session,
            CellTestsContext(
                materialize_upstreams=self._materialize_upstreams,
                forget_moved=self.forget_moved,
                compute_provenance=self._compute_cell_provenance,
                moved=self._moved,
                manifest_tables=self._manifest_tables,
                prepare_mounts=self._prepare_mounts,
                resolve_runtime_env=self._resolve_effective_runtime_env,
                fanout_variants=self._fanout_variants,
                load_input_blobs=self._load_input_blobs,
                add_dataset_inputs=self._add_dataset_inputs,
                harness_env=self._harness_env,
            ),
            cell_id,
            test_source,
        )

    async def execute_batch(
        self,
        cell_specs: list[dict[str, Any]],
        *,
        use_cache: bool = True,
        batch_timeout_seconds: float = 600.0,
        cell_timeout_seconds: float = DEFAULT_CELL_TIMEOUT_SECONDS,
        on_cell_event: Callable[[BatchCellResult], Awaitable[None]] | None = None,
        on_cell_start: Callable[[str], Awaitable[None]] | None = None,
        provenance: Mapping[str, _CellProvenance] | None = None,
    ) -> BatchExecutionResult:
        """Execute a sequence of cells in one harness subprocess.

        ``cell_specs`` are in notebook order, each with ``cell_id``, ``source``,
        ``env`` and a resolved ``mount_manifest``; the caller partitions the
        notebook. ``use_cache=False`` is rerun-all. ``on_cell_event`` is awaited
        as each ``BatchCellResult`` is recorded, so frames stream in real time;
        ``on_cell_start`` is awaited with a cell's id as the harness starts it.
        ``provenance`` is each cell's, as the caller resolved it to pin the spec's
        inputs; a cell without one is resolved at its cache check.
        """
        return await self._run_batch(
            cell_specs,
            use_cache=use_cache,
            batch_timeout_seconds=batch_timeout_seconds,
            cell_timeout_seconds=cell_timeout_seconds,
            on_cell_event=on_cell_event,
            on_cell_start=on_cell_start,
            provenance=provenance,
        )

    @contextmanager
    def one_run(self) -> Generator[None]:
        """Treat every execution inside the block as one run of the notebook.

        Inside it each cell executes at most once; a repeat request returns the
        first result. Otherwise a driver running many cells (``strata run``,
        the cascade, Run All) re-executes a ``# @nocache`` producer once per
        consumer, repeating side effects and giving consumers different values.
        Upstream materialization is itself a request, so it is covered too.

        A cache-bypassing request is satisfied only by an earlier real
        execution, not a cache hit, so rerun-all stays honest. Re-entrant: a
        nested block joins the open run.
        """
        if self._run_scope is not None:
            yield
            return
        self._run_scope = {}
        try:
            yield
        finally:
            self._run_scope = None

    async def _execute_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
        init: str | None = None,
    ) -> CellExecutionResult:
        """Shared execution entrypoint with explicit cache/materialization policy.

        Every mode and language comes through here, so a run scope records each
        cell. A request with no run open becomes its own run, so a diamond over a
        ``# @nocache`` producer reads it once; the scope closes on return.
        """
        if self._run_scope is None:
            with self.one_run():
                return await self._execute_cell(
                    cell_id,
                    source,
                    timeout_seconds,
                    materialize_upstreams=materialize_upstreams,
                    use_cache=use_cache,
                    init=init,
                )
        scope = self._run_scope
        earlier = scope.get(cell_id)
        if earlier is not None and (use_cache or not earlier.cache_hit):
            return earlier
        refused = self._annotation_refusal(cell_id, source)
        if refused is not None:
            return refused
        # The worker registry lives in the server's store: one read per run, off the loop,
        # serves every worker lookup in it.
        notebook_state = self.session.notebook_state
        worker = effective_worker_name(
            self.session.notebook_state, cell_id, parse_annotations(source).worker
        )
        if init is not None:
            self._branch_inits[cell_id] = init
        try:
            with one_policy_read(notebook_state, await read_worker_policy(notebook_state, worker)):
                result = await self._dispatch_cell(
                    cell_id,
                    source,
                    timeout_seconds,
                    materialize_upstreams=materialize_upstreams,
                    use_cache=use_cache,
                )
        finally:
            self._branch_inits.pop(cell_id, None)
            # A run the pipeline left open (it raised, or the cell failed) ends here.
            await self._close_open_run(cell_id)
        if result.success:
            scope[cell_id] = result
        return result

    def _annotation_refusal(self, cell_id: str, source: str) -> CellExecutionResult | None:
        """A failed result when an annotation cannot be read, before anything is staged.

        Otherwise the dropped directive makes the run, and its cache entry, a
        computation other than the one the source declares.
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            return None
        if cell.source != source:
            cell = cell.model_copy(update={"source": source})
        diagnostic = blocking_diagnostic(cell, self.session.notebook_state, self.session.analysis)
        if diagnostic is None:
            return None
        line = f"line {diagnostic.line}: " if diagnostic.line else ""
        result = CellExecutionResult(
            cell_id=cell_id,
            success=False,
            error=f"{line}{diagnostic.message} [{diagnostic.code}]",
            error_code=diagnostic.code,
        )
        self.session.apply_execution_result_metadata(cell_id, result)
        return result

    async def _dispatch_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
    ) -> CellExecutionResult:
        """Route one execution to its language / loop / fan-out pipeline."""
        annotations = parse_annotations(source)
        timeout_seconds = self._resolve_effective_timeout(
            cell_id,
            timeout_seconds,
            annotations.timeout,
        )

        # Only a loop annotation well-formed enough to run; validation flags malformed ones.
        if (
            annotations.loop is not None
            and annotations.loop.max_iter > 0
            and annotations.loop.carry
        ):
            start_time = time.time()
            if cell_id in self._materializing:
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=(
                        f"Cycle detected: cell {cell_id} is already being "
                        f"materialised (stack: {self._materializing})"
                    ),
                )
            self._materializing.add(cell_id)
            try:
                with _loop_folders(annotations.output_dirs) as folders:
                    loop_result = await self._execute_loop_cell(
                        cell_id,
                        source,
                        annotations.loop,
                        timeout_seconds,
                        start_time,
                        materialize_upstreams=materialize_upstreams,
                        use_cache=use_cache,
                        folders=folders,
                    )
            finally:
                self._materializing.discard(cell_id)
            # As the Python path does: a CLI run has no WS handler to persist the console.
            if not loop_result.cache_hit:
                self.session.persist_display_outputs(
                    cell_id, loop_result.display_outputs if loop_result.success else None
                )
            self.session.apply_execution_result_metadata(cell_id, loop_result)
            return loop_result
        effective_worker = effective_worker_name(
            self.session.notebook_state, cell_id, annotations.worker
        )
        worker_spec = resolve_worker_spec(
            self.session.notebook_state,
            effective_worker,
        )
        if not worker_supports_notebook_execution(worker_spec):
            policy_error = get_worker_execution_error(
                self.session.notebook_state,
                effective_worker,
            )
            unsupported = (
                "is unknown; it is not a worker this notebook can use"
                if worker_spec is None
                else "is not implemented yet"
            )
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                error=policy_error
                or f"Execution failed: worker '{effective_worker}' {unsupported}",
            )

        start_time = time.time()

        # --- cycle guard --------------------------------------------------
        if cell_id in self._materializing:
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                error=(
                    f"Cycle detected: cell {cell_id} is already being "
                    f"materialised (stack: {self._materializing})"
                ),
            )
        self._materializing.add(cell_id)

        try:
            with output_dirs.assembling() if annotations.output_dirs else nullcontext():
                return await self._materialize_cell(
                    cell_id,
                    source,
                    timeout_seconds,
                    start_time,
                    materialize_upstreams=materialize_upstreams,
                    use_cache=use_cache,
                )
        finally:
            self._materializing.discard(cell_id)

    def _remote_execution_metadata(
        self,
        worker_spec: Any,
        remote_build_id: str | None = None,
        remote_build_state: str | None = None,
        remote_error_code: str | None = None,
    ) -> dict[str, str]:
        """Return UI-facing remote execution metadata for a worker."""
        if worker_spec is None or worker_spec.backend == WorkerBackendType.LOCAL:
            return {}

        metadata = {
            "remote_worker": str(worker_spec.name),
            "remote_transport": worker_transport(worker_spec),
        }
        if remote_build_id:
            metadata["remote_build_id"] = remote_build_id
        if remote_build_state:
            metadata["remote_build_state"] = remote_build_state
        if remote_error_code:
            metadata["remote_error_code"] = remote_error_code
        return metadata

    def _resolve_effective_timeout(
        self,
        cell_id: str,
        timeout_seconds: float,
        annotation_timeout: float | None,
    ) -> float:
        """Resolve the effective timeout with annotation precedence."""
        if annotation_timeout is not None:
            return annotation_timeout

        cell = self.session.notebook_state.get_cell(cell_id)
        if cell and cell.timeout is not None:
            return cell.timeout

        notebook_timeout = self.session.notebook_state.timeout
        if notebook_timeout is not None:
            return notebook_timeout

        return timeout_seconds

    def _timeout_message(self, cell_id: str | None, timeout_seconds: float) -> str:
        """``cell_timeout_message`` naming the lever ``_resolve_effective_timeout`` took."""
        cell = self.session.notebook_state.get_cell(cell_id) if cell_id else None
        if cell is not None and parse_annotations(cell.source).timeout is not None:
            return cell_timeout_message(timeout_seconds, set_by="annotation")
        if (
            cell is not None and cell.timeout is not None
        ) or self.session.notebook_state.timeout is not None:
            return cell_timeout_message(timeout_seconds, set_by="notebook")
        return cell_timeout_message(timeout_seconds)

    def _analysis_for(self, cell_id: str, source: str) -> cg.Analysis:
        """The session's analysis, or one that read *cell_id* as *source*."""
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise ValueError(f"Cell {cell_id} not found")
        return analysis_with(
            cell.model_copy(update={"source": source}),
            self.session.notebook_state,
            self.session.analysis,
        )

    def _resolve_effective_runtime_env(
        self,
        cell_id: str,
        annotation_env: dict[str, str],
    ) -> dict[str, str]:
        """Resolve the effective runtime env with annotation precedence."""
        cell = self.session.notebook_state.get_cell(cell_id)
        runtime_env = drop_blanked_secrets(cell.env) if cell is not None else {}
        runtime_env.update(annotation_env)
        return runtime_env

    # ------------------------------------------------------------------
    # Provenance computation (shared by every cell-kind path)
    # ------------------------------------------------------------------

    async def _compute_cell_provenance(
        self,
        cell_id: str,
        source: str,
        *,
        annotations: CellAnnotations | None = None,
        mount_specs: list[MountSpec] | None = None,
        mount_fingerprints: list[str] | None = None,
        has_rw_mount: bool | None = None,
    ) -> _CellProvenance:
        """Compute the standard provenance triplet for a cell.

        Every cell kind must feed the store the same hash ``compute_staleness``
        recomputes; drifting any ingredient (runtime env, worker identity, mount
        fingerprints, env-key narrowing) leaves the cell stale forever. Optional
        precomputed arguments let callers skip resolving mounts again. A fan-out
        instance's is this ``for_instance``.
        """
        if annotations is None:
            annotations = parse_annotations(source)
        if mount_specs is None:
            mount_specs = self._resolve_cell_mount_specs(cell_id, source)
        effective_worker = effective_worker_name(
            self.session.notebook_state, cell_id, annotations.worker
        )
        weights: list[InputWeight] = []
        machine_cache = self._mount_cache_for(effective_worker)
        listings: dict[str, dict[str, dict[str, Any]]] = {}
        mount_approvals: list[str] = []
        if mount_fingerprints is None or has_rw_mount is None:
            mount_fingerprints, has_rw_mount = await self._fingerprint_mounts(
                mount_specs,
                weights=weights,
                cache=machine_cache,
                listings=listings,
                approvals=mount_approvals,
                only=_each_files(annotations.each),
            )
        else:
            mount_approvals = list(mount_fingerprints)
        if machine_cache is None and listings:
            weights = await self._ask_worker_cache(effective_worker, mount_specs, weights, listings)

        cell_state = self.session.notebook_state.get_cell(cell_id)
        # Keyed on the source this run executes, which a run started before the editor's
        # flush has and the session does not yet.
        analysis = self._analysis_for(cell_id, source)
        source_hash = analysis.cells[cell_id].definition_key
        assert source_hash is not None  # every cell has one
        runtime_env = self._resolve_effective_runtime_env(cell_id, annotations.env)
        runtime_identity = worker_runtime_identity(self.session.notebook_state, effective_worker)
        declared_env_keys = set(annotations.env) | set(
            getattr(cell_state, "env_overrides", {}) or {}
        )
        provenance_env = narrow_env_for_provenance(
            source,
            runtime_env,
            declared_env_keys,
            language=getattr(cell_state, "language", "python"),
        )
        env_hash = compute_execution_env_hash(
            self.session.path,
            provenance_env,
            runtime_identity=runtime_identity,
        )
        (
            config,
            table_fingerprints,
            table_snapshots,
            table_weights,
            table_refs,
            table_locations,
        ) = await self._weigh_tables(annotations, source)
        weights += table_weights
        table_reads = []
        if annotations.tables:
            from strata.notebook.table_index import table_read

            read_config = config or self._lake_config()
            table_reads = [
                table_read(spec.uri, table_snapshots.get(spec.name), read_config)
                for spec in annotations.tables
            ]
        (
            fetch_fingerprints,
            fetched,
            fetch_refs,
            fetch_times,
            fetch_error,
            fetch_error_code,
        ) = await self._resolve_fetches(annotations.fetches, weights=weights)
        self._fetch_refs[cell_id] = fetch_refs
        # Every output this run stages, whatever stages it, carries the tables it read.
        self._table_refs[cell_id] = table_refs
        self.session.get_artifact_manager().record_table_locations(cell_id, table_locations)
        self._fetch_times[cell_id] = fetch_times
        dataset_fingerprints, datasets, dataset_error = await self._resolve_datasets(
            annotations.datasets
        )
        self._dataset_refs[cell_id] = {
            dataset.resolved.lineage_uri: dataset.local_ref for dataset in datasets.values()
        }
        # Provenance is also made for upstreams that only hit the cache: only news is sent.
        # An empty list is news too: it clears what the cell showed.
        if self.session.record_cell_inputs(cell_id, effective_worker, weights):
            await self._broadcast_cell_inputs(cell_id, effective_worker, weights)
        read_confirmation = self._read_confirmation(
            cell_id,
            effective_worker,
            weights,
            input_fingerprint(
                mount_approvals,
                table_fingerprints,
                [f.url for f in annotations.fetches],
                runtime_identity=runtime_identity,
                config=self._lake_config(),
            ),
        )
        key = cell_key(
            self.session,
            analysis,
            cell_id,
            mounts=mount_fingerprints,
            tables=table_fingerprints,
            fetches=fetch_fingerprints,
            datasets=dataset_fingerprints,
            env_hash=env_hash,
        )

        return _CellProvenance(
            session=self.session,
            analysis=analysis,
            cell_id=cell_id,
            annotations=annotations,
            source_hash=source_hash,
            runtime_env=runtime_env,
            effective_worker=effective_worker,
            runtime_identity=runtime_identity,
            env_hash=env_hash,
            input_hashes=key.upstream,
            mount_specs=mount_specs,
            mount_fingerprints=mount_fingerprints,
            has_rw_mount=has_rw_mount,
            table_fingerprints=table_fingerprints,
            table_snapshots=table_snapshots,
            provenance_hash=key.provenance_hash,
            key_parts=key.parts,
            fetch_fingerprints=fetch_fingerprints,
            fetched=fetched,
            fetch_error=fetch_error,
            fetch_error_code=fetch_error_code,
            dataset_fingerprints=dataset_fingerprints,
            datasets=datasets,
            dataset_error=dataset_error,
            table_reads=table_reads,
            input_weights=weights,
            read_confirmation=read_confirmation,
        )

    # ------------------------------------------------------------------
    # The cell materialization pipeline
    # ------------------------------------------------------------------

    async def _materialize_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float,
        start_time: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
    ) -> CellExecutionResult:
        """Dispatch a cell to its language executor; a missing cell uses the Python one."""
        from strata.notebook.languages import get_language_executor

        self.forget_moved(cell_id)

        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is not None:
            cell.cache_hit = False
            language_executor = get_language_executor(cell.language)
            # Only the Python loader expands a manifest; another reader would see its bytes.
            read = _each_output_read(self.session.dag, cell)
            if read is not None and cell.language != CellLanguage.PYTHON:
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=f"`{read}` is the output of an @each cell, which only a Python cell "
                    "can read yet",
                    execution_method="error",
                )
        else:
            language_executor = get_language_executor(CellLanguage.PYTHON)

        return await language_executor.execute(
            self,
            cell_id,
            source,
            start_time,
            timeout_seconds=timeout_seconds,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    def _python_context(self) -> PythonContext:
        return PythonContext(
            execute_each_cell=self._execute_each_cell,
            execute_fanout_cell=self._execute_fanout_cell,
            fanout_variants=self._fanout_variants,
            materialize_upstreams=self._materialize_upstreams,
            compute_provenance=self._compute_cell_provenance,
            remote_execution_metadata=self._remote_execution_metadata,
            manifest_tables=self._manifest_tables,
            find_or_pull_outputs=self._find_or_pull_outputs,
            validate_output_hit=self._validate_output_hit,
            promote_reverted_leaf=self._promote_reverted_leaf,
            moved=self._moved,
            table_write_refusal=self._table_write_refusal,
            restore_gone_tables=self._restore_gone_tables,
            load_input_blobs=self._load_input_blobs,
            add_dataset_inputs=self._add_dataset_inputs,
            prepare_mounts=self._prepare_mounts,
            local_runs=self._local_runs,
            input_refs=self._input_refs,
            restore_manifest=self._restore_manifest,
            dispatch_execution=self._dispatch_execution,
            parse_result=self._parse_result,
            collect_output_dirs_off_loop=self._collect_output_dirs_off_loop,
            keep_failed_output_dirs=self._keep_failed_output_dirs,
            store_outputs=self._store_outputs,
            close_open_run=self._close_open_run,
            push_to_team_store=self._push_to_team_store,
            store_result_displays=self._store_result_displays,
            store_console_outputs=self._store_console_outputs,
            store_mount_writes=self._store_mount_writes,
            record_good_run=self._record_good_run,
            timeout_message=self._timeout_message,
            sync_back_mounts=self._mount_resolver.sync_back,
            received_output_files=self._received_output_files,
            folder_listings=self._folder_listings,
            each_items=self._each_items,
            each_listed=self._each_listed,
            branch_inits=self._branch_inits,
            open_runs=self._open_runs,
            variant_writes=self._variant_writes,
        )

    async def _execute_python_cell(
        self,
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
        """Run a Python cell (see ``cell_paths/python.py``)."""
        return await run_python_cell(
            self.session,
            self._python_context(),
            cell_id,
            source,
            timeout_seconds,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
            fanout_variant=fanout_variant,
            cell_provenance=cell_provenance,
        )

    def _canonical_output(
        self, cell_id: str, var_name: str, provenance_hash: str
    ) -> ArtifactVersion | None:
        """The cell's latest output *var_name*, if it is the one under ``provenance_hash``.

        find_by_provenance can return a row from another notebook in the same store; only
        this says the cell's own output is that result.
        """
        artifact_mgr = self.session.get_artifact_manager()
        latest = artifact_mgr.artifact_store.get_latest_version(
            artifact_mgr.cell_artifact_id(cell_id, var_name)
        )
        if latest is None or latest.provenance_hash != derive_subkey(provenance_hash, var_name):
            return None
        return latest

    async def _find_or_pull_outputs(
        self,
        cell_id: str,
        provenance_hash: str,
        consumed_vars: set[str],
        *,
        source_hash: str,
        source: str,
        env_hash: str,
        variant: str | None = None,
    ) -> tuple[ArtifactVersion | None, TeamPull | None]:
        """A row under the cell's key, asking the team store only if the store has none.

        A row that turns out not to be this cell's own output is a miss with no pull (batch
        differs: it pulls whenever its canonical outputs are missing). A pull writes each
        variable under its canonical local id, so validation runs against real artifacts.
        """
        artifact_mgr = self.session.get_artifact_manager()
        if not consumed_vars:
            return artifact_mgr.find_cached(provenance_hash), None
        first_key = derive_subkey(provenance_hash, sorted(consumed_vars)[0])
        cached = artifact_mgr.find_cached(first_key)
        team_pull: TeamPull | None = None
        if cached is None:
            team_pull = await self._pull_from_team_store(
                cell_id=cell_id,
                provenance_hash=provenance_hash,
                consumed_vars=consumed_vars,
                source_hash=source_hash,
                source=source,
                env_hash=env_hash,
                input_versions=self._input_refs(cell_id, variant),
                variant=variant,
            )
            cached = artifact_mgr.find_cached(first_key) if team_pull is not None else None
        return cached, team_pull

    async def _gone_tables(self, source: str) -> list[TableWriteSpec]:
        """The ``@write_table`` lines of *source* whose table no longer exists."""
        specs = parse_annotations(source).table_writes
        if not specs:
            return []
        return await anyio.to_thread.run_sync(
            functools.partial(
                gone_tables,
                specs,
                config=self._lake_config(),
                env=dict(self.session.notebook_state.env),
                notebook=self.session.path,
            )
        )

    def _write_back(
        self, cell_id: str, specs: list[TableWriteSpec], variant: str | None = None
    ) -> None:
        """Write a hit's stored outputs into *specs*' tables, as its run wrote them.

        A hit's outputs are each variable's latest version (a revert has promoted them).

        Raises:
            TableWriteError: as for a run's write.
        """
        manager = self.session.get_artifact_manager()
        outputs: dict[str, StagedOutput] = {}
        for spec in specs:
            version = manager.artifact_store.get_latest_version(
                manager.cell_artifact_id(cell_id, spec.variable, variant=variant)
            )
            if version is None:
                continue  # The write names it as not produced.
            outputs[spec.variable] = StagedOutput(
                artifact_id=version.id,
                version=version.version,
                blob=manager.load_artifact_data(version.id, version.version),
                content_type=_artifact_content_type(version),
                provenance_hash=version.provenance_hash,
            )
        self._write_cell_tables(specs, outputs)

    async def _restore_gone_tables(self, cell_id: str, source: str) -> None:
        """Before a hit is served, write its stored outputs into any table that is gone.

        Only those: a table still there holds them already, and an append would double.

        Raises:
            TableWriteError: as for a run's write.
        """
        gone = await self._gone_tables(source)
        if gone:
            await anyio.to_thread.run_sync(self._write_back, cell_id, gone)

    def _validate_output_hit(
        self,
        cell: Any,
        cell_id: str,
        provenance_hash: str,
        consumed_vars: set[str],
        table_writes: list[TableWriteSpec],
        cached_artifact: ArtifactVersion,
        current_displays: list[Any],
        cached_displays: list[Any],
        variant: str | None = None,
    ) -> tuple[ArtifactVersion | None, list[Any]]:
        """A row under the cell's key as the cell's own: each consumed variable's canonical
        output, promoted by a revert. The artifact is None for a miss.
        """
        promoted_count = self._promote_reverted_outputs(
            cell_id,
            provenance_hash,
            consumed_vars,
            table_writes,
            len(current_displays),
            variant,
        )
        if promoted_count is None:
            return None, cached_displays
        if not promoted_count or cell is None:
            return cached_artifact, cached_displays
        artifact_mgr = self.session.get_artifact_manager()
        # The display resolver saw pre-promotion state, and cached_artifact names the superseded
        # version, which would disagree with cell.artifact_uris re-read from latest.
        refreshed = artifact_mgr.artifact_store.get_latest_version(
            artifact_mgr.cell_artifact_id(cell_id, sorted(consumed_vars)[0], variant=variant)
        )
        return (
            refreshed if refreshed is not None else cached_artifact,
            self.session.resolve_cached_display_outputs(
                cell_id, provenance_hash, current_displays, variant
            ),
        )

    def _recorded_display_count(
        self, cell_id: str, provenance_hash: str, variant: str | None
    ) -> int | None:
        """Display-output count recorded by the run cached under ``provenance_hash``, or None."""
        from strata.notebook.session import stored_display

        store = self.session.get_artifact_manager().artifact_store
        display_id = self.session.get_artifact_manager().cell_artifact_id(
            cell_id, "__display__0", variant=variant
        )
        display_prov = derive_subkey(provenance_hash, "__display__0")
        latest = store.get_latest_version(display_id)
        artifact = (
            latest
            if latest is not None and latest.provenance_hash == display_prov
            else store.find_version_by_provenance(display_id, display_prov)
        )
        if artifact is None:
            return None
        described = stored_display(artifact)
        return described[1] if described is not None else None

    def _promote_reverted_outputs(
        self,
        cell_id: str,
        provenance_hash: str,
        variables: Iterable[str],
        table_writes: list[TableWriteSpec],
        current_display_count: int,
        variant: str | None = None,
    ) -> int | None:
        """Make each variable's, display's and console's version under ``provenance_hash``
        latest again.

        A mismatched latest may still have the right result under an older version (reverting
        an edit P1 -> P2 -> P1); promoting re-points "latest" at it without running the cell.
        All are checked before any write. Returns how many were promoted, or None for a miss.
        """
        artifact_mgr = self.session.get_artifact_manager()
        store = artifact_mgr.artifact_store
        to_promote: list[tuple[str, int]] = []
        invalid: tuple[str, ArtifactVersion | None, str] | None = None
        # A later run replaced or merged into these tables; promoting would not undo it.
        rewrites = {spec.variable for spec in table_writes if spec.mode != "append"}
        # sorted() keeps the reported variable and promotion order stable across runs.
        for var_name in sorted(variables):
            canonical_id = artifact_mgr.cell_artifact_id(cell_id, var_name, variant=variant)
            var_prov = derive_subkey(provenance_hash, var_name)
            canonical_art = store.get_latest_version(canonical_id)
            if canonical_art is not None and canonical_art.provenance_hash == var_prov:
                continue
            older = (
                None
                if var_name in rewrites
                else store.find_version_by_provenance(canonical_id, var_prov)
            )
            if older is None or not store.blob_exists(canonical_id, older.version):
                invalid = (canonical_id, canonical_art, var_prov)
                break
            to_promote.append((canonical_id, older.version))

        # Display outputs share the provenance; without promoting them a reverted cell hits with
        # its plot gone. Restore the count the reverted run recorded (the current list holds the
        # later run's, or none after a failure); older displays use the current list.
        # An instance's are found by provenance, as its console is, so need no promoting.
        if invalid is None and variant is None:
            display_count = self._recorded_display_count(cell_id, provenance_hash, variant)
            if display_count is None:
                display_count = current_display_count
            for index in range(display_count):
                display_id = artifact_mgr.cell_artifact_id(
                    cell_id, f"__display__{index}", variant=variant
                )
                display_prov = derive_subkey(provenance_hash, f"__display__{index}")
                display_art = store.get_latest_version(display_id)
                if display_art is not None and display_art.provenance_hash == display_prov:
                    continue
                older_display = store.find_version_by_provenance(display_id, display_prov)
                if older_display is None or not store.blob_exists(
                    display_id, older_display.version
                ):
                    invalid = (display_id, display_art, display_prov)
                    break
                to_promote.append((display_id, older_display.version))

        # The console the reverted run printed, or the hit replays a later run's. A run with
        # none on record still hits: its outputs are all a hit needs.
        if invalid is None and variant is None:
            console_id = artifact_mgr.cell_artifact_id(cell_id, "__console__")
            console_prov = derive_subkey(provenance_hash, "__console__")
            console_art = store.get_latest_version(console_id)
            if console_art is None or console_art.provenance_hash != console_prov:
                older_console = store.find_version_by_provenance(console_id, console_prov)
                if older_console is not None and store.blob_exists(
                    console_id, older_console.version
                ):
                    to_promote.append((console_id, older_console.version))

        if invalid is not None:
            artifact_id, latest, expected = invalid
            logger.info(
                "Cache hit for cell %s invalidated: canonical artifact %s %s "
                "(expected provenance %s).",
                cell_id,
                artifact_id,
                "not found"
                if latest is None
                else f"has stale provenance {latest.provenance_hash[:12]}",
                expected[:12],
            )
            return None
        for artifact_id, source_version in to_promote:
            promoted = store.promote_version(artifact_id, source_version)
            if promoted is None:
                # The blob vanished since the check (likely a GC pass). Run the cell.
                logger.info(
                    "Cell %s could not promote %s@v=%d; re-executing.",
                    cell_id,
                    artifact_id,
                    source_version,
                )
                return None
            logger.info(
                "Cell %s reverted to a known provenance: promoted "
                "%s@v=%d to @v=%d instead of re-executing.",
                cell_id,
                artifact_id,
                source_version,
                promoted.version,
            )
        return len(to_promote)

    def _promote_reverted_leaf(
        self, cell_id: str, provenance_hash: str, current_display_count: int
    ) -> bool:
        """Make a leaf's console and displays from an earlier run under ``provenance_hash``
        latest again. All or none is checked before writing; True when promoted.
        """
        artifact_mgr = self.session.get_artifact_manager()
        store = artifact_mgr.artifact_store
        display_count = self._recorded_display_count(cell_id, provenance_hash, None)
        if display_count is None:
            # A run without displays leaves no display artifact; its console says so.
            console = store.find_version_by_provenance(
                artifact_mgr.cell_artifact_id(cell_id, "__console__"),
                derive_subkey(provenance_hash, "__console__"),
            )
            params = console.params if console is not None else {}
            recorded = params.get("display_count")
            display_count = int(recorded) if recorded is not None else current_display_count
        labels = ["__console__", *(f"__display__{index}" for index in range(display_count))]
        to_promote: list[tuple[str, int]] = []
        for label in labels:
            artifact_id = artifact_mgr.cell_artifact_id(cell_id, label)
            expected = derive_subkey(provenance_hash, label)
            latest = store.get_latest_version(artifact_id)
            if latest is not None and latest.provenance_hash == expected:
                continue
            older = store.find_version_by_provenance(artifact_id, expected)
            if older is None or not store.blob_exists(artifact_id, older.version):
                return False
            to_promote.append((artifact_id, older.version))
        for artifact_id, version in to_promote:
            if store.promote_version(artifact_id, version) is None:
                # The blob vanished since the check (likely a GC pass); run the cell.
                return False
            logger.info(
                "Cell %s reverted to a known provenance: promoted %s@v=%d.",
                cell_id,
                artifact_id,
                version,
            )
        return bool(to_promote)

    def _fanout_variants(self, cell_id: str) -> tuple[str, ...] | None:
        """The variant names if ``cell_id`` is a ``@per_variant`` fan-out cell, else None.

        In the order of the group's tabs, so its console and displays read as the tabs do.
        """
        dag = self.session.dag
        fanout = dag.fanout_cells.get(cell_id) if dag else None
        if dag is None or fanout is None:
            return None
        # The group's members are in cell order, which is the order of its tabs.
        group = next(g for g in dag.variant_groups if g.group == fanout.group)
        return tuple(name for _, name in group.members)

    def _fanout_context(self) -> FanoutContext:
        return FanoutContext(
            materialize_upstreams=self._materialize_upstreams,
            compute_provenance=self._compute_cell_provenance,
            execute_python_cell=self._execute_python_cell,
            moved=self._moved,
            promote_reverted_outputs=self._promote_reverted_outputs,
            gone_tables=self._gone_tables,
            write_back=self._write_back,
            store_console_outputs=self._store_console_outputs,
            prepare_mounts=self._prepare_mounts,
            input_refs=self._input_refs,
            write_cell_tables=self._write_cell_tables,
            variant_writes=self._variant_writes,
            each_items=self._each_items,
            each_listed=self._each_listed,
            branch_inits=self._branch_inits,
            on_variant_complete=self.on_variant_complete,
        )

    async def _execute_fanout_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float,
        start_time: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
        variant_names: tuple[str, ...],
    ) -> CellExecutionResult:
        """Run a ``# @per_variant`` cell (see ``cell_paths/fanout.py``)."""
        return await run_fanout_cell(
            self.session,
            self._fanout_context(),
            cell_id,
            source,
            timeout_seconds,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
            variant_names=variant_names,
        )

    async def _execute_each_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float,
        start_time: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
    ) -> CellExecutionResult:
        """Run a ``# @each`` cell (see ``cell_paths/fanout.py``)."""
        return await run_each_cell(
            self.session,
            self._fanout_context(),
            cell_id,
            source,
            timeout_seconds,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    def _r_context(self) -> RContext:
        return RContext(
            materialize_upstreams=self._materialize_upstreams,
            compute_provenance=self._compute_cell_provenance,
            remote_execution_metadata=self._remote_execution_metadata,
            find_or_pull_outputs=self._find_or_pull_outputs,
            validate_output_hit=self._validate_output_hit,
            promote_reverted_outputs=self._promote_reverted_outputs,
            moved=self._moved,
            table_write_refusal=self._table_write_refusal,
            restore_gone_tables=self._restore_gone_tables,
            load_input_blobs=self._load_input_blobs,
            dispatch_http_executor=self._dispatch_http_executor,
            prepare_mounts=self._prepare_mounts,
            write_manifest=self._write_manifest,
            run_r_harness=self._run_r_harness,
            parse_result=self._parse_result,
            collect_output_dirs_off_loop=self._collect_output_dirs_off_loop,
            keep_failed_output_dirs=self._keep_failed_output_dirs,
            store_outputs=self._store_outputs,
            push_to_team_store=self._push_to_team_store,
            store_result_displays=self._store_result_displays,
            sync_back_mounts=self._mount_resolver.sync_back,
            store_console_outputs=self._store_console_outputs,
            store_mount_writes=self._store_mount_writes,
            record_good_run=self._record_good_run,
            timeout_message=self._timeout_message,
            received_output_files=self._received_output_files,
            folder_listings=self._folder_listings,
        )

    async def _execute_r_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float,
        start_time: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
    ) -> CellExecutionResult:
        """Run an R cell (see ``cell_paths/r.py``)."""
        return await run_r_cell(
            self.session,
            self._r_context(),
            cell_id,
            source,
            timeout_seconds,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    def _dispatch_context(self) -> DispatchContext:
        return DispatchContext(
            origin=self.origin,
            credential_resolver=self._credential_resolver,
            note_worker_mount_bytes=self._note_worker_mount_bytes,
            timeout_message=self._timeout_message,
            broadcast_remote_phase=self._broadcast_remote_phase,
            lake_config=self._lake_config,
            received_output_files=self._received_output_files,
        )

    async def _dispatch_execution(
        self,
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
        """Run a cell on its worker's backend (see ``cell_paths/dispatch.py``)."""
        return await dispatch_execution(
            self.session,
            DispatchRoutes(
                local=self._dispatch_local,
                embedded=self._dispatch_embedded_executor,
                http=self._dispatch_http_executor,
            ),
            worker_spec,
            source,
            input_specs,
            mount_specs,
            output_dir,
            result_dir,
            venv_path,
            runtime_env,
            timeout_seconds,
            remote_build_id=remote_build_id,
            mutation_defines=mutation_defines,
            tables=tables,
            cell_id=cell_id,
            cell_provenance_hash=cell_provenance_hash,
            local_run=local_run,
            restore=restore,
            resolved_mounts=resolved_mounts,
        )

    async def _dispatch_local(
        self,
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
        """Run on this machine (see ``cell_paths/python.py``)."""
        return await dispatch_local(
            self.session,
            LocalDispatchContext(
                prepare_mounts=self._prepare_mounts,
                local_runs=self._local_runs,
                run_token=self._run_token,
                write_manifest=self._write_manifest,
                run_harness=self._run_harness,
                # The session's by default, as an R cell's: a CLI or MCP run is warm too.
                pool=self.pool if self.pool is not None else self.session.warm_pool,
            ),
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

    async def _dispatch_embedded_executor(
        self,
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
        """Run the bundle-based path here (see ``cell_paths/dispatch.py``)."""
        return await dispatch_embedded(
            EmbeddedContext(
                prepare_mounts=self._prepare_mounts,
                run_token=self._run_token,
                write_manifest=self._write_manifest,
                run_harness=self._run_harness,
            ),
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

    async def _locked_environment(
        self, worker_spec: Any, language: str = "python"
    ) -> dict[str, str] | None:
        """The notebook's lock, for a worker that runs in it (see ``cell_paths/dispatch.py``)."""
        return await locked_environment(
            self.session,
            worker_spec,
            language=language,
        )

    async def _dispatch_http_executor(
        self,
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
        """Run a cell over HTTP (see ``cell_paths/dispatch.py``)."""
        return await dispatch_http(
            self.session,
            self._dispatch_context(),
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
            language=language,
        )

    async def _await_accepted_job(
        self,
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
        """Follow a job a worker accepted (see ``cell_paths/dispatch.py``)."""
        return await await_accepted_job(
            self._dispatch_context(),
            accepted,
            submit_url=submit_url,
            headers=headers,
            timeout_seconds=timeout_seconds,
            cancel=cancel,
            worker_spec=worker_spec,
            cell_id=cell_id,
            receive_bundle=receive_bundle,
        )

    def _mount_cache_for(self, worker_name: str) -> MountCache | None:
        """The mount cache a cell on *worker_name* reads through, if it is this server's."""
        worker = resolve_worker_spec(self.session.notebook_state, worker_name)
        if worker is None or worker.backend == WorkerBackendType.LOCAL:
            return MountCache.from_config(self._lake_config())
        if is_embedded_executor_worker(worker):
            return MountCache.from_config(self._lake_config())
        return None

    async def _ask_worker_cache(
        self,
        worker_name: str,
        mount_specs: list[MountSpec],
        weights: list[InputWeight],
        listings: dict[str, dict[str, dict[str, Any]]],
    ) -> list[InputWeight]:
        """*weights* with the mounts' cached shares a remote worker reports, if it does."""
        worker = resolve_worker_spec(self.session.notebook_state, worker_name)
        if worker is None:
            return weights
        specs = {m.name: m for m in mount_specs}
        names = sorted(listings)
        answer = await worker_cached_bytes(
            worker,
            _resolve_worker_token(worker),
            [
                {
                    "spec": specs[name].model_dump(mode="json"),
                    "entries": listing_entries(listings[name]),
                }
                for name in names
            ],
        )
        if answer is None:
            return weights
        shares = dict(zip(names, answer, strict=True))
        return [
            w.model_copy(update={"cached_bytes": shares[w.name]})
            if w.kind == "mount" and w.name in shares
            else w
            for w in weights
        ]

    async def _broadcast_cell_inputs(
        self, cell_id: str, machine: str, weights: list[InputWeight]
    ) -> None:
        """Tell the notebook what the cell's inputs weigh. Best effort, like the phase."""
        try:
            from strata.notebook.protocol import MessageType
            from strata.notebook.ws import _broadcast_message, _make_message, next_notebook_sequence
            from strata.notebook.ws_payloads import cell_inputs_payload

            session_id = self.session.id
            await _broadcast_message(
                session_id,
                _make_message(
                    MessageType.CELL_INPUTS,
                    next_notebook_sequence(session_id),
                    cell_inputs_payload(cell_id, machine, weights),
                ),
            )
        except Exception:
            logger.debug("cell inputs broadcast failed", exc_info=True)

    async def _broadcast_remote_phase(
        self,
        cell_id: str | None,
        worker_spec: Any,
        phase: str,
        *,
        queue: tuple[int | None, str | None] = (None, None),
    ) -> None:
        """Move the cell's badge (see ``cell_paths/dispatch.py``)."""
        return await broadcast_remote_phase(
            self.session,
            cell_id,
            worker_spec,
            phase,
            queue=queue,
        )

    def _stage_signed_transport_inputs(
        self,
        *,
        artifact_store: Any,
        build_id: str,
        input_specs: dict[str, dict[str, Any]],
        output_dir: Path,
        tenant_id: str | None,
        principal_id: str | None,
    ) -> tuple[dict[str, dict[str, Any]], list[tuple[str, int]]]:
        """Stage upstream blobs for signed transport (see ``cell_paths/dispatch.py``)."""
        return stage_signed_transport_inputs(
            self.session,
            artifact_store=artifact_store,
            build_id=build_id,
            input_specs=input_specs,
            output_dir=output_dir,
            tenant_id=tenant_id,
            principal_id=principal_id,
        )

    # ------------------------------------------------------------------
    # ①½ Resolve mounts
    # ------------------------------------------------------------------

    def _resolve_cell_mount_specs(
        self,
        cell_id: str,
        source: str,
    ) -> list[MountSpec]:
        """Resolve a cell's mount declarations: annotation, then cell meta, then notebook."""
        cell = self.session.notebook_state.get_cell(cell_id)

        cell_mounts_spec = mounts_of(self.session.notebook_state, cell, source) if cell else []

        annotations = parse_annotations(source)
        annotation_mounts = annotations.mounts

        merged = resolve_cell_mounts(
            [],
            cell_mounts_spec,
            annotation_mounts,
        )

        return merged

    async def _fingerprint_mounts(
        self,
        mount_specs: list[MountSpec],
        *,
        weights: list[InputWeight] | None = None,
        cache: MountCache | None = None,
        listings: dict[str, dict[str, dict[str, Any]]] | None = None,
        approvals: list[str] | None = None,
        only: Mapping[str, str] | None = None,
    ) -> tuple[list[str], bool]:
        """Compute mount fingerprints without preparing local materializations.

        *weights*, when given, gets each remote read-only mount's weight from the same
        listing, its cached share counted in *cache* (``None``: unknown). *listings* gets
        the listing of each mount whose share is unknown, for the machine to be asked.
        *approvals* gets what a yes to reading each mount is kept for. *only* gives, by
        mount name, the glob of the files a run reads, the rest of the mount weighing nothing.
        """
        self._mount_resolver.credential_resolver = self._credential_resolver()
        mount_fingerprints: list[str] = []
        has_rw_mount = False
        for mount in sorted(mount_specs, key=lambda item: item.name):
            # Listing an object store blocks: keep it off the event loop.
            fingerprint, weight, approval = await asyncio.to_thread(
                weigh_mount_sync,
                self._mount_resolver,
                mount,
                cache,
                (only or {}).get(mount.name),
            )
            if approvals is not None and approval is not None:
                approvals.append(approval)
            if listings is not None and weight is not None and weight.listing is not None:
                listings[mount.name] = weight.listing
            if weights is not None and weight is not None:
                weights.append(
                    InputWeight(
                        kind="mount",
                        name=mount.name,
                        uri=mount.uri,
                        credential=mount.credential,
                        bytes=weight.bytes,
                        cached_bytes=weight.cached_bytes,
                    )
                )
            if fingerprint is None:
                has_rw_mount = True
            else:
                mount_fingerprints.append(fingerprint)
        return mount_fingerprints, has_rw_mount

    def _credential_resolver(self) -> CredentialResolver:
        """Named credentials as this notebook sees them right now.

        Rebuilt per use: a secret manager fills the env after the session opens,
        and a rotated value must reach the next run. Who may write through a credential is
        judged by the principal who started this run.
        """
        from strata.auth.principal import get_principal

        principal = get_principal()
        return CredentialResolver.from_config(
            self._lake_config(),
            env=dict(self.session.notebook_state.env),
            notebook=self.session.path,
            principal=principal.id if principal is not None else None,
        )

    def _note_worker_mount_bytes(
        self, cell_id: str | None, result: dict[str, Any], mount_specs: list[MountSpec]
    ) -> None:
        """A worker prepared the cell's mounts: keep what it reports they downloaded."""
        if cell_id is not None:
            self._mounts_moved.setdefault(cell_id, {}).update(
                _worker_mount_bytes(result, mount_specs)
            )

    def forget_moved(self, cell_id: str) -> None:
        """A run of *cell_id* starts: what an earlier one read is not its own."""
        self._mounts_moved.pop(cell_id, None)
        self._tables_moved.pop(cell_id, None)

    def _moved(self, cell_id: str) -> tuple[dict[str, int | None], dict[str, int]]:
        """What *cell_id*'s run moved: by mount name, and by table URI."""
        return self._mounts_moved.get(cell_id, {}), self._tables_moved.get(cell_id, {})

    async def _prepare_mounts(
        self,
        mount_specs: list[MountSpec],
        cell_id: str | None = None,
        *,
        only: Mapping[str, str] | None = None,
    ) -> dict[str, ResolvedMount]:
        """Prepare local mount materializations for local execution paths.

        With *cell_id*, what each view downloaded is added to the cell's moved bytes. *only*
        limits what a remote mount downloads (``MountResolver.prepare_mounts``).
        """
        if not mount_specs:
            return {}
        self._mount_resolver.credential_resolver = self._credential_resolver()
        ensure_writable(self._mount_resolver.credential_resolver, mount_specs)
        self._mount_resolver.mount_cache = MountCache.from_config(self._lake_config())
        resolved = await self._mount_resolver.prepare_mounts(mount_specs, only=only)
        if cell_id is not None:
            moved = self._mounts_moved.setdefault(cell_id, {})
            for name, mount in resolved.items():
                if mount.bytes_moved is not None:
                    moved[name] = (moved.get(name) or 0) + mount.bytes_moved
        staged = [rm.staging_dir for rm in resolved.values() if rm.staging_dir is not None]
        if staged:
            try:
                harness_user = resolve_harness_user()
            except LocalExecutionRefused:
                harness_user = None  # the run is refused where the cell would start
            for staging in staged:
                # The cell writes the staging copy; sync_back reads it without following links.
                let_through(self._mount_resolver.cache_dir, harness_user)
                let_through(staging.parent, harness_user)
                hand_over(staging, harness_user)
        return resolved

    async def _resolve_fetches(
        self, fetch_specs: list[FetchSpec], *, weights: list[InputWeight] | None = None
    ) -> tuple[
        list[str], dict[str, Path], dict[str, str], dict[str, float], str | None, str | None
    ]:
        """Check every ``@fetch`` right before a run and fingerprint its bytes.

        Always rechecked (``max_age=0``) so the run records what the URL served
        when it ran. Returns fingerprints, a path per name, the lineage input per
        URL, when each URL's bytes were downloaded, and the first failure and its
        code, if any. *weights*, when given, gets each fetch's size, cached unless this
        check downloaded it.
        """
        if not fetch_specs:
            return [], {}, {}, {}, None, None
        from strata.notebook.fetch import FetchCache, FetchError, guard_settings

        allowed_hosts, allow_local = guard_settings(self._lake_config())
        cache = FetchCache(self.session.path, allowed_hosts=allowed_hosts, allow_local=allow_local)
        fingerprints: list[str] = []
        fetched: dict[str, Path] = {}
        refs: dict[str, str] = {}
        times: dict[str, float] = {}
        error: str | None = None
        error_code: str | None = None
        loop = asyncio.get_running_loop()
        for spec in sorted(fetch_specs, key=lambda item: item.name):
            try:
                result = await loop.run_in_executor(
                    None, lambda s=spec: cache.resolve(s, max_age=0)
                )
            except FetchError as exc:
                if error is None:
                    error, error_code = str(exc), exc.code
                fingerprints.append(cache.fingerprint(spec, max_age=float("inf")))
                continue
            fingerprints.append(result.fingerprint(spec))
            fetched[spec.name] = result.path
            if weights is not None:
                size = result.path.stat().st_size
                weights.append(
                    InputWeight(
                        kind="fetch",
                        name=spec.name,
                        uri=spec.url,
                        credential=None,
                        bytes=size,
                        cached_bytes=0 if result.downloaded else size,
                    )
                )
            refs[spec.url] = f"sha256:{result.sha256}"
            if result.fetched_at is not None:
                times[spec.url] = result.fetched_at
        return fingerprints, fetched, refs, times, error, error_code

    def _fetch_params(self, cell_id: str) -> dict[str, str]:
        """Transform params recording when each fetched input was downloaded, if any."""
        times = self._fetch_times.get(cell_id)
        return {"fetched_at": json.dumps(times, sort_keys=True)} if times else {}

    async def _resolve_datasets(
        self, dataset_specs: list[DatasetSpec]
    ) -> tuple[list[str], dict[str, DatasetInput], str | None]:
        """Resolve every ``@dataset`` before a run and copy it into the notebook's store.

        The answer is handed to the session so staleness agrees with what the
        run recorded. Returns fingerprints, the input per variable, and the
        first failure.
        """
        if not dataset_specs:
            return [], {}, None
        from strata.notebook.datasets import (
            DatasetError,
            copy_into,
            registry_for,
            unresolved_fingerprint,
        )

        store = self.session.get_artifact_manager().artifact_store
        tenant = self.session.opened_by[1] if self.session.opened_by else None
        fingerprints: list[str] = []
        datasets: dict[str, DatasetInput] = {}
        error: str | None = None
        for spec in sorted(dataset_specs, key=lambda item: item.name):
            try:
                registry = registry_for(self._lake_config(), tenant)
                resolved = await anyio.to_thread.run_sync(registry.resolve, spec)
                dataset = await anyio.to_thread.run_sync(copy_into, registry, resolved, store)
            except DatasetError as exc:
                error = error or str(exc)
                fingerprint = unresolved_fingerprint(spec)
            else:
                fingerprint = resolved.fingerprint
                datasets[spec.name] = dataset
            self.session.remember_dataset_fingerprint(spec, fingerprint)
            fingerprints.append(fingerprint)
        return fingerprints, datasets, error

    def _add_dataset_inputs(
        self,
        input_specs: dict[str, Any],
        datasets: dict[str, DatasetInput],
        output_dir: Path,
    ) -> None:
        """Write each dataset's bytes from the notebook's store into *output_dir* as an input."""
        for name, dataset in sorted(datasets.items()):
            artifact_id, _, version = dataset.local_ref.partition("@v=")
            ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(dataset.content_type, "")
            file_name = f"__dataset_{name}{ext}"
            if not self._write_input_file(
                output_dir / file_name, artifact_id, int(version), dataset.content_type
            ):
                continue
            input_specs[name] = {
                "content_type": dataset.content_type,
                "file": file_name,
                "uri": f"strata://artifact/{dataset.local_ref}",
            }

    def _input_refs(self, cell_id: str, variant: str | None = None) -> dict[str, str]:
        """Inputs an artifact of *cell_id* records: upstreams, fetch digests, dataset
        versions, and each table read with the snapshot it resolved.

        With ``variant`` set, a chained ``# @per_variant`` cell records only the
        upstream instance it zipped to.
        """
        return {
            **self.session.collect_input_refs(cell_id, variant=variant),
            **self._fetch_refs.get(cell_id, {}),
            **self._dataset_refs.get(cell_id, {}),
            **self._table_refs.get(cell_id, {}),
        }

    async def _weigh_tables(
        self, annotations: CellAnnotations, source: str
    ) -> tuple[
        Any, list[str], dict[str, int | None], list[InputWeight], dict[str, str], dict[str, str]
    ]:
        """A cell's tables: the config they resolve in, fingerprints, snapshots, weights,
        the snapshot each read resolved by URI (what its outputs record), and the
        location each loaded read found its table at, by URI.

        A DuckDB cell's catalog tables are inputs like its @table ones, though only the
        @table ones are weighed.
        """
        tables = list(annotations.tables)
        config = None
        # Imported only for SQL cells: the SQL package needs the [sql] extra.
        if annotations.sql is not None:
            from strata.notebook.sql.lake import lake_tables, with_notebook_catalogs

            tables += lake_tables(self.session.notebook_state, source)
            config = with_notebook_catalogs(self._lake_config(), self.session.notebook_state)
        from strata.notebook.tables import table_credential

        sizes: dict[str, int | None] = {}
        locations: dict[str, str] = {}
        fingerprints, snapshots = await self._fingerprint_tables(
            tables, config, sizes=sizes, locations=locations
        )
        weights = [
            InputWeight(
                kind="table",
                name=spec.name,
                uri=spec.uri,
                credential=table_credential(spec.uri, config or self._lake_config()),
                bytes=sizes[spec.name],
                cached_bytes=None,
            )
            for spec in annotations.tables
            if spec.name in sizes
        ]
        # A table with no snapshot yet is not an input the record can name.
        refs = {
            spec.uri: str(snapshots[spec.name])
            for spec in tables
            if snapshots.get(spec.name) is not None
        }
        found = {spec.uri: locations[spec.name] for spec in tables if spec.name in locations}
        return config, fingerprints, snapshots, weights, refs, found

    async def estimate_reads(self, cell_id: str, source: str) -> ReadEstimate:
        """What *cell_id*'s inputs would move onto its machine, weighed before it runs.

        The weights its provenance makes, except that a ``@fetch`` is asked with a HEAD
        rather than downloaded.
        """
        annotations = parse_annotations(source)
        mount_specs = self._resolve_cell_mount_specs(cell_id, source)
        machine = effective_worker_name(self.session.notebook_state, cell_id, annotations.worker)
        weights: list[InputWeight] = []
        cache = self._mount_cache_for(machine)
        listings: dict[str, dict[str, dict[str, Any]]] = {}
        mount_approvals: list[str] = []
        await self._fingerprint_mounts(
            mount_specs,
            weights=weights,
            cache=cache,
            listings=listings,
            approvals=mount_approvals,
            only=_each_files(annotations.each),
        )
        if cache is None and listings:
            weights = await self._ask_worker_cache(machine, mount_specs, weights, listings)
        _, table_fingerprints, _, table_weights, _, _ = await self._weigh_tables(
            annotations, source
        )
        weights += table_weights
        if annotations.fetches:
            from strata.notebook.fetch import FetchCache, guard_settings

            allowed_hosts, allow_local = guard_settings(self._lake_config())
            fetches = FetchCache(
                self.session.path, allowed_hosts=allowed_hosts, allow_local=allow_local
            )
            for spec in sorted(annotations.fetches, key=lambda item: item.name):
                size, cached = await asyncio.to_thread(fetches.probe, spec)
                weights.append(
                    InputWeight(
                        kind="fetch", name=spec.name, uri=spec.url, bytes=size, cached_bytes=cached
                    )
                )
        config = self._lake_config()
        fingerprint = input_fingerprint(
            mount_approvals,
            table_fingerprints,
            [f.url for f in annotations.fetches],
            runtime_identity=worker_runtime_identity(self.session.notebook_state, machine),
            config=config,
        )
        return estimate(cell_id, machine, weights, fingerprint, config)

    def _read_confirmation(
        self, cell_id: str, machine: str, weights: list[InputWeight], fingerprint: str
    ) -> ReadConfirmation | None:
        """What the reads index records about asking first, for the run about to start."""
        config = self._lake_config()
        if not read_confirm_asks(config):
            return None
        read = estimate(cell_id, machine, weights, fingerprint, config)
        return confirmation(
            read, config, self.session.reads_approved(cell_id, machine, fingerprint)
        )

    async def _fingerprint_tables(
        self,
        table_specs: list[TableSpec],
        config: Any = None,
        *,
        sizes: dict[str, int | None] | None = None,
        locations: dict[str, str] | None = None,
    ) -> tuple[list[str], dict[str, int | None]]:
        """Resolve table snapshots for provenance hashing (see tables.py).

        Catalog I/O runs off the event loop. An unreachable catalog yields a
        random fingerprint (cell shows stale) rather than raising, since this
        also runs on notebook open. *config* defaults to the server's.
        """
        if not table_specs:
            return [], {}
        from strata.notebook.tables import fingerprint_tables

        config = config or self._lake_config()
        env = dict(self.session.notebook_state.env)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                fingerprint_tables,
                table_specs,
                config,
                env,
                notebook=self.session.path,
                sizes=sizes,
                locations=locations,
            ),
        )

    def _lake_config(self):
        """Server config when running inside the server, else loaded once for this executor.

        Several reads a cell: loading each one cost an ``@each`` item a fifth of its run.
        """
        try:
            from strata.server import get_state

            return get_state().config
        except RuntimeError:
            if self._loaded_config is None:
                from strata.config import StrataConfig

                self._loaded_config = StrataConfig.load()
            return self._loaded_config

    def _ambient_strata_url(self) -> str:
        """Server URL the harness binds the injected ``strata`` client to.

        This server's ``server_url``, unless ``notebook_remote_store_url`` points
        at a shared remote store. Not part of provenance. A config with no URL
        yields ``""`` and the harness injects no ``strata``.
        """
        config = self._lake_config()
        remote = getattr(config, "notebook_remote_store_url", None)
        if remote:
            return str(remote)
        return str(getattr(config, "server_url", "") or "")

    def _harness_env(self, extra: dict[str, str] | None = None) -> dict[str, str] | None:
        """The environment to spawn a cell subprocess with.

        ``None`` when nothing is filtered or added, so the spawn inherits.
        """
        from strata.notebook.harness_env import harness_env

        allowlist = list(getattr(self._lake_config(), "notebook_harness_env_allowlist", []) or [])
        if not allowlist and not extra:
            return None
        return harness_env(allowlist, extra)

    def _ambient_promote_url(self) -> str:
        """Where a cell's ``strata.promote`` posts, or ``""`` when it cannot.

        Always this server, even with a team store, because only this process
        can read the notebook's artifacts. Empty without a team store, where
        promoting means nothing.
        """
        config = self._lake_config()
        if not getattr(config, "notebook_remote_store_url", None):
            return ""
        server_url = str(getattr(config, "server_url", "") or "").rstrip("/")
        if not server_url:
            return ""
        return f"{server_url}/v1/notebooks/{self.session.id}"

    @contextmanager
    def _run_token(self, timeout_seconds: float) -> Generator[str, None, None]:
        """A token the cell's ambient client calls this server with, for the run's length.

        The server counts the run's scans under it. Run by an authenticated principal, the
        cell also acts as them (see ``strata.auth.run_tokens``).
        """
        from strata.auth.principal import get_principal
        from strata.auth.run_tokens import run_token

        with run_token(
            get_principal(), self.session.id, timeout_seconds, meter=self._tables_moved
        ) as token:
            yield token

    def _cell_strata_url(self) -> str:
        """Where a cell's ambient client points: this server, never the team store.

        The manifest is readable by the cell, and a team-store token there would
        let the cell assert any ``X-Strata-Principal``.
        """
        config = self._lake_config()
        return str(getattr(config, "server_url", "") or "")

    def _ambient_strata_headers(self) -> dict[str, str]:
        """Auth headers the ambient client sends to a remote store; empty for the local server."""
        config = self._lake_config()
        if not getattr(config, "notebook_remote_store_url", None):
            return {}
        from strata.auth.principal import remote_store_headers

        return remote_store_headers(config)

    async def _pull_from_team_store(
        self,
        *,
        cell_id: str,
        provenance_hash: str,
        consumed_vars: set[str],
        source_hash: str,
        source: str,
        env_hash: str,
        input_versions: dict[str, str],
        variant: str | None = None,
    ) -> TeamPull | None:
        """Try to serve this cell from a colleague's result.

        Runs only after a local miss. ``None`` when the feature is off, no store
        is configured, or the store lacks any consumed variable; the caller then
        runs the cell. The client is per pull: setup is cheap next to the run.
        """
        config = self._lake_config()
        if not getattr(config, "notebook_team_cache_enabled", False):
            return None
        if _team_cache_publish_policy(config) == "off":
            # "off" disables the whole feature; unsetting the URL would also remove the ambient
            # client a cell uses.
            return None
        base_url = getattr(config, "notebook_remote_store_url", None)
        if not base_url:
            return None

        store = TeamStore(str(base_url), self._ambient_strata_headers())
        try:
            return await pull_cell_outputs(
                store,
                self.session.get_artifact_manager(),
                cell_id=cell_id,
                provenance_hash=provenance_hash,
                consumed_vars=consumed_vars,
                source_hash=source_hash,
                source=source,
                env_hash=env_hash,
                input_versions=input_versions,
                variant=variant,
            )
        finally:
            await store.aclose()

    async def _push_to_team_store(
        self,
        *,
        cell_id: str,
        variant: str | None = None,
    ) -> None:
        """Offer this cell's fresh outputs to the team store.

        Runs only after the cell succeeded and stored its artifacts. Failures
        are swallowed: a shared-cache problem must not fail a finished cell.
        Gated on the same switch as the pull.
        """
        config = self._lake_config()
        if not getattr(config, "notebook_team_cache_enabled", False):
            return
        if _team_cache_publish_policy(config) != "all":
            # Under "promoted", sharing is deliberate. Pulls still happen.
            return
        base_url = getattr(config, "notebook_remote_store_url", None)
        if not base_url:
            return
        if self.session.dag is None:
            return
        consumed_vars = self.session.dag.consumed_variables.get(cell_id, set())
        if not consumed_vars:
            return

        try:
            # Publishing from an environment that doesn't match ``uv.lock`` would stamp the
            # artifact with the wrong environment, and first-writer-wins makes it everyone's
            # answer. Refuse the publish, not the run: the result stays local, and a broken sync
            # stays the owner's problem.
            attestation_error = self.session.environment_attestation_error()
            if attestation_error is not None:
                logger.warning(
                    "Not publishing cell %s to the team store: %s, so its provenance "
                    "would describe an environment the result was not built in. "
                    "Re-run the environment sync.",
                    cell_id,
                    attestation_error,
                )
                return

            store = TeamStore(str(base_url), self._ambient_strata_headers())
            try:
                await publish_cell_outputs(
                    store,
                    self.session.get_artifact_manager(),
                    cell_id=cell_id,
                    consumed_vars=consumed_vars,
                    variant=variant,
                )
            finally:
                await store.aclose()
        except Exception:
            logger.exception("Could not offer cell %s's outputs to the team store", cell_id)

    def _manifest_tables(
        self,
        table_specs: list[TableSpec],
        table_snapshots: dict[str, int | None],
    ) -> dict[str, dict[str, Any]]:
        """Build the manifest ``tables`` block from resolved snapshots.

        A table with no snapshots yet is injected with ``snapshot_id`` None.

        Raises:
            RuntimeError: If two ``@table`` declarations share a name, or a
                declared table's snapshot could not be resolved.
        """
        # Snapshots are keyed by name, so duplicates collapse: one wins injection while both
        # feed provenance. Reject first.
        seen: set[str] = set()
        duplicates: set[str] = set()
        for spec in table_specs:
            if spec.name in seen:
                duplicates.add(spec.name)
            seen.add(spec.name)
        if duplicates:
            raise RuntimeError(
                "duplicate @table name(s): "
                + ", ".join(sorted(duplicates))
                + "; each @table must have a unique name within the cell"
            )

        from strata.iceberg.catalog import in_notebook

        tables: dict[str, dict[str, Any]] = {}
        for spec in table_specs:
            if spec.name not in table_snapshots:
                raise RuntimeError(
                    f"@table {spec.name}: could not resolve a snapshot for "
                    f"{spec.uri!r}; is the catalog reachable?"
                )
            # The cell asks the server, whose directory is not the notebook's.
            tables[spec.name] = {
                "uri": in_notebook(spec.uri, self.session.path),
                "snapshot_id": table_snapshots[spec.name],
            }
        return tables

    def _write_manifest(
        self,
        source: str,
        input_specs: dict[str, dict[str, str]],
        output_dir: Path,
        runtime_env: dict[str, str],
        resolved_mounts: dict[str, ResolvedMount],
        mutation_defines: list[str] | None = None,
        loop_config: dict[str, Any] | None = None,
        tables: dict[str, dict[str, Any]] | None = None,
        cell_id: str | None = None,
        folder_mounts: dict[str, dict[str, str]] | None = None,
        run_token: str | None = None,
        restore: dict[str, Any] | None = None,
    ) -> Path:
        """Write the harness manifest for one local execution.

        *folder_mounts* replaces the run's own ``# @output_dir`` folders: a loop's live
        outside each iteration's run directory. A *run_token* is not written, only
        announced: it follows on the harness's stdin. *restore* names the release and
        the part files ``strata.restore()`` returns (``_restore_manifest``).
        """
        manifest_mounts = {
            name: {
                "uri": rm.spec.uri,
                "mode": rm.spec.mode.value,
                "local_path": str(rm.local_path),
            }
            for name, rm in resolved_mounts.items()
        }
        # Made here, in the run directory, so its hand-over covers the folders.
        manifest_mounts.update(
            output_dirs.mount_entries(source, output_dir)
            if folder_mounts is None
            else folder_mounts
        )
        manifest: dict[str, Any] = {
            "source": source,
            "inputs": input_specs,
            "output_dir": str(output_dir),
            "mounts": manifest_mounts,
            "tables": tables or {},
            "env": runtime_env,
            "mutation_defines": list(mutation_defines or []),
            "strata_url": self._cell_strata_url(),
            "strata_cell_id": cell_id,
            "strata_promote_url": self._ambient_promote_url(),
            **self._work_dir_fields(),
        }
        if loop_config is not None:
            manifest["loop"] = loop_config
        if run_token is not None:
            manifest["run_token_on_stdin"] = True
        if restore is not None:
            manifest["restore"] = restore
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        return manifest_path

    def _work_dir_fields(self) -> dict[str, str]:
        """Where a local cell runs, its notebook's ``work/``, made and handed over here.

        Relative: every local harness starts in the notebook's directory, and a relative
        change needs no directory above it to be searchable by the cell's user.
        """
        try:
            harness_user = resolve_harness_user()
        except LocalExecutionRefused:
            harness_user = None  # the run is refused where the cell would start
        prepare_work_dir(self.session.path, harness_user)
        return {"work_dir": WORK_DIR, "notebook_dir": str(self.session.path)}

    # ------------------------------------------------------------------
    # Prompt cell execution (LLM path)
    # ------------------------------------------------------------------

    async def _execute_prompt_cell(
        self,
        cell_id: str,
        source: str,
        start_time: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
    ) -> CellExecutionResult:
        """Execute a prompt cell via the LLM provider."""
        return await run_prompt_cell(
            self.session,
            self._path_context(),
            cell_id,
            source,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
            on_delta=self.on_prompt_delta,
        )

    async def _execute_sql_cell(
        self,
        cell_id: str,
        source: str,
        start_time: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
    ) -> CellExecutionResult:
        """Execute a SQL cell via ``strata.notebook.sql.cell_executor``."""
        return await run_sql_cell(
            self.session,
            self._path_context(),
            cell_id,
            source,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    async def _execute_widget_cell(
        self,
        cell_id: str,
        source: str,
        start_time: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
    ) -> CellExecutionResult:
        """Execute a widget cell. It has no upstream inputs and no subprocess."""
        del materialize_upstreams  # widgets have no upstream inputs

        return await run_widget_cell(
            self.session, self._path_context(), cell_id, source, start_time, use_cache=use_cache
        )

    def _path_context(self) -> PathContext:
        """What an extracted path shares. Built per call, so a replaced method still holds."""
        return PathContext(
            compute_provenance=self._compute_cell_provenance,
            moved=self._moved,
            materialize_upstreams_or_failure=self._materialize_upstreams_or_failure,
            store_inline_display_outputs=self._store_inline_display_outputs,
        )

    # ------------------------------------------------------------------
    # ① Materialise upstream cells
    # ------------------------------------------------------------------

    async def _materialize_upstreams_or_failure(
        self, cell_id: str, start_time: float, execution_method: str
    ) -> CellExecutionResult | None:
        """Materialize the upstreams, or return the failure as this cell's result.

        SQL, prompt and loop cells would otherwise let the error escape and skip
        the bookkeeping a returned result gets, so neither this cell nor the
        upstream that broke would be announced.
        """
        try:
            await self._materialize_upstreams(cell_id)
        except Exception as exc:  # noqa: BLE001 - reported as this cell's failure
            result = CellExecutionResult(
                cell_id=cell_id,
                success=False,
                error=str(exc),
                duration_ms=int((time.time() - start_time) * 1000),
                execution_method=execution_method,
            )
            self.session.apply_execution_result_metadata(cell_id, result)
            return result
        return None

    async def _materialize_upstreams(self, cell_id: str) -> None:
        """Ensure every upstream variable has a current artifact.

        Always calls ``execute_cell`` on each upstream: its provenance check
        makes an unchanged upstream a cache hit, while an edited one re-executes
        (an existence check would miss that).
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None or not cell.upstream_ids:
            return

        # Execute each upstream once even if it produces several referenced variables.
        executed_upstreams: set[str] = set()
        # Report the first failure only after trying every upstream, so all broken siblings
        # surface in one run instead of one round trip each.
        first_failure: tuple[str, CellExecutionResult] | None = None

        for upstream_id in cell.upstream_ids:
            if upstream_id in executed_upstreams:
                continue

            upstream_cell = self.session.notebook_state.get_cell(upstream_id)
            if upstream_cell is None:
                continue

            # Always materialize: execute_cell() returns at once on a cache hit. Capture the
            # error before the run clears it: a client shown it needs the replacing result.
            carried_error = upstream_cell.error is not None
            result = await self.execute_cell(
                upstream_id,
                upstream_cell.source,
            )
            if not result.success:
                # The language wrapper already recorded the error and cleared the output, as in a
                # direct run, so it reads as `error`, not never-ran.
                self.upstream_results.setdefault(upstream_id, result)
                if first_failure is None:
                    first_failure = (upstream_id, result)
                continue
            # Mark it ran. Otherwise the next staleness pass must infer it, conservatively for
            # languages with their own cache scheme, leaving e.g. a SQL upstream `idle` with no
            # result while its value is current and in use.
            upstream_cell.status = CellStatus.READY
            if carried_error:
                self.upstream_results.setdefault(upstream_id, result)
            executed_upstreams.add(upstream_id)

        if first_failure is not None:
            failed_id, failed_result = first_failure
            raise RuntimeError(
                f"Failed to materialise upstream cell {failed_id}: {failed_result.error}"
            )

    # ------------------------------------------------------------------
    # ④-a Load input blobs (guaranteed to exist after step ①)
    # ------------------------------------------------------------------

    def _load_input_blobs(
        self,
        cell_id: str,
        output_dir: Path,
        *,
        fanout_variant: str | None = None,
    ) -> dict[str, Any]:
        """Write upstream variable blobs from the artifact store into *output_dir*.

        Upstreams must already be materialized. A sweep read binds what cellgraph's
        ``variant_read`` says, the rule the key reads by: with ``fanout_variant`` set,
        the cell's own group as that variant's scalar; every other sweep read as a
        ``{variant: value}`` dict.
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            return {}

        artifact_mgr = self.session.get_artifact_manager()
        notebook_id = self.session.notebook_state.id
        # A value is a single-artifact spec (``{content_type, file, uri}``) or a sweep bundle
        # (``{kind: "sweep_dict", variants: {name: spec}}``).
        input_specs: dict[str, Any] = {}

        def _load_artifact_spec(artifact_id: str, file_stem: str) -> dict[str, str] | None:
            """Write one artifact's blob to *file_stem*; return its spec, or None if absent."""
            artifact = artifact_mgr.artifact_store.get_latest_version(artifact_id)
            if artifact is None:
                return None
            content_type = artifact.params.get("content_type") or "pickle/object"
            ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".pickle")
            input_file = output_dir / f"{safe_filename_stem(file_stem)}{ext}"
            self._write_ready_input_file(input_file, artifact_id, artifact.version, content_type)
            return {
                "content_type": content_type,
                "file": f"{safe_filename_stem(file_stem)}{ext}",
                "uri": f"strata://artifact/{artifact.id}@v={artifact.version}",
            }

        def _load_versioned_spec(uri: str, file_stem: str) -> dict[str, str] | None:
            """Resolve a pinned ``strata://artifact/{id}@v={n}`` URI to a file spec."""
            ref = uri.removeprefix("strata://artifact/")
            if "@v=" not in ref:
                return None
            art_id, _, ver = ref.partition("@v=")
            try:
                version = int(ver)
            except ValueError:
                return None
            artifact = artifact_mgr.artifact_store.get_artifact(art_id, version)
            if artifact is None:
                return None
            content_type = artifact.params.get("content_type") or "pickle/object"
            ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".pickle")
            input_file = output_dir / f"{safe_filename_stem(file_stem)}{ext}"
            self._write_ready_input_file(input_file, art_id, version, content_type)
            return {"content_type": content_type, "file": f"{safe_filename_stem(file_stem)}{ext}"}

        def _attach_injected(var_name: str, spec: dict[str, Any]) -> None:
            """Resolve a module/cell spec's injected upstream values to files for the harness."""
            if spec.get("content_type") != "module/cell":
                return
            try:
                descriptor = json.loads((output_dir / spec["file"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            injected = descriptor.get("injected") or {}
            resolved: dict[str, dict[str, str]] = {}
            for name, uri in injected.items():
                sub = _load_versioned_spec(uri, f"{var_name}__inj__{name}")
                if sub is None:
                    logger.error(
                        "Injected value '%s' for module export '%s' could not be resolved (%s).",
                        name,
                        var_name,
                        uri,
                    )
                    continue
                resolved[name] = sub
            if resolved:
                spec["injected"] = resolved

        analysis = self.session.analysis
        assert analysis is not None  # set with the DAG
        # An @each instance reads its item, and not the collection it came from.
        each_spec = analysis.cells[cell_id].annotations.get("each") if fanout_variant else None
        each_item = self._each_items.get(cell_id, {}).get(fanout_variant or "")
        if each_spec is not None and each_item is not None and each_item.value is not None:
            item_file = f"__each_{safe_filename_stem(each_spec['name'])}.json"
            (output_dir / item_file).write_bytes(each_item.value)
            input_specs[each_spec["name"]] = {"content_type": "json/object", "file": item_file}
        for upstream_id in cell.upstream_ids:
            upstream_cell = self.session.notebook_state.get_cell(upstream_id)
            if upstream_cell is None:
                continue

            # builtin_references holds builtin-shadowing names (``input``) the display-facing
            # list filters out. Only names wired from this upstream load, so a shadowed earlier
            # definer never overwrites the real producer's value.
            wired = self.session.wired_variables(cell_id, upstream_id)
            referenced_vars = [
                v for v in (*cell.references, *cell.builtin_references) if v in wired
            ]

            for var_name in referenced_vars:
                if each_spec is not None and var_name == each_spec["source"]:
                    continue
                artifact_id = f"nb_{notebook_id}_cell_{upstream_id}_var_{var_name}"
                try:
                    read = analysis.variant_read(cell_id, upstream_id, var_name, fanout_variant)
                    if read is not None:
                        # Partial on a missing variant. A failed one never gets here:
                        # _materialize_upstreams raises first. A collapse of a fan-out binds
                        # even with no instance; a group with no member left stays unbound.
                        if read.fanout and not read.scalar:
                            input_specs.setdefault(var_name, {"kind": "sweep_dict", "variants": {}})
                        for name in read.variants:
                            # A fan-out's instances live under ``@variant=`` subkeys of one cell.
                            vid = (
                                artifact_mgr.cell_artifact_id(upstream_id, var_name, variant=name)
                                if read.fanout
                                else artifact_id
                            )
                            stem = var_name if read.scalar else f"{var_name}__{name}"
                            spec = _load_artifact_spec(vid, stem)
                            if spec is None:
                                if not read.fanout:
                                    logger.error(
                                        "Sweep variant '%s' of '%s' has no artifact; dropping it.",
                                        name,
                                        var_name,
                                    )
                                continue
                            if read.scalar:
                                input_specs[var_name] = spec
                            else:
                                input_specs.setdefault(
                                    var_name, {"kind": "sweep_dict", "variants": {}}
                                )["variants"][name] = spec
                        continue

                    latest = artifact_mgr.artifact_store.get_latest_version(artifact_id)
                    if (
                        latest is not None
                        and latest.params.get("content_type") == each.CONTENT_TYPE
                    ):
                        # An @each cell's output: the {key: value} dict of its manifest's items.
                        blob = artifact_mgr.load_artifact_data(artifact_id, latest.version)
                        items = {}
                        for key, instance_id, version in each.manifest_refs(artifact_id, blob):
                            item_spec = _load_versioned_spec(
                                f"strata://artifact/{instance_id}@v={version}",
                                f"{var_name}__{each.instance_name(key)}",
                            )
                            if item_spec is None:
                                raise each.EachError(
                                    f"`{var_name}`: item {key!r} is {instance_id}@v={version}, "
                                    "which is gone"
                                )
                            items[key] = item_spec
                        input_specs[var_name] = {"kind": "sweep_dict", "variants": items}
                        continue

                    spec = _load_artifact_spec(artifact_id, var_name)
                    if spec is None:
                        # An import binding isn't always materialized (remote workers don't ship
                        # module blobs back) and the consumer re-imports it. Other gaps are errors.
                        if var_name in imported_names(upstream_cell.source):
                            logger.debug(
                                "Module-typed upstream '%s' has no artifact "
                                "(producer cell %s); the consuming cell re-imports it.",
                                var_name,
                                upstream_id,
                            )
                        else:
                            logger.error(
                                "Artifact %s still missing after upstream "
                                "materialisation; skipping variable '%s'.",
                                artifact_id,
                                var_name,
                            )
                        continue

                    _attach_injected(var_name, spec)
                    input_specs[var_name] = spec
                    logger.info(
                        "Loaded input %s from artifact store (%s)",
                        var_name,
                        artifact_id,
                    )
                except each.EachError:
                    raise  # a missing value would otherwise read as a NameError in the cell
                except Exception:
                    logger.exception(
                        "Failed to load input %s from artifact store",
                        var_name,
                    )

        return input_specs

    # ------------------------------------------------------------------
    # ⑤ Store output artifacts
    # ------------------------------------------------------------------

    def _release_nothing(self, cell_id: str, local_run: LocalRun | None) -> bool:
        """Complete a run that stores no variable: an empty ``final`` release ends its build."""
        if local_run is None:
            return True
        try:
            self._local_runs().finalize(local_run, [])
        except Exception:
            logger.exception("Failed to complete the run of cell %s", cell_id)
        return True

    def _store_outputs(
        self,
        cell_id: str,
        output_dir: Path,
        provenance_hash: str,
        input_hashes: list[str],
        *,
        source_hash: str = "",
        source: str = "",
        env_hash: str = "",
        variant: str | None = None,
        build_env: str = "",
        build_duration_ms: float = 0.0,
        hardware: dict[str, Any] | None = None,
        collected: dict[str, bytes | Exception] | None = None,
        local_run: LocalRun | None = None,
        stochastic: bool = False,
        run_outputs: dict[str, Any] | None = None,
    ) -> bool:
        """Persist consumed output variables as artifacts; True iff all were stored.

        With ``store_leaf_outputs``, a leaf cell's variables are stored too, best-effort.
        *collected* holds folders already collected by :meth:`_collect_output_dirs`.
        *run_outputs* is what the harness reported, which says why a value left no file.

        Raises:
            OutputNotSaved: a variable another cell reads left no file.

        With ``variant`` set, ids get an ``@variant={name}`` suffix so fan-out
        instances do not collide. With a *local_run*, the outputs become its ``final``
        release; *stochastic* keys them by the run (``# @stochastic``) so every run's
        persist beside the cached key, which none of them matches.
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None or self.session.dag is None:
            return self._release_nothing(cell_id, local_run)

        artifact_mgr = self.session.get_artifact_manager()
        input_versions = self._input_refs(cell_id, variant)
        consumed_vars = self.session.dag.consumed_variables.get(cell_id, set())

        try:
            output_files = list(output_dir.iterdir())
        except Exception:
            output_files = []

        logger.info(
            "_store_outputs %s: consumed_vars=%s output_files=%s",
            cell_id,
            consumed_vars,
            [f.name for f in output_files],
        )

        # Kept for the record only: no input resolves through them, so a value that did not
        # serialize is skipped rather than failing the cell.
        leaf_vars = set(cell.defines) if not consumed_vars and self.store_leaf_outputs else set()
        if not consumed_vars and not leaf_vars:
            return self._release_nothing(cell_id, local_run)

        all_stored = True
        staged: list[StagedVersion] = []
        staged_names: list[tuple[str, str]] = []
        staged_outputs: dict[str, StagedOutput] = {}

        # ``.rds`` is R-only (harness.R's RDS fallback), with its own content_type so
        # consumers recognize it without scanning bytes.
        content_type_map = {
            ".arrow": "arrow/ipc",
            ".json": "json/object",
            ".pickle": "pickle/object",
            ".module.json": "module/import",
            ".cell_module.json": "module/cell",
            ".cell_instance.pickle": "module/cell-instance",
            ".rds": "application/x-r-rds",
        }
        # Order matters: module/cell/import values write a specific descriptor AND a generic
        # ``.pickle`` sidecar, so the descriptor must match first.
        output_exts = [
            ".arrow",
            ".cell_module.json",
            ".cell_instance.pickle",
            ".module.json",
            ".json",
            ".pickle",
            ".rds",
        ]

        def output_key(var_name: str) -> str:
            if stochastic and local_run is not None:
                return run_output_key(provenance_hash, local_run.build_id, var_name)
            return derive_subkey(provenance_hash, var_name)

        output_dir_names = set(parse_annotations(source or cell.source).output_dirs)
        for var_name in consumed_vars | leaf_vars:
            if var_name in output_dir_names:
                try:
                    staged.append(
                        self._stage_output_dir(
                            cell_id,
                            var_name,
                            output_dir,
                            provenance_hash,
                            input_versions,
                            source_hash=source_hash,
                            source=source,
                            env_hash=env_hash,
                            variant=variant,
                            collected=(collected or {}).get(var_name),
                            output_key=output_key(var_name),
                            computation_key=provenance_hash,
                        )
                    )
                except UnsafeRunFile:
                    artifact_mgr.discard_cell_outputs(staged)
                    raise
                except Exception:
                    logger.exception("Failed to keep output folder %s of %s", var_name, cell_id)
                    all_stored = False
                    break
                staged_names.append((var_name, output_dirs.CONTENT_TYPE))
                continue
            # Python writes a case-safe stem (``Data-<hash>.json``), R the plain name. Try every
            # safe-stem candidate, then the plain name only if no safe-stem file exists at all
            # (never per-ext), so a case-differing sibling like ``data.arrow`` is never taken
            # for ``Data``.
            safe = safe_filename_stem(var_name)
            stems = [safe] if safe == var_name else [safe, var_name]
            output_file: Path | None = None
            ext = ""
            for stem in stems:
                for candidate_ext in output_exts:
                    candidate = output_dir / f"{stem}{candidate_ext}"
                    if candidate.exists():
                        output_file, ext = candidate, candidate_ext
                        break
                if output_file is not None:
                    break

            if output_file is None and var_name in leaf_vars:
                continue
            if output_file is None:
                logger.warning(
                    "_store_outputs %s: no output file for consumed var %s "
                    "(looked for %s{.arrow,.json,.pickle,.cell_module.json,"
                    ".cell_instance.pickle,.rds} in %s)",
                    cell_id,
                    var_name,
                    safe,
                    output_dir,
                )
                artifact_mgr.discard_cell_outputs(staged)
                raise output_not_saved(var_name, run_outputs)

            try:
                blob_data = read_run_file(output_dir, output_file.name)

                content_type = content_type_map.get(ext, "pickle/object")
                var_provenance = output_key(var_name)

                staged.append(
                    artifact_mgr.stage_cell_output(
                        cell_id=cell_id,
                        variable_name=var_name,
                        blob_data=blob_data,
                        content_type=content_type,
                        provenance_hash=var_provenance,
                        input_versions=input_versions,
                        source_hash=source_hash,
                        source=source,
                        env_hash=env_hash,
                        variant=variant,
                        build_env=build_env,
                        build_duration_ms=build_duration_ms,
                        hardware=hardware,
                        extra_params=self._fetch_params(cell_id),
                        computation_key=provenance_hash,
                    )
                )
                staged_names.append((var_name, content_type))
                staged_outputs[var_name] = StagedOutput(
                    artifact_id=staged[-1].artifact_id,
                    version=staged[-1].version,
                    blob=blob_data,
                    content_type=content_type,
                    provenance_hash=var_provenance,
                )
            except UnsafeRunFile:
                artifact_mgr.discard_cell_outputs(staged)
                raise
            except Exception:
                logger.exception(
                    "Failed to store output %s for cell %s",
                    var_name,
                    cell_id,
                )
                if var_name in leaf_vars:
                    continue
                all_stored = False
                break

        # One run's outputs become current together or not at all: a downstream cell
        # must never read one variable from this run and another from an earlier one.
        table_writes = parse_annotations(source or cell.source).table_writes
        # An @each cell writes once, over every item, with its manifest.
        if variant in self._each_items.get(cell_id, {}):
            table_writes = []
        if all_stored and table_writes:
            # Before the outputs become current: a failed write keeps none of them, so the
            # next run is not a cache hit and writes again.
            try:
                self._write_cell_tables(table_writes, staged_outputs)
            except TableWriteError:
                artifact_mgr.discard_cell_outputs(staged)
                raise

        try:
            if not all_stored:
                artifact_mgr.discard_cell_outputs(staged)
                return False
            # A local run's outputs are its final release, which completes its build.
            stored = (
                self._local_runs().finalize(local_run, staged)
                if local_run is not None
                else artifact_mgr.finalize_cell_outputs(staged)
            )
        except Exception:
            logger.exception("Failed to store the outputs of cell %s", cell_id)
            artifact_mgr.discard_cell_outputs(staged)
            # A leaf's record is best-effort and never fails the cell.
            return not consumed_vars

        for (var_name, content_type), artifact_version in zip(staged_names, stored, strict=True):
            if var_name not in leaf_vars:
                uri = f"strata://artifact/{artifact_version.id}@v={artifact_version.version}"
                cell.artifact_uris[var_name] = uri
                cell.artifact_uri = uri  # backward compat
            logger.info(
                "Stored output %s for cell %s as %s@v=%d (%d bytes, %s)",
                var_name,
                cell_id,
                artifact_version.id,
                artifact_version.version,
                artifact_version.byte_size or 0,
                content_type,
            )

        return True

    def _write_input_file(
        self, path: Path, artifact_id: str, version: int, content_type: str
    ) -> bool:
        """Copy an input's bytes for the harness; ``False`` if its blob is gone.

        An output folder is rebuilt beside them, from its manifest, which is parsed whole.
        """
        artifact_mgr = self.session.get_artifact_manager()
        if not artifact_mgr.copy_blob(artifact_id, version, path):
            return False
        if content_type == output_dirs.CONTENT_TYPE:
            folder = path.with_name(path.name.removesuffix(output_dirs.EXTENSION) + ".dir")
            # The notebook's store is untenanted.
            output_dirs.materialize(
                artifact_mgr.artifact_store, artifact_id, path.read_bytes(), folder, tenant=None
            )
        return True

    def _write_ready_input_file(
        self, path: Path, artifact_id: str, version: int, content_type: str
    ) -> None:
        """As ``_write_input_file``; raises ``ValueError`` if the version is not ready or gone."""
        self.session.get_artifact_manager().ensure_readable(artifact_id, version)
        if not self._write_input_file(path, artifact_id, version, content_type):
            raise ValueError(f"Blob data not found for {artifact_id}@v={version}")

    def _collect_output_dir(self, cell_id: str, name: str, output_dir: Path | None) -> bytes:
        """Store a ``# @output_dir`` folder's files and return its manifest.

        Files a worker uploaded during the run are in already; its result lists which of
        them the folder still holds, and without a result (a lost worker) all count.
        """
        received = self._received_output_files.get(cell_id, {}).get(name, {})
        streamed = output_dirs.read_streamed(output_dir) if output_dir is not None else None
        if streamed is not None:
            listed = set(streamed.get(name, ()))
            received = {rel: entry for rel, entry in received.items() if rel in listed}
        return output_dirs.collect(
            self.session.get_artifact_manager(),
            cell_id,
            output_dir,
            name,
            received,
        )

    async def _collect_output_dirs_off_loop(
        self, cell_id: str, source: str, output_dir: Path
    ) -> dict[str, bytes | Exception] | None:
        """:meth:`_collect_output_dirs` in a thread, or ``None`` for a cell with no folder.

        Called before the run's provenance is recorded and its outputs stored, since
        nothing may see those two apart.
        """
        if not parse_annotations(source).output_dirs:
            return None
        return await anyio.to_thread.run_sync(
            self._collect_output_dirs, cell_id, source, output_dir
        )

    def _collect_output_dirs(
        self, cell_id: str, source: str, output_dir: Path
    ) -> dict[str, bytes | Exception]:
        """Each folder's manifest, or what collecting it raised: run in a thread, since a
        folder can hold gigabytes. Staging reports a failure as if it had collected it."""
        collected: dict[str, bytes | Exception] = {}
        for name in parse_annotations(source).output_dirs:
            try:
                collected[name] = self._collect_output_dir(cell_id, name, output_dir)
            except Exception as exc:
                collected[name] = exc
        return collected

    def _stage_output_dir(
        self,
        cell_id: str,
        name: str,
        output_dir: Path | None,
        provenance_hash: str,
        input_versions: dict[str, str],
        *,
        source_hash: str,
        source: str,
        env_hash: str,
        variant: str | None = None,
        collected: bytes | Exception | None = None,
        output_key: str | None = None,
        computation_key: str | None = None,
    ) -> StagedVersion:
        """Keep a ``# @output_dir`` folder's files and stage its manifest as the output.

        *collected* is the folder already collected by :meth:`_collect_output_dirs`.
        *output_key* replaces the output's derived key (a ``# @stochastic`` run's).
        """
        artifact_mgr = self.session.get_artifact_manager()
        if isinstance(collected, Exception):
            raise collected
        manifest = (
            collected
            if collected is not None
            else self._collect_output_dir(cell_id, name, output_dir)
        )
        self._folder_listings.setdefault(cell_id, {})[name] = output_dirs.listing(name, manifest)
        return artifact_mgr.stage_cell_output(
            cell_id=cell_id,
            variable_name=name,
            blob_data=manifest,
            content_type=output_dirs.CONTENT_TYPE,
            provenance_hash=output_key or derive_subkey(provenance_hash, name),
            input_versions=input_versions,
            source_hash=source_hash,
            source=source,
            env_hash=env_hash,
            variant=variant,
            computation_key=computation_key,
        )

    def _restore_manifest(self, run: LocalRun, output_dir: Path) -> dict[str, Any] | None:
        """The manifest's ``restore`` entry (see ``cell_paths/python.py``)."""
        return restore_manifest(self.session, self._write_ready_input_file, run, output_dir)

    def _keep_failed_output_dirs(
        self,
        cell_id: str,
        source: str,
        output_dir: Path | None,
        provenance_hash: str,
        env_hash: str,
        collected: dict[str, bytes | Exception] | None = None,
        *,
        variant: str | None = None,
    ) -> None:
        """Keep what a failed run left in its ``# @output_dir`` folders, as failed versions.

        ``output_dir`` is ``None`` when no result came back: what a worker uploaded is all.
        *collected* holds folders already collected by :meth:`_collect_output_dirs`.
        *variant* is the ``# @per_variant`` instance that failed, kept under its own id.

        A failed version is never a cache hit or an input; lineage shows it, and its files
        are listed and downloaded through ``GET .../cells/{cell_id}/folders/{name}``. Best
        effort: the run has already failed.
        """
        names = parse_annotations(source).output_dirs
        if not names:
            return
        artifact_mgr = self.session.get_artifact_manager()
        input_versions = self._input_refs(cell_id)
        for name in names:
            try:
                staged = self._stage_output_dir(
                    cell_id,
                    name,
                    output_dir,
                    provenance_hash,
                    input_versions,
                    source_hash=compute_source_hash(source),
                    source=source,
                    env_hash=env_hash,
                    variant=variant,
                    collected=(collected or {}).get(name),
                )
                artifact_mgr.discard_cell_outputs([staged])
            except Exception:
                logger.exception("Failed to keep output folder %s of failed %s", name, cell_id)

    def _write_cell_tables(
        self, table_writes: list[TableWriteSpec], outputs: dict[str, StagedOutput]
    ) -> None:
        """Write ``@write_table`` variables with the server's catalogs and this notebook's env."""
        written = write_cell_tables(
            table_writes,
            outputs,
            config=self._lake_config(),
            env=dict(self.session.notebook_state.env),
            notebook=self.session.path,
        )
        for write in written:
            logger.info("Wrote table %s (snapshot %d)", write.table, write.snapshot_id)

    def _table_write_refusal(self, source: str) -> str | None:
        """Why this run may not write its ``@write_table`` tables, checked before it runs."""
        try:
            authorize_table_writes(
                parse_annotations(source).table_writes,
                config=self._lake_config(),
                env=dict(self.session.notebook_state.env),
                notebook=self.session.path,
            )
        except TableWriteError as exc:
            return str(exc)
        return None

    def _store_console_outputs(
        self,
        cell_id: str,
        provenance_hash: str,
        stdout: str,
        stderr: str,
        input_hashes: list[str],
        *,
        source_hash: str = "",
        source: str = "",
        env_hash: str = "",
        display_count: int | None = None,
        duration_ms: float,
        variant: str | None = None,
    ) -> None:
        """Persist a cell's console output as a provenance-keyed artifact.

        So a hit replays what its own run printed, and a cell that only prints can
        cache-hit at all. Stored even when both streams are empty: that is what the run
        printed, and a leaf has no other artifact, so the console is the record that it
        ran under this provenance (cache hit, ready on reopen).
        ``display_count`` records how many displays the run made, which a run with
        none leaves no display artifact to say. ``duration_ms`` is what the run cost,
        hence what a hit on this record saves.
        """
        artifact_mgr = self.session.get_artifact_manager()
        input_versions = self._input_refs(cell_id, variant)
        blob = json.dumps({"stdout": stdout, "stderr": stderr}).encode("utf-8")
        params = self._fetch_params(cell_id)
        if display_count is not None:
            params["display_count"] = str(display_count)
        artifact_mgr.store_cell_output(
            cell_id=cell_id,
            variable_name="__console__",
            blob_data=blob,
            content_type="json/object",
            provenance_hash=derive_subkey(provenance_hash, "__console__"),
            input_versions=input_versions,
            source_hash=source_hash,
            source=source,
            env_hash=env_hash,
            build_duration_ms=duration_ms,
            extra_params=params,
            variant=variant,
        )

    def _store_mount_writes(
        self,
        cell_id: str,
        provenance_hash: str,
        writes: list[MountWrite],
        *,
        source_hash: str,
        source: str,
        env_hash: str,
    ) -> None:
        """Record what a run wrote through each read-write mount, as ``__mount__<name>``.

        A read-write cell never cache-hits, so this is the run's record of its side effect:
        the files it created or changed, with sizes and sha256 digests, linked to the inputs
        it read. Each run adds a version, so earlier runs' records stay readable.
        """
        artifact_mgr = self.session.get_artifact_manager()
        input_versions = self._input_refs(cell_id)
        for write in writes:
            blob = json.dumps(
                {"uri": write.uri, "files": [f.model_dump() for f in write.files]},
                sort_keys=True,
            ).encode("utf-8")
            label = f"__mount__{write.mount}"
            try:
                artifact_mgr.store_cell_output(
                    cell_id=cell_id,
                    variable_name=label,
                    blob_data=blob,
                    content_type="json/object",
                    provenance_hash=derive_subkey(provenance_hash, label),
                    input_versions=input_versions,
                    source_hash=source_hash,
                    source=source,
                    env_hash=env_hash,
                    extra_params={"mount_uri": write.uri},
                )
            except Exception:
                # The files are already written and synced; losing the record is not a
                # reason to fail the run.
                logger.exception("Failed to record writes through mount %s", write.mount)

    def _record_good_run(
        self,
        cell_id: str,
        provenance_hash: str,
        source_hash: str,
        env_hash: str,
        prov: _CellProvenance,
        writes: list[MountWrite],
    ) -> None:
        """Record a run that stored its outputs and synced its mounts as the cell's last good one.

        Not before the sync-back: a run that fails there is not the last good run, and leaves
        the reads, tables and writes indexes as they were.
        """
        try:
            self.session.record_successful_execution_provenance(
                cell_id,
                provenance_hash,
                source_hash,
                env_hash,
                table_reads=prov.table_reads,
                reads=prov.reads(*self._moved(cell_id)),
                key_parts=prov.key_parts,
                writes=_writes_entry(writes),
            )
        except (OSError, NotebookQuiesced):
            # The outputs are stored and the files synced; losing the record is not a
            # reason to fail the run.
            logger.exception("Failed to record the run of cell %s", cell_id)

    def _store_result_displays(
        self,
        result: CellExecutionResult,
        output_dir: Path,
        provenance_hash: str,
        input_hashes: list[str],
        *,
        source_hash: str,
        source: str,
        env_hash: str,
        variant: str | None = None,
    ) -> None:
        """Store a single run's displays and put what was stored on its *result*."""
        result.display_outputs = self._store_display_outputs(
            result.cell_id,
            output_dir,
            provenance_hash,
            input_hashes,
            result.display_outputs,
            source_hash=source_hash,
            source=source,
            env_hash=env_hash,
            variant=variant,
        )
        result.display_output = result.display_outputs[-1] if result.display_outputs else None

    def _store_inline_display_outputs(
        self,
        cell_id: str,
        provenance_hash: str,
        display_outputs: list[dict[str, Any]],
        *,
        source_hash: str = "",
        source: str = "",
        env_hash: str = "",
    ) -> list[dict[str, Any]]:
        """Persist displays a cell built in memory rather than as output files.

        For SQL cells, which render markdown in-process and have no output
        directory. The artifacts match ``_store_display_outputs``: keyed on the
        generic provenance under ``__display__{i}`` with their own description.
        """
        if not display_outputs:
            return []

        artifact_mgr = self.session.get_artifact_manager()
        input_versions = self._input_refs(cell_id)
        stored: list[dict[str, Any]] = []

        for index, display_output in enumerate(display_outputs):
            text = display_output.get("markdown_text")
            content_type = str(display_output.get("content_type", "")).strip()
            if not isinstance(text, str) or content_type != "text/markdown":
                stored.append(dict(display_output))
                continue
            blob = text.encode()
            # Set before the artifact records its description, so a cached display reports its
            # size, not 0.
            entry = {**display_output, "bytes": len(blob)}
            display_provenance = derive_subkey(provenance_hash, f"__display__{index}")
            canonical_id = artifact_mgr.cell_artifact_id(cell_id, f"__display__{index}")
            # A cache hit comes through here too; reuse the stored version instead of writing an
            # identical blob under a new one each time.
            canonical = artifact_mgr.artifact_store.get_latest_version(canonical_id)
            if canonical is not None and canonical.provenance_hash == display_provenance:
                version = canonical.version
            else:
                version = artifact_mgr.store_cell_output(
                    cell_id=cell_id,
                    variable_name=f"__display__{index}",
                    blob_data=blob,
                    content_type=content_type,
                    provenance_hash=display_provenance,
                    input_versions=input_versions,
                    source_hash=source_hash,
                    source=source,
                    env_hash=env_hash,
                    extra_params={
                        **display_metadata_params(entry, len(display_outputs)),
                        **self._fetch_params(cell_id),
                    },
                ).version
            entry["artifact_uri"] = f"strata://artifact/{canonical_id}@v={version}"
            stored.append(entry)

        return stored

    def _store_display_outputs(
        self,
        cell_id: str,
        output_dir: Path,
        provenance_hash: str,
        input_hashes: list[str],
        display_outputs: list[dict[str, Any]] | None,
        *,
        source_hash: str = "",
        source: str = "",
        env_hash: str = "",
        variant: str | None = None,
    ) -> list[dict[str, Any]]:
        """Persist ordered cell display outputs as canonical artifacts.

        The run's ``# @output_dir`` listings come first, so the cell's last display stays last.
        A fan-out instance's are stored under its *variant*.
        """
        listings = self._folder_listings.pop(cell_id, {})
        display_outputs = [
            {
                "content_type": "text/markdown",
                "markdown_text": listings[name],
                "bytes": len(listings[name].encode("utf-8")),
                "preview": None,
            }
            for name in sorted(listings)
        ] + list(display_outputs or [])
        if not display_outputs:
            return []

        artifact_mgr = self.session.get_artifact_manager()
        input_versions = self._input_refs(cell_id, variant)
        stored_displays: list[dict[str, Any]] = []

        for index, display_output in enumerate(display_outputs):
            file_name = str(display_output.get("file", "")).strip()
            content_type = str(display_output.get("content_type", "")).strip()
            markdown_text = display_output.get("markdown_text")
            if not file_name and content_type == "text/markdown" and isinstance(markdown_text, str):
                # Made by the server (a folder's listing), not written by the cell.
                blob_data = markdown_text.encode("utf-8")
            else:
                if not file_name or not content_type:
                    continue
                if not (output_dir / file_name).exists():
                    continue
                blob_data = read_run_file(output_dir, file_name)
            row_count = display_output.get("rows")
            display_provenance = derive_subkey(provenance_hash, f"__display__{index}")
            artifact_version = artifact_mgr.store_cell_output(
                cell_id=cell_id,
                variable_name=f"__display__{index}",
                blob_data=blob_data,
                content_type=content_type,
                row_count=row_count if isinstance(row_count, int) else None,
                variant=variant,
                provenance_hash=display_provenance,
                input_versions=input_versions,
                source_hash=source_hash,
                source=source,
                env_hash=env_hash,
                extra_params={
                    **display_metadata_params(display_output, len(display_outputs)),
                    **self._fetch_params(cell_id),
                },
            )
            display_uri = f"strata://artifact/{artifact_version.id}@v={artifact_version.version}"
            stored_display = dict(display_output)
            stored_display["artifact_uri"] = display_uri
            stored_display["computed_at"] = artifact_version.created_at
            stored_displays.append(stored_display)

        return stored_displays

    # ------------------------------------------------------------------
    # Harness helpers
    # ------------------------------------------------------------------

    def _harness_command(
        self, manifest_path: Path, venv_python: Path, harness_user: HarnessUser | None
    ) -> list[str] | None:
        """The cold Python harness's argv (see ``cell_paths/python.py``); a test replaces it."""
        return python_harness_command(
            self.harness_path, self.session.path, manifest_path, venv_python, harness_user
        )

    async def _run_harness(
        self,
        manifest_path: Path,
        venv_python: Path | None,
        timeout_seconds: float,
        *,
        run_token: str | None = None,
        on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Run the Python harness (see ``cell_paths/python.py``)."""
        return await run_harness(
            self.session,
            self._harness_command,
            self._harness_env,
            manifest_path,
            venv_python,
            timeout_seconds,
            run_token=run_token,
            on_event=on_event,
        )

    async def _run_r_harness(
        self,
        manifest_path: Path,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Run the R harness (see ``cell_paths/r.py``)."""
        return await run_r_harness(
            self.session.path,
            self.r_harness_path,
            self._harness_env,
            manifest_path,
            timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Loop cell execution (sequential, fresh subprocess per iteration)
    # ------------------------------------------------------------------

    async def _execute_loop_cell(
        self,
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
        """Execute a loop cell (see ``cell_paths/loop.py``)."""
        return await run_loop_cell(
            self.session,
            LoopContext(
                materialize_upstreams_or_failure=self._materialize_upstreams_or_failure,
                resolve_cell_mount_specs=self._resolve_cell_mount_specs,
                fingerprint_mounts=self._fingerprint_mounts,
                resolve_runtime_env=self._resolve_effective_runtime_env,
                compute_provenance=self._compute_cell_provenance,
                manifest_tables=self._manifest_tables,
                moved=self._moved,
                promote_reverted_outputs=self._promote_reverted_outputs,
                restore_gone_tables=self._restore_gone_tables,
                canonical_output=self._canonical_output,
                table_write_refusal=self._table_write_refusal,
                prepare_mounts=self._prepare_mounts,
                load_input_blobs=self._load_input_blobs,
                keep_failed_output_dirs=self._keep_failed_output_dirs,
                write_manifest=self._write_manifest,
                run_token=self._run_token,
                run_harness=self._run_harness,
                input_refs=self._input_refs,
                write_cell_tables=self._write_cell_tables,
                store_display_outputs=self._store_display_outputs,
                store_console_outputs=self._store_console_outputs,
                folder_listings=self._folder_listings,
                on_iteration_complete=self.on_iteration_complete,
            ),
            cell_id,
            source,
            loop,
            timeout_seconds,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
            folders=folders,
        )

    def _parse_result(
        self,
        cell_id: str,
        result: dict,
        duration_ms: float,
        execution_method: str = "cold",
    ) -> CellExecutionResult:
        """Parse harness result into a CellExecutionResult."""
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error")
            stderr = result.get("stderr", "")
            detected = _detect_missing_module(error_msg, stderr)
            suggest_lang, suggest_pkg = detected if detected else (None, None)
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                stdout=result.get("stdout", ""),
                stderr=stderr,
                error=error_msg,
                traceback=result.get("traceback") or None,
                traceback_full=result.get("traceback_full") or None,
                duration_ms=duration_ms,
                execution_method=execution_method,
                suggest_install=suggest_pkg,
                suggest_install_language=suggest_lang,
            )

        outputs = {}
        variables = result.get("variables", {})
        for var_name, output_meta in variables.items():
            if "error" in output_meta:
                outputs[var_name] = {
                    "content_type": "error",
                    "error": output_meta["error"],
                    "type": output_meta.get("type", "unknown"),
                }
            else:
                outputs[var_name] = output_meta

        mutation_warnings = result.get("mutation_warnings", [])
        raw_displays = result.get("displays")
        display_outputs = (
            [display for display in raw_displays if isinstance(display, dict)]
            if isinstance(raw_displays, list)
            else []
        )
        if not display_outputs:
            display_output = outputs.get("_")
            if isinstance(display_output, dict):
                display_outputs = [display_output]

        return CellExecutionResult(
            cell_id=cell_id,
            success=True,
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            outputs=outputs,
            display_outputs=display_outputs,
            duration_ms=duration_ms,
            execution_method=execution_method,
            mutation_warnings=mutation_warnings,
        )

    # ------------------------------------------------------------------
    # Run-all batching
    # ------------------------------------------------------------------

    def _batch_context(self) -> BatchContext:
        return BatchContext(
            harness_path=self.harness_path,
            harness_env=self._harness_env,
            work_dir_fields=self._work_dir_fields,
            run_token=self._run_token,
            compute_provenance=self._compute_cell_provenance,
            canonical_output=self._canonical_output,
            input_refs=self._input_refs,
            moved=self._moved,
            promote_reverted_outputs=self._promote_reverted_outputs,
            restore_gone_tables=self._restore_gone_tables,
            promote_reverted_leaf=self._promote_reverted_leaf,
            pull_from_team_store=self._pull_from_team_store,
            push_to_team_store=self._push_to_team_store,
            store_outputs=self._store_outputs,
            store_console_outputs=self._store_console_outputs,
            store_display_outputs=self._store_display_outputs,
            write_input_file=self._write_input_file,
        )

    async def _run_batch(
        self,
        cell_specs: list[dict[str, Any]],
        *,
        use_cache: bool,
        batch_timeout_seconds: float,
        cell_timeout_seconds: float = DEFAULT_CELL_TIMEOUT_SECONDS,
        on_cell_event: Callable[[BatchCellResult], Awaitable[None]] | None = None,
        on_cell_start: Callable[[str], Awaitable[None]] | None = None,
        provenance: Mapping[str, _CellProvenance] | None = None,
    ) -> BatchExecutionResult:
        """Run a sequence of cells in one harness subprocess (see ``cell_paths/batch.py``)."""
        return await run_batch(
            self.session,
            self._batch_context(),
            cell_specs,
            use_cache=use_cache,
            batch_timeout_seconds=batch_timeout_seconds,
            cell_timeout_seconds=cell_timeout_seconds,
            on_cell_event=on_cell_event,
            on_cell_start=on_cell_start,
            provenance=provenance,
        )


# ---------------------------------------------------------------------------
# Batch helpers (module-level)
# ---------------------------------------------------------------------------


# Display-output keys valid only for the run in hand: the harness's temp file name,
# the inline data URL and markdown text hydration rebuilds, and the store's URI.
_DISPLAY_TRANSIENT_KEYS = frozenset({"file", "inline_data_url", "markdown_text", "artifact_uri"})


def display_metadata_params(display_output: dict[str, Any], count: int) -> dict[str, str]:
    """What a display artifact records about itself, as transform params.

    Stored with the bytes so a cache hit serves this value's own preview, not
    whatever the cell shows now. ``count`` is how many displays the run
    produced, so a restore brings back the whole set.
    """
    metadata = {k: v for k, v in display_output.items() if k not in _DISPLAY_TRANSIENT_KEYS}
    return {
        "display": json.dumps(metadata, sort_keys=True, default=str),
        "display_count": str(count),
    }


# The variable a cell output's id names, before any ``@ckpt=`` / ``@variant=`` label.
# ---------------------------------------------------------------------------
# Batchability + partitioning
# ---------------------------------------------------------------------------


def _each_output_read(dag: Any, cell: Any) -> str | None:
    """The first variable *cell* reads from a ``# @each`` cell, if it reads one."""
    if dag is None:
        return None
    for ref in (*cell.references, *cell.builtin_references):
        producer = dag.read_producer(cell.id, ref)
        if isinstance(producer, str) and producer in dag.each_cells:
            return ref
    return None


def is_cell_batchable(executor: CellExecutor, cell: Any) -> bool:
    """Return whether ``cell`` is eligible for run-all batching.

    Delegates to the language executor's ``is_batchable`` (only Python can be).
    Sweep members and sweep consumers never batch: in a shared namespace
    sibling variants overwrite each other and the ``{variant: value}`` dict
    never forms.
    """
    from strata.notebook.languages import get_language_executor

    # A batch never sees a fetch's bytes; a fetching cell runs single-cell, where the
    # fetch is checked and injected.
    annotations = parse_annotations(cell.source)
    if annotations.fetches or annotations.datasets or annotations.each is not None:
        return False

    dag = executor.session.dag
    # Run single-cell, where the gate refuses it and Run All blocks its dependents.
    if (
        blocking_diagnostic(cell, executor.session.notebook_state, executor.session.analysis)
        is not None
    ):
        return False
    if dag is not None:
        modes = executor.session.notebook_state.variant_modes
        if cell.variant_group and modes.get(cell.variant_group) == "sweep":
            return False
        if any(
            isinstance(dag.read_producer(cell.id, ref), SweepProducer)
            for ref in (*cell.references, *cell.builtin_references)
        ):
            return False
        # An @each output is a manifest the single-cell loader expands.
        if _each_output_read(dag, cell) is not None:
            return False

    return get_language_executor(cell.language).is_batchable(cell, executor)


def partition_batchable_runs(
    executor: CellExecutor, cells: list[Any]
) -> list[tuple[str, list[Any]]]:
    """Group consecutive batchable cells, in notebook order.

    Returns ``[(kind, [cell, ...]), ...]`` with ``kind`` ``"batch"`` (one or more
    cells) or ``"single"`` (exactly one).
    """
    runs: list[tuple[str, list[Any]]] = []
    current: list[Any] = []

    for cell in cells:
        if is_cell_batchable(executor, cell):
            current.append(cell)
        else:
            if current:
                runs.append(("batch", current))
                current = []
            runs.append(("single", [cell]))

    if current:
        runs.append(("batch", current))

    return runs
