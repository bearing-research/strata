"""Cell executor: materialize a notebook cell.

``execute_cell`` recursively materializes upstream inputs, computes provenance
(``sha256(sorted_input_hashes + source_hash + env_hash)``), returns a cache hit
if one exists, and otherwise runs the harness and stores every consumed variable.
It works on any cell; the cascade planner only previews what will run.

This is deliberately a separate pipeline from Core's ``materialize`` SDK
(``POST /v1/materialize``): cells are in-process, multi-output and
source-as-transform. The two share only the artifact store
(``find_by_provenance`` / ``put``) and ``notebook.provenance.derive_subkey``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urljoin, urlparse, urlsplit, urlunparse

import httpx

# Stdlib json, not orjson: executor.py is imported via ``strata.config`` and must load
# with core deps only (orjson is in the ``[notebook]`` extra, absent from the Docker
# image). Batch frames are small dicts.
from strata.artifact_store import ArtifactVersion, StagedVersion, get_artifact_store
from strata.artifact_store import TransformSpec as ArtifactTransformSpec
from strata.blob_store import BLOB_STREAM_CHUNK_BYTES
from strata.notebook import console_relay
from strata.notebook.analyzer import imported_names
from strata.notebook.annotations import CellAnnotations, LoopAnnotation, parse_annotations
from strata.notebook.credentials import CredentialResolver
from strata.notebook.dag import SweepProducer
from strata.notebook.dependencies import UV_NOT_FOUND_MESSAGE, resolve_uv, uv_env
from strata.notebook.env import compute_execution_env_hash, narrow_env_for_provenance
from strata.notebook.harness_user import (
    HarnessUser,
    LocalExecutionRefused,
    hand_over,
    identity_env,
    resolve_harness_user,
    spawn_kwargs,
)
from strata.notebook.immutability import MutationWarning
from strata.notebook.models import (
    CellLanguage,
    CellOutput,
    CellStatus,
    CellTestCase,
    CellTestResult,
    DatasetSpec,
    FetchSpec,
    MountMode,
    MountSpec,
    TableSpec,
    WorkerBackendType,
)
from strata.notebook.module_export import build_module_export_plan, runtime_binding_names
from strata.notebook.mounts import (
    MountCredentials,
    MountResolver,
    ResolvedMount,
    mount_fingerprint,
    resolve_cell_mounts,
)
from strata.notebook.process_tree import (
    SUBPROCESS_LINE_LIMIT,
    subprocess_kwargs_for_new_group,
    terminate_subprocess_tree,
)
from strata.notebook.provenance import (
    compute_provenance_hash,
    compute_source_hash,
    derive_subkey,
    safe_filename_stem,
)
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
from strata.notebook.team_store import (
    TeamPull,
    TeamStore,
    publish_cell_outputs,
    pull_cell_outputs,
)
from strata.notebook.workers import (
    SIGNED_TRANSPORTS,
    get_worker_execution_error,
    is_embedded_executor_worker,
    is_http_executor_worker,
    resolve_worker_spec,
    worker_runtime_identity,
    worker_supports_notebook_execution,
    worker_transport,
)
from strata.notebook.writer import drop_blanked_secrets
from strata.tracing import current_trace_context, trace_span
from strata.transforms.build_store import get_build_store
from strata.types import EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION

if TYPE_CHECKING:
    from strata.notebook.datasets import DatasetInput
    from strata.notebook.pool import WarmProcessPool
    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)

# Fallback when no ``# @timeout`` or override applies. Generous because I/O-bound
# cells are common; a hung cell still dies here and the UI can interrupt sooner.
# Mirrors ``StrataConfig.scan_timeout_seconds``.
DEFAULT_CELL_TIMEOUT_SECONDS = 300.0


def cell_timeout_message(timeout_seconds: float) -> str:
    """A timed-out-cell error that names the remedy.

    Points at all three levers (per-cell annotation, notebook default, CLI flag),
    since a bare "timed out" does not say the limit is configurable.
    """
    return (
        f"Cell execution timed out after {timeout_seconds}s. Raise the limit with a "
        f"'# @timeout <seconds>' annotation on the cell, a 'timeout' key in "
        f"notebook.toml, or 'strata run --timeout <seconds>' for a one-off run."
    )


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


def _artifact_content_type(artifact: Any) -> str:
    """Read an artifact's stored content_type from its transform_spec params."""
    spec_json = getattr(artifact, "transform_spec", None)
    if not spec_json:
        return "pickle/object"
    try:
        spec = json.loads(spec_json)
    except (ValueError, TypeError):
        return "pickle/object"
    ct = spec.get("params", {}).get("content_type")
    return str(ct) if isinstance(ct, str) and ct else "pickle/object"


def _loop_run_token(artifact: Any) -> str:
    """The token of the loop run that stored an ``@iter=k`` artifact."""
    spec = json.loads(artifact.transform_spec or "{}")
    return str(spec.get("params", {}).get("loop_run", ""))


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


@dataclass(kw_only=True, frozen=True)
class _CellProvenance:
    """Inputs and result of the standard cell-provenance computation.

    Every cell kind (default, prompt, sql, loop) must produce the same hash for
    the same inputs, since ``compute_staleness`` recomputes it on reopen. The
    intermediates are kept so callers can reuse them.
    """

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


@dataclass(kw_only=True)
class CellExecutionResult:
    """Result from executing a cell.

    ``display_output`` is a legacy shim for the last of ``display_outputs``;
    the two are kept consistent. ``execution_method`` is ``cold``, ``warm`` or ``cached``.
    """

    cell_id: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    display_outputs: list[dict[str, Any]] = field(default_factory=list)
    display_output: dict[str, Any] | None = None
    duration_ms: float = 0
    error: str | None = None
    # A stable name for the failure a client can match on (e.g. ``fetch_pin_mismatch``);
    # ``None`` for failures that have none.
    error_code: str | None = None
    # The harness's formatted traceback, kept apart from ``error`` so the UI's one-line
    # pill stays one line.
    traceback: str | None = None
    cache_hit: bool = False
    artifact_uri: str | None = None
    execution_method: str = "cold"  # cold, warm, cached
    mutation_warnings: list[MutationWarning] = field(default_factory=list)
    # Retries the prompt-cell validate-and-retry loop used; 0 on first-try pass or for
    # non-schema cells.
    validation_retries: int = 0
    suggest_install: str | None = None  # e.g. "requests" (Python) or "arrow" (R)
    # Frontend dispatches ``uv add`` for ``"python"``, ``install.packages()`` for ``"r"``.
    # ``None`` when ``suggest_install`` is ``None``.
    suggest_install_language: str | None = None
    remote_worker: str | None = None
    remote_transport: str | None = None
    remote_build_id: str | None = None
    remote_build_state: str | None = None
    remote_error_code: str | None = None
    # Explicit: none of the fields below reliably implies a team hit (an unauthenticated
    # store publishes anonymously; a publisher may record no duration).
    from_team_cache: bool = False
    # Who computed a team-store hit. ``None`` for a local hit or a real run; otherwise a
    # team hit with no author reads as a bug, not a saving.
    team_cache_principal: str | None = None
    # Platform the team hit was built on, e.g. ``cpython-3.14-linux-x86_64``. Provenance
    # covers the lockfile, not the platform, so a hit can cross machines; this discloses
    # which. Empty unless a team hit recorded one.
    team_cache_build_env: str = ""
    # The publisher's run cost, i.e. what this hit saved. Carried, not inferred: savings
    # are priced against the last local run, which a teammate-served cell never had.
    team_cache_saved_ms: int = 0
    # The promotion the team hit came from, when one put it in the store.
    team_cache_promotion: str | None = None

    def __post_init__(self) -> None:
        # Legacy shim: accept either `display_outputs` or `display_output` and keep both
        # consistent.
        if not self.display_outputs and self.display_output is not None:
            self.display_outputs = [self.display_output]
        elif self.display_output is None and self.display_outputs:
            self.display_output = self.display_outputs[-1]

    def apply_remote_metadata(
        self,
        *,
        remote_worker: str | None = None,
        remote_transport: str | None = None,
        remote_build_id: str | None = None,
        remote_build_state: str | None = None,
        remote_error_code: str | None = None,
    ) -> CellExecutionResult:
        """Attach remote execution metadata to this result."""
        if remote_worker:
            self.remote_worker = remote_worker
        if remote_transport:
            self.remote_transport = remote_transport
        if remote_build_id:
            self.remote_build_id = remote_build_id
        if remote_build_state:
            self.remote_build_state = remote_build_state
        if remote_error_code:
            self.remote_error_code = remote_error_code
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to the REST ``/agent`` execution-result wire shape.

        The WebSocket frame uses a different, slimmer shape
        (``_execution_result_payload`` in ws.py).
        """
        payload = asdict(self)
        # Wire rename: success: bool -> status: "ready" | "error".
        payload["status"] = "ready" if payload.pop("success") else "error"
        payload["displays"] = payload.pop("display_outputs")
        payload["display"] = payload.pop("display_output")
        # Optional metadata fields are omitted from the wire when falsy.
        for opt_field in (
            "error_code",
            "validation_retries",
            "suggest_install",
            "suggest_install_language",
            "remote_worker",
            "remote_transport",
            "remote_build_id",
            "remote_build_state",
            "remote_error_code",
        ):
            if not payload[opt_field]:
                payload.pop(opt_field)
        return payload


@dataclass(kw_only=True)
class BatchCellResult:
    """Per-cell outcome inside a batched run-all execution."""

    cell_id: str
    status: str  # "ok" | "cache_hit" | "cell_error" | "persist_failed" | "not_run"
    error: str | None = None
    traceback: str | None = None
    stdout: str = ""
    stderr: str = ""
    # Enough payload for ws._execute_run_all to fan out cell_output frames without going
    # back to the harness or the store.
    outputs: dict[str, Any] = field(default_factory=dict)
    display_outputs: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False
    mutation_warnings: list[MutationWarning] = field(default_factory=list)


@dataclass(kw_only=True)
class BatchExecutionResult:
    """Outcome of one ``CellExecutor.execute_batch``: which cells still need single-cell runs."""

    cell_results: list[BatchCellResult]
    completed: bool  # True if batch_end with reason=complete
    failed_cell_id: str | None = None
    end_reason: str = "complete"  # "complete" | "cell_error" | "persist_failed" | "subprocess_died"


# Module-level so a test can make polling immediate.
_JOB_POLL_SECONDS = 1.0

# Patchable for the same reason.
_monotonic = time.monotonic


class RemoteExecutionError(RuntimeError):
    """Execution failure with structured remote metadata for notebook UX."""

    def __init__(
        self,
        message: str,
        *,
        remote_build_state: str | None = None,
        remote_error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.remote_build_state = remote_build_state
        self.remote_error_code = remote_error_code


def _team_cache_publish_policy(config: Any) -> str:
    """What the team cache offers outward: ``all``, ``promoted`` or ``off``.

    Read defensively: ``_lake_config`` may return an older config object, and a
    missing setting means the pre-existing behaviour.
    """
    return str(getattr(config, "notebook_team_cache_publish", "all") or "all")


class CellExecutor:
    """Materialize notebook cells (cache-or-build per cell).

    ``execute_cell`` ensures upstream artifacts exist (recursively on a miss),
    checks this cell's cache, and executes and stores on a miss.
    """

    def __init__(
        self,
        session: NotebookSession,
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
        # When each of those bytes was downloaded, ``{url: unix time}``, for the record.
        self._fetch_times: dict[str, dict[str, float]] = {}
        # Same for ``@dataset``: ``{strata://name/<reference>: <id>@v=<n>}``.
        self._dataset_refs: dict[str, dict[str, str]] = {}
        # Also store the variables of a cell nothing downstream reads. Off by default: only
        # consumed variables are kept, and a leaf's result can be large.
        self.store_leaf_outputs = False
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
    ) -> CellExecutionResult:
        """Materialise a cell: ensure inputs, check cache, execute, store."""
        return await self._execute_cell(
            cell_id,
            source,
            timeout_seconds,
            materialize_upstreams=True,
            use_cache=True,
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

    async def run_cell_tests(self, cell_id: str, test_source: str) -> CellTestResult:
        """Run a Python cell's unit tests and persist the result.

        Tests target the functions/classes the cell defines, which are never
        stored as artifacts, so this re-executes the source with its upstream
        inputs injected and runs ``pytest`` in the notebook venv. It does not go
        through the harness and never touches the artifact cache or cell status.
        """
        # cell_test_runner is stdlib-only, kept beside its single caller.
        from strata.notebook.cell_test_runner import (
            PytestUnavailableError,
            run_cell_tests_in_dir,
        )
        from strata.notebook.runtime_state import persist_cell_test_result

        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise FileNotFoundError(f"Cell {cell_id} not found")
        venv_python = self.session.venv_python
        if venv_python is None:
            raise RuntimeError(_no_interpreter_message(self.session))

        source = cell.source
        await self._materialize_upstreams(cell_id)

        # The same fetch, mount, dataset, table and env inputs a run of the cell gets.
        annotations = parse_annotations(source)
        mount_specs = self._resolve_cell_mount_specs(cell_id, source)
        prov = await self._compute_cell_provenance(
            cell_id, source, annotations=annotations, mount_specs=mount_specs
        )
        problem = prov.fetch_error or prov.dataset_error
        tables: dict[str, dict[str, Any]] = {}
        if problem is None:
            try:
                tables = self._manifest_tables(annotations.tables, prov.table_snapshots)
            except RuntimeError as exc:
                problem = str(exc)
        mount_specs = [
            *mount_specs,
            *(
                MountSpec(name=name, uri=path.resolve().as_uri(), mode=MountMode.READ_ONLY)
                for name, path in prov.fetched.items()
            ),
        ]
        resolved_mounts = await self._prepare_mounts(mount_specs)
        runtime_env = self._resolve_effective_runtime_env(cell_id, annotations.env)

        # Tests run the cell's source, so they are refused or dropped to the harness user
        # like any harness.
        refused: dict[str, Any] | None = None
        harness_user = None
        try:
            harness_user = resolve_harness_user()
        except LocalExecutionRefused as exc:
            problem = str(exc)
        if problem is not None:
            refused = {
                "passed": 0,
                "failed": 0,
                "errored": 1,
                "skipped": 0,
                "tests": [
                    {"name": "<refused>", "nodeid": "", "outcome": "error", "message": problem}
                ],
            }

        pytest_unavailable = False
        auto_installed: list[str] = []
        with tempfile.TemporaryDirectory(prefix="strata_celltest_") as tmp:
            blob_dir = Path(tmp) / "inputs"
            blob_dir.mkdir()
            input_specs = self._load_input_blobs(cell_id, blob_dir)
            self._add_dataset_inputs(input_specs, prov.datasets, blob_dir)

            # The run directory is created inside this server-private one; a harness user must
            # reach it.
            hand_over(Path(tmp), harness_user)

            def _run(rundir_name: str) -> dict[str, Any]:
                return run_cell_tests_in_dir(
                    rundir=Path(tmp) / rundir_name,
                    venv_python=venv_python,
                    cell_source=source,
                    test_source=test_source,
                    inputs=input_specs,
                    input_dir=blob_dir,
                    mounts={name: str(rm.local_path) for name, rm in resolved_mounts.items()},
                    tables=tables,
                    env=identity_env(self._harness_env(runtime_env), harness_user),
                    run_as=harness_user,
                )

            empty_raw: dict[str, Any] = {
                "passed": 0,
                "failed": 0,
                "errored": 0,
                "skipped": 0,
                "tests": [],
            }
            raw: dict[str, Any]
            try:
                raw = refused if refused is not None else await asyncio.to_thread(_run, "run")
            except PytestUnavailableError:
                # Auto-provision pytest into the dev group and retry once. Dev tools are excluded
                # from the env hash, so cell caches survive. On failure, fall back to the actionable
                # pytest_unavailable flag.
                from strata.notebook.dependencies import ensure_dev_tool

                install = await asyncio.to_thread(ensure_dev_tool, self.session.path, "pytest")
                if install.success:
                    auto_installed = ["pytest"]
                    try:
                        raw = await asyncio.to_thread(_run, "run")
                    except PytestUnavailableError:
                        pytest_unavailable = True
                        raw = empty_raw
                else:
                    logger.warning(
                        "Auto-install of pytest failed for notebook %s: %s",
                        self.session.path,
                        install.error,
                    )
                    pytest_unavailable = True
                    raw = empty_raw

        # After the run: a pytest auto-install can change the lockfile, which is an input.
        source_hash, test_source_hash, input_fingerprint = self.session.cell_test_fingerprint(
            cell_id, source, test_source
        )
        result = CellTestResult(
            passed=raw["passed"],
            failed=raw["failed"],
            errored=raw["errored"],
            skipped=raw.get("skipped", 0),
            tests=[CellTestCase(**t) for t in raw["tests"]],
            cell_source_hash=source_hash,
            test_source_hash=test_source_hash,
            input_fingerprint=input_fingerprint,
            ran_at=int(time.time() * 1000),
            pytest_unavailable=pytest_unavailable,
            auto_installed=auto_installed,
        )

        persist_cell_test_result(self.session.path, cell_id, result.model_dump())
        cell.test_result = result
        cell.test_source = test_source
        return result

    async def execute_batch(
        self,
        cell_specs: list[dict[str, Any]],
        *,
        use_cache: bool = True,
        batch_timeout_seconds: float = 600.0,
        cell_timeout_seconds: float = DEFAULT_CELL_TIMEOUT_SECONDS,
        on_cell_event: Callable[[BatchCellResult], Awaitable[None]] | None = None,
    ) -> BatchExecutionResult:
        """Execute a sequence of cells in one harness subprocess.

        ``cell_specs`` are in notebook order, each with ``cell_id``, ``source``,
        ``env`` and a resolved ``mount_manifest``; the caller partitions the
        notebook. ``use_cache=False`` is rerun-all. ``on_cell_event`` is awaited
        as each ``BatchCellResult`` is recorded, so frames stream in real time.
        """
        return await self._run_batch(
            cell_specs,
            use_cache=use_cache,
            batch_timeout_seconds=batch_timeout_seconds,
            cell_timeout_seconds=cell_timeout_seconds,
            on_cell_event=on_cell_event,
        )

    @contextmanager
    def one_run(self) -> Iterator[None]:
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
                )
        scope = self._run_scope
        earlier = scope.get(cell_id)
        if earlier is not None and (use_cache or not earlier.cache_hit):
            return earlier
        result = await self._dispatch_cell(
            cell_id,
            source,
            timeout_seconds,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )
        if result.success:
            scope[cell_id] = result
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
                return await self._execute_loop_cell(
                    cell_id,
                    source,
                    annotations.loop,
                    timeout_seconds,
                    start_time,
                    materialize_upstreams=materialize_upstreams,
                    use_cache=use_cache,
                )
            finally:
                self._materializing.discard(cell_id)
        effective_worker = self._resolve_effective_worker(cell_id, annotations.worker)
        worker_spec = resolve_worker_spec(
            self.session.notebook_state,
            effective_worker,
        )
        if not worker_supports_notebook_execution(worker_spec):
            policy_error = get_worker_execution_error(
                self.session.notebook_state,
                effective_worker,
            )
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                error=policy_error
                or (f"Execution failed: worker '{effective_worker}' is not implemented yet"),
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

    def _resolve_effective_worker(
        self,
        cell_id: str,
        annotation_worker: str | None,
    ) -> str:
        """Resolve the effective worker with annotation precedence."""
        if annotation_worker:
            return annotation_worker

        cell = self.session.notebook_state.get_cell(cell_id)
        if cell and cell.worker:
            return cell.worker

        notebook_worker = self.session.notebook_state.worker
        if notebook_worker:
            return notebook_worker

        return "local"

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
        precomputed arguments let callers skip resolving mounts again.
        """
        if annotations is None:
            annotations = parse_annotations(source)
        if mount_specs is None:
            mount_specs = self._resolve_cell_mount_specs(cell_id, source)
        if mount_fingerprints is None or has_rw_mount is None:
            mount_fingerprints, has_rw_mount = await self._fingerprint_mounts(mount_specs)

        source_hash = compute_source_hash(source)
        runtime_env = self._resolve_effective_runtime_env(cell_id, annotations.env)
        effective_worker = self._resolve_effective_worker(cell_id, annotations.worker)
        runtime_identity = worker_runtime_identity(self.session.notebook_state, effective_worker)
        cell_state = self.session.notebook_state.get_cell(cell_id)
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
        input_hashes = self._collect_input_hashes(cell_id)
        # A DuckDB cell's catalog tables are inputs like its @table ones. Imported only for
        # SQL cells: the SQL package needs the [sql] extra.
        tables = list(annotations.tables)
        config = None
        if annotations.sql is not None:
            from strata.notebook.sql.lake import lake_tables, with_notebook_catalogs

            tables += lake_tables(self.session.notebook_state, source)
            config = with_notebook_catalogs(self._lake_config(), self.session.notebook_state)
        table_fingerprints, table_snapshots = await self._fingerprint_tables(tables, config)
        (
            fetch_fingerprints,
            fetched,
            fetch_refs,
            fetch_times,
            fetch_error,
            fetch_error_code,
        ) = await self._resolve_fetches(annotations.fetches)
        self._fetch_refs[cell_id] = fetch_refs
        self._fetch_times[cell_id] = fetch_times
        dataset_fingerprints, datasets, dataset_error = await self._resolve_datasets(
            annotations.datasets
        )
        self._dataset_refs[cell_id] = {
            dataset.resolved.lineage_uri: dataset.local_ref for dataset in datasets.values()
        }
        provenance_hash = compute_provenance_hash(
            input_hashes
            + mount_fingerprints
            + table_fingerprints
            + fetch_fingerprints
            + dataset_fingerprints,
            source_hash,
            env_hash,
        )

        return _CellProvenance(
            annotations=annotations,
            source_hash=source_hash,
            runtime_env=runtime_env,
            effective_worker=effective_worker,
            runtime_identity=runtime_identity,
            env_hash=env_hash,
            input_hashes=input_hashes,
            mount_specs=mount_specs,
            mount_fingerprints=mount_fingerprints,
            has_rw_mount=has_rw_mount,
            table_fingerprints=table_fingerprints,
            table_snapshots=table_snapshots,
            provenance_hash=provenance_hash,
            fetch_fingerprints=fetch_fingerprints,
            fetched=fetched,
            fetch_error=fetch_error,
            fetch_error_code=fetch_error_code,
            dataset_fingerprints=dataset_fingerprints,
            datasets=datasets,
            dataset_error=dataset_error,
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

        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is not None:
            cell.cache_hit = False
            language_executor = get_language_executor(cell.language)
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

    async def _execute_python_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float,
        start_time: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
        fanout_group: str | None = None,
        fanout_variant: str | None = None,
    ) -> CellExecutionResult:
        """Python cell pipeline: provenance, upstreams, cache check, harness, persist.

        A top-level call on a ``# @per_variant`` cell redirects to
        :meth:`_execute_fanout_cell`, which calls back once per variant with
        ``fanout_variant`` set; provenance, cache, inputs and artifacts are then
        scoped to that variant.
        """
        remote_metadata: dict[str, str] = {}
        try:
            # Re-fetch the cell: the dispatch wrapper already reset ``cache_hit``, but the rest of
            # the pipeline needs ``cell`` in scope.
            cell = self.session.notebook_state.get_cell(cell_id)

            # A @per_variant cell fans out: the orchestrator re-enters this pipeline once per
            # variant with fanout_variant set.
            if fanout_variant is None:
                fanout = self._fanout_info(cell_id)
                if fanout is not None:
                    return await self._execute_fanout_cell(
                        cell_id,
                        source,
                        timeout_seconds,
                        start_time,
                        materialize_upstreams=materialize_upstreams,
                        use_cache=use_cache,
                        group=fanout[0],
                        variant_names=fanout[1],
                    )

            # ① Materialize every upstream whose artifact is missing (recursive; each upstream
            # miss executes its own upstreams).
            if materialize_upstreams:
                await self._materialize_upstreams(cell_id)

            # ② Provenance. Every cell-kind path uses the same helper so the hash agrees with
            # ``compute_staleness`` on re-open.
            prov = await self._compute_cell_provenance(cell_id, source)
            source_hash = prov.source_hash
            runtime_env = prov.runtime_env
            effective_worker = prov.effective_worker
            env_hash = prov.env_hash
            input_hashes = prov.input_hashes
            mount_specs = prov.mount_specs
            mount_fingerprints = prov.mount_fingerprints
            provenance_hash = prov.provenance_hash

            # Scope provenance to this variant so each instance caches independently.
            if fanout_variant is not None:
                provenance_hash = derive_subkey(provenance_hash, f"variant={fanout_variant}")

            # RW mounts have side effects: not cacheable. A rerun bypasses the cache but still
            # records a cacheable leaf's console, or the next plain run replays an older one.
            cacheable = not prov.has_rw_mount and not prov.annotations.nocache
            if not cacheable:
                use_cache = False

            worker_spec = resolve_worker_spec(
                self.session.notebook_state,
                effective_worker,
            )
            remote_metadata = self._remote_execution_metadata(worker_spec)

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
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=prov.fetch_error,
                    error_code=prov.fetch_error_code,
                    execution_method="error",
                )
            # Likewise a dataset the registry could not resolve or hand over.
            if prov.dataset_error is not None:
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=prov.dataset_error,
                    execution_method="error",
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
                manifest_tables = self._manifest_tables(
                    prov.annotations.tables, prov.table_snapshots
                )
            except RuntimeError as e:
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=str(e),
                    execution_method="error",
                )

            # ③ Cache check for THIS cell.
            artifact_mgr = self.session.get_artifact_manager()
            consumed_vars = (
                self.session.dag.consumed_variables.get(cell_id, set())
                if self.session.dag
                else set()
            )

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
                self.session._resolve_cached_display_outputs(
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
                self.session._resolve_cached_console(cell_id, provenance_hash)
                if (cell is not None and use_cache and not consumed_vars)
                else None
            )
            team_pull: TeamPull | None = None
            if use_cache:
                if consumed_vars:
                    first_var = sorted(consumed_vars)[0]
                    var_prov = derive_subkey(provenance_hash, first_var)
                    cached_artifact = artifact_mgr.find_cached(var_prov)
                else:
                    cached_artifact = artifact_mgr.find_cached(provenance_hash)

                # Team cache tier, only after a local miss so an ordinary hit pays nothing. A pull
                # writes each variable under its canonical local id, so the re-probe and validation
                # below run against real local artifacts.
                if cached_artifact is None and consumed_vars:
                    team_pull = await self._pull_from_team_store(
                        cell_id=cell_id,
                        provenance_hash=provenance_hash,
                        consumed_vars=consumed_vars,
                        source_hash=source_hash,
                        source=source,
                        env_hash=env_hash,
                        input_versions=self._input_refs(cell_id, fanout_variant),
                        variant=fanout_variant,
                    )
                    if team_pull is not None:
                        cached_artifact = artifact_mgr.find_cached(
                            derive_subkey(provenance_hash, sorted(consumed_vars)[0])
                        )

            # Validate the hit: find_by_provenance can return artifacts from other notebooks in
            # the same DB, so each consumed variable's LOCAL canonical artifact must exist AND
            # carry the expected provenance hash.
            if use_cache and cached_artifact is not None and consumed_vars:
                # Two passes. A mismatched variable may still have the right result under an older
                # version (reverting an edit P1 -> P2 -> P1); promoting re-points "latest" at it
                # without running the cell. Decide before writing: if one variable can't be promoted
                # the cell runs anyway, and an applied promotion would disagree with the rest.
                to_promote: list[tuple[str, int]] = []
                invalid: tuple[str, ArtifactVersion | None, str] | None = None
                # sorted() keeps the reported variable and promotion order stable across runs.
                for var_name in sorted(consumed_vars):
                    canonical_id = artifact_mgr.cell_artifact_id(
                        cell_id, var_name, variant=fanout_variant
                    )
                    var_prov = derive_subkey(provenance_hash, var_name)
                    canonical_art = artifact_mgr.artifact_store.get_latest_version(
                        canonical_id,
                    )
                    if canonical_art is not None and canonical_art.provenance_hash == var_prov:
                        continue
                    older = artifact_mgr.artifact_store.find_version_by_provenance(
                        canonical_id, var_prov
                    )
                    if older is None or not artifact_mgr.artifact_store.blob_exists(
                        canonical_id, older.version
                    ):
                        invalid = (canonical_id, canonical_art, var_prov)
                        break
                    to_promote.append((canonical_id, older.version))

                # Display outputs share the provenance and resolved to [] above, since their latest
                # carries the edited hash; without promoting them a reverted cell hits with its plot
                # gone. Restore the count the reverted run recorded (the current list holds the
                # later run's, or none after a failure); older displays use the current list.
                if invalid is None:
                    display_count = self._recorded_display_count(
                        cell_id, provenance_hash, fanout_variant
                    )
                    if display_count is None:
                        display_count = len(current_display_outputs)
                    for index in range(display_count):
                        display_id = artifact_mgr.cell_artifact_id(
                            cell_id, f"__display__{index}", variant=fanout_variant
                        )
                        display_prov = derive_subkey(provenance_hash, f"__display__{index}")
                        display_art = artifact_mgr.artifact_store.get_latest_version(display_id)
                        if display_art is not None and display_art.provenance_hash == display_prov:
                            continue
                        older_display = artifact_mgr.artifact_store.find_version_by_provenance(
                            display_id, display_prov
                        )
                        if older_display is None or not artifact_mgr.artifact_store.blob_exists(
                            display_id, older_display.version
                        ):
                            invalid = (display_id, display_art, display_prov)
                            break
                        to_promote.append((display_id, older_display.version))

                if invalid is not None:
                    canonical_id, canonical_art, var_prov = invalid
                    logger.info(
                        "Cache hit for cell %s invalidated: "
                        "canonical artifact %s %s "
                        "(provenance hit was %s@v=%d, "
                        "expected provenance %s).",
                        cell_id,
                        canonical_id,
                        "not found"
                        if canonical_art is None
                        else f"has stale provenance {canonical_art.provenance_hash[:12]}",
                        cached_artifact.id,
                        cached_artifact.version,
                        var_prov[:12],
                    )
                    cached_artifact = None
                else:
                    for canonical_id, source_version in to_promote:
                        promoted = artifact_mgr.artifact_store.promote_version(
                            canonical_id, source_version
                        )
                        if promoted is None:
                            # The blob vanished since the check (likely a GC pass). Run the cell.
                            logger.info(
                                "Cell %s could not promote %s@v=%d; re-executing.",
                                cell_id,
                                canonical_id,
                                source_version,
                            )
                            cached_artifact = None
                            break
                        logger.info(
                            "Cell %s reverted to a known provenance: promoted "
                            "%s@v=%d to @v=%d instead of re-executing.",
                            cell_id,
                            canonical_id,
                            source_version,
                            promoted.version,
                        )
                    if cached_artifact is not None and to_promote and cell is not None:
                        # The display resolver saw pre-promotion state; re-resolve.
                        cached_display_outputs = self.session._resolve_cached_display_outputs(
                            cell_id,
                            provenance_hash,
                            current_display_outputs,
                        )
                        # Likewise cached_artifact names the superseded version, which would
                        # disagree with cell.artifact_uris re-read from latest below.
                        first_var = sorted(consumed_vars)[0]
                        refreshed = artifact_mgr.artifact_store.get_latest_version(
                            artifact_mgr.cell_artifact_id(
                                cell_id, first_var, variant=fanout_variant
                            )
                        )
                        if refreshed is not None:
                            cached_artifact = refreshed

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
                if remote_metadata.get("remote_transport") == "signed":
                    remote_metadata.setdefault("remote_build_state", "ready")
                duration_ms = (time.time() - start_time) * 1000
                if cell:
                    cell.cache_hit = True
                    cell.display_outputs = list(cached_display_outputs)
                    cell.display_output = (
                        cached_display_outputs[-1] if cached_display_outputs else None
                    )
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
                    stdout=cached_console[0] if cached_console else "",
                    stderr=cached_console[1] if cached_console else "",
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
                            cached_display_outputs[-1].artifact_uri
                            if cached_display_outputs
                            else None
                        )
                    ),
                    execution_method="cached",
                    team_cache_principal=team_pull.principal if team_pull else None,
                    team_cache_build_env=team_pull.build_env if team_pull else "",
                    team_cache_saved_ms=team_pull.saved_ms if team_pull else 0,
                    team_cache_promotion=team_pull.promotion if team_pull else None,
                    from_team_cache=team_pull is not None,
                ).apply_remote_metadata(**remote_metadata)
                self.session.record_successful_execution_provenance(
                    cell_id,
                    provenance_hash,
                    source_hash,
                    env_hash,
                )
                self.session.apply_execution_result_metadata(cell_id, cached_result)
                return cached_result

            # ④ Cache miss: execute the cell.
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                remote_build_id = (
                    f"nbbuild-{uuid.uuid4().hex[:12]}"
                    if worker_spec is not None and worker_transport(worker_spec) == "signed"
                    else None
                )
                remote_metadata = self._remote_execution_metadata(
                    worker_spec,
                    remote_build_id=remote_build_id,
                )

                # Force may skip upstream materialization, so missing inputs are allowed to surface
                # at execution time.
                input_specs = self._load_input_blobs(
                    cell_id,
                    output_dir,
                    fanout_group=fanout_group,
                    fanout_variant=fanout_variant,
                )
                if fetches_as_inputs:
                    _add_fetch_inputs(input_specs, prov.fetched, output_dir)
                self._add_dataset_inputs(input_specs, prov.datasets, output_dir)

                venv_path = self.session.venv_python

                (
                    result,
                    result_output_dir,
                    execution_method,
                    resolved_mounts,
                ) = await self._dispatch_execution(
                    worker_spec,
                    source,
                    input_specs,
                    mount_specs,
                    output_dir,
                    venv_path,
                    runtime_env,
                    timeout_seconds,
                    remote_build_id=remote_build_id,
                    mutation_defines=list(getattr(cell, "mutation_defines", []) or []),
                    tables=manifest_tables,
                    cell_id=cell_id,
                    cell_provenance_hash=provenance_hash,
                )
                if remote_build_id and remote_metadata.get("remote_transport") == "signed":
                    remote_metadata["remote_build_state"] = "ready"

                duration_ms = (time.time() - start_time) * 1000
                exec_result = self._parse_result(
                    cell_id,
                    result,
                    duration_ms,
                    execution_method,
                ).apply_remote_metadata(**remote_metadata)

                # ⑤ Store artifacts for consumed variables.
                if exec_result.success:
                    module_export_error = self._write_module_export_outputs(
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
                    self.session.record_successful_execution_provenance(
                        cell_id,
                        provenance_hash,
                        source_hash,
                        env_hash,
                    )
                    stored_ok = self._store_outputs(
                        cell_id,
                        result_output_dir,
                        provenance_hash,
                        input_hashes,
                        source_hash=source_hash,
                        source=source,
                        env_hash=env_hash,
                        variant=fanout_variant,
                        # Reported by whatever ran the cell (venv or remote worker); this process
                        # may not be on that machine.
                        build_env=str(result.get("build_env") or ""),
                        hardware=result.get("hardware") or {},
                        # What a teammate is told they saved; they never ran the cell, so it travels
                        # with the bytes.
                        build_duration_ms=exec_result.duration_ms,
                    )
                    if not stored_ok:
                        logger.error(
                            "Cell %s executed OK but artifact storage failed.",
                            cell_id,
                        )
                        exec_result = CellExecutionResult(
                            cell_id=cell_id,
                            success=False,
                            stdout=exec_result.stdout,
                            stderr=exec_result.stderr,
                            outputs=exec_result.outputs,
                            duration_ms=exec_result.duration_ms,
                            error=(
                                "Cell executed successfully but failed to "
                                "store output artifacts. Check server logs."
                            ),
                            execution_method=exec_result.execution_method,
                        ).apply_remote_metadata(**remote_metadata)

                    if stored_ok:
                        # Only once artifacts are on disk: the publish reads them back, so a pull
                        # reproduces exactly what a local run stored.
                        await self._push_to_team_store(cell_id=cell_id, variant=fanout_variant)

                    if exec_result.success:
                        exec_result.display_outputs = self._store_display_outputs(
                            cell_id,
                            result_output_dir,
                            provenance_hash,
                            input_hashes,
                            exec_result.display_outputs,
                            source_hash=source_hash,
                            source=source,
                            env_hash=env_hash,
                        )
                        exec_result.display_output = (
                            exec_result.display_outputs[-1] if exec_result.display_outputs else None
                        )
                        # Cache a leaf cell's console by provenance so a re-run replays it.
                        if cacheable and not consumed_vars:
                            self._store_console_outputs(
                                cell_id,
                                provenance_hash,
                                exec_result.stdout,
                                exec_result.stderr,
                                input_hashes,
                                source_hash=source_hash,
                                source=source,
                                env_hash=env_hash,
                            )

                    # ⑥ Sync back read-write mounts.
                    if exec_result.success and resolved_mounts:
                        try:
                            await self._mount_resolver.sync_back(resolved_mounts)
                        except Exception as exc:
                            logger.exception(
                                "Failed to sync-back RW mounts for cell %s",
                                cell_id,
                            )
                            exec_result = CellExecutionResult(
                                cell_id=cell_id,
                                success=False,
                                stdout=exec_result.stdout,
                                stderr=exec_result.stderr,
                                outputs=exec_result.outputs,
                                duration_ms=exec_result.duration_ms,
                                error=(
                                    "Cell executed successfully but failed to sync "
                                    f"read-write mounts: {exc}"
                                ),
                                execution_method=exec_result.execution_method,
                                mutation_warnings=exec_result.mutation_warnings,
                            ).apply_remote_metadata(**remote_metadata)

                self.session.persist_display_outputs(
                    cell_id,
                    exec_result.display_outputs if exec_result.success else None,
                )
                self.session.apply_execution_result_metadata(cell_id, exec_result)
                return exec_result

        except RemoteExecutionError as e:
            duration_ms = (time.time() - start_time) * 1000
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
            self.session.persist_display_output(cell_id, None)
            self.session.apply_execution_result_metadata(cell_id, error_result)
            return error_result
        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            timeout_result = CellExecutionResult(
                cell_id=cell_id,
                success=False,
                duration_ms=duration_ms,
                error=cell_timeout_message(timeout_seconds),
            ).apply_remote_metadata(**remote_metadata)
            self.session.persist_display_output(cell_id, None)
            self.session.apply_execution_result_metadata(cell_id, timeout_result)
            return timeout_result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_result = CellExecutionResult(
                cell_id=cell_id,
                success=False,
                duration_ms=duration_ms,
                error=f"Execution failed: {e}",
            ).apply_remote_metadata(**remote_metadata)
            self.session.persist_display_output(cell_id, None)
            self.session.apply_execution_result_metadata(cell_id, error_result)
            return error_result

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

    def _fanout_info(self, cell_id: str) -> tuple[str, tuple[str, ...]] | None:
        """``(group, variant_names)`` if ``cell_id`` is a ``@per_variant`` fan-out cell, else None.

        That is, its outputs resolve to a ``SweepProducer`` with ``fanout_cell == cell_id``.
        """
        dag = self.session.dag
        if dag is None:
            return None
        for producer in dag.variable_producer.values():
            if isinstance(producer, SweepProducer) and producer.fanout_cell == cell_id:
                return producer.group, tuple(name for name, _ in producer.variants)
        return None

    async def _execute_fanout_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float,
        start_time: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
        group: str,
        variant_names: tuple[str, ...],
    ) -> CellExecutionResult:
        """Run a ``# @per_variant`` cell once per variant of its fan-out group.

        Each instance binds that variant's scalar and stores ``@variant={name}``
        artifacts, which a downstream consumer collapses back to a
        ``{variant: value}`` dict. Succeeds iff every variant succeeds.
        Upstreams are materialised once, not per variant.
        """
        if materialize_upstreams:
            await self._materialize_upstreams(cell_id)

        results: list[tuple[str, CellExecutionResult]] = []
        total = len(variant_names)
        for index, name in enumerate(variant_names):
            variant_start = time.time()
            res = await self._execute_python_cell(
                cell_id,
                source,
                timeout_seconds,
                start_time,
                materialize_upstreams=False,
                use_cache=use_cache,
                fanout_group=group,
                fanout_variant=name,
            )
            results.append((name, res))
            if self.on_variant_complete is not None:
                try:
                    await self.on_variant_complete(
                        {
                            "cell_id": cell_id,
                            "variant": name,
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
                        name,
                    )

        duration_ms = (time.time() - start_time) * 1000
        failures = [(name, r) for name, r in results if not r.success]
        stdout = "\n".join(f"[variant={name}]\n{r.stdout}" for name, r in results if r.stdout)
        stderr = "\n".join(f"[variant={name}]\n{r.stderr}" for name, r in results if r.stderr)
        if failures:
            first_name, first = failures[0]
            aggregate = CellExecutionResult(
                cell_id=cell_id,
                success=False,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                error=f"Fan-out variant '{first_name}' failed: {first.error}",
                execution_method="fanout",
            )
        else:
            aggregate = CellExecutionResult(
                cell_id=cell_id,
                success=True,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                execution_method="fanout",
            )
            # Record the BASE provenance: compute_staleness recomputes the base hash, so a
            # variant-scoped value would leave the fan-out cell perpetually stale on re-open.
            try:
                prov = await self._compute_cell_provenance(cell_id, source)
                self.session.record_successful_execution_provenance(
                    cell_id,
                    prov.provenance_hash,
                    compute_source_hash(source),
                    prov.env_hash,
                )
            except Exception:
                logger.exception("Failed to record fan-out base provenance for cell %s", cell_id)
        self.session.apply_execution_result_metadata(cell_id, aggregate)
        return aggregate

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
        """R cell pipeline: provenance, upstreams, cache check, ``Rscript`` harness, persist.

        Runs locally (warm pool, then cold ``Rscript``) or on an HTTP executor
        worker. Cache, mounts and storage are shared with Python: they key off
        provenance hashes and file extensions, which the R harness produces identically.
        """
        remote_metadata: dict[str, str] = {}
        try:
            cell = self.session.notebook_state.get_cell(cell_id)

            if materialize_upstreams:
                await self._materialize_upstreams(cell_id)

            prov = await self._compute_cell_provenance(cell_id, source)
            source_hash = prov.source_hash
            runtime_env = prov.runtime_env
            env_hash = prov.env_hash
            input_hashes = prov.input_hashes
            provenance_hash = prov.provenance_hash
            if prov.annotations.datasets:
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error="@dataset is not supported on R cells; read the dataset in a "
                    "Python cell upstream",
                    execution_method="error",
                )
            # As in Python: an unchecked fetch fails the run, and each fetched file arrives as a
            # read-only mount that harness.R binds to its name as a path string.
            if prov.fetch_error is not None:
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=prov.fetch_error,
                    error_code=prov.fetch_error_code,
                    execution_method="error",
                )
            worker_spec = resolve_worker_spec(self.session.notebook_state, prov.effective_worker)
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
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error=f"R cells run locally or on an executor worker; worker "
                    f"'{worker_spec.name}' is neither",
                    execution_method="error",
                )
            if remote is not None and prov.fetched:
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=False,
                    error="@fetch on an R cell is read on this machine; run the cell "
                    "locally or fetch in a Python cell upstream",
                    execution_method="error",
                )
            remote_metadata = self._remote_execution_metadata(worker_spec)
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

            artifact_mgr = self.session.get_artifact_manager()
            consumed_vars = (
                self.session.dag.consumed_variables.get(cell_id, set())
                if self.session.dag
                else set()
            )

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
                self.session._resolve_cached_display_outputs(
                    cell_id,
                    provenance_hash,
                    current_display_outputs,
                )
                if (cell is not None and use_cache)
                else []
            )
            team_pull: TeamPull | None = None
            if use_cache:
                if consumed_vars:
                    first_var = sorted(consumed_vars)[0]
                    var_prov = derive_subkey(provenance_hash, first_var)
                    cached_artifact = artifact_mgr.find_cached(var_prov)
                else:
                    cached_artifact = artifact_mgr.find_cached(provenance_hash)

                # Same team-cache tier as Python: the store keys off provenance and blobs, both
                # language-agnostic, and the two paths must not drift.
                if cached_artifact is None and consumed_vars:
                    team_pull = await self._pull_from_team_store(
                        cell_id=cell_id,
                        provenance_hash=provenance_hash,
                        consumed_vars=consumed_vars,
                        source_hash=source_hash,
                        source=source,
                        env_hash=env_hash,
                        input_versions=self._input_refs(cell_id),
                    )
                    if team_pull is not None:
                        cached_artifact = artifact_mgr.find_cached(
                            derive_subkey(provenance_hash, sorted(consumed_vars)[0])
                        )

            notebook_id = self.session.notebook_state.id
            if use_cache and cached_artifact is not None and consumed_vars:
                for var_name in consumed_vars:
                    canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{var_name}"
                    var_prov = derive_subkey(provenance_hash, var_name)
                    canonical_art = artifact_mgr.artifact_store.get_latest_version(
                        canonical_id,
                    )
                    if canonical_art is None or canonical_art.provenance_hash != var_prov:
                        cached_artifact = None
                        break

            if cached_artifact is not None or (not consumed_vars and cached_display_outputs):
                duration_ms = (time.time() - start_time) * 1000
                if cell:
                    cell.cache_hit = True
                    cell.display_outputs = list(cached_display_outputs)
                    cell.display_output = (
                        cached_display_outputs[-1] if cached_display_outputs else None
                    )
                    for var_name in consumed_vars:
                        canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{var_name}"
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
                            cached_display_outputs[-1].artifact_uri
                            if cached_display_outputs
                            else None
                        )
                    ),
                    execution_method="cached",
                    team_cache_principal=team_pull.principal if team_pull else None,
                    team_cache_build_env=team_pull.build_env if team_pull else "",
                    team_cache_saved_ms=team_pull.saved_ms if team_pull else 0,
                    team_cache_promotion=team_pull.promotion if team_pull else None,
                    from_team_cache=team_pull is not None,
                ).apply_remote_metadata(**remote_metadata)
                self.session.record_successful_execution_provenance(
                    cell_id,
                    provenance_hash,
                    source_hash,
                    env_hash,
                )
                self.session.apply_execution_result_metadata(cell_id, cached_result)
                return cached_result

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                input_specs = self._load_input_blobs(cell_id, output_dir)
                result_output_dir = output_dir
                if remote is not None:
                    remote_build_id = (
                        f"nbbuild-{uuid.uuid4().hex[:12]}"
                        if worker_transport(remote) == "signed"
                        else None
                    )
                    remote_metadata = self._remote_execution_metadata(
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
                        ) = await self._dispatch_http_executor(
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
                    resolved_mounts = await self._prepare_mounts(mount_specs)
                    manifest_path = self._write_manifest(
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
                    r_pool = getattr(self.session, "r_warm_pool", None)
                    if r_pool is not None:
                        from strata.notebook.pool import PooledCellExecutor

                        pool_result = await PooledCellExecutor.execute_with_pool(
                            r_pool,
                            manifest_path,
                            self.session.path,
                            timeout_seconds,
                        )
                        if pool_result is not None:
                            result = pool_result
                            execution_method = "warm"
                    if result is None:
                        result = await self._run_r_harness(manifest_path, timeout_seconds)

                duration_ms = (time.time() - start_time) * 1000
                exec_result = self._parse_result(
                    cell_id, result, duration_ms, execution_method
                ).apply_remote_metadata(**remote_metadata)

                if exec_result.success:
                    self.session.record_successful_execution_provenance(
                        cell_id,
                        provenance_hash,
                        source_hash,
                        env_hash,
                    )
                    stored_ok = self._store_outputs(
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
                    )
                    if not stored_ok:
                        logger.error(
                            "R cell %s executed OK but artifact storage failed.",
                            cell_id,
                        )
                        exec_result = CellExecutionResult(
                            cell_id=cell_id,
                            success=False,
                            stdout=exec_result.stdout,
                            stderr=exec_result.stderr,
                            outputs=exec_result.outputs,
                            duration_ms=exec_result.duration_ms,
                            error=(
                                "Cell executed successfully but failed to "
                                "store output artifacts. Check server logs."
                            ),
                            execution_method=exec_result.execution_method,
                        )

                    if stored_ok:
                        await self._push_to_team_store(cell_id=cell_id)

                    if exec_result.success:
                        exec_result.display_outputs = self._store_display_outputs(
                            cell_id,
                            result_output_dir,
                            provenance_hash,
                            input_hashes,
                            exec_result.display_outputs,
                            source_hash=source_hash,
                            source=source,
                            env_hash=env_hash,
                        )
                        exec_result.display_output = (
                            exec_result.display_outputs[-1] if exec_result.display_outputs else None
                        )

                    if exec_result.success and resolved_mounts:
                        try:
                            await self._mount_resolver.sync_back(resolved_mounts)
                        except Exception as exc:
                            logger.exception(
                                "Failed to sync-back RW mounts for R cell %s",
                                cell_id,
                            )
                            exec_result = CellExecutionResult(
                                cell_id=cell_id,
                                success=False,
                                stdout=exec_result.stdout,
                                stderr=exec_result.stderr,
                                outputs=exec_result.outputs,
                                duration_ms=exec_result.duration_ms,
                                error=(
                                    "Cell executed successfully but failed to sync "
                                    f"read-write mounts: {exc}"
                                ),
                                execution_method=exec_result.execution_method,
                                mutation_warnings=exec_result.mutation_warnings,
                            )

                self.session.persist_display_outputs(
                    cell_id,
                    exec_result.display_outputs if exec_result.success else None,
                )
                self.session.apply_execution_result_metadata(cell_id, exec_result)
                return exec_result

        except RemoteExecutionError as e:
            duration_ms = (time.time() - start_time) * 1000
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
            self.session.persist_display_output(cell_id, None)
            self.session.apply_execution_result_metadata(cell_id, error_result)
            return error_result
        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            timeout_result = CellExecutionResult(
                cell_id=cell_id,
                success=False,
                duration_ms=duration_ms,
                error=f"R cell execution timed out after {timeout_seconds}s",
            ).apply_remote_metadata(**remote_metadata)
            self.session.persist_display_output(cell_id, None)
            self.session.apply_execution_result_metadata(cell_id, timeout_result)
            return timeout_result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_result = CellExecutionResult(
                cell_id=cell_id,
                success=False,
                duration_ms=duration_ms,
                error=f"R execution failed: {e}",
            )
            self.session.persist_display_output(cell_id, None)
            self.session.apply_execution_result_metadata(cell_id, error_result)
            return error_result

    async def _dispatch_execution(
        self,
        worker_spec: Any,
        source: str,
        input_specs: dict[str, dict[str, str]],
        mount_specs: list[MountSpec],
        output_dir: Path,
        venv_path: Path | None,
        runtime_env: dict[str, str],
        timeout_seconds: float,
        remote_build_id: str | None = None,
        mutation_defines: list[str] | None = None,
        tables: dict[str, dict[str, Any]] | None = None,
        cell_id: str | None = None,
        cell_provenance_hash: str | None = None,
    ) -> tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]:
        """Dispatch one cell execution through the selected worker backend."""
        if worker_spec.backend == WorkerBackendType.LOCAL:
            return await self._dispatch_local(
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
            )

        if is_embedded_executor_worker(worker_spec):
            return await self._dispatch_embedded_executor(
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
            )

        if is_http_executor_worker(worker_spec):
            # Parent span for the remote side: its context travels in headers and the manifest,
            # so worker (and pool) spans join this trace.
            with trace_span(
                "notebook.dispatch",
                worker=worker_spec.name,
                build_id=remote_build_id,
                notebook_id=self.session.notebook_state.id,
                cell_id=cell_id,
            ):
                return await self._dispatch_http_executor(
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
    ) -> tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]:
        """Run the direct local execution path."""
        result = None
        execution_method = "cold"
        resolved_mounts = await self._prepare_mounts(mount_specs)
        manifest_path = self._write_manifest(
            source,
            input_specs,
            output_dir,
            runtime_env,
            resolved_mounts,
            mutation_defines=mutation_defines,
            tables=tables,
            cell_id=cell_id,
        )

        # Without an interpreter nothing runs, not even in a pre-started warm process:
        # _run_harness refuses below.
        if self.pool is not None and venv_path is not None:
            from strata.notebook.pool import PooledCellExecutor

            pool_result = await PooledCellExecutor.execute_with_pool(
                self.pool,
                manifest_path,
                self.session.path,
                timeout_seconds,
            )
            if pool_result is not None:
                result = pool_result
                execution_method = "warm"
                logger.debug(
                    "Executed cell %s with warm process",
                    manifest_path.parent.name,
                )

        if result is None:
            result = await self._run_harness(
                manifest_path,
                venv_path,
                timeout_seconds,
            )

        return result, output_dir, execution_method, resolved_mounts

    async def _dispatch_embedded_executor(
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
    ) -> tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]:
        """Run the bundle-based executor path locally for supported executor workers."""
        resolved_mounts = await self._prepare_mounts(mount_specs)
        manifest_path = self._write_manifest(
            source,
            input_specs,
            output_dir,
            runtime_env,
            resolved_mounts,
            mutation_defines=mutation_defines,
            tables=tables,
            cell_id=cell_id,
        )
        result = await self._run_harness(manifest_path, venv_path, timeout_seconds)

        bundle_path = output_dir / "notebook-output-bundle.tar"
        pack_notebook_output_bundle(bundle_path, result, output_dir)

        unpacked_dir = output_dir / "_executor_result"
        unpacked_result = unpack_notebook_output_bundle(bundle_path, unpacked_dir)
        return unpacked_result, unpacked_dir, "executor", resolved_mounts

    async def _locked_environment(
        self, worker_spec: Any, language: str = "python"
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
            spec = r_environment_spec(self.session.path)
            feature = "locked_r_environments"
        elif language == "python":
            spec = environment_spec(
                self.session.path, read_requested_python_minor(self.session.path)
            )
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
        """Run a cell through an external notebook executor over HTTP."""
        for mount in mount_specs:
            if mount.uri.startswith("file://"):
                raise RuntimeError(
                    f"Remote executor workers do not support file:// mounts: '{mount.name}'"
                )

        executor_url = str(worker_spec.config.url or "").strip()
        if not executor_url:
            raise RuntimeError(f"Executor worker '{worker_spec.name}' is missing config.url")

        worker_token = _resolve_worker_token(worker_spec)
        transport = str(worker_spec.config.transport or "direct").strip().lower()
        if transport in SIGNED_TRANSPORTS:
            return await self._dispatch_http_executor_with_manifest(
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
        environment = await self._locked_environment(worker_spec, language)
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

        for spec in input_specs.values():
            _add_file(str(spec["file"]))
            # Ship the injected blobs alongside the module descriptor.
            for inj in (spec.get("injected") or {}).values():
                _add_file(str(inj["file"]))

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
                    cell_timeout_message(timeout_seconds),
                    remote_error_code="TIMEOUT",
                )
            if response.status_code != 200:
                await response.aread()
                detail = self._extract_remote_error(response)
                raise RemoteExecutionError(
                    f"Remote executor '{worker_spec.name}' returned "
                    f"{response.status_code}: {detail}",
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
                    finished = await self._await_accepted_job(
                        response,
                        submit_url=executor_url,
                        headers=headers,
                        timeout_seconds=timeout_seconds,
                        cancel=lambda: self._cancel_remote_execution(
                            executor_url, build_id, worker_token
                        ),
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
                await asyncio.shield(
                    self._cancel_remote_execution(executor_url, build_id, worker_token)
                )
                raise
            except httpx.TimeoutException as exc:
                raise RemoteExecutionError(
                    cell_timeout_message(timeout_seconds),
                    remote_error_code="TIMEOUT",
                ) from exc
            except httpx.HTTPError as exc:
                raise RemoteExecutionError(
                    f"Remote executor request failed for worker '{worker_spec.name}': {exc}",
                    remote_error_code="REQUEST_FAILED",
                ) from exc
        finally:
            for handle in input_file_handles:
                handle.close()

        unpacked_dir = output_dir / "_executor_result"
        unpacked_result = unpack_notebook_output_bundle(bundle_path, unpacked_dir)
        return unpacked_result, unpacked_dir, "executor", {}

    async def _dispatch_http_executor_with_manifest(
        self,
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
        from strata.auth import get_principal
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

        base_url = str(worker_spec.config.strata_url or "").strip() or state.config.server_url
        principal = get_principal()
        tenant_id = principal.tenant if principal is not None else None
        principal_id = principal.id if principal is not None else None

        build_id = build_id or f"nbbuild-{uuid.uuid4().hex[:12]}"
        artifact_id = f"nb_remote_{self.session.notebook_state.id}_{build_id}"
        trace_context = current_trace_context()
        artifact_version: int | None = None
        failure_recorded = False

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
                    artifact_store.fail_artifact(artifact_id, artifact_version)
                except Exception:
                    logger.exception(
                        "Failed to mark notebook artifact %s@v=%s as failed",
                        artifact_id,
                        artifact_version,
                    )

        staged_input_specs, input_artifacts = self._stage_signed_transport_inputs(
            artifact_store=artifact_store,
            build_id=build_id,
            input_specs=input_specs,
            output_dir=output_dir,
            tenant_id=tenant_id,
            principal_id=principal_id,
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
            "_dispatch_mode": "external",
        }
        if language != "python":
            build_params["language"] = language
        # As the v1 path. In ``params``, because they change what the cell computes.
        _add_harness_params(build_params, mutation_defines, tables)
        environment = await self._locked_environment(worker_spec, language)
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

        try:
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
            )
            build_store.start_build(build_id)

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
                        "notebook_id": self.session.notebook_state.id,
                        "cell_id": cell_id,
                        "cell_provenance_hash": cell_provenance_hash,
                        # W3C trace context, for a worker reached without the headers (a
                        # dispatcher that forwards only the body).
                        **trace_context,
                    },
                    input_artifacts=input_artifacts,
                    max_output_bytes=state.config.max_transform_output_bytes,
                    blob_store=(
                        artifact_store.blob_store if state.config.artifact_presigned_urls else None
                    ),
                    # The worker uploads and finalizes after provisioning plus the whole
                    # cell run (and 5 minutes for the upload), so the setting is a floor.
                    url_expiry_seconds=max(
                        state.config.signed_url_expiry_seconds,
                        timeout_seconds + state.config.worker_provisioning_timeout_seconds + 300.0,
                    ),
                )
            ).to_dict()

            manifest_execute_url = self._manifest_execute_url(executor_url)
            worker_token = _resolve_worker_token(worker_spec)
            headers = {**trace_context}
            if worker_token:
                headers["Authorization"] = f"Bearer {worker_token}"
            # Before the worker can send a chunk; keyed by the session id, as the sockets are.
            relay = (
                console_relay.relaying(
                    build_id,
                    self.session.id,
                    cell_id,
                    shared_store=build_store if state.config.node_advertised_url else None,
                )
                if cell_id
                else nullcontext()
            )
            async with relay:
                async with httpx.AsyncClient(timeout=max(timeout_seconds + 10.0, 30.0)) as client:
                    response = await client.post(
                        manifest_execute_url, json=manifest, headers=headers
                    )
                if response.status_code == 202:
                    response = await self._await_accepted_job(
                        response,
                        submit_url=manifest_execute_url,
                        headers=headers,
                        timeout_seconds=timeout_seconds,
                        cancel=lambda: self._cancel_remote_execution(
                            executor_url, build_id, worker_token
                        ),
                        worker_spec=worker_spec,
                        cell_id=cell_id,
                    )
        except RemoteExecutionError as exc:
            _mark_failed(str(exc), exc.remote_error_code or "EXECUTOR_ERROR")
            raise
        except asyncio.CancelledError:
            _mark_failed("Notebook manifest execution cancelled", "CANCELLED")
            # Shielded: inside a cancellation an unshielded await is cancelled before the worker
            # hears.
            await asyncio.shield(
                self._cancel_remote_execution(executor_url, build_id, worker_token)
            )
            raise
        except httpx.TimeoutException as exc:
            _mark_failed("Notebook manifest execution timed out", "TIMEOUT")
            raise RemoteExecutionError(
                cell_timeout_message(timeout_seconds),
                remote_build_state="failed",
                remote_error_code="TIMEOUT",
            ) from exc
        except httpx.HTTPError as exc:
            _mark_failed(
                f"Remote executor request failed for worker '{worker_spec.name}': {exc}",
                "REQUEST_FAILED",
            )
            raise RemoteExecutionError(
                f"Remote executor request failed for worker '{worker_spec.name}': {exc}",
                remote_build_state="failed",
                remote_error_code="REQUEST_FAILED",
            ) from exc
        except Exception as exc:
            _mark_failed(str(exc), "SETUP_FAILED")
            raise RemoteExecutionError(
                f"Remote executor setup failed for worker '{worker_spec.name}': {exc}",
                remote_build_state="failed",
                remote_error_code="SETUP_FAILED",
            ) from exc

        try:
            if response.status_code == 408:
                _mark_failed("Notebook manifest execution timed out", "TIMEOUT")
                raise RemoteExecutionError(
                    cell_timeout_message(timeout_seconds),
                    remote_build_state="failed",
                    remote_error_code="TIMEOUT",
                )
            if response.status_code != 200:
                detail = self._extract_remote_error(response)
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
                    else (
                        f"Remote executor '{worker_spec.name}' returned "
                        f"{response.status_code}: {detail}"
                    )
                )
                _mark_failed(
                    error_message,
                    error_code,
                )
                refreshed_build = build_store.get_build(build_id)
                raise RemoteExecutionError(
                    error_message,
                    remote_build_state=(
                        refreshed_build.state if refreshed_build is not None else "failed"
                    ),
                    remote_error_code=error_code,
                )

            build = build_store.get_build(build_id)
            if build is None or build.state != "ready":
                build_error_message = build.error_message if build is not None else None
                build_error_code = (
                    build.error_code if build is not None and build.error_code else None
                )
                raise RemoteExecutionError(
                    build_error_message
                    or f"Notebook build {build_id} did not complete successfully",
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

            bundle_path = output_dir / "notebook-output-bundle.tar"
            with reader_cm as blob_reader, open(bundle_path, "wb") as dst:
                while True:
                    chunk = blob_reader.read(BLOB_STREAM_CHUNK_BYTES)
                    if not chunk:
                        break
                    dst.write(chunk)

            try:
                read_notebook_output_bundle_manifest_path(bundle_path)
            except Exception as exc:
                _mark_failed(
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
            _mark_failed("Notebook manifest execution cancelled", "CANCELLED")
            raise
        except RemoteExecutionError:
            raise
        except Exception as exc:
            _mark_failed(str(exc), "EXECUTOR_ERROR")
            raise RemoteExecutionError(
                str(exc),
                remote_build_state="failed",
                remote_error_code="EXECUTOR_ERROR",
            ) from exc

    def _cancel_url(self, executor_url: str, build_id: str) -> str:
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

    async def _cancel_remote_execution(
        self, executor_url: str, build_id: str, worker_token: str | None
    ) -> None:
        """Ask a worker to stop an execution we have stopped waiting for.

        The result could never land (the build is already failed), but this
        frees the machine. Every failure is logged and swallowed, since raising
        during cancellation would replace the reason the cell stopped.
        """
        headers = {"Authorization": f"Bearer {worker_token}"} if worker_token else None
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    self._cancel_url(executor_url, build_id), headers=headers
                )
            if response.status_code >= 400:
                logger.info(
                    "Worker refused cancel for build %s: HTTP %d",
                    build_id,
                    response.status_code,
                )
        except Exception as exc:
            logger.info("Could not cancel build %s on the worker: %s", build_id, exc)

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
        """Follow a job a worker accepted with 202 until it finishes.

        Queueing, booting and env pulls count against
        ``worker_provisioning_timeout_seconds``; the cell's timeout starts when
        the job reports ``running``. Either deadline cancels the job. The job URL
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
            getattr(self._lake_config(), "worker_provisioning_timeout_seconds", 600.0)
        )

        await self._broadcast_remote_phase(cell_id, worker_spec, "starting")
        started = _monotonic()
        running_since: float | None = None
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                try:
                    async with client.stream("GET", job_url, headers=headers) as reply:
                        if (
                            receive_bundle is not None
                            and reply.status_code == 200
                            and reply.headers.get("content-type", "").startswith(
                                "application/x-tar"
                            )
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
                        f"{reply.status_code}: {self._extract_remote_error(reply)}",
                        remote_build_state="failed",
                        remote_error_code="JOB_STATUS_FAILED",
                    )
                job = reply.json()
                state = str(job.get("state", ""))
                if state in ("finished", "failed"):
                    status_code = int(
                        job.get("status_code") or (200 if state == "finished" else 502)
                    )
                    content = job.get("result")
                    if content is None:
                        content = {"detail": job.get("error") or f"job {state}"}
                    return httpx.Response(status_code, json=content)

                now = _monotonic()
                if state == "running":
                    if running_since is None:
                        running_since = now
                        await self._broadcast_remote_phase(cell_id, worker_spec, "running")
                    if now - running_since > timeout_seconds:
                        await cancel()
                        raise RemoteExecutionError(
                            cell_timeout_message(timeout_seconds),
                            remote_build_state="failed",
                            remote_error_code="TIMEOUT",
                        )
                elif now - started > provisioning_limit:
                    await cancel()
                    raise RemoteExecutionError(
                        f"Remote executor '{worker_spec.name}' did not start the job within "
                        f"{provisioning_limit:g}s (last state: {state or 'unknown'}); set "
                        "STRATA_WORKER_PROVISIONING_TIMEOUT_SECONDS, or keep a machine warm",
                        remote_build_state="failed",
                        remote_error_code="PROVISIONING_TIMEOUT",
                    )
                await asyncio.sleep(_JOB_POLL_SECONDS)

    async def _broadcast_remote_phase(
        self, cell_id: str | None, worker_spec: Any, phase: str
    ) -> None:
        """Move the cell's badge between ``starting`` (provisioning) and ``running``.

        Best effort: a failed broadcast must not fail the run.
        """
        if not cell_id:
            return
        try:
            from strata.notebook.protocol import MessageType
            from strata.notebook.ws import _broadcast_message, _make_message, next_notebook_sequence
            from strata.notebook.ws_payloads import cell_status_payload

            notebook_id = self.session.notebook_state.id
            await _broadcast_message(
                notebook_id,
                _make_message(
                    MessageType.CELL_STATUS,
                    next_notebook_sequence(notebook_id),
                    cell_status_payload(
                        cell_id,
                        "running",
                        remote_worker=worker_spec.name,
                        remote_transport=worker_transport(worker_spec),
                        remote_build_state=phase,
                    ),
                ),
            )
        except Exception:
            logger.debug("remote phase broadcast failed", exc_info=True)

    def _manifest_execute_url(self, executor_url: str) -> str:
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
            blob_data = input_path.read_bytes()
            source_token = source_uri or f"local:{file_name}"
            source_hash = hashlib.sha256(source_token.encode("utf-8")).hexdigest()[:16]
            artifact_id = (
                f"nb_remote_input_{self.session.notebook_state.id}_{source_hash}_{var_name}"
            )
            provenance_hash = hashlib.sha256(
                json.dumps(
                    {
                        "source": source_token,
                        "content_type": content_type,
                        "byte_hash": hashlib.sha256(blob_data).hexdigest(),
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
            artifact_store.write_blob(artifact_id, version, blob_data)
            finalized = artifact_store.finalize_artifact(
                artifact_id,
                version,
                schema_json=json.dumps({"content_type": content_type}),
                row_count=0,
                byte_size=len(blob_data),
                content_sha256=hashlib.sha256(blob_data).hexdigest(),
            )
            if finalized is None:
                raise RuntimeError(
                    f"Failed to finalize staged notebook input artifact for {var_name}"
                )
            input_artifacts.append((finalized.id, finalized.version))
            return f"strata://artifact/{finalized.id}@v={finalized.version}"

        for var_name, spec in sorted(input_specs.items()):
            file_name = str(spec.get("file", "")).strip()
            if not file_name:
                raise RuntimeError(
                    "Signed notebook executor transport is missing a local "
                    f"input file for {var_name}"
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

    def _parse_artifact_uri(self, input_uri: str) -> tuple[str, int]:
        """Parse a canonical artifact URI into (artifact_id, version)."""
        import re

        match = re.fullmatch(r"strata://artifact/([^@]+)@v=(\d+)", input_uri)
        if match is None:
            raise RuntimeError(
                "Signed notebook executor transport only supports artifact inputs, "
                f"got {input_uri!r}"
            )
        return match.group(1), int(match.group(2))

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

        # Cell-level mounts already include notebook defaults from parser.py.
        cell_mounts_spec = cell.mounts if cell else []

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
    ) -> tuple[list[str], bool]:
        """Compute mount fingerprints without preparing local materializations."""
        self._mount_resolver.credential_resolver = self._credential_resolver()
        mount_fingerprints: list[str] = []
        has_rw_mount = False
        for mount in sorted(mount_specs, key=lambda item: item.name):
            fingerprint = await mount_fingerprint(self._mount_resolver, mount)
            if fingerprint is None:
                has_rw_mount = True
            else:
                mount_fingerprints.append(fingerprint)
        return mount_fingerprints, has_rw_mount

    def _credential_resolver(self) -> CredentialResolver:
        """Named credentials as this notebook sees them right now.

        Rebuilt per use: a secret manager fills the env after the session opens,
        and a rotated value must reach the next run.
        """
        return CredentialResolver.from_config(
            self._lake_config(), env=dict(self.session.notebook_state.env)
        )

    async def _prepare_mounts(
        self,
        mount_specs: list[MountSpec],
    ) -> dict[str, ResolvedMount]:
        """Prepare local mount materializations for local execution paths."""
        if not mount_specs:
            return {}
        self._mount_resolver.credential_resolver = self._credential_resolver()
        return await self._mount_resolver.prepare_mounts(mount_specs)

    async def _resolve_fetches(
        self, fetch_specs: list[FetchSpec]
    ) -> tuple[
        list[str], dict[str, Path], dict[str, str], dict[str, float], str | None, str | None
    ]:
        """Check every ``@fetch`` right before a run and fingerprint its bytes.

        Always rechecked (``max_age=0``) so the run records what the URL served
        when it ran. Returns fingerprints, a path per name, the lineage input per
        URL, when each URL's bytes were downloaded, and the first failure and its
        code, if any.
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
                resolved = await asyncio.to_thread(registry.resolve, spec)
                dataset = await asyncio.to_thread(copy_into, registry, resolved, store)
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
        store = self.session.get_artifact_manager().artifact_store
        for name, dataset in sorted(datasets.items()):
            artifact_id, _, version = dataset.local_ref.partition("@v=")
            blob = store.read_blob(artifact_id, int(version))
            if blob is None:
                continue
            ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(dataset.content_type, "")
            file_name = f"__dataset_{name}{ext}"
            (output_dir / file_name).write_bytes(blob)
            input_specs[name] = {
                "content_type": dataset.content_type,
                "file": file_name,
                "uri": f"strata://artifact/{dataset.local_ref}",
            }

    def _input_refs(self, cell_id: str, variant: str | None = None) -> dict[str, str]:
        """Inputs an artifact of *cell_id* records: upstreams, fetch digests, dataset versions.

        With ``variant`` set, a chained ``# @per_variant`` cell records only the
        upstream instance it zipped to.
        """
        return {
            **self.session._collect_input_refs(cell_id, variant=variant),
            **self._fetch_refs.get(cell_id, {}),
            **self._dataset_refs.get(cell_id, {}),
        }

    async def _fingerprint_tables(
        self,
        table_specs: list[TableSpec],
        config: Any = None,
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
        return await loop.run_in_executor(None, fingerprint_tables, table_specs, config, env)

    def _lake_config(self):
        """Server config when running inside the server, else loaded fresh."""
        try:
            from strata.server import get_state

            return get_state().config
        except RuntimeError:
            from strata.config import StrataConfig

            return StrataConfig.load()

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
        from strata.auth import remote_store_headers

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

        tables: dict[str, dict[str, Any]] = {}
        for spec in table_specs:
            if spec.name not in table_snapshots:
                raise RuntimeError(
                    f"@table {spec.name}: could not resolve a snapshot for "
                    f"{spec.uri!r}; is the catalog reachable?"
                )
            tables[spec.name] = {"uri": spec.uri, "snapshot_id": table_snapshots[spec.name]}
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
    ) -> Path:
        """Write the harness manifest for one local execution."""
        manifest_mounts = {
            name: {
                "uri": rm.spec.uri,
                "mode": rm.spec.mode.value,
                "local_path": str(rm.local_path),
            }
            for name, rm in resolved_mounts.items()
        }
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
        }
        if loop_config is not None:
            manifest["loop"] = loop_config
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        return manifest_path

    def _extract_remote_error(self, response: httpx.Response) -> str:
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
        from strata.notebook.llm.config import llm_config_for_session
        from strata.notebook.prompt_executor import execute_prompt_cell

        if materialize_upstreams:
            failure = await self._materialize_upstreams_or_failure(cell_id, start_time, "prompt")
            if failure is not None:
                return failure

        llm_config = llm_config_for_session(self.session)
        if llm_config is None:
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                outputs={},
                stdout="",
                stderr="",
                error=(
                    "LLM not configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
                    "STRATA_AI_API_KEY in the notebook's environment (the Runtime panel, "
                    "or [env] in notebook.toml), or STRATA_AI_API_KEY where the server starts. "
                    "The server's key is not sent to an [ai] base_url of the notebook's own."
                ),
                cache_hit=False,
                duration_ms=int((time.time() - start_time) * 1000),
                execution_method="llm",
            )

        # The prompt executor caches by its own provenance (rendered text + model), which
        # compute_staleness can't see. Record the standard hash so "can_preserve_ready" matches.
        prov = await self._compute_cell_provenance(cell_id, source)
        standard_provenance = prov.provenance_hash

        result_dict = await execute_prompt_cell(
            self.session,
            cell_id,
            source,
            llm_config,
            use_cache=use_cache,
            on_delta=self.on_prompt_delta,
        )

        if result_dict.get("success"):
            # Persisted, like every other cell kind's, or a prompt cell is idle after a restart
            # and re-issues a paid call.
            self.session.record_successful_execution_provenance(
                cell_id,
                standard_provenance,
                prov.source_hash,
                prov.env_hash,
            )

        return CellExecutionResult(
            cell_id=cell_id,
            success=result_dict["success"],
            outputs=result_dict["outputs"],
            display_outputs=result_dict.get("display_outputs") or [],
            display_output=result_dict.get("display_output"),
            stdout=result_dict.get("stdout", ""),
            stderr=result_dict.get("stderr", ""),
            error=result_dict.get("error"),
            cache_hit=result_dict.get("cache_hit", False),
            duration_ms=result_dict.get("duration_ms", 0),
            execution_method=result_dict.get("execution_method", "llm"),
            artifact_uri=result_dict.get("artifact_uri"),
            mutation_warnings=result_dict.get("mutation_warnings", []),
            validation_retries=int(result_dict.get("validation_retries", 0) or 0),
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
        from strata.notebook.sql.cell_executor import execute_sql_cell

        if materialize_upstreams:
            failure = await self._materialize_upstreams_or_failure(cell_id, start_time, "sql")
            if failure is not None:
                return failure

        result_dict = await execute_sql_cell(
            self.session,
            cell_id,
            source,
            use_cache=use_cache,
        )

        if result_dict.get("success"):
            cell = self.session.notebook_state.get_cell(cell_id)
            if cell is not None:
                cell.cache_hit = bool(result_dict.get("cache_hit"))

            # compute_staleness compares the generic triplet (inputs + source + env) to
            # ``last_provenance_hash`` and can't see the SQL-specific hash. Persist the generic
            # one, as ``_execute_prompt_cell`` does, so reopen marks the cell READY.
            prov = await self._compute_cell_provenance(cell_id, source)
            self.session.record_successful_execution_provenance(
                cell_id,
                prov.provenance_hash,
                prov.source_hash,
                prov.env_hash,
            )
            # Back the result-table preview with an artifact. Otherwise `save_cell_output` refuses
            # it, export drops it (``markdown_text`` is stripped at persist time and re-fetched
            # via the uri), and staleness can't resolve a cached display, so an upstream SQL cell
            # shows nothing or a stale table.
            stored_displays = self._store_inline_display_outputs(
                cell_id,
                prov.provenance_hash,
                result_dict.get("display_outputs") or [],
                source_hash=prov.source_hash,
                source=source,
                env_hash=prov.env_hash,
            )
            if stored_displays:
                result_dict["display_outputs"] = stored_displays
                result_dict["display_output"] = stored_displays[-1]
            # Record them where reopen and export read from, as the Python and R paths do.
            self.session.persist_display_outputs(
                cell_id, result_dict.get("display_outputs") or None
            )
        else:
            # A failed run clears what the cell showed, as in Python and R; otherwise the last
            # good table reads as a current result.
            self.session.persist_display_outputs(cell_id, None)

        # ``execute_sql_cell`` times only its own work; ``start_time`` also covers upstream
        # materialization and dispatch.
        duration_ms = (time.time() - start_time) * 1000

        result = CellExecutionResult(
            cell_id=cell_id,
            success=result_dict["success"],
            outputs=result_dict["outputs"],
            display_outputs=result_dict.get("display_outputs") or [],
            display_output=result_dict.get("display_output"),
            stdout=result_dict.get("stdout", ""),
            stderr=result_dict.get("stderr", ""),
            error=result_dict.get("error"),
            cache_hit=result_dict.get("cache_hit", False),
            duration_ms=int(duration_ms),
            execution_method=result_dict.get("execution_method", "sql"),
            artifact_uri=result_dict.get("artifact_uri"),
            mutation_warnings=result_dict.get("mutation_warnings", []),
        )
        # Record the run's error against its source and clear it on success, as Python and R
        # do, so a fixed query doesn't carry the previous error.
        self.session.apply_execution_result_metadata(cell_id, result)
        return result

    async def _execute_widget_cell(
        self,
        cell_id: str,
        source: str,
        start_time: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
    ) -> CellExecutionResult:
        """Execute a widget cell: materialize each control's value artifact.

        No upstreams and no subprocess. Like prompt/SQL it persists the generic
        provenance triplet so ``compute_staleness`` keeps it READY when values
        are unchanged.
        """
        del materialize_upstreams  # widgets have no upstream inputs

        from strata.notebook.widget_executor import execute_widget_cell

        result_dict = execute_widget_cell(self.session, cell_id, source, use_cache=use_cache)

        if result_dict.get("success"):
            # Publish each control's value artifact onto ``artifact_uris``, as a Python cell does
            # for its outputs. Otherwise downstream ``_collect_input_hashes`` sees no upstream
            # artifact and cache-hits the old output: dragging a slider never updates downstream.
            cell = self.session.notebook_state.get_cell(cell_id)
            if cell is not None:
                cell.artifact_uris = {
                    name: out["artifact_uri"]
                    for name, out in (result_dict.get("outputs") or {}).items()
                    if isinstance(out, dict) and out.get("artifact_uri")
                }

            prov = await self._compute_cell_provenance(cell_id, source)
            self.session.record_successful_execution_provenance(
                cell_id,
                prov.provenance_hash,
                prov.source_hash,
                prov.env_hash,
            )

        return CellExecutionResult(
            cell_id=cell_id,
            success=result_dict["success"],
            outputs=result_dict.get("outputs", {}),
            display_outputs=result_dict.get("display_outputs") or [],
            error=result_dict.get("error"),
            cache_hit=result_dict.get("cache_hit", False),
            duration_ms=int((time.time() - start_time) * 1000),
            execution_method=result_dict.get("execution_method", "widget"),
            artifact_uri=result_dict.get("artifact_uri"),
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
    # ② Collect input hashes (upstream artifacts are guaranteed to exist)
    # ------------------------------------------------------------------

    def _collect_input_hashes(self, cell_id: str) -> list[str]:
        """Provenance hashes from upstream artifacts, after ``_materialize_upstreams``.

        Delegates to ``session._collect_input_hashes`` so the hash matches what
        ``compute_staleness`` recomputes.
        """
        return self.session._collect_input_hashes(cell_id)

    # ------------------------------------------------------------------
    # ④-a Load input blobs (guaranteed to exist after step ①)
    # ------------------------------------------------------------------

    def _load_input_blobs(
        self,
        cell_id: str,
        output_dir: Path,
        *,
        fanout_group: str | None = None,
        fanout_variant: str | None = None,
    ) -> dict[str, Any]:
        """Write upstream variable blobs from the artifact store into *output_dir*.

        Upstreams must already be materialized. With ``fanout_variant`` set, a
        reference from the cell's own ``fanout_group`` binds only that variant's
        scalar; other sweep groups, and upstream fan-out cells, collapse to a
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
        dag = self.session.dag

        def _load_artifact_spec(artifact_id: str, file_stem: str) -> dict[str, str] | None:
            """Write one artifact's blob to *file_stem*; return its spec, or None if absent."""
            artifact = artifact_mgr.artifact_store.get_latest_version(artifact_id)
            if artifact is None:
                return None
            blob_data = artifact_mgr.load_artifact_data(artifact_id, artifact.version)
            content_type = "pickle/object"
            if artifact.transform_spec:
                try:
                    params = json.loads(artifact.transform_spec).get("params", {})
                except ValueError:
                    params = {}  # malformed transform_spec → default content type
                content_type = params.get("content_type") or content_type
            ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".pickle")
            input_file = output_dir / f"{safe_filename_stem(file_stem)}{ext}"
            with open(input_file, "wb") as f:
                f.write(blob_data)
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
            content_type = "pickle/object"
            if artifact.transform_spec:
                try:
                    params = json.loads(artifact.transform_spec).get("params", {})
                except ValueError:
                    params = {}
                content_type = params.get("content_type") or content_type
            ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".pickle")
            input_file = output_dir / f"{safe_filename_stem(file_stem)}{ext}"
            with open(input_file, "wb") as f:
                f.write(artifact_mgr.load_artifact_data(art_id, version))
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
                producer = dag.variable_producer.get(var_name) if dag else None
                artifact_id = f"nb_{notebook_id}_cell_{upstream_id}_var_{var_name}"
                try:
                    if isinstance(producer, SweepProducer):
                        if producer.fanout_cell is not None:
                            # Upstream is a @per_variant fan-out cell: its outputs
                            # live under ``@variant=`` subkeys of one cell.
                            if fanout_variant is not None and producer.group == fanout_group:
                                # Chained fan-out: this instance zips to the upstream
                                # instance of the same variant (scalar bind).
                                vid = artifact_mgr.cell_artifact_id(
                                    producer.fanout_cell, var_name, variant=fanout_variant
                                )
                                spec = _load_artifact_spec(vid, var_name)
                                if spec is not None:
                                    input_specs[var_name] = spec
                                continue
                            # Collapse consumer: gather all variants into a dict
                            # (partial set on any missing variant).
                            bundle = input_specs.setdefault(
                                var_name, {"kind": "sweep_dict", "variants": {}}
                            )
                            for vname, _cid in producer.variants:
                                vid = artifact_mgr.cell_artifact_id(
                                    producer.fanout_cell, var_name, variant=vname
                                )
                                vspec = _load_artifact_spec(vid, f"{var_name}__{vname}")
                                if vspec is not None:
                                    bundle["variants"][vname] = vspec
                            continue

                        variant_name = next(
                            (name for name, cid in producer.variants if cid == upstream_id),
                            None,
                        )
                        if variant_name is None:
                            continue

                        if fanout_variant is not None and producer.group == fanout_group:
                            # This cell fans out over this group: bind only its own variant as a
                            # SCALAR. Chained fan-out zips by variant name.
                            if variant_name != fanout_variant:
                                continue
                            spec = _load_artifact_spec(artifact_id, var_name)
                            if spec is not None:
                                input_specs[var_name] = spec
                            continue

                        spec = _load_artifact_spec(artifact_id, f"{var_name}__{variant_name}")
                        if spec is None:
                            # A variant with no artifact is dropped. A failed one never gets here:
                            # _materialize_upstreams raises first.
                            logger.error(
                                "Sweep variant '%s' of '%s' has no artifact; "
                                "dropping it from the dict.",
                                variant_name,
                                var_name,
                            )
                            continue
                        bundle = input_specs.setdefault(
                            var_name, {"kind": "sweep_dict", "variants": {}}
                        )
                        bundle["variants"][variant_name] = spec
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
                except Exception:
                    logger.exception(
                        "Failed to load input %s from artifact store",
                        var_name,
                    )

        return input_specs

    # ------------------------------------------------------------------
    # ⑤ Store output artifacts
    # ------------------------------------------------------------------

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
    ) -> bool:
        """Persist consumed output variables as artifacts; True iff all were stored.

        With ``store_leaf_outputs``, a leaf cell's variables are stored too, best-effort.

        With ``variant`` set, ids get an ``@variant={name}`` suffix so fan-out
        instances do not collide.
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None or self.session.dag is None:
            return True

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
            return True

        all_stored = True
        staged: list[StagedVersion] = []
        staged_names: list[tuple[str, str]] = []

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

        for var_name in consumed_vars | leaf_vars:
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
                all_stored = False
                break

            try:
                with open(output_file, "rb") as f:
                    blob_data = f.read()

                content_type = content_type_map.get(ext, "pickle/object")
                var_provenance = derive_subkey(provenance_hash, var_name)

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
                    )
                )
                staged_names.append((var_name, content_type))
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
        try:
            if not all_stored:
                artifact_mgr.discard_cell_outputs(staged)
                return False
            stored = artifact_mgr.finalize_cell_outputs(staged)
        except Exception:
            logger.exception("Failed to store the outputs of cell %s", cell_id)
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
    ) -> None:
        """Persist a leaf cell's console output as a provenance-keyed artifact.

        So a cell that only prints can still cache-hit and replay its output. Stored
        even when both streams are empty: a leaf has no other artifact, so the console
        is the record that it ran under this provenance (cache hit, ready on reopen).
        """
        artifact_mgr = self.session.get_artifact_manager()
        input_versions = self._input_refs(cell_id)
        blob = json.dumps({"stdout": stdout, "stderr": stderr}).encode("utf-8")
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
            extra_params=self._fetch_params(cell_id),
        )

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
        notebook_id = self.session.notebook_state.id
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
            canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var___display__{index}"
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
    ) -> list[dict[str, Any]]:
        """Persist ordered cell display outputs as canonical artifacts."""
        if not display_outputs:
            return []

        artifact_mgr = self.session.get_artifact_manager()
        input_versions = self._input_refs(cell_id)
        stored_displays: list[dict[str, Any]] = []

        for index, display_output in enumerate(display_outputs):
            file_name = str(display_output.get("file", "")).strip()
            content_type = str(display_output.get("content_type", "")).strip()
            if not file_name or not content_type:
                continue

            output_file = output_dir / file_name
            if not output_file.exists():
                continue

            blob_data = output_file.read_bytes()
            row_count = display_output.get("rows")
            display_provenance = derive_subkey(provenance_hash, f"__display__{index}")
            artifact_version = artifact_mgr.store_cell_output(
                cell_id=cell_id,
                variable_name=f"__display__{index}",
                blob_data=blob_data,
                content_type=content_type,
                row_count=row_count if isinstance(row_count, int) else None,
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
            stored_displays.append(stored_display)

        return stored_displays

    def _write_module_export_outputs(
        self,
        cell_id: str,
        source: str,
        output_dir: Path,
        provenance_hash: str,
        outputs: dict[str, Any],
    ) -> str | None:
        """Write synthetic module artifacts for cross-cell defs/classes.

        Returns an error string when a downstream-consumed definition cannot be
        exported safely.
        """
        if self.session.dag is None:
            return None

        consumed_vars = self.session.dag.consumed_variables.get(cell_id, set())
        if not consumed_vars:
            return None

        # Upstream-produced names can be hydrated into the synthetic module at load time
        # instead of blocking a def/class that closes over them.
        producer = self.session.dag.variable_producer
        cells_by_id = {c.id: c for c in self.session.notebook_state.cells}
        this_cell = cells_by_id.get(cell_id)
        cross_cell = frozenset(
            v
            for v in ([*this_cell.references, *this_cell.builtin_references] if this_cell else [])
            if isinstance(producer.get(v), str) and producer[v] != cell_id
        )
        # Same-cell runtime values (a loaded model, a cwd-derived path) can be hydrated too.
        same_cell_runtime = runtime_binding_names(source)
        injectable = cross_cell | same_cell_runtime

        export_plan = build_module_export_plan(source, injectable=injectable)
        exportable_vars = sorted(set(export_plan.exported_symbols) & set(consumed_vars))
        blocked_vars = sorted(export_plan.blocking_symbols & set(consumed_vars))
        if not exportable_vars and not blocked_vars:
            return None

        if not export_plan.is_exportable:
            joined_vars = ", ".join(sorted(set(exportable_vars) | set(blocked_vars)))
            return (
                "This cell defines reusable code used downstream "
                f"({joined_vars}), but it cannot be shared across cells yet: "
                f"{export_plan.format_error()}"
            )

        # Constants alone don't trigger module-export (``x = 1`` stays a plain int). They
        # ride the module only when the cell also exports a def/class.
        code_exports = [
            name
            for name in exportable_vars
            if export_plan.exported_symbols[name].kind in ("function", "async function", "class")
        ]
        if not code_exports:
            return None

        # Pin each injected name to the upstream variable's versioned artifact URI. If one
        # can't be pinned, block rather than ship a module that NameErrors at call time.
        injected_refs: dict[str, str] = {}
        for name in sorted(export_plan.injected_inputs):
            prod_id = producer.get(name)
            if isinstance(prod_id, str) and prod_id != cell_id:
                prod_cell = cells_by_id.get(prod_id)
                uri = prod_cell.artifact_uris.get(name) if prod_cell is not None else None
            else:
                # Same-cell: persist the runtime value now (store is idempotent by provenance, so
                # the later _store_outputs pass is a no-op) and pin its URI.
                uri = self._store_same_cell_injected(
                    cell_id, name, output_dir, outputs, provenance_hash, source
                )
            if uri is None:
                return (
                    "This cell defines reusable code used downstream "
                    f"({', '.join(sorted(set(exportable_vars) | set(blocked_vars)))}), "
                    f"but a value it closes over (`{name}`) isn't available to "
                    "share yet."
                )
            injected_refs[name] = uri

        source_hash = compute_source_hash(source)
        notebook_id = self.session.notebook_state.id
        # Fold the injected identity into the module name so sys.modules can't alias two
        # hydrations of the same slice.
        injected_tag = hashlib.sha256(
            "|".join(f"{k}={v}" for k, v in sorted(injected_refs.items())).encode()
        ).hexdigest()

        module_suffix = source_hash[:12]
        if injected_refs:
            module_suffix = f"{source_hash[:12]}_{injected_tag[:8]}"

        for var_name in exportable_vars:
            symbol = export_plan.exported_symbols[var_name]
            descriptor = {
                "module_name": (f"nb_{notebook_id}_{cell_id}_{var_name}_{module_suffix}"),
                "symbol_name": var_name,
                "kind": symbol.kind,
                "source": export_plan.module_source,
                "provenance_hash": derive_subkey(provenance_hash, var_name),
                "injected": injected_refs,
            }
            output_file = output_dir / f"{safe_filename_stem(var_name)}.cell_module.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(descriptor, f)

            outputs[var_name] = {
                "content_type": "module/cell",
                "file": output_file.name,
                "bytes": output_file.stat().st_size,
                "type": symbol.kind,
                "preview": f"<{symbol.kind} {var_name}>",
            }

        return None

    def _store_same_cell_injected(
        self,
        cell_id: str,
        var_name: str,
        output_dir: Path,
        outputs: dict[str, Any],
        provenance_hash: str,
        source: str,
    ) -> str | None:
        """Store a same-cell value an exported def closes over; return its pinned URI or None.

        The harness already wrote its blob to *output_dir*. ``store_cell_output``
        dedups by provenance, so a later ``_store_outputs`` of it is a no-op.
        """
        spec = outputs.get(var_name)
        if not isinstance(spec, dict) or "file" not in spec or "error" in spec:
            return None
        blob_path = output_dir / spec["file"]
        if not blob_path.exists():
            return None
        artifact_mgr = self.session.get_artifact_manager()
        artifact_version = artifact_mgr.store_cell_output(
            cell_id=cell_id,
            variable_name=var_name,
            blob_data=blob_path.read_bytes(),
            content_type=spec.get("content_type", "pickle/object"),
            provenance_hash=derive_subkey(provenance_hash, var_name),
            source_hash=compute_source_hash(source),
            source=source,
        )
        return f"strata://artifact/{artifact_version.id}@v={artifact_version.version}"

    # ------------------------------------------------------------------
    # Harness helpers
    # ------------------------------------------------------------------

    def _harness_command(
        self, manifest_path: Path, venv_python: Path, harness_user: HarnessUser | None
    ) -> list[str] | None:
        """The argv that starts the cold Python harness; ``None`` without uv.

        A separate method so a test can skip ``uv run`` without replacing the
        spawn's environment, user and refusal logic.
        """
        if harness_user is not None:
            # Straight to the venv interpreter: ``uv run`` wants a writable uv cache a separate
            # user would need arranged, and the venv is already synced.
            return [str(venv_python), str(self.harness_path), str(manifest_path)]
        uv = resolve_uv()
        if uv is None:
            return None
        return [
            uv,
            "run",
            "--directory",
            str(self.session.path),
            "python",
            str(self.harness_path),
            str(manifest_path),
        ]

    async def _run_harness(
        self,
        manifest_path: Path,
        venv_python: Path | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Run the harness via uv, or as the harness user; no-op if ``venv_python`` is None."""
        if venv_python is None:
            return {
                "success": False,
                "error": _no_interpreter_message(self.session),
                "stderr": "",
                "stdout": "",
                "variables": {},
            }
        try:
            harness_user = resolve_harness_user()
        except LocalExecutionRefused as exc:
            return _refused_result(exc)
        cmd = self._harness_command(manifest_path, venv_python, harness_user)
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
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.session.path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # ``uv run`` must find the notebook's .venv, not the server's environment.
            env=uv_env(identity_env(self._harness_env(), harness_user)),
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
                "Cell execution cancelled; terminating harness subprocess tree pid=%s",
                proc.pid,
            )
            try:
                await asyncio.shield(terminate_subprocess_tree(proc))
            except Exception:
                logger.exception(
                    "Failed to terminate cancelled harness subprocess tree pid=%s",
                    proc.pid,
                )
            raise
        except TimeoutError:
            await terminate_subprocess_tree(proc)
            raise TimeoutError()

        # Separate from manifest.json and user files like result.json. Absent means the
        # harness crashed before its finally block (typically an import error): surface
        # stderr instead of "Unknown error".
        result_path = manifest_path.parent / "harness-result.json"
        if not result_path.exists():
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
            return {
                "success": False,
                "error": (
                    stderr_text.strip() or "Harness exited without producing a result manifest"
                ),
                "stderr": stderr_text,
                "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
                "variables": {},
            }
        with open(result_path) as f:
            return json.load(f)

    async def _run_r_harness(
        self,
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

        cmd = [rscript, str(self.r_harness_path), str(manifest_path)]

        from strata.notebook.process_tree import (
            subprocess_kwargs_for_new_group,
            terminate_subprocess_tree,
        )

        hand_over(manifest_path.parent, harness_user)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.session.path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=identity_env(self._harness_env(), harness_user),
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
                "error": (
                    stderr_text.strip() or "Rscript exited without producing a result manifest"
                ),
                "stderr": stderr_text,
                "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
                "variables": {},
            }
        with open(result_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Loop cell execution (sequential, fresh subprocess per iteration)
    # ------------------------------------------------------------------

    _LOOP_CONTENT_TYPE_EXT = {
        "arrow/ipc": ".arrow",
        "json/object": ".json",
        "pickle/object": ".pickle",
        "module/import": ".module.json",
        "module/cell": ".cell_module.json",
        "module/cell-instance": ".cell_instance.pickle",
    }

    def _cached_loop_result(
        self,
        cell_id: str,
        loop: LoopAnnotation,
        cell_provenance: str,
        start_time: float,
    ) -> CellExecutionResult | None:
        """The loop's own result from a previous identical run, or None.

        A hit needs the carry and every consumed variable to be the cell's canonical
        rows under this provenance: a duplicate's row is not this cell's result, and
        an output read only since the last run was never stored.
        """
        artifact_mgr = self.session.get_artifact_manager()
        if artifact_mgr.find_cached(derive_subkey(cell_provenance, loop.carry)) is None:
            return None
        consumed_vars = (
            self.session.dag.consumed_variables.get(cell_id, set()) if self.session.dag else set()
        )
        notebook_id = self.session.notebook_state.id
        uris: dict[str, str] = {}
        for var_name in sorted({loop.carry, *consumed_vars}):
            canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{var_name}"
            canonical = artifact_mgr.artifact_store.get_latest_version(canonical_id)
            if canonical is None or canonical.provenance_hash != derive_subkey(
                cell_provenance, var_name
            ):
                return None
            uris[var_name] = f"strata://artifact/{canonical.id}@v={canonical.version}"
        uri = uris[loop.carry]
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is not None:
            cell.cache_hit = True
        self._set_loop_artifact_uris(cell_id, uri, consumed_vars, uris)
        result = CellExecutionResult(
            cell_id=cell_id,
            success=True,
            outputs={},
            duration_ms=(time.time() - start_time) * 1000,
            cache_hit=True,
            artifact_uri=uri,
            execution_method="cached",
        )
        self.session.apply_execution_result_metadata(cell_id, result)
        return result

    def _set_loop_artifact_uris(
        self,
        cell_id: str,
        carry_uri: str,
        consumed_vars: set[str],
        uris: dict[str, str],
    ) -> None:
        """Record a loop's outputs on the cell the way ``_store_outputs`` does: consumed only."""
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            return
        for var_name, uri in uris.items():
            if var_name in consumed_vars:
                cell.artifact_uris[var_name] = uri
        cell.artifact_uri = carry_uri

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
    ) -> CellExecutionResult:
        """Execute a loop cell by running the body up to ``loop.max_iter`` times.

        Local worker only, no rw mounts. Each iteration is a fresh harness
        subprocess; the carry passes in as a named input and out as a named
        output, stored with an ``@iter=k`` suffix.
        """
        annotations = parse_annotations(source)
        effective_worker = self._resolve_effective_worker(cell_id, annotations.worker)
        if effective_worker != "local":
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                error=(
                    f"Loop cells currently run only on worker 'local'; got "
                    f"'{effective_worker}'. Remote-worker loops are not in "
                    f"the Phase 1 scope."
                ),
                execution_method="loop",
            )

        if materialize_upstreams:
            failure = await self._materialize_upstreams_or_failure(cell_id, start_time, "loop")
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

        mount_specs = self._resolve_cell_mount_specs(cell_id, source)
        _, has_rw_mount = await self._fingerprint_mounts(mount_specs)
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

        runtime_env = self._resolve_effective_runtime_env(cell_id, annotations.env)

        # Computed up front so the loop is skipped on a cache hit. The loop dispatch precedes
        # ``execute_cell``'s cache check, and ``_materialize_upstreams`` calls ``execute_cell``
        # on every upstream assuming it caches, so otherwise any downstream run re-runs the loop.
        prov = await self._compute_cell_provenance(
            cell_id,
            source,
            annotations=annotations,
            mount_specs=mount_specs,
        )
        cell_provenance = prov.provenance_hash
        env_hash = prov.env_hash
        source_hash = prov.source_hash
        carry_var_provenance = derive_subkey(cell_provenance, loop.carry)
        if use_cache:
            cached = self._cached_loop_result(cell_id, loop, cell_provenance, start_time)
            if cached is not None:
                return cached

        try:
            carry_blob, carry_content_type = self._resolve_loop_seed(cell_id, loop)
        except ValueError as exc:
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                error=str(exc),
                execution_method="loop",
            )

        artifact_mgr = self.session.get_artifact_manager()

        # Downstream-consumed variables beyond the carry must be persisted too, or a later
        # cell reading e.g. ``final_metrics`` finds the name unbound.
        consumed_vars = (
            self.session.dag.consumed_variables.get(cell_id, set()) if self.session.dag else set()
        )
        extra_consumed = sorted(consumed_vars - {loop.carry})
        # var -> (blob, content_type), captured each iteration (the tmpdir dies with the
        # ``with`` block); the last capture is the final state.
        extra_blobs: dict[str, tuple[bytes, str]] = {}
        # An in-place update (``state["i"] += 1``) keeps the carry's id(), so the harness
        # serializes it only when it is listed as mutated.
        loop_cell = self.session.notebook_state.get_cell(cell_id)
        mutation_defines = sorted(
            {loop.carry, *(loop_cell.mutation_defines if loop_cell is not None else [])}
        )

        final_artifact_uri: str | None = None
        final_result: dict[str, Any] | None = None
        loop_run = uuid.uuid4().hex
        combined_stdout: list[str] = []
        combined_stderr: list[str] = []
        all_mutation_warnings: list[MutationWarning] = []

        for k in range(loop.max_iter):
            with tempfile.TemporaryDirectory(prefix=f"strata_loop_iter_{k}_") as tmpdir:
                output_dir = Path(tmpdir)

                # This writes the upstream seed for the carry too; it is overwritten below so iter k
                # sees iter k-1's output.
                input_specs = self._load_input_blobs(cell_id, output_dir)

                ext = self._LOOP_CONTENT_TYPE_EXT.get(carry_content_type, ".pickle")
                carry_file = f"{loop.carry}{ext}"
                (output_dir / carry_file).write_bytes(carry_blob)
                input_specs[loop.carry] = {
                    "content_type": carry_content_type,
                    "file": carry_file,
                }

                resolved_mounts = await self._prepare_mounts(mount_specs)
                loop_config: dict[str, Any] | None = (
                    {"until_expr": loop.until_expr, "iteration": k}
                    if loop.until_expr is not None
                    else {"iteration": k}
                )
                manifest_path = self._write_manifest(
                    source,
                    input_specs,
                    output_dir,
                    runtime_env,
                    resolved_mounts,
                    mutation_defines=mutation_defines,
                    loop_config=loop_config,
                    cell_id=cell_id,
                )

                venv_path = self.session.venv_python
                try:
                    result = await self._run_harness(manifest_path, venv_path, timeout_seconds)
                except TimeoutError:
                    duration_ms = (time.time() - start_time) * 1000
                    return CellExecutionResult(
                        cell_id=cell_id,
                        success=False,
                        error=(
                            f"Loop cell iter {k} timed out after "
                            f"{timeout_seconds}s (per-iteration timeout)."
                        ),
                        stdout="\n".join(combined_stdout),
                        stderr="\n".join(combined_stderr),
                        duration_ms=duration_ms,
                        execution_method="loop",
                    )

                combined_stdout.append(result.get("stdout", ""))
                combined_stderr.append(result.get("stderr", ""))
                all_mutation_warnings.extend(result.get("mutation_warnings", []))
                final_result = result

                if not result.get("success", False):
                    traceback_text = result.get("traceback") or None
                    # ``str()`` of a bare ``assert`` is empty; the traceback's last line is not.
                    error_msg = result.get("error") or (
                        traceback_text.strip().splitlines()[-1]
                        if traceback_text
                        else "Unknown error"
                    )
                    duration_ms = (time.time() - start_time) * 1000
                    return CellExecutionResult(
                        cell_id=cell_id,
                        success=False,
                        error=f"Loop cell iter {k} failed: {error_msg}",
                        traceback=traceback_text,
                        stdout="\n".join(combined_stdout),
                        stderr="\n".join(combined_stderr),
                        duration_ms=duration_ms,
                        execution_method="loop",
                        mutation_warnings=all_mutation_warnings,
                    )

                loop_state = result.get("loop") or {}
                if loop_state.get("error"):
                    # A predicate that cannot be evaluated would otherwise run every iteration and
                    # report success.
                    return CellExecutionResult(
                        cell_id=cell_id,
                        success=False,
                        error=f"Loop cell iter {k}: {loop_state['error']}",
                        stdout="\n".join(combined_stdout),
                        stderr="\n".join(combined_stderr),
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
                    return CellExecutionResult(
                        cell_id=cell_id,
                        success=False,
                        error=(
                            f"Loop cell iter {k} did not produce carry "
                            f"variable '{loop.carry}': {detail}. The cell "
                            f"body must rebind `{loop.carry}` every "
                            f"iteration."
                        ),
                        stdout="\n".join(combined_stdout),
                        stderr="\n".join(combined_stderr),
                        duration_ms=duration_ms,
                        execution_method="loop",
                        mutation_warnings=all_mutation_warnings,
                    )

                new_content_type = str(carry_meta.get("content_type", "pickle/object"))
                new_carry_file = carry_meta.get("file")
                if not isinstance(new_carry_file, str):
                    raise RuntimeError(
                        f"Loop cell iter {k} produced carry metadata without "
                        f"a 'file' entry: {carry_meta}"
                    )
                new_carry_path = output_dir / new_carry_file
                if not new_carry_path.exists():
                    raise RuntimeError(
                        f"Loop cell iter {k} carry file not produced by harness: {new_carry_path}"
                    )
                new_carry_blob = new_carry_path.read_bytes()

                # Capture while the iteration tmpdir is still alive.
                for extra_var in extra_consumed:
                    extra_meta = result.get("variables", {}).get(extra_var)
                    if (
                        not isinstance(extra_meta, dict)
                        or extra_meta.get("content_type") == "error"
                    ):
                        continue
                    extra_file = extra_meta.get("file")
                    if not isinstance(extra_file, str):
                        continue
                    extra_path = output_dir / extra_file
                    if extra_path.exists():
                        extra_blobs[extra_var] = (
                            extra_path.read_bytes(),
                            str(extra_meta.get("content_type", "pickle/object")),
                        )

                # Chains through the previous iteration's carry bytes so identical chains are
                # detectable.
                prev_carry_hash = hashlib.sha256(carry_blob).hexdigest()
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
                final_artifact_uri = f"strata://artifact/{artifact.id}@v={artifact.version}"

                carry_blob = new_carry_blob
                carry_content_type = new_content_type

                iter_duration_ms = (time.time() - start_time) * 1000
                if self.on_iteration_complete is not None:
                    try:
                        await self.on_iteration_complete(
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
        # ``carry_var_provenance`` were computed before the loop for the cache check.

        # Environment identity for lineage and the team cache; the cost is the whole loop,
        # which is what a teammate skipping it saves.
        loop_build_env = str((final_result or {}).get("build_env") or "")
        loop_duration_ms = (time.time() - start_time) * 1000

        # Record inputs so a training loop's output has a lineage graph.
        loop_input_versions = self._input_refs(cell_id)

        # Staged, then finalized with the extra outputs below so all become current together.
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
                    provenance_hash=derive_subkey(cell_provenance, extra_var),
                    input_versions=loop_input_versions,
                    source_hash=source_hash,
                    source=source,
                    env_hash=env_hash,
                    build_env=loop_build_env,
                    build_duration_ms=loop_duration_ms,
                )
            )
            extra_outputs[extra_var] = {
                "content_type": extra_content_type,
                "file": (
                    f"{extra_var}{self._LOOP_CONTENT_TYPE_EXT.get(extra_content_type, '.pickle')}"
                ),
            }

        stored = artifact_mgr.finalize_cell_outputs(staged)
        canonical_artifact = stored[0]
        canonical_uri = f"strata://artifact/{canonical_artifact.id}@v={canonical_artifact.version}"
        # Downstream provenance hashes these, as for any other cell's consumed outputs.
        self._set_loop_artifact_uris(
            cell_id,
            canonical_uri,
            consumed_vars,
            {
                name: f"strata://artifact/{artifact.id}@v={artifact.version}"
                for name, artifact in zip([loop.carry, *extra_outputs], stored, strict=True)
            },
        )

        # Lets ``compute_staleness`` hit the "uncached ready" path for leaf loop cells.
        self.session.record_successful_execution_provenance(
            cell_id,
            cell_provenance,
            source_hash,
            env_hash,
        )
        # A leaf's console is its record of the run, so a cold open reads it ready.
        if not consumed_vars and not annotations.nocache:
            self._store_console_outputs(
                cell_id,
                cell_provenance,
                "\n".join(combined_stdout),
                "\n".join(combined_stderr),
                prov.input_hashes,
                source_hash=source_hash,
                source=source,
                env_hash=env_hash,
            )

        duration_ms = loop_duration_ms
        raw_displays = final_result.get("displays") if final_result else None
        display_outputs = (
            [d for d in raw_displays if isinstance(d, dict)]
            if isinstance(raw_displays, list)
            else []
        )

        return CellExecutionResult(
            cell_id=cell_id,
            success=True,
            stdout="\n".join(combined_stdout),
            stderr="\n".join(combined_stderr),
            outputs={
                loop.carry: {
                    "content_type": carry_content_type,
                    "file": (
                        f"{loop.carry}"
                        f"{self._LOOP_CONTENT_TYPE_EXT.get(carry_content_type, '.pickle')}"
                    ),
                },
                **extra_outputs,
            },
            display_outputs=display_outputs,
            duration_ms=duration_ms,
            artifact_uri=canonical_uri,
            execution_method="loop",
            mutation_warnings=all_mutation_warnings,
        )

    def _resolve_loop_seed(
        self,
        cell_id: str,
        loop: LoopAnnotation,
    ) -> tuple[bytes, str]:
        """Resolve the iter-0 carry seed as ``(blob_bytes, content_type)``.

        From ``# @loop start_from=<cell>@iter=<k>``, else the latest artifact of
        the upstream cell defining the carry. Raises ``ValueError`` if neither exists.
        """
        artifact_mgr = self.session.get_artifact_manager()

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
            # left over from an older, longer run.
            first = artifact_mgr.get_iteration_artifact(loop.start_from_cell, loop.carry, 0)
            if first is None or _loop_run_token(first) != _loop_run_token(artifact):
                raise ValueError(
                    f"Loop seed start_from={loop.start_from_cell}@iter={loop.start_from_iter} "
                    f"is left over from an older run of cell {loop.start_from_cell}: its "
                    f"latest run did not reach iteration {loop.start_from_iter}. Run that "
                    f"cell through iteration {loop.start_from_iter} first."
                )
            blob = artifact_mgr.artifact_store.read_blob(artifact_id, artifact.version)
            if blob is None:
                raise ValueError(f"Loop seed blob missing for {artifact_id}@v={artifact.version}.")
            return blob, _artifact_content_type(artifact)

        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise ValueError(f"Loop cell {cell_id!r} not found in notebook state.")

        notebook_id = self.session.notebook_state.id
        for upstream_id in cell.upstream_ids:
            upstream_cell = self.session.notebook_state.get_cell(upstream_id)
            if upstream_cell is None or loop.carry not in self.session.wired_variables(
                cell_id, upstream_id
            ):
                continue
            upstream_artifact_id = f"nb_{notebook_id}_cell_{upstream_id}_var_{loop.carry}"
            artifact = artifact_mgr.artifact_store.get_latest_version(upstream_artifact_id)
            if artifact is None or artifact.state not in ("ready", "superseded"):
                continue
            blob = artifact_mgr.load_artifact_data(upstream_artifact_id, artifact.version)
            return blob, _artifact_content_type(artifact)

        raise ValueError(
            f"Cannot resolve loop carry '{loop.carry}': no upstream cell "
            f"defines it and no @loop start_from annotation is set. "
            f"Define `{loop.carry}` in an upstream cell or add "
            f"`# @loop start_from=<cell>@iter=<k>`."
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

    async def _run_batch(
        self,
        cell_specs: list[dict[str, Any]],
        *,
        use_cache: bool,
        batch_timeout_seconds: float,
        cell_timeout_seconds: float = DEFAULT_CELL_TIMEOUT_SECONDS,
        on_cell_event: Callable[[BatchCellResult], Awaitable[None]] | None = None,
    ) -> BatchExecutionResult:
        """Run a sequence of cells in one harness subprocess (see ``execute_batch``).

        Per-cell stdout/stderr attribution is not implemented.
        """
        # As in single-cell: no interpreter, no run.
        venv_python = self.session.venv_python
        if venv_python is None:
            return _refused_batch(cell_specs, _no_interpreter_message(self.session))

        try:
            harness_user = resolve_harness_user()
        except LocalExecutionRefused as exc:
            # The run-all dispatcher doesn't batch on a refusing host (single-cell still serves
            # cache hits), so only a direct batch call gets here.
            return _refused_batch(cell_specs, str(exc))

        batch_tmpdir = Path(
            tempfile.mkdtemp(
                prefix=f"strata_batch_{uuid.uuid4().hex[:8]}_",
                dir=str(self.session.path / ".strata"),
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
        end_reason = "subprocess_died"
        failed_cell_id: str | None = None

        try:
            upstream_inputs = self._batch_resolve_upstream_inputs(cell_specs, batch_tmpdir)
            manifest = {
                "cells": cell_specs,
                "upstream_inputs": upstream_inputs,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            env = identity_env(
                self._harness_env(
                    {
                        "STRATA_BATCH_FRAME_FD": str(frame_w),
                        "STRATA_BATCH_RESP_FD": str(resp_r),
                        "STRATA_BATCH_OUTPUT_DIR": str(batch_tmpdir),
                    }
                ),
                harness_user,
            )
            hand_over(batch_tmpdir, harness_user)

            # New process group so a timeout SIGKILL reaches every descendant. Mirrors
            # _run_harness.
            proc = await asyncio.create_subprocess_exec(
                str(venv_python),
                str(self.harness_path),
                "--batch",
                str(manifest_path),
                env=env,
                pass_fds=(frame_w, resp_r),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.session.path),
                **spawn_kwargs(harness_user),
                **subprocess_kwargs_for_new_group(),
            )

            # Close the parent's copies of harness-side fds so EOF detection works on exit.
            os.close(frame_w)
            os.close(resp_r)
            frame_w = -1  # mark already-closed for finally
            resp_r = -1

            # Drain stdout/stderr concurrently to prevent pipe-buffer deadlock (no per-cell
            # attribution).
            assert proc.stdout is not None
            assert proc.stderr is not None
            stdout_task = asyncio.create_task(_drain_stream(proc.stdout))
            stderr_task = asyncio.create_task(_drain_stream(proc.stderr))

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
                    self._batch_service_loop(
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
                        use_cache=use_cache,
                        on_cell_event=on_cell_event,
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

    async def _batch_service_loop(
        self,
        frame_reader: asyncio.StreamReader,
        send_response: Callable[[dict], None],
        cell_results: dict[str, BatchCellResult],
        batch_tmpdir: Path,
        *,
        executed_sources: dict[str, str],
        use_cache: bool,
        on_cell_event: Callable[[BatchCellResult], Awaitable[None]] | None = None,
        watchdog_state: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        """Read frames and service requests until ``batch_end``.

        Returns ``(end_reason, failed_cell_id)``. ``on_cell_event`` is awaited
        after each ``BatchCellResult`` is recorded.
        """
        active_cell_id: str | None = None
        end_reason = "subprocess_died"
        failed_cell_id: str | None = None

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
                        pass
                    break
                try:
                    line = await asyncio.wait_for(frame_reader.readline(), timeout=remaining)
                except TimeoutError:
                    watchdog_state["timed_out"] = True
                    watchdog_state["timeout_cell_id"] = watchdog_state["active_cell_id"]
                    try:
                        await terminate_subprocess_tree(watchdog_state["proc"])
                    except Exception:
                        pass
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
                # Watchdog window: from cell_start until a completion frame (cache_hit output,
                # cell_error, or persist).
                if watchdog_state is not None and active_cell_id:
                    watchdog_state["active_cell_id"] = active_cell_id
                    watchdog_state["active_started_at"] = time.time()
            elif ftype == "cache_check":
                response = await _serviced(
                    self._batch_service_cache_check(
                        payload.get("cell_id", ""),
                        batch_tmpdir,
                        executed_sources=executed_sources,
                        use_cache=use_cache,
                    )
                )
                # Answered once here instead of at each cache-check return; the cell's ambient
                # client needs it to name what it promotes.
                response["input_uris"] = self._upstream_artifact_uris(payload.get("cell_id", ""))
                send_response(response)
            elif ftype == "persist":
                response = await _serviced(
                    self._batch_service_persist(
                        payload,
                        batch_tmpdir,
                        executed_sources=executed_sources,
                    )
                )
                cell_id_pl = payload["cell_id"]
                if response.get("ok"):
                    # Prefer post-persist display metadata (carries artifact_uri) over the harness's
                    # pre-persist payload, as single-cell does.
                    display_outputs = response.get("display_outputs") or (
                        payload.get("display_outputs") or []
                    )
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
                        stdout=payload.get("stdout", ""),
                        stderr=payload.get("stderr", ""),
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

    def _batch_resolve_upstream_inputs(
        self,
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
            cell = self.session.notebook_state.get_cell(cell_id)
            if cell is None:
                continue
            for upstream_id in cell.upstream_ids:
                if upstream_id in batch_cell_ids:
                    continue
                upstream_cell = self.session.notebook_state.get_cell(upstream_id)
                if upstream_cell is None:
                    continue
                wired = self.session.wired_variables(cell_id, upstream_id)
                for var_name, uri in upstream_cell.artifact_uris.items():
                    if var_name in inputs or var_name not in wired:
                        continue
                    spec_dict = self._materialize_artifact_to_dir(uri, upstream_dir, var_name)
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

    def _materialize_artifact_to_dir(
        self,
        uri: str,
        target_dir: Path,
        var_name: str,
    ) -> dict[str, str] | None:
        """Write an artifact's blob to ``target_dir/<var><ext>``; return the spec or None."""
        artifact_id, version = self._parse_artifact_uri(uri)
        if not artifact_id:
            return None
        artifact_mgr = self.session.get_artifact_manager()
        art = artifact_mgr.artifact_store.get_artifact(artifact_id, version)
        if art is None:
            return None
        blob = artifact_mgr.artifact_store.read_blob(artifact_id, version)
        if blob is None:
            return None
        content_type = _artifact_content_type(art)
        ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".bin")
        file_name = f"{safe_filename_stem(var_name)}{ext}"
        (target_dir / file_name).write_bytes(blob)
        return {"content_type": content_type, "file": file_name}

    def _upstream_artifact_uris(self, cell_id: str) -> dict[str, dict[str, str]]:
        """The inputs this cell reads, as the manifest's ``{variable: {"uri": ...}}``.

        ``strata.promote`` needs inputs named as the cell names them. A batch has
        no per-cell manifest, but in-batch upstreams have persisted by the time a
        cell checks its cache, so the map is built here.
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            return {}
        inputs: dict[str, dict[str, str]] = {}
        for upstream_id in cell.upstream_ids:
            upstream = self.session.notebook_state.get_cell(upstream_id)
            if upstream is None:
                continue
            wired = self.session.wired_variables(cell_id, upstream_id)
            for var_name, uri in upstream.artifact_uris.items():
                if var_name in wired:
                    inputs.setdefault(var_name, {"uri": uri})
        return inputs

    def _materialize_batch_cache_hit(
        self,
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
        artifact_mgr = self.session.get_artifact_manager()
        notebook_id = self.session.notebook_state.id
        cached_outputs: dict[str, dict[str, str]] = {}
        for var_name in consumed_vars:
            canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{var_name}"
            var_prov = derive_subkey(provenance_hash, var_name)
            canonical_art = artifact_mgr.artifact_store.get_latest_version(canonical_id)
            if canonical_art is None or canonical_art.provenance_hash != var_prov:
                return None
            blob = artifact_mgr.artifact_store.read_blob(canonical_art.id, canonical_art.version)
            if blob is None:
                return None
            content_type = _artifact_content_type(canonical_art)
            ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".bin")
            file_name = f"{safe_filename_stem(var_name)}{ext}"
            (cell_output_dir / file_name).write_bytes(blob)
            cached_outputs[var_name] = {
                "content_type": content_type,
                "file": file_name,
            }
            # Populate artifact_uris, as single-cell does, so later batch cells resolve via
            # _collect_input_hashes.
            uri = f"strata://artifact/{canonical_art.id}@v={canonical_art.version}"
            cell.artifact_uris[var_name] = uri
            cell.artifact_uri = uri
        return cached_outputs

    async def _batch_service_cache_check(
        self,
        cell_id: str,
        batch_tmpdir: Path,
        *,
        executed_sources: dict[str, str],
        use_cache: bool,
    ) -> dict[str, Any]:
        """Service a ``cache_check`` request from the batch harness.

        On a hit, writes cached blobs to ``batch_tmpdir/<cell_id>/{var}{ext}`` and
        sets ``cell.artifact_uris`` and ``display_outputs`` as single-cell does.
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            return {"cache_hit": False, "provenance_hash": ""}

        # The partition's source, not the session's: a mid-batch edit must not decide whether
        # this run is a hit.
        source = executed_sources.get(cell_id, cell.source)
        try:
            prov = await self._compute_cell_provenance(cell_id, source)
        except Exception as exc:
            logger.warning("Batch cache_check provenance failed for %s: %s", cell_id, exc)
            return {"cache_hit": False, "provenance_hash": ""}

        provenance_hash = prov.provenance_hash
        # ``# @nocache`` marks an effect the artifact doesn't capture (a write, a POST, a
        # clock read); Run All must not serve it from cache either.
        if not use_cache or prov.annotations.nocache:
            return {"cache_hit": False, "provenance_hash": provenance_hash}

        cell_output_dir = batch_tmpdir / cell_id
        cell_output_dir.mkdir(parents=True, exist_ok=True)
        notebook_id = self.session.notebook_state.id
        artifact_mgr = self.session.get_artifact_manager()
        consumed_vars = (
            self.session.dag.consumed_variables.get(cell_id, set())
            if self.session.dag is not None
            else set()
        )

        # Hydrate cached displays the way single-cell does
        # (session._resolve_cached_display_outputs): rich models when every artifact exists
        # with matching provenance, else []. Their blobs then go to the per-cell tmpdir.
        existing_display_outputs = cell.display_outputs or (
            [cell.display_output] if cell.display_output is not None else []
        )
        cached_display_models = self.session._resolve_cached_display_outputs(
            cell_id, provenance_hash, existing_display_outputs
        )
        cached_displays: list[dict[str, Any]] = []
        for index, cached in enumerate(cached_display_models):
            display_artifact_id = f"nb_{notebook_id}_cell_{cell_id}_var___display__{index}"
            display_art = artifact_mgr.artifact_store.get_latest_version(display_artifact_id)
            if display_art is None:
                continue  # Should not happen: the helper validated existence.
            blob = artifact_mgr.artifact_store.read_blob(display_art.id, display_art.version)
            if blob is None:
                continue
            content_type = _artifact_content_type(display_art)
            ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".bin")
            file_name = f"__display__{index}{ext}"
            (cell_output_dir / file_name).write_bytes(blob)
            meta = cached.model_dump()
            meta["file"] = file_name
            cached_displays.append(meta)

        # A leaf's record is its console (empty when it printed nothing), as in single-cell;
        # without one, or any cached display, there is nothing to serve, so run it.
        cached_console = (
            self.session._resolve_cached_console(cell_id, provenance_hash)
            if not consumed_vars
            else None
        )
        if not consumed_vars and not cached_displays and cached_console is None:
            return {"cache_hit": False, "provenance_hash": provenance_hash}

        cached_outputs = self._materialize_batch_cache_hit(
            cell, cell_id, provenance_hash, consumed_vars, cell_output_dir
        )
        if cached_outputs is None and consumed_vars:
            # Team cache tier, in single-run order: only after a local miss. A pull writes each
            # variable under its canonical local id, so the same probe decides afterwards.
            team_pull = await self._pull_from_team_store(
                cell_id=cell_id,
                provenance_hash=provenance_hash,
                consumed_vars=set(consumed_vars),
                source_hash=prov.source_hash,
                source=source,
                env_hash=prov.env_hash,
                input_versions=self._input_refs(cell_id),
            )
            if team_pull is not None:
                cached_outputs = self._materialize_batch_cache_hit(
                    cell, cell_id, provenance_hash, consumed_vars, cell_output_dir
                )
        if cached_outputs is None:
            return {"cache_hit": False, "provenance_hash": provenance_hash}

        cell.cache_hit = True
        if cached_display_models:
            cell.display_outputs = list(cached_display_models)
            cell.display_output = cached_display_models[-1]

        return {
            "cache_hit": True,
            "provenance_hash": provenance_hash,
            "cached_outputs": cached_outputs,
            "cached_displays": cached_displays,
            "stdout": cached_console[0] if cached_console else "",
            "stderr": cached_console[1] if cached_console else "",
        }

    async def _batch_service_persist(
        self,
        payload: dict[str, Any],
        batch_tmpdir: Path,
        *,
        executed_sources: dict[str, str],
    ) -> dict[str, Any]:
        """Service a ``persist`` request from the batch harness.

        Uses the single-cell chain (``_write_module_export_outputs``,
        ``_store_outputs``, ``_store_display_outputs``).
        """
        cell_id = payload.get("cell_id", "")
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            return {"ok": False, "error": f"cell {cell_id} not found"}

        cell_output_dir = batch_tmpdir / cell_id
        if not cell_output_dir.exists():
            return {"ok": False, "error": f"output dir missing for {cell_id}"}

        # Everything recorded describes the run, so read the source that ran: hashing an edit
        # made mid-batch would file outputs under the wrong key (green on reopen, wrong bytes).
        executed_source = executed_sources.get(cell_id, cell.source)
        try:
            prov = await self._compute_cell_provenance(cell_id, executed_source)
        except Exception as exc:
            return {"ok": False, "error": f"provenance compute failed: {exc}"}

        provenance_hash = prov.provenance_hash
        source_hash = prov.source_hash
        env_hash = prov.env_hash
        input_hashes = self._collect_input_hashes(cell_id)

        # Module export writes synthetic .cell_module.json / .cell_instance.pickle for
        # top-level defs/classes; _store_outputs picks them up. An error string means a
        # consumed def/class can't be exported safely: refuse to persist and signal the
        # harness so the cell errors, as single-cell does.
        try:
            module_export_error = self._write_module_export_outputs(
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

        stored_ok = self._store_outputs(
            cell_id,
            cell_output_dir,
            provenance_hash,
            input_hashes,
            source_hash=source_hash,
            source=executed_source,
            env_hash=env_hash,
        )

        # Carry the post-persist metadata (with artifact_uri) in the ack so the dispatcher
        # broadcasts the URI-bearing version, as single-cell does.
        display_outputs_meta = payload.get("display_outputs") or []
        persisted_displays: list[dict[str, Any]] = []
        if display_outputs_meta:
            try:
                persisted_displays = self._store_display_outputs(
                    cell_id,
                    cell_output_dir,
                    provenance_hash,
                    input_hashes,
                    display_outputs_meta,
                    source_hash=source_hash,
                    source=cell.source,
                    env_hash=env_hash,
                )
            except Exception as exc:
                logger.warning("Batch display persist failed for %s: %s", cell_id, exc)

        if not stored_ok:
            return {"ok": False, "error": "store_outputs returned False"}

        # A leaf's console is its record, as in single-cell. Rerun All stores it too, or the
        # next Run All replays an older run's; only ``# @nocache`` stores none.
        consumed_vars = (
            self.session.dag.consumed_variables.get(cell_id, set())
            if self.session.dag is not None
            else set()
        )
        if not prov.annotations.nocache and not consumed_vars:
            self._store_console_outputs(
                cell_id,
                provenance_hash,
                payload.get("stdout", ""),
                payload.get("stderr", ""),
                input_hashes,
                source_hash=source_hash,
                source=executed_source,
                env_hash=env_hash,
            )

        # Offer it to the team, as a single run does. Inert unless configured, and it
        # swallows its own failures: a shared-cache problem must not fail a finished cell.
        await self._push_to_team_store(cell_id=cell_id)

        # Record provenance + execution as single-cell does.
        try:
            self.session.record_successful_execution_provenance(
                cell_id, provenance_hash, source_hash, env_hash
            )
        except Exception:
            pass

        # So a later cache check finds rich CellOutput objects. The dispatcher does this too,
        # but direct execute_batch calls (tests, REPL) need it here.
        if persisted_displays:
            cell.display_outputs = [CellOutput(**d) for d in persisted_displays]
            cell.display_output = cell.display_outputs[-1]
        # Persist to runtime.json, as single-cell does, or the display is lost on reopen.
        self.session.persist_display_outputs(cell_id, persisted_displays or None)

        uri = cell.artifact_uri or ""
        return {"ok": True, "uri": uri, "display_outputs": persisted_displays}


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


_ARTIFACT_EXT_BY_CONTENT_TYPE: dict[str, str] = {
    "arrow/ipc": ".arrow",
    "json/object": ".json",
    "pickle/object": ".pickle",
    "module/import": ".module.json",
    "module/cell": ".cell_module.json",
    "module/cell-instance": ".cell_instance.pickle",
}


def _refused_result(exc: LocalExecutionRefused) -> dict[str, Any]:
    """A harness result for cell code this host would not start."""
    return {"success": False, "error": str(exc), "stderr": "", "stdout": "", "variables": {}}


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


def _no_interpreter_message(session: NotebookSession) -> str:
    """Why cell code did not run: the session has no interpreter.

    PATH ``python`` is never used instead: its result would be stored under
    this notebook's provenance.
    """
    return session.environment_execution_block_message() or (
        "Notebook environment is not ready: it has no Python interpreter."
    )


async def _drain_stream(stream: asyncio.StreamReader) -> None:
    """Read and discard ``stream`` until EOF, so chatty native code cannot fill the pipe."""
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return


# ---------------------------------------------------------------------------
# Batchability + partitioning
# ---------------------------------------------------------------------------


def is_cell_batchable(executor: CellExecutor, cell: Any) -> bool:
    """Return whether ``cell`` is eligible for run-all batching.

    Delegates to the language executor's ``is_batchable`` (only Python can be).
    Sweep members and sweep consumers never batch: in a shared namespace
    sibling variants overwrite each other and the ``{variant: value}`` dict
    never forms.
    """
    from strata.notebook.dag import SweepProducer
    from strata.notebook.languages import get_language_executor

    # A batch never sees a fetch's bytes; a fetching cell runs single-cell, where the
    # fetch is checked and injected.
    annotations = parse_annotations(cell.source)
    if annotations.fetches or annotations.datasets:
        return False

    dag = executor.session.dag
    if dag is not None:
        modes = executor.session.notebook_state.variant_modes
        if cell.variant_group and modes.get(cell.variant_group) == "sweep":
            return False
        if any(
            isinstance(dag.variable_producer.get(ref), SweepProducer)
            for ref in (*cell.references, *cell.builtin_references)
        ):
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
