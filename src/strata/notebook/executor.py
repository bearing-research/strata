"""Cell executor — materialize a notebook cell.

Each ``execute_cell`` call materializes one cell: ensure upstream inputs
exist, compute provenance, cache-check, execute on miss, persist outputs.

1. **Materialize upstream inputs** — for every upstream variable this cell
   needs, look in the artifact store.  Cache hit → done.  Cache miss →
   recursively ``execute_cell`` on the upstream so it produces the artifact.
2. **Compute provenance** — now that all upstream artifacts exist we can
   build the deterministic hash ``sha256(sorted_input_hashes + source_hash
   + env_hash)``.
3. **Cache check** — if an artifact with matching provenance already exists,
   return immediately (cache hit).
4. **Execute** — spawn the harness subprocess, passing resolved input blobs.
5. **Store outputs** — persist every consumed variable as an artifact.

The cascade planner (``cascade.py``) is a *UI-level* optimisation that
previews which cells will run.  The executor itself is self-contained: you
can call ``execute_cell`` on *any* cell and it will recursively materialise
the full upstream DAG.

Relationship to Strata Core's ``materialize`` SDK
-------------------------------------------------

Strata Core exposes a separate ``materialize(inputs, transform) → artifact``
primitive (``client.materialize`` / ``POST /v1/materialize``) for materializing
*transforms* — server-registered executors like ``scan@v1`` keyed by
``(table_identity, snapshot_id, columns, filters)``.

This module is a parallel pipeline for materializing *cells* — ad-hoc Python
source executed in a subprocess harness, keyed by ``(source_hash, env_hash,
mount_fingerprints, input_hashes)``, with multi-output fan-out (one artifact
per consumed variable via ``derive_subkey``).

The two pipelines deliberately do not share the materialize entry point —
the SDK shape (HTTP, single-output, registered transforms) does not fit
the notebook shape (in-process, multi-output, source-as-transform). They
share the substrate: ``artifact_store.find_by_provenance`` / ``put`` and the
``derive_subkey`` helper in ``notebook.provenance``.
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
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import httpx
import orjson

from strata.artifact_store import TransformSpec as ArtifactTransformSpec
from strata.artifact_store import get_artifact_store
from strata.blob_store import BLOB_STREAM_CHUNK_BYTES
from strata.notebook.annotations import CellAnnotations, LoopAnnotation, parse_annotations
from strata.notebook.env import compute_execution_env_hash, narrow_env_for_provenance
from strata.notebook.immutability import MutationWarning
from strata.notebook.models import CellLanguage, MountSpec, WorkerBackendType
from strata.notebook.module_export import build_module_export_plan
from strata.notebook.mounts import (
    MountCredentials,
    MountFingerprinter,
    MountResolver,
    ResolvedMount,
    parse_mount_uri,
    resolve_cell_mounts,
)
from strata.notebook.provenance import (
    compute_provenance_hash,
    compute_source_hash,
    derive_subkey,
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
from strata.notebook.workers import (
    get_worker_execution_error,
    is_embedded_executor_worker,
    is_http_executor_worker,
    resolve_worker_spec,
    worker_runtime_identity,
    worker_supports_notebook_execution,
    worker_transport,
)
from strata.transforms.build_store import get_build_store
from strata.transforms.signed_urls import generate_build_manifest
from strata.types import EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION

if TYPE_CHECKING:
    from strata.notebook.pool import WarmProcessPool
    from strata.notebook.session import NotebookSession

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


def _resolve_worker_token(worker_spec: Any) -> str | None:
    """Look up the shared-secret token for an HTTP executor worker.

    Two config shapes, in priority order:

    1. ``config.token_env``: name of an env var holding the token. Preferred
       — keeps the secret out of committed notebook.toml files. Common
       pattern: ``token_env = "STRATA_GPU_WORKER_TOKEN"``, the operator
       exports the env var when running the Strata server.
    2. ``config.token``: literal token string. Convenient for local dev,
       never commit a real one.

    Returns ``None`` if neither is set (worker is reached without auth,
    backward-compatible with deployments that don't enforce a token).
    """
    config = getattr(worker_spec, "config", None) or {}
    token_env = str(config.get("token_env") or "").strip()
    if token_env:
        value = os.environ.get(token_env, "").strip()
        if value:
            return value
    literal = str(config.get("token") or "").strip()
    if literal:
        return literal
    return None


def _detect_missing_module(error: str, stderr: str) -> str | None:
    """Parse ModuleNotFoundError to extract package name.

    Returns the PyPI package name to suggest for ``uv add``, or None.
    """
    import re

    combined = f"{error}\n{stderr}"
    # Match full form: ModuleNotFoundError: No module named 'pkg'
    # or short form from harness: No module named 'pkg'
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", combined)
    if not m:
        return None
    module = m.group(1).split(".")[0]  # top-level module
    return _MODULE_TO_PACKAGE.get(module, module)


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


@dataclass(kw_only=True, frozen=True)
class _CellProvenance:
    """Inputs and outputs of the standard cell-provenance computation.

    Every cell-kind path (default, prompt, sql, loop) must produce the
    *same* provenance hash for a given (source, env, inputs, mounts)
    tuple — that hash is what ``compute_staleness`` recomputes on cell
    re-open to decide whether to re-run. The fields here are the
    ingredients plus the resulting hash, kept together so callers can
    re-use intermediate values (env_hash for record_successful_execution,
    annotations for downstream dispatch) without re-deriving them.
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
    provenance_hash: str


@dataclass(kw_only=True)
class CellExecutionResult:
    """Result from executing a cell.

    Attributes:
        cell_id: ID of the executed cell
        success: Whether execution succeeded
        stdout: Captured standard output
        stderr: Captured standard error
        outputs: Dict of output variable name -> metadata
        display_outputs: Ordered visible display output metadata
        display_output: Primary visible display output metadata (legacy last-item shim)
        duration_ms: Execution duration in milliseconds
        error: Error message if execution failed
        cache_hit: Whether execution was skipped due to cache hit
        artifact_uri: URI of stored artifact (if any)
        execution_method: How the cell was executed (cached, warm, cold)
        mutation_warnings: List of mutation warnings (M6)
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
    cache_hit: bool = False
    artifact_uri: str | None = None
    execution_method: str = "cold"  # cold, warm, cached
    mutation_warnings: list[MutationWarning] = field(default_factory=list)
    # Number of retries the prompt-cell validate-and-retry loop
    # consumed. 0 on first-try pass or for non-prompt / non-schema
    # cells. Surfaced so the UI can show "validated after N retries"
    # when non-zero.
    validation_retries: int = 0
    suggest_install: str | None = None  # e.g. "requests"
    remote_worker: str | None = None
    remote_transport: str | None = None
    remote_build_id: str | None = None
    remote_build_state: str | None = None
    remote_error_code: str | None = None

    def __post_init__(self) -> None:
        # Legacy shim: accept either `display_outputs` or `display_output`
        # from callers and make both views consistent on the instance.
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

        The WebSocket execution-finished frame uses a slimmer, different
        shape — see ``_execution_result_payload`` in ws.py.
        """
        payload = asdict(self)
        # success: bool → status: "ready" | "error" (wire rename + transform)
        payload["status"] = "ready" if payload.pop("success") else "error"
        # display_outputs / display_output → displays / display on the wire
        payload["displays"] = payload.pop("display_outputs")
        payload["display"] = payload.pop("display_output")
        # Optional metadata fields are omitted from the wire when falsy.
        for opt_field in (
            "validation_retries",
            "suggest_install",
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


@dataclass(kw_only=True)
class BatchExecutionResult:
    """Outcome of one ``CellExecutor.execute_batch`` invocation.

    The dispatcher (``ws._execute_run_all``) uses this to decide which
    cells still need to run via single-cell mode after the batch
    completes or terminates early.
    """

    cell_results: list[BatchCellResult]
    completed: bool  # True if batch_end with reason=complete
    failed_cell_id: str | None = None
    end_reason: str = "complete"  # "complete" | "cell_error" | "persist_failed" | "subprocess_died"


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


class CellExecutor:
    """Materialize notebook cells (cache-or-build per cell).

    Each ``execute_cell`` call ensures all upstream artifacts exist
    (recursively materialising them on cache miss), then checks the
    cache for this cell, and finally executes + stores on a miss.

    This is the notebook-side parallel to Strata Core's transform
    ``materialize`` SDK — see the module docstring for the seam.

    Attributes:
        session: NotebookSession for the notebook
        harness_path: Path to the harness script
        pool: Optional WarmProcessPool for fast execution (M6)
    """

    def __init__(
        self,
        session: NotebookSession,
        pool: WarmProcessPool | None = None,
        mount_credentials: MountCredentials | None = None,
    ):
        self.session = session
        self.harness_path = Path(__file__).parent / "harness.py"
        self.pool = pool
        # Guard against DAG cycles during recursive materialisation.
        # Per-instance is correct: cycles are only meaningful within a single
        # execute_cell() recursive tree. Each top-level call creates a fresh
        # CellExecutor, so the guard resets between independent executions.
        self._materializing: set[str] = set()
        self._mount_resolver = MountResolver(
            cache_dir=session.path / ".strata" / "mount_cache",
            credentials=mount_credentials,
        )
        # Optional callback fired after every loop iteration completes. Set
        # by the WS handler so the frontend can update its progress badge
        # in real time; unset for non-streaming callers (REST, CLI).
        self.on_iteration_complete: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float = 30,
        *,
        skip_upstream_materialization: bool = False,
    ) -> CellExecutionResult:
        """Materialise a cell: ensure inputs → cache check → execute → store.

        Single public entry point for cell materialization. See the module
        docstring for how this relates to Strata Core's transform
        ``materialize`` SDK (they are deliberately separate pipelines that
        share the artifact-store substrate).

        ``skip_upstream_materialization`` is the seam used by batch
        continuation (``execute_batch`` → fall-back loop after a batched
        cell errors). When ``True``, the cell trusts whatever artifacts
        are already persisted and does not recursively re-execute its
        upstreams. Direct-dependency-on-failed-upstream cells will hit
        clean missing-artifact errors rather than re-running the cell
        that already failed once. Cache lookup for the target cell is
        unchanged — that's what differentiates this from
        ``execute_cell_force``.
        """
        return await self._execute_cell(
            cell_id,
            source,
            timeout_seconds,
            materialize_upstreams=not skip_upstream_materialization,
            use_cache=True,
        )

    async def execute_cell_force(
        self, cell_id: str, source: str, timeout_seconds: float = 30
    ) -> CellExecutionResult:
        """Execute a cell using the currently available upstream artifacts only.

        This bypasses recursive upstream materialization and skips the target
        cell cache lookup so "Run this only" performs a real execution against
        whatever inputs are currently present.
        """
        return await self._execute_cell(
            cell_id,
            source,
            timeout_seconds,
            materialize_upstreams=False,
            use_cache=False,
        )

    async def execute_cell_rerun(
        self, cell_id: str, source: str, timeout_seconds: float = 30
    ) -> CellExecutionResult:
        """Force re-execution of a cell while still materializing upstreams.

        Upstreams are resolved through the normal cache (stale upstreams are
        re-materialized from the artifact store on cache hit); only the target
        cell's cache is bypassed so a real execution happens.
        """
        return await self._execute_cell(
            cell_id,
            source,
            timeout_seconds,
            materialize_upstreams=True,
            use_cache=False,
        )

    async def execute_batch(
        self,
        cell_specs: list[dict[str, Any]],
        *,
        use_cache: bool = True,
        batch_timeout_seconds: float = 600.0,
    ) -> BatchExecutionResult:
        """Execute a sequence of cells in one harness subprocess.

        ``cell_specs`` is the list of cells in notebook order, each a dict
        with keys ``cell_id``, ``source``, ``env``, ``mount_manifest``
        (already resolved). Caller (``ws._execute_run_all`` in PR-b3) is
        responsible for partitioning the notebook into batchable runs;
        this method just executes whatever it's given.

        Returns a ``BatchExecutionResult`` so the caller can decide which
        cells still need to run via single-cell mode (workers, post-batch-
        failure continuation, etc.). See issue #26 for the full design.

        ``use_cache=False`` bypasses the per-cell cache check (rerun-all
        semantics).
        """
        return await self._run_batch(
            cell_specs,
            use_cache=use_cache,
            batch_timeout_seconds=batch_timeout_seconds,
        )

    async def _execute_cell(
        self,
        cell_id: str,
        source: str,
        timeout_seconds: float,
        *,
        materialize_upstreams: bool,
        use_cache: bool,
    ) -> CellExecutionResult:
        """Shared execution entrypoint with explicit cache/materialization policy."""
        annotations = parse_annotations(source)
        timeout_seconds = self._resolve_effective_timeout(
            cell_id,
            timeout_seconds,
            annotations.timeout,
        )

        # Loop-cell dispatcher: only if the annotation is well-formed enough
        # to run. Validation diagnostics surface malformed loops separately.
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
        runtime_env = dict(cell.env) if cell is not None else {}
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

        Every cell-kind path (default, prompt, sql, loop) feeds its
        artifact-store writes the *same* hash so ``compute_staleness``
        recognises the cell as ready on re-open. Drifting any of the
        ingredients (runtime_env, worker identity, mount fingerprints,
        env-key narrowing) makes the cell appear stale forever.

        Optional precomputed arguments let callers reuse work — the
        non-loop ``_materialize_cell`` path resolves mounts before the cache
        check, the loop path resolves them once per iteration, so it
        passes them in instead of paying the cost again.
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
        provenance_env = narrow_env_for_provenance(source, runtime_env, declared_env_keys)
        env_hash = compute_execution_env_hash(
            self.session.path,
            provenance_env,
            runtime_identity=runtime_identity,
        )
        input_hashes = self._collect_input_hashes(cell_id)
        provenance_hash = compute_provenance_hash(
            input_hashes + mount_fingerprints,
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
            provenance_hash=provenance_hash,
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
        remote_metadata: dict[str, str] = {}
        try:
            cell = self.session.notebook_state.get_cell(cell_id)
            if cell is not None:
                cell.cache_hit = False

            # Prompt cells use a dedicated executor (LLM call, no subprocess)
            if cell is not None and cell.language == CellLanguage.PROMPT:
                return await self._execute_prompt_cell(
                    cell_id,
                    source,
                    start_time,
                    materialize_upstreams=materialize_upstreams,
                    use_cache=use_cache,
                )

            # SQL cells use a dedicated executor (ADBC query, no subprocess).
            if cell is not None and cell.language == CellLanguage.SQL:
                return await self._execute_sql_cell(
                    cell_id,
                    source,
                    start_time,
                    materialize_upstreams=materialize_upstreams,
                    use_cache=use_cache,
                )

            # Markdown cells are pure prose — no execution, no subprocess,
            # no provenance chain. Return success with no display outputs:
            # the frontend already renders the source in-place via the
            # cell's preview view, so emitting it again as a display
            # output would just duplicate the same content in the output
            # panel below the editor.
            if cell is not None and cell.language == CellLanguage.MARKDOWN:
                # ``start_time`` is wall-clock (``time.time()``), so subtract
                # in the same clock — mixing ``monotonic()`` here produces a
                # ~1.7e12 ms negative because the two clocks have different
                # epochs.
                duration_ms = (time.time() - start_time) * 1000
                return CellExecutionResult(
                    cell_id=cell_id,
                    success=True,
                    duration_ms=duration_ms,
                    execution_method="cached",
                    cache_hit=True,
                )

            # ① Materialise every upstream cell whose artifact is missing.
            #   This is the recursive ``materialize`` call — each upstream
            #   that is a cache miss will itself execute its own upstreams.
            if materialize_upstreams:
                await self._materialize_upstreams(cell_id)

            # ① Compute the standard provenance triplet (annotations,
            # mounts, env_hash, input_hashes, source_hash → provenance_hash).
            # Every cell-kind path uses the same helper so the hash stays
            # consistent with ``compute_staleness`` on re-open.
            prov = await self._compute_cell_provenance(cell_id, source)
            source_hash = prov.source_hash
            runtime_env = prov.runtime_env
            effective_worker = prov.effective_worker
            env_hash = prov.env_hash
            input_hashes = prov.input_hashes
            mount_specs = prov.mount_specs
            mount_fingerprints = prov.mount_fingerprints
            provenance_hash = prov.provenance_hash

            # RW mounts make the cell non-cacheable (side effects).
            if prov.has_rw_mount:
                use_cache = False

            worker_spec = resolve_worker_spec(
                self.session.notebook_state,
                effective_worker,
            )
            remote_metadata = self._remote_execution_metadata(worker_spec)

            logger.info(
                "execute_cell %s: source_hash=%s env_hash=%s "
                "input_hashes=%s mount_fps=%s provenance=%s",
                cell_id,
                source_hash[:12],
                env_hash[:12],
                [h[:12] for h in input_hashes],
                [fp[:20] for fp in mount_fingerprints],
                provenance_hash[:12],
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
            cached_display_outputs = (
                self.session._resolve_cached_display_outputs(
                    cell_id,
                    provenance_hash,
                    current_display_outputs,
                )
                if cell is not None
                else []
            )
            if use_cache:
                if consumed_vars:
                    first_var = sorted(consumed_vars)[0]
                    var_prov = derive_subkey(provenance_hash, first_var)
                    cached_artifact = artifact_mgr.find_cached(var_prov)
                else:
                    cached_artifact = artifact_mgr.find_cached(provenance_hash)

            # Validate cache hit: every consumed variable must have a
            # canonical artifact whose provenance matches.  The global
            # find_by_provenance can return artifacts from old notebook
            # sessions (same SQLite DB, different notebook_id).  We must
            # verify the LOCAL canonical artifact exists AND has the
            # expected provenance hash — not just that it exists.
            notebook_id = self.session.notebook_state.id
            if use_cache and cached_artifact is not None and consumed_vars:
                for var_name in consumed_vars:
                    canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{var_name}"
                    var_prov = derive_subkey(provenance_hash, var_name)
                    canonical_art = artifact_mgr.artifact_store.get_latest_version(
                        canonical_id,
                    )
                    if canonical_art is None or canonical_art.provenance_hash != var_prov:
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
                        break

            logger.info(
                "execute_cell %s: consumed_vars=%s use_cache=%s cache_hit=%s",
                cell_id,
                consumed_vars,
                use_cache,
                cached_artifact is not None or bool(cached_display_outputs),
            )

            if cached_artifact is not None or (not consumed_vars and cached_display_outputs):
                if remote_metadata.get("remote_transport") == "signed":
                    remote_metadata.setdefault("remote_build_state", "ready")
                # Cache hit — update cell state and return.
                duration_ms = (time.time() - start_time) * 1000
                if cell:
                    cell.cache_hit = True
                    cell.display_outputs = list(cached_display_outputs)
                    cell.display_output = (
                        cached_display_outputs[-1] if cached_display_outputs else None
                    )
                    # Populate per-variable URIs from canonical artifacts
                    for var_name in consumed_vars:
                        canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{var_name}"
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
                ).apply_remote_metadata(**remote_metadata)
                self.session.record_successful_execution_provenance(
                    cell_id,
                    provenance_hash,
                    source_hash,
                    env_hash,
                )
                self.session.apply_execution_result_metadata(cell_id, cached_result)
                return cached_result

            # ④ Cache miss — execute the cell.
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

                # Load upstream blobs into output_dir for the harness.
                # Force execution may intentionally skip upstream materialization,
                # so missing inputs are allowed to surface at execution time.
                input_specs = self._load_input_blobs(cell_id, output_dir)

                venv_path = self.session.venv_python or Path("python")

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

                # ⑤ Store output artifacts for consumed variables.
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
                        env_hash=env_hash,
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

                    if exec_result.success:
                        exec_result.display_outputs = self._store_display_outputs(
                            cell_id,
                            result_output_dir,
                            provenance_hash,
                            input_hashes,
                            exec_result.display_outputs,
                            source_hash=source_hash,
                            env_hash=env_hash,
                        )
                        exec_result.display_output = (
                            exec_result.display_outputs[-1] if exec_result.display_outputs else None
                        )

                    # ⑥ Sync-back read-write mounts after successful execution.
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
                error=f"Cell execution timed out after {timeout_seconds}s",
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

    async def _dispatch_execution(
        self,
        worker_spec: Any,
        source: str,
        input_specs: dict[str, dict[str, str]],
        mount_specs: list[MountSpec],
        output_dir: Path,
        venv_path: Path,
        runtime_env: dict[str, str],
        timeout_seconds: float,
        remote_build_id: str | None = None,
        mutation_defines: list[str] | None = None,
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
            )

        if is_http_executor_worker(worker_spec):
            return await self._dispatch_http_executor(
                worker_spec,
                source,
                input_specs,
                mount_specs,
                output_dir,
                runtime_env,
                timeout_seconds,
                remote_build_id=remote_build_id,
            )

        raise RuntimeError(f"Unsupported worker backend: {worker_spec.backend.value}")

    async def _dispatch_local(
        self,
        source: str,
        input_specs: dict[str, dict[str, str]],
        mount_specs: list[MountSpec],
        output_dir: Path,
        venv_path: Path,
        runtime_env: dict[str, str],
        timeout_seconds: float,
        mutation_defines: list[str] | None = None,
    ) -> tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]:
        """Run the existing direct local execution path."""
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
        )

        if self.pool is not None:
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
        venv_path: Path,
        runtime_env: dict[str, str],
        timeout_seconds: float,
        mutation_defines: list[str] | None = None,
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
        )
        result = await self._run_harness(manifest_path, venv_path, timeout_seconds)

        bundle_path = output_dir / "notebook-output-bundle.tar"
        pack_notebook_output_bundle(bundle_path, result, output_dir)

        unpacked_dir = output_dir / "_executor_result"
        unpacked_result = unpack_notebook_output_bundle(bundle_path, unpacked_dir)
        return unpacked_result, unpacked_dir, "executor", resolved_mounts

    async def _dispatch_http_executor(
        self,
        worker_spec: Any,
        source: str,
        input_specs: dict[str, dict[str, str]],
        mount_specs: list[MountSpec],
        output_dir: Path,
        runtime_env: dict[str, str],
        timeout_seconds: float,
        remote_build_id: str | None = None,
    ) -> tuple[dict[str, Any], Path, str, dict[str, ResolvedMount]]:
        """Run a cell through an external notebook executor over HTTP."""
        for mount in mount_specs:
            if mount.uri.startswith("file://"):
                raise RuntimeError(
                    f"Remote executor workers do not support file:// mounts: '{mount.name}'"
                )

        executor_url = str(worker_spec.config.get("url", "")).strip()
        if not executor_url:
            raise RuntimeError(f"Executor worker '{worker_spec.name}' is missing config.url")

        worker_token = _resolve_worker_token(worker_spec)
        transport = str(worker_spec.config.get("transport", "direct")).strip().lower()
        if transport in {"signed", "manifest", "build"}:
            return await self._dispatch_http_executor_with_manifest(
                worker_spec,
                source,
                input_specs,
                mount_specs,
                output_dir,
                runtime_env,
                timeout_seconds,
                build_id=remote_build_id,
            )

        metadata = {
            "protocol_version": EXECUTOR_PROTOCOL_VERSION,
            "build_id": f"notebook-{uuid.uuid4().hex[:12]}",
            "tenant": None,
            "principal": None,
            # ``sorted(input_specs)`` round-trips through ``str(list)`` —
            # the legacy inline form used an f-string which called
            # ``__str__`` on the list. Preserve the same byte format so
            # any cached transport hashes keyed off this value stay valid.
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
            "inputs": [
                {
                    "name": var_name,
                    "format": str(spec.get("content_type", "pickle/object")),
                    "uri": None,
                    "byte_size": (output_dir / str(spec["file"])).stat().st_size,
                }
                for var_name, spec in sorted(input_specs.items())
            ],
        }

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
        for spec in input_specs.values():
            file_name = str(spec["file"])
            input_path = output_dir / file_name
            handle = open(input_path, "rb")
            input_file_handles.append(handle)
            files.append(
                (
                    file_name,
                    (
                        file_name,
                        handle,
                        "application/octet-stream",
                    ),
                )
            )

        timeout = max(timeout_seconds + 5.0, 30.0)
        headers = {
            EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION,
        }
        if worker_token:
            headers["Authorization"] = f"Bearer {worker_token}"
        bundle_path = output_dir / "notebook-output-bundle.tar"
        try:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        executor_url,
                        files=files,
                        headers=headers,
                    ) as response:
                        if response.status_code == 408:
                            raise RemoteExecutionError(
                                f"Cell execution timed out after {timeout_seconds}s",
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
                        notebook_protocol = response.headers.get(
                            "X-Strata-Notebook-Executor-Protocol"
                        )
                        if (
                            notebook_protocol
                            and notebook_protocol != NOTEBOOK_EXECUTOR_PROTOCOL_VERSION
                        ):
                            raise RemoteExecutionError(
                                f"Remote executor '{worker_spec.name}' returned unsupported "
                                f"notebook protocol version {notebook_protocol!r}",
                                remote_error_code="PROTOCOL_ERROR",
                            )

                        with open(bundle_path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
            except httpx.TimeoutException as exc:
                raise RemoteExecutionError(
                    f"Cell execution timed out after {timeout_seconds}s",
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
        build_store = get_build_store(artifact_dir / "artifacts.sqlite")
        if artifact_store is None or build_store is None:
            raise RuntimeError("Build store is not initialized")

        executor_url = str(worker_spec.config.get("url", "")).strip()
        if not executor_url:
            raise RuntimeError(f"Executor worker '{worker_spec.name}' is missing config.url")

        base_url = str(worker_spec.config.get("strata_url", "")).strip() or state.config.server_url
        principal = get_principal()
        tenant_id = principal.tenant if principal is not None else None
        principal_id = principal.id if principal is not None else None

        build_id = build_id or f"nbbuild-{uuid.uuid4().hex[:12]}"
        artifact_id = f"nb_remote_{self.session.notebook_state.id}_{build_id}"
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
                    "params": build_params,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        transform_spec = ArtifactTransformSpec(
            executor=NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            params=build_params,
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
                params=build_params,
            )
            build_store.start_build(build_id)

            manifest = generate_build_manifest(
                base_url=base_url,
                build_id=build_id,
                metadata={
                    "build_id": build_id,
                    "artifact_id": artifact_id,
                    "version": artifact_version,
                    "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
                    "params": build_params,
                },
                input_artifacts=input_artifacts,
                max_output_bytes=state.config.max_transform_output_bytes,
                url_expiry_seconds=state.config.signed_url_expiry_seconds,
            ).to_dict()

            manifest_execute_url = self._manifest_execute_url(executor_url)
            worker_token = _resolve_worker_token(worker_spec)
            headers = {"Authorization": f"Bearer {worker_token}"} if worker_token else None
            async with httpx.AsyncClient(timeout=max(timeout_seconds + 10.0, 30.0)) as client:
                response = await client.post(manifest_execute_url, json=manifest, headers=headers)
        except asyncio.CancelledError:
            _mark_failed("Notebook manifest execution cancelled", "CANCELLED")
            raise
        except httpx.TimeoutException as exc:
            _mark_failed("Notebook manifest execution timed out", "TIMEOUT")
            raise RemoteExecutionError(
                f"Cell execution timed out after {timeout_seconds}s",
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
                    f"Cell execution timed out after {timeout_seconds}s",
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
        input_specs: dict[str, dict[str, str]],
        output_dir: Path,
        tenant_id: str | None,
        principal_id: str | None,
    ) -> tuple[dict[str, dict[str, str]], list[tuple[str, int]]]:
        """Stage notebook upstream blobs into the service artifact store for signed transport."""
        staged_specs: dict[str, dict[str, str]] = {}
        input_artifacts: list[tuple[str, int]] = []

        for var_name, spec in sorted(input_specs.items()):
            file_name = str(spec.get("file", "")).strip()
            if not file_name:
                raise RuntimeError(
                    "Signed notebook executor transport is missing a local "
                    f"input file for {var_name}"
                )
            input_path = output_dir / file_name
            if not input_path.exists():
                raise RuntimeError(
                    f"Signed notebook executor transport could not find input file {file_name!r}"
                )

            blob_data = input_path.read_bytes()
            content_type = str(spec.get("content_type", "pickle/object"))
            source_uri = str(spec.get("uri", "")).strip()
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
            )
            if finalized is None:
                raise RuntimeError(
                    f"Failed to finalize staged notebook input artifact for {var_name}"
                )

            staged_uri = f"strata://artifact/{finalized.id}@v={finalized.version}"
            staged_specs[var_name] = {
                "uri": staged_uri,
                "content_type": content_type,
            }
            input_artifacts.append((finalized.id, finalized.version))

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
        """Resolve all mount declarations for a cell.

        Priority: annotation > cell-meta > notebook-level.
        """
        cell = self.session.notebook_state.get_cell(cell_id)

        # Cell-level mounts already include notebook defaults from parser.py.
        cell_mounts_spec = cell.mounts if cell else []

        # Annotation mounts (from # @mount in source)
        annotations = parse_annotations(source)
        annotation_mounts = annotations.mounts

        # Merge with priority
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
        mount_fingerprints: list[str] = []
        has_rw_mount = False
        credentials = self._mount_resolver.credentials
        for mount in sorted(mount_specs, key=lambda item: item.name):
            scheme, _ = parse_mount_uri(mount.uri)
            storage_options = {**credentials.get(scheme, {}), **mount.options} or None
            fingerprint = await MountFingerprinter.fingerprint_mount(
                mount, storage_options=storage_options
            )
            if fingerprint is None:
                has_rw_mount = True
            else:
                mount_fingerprints.append(f"{mount.name}:{fingerprint}")
        return mount_fingerprints, has_rw_mount

    async def _prepare_mounts(
        self,
        mount_specs: list[MountSpec],
    ) -> dict[str, ResolvedMount]:
        """Prepare local mount materializations for local execution paths."""
        if not mount_specs:
            return {}
        return await self._mount_resolver.prepare_mounts(mount_specs)

    def _write_manifest(
        self,
        source: str,
        input_specs: dict[str, dict[str, str]],
        output_dir: Path,
        runtime_env: dict[str, str],
        resolved_mounts: dict[str, ResolvedMount],
        mutation_defines: list[str] | None = None,
        loop_config: dict[str, Any] | None = None,
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
            "env": runtime_env,
            "mutation_defines": list(mutation_defines or []),
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
        from strata.notebook.prompt_executor import execute_prompt_cell
        from strata.notebook.routes import _get_llm_config

        if materialize_upstreams:
            await self._materialize_upstreams(cell_id)

        llm_config = _get_llm_config(self.session)
        if llm_config is None:
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                outputs={},
                stdout="",
                stderr="",
                error=(
                    "LLM not configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                    "or STRATA_AI_API_KEY in the Runtime Panel env vars."
                ),
                cache_hit=False,
                duration_ms=int((time.time() - start_time) * 1000),
                execution_method="llm",
            )

        # Compute and record the standard provenance hash so the staleness
        # checker can recognise this cell as "ready" on subsequent recomputes.
        # The prompt executor uses its own provenance (rendered text + model)
        # for artifact caching, which is correct for dedup but invisible to
        # compute_staleness. Recording the standard hash lets the
        # "can_preserve_ready" path match.
        prov = await self._compute_cell_provenance(cell_id, source)
        standard_provenance = prov.provenance_hash

        result_dict = await execute_prompt_cell(
            self.session,
            cell_id,
            source,
            llm_config,
            use_cache=use_cache,
        )

        if result_dict.get("success"):
            cell = self.session.notebook_state.get_cell(cell_id)
            if cell is not None:
                cell.last_provenance_hash = standard_provenance

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
            await self._materialize_upstreams(cell_id)

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

            # SQL cells need to participate in the notebook's standard
            # staleness machinery so a recompute or reopen correctly
            # marks them READY. ``compute_staleness`` always recomputes
            # the *generic* provenance triplet (input hashes + source
            # hash + env hash) and compares against
            # ``cell.last_provenance_hash``; the SQL-specific hash the
            # cell_executor folds is invisible to that path. Mirror
            # ``_execute_prompt_cell`` and persist the generic triplet
            # via ``record_successful_execution_provenance``.
            prov = await self._compute_cell_provenance(cell_id, source)
            self.session.record_successful_execution_provenance(
                cell_id,
                prov.provenance_hash,
                prov.source_hash,
                prov.env_hash,
            )

        # Account for the duration the wrapper itself adds (materialize
        # upstreams, dispatch overhead). ``execute_sql_cell`` measures
        # only its own work; the caller's ``start_time`` is the right
        # reference for the cell's total duration.
        duration_ms = (time.time() - start_time) * 1000

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
            duration_ms=int(duration_ms),
            execution_method=result_dict.get("execution_method", "sql"),
            artifact_uri=result_dict.get("artifact_uri"),
            mutation_warnings=result_dict.get("mutation_warnings", []),
        )

    # ------------------------------------------------------------------
    # ① Materialise upstream cells
    # ------------------------------------------------------------------

    async def _materialize_upstreams(self, cell_id: str) -> None:
        """Ensure every upstream variable has a *current* artifact.

        Always recursively calls ``execute_cell`` on each upstream.
        This is correct because ``execute_cell`` has its own provenance-
        based cache check — an unchanged upstream returns instantly as a
        cache hit, while a stale upstream (e.g. source edited) re-executes.

        The previous approach only checked artifact *existence*, which
        missed the case where an upstream's source changed but its old
        artifact still existed in the store.
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None or not cell.upstream_ids:
            return

        # We only need to execute a given upstream once even if it
        # produces multiple variables we reference.
        executed_upstreams: set[str] = set()

        for upstream_id in cell.upstream_ids:
            if upstream_id in executed_upstreams:
                continue

            upstream_cell = self.session.notebook_state.get_cell(upstream_id)
            if upstream_cell is None:
                continue

            # Always materialise the upstream. execute_cell() will
            # return immediately on cache hit (provenance matches),
            # or re-execute if the upstream is stale.
            result = await self.execute_cell(
                upstream_id,
                upstream_cell.source,
            )
            if not result.success:
                raise RuntimeError(
                    f"Failed to materialise upstream cell {upstream_id}: {result.error}"
                )
            executed_upstreams.add(upstream_id)

    # ------------------------------------------------------------------
    # ② Collect input hashes (upstream artifacts are guaranteed to exist)
    # ------------------------------------------------------------------

    def _collect_input_hashes(self, cell_id: str) -> list[str]:
        """Read provenance hashes from upstream artifacts.

        Called *after* ``_materialize_upstreams`` so every upstream
        artifact is populated. Uses per-variable ``artifact_uris`` dict
        when available, falling back to the legacy ``artifact_uri`` field.
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None or not cell.upstream_ids:
            return []

        artifact_mgr = self.session.get_artifact_manager()
        hashes: list[str] = []

        for upstream_id in cell.upstream_ids:
            upstream_cell = self.session.notebook_state.get_cell(upstream_id)
            if upstream_cell is None:
                continue

            # Collect URIs: prefer per-variable dict, fall back to single URI
            uris = list(upstream_cell.artifact_uris.values())
            if not uris and upstream_cell.artifact_uri:
                uris = [upstream_cell.artifact_uri]

            for uri in sorted(uris):  # sorted for deterministic ordering
                try:
                    parts = uri.split("/")
                    artifact_id = parts[-1].split("@")[0]
                    version = int(parts[-1].split("@v=")[1])
                    artifact = artifact_mgr.artifact_store.get_artifact(
                        artifact_id,
                        version,
                    )
                    if artifact:
                        hashes.append(artifact.provenance_hash)
                except (IndexError, ValueError):
                    pass

        return hashes

    # ------------------------------------------------------------------
    # ④-a Load input blobs (guaranteed to exist after step ①)
    # ------------------------------------------------------------------

    def _load_input_blobs(
        self,
        cell_id: str,
        output_dir: Path,
    ) -> dict[str, dict[str, str]]:
        """Load upstream variable blobs from the artifact store.

        All upstream artifacts are guaranteed to exist because
        ``_materialize_upstreams`` has already run.  This method simply
        reads blobs and writes them to *output_dir* for the harness.
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            return {}

        artifact_mgr = self.session.get_artifact_manager()
        notebook_id = self.session.notebook_state.id
        input_specs: dict[str, dict[str, str]] = {}

        for upstream_id in cell.upstream_ids:
            upstream_cell = self.session.notebook_state.get_cell(upstream_id)
            if upstream_cell is None:
                continue

            referenced_vars = [v for v in cell.references if v in upstream_cell.defines]

            for var_name in referenced_vars:
                artifact_id = f"nb_{notebook_id}_cell_{upstream_id}_var_{var_name}"
                try:
                    artifact = artifact_mgr.artifact_store.get_latest_version(
                        artifact_id,
                    )
                    if artifact is None:
                        # Should not happen after _materialize_upstreams,
                        # but guard defensively.
                        logger.error(
                            "Artifact %s still missing after upstream "
                            "materialisation — skipping variable '%s'.",
                            artifact_id,
                            var_name,
                        )
                        continue

                    blob_data = artifact_mgr.load_artifact_data(
                        artifact_id,
                        artifact.version,
                    )

                    # Determine content type.
                    content_type = "pickle/object"
                    if artifact.transform_spec:
                        try:
                            spec = json.loads(artifact.transform_spec)
                            ct = spec.get("params", {}).get("content_type")
                            if ct:
                                content_type = ct
                        except (ValueError, KeyError):
                            pass

                    ext_map = {
                        "arrow/ipc": ".arrow",
                        "json/object": ".json",
                        "pickle/object": ".pickle",
                        "module/import": ".module.json",
                        "module/cell": ".cell_module.json",
                        "module/cell-instance": ".cell_instance.pickle",
                    }
                    ext = ext_map.get(content_type, ".pickle")
                    input_file = output_dir / f"{var_name}{ext}"
                    with open(input_file, "wb") as f:
                        f.write(blob_data)

                    input_specs[var_name] = {
                        "content_type": content_type,
                        "file": f"{var_name}{ext}",
                        "uri": (f"strata://artifact/{artifact.id}@v={artifact.version}"),
                    }
                    logger.info(
                        "Loaded input %s from artifact store (%s@v=%d, %d bytes, %s)",
                        var_name,
                        artifact_id,
                        artifact.version,
                        len(blob_data),
                        content_type,
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
        env_hash: str = "",
    ) -> bool:
        """Persist consumed output variables as artifacts.

        Returns True if every consumed variable was stored, False otherwise.
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None or self.session.dag is None:
            return True

        artifact_mgr = self.session.get_artifact_manager()
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

        if not consumed_vars:
            return True

        all_stored = True

        for var_name in consumed_vars:
            found = False
            for ext in [
                ".arrow",
                ".cell_module.json",
                ".cell_instance.pickle",
                ".module.json",
                ".json",
                ".pickle",
            ]:
                output_file = output_dir / f"{var_name}{ext}"
                if output_file.exists():
                    found = True
                    try:
                        with open(output_file, "rb") as f:
                            blob_data = f.read()

                        content_type_map = {
                            ".arrow": "arrow/ipc",
                            ".json": "json/object",
                            ".pickle": "pickle/object",
                            ".module.json": "module/import",
                            ".cell_module.json": "module/cell",
                            ".cell_instance.pickle": "module/cell-instance",
                        }
                        content_type = content_type_map.get(ext, "pickle/object")

                        var_provenance = derive_subkey(provenance_hash, var_name)

                        artifact_version = artifact_mgr.store_cell_output(
                            cell_id=cell_id,
                            variable_name=var_name,
                            blob_data=blob_data,
                            content_type=content_type,
                            provenance_hash=var_provenance,
                            input_versions={h: h for h in input_hashes},
                            source_hash=source_hash,
                            env_hash=env_hash,
                        )
                        uri = (
                            f"strata://artifact/{artifact_version.id}@v={artifact_version.version}"
                        )
                        cell.artifact_uris[var_name] = uri
                        cell.artifact_uri = uri  # backward compat
                        logger.info(
                            "Stored output %s for cell %s as %s@v=%d (%d bytes, %s)",
                            var_name,
                            cell_id,
                            artifact_version.id,
                            artifact_version.version,
                            len(blob_data),
                            content_type,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to store output %s for cell %s",
                            var_name,
                            cell_id,
                        )
                        all_stored = False
                    break

            if not found:
                logger.warning(
                    "_store_outputs %s: no output file for consumed var %s "
                    "("
                    "looked for %s.arrow/.json/.pickle/"
                    ".cell_module.json/.cell_instance.pickle in %s"
                    ")",
                    cell_id,
                    var_name,
                    var_name,
                    output_dir,
                )
                all_stored = False

        return all_stored

    def _store_display_outputs(
        self,
        cell_id: str,
        output_dir: Path,
        provenance_hash: str,
        input_hashes: list[str],
        display_outputs: list[dict[str, Any]] | None,
        *,
        source_hash: str = "",
        env_hash: str = "",
    ) -> list[dict[str, Any]]:
        """Persist ordered cell display outputs as canonical artifacts."""
        if not display_outputs:
            return []

        artifact_mgr = self.session.get_artifact_manager()
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
                input_versions={h: h for h in input_hashes},
                source_hash=source_hash,
                env_hash=env_hash,
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
        exported safely under the current V1 rules.
        """
        if self.session.dag is None:
            return None

        consumed_vars = self.session.dag.consumed_variables.get(cell_id, set())
        if not consumed_vars:
            return None

        export_plan = build_module_export_plan(source)
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

        # Constants alone shouldn't trigger module-export — a cell
        # whose only consumed output is ``x = 1`` should serialize ``x``
        # as a regular int. We only route constants through module-
        # export when the cell *also* exports a def/class; that's when
        # the synthetic module is being built anyway and putting the
        # constant on it keeps the name available alongside the defs.
        code_exports = [
            name
            for name in exportable_vars
            if export_plan.exported_symbols[name].kind in ("function", "async function", "class")
        ]
        if not code_exports:
            return None

        source_hash = compute_source_hash(source)
        notebook_id = self.session.notebook_state.id

        for var_name in exportable_vars:
            symbol = export_plan.exported_symbols[var_name]
            descriptor = {
                "module_name": (f"nb_{notebook_id}_{cell_id}_{var_name}_{source_hash[:12]}"),
                "symbol_name": var_name,
                "kind": symbol.kind,
                "source": export_plan.module_source,
                "provenance_hash": derive_subkey(provenance_hash, var_name),
            }
            output_file = output_dir / f"{var_name}.cell_module.json"
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

    # ------------------------------------------------------------------
    # Harness helpers
    # ------------------------------------------------------------------

    async def _run_harness(
        self,
        manifest_path: Path,
        venv_python: Path,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Run the harness script via uv."""
        cmd = [
            "uv",
            "run",
            "--directory",
            str(self.session.path),
            "python",
            str(self.harness_path),
            str(manifest_path),
        ]

        # Spawn the harness as the leader of a new process group so
        # SIGTERM / SIGKILL can target the entire descendant tree on
        # cancel (PyTorch DataLoader workers, multiprocessing pools,
        # …). Without this, ``proc.kill()`` only reaches the harness
        # and leaks every child it spawned.
        from strata.notebook.process_tree import (
            subprocess_kwargs_for_new_group,
            terminate_subprocess_tree,
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.session.path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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

        # Harness writes output to harness-result.json (separate
        # from the input manifest.json AND from any user variable
        # files like result.json). File absent → harness crashed
        # before reaching its finally block (typically a
        # ModuleNotFoundError at import time) — surface stderr so
        # the user sees what actually broke instead of an opaque
        # "Unknown error".
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

    # ------------------------------------------------------------------
    # Loop cell execution (Phase 1 — local worker, sequential iterations,
    # fresh subprocess per iter, no warm-pool reuse, no per-iter cache).
    # ------------------------------------------------------------------

    _LOOP_CONTENT_TYPE_EXT = {
        "arrow/ipc": ".arrow",
        "json/object": ".json",
        "pickle/object": ".pickle",
        "module/import": ".module.json",
        "module/cell": ".cell_module.json",
        "module/cell-instance": ".cell_instance.pickle",
    }

    async def _execute_loop_cell(
        self,
        cell_id: str,
        source: str,
        loop: LoopAnnotation,
        timeout_seconds: float,
        start_time: float,
        *,
        materialize_upstreams: bool,
    ) -> CellExecutionResult:
        """Execute a loop cell by running the body up to ``loop.max_iter`` times.

        Phase 1 constraints: local worker only, no RW mounts, no cache
        reuse across iterations on re-run (cache resumption is Phase 2).
        Every iteration spawns a fresh harness subprocess; the carry
        variable is passed in as a regular named input and read back as a
        regular named output. Iter k's new carry is stored as an artifact
        with an ``@iter=k`` suffix on its id.
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
            await self._materialize_upstreams(cell_id)

        mount_specs = self._resolve_cell_mount_specs(cell_id, source)
        _, has_rw_mount = await self._fingerprint_mounts(mount_specs)
        if has_rw_mount:
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                error=(
                    "Loop cells do not support rw mounts — they would make "
                    "per-iteration caching incorrect. Use an ro mount or "
                    "move the side-effect to a non-loop cell."
                ),
                execution_method="loop",
            )

        runtime_env = self._resolve_effective_runtime_env(cell_id, annotations.env)

        try:
            carry_blob, carry_content_type = self._resolve_loop_seed(cell_id, loop)
        except ValueError as exc:
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                error=str(exc),
                execution_method="loop",
            )

        source_hash = compute_source_hash(source)
        artifact_mgr = self.session.get_artifact_manager()

        final_artifact_uri: str | None = None
        final_result: dict[str, Any] | None = None
        combined_stdout: list[str] = []
        combined_stderr: list[str] = []
        all_mutation_warnings: list[MutationWarning] = []

        for k in range(loop.max_iter):
            with tempfile.TemporaryDirectory(prefix=f"strata_loop_iter_{k}_") as tmpdir:
                output_dir = Path(tmpdir)

                # Load upstream inputs first — this writes files into
                # output_dir, *including* the upstream seed for the carry
                # variable. We overwrite that seed with the current
                # iteration's carry below so iter k sees iter k-1's
                # output rather than the original upstream value.
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
                    loop_config=loop_config,
                )

                venv_path = self.session.venv_python or Path("python")
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
                    error_msg = result.get("error", "Unknown error")
                    duration_ms = (time.time() - start_time) * 1000
                    return CellExecutionResult(
                        cell_id=cell_id,
                        success=False,
                        error=f"Loop cell iter {k} failed: {error_msg}",
                        stdout="\n".join(combined_stdout),
                        stderr="\n".join(combined_stderr),
                        duration_ms=duration_ms,
                        execution_method="loop",
                        mutation_warnings=all_mutation_warnings,
                    )

                # Extract the new carry from the harness result.
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

                # Per-iteration provenance: chains through the previous iter's
                # carry bytes so re-runs with identical chains are detectable.
                prev_carry_hash = hashlib.sha256(carry_blob).hexdigest()
                iter_provenance = derive_subkey(source_hash, prev_carry_hash, f"iter={k}")

                artifact = artifact_mgr.store_cell_output(
                    cell_id=cell_id,
                    variable_name=loop.carry,
                    blob_data=new_carry_blob,
                    content_type=new_content_type,
                    provenance_hash=iter_provenance,
                    source_hash=source_hash,
                    iteration=k,
                )
                final_artifact_uri = f"strata://artifact/{artifact.id}@v={artifact.version}"

                carry_blob = new_carry_blob
                carry_content_type = new_content_type

                loop_state = result.get("loop") or {}
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

        # Also store the final iteration's carry under the non-iter
        # canonical id (``nb_..._var_<name>``) so downstream cells can
        # resolve it via the normal _load_input_blobs path, which looks
        # up the latest version of the canonical id. Without this, a
        # downstream cell referencing the carry variable would miss the
        # loop cell's output entirely even though the iter artifacts
        # are all there.
        #
        # Crucially, the canonical artifact's provenance must match what
        # ``compute_staleness`` recomputes on re-open — same per-variable
        # scheme used by the non-loop path ``sha256(prov:var_name)``
        # where ``prov`` comes from ``compute_provenance_hash`` over
        # narrow env + input hashes + mount fingerprints + source. If
        # we stored a custom hash here the loop cell would always look
        # stale on subsequent staleness computations.
        # Use the shared provenance helper — annotations and mount_specs
        # are already resolved above, so pass them in to avoid redoing
        # parse_annotations + mount discovery.
        prov = await self._compute_cell_provenance(
            cell_id,
            source,
            annotations=annotations,
            mount_specs=mount_specs,
        )
        cell_provenance = prov.provenance_hash
        env_hash = prov.env_hash
        carry_var_provenance = derive_subkey(cell_provenance, loop.carry)

        canonical_artifact = artifact_mgr.store_cell_output(
            cell_id=cell_id,
            variable_name=loop.carry,
            blob_data=carry_blob,
            content_type=carry_content_type,
            provenance_hash=carry_var_provenance,
            source_hash=source_hash,
        )
        canonical_uri = f"strata://artifact/{canonical_artifact.id}@v={canonical_artifact.version}"

        # Record the cell-level provenance on the session so
        # ``compute_staleness`` can hit the "uncached ready" path for
        # leaf loop cells that don't produce an upstream carry.
        self.session.record_successful_execution_provenance(
            cell_id,
            cell_provenance,
            source_hash,
            env_hash,
        )

        duration_ms = (time.time() - start_time) * 1000
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
                }
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

        Either a ``# @loop start_from=<cell>@iter=<k>`` reference is used,
        or an upstream cell in the DAG defines the carry variable and its
        latest artifact seeds iter 0. Raises ``ValueError`` when neither
        path yields a blob.
        """
        artifact_mgr = self.session.get_artifact_manager()

        if loop.start_from_cell is not None and loop.start_from_iter is not None:
            artifact_id = artifact_mgr.cell_artifact_id(
                loop.start_from_cell, loop.carry, loop.start_from_iter
            )
            artifact = artifact_mgr.artifact_store.get_latest_version(artifact_id)
            if artifact is None or artifact.state != "ready":
                raise ValueError(
                    f"Loop seed artifact not found for "
                    f"start_from={loop.start_from_cell}@iter={loop.start_from_iter}. "
                    f"Run that cell through iteration {loop.start_from_iter} first."
                )
            blob = artifact_mgr.artifact_store.blob_store.read_blob(artifact_id, artifact.version)
            if blob is None:
                raise ValueError(f"Loop seed blob missing for {artifact_id}@v={artifact.version}.")
            return blob, _artifact_content_type(artifact)

        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            raise ValueError(f"Loop cell {cell_id!r} not found in notebook state.")

        notebook_id = self.session.notebook_state.id
        for upstream_id in cell.upstream_ids:
            upstream_cell = self.session.notebook_state.get_cell(upstream_id)
            if upstream_cell is None or loop.carry not in upstream_cell.defines:
                continue
            upstream_artifact_id = f"nb_{notebook_id}_cell_{upstream_id}_var_{loop.carry}"
            artifact = artifact_mgr.artifact_store.get_latest_version(upstream_artifact_id)
            if artifact is None or artifact.state != "ready":
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
            suggest = _detect_missing_module(error_msg, stderr)
            return CellExecutionResult(
                cell_id=cell_id,
                success=False,
                stdout=result.get("stdout", ""),
                stderr=stderr,
                error=error_msg,
                duration_ms=duration_ms,
                execution_method=execution_method,
                suggest_install=suggest,
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
    # Run-all batching (PR-b2 of issue #26)
    # ------------------------------------------------------------------

    async def _run_batch(
        self,
        cell_specs: list[dict[str, Any]],
        *,
        use_cache: bool,
        batch_timeout_seconds: float,
    ) -> BatchExecutionResult:
        """Run a sequence of cells in one harness subprocess.

        See ``execute_batch`` for public docs. This handles all the
        subprocess + pipe wiring + frame protocol service. PR-b2 ships
        the happy/error paths; per-cell watchdog and full
        stdout/stderr attribution are deferred to PR-b3.
        """
        # Match single-cell's fallback: venv interpreter if synced,
        # otherwise PATH python (e.g. test environments that don't go
        # through the full env-sync flow).
        venv_python = self.session.venv_python or Path("python")

        batch_tmpdir = Path(
            tempfile.mkdtemp(
                prefix=f"strata_batch_{uuid.uuid4().hex[:8]}_",
                dir=str(self.session.path / ".strata"),
            )
        )
        manifest_path = batch_tmpdir / "batch_manifest.json"

        # Pipe pairs: frame_r/w (harness → parent), resp_r/w (parent → harness).
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
            manifest_path.write_bytes(orjson.dumps(manifest))

            env = {
                **os.environ,
                "STRATA_BATCH_FRAME_FD": str(frame_w),
                "STRATA_BATCH_RESP_FD": str(resp_r),
                "STRATA_BATCH_OUTPUT_DIR": str(batch_tmpdir),
            }

            # Spawn the batch harness as the leader of a new process
            # group so a SIGKILL on timeout reaches every descendant
            # (multiprocessing workers, DataLoader pools, …). Mirrors
            # single-cell's _run_harness at L2486.
            from strata.notebook.process_tree import (
                subprocess_kwargs_for_new_group,
                terminate_subprocess_tree,
            )

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
                **subprocess_kwargs_for_new_group(),
            )

            # Close parent-side copies of the harness-side fds so the
            # harness's EOF detection works correctly when the subprocess
            # exits.
            os.close(frame_w)
            os.close(resp_r)
            frame_w = -1  # mark already-closed for finally
            resp_r = -1

            # Drain harness stdout/stderr concurrently. PR-b2 just sinks
            # them to /dev/null to prevent pipe-buffer deadlock; full
            # per-cell attribution lands in PR-b3.
            assert proc.stdout is not None
            assert proc.stderr is not None
            stdout_task = asyncio.create_task(_drain_stream(proc.stdout))
            stderr_task = asyncio.create_task(_drain_stream(proc.stderr))

            # Wrap parent-side frame pipe as an asyncio StreamReader.
            loop = asyncio.get_running_loop()
            frame_reader = asyncio.StreamReader(loop=loop)
            frame_protocol = asyncio.StreamReaderProtocol(frame_reader, loop=loop)
            frame_file = os.fdopen(frame_r, "rb")
            frame_r = -1  # ownership transferred to file
            await loop.connect_read_pipe(lambda: frame_protocol, frame_file)

            # Write side uses sync os.write — kernel pipe buffer absorbs
            # the small JSON responses we send.
            resp_w_fd = resp_w
            resp_w = -1  # ownership transferred

            def send_response(payload: dict) -> None:
                line = orjson.dumps(payload) + b"\n"
                os.write(resp_w_fd, line)

            try:
                end_reason, failed_cell_id = await asyncio.wait_for(
                    self._batch_service_loop(
                        frame_reader,
                        send_response,
                        cell_results,
                        batch_tmpdir,
                        use_cache=use_cache,
                    ),
                    timeout=batch_timeout_seconds,
                )
                completed = end_reason == "complete"
            except TimeoutError:
                # SIGTERM the whole descendant tree, not just the
                # harness. User code may have spawned children
                # (multiprocessing pools, native threads) that proc.kill()
                # alone would leak.
                await terminate_subprocess_tree(proc)
                end_reason = "subprocess_died"
                completed = False
            finally:
                # Always close the response fd so the harness sees EOF
                # if it's still alive.
                try:
                    os.close(resp_w_fd)
                except OSError:
                    pass

            await proc.wait()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        finally:
            # Close any pipe fds still owned by the parent.
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
        use_cache: bool,
    ) -> tuple[str, str | None]:
        """Read frames and dispatch service requests until ``batch_end``.

        Returns (end_reason, failed_cell_id).
        """
        active_cell_id: str | None = None
        end_reason = "subprocess_died"
        failed_cell_id: str | None = None

        while True:
            line = await frame_reader.readline()
            if not line:
                # Pipe closed without batch_end — subprocess died.
                break
            try:
                frame = orjson.loads(line)
            except orjson.JSONDecodeError:
                logger.warning("Batch harness emitted unparseable frame: %r", line[:200])
                continue

            ftype = frame.get("type")
            payload = frame.get("payload") or {}

            if ftype == "cell_start":
                active_cell_id = payload.get("cell_id")
            elif ftype == "cache_check":
                response = await self._batch_service_cache_check(
                    payload.get("cell_id", ""),
                    batch_tmpdir,
                    use_cache=use_cache,
                )
                send_response(response)
            elif ftype == "persist":
                response = await self._batch_service_persist(payload, batch_tmpdir)
                cell_id_pl = payload["cell_id"]
                if response.get("ok"):
                    cell_results[cell_id_pl] = BatchCellResult(
                        cell_id=cell_id_pl,
                        status="ok",
                        stdout=payload.get("stdout", ""),
                        stderr=payload.get("stderr", ""),
                    )
                else:
                    # Persist rejected (module-export violation, store_outputs
                    # failure, etc.) — harness will see persist_err, emit
                    # batch_end(reason="persist_failed"), and exit. Record
                    # the failure here so the dispatcher (PR-b3) doesn't see
                    # this cell as "not_run".
                    cell_results[cell_id_pl] = BatchCellResult(
                        cell_id=cell_id_pl,
                        status="persist_failed",
                        error=response.get("error"),
                        stdout=payload.get("stdout", ""),
                        stderr=payload.get("stderr", ""),
                    )
                send_response(response)
            elif ftype == "cell_output":
                if payload.get("cache_hit") and active_cell_id:
                    cell_results[active_cell_id] = BatchCellResult(
                        cell_id=active_cell_id,
                        status="cache_hit",
                    )
            elif ftype == "cell_error":
                cell_id = payload.get("cell_id", "")
                cell_results[cell_id] = BatchCellResult(
                    cell_id=cell_id,
                    status="cell_error",
                    error=payload.get("error"),
                    traceback=payload.get("traceback"),
                    stdout=payload.get("stdout", ""),
                    stderr=payload.get("stderr", ""),
                )
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

        The harness seeds its namespace from these on entry. Variables
        whose producing cell is also inside the batch are handled by the
        in-batch flow and shouldn't be in this dict.
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
                for var_name, uri in upstream_cell.artifact_uris.items():
                    if var_name in inputs:
                        continue
                    spec_dict = self._materialize_artifact_to_dir(uri, upstream_dir, var_name)
                    if spec_dict is not None:
                        inputs[var_name] = spec_dict

        # Inputs paths are relative to batch_tmpdir (since the harness
        # passes output_dir=batch_tmpdir to deserialize_inputs).
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
        """Read an artifact by URI and write its blob to ``target_dir/<var><ext>``.

        Returns ``{content_type, file}`` or None on failure.
        """
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
        file_name = f"{var_name}{ext}"
        (target_dir / file_name).write_bytes(blob)
        return {"content_type": content_type, "file": file_name}

    async def _batch_service_cache_check(
        self,
        cell_id: str,
        batch_tmpdir: Path,
        *,
        use_cache: bool,
    ) -> dict[str, Any]:
        """Service a ``cache_check`` request from the batch harness.

        Computes provenance via the existing single-cell helpers and
        checks the artifact store. On hit, materializes cached blobs to
        ``batch_tmpdir/<cell_id>/{var}{ext}`` AND mirrors single-cell
        L823-844 (populates ``cell.artifact_uris`` + ``display_outputs``).
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            return {"cache_hit": False, "provenance_hash": ""}

        source = cell.source
        try:
            prov = await self._compute_cell_provenance(cell_id, source)
        except Exception as exc:
            logger.warning("Batch cache_check provenance failed for %s: %s", cell_id, exc)
            return {"cache_hit": False, "provenance_hash": ""}

        provenance_hash = prov.provenance_hash
        if not use_cache:
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

        # A cell with no consumed vars has nothing meaningful to cache
        # (PR-b3 handles display-only cache hits). Treat as miss so the
        # cell executes for its side effects.
        if not consumed_vars:
            return {"cache_hit": False, "provenance_hash": provenance_hash}

        cached_outputs: dict[str, dict[str, str]] = {}
        for var_name in consumed_vars:
            canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{var_name}"
            var_prov = derive_subkey(provenance_hash, var_name)
            canonical_art = artifact_mgr.artifact_store.get_latest_version(canonical_id)
            if canonical_art is None or canonical_art.provenance_hash != var_prov:
                return {"cache_hit": False, "provenance_hash": provenance_hash}
            blob = artifact_mgr.artifact_store.read_blob(canonical_art.id, canonical_art.version)
            if blob is None:
                return {"cache_hit": False, "provenance_hash": provenance_hash}
            content_type = _artifact_content_type(canonical_art)
            ext = _ARTIFACT_EXT_BY_CONTENT_TYPE.get(content_type, ".bin")
            file_name = f"{var_name}{ext}"
            (cell_output_dir / file_name).write_bytes(blob)
            cached_outputs[var_name] = {
                "content_type": content_type,
                "file": file_name,
            }
            # Mirror single-cell L823-844 — populate artifact_uris so
            # downstream cells in the batch resolve via _collect_input_hashes.
            uri = f"strata://artifact/{canonical_art.id}@v={canonical_art.version}"
            cell.artifact_uris[var_name] = uri
            cell.artifact_uri = uri
        cell.cache_hit = True

        # Display outputs aren't covered by PR-b2's minimal cache-hit
        # mirror — they ship in PR-b3 alongside the broader test suite.
        return {
            "cache_hit": True,
            "provenance_hash": provenance_hash,
            "cached_outputs": cached_outputs,
            "cached_displays": [],
        }

    async def _batch_service_persist(
        self,
        payload: dict[str, Any],
        batch_tmpdir: Path,
    ) -> dict[str, Any]:
        """Service a ``persist`` request from the batch harness.

        Calls the existing single-cell persistence chain:
        ``_write_module_export_outputs`` → ``_store_outputs`` →
        ``_store_display_outputs``. Same code paths single-cell uses
        post-L924; keeps ``cell.artifact_uris`` compatible.
        """
        cell_id = payload.get("cell_id", "")
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            return {"ok": False, "error": f"cell {cell_id} not found"}

        cell_output_dir = batch_tmpdir / cell_id
        if not cell_output_dir.exists():
            return {"ok": False, "error": f"output dir missing for {cell_id}"}

        # Recompute provenance (same as cache_check). Deterministic given
        # the cell's current source + upstream artifact_uris.
        try:
            prov = await self._compute_cell_provenance(cell_id, cell.source)
        except Exception as exc:
            return {"ok": False, "error": f"provenance compute failed: {exc}"}

        provenance_hash = prov.provenance_hash
        source_hash = prov.source_hash
        env_hash = prov.env_hash
        input_hashes = self._collect_input_hashes(cell_id)

        # Module export: scans source AST, writes synthetic .cell_module.json /
        # .cell_instance.pickle into cell_output_dir if there are top-level
        # defs/classes. _store_outputs picks them up in the same iteration.
        # A returned error string means a downstream-consumed def/class can't
        # be safely exported under the current V1 rules — single-cell mode
        # converts that into success=False (executor.py L995). Batch must
        # do the same: refuse to persist, signal the harness, so the batch
        # ends and the cell shows as errored.
        try:
            module_export_error = self._write_module_export_outputs(
                cell_id,
                cell.source,
                cell_output_dir,
                provenance_hash,
                {},  # outputs dict; module export reads from AST, not this
            )
        except Exception as exc:
            return {"ok": False, "error": f"module export raised: {exc}"}
        if module_export_error:
            return {"ok": False, "error": f"module export rejected: {module_export_error}"}

        # Persist consumed variables via the existing helper.
        stored_ok = self._store_outputs(
            cell_id,
            cell_output_dir,
            provenance_hash,
            input_hashes,
            source_hash=source_hash,
            env_hash=env_hash,
        )

        # Persist display outputs.
        display_outputs_meta = payload.get("display_outputs") or []
        if display_outputs_meta:
            try:
                self._store_display_outputs(
                    cell_id,
                    cell_output_dir,
                    provenance_hash,
                    input_hashes,
                    display_outputs_meta,
                    source_hash=source_hash,
                    env_hash=env_hash,
                )
            except Exception as exc:
                logger.warning("Batch display persist failed for %s: %s", cell_id, exc)

        if not stored_ok:
            return {"ok": False, "error": "store_outputs returned False"}

        # Record provenance + execution as single-cell does.
        try:
            self.session.record_successful_execution_provenance(
                cell_id, provenance_hash, source_hash, env_hash
            )
        except Exception:
            pass

        uri = cell.artifact_uri or ""
        return {"ok": True, "uri": uri}


# ---------------------------------------------------------------------------
# Batch helpers (module-level)
# ---------------------------------------------------------------------------


_ARTIFACT_EXT_BY_CONTENT_TYPE: dict[str, str] = {
    "arrow/ipc": ".arrow",
    "json/object": ".json",
    "pickle/object": ".pickle",
    "module/import": ".module.json",
    "module/cell": ".cell_module.json",
    "module/cell-instance": ".cell_instance.pickle",
}


def _artifact_content_type(artifact: Any) -> str:
    """Extract content_type from an ``ArtifactVersion``'s transform_spec.

    Content type lives in ``transform_spec.params.content_type`` — same
    pattern ``NotebookArtifactManager.get_artifact_preview`` reads from.
    Returns ``"pickle/object"`` if missing (safe default — round-trips
    via cloudpickle).
    """
    spec_json = getattr(artifact, "transform_spec", None)
    if not spec_json:
        return "pickle/object"
    try:
        spec = json.loads(spec_json)
        return spec.get("params", {}).get("content_type") or "pickle/object"
    except (ValueError, KeyError, TypeError):
        return "pickle/object"


async def _drain_stream(stream: asyncio.StreamReader) -> None:
    """Read and discard everything from ``stream`` until EOF.

    PR-b2 sinks raw stdout/stderr to /dev/null to prevent kernel pipe
    buffer deadlocks on chatty native code. PR-b3 will attribute reads
    to the currently-active cell via the most-recent ``cell_start`` frame.
    """
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
