"""Notebook worker catalog helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from strata.notebook.models import NotebookState, WorkerBackendType, WorkerSpec
from strata.notebook.remote_executor import NOTEBOOK_EXECUTOR_TRANSFORM_REF

_HEALTH_CACHE_TTL_SECONDS = 5.0
_HEALTH_HISTORY_LIMIT = 6


@dataclass(frozen=True)
class ManagedWorkerRecord:
    """Service-managed worker config plus operational policy flags."""

    worker: WorkerSpec
    enabled: bool = True


@dataclass(frozen=True)
class WorkerHealthSnapshot:
    """Cached result of a worker health probe."""

    checked_at: float
    health: str
    error: str | None = None
    duration_ms: int | None = None


@dataclass(frozen=True)
class WorkerHealthRecord:
    """Cached worker health plus a short recent probe trail."""

    latest: WorkerHealthSnapshot
    history: tuple[WorkerHealthSnapshot, ...]
    probe_count: int
    healthy_probe_count: int
    unavailable_probe_count: int
    unknown_probe_count: int
    consecutive_failures: int
    last_healthy_at: float | None = None
    last_unavailable_at: float | None = None
    last_unknown_at: float | None = None
    last_status_change_at: float | None = None


_worker_health_cache: dict[str, WorkerHealthRecord] = {}


@dataclass(frozen=True)
class WorkerPolicy:
    """Effective worker policy for one notebook in the current deployment mode."""

    service_mode: bool
    definitions_editable: bool
    effective_workers: dict[str, WorkerSpec]
    server_workers: dict[str, ManagedWorkerRecord]


def get_builtin_local_worker() -> WorkerSpec:
    """Return the implicit built-in local worker."""
    return WorkerSpec(name="local", backend=WorkerBackendType.LOCAL)


def _parse_managed_worker_records(raw_specs: Any) -> list[ManagedWorkerRecord]:
    """Parse service-managed worker entries from an untyped config list."""
    if not isinstance(raw_specs, list):
        return []

    parsed: list[ManagedWorkerRecord] = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            continue
        try:
            parsed.append(
                ManagedWorkerRecord(
                    worker=WorkerSpec.model_validate(raw_spec),
                    enabled=bool(raw_spec.get("enabled", True)),
                )
            )
        except Exception:
            continue
    return parsed


def _serialize_managed_worker_records(
    records: list[ManagedWorkerRecord],
) -> list[dict[str, Any]]:
    """Serialize service-managed worker entries for config and responses."""
    return [
        {
            **record.worker.model_dump(mode="json"),
            "enabled": record.enabled,
        }
        for record in records
    ]


def _load_worker_policy(notebook_state: NotebookState) -> WorkerPolicy:
    """Load the effective notebook worker policy for the current deployment."""
    try:
        from strata.server import get_state

        state = get_state()
        config = state.config
        service_mode = config.deployment_mode == "service"
        server_workers = {
            record.worker.name: record
            for record in _parse_managed_worker_records(
                config.transforms_config.get("notebook_workers", [])
            )
        }
    except Exception:
        service_mode = False
        server_workers = {}

    builtin = get_builtin_local_worker()
    if service_mode:
        effective_workers = {
            "local": builtin,
            **{name: record.worker for name, record in server_workers.items() if record.enabled},
        }
    else:
        effective_workers = {
            "local": builtin,
            **{worker.name: worker for worker in notebook_state.workers},
        }

    return WorkerPolicy(
        service_mode=service_mode,
        definitions_editable=not service_mode,
        effective_workers=effective_workers,
        server_workers=server_workers,
    )


def notebook_worker_definitions_editable(notebook_state: NotebookState) -> bool:
    """Return whether notebook-scoped worker definitions can be edited."""
    return _load_worker_policy(notebook_state).definitions_editable


def get_server_managed_workers() -> list[WorkerSpec]:
    """Return the configured service-mode notebook worker registry."""
    return [record.worker for record in get_server_managed_worker_records()]


def get_server_managed_worker_records() -> list[ManagedWorkerRecord]:
    """Return the configured service-mode notebook worker registry."""
    try:
        from strata.server import get_state

        return _parse_managed_worker_records(
            get_state().config.transforms_config.get("notebook_workers", [])
        )
    except Exception:
        return []


def set_server_managed_workers(workers: list[WorkerSpec]) -> list[WorkerSpec]:
    """Replace the configured service-mode notebook worker registry."""
    replace_server_managed_worker_records(
        [ManagedWorkerRecord(worker=worker, enabled=True) for worker in workers]
    )
    return get_server_managed_workers()


def create_server_managed_worker_record(
    record: ManagedWorkerRecord,
) -> list[ManagedWorkerRecord]:
    """Create one service-managed worker entry.

    Raises:
        ValueError: If a worker with the same name already exists.
    """
    records = get_server_managed_worker_records()
    if any(existing.worker.name == record.worker.name for existing in records):
        raise ValueError(record.worker.name)
    return replace_server_managed_worker_records([*records, record])


def update_server_managed_worker_record(
    worker_name: str,
    record: ManagedWorkerRecord,
) -> list[ManagedWorkerRecord]:
    """Replace one service-managed worker entry in place.

    Allows renaming the worker as long as the new name does not collide with
    another configured worker.

    Raises:
        KeyError: If the referenced worker does not exist.
        ValueError: If the requested new worker name would collide.
    """
    records = get_server_managed_worker_records()
    next_records: list[ManagedWorkerRecord] = []
    updated = False

    for existing in records:
        if existing.worker.name == worker_name:
            updated = True
            continue
        next_records.append(existing)

    if not updated:
        raise KeyError(worker_name)

    if any(existing.worker.name == record.worker.name for existing in next_records):
        raise ValueError(record.worker.name)

    insert_at = next(
        (index for index, existing in enumerate(records) if existing.worker.name == worker_name),
        len(next_records),
    )
    next_records.insert(insert_at, record)
    return replace_server_managed_worker_records(next_records)


def replace_server_managed_worker_records(
    records: list[ManagedWorkerRecord],
) -> list[ManagedWorkerRecord]:
    """Replace the configured service-mode notebook worker registry."""
    from strata.server import get_state

    state = get_state()
    state.config.transforms_config["notebook_workers"] = _serialize_managed_worker_records(records)
    return get_server_managed_worker_records()


def set_server_managed_worker_enabled(
    worker_name: str,
    enabled: bool,
) -> list[ManagedWorkerRecord]:
    """Enable or disable one service-managed worker by name."""
    records = get_server_managed_worker_records()
    updated = False
    next_records: list[ManagedWorkerRecord] = []

    for record in records:
        if record.worker.name == worker_name:
            next_records.append(ManagedWorkerRecord(worker=record.worker, enabled=enabled))
            updated = True
        else:
            next_records.append(record)

    if not updated:
        raise KeyError(worker_name)

    return replace_server_managed_worker_records(next_records)


def delete_server_managed_worker_record(worker_name: str) -> list[ManagedWorkerRecord]:
    """Delete one service-managed worker by name."""
    records = get_server_managed_worker_records()
    next_records = [record for record in records if record.worker.name != worker_name]
    if len(next_records) == len(records):
        raise KeyError(worker_name)
    return replace_server_managed_worker_records(next_records)


def validate_worker_assignment(
    notebook_state: NotebookState,
    worker_name: str | None,
) -> str | None:
    """Validate a requested worker assignment against deployment policy.

    Returns ``None`` when the assignment is allowed. In personal mode, worker
    assignments remain permissive so notebooks can reference future workers.
    In service mode, assignments must resolve against the server-managed
    worker registry.
    """
    normalized_name = (worker_name or "").strip()
    if not normalized_name or normalized_name == "local":
        return None

    policy = _load_worker_policy(notebook_state)
    if not policy.service_mode:
        return None

    server_record = policy.server_workers.get(normalized_name)
    if server_record is not None and not server_record.enabled:
        return (
            f"Worker '{normalized_name}' is disabled by server policy. "
            "Choose an enabled server-managed worker."
        )

    if normalized_name in policy.effective_workers:
        return None

    return (
        f"Worker '{normalized_name}' is not allowed in service mode. "
        "Choose a server-managed worker."
    )


def get_worker_execution_error(
    notebook_state: NotebookState,
    worker_name: str | None,
) -> str | None:
    """Return an execution-facing worker policy error, if any."""
    normalized_name = (worker_name or "").strip() or "local"
    policy_error = validate_worker_assignment(notebook_state, normalized_name)
    if policy_error is not None:
        return f"Execution failed: {policy_error}"
    return None


def resolve_worker_spec(
    notebook_state: NotebookState,
    worker_name: str | None,
) -> WorkerSpec | None:
    """Resolve a worker name against the effective worker policy.

    ``None`` and ``"local"`` map to the implicit built-in local worker.
    In personal mode, notebook-scoped worker definitions are visible here.
    In service mode, only the server-managed registry is visible.
    """
    normalized_name = (worker_name or "").strip()
    if not normalized_name or normalized_name == "local":
        return get_builtin_local_worker()

    return _load_worker_policy(notebook_state).effective_workers.get(normalized_name)


def worker_runtime_identity(
    notebook_state: NotebookState,
    worker_name: str | None,
) -> str | None:
    """Return the runtime identity that should participate in provenance."""
    worker = resolve_worker_spec(notebook_state, worker_name)
    if worker is None:
        return None

    if worker.runtime_id:
        return f"{worker.backend.value}:{worker.runtime_id}"

    # exclude_none reproduces the old raw-dict shape (only set keys + extras), so
    # the provenance hash is unchanged for existing configs.
    config_dict = worker.config.model_dump(mode="json", exclude_none=True)
    if config_dict:
        config_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        return f"{worker.backend.value}:{worker.name}:{config_hash}"

    return f"{worker.backend.value}:{worker.name}"


def is_embedded_executor_worker(worker: WorkerSpec | None) -> bool:
    """Return True when the worker uses the local embedded executor path."""
    if worker is None or worker.backend != WorkerBackendType.EXECUTOR:
        return False
    url = str(worker.config.url or "").strip()
    return url in {"embedded://local", "embedded://notebook"}


def is_http_executor_worker(worker: WorkerSpec | None) -> bool:
    """Return True when the worker points at an HTTP notebook executor."""
    if worker is None or worker.backend != WorkerBackendType.EXECUTOR:
        return False
    url = str(worker.config.url or "").strip()
    scheme = urlparse(url).scheme.lower()
    return scheme in {"http", "https"}


def worker_supports_notebook_execution(worker: WorkerSpec | None) -> bool:
    """Return whether notebook execution can run on this worker today."""
    if worker is None:
        return False
    if worker.backend == WorkerBackendType.LOCAL:
        return True
    return is_embedded_executor_worker(worker) or is_http_executor_worker(worker)


def worker_transport(worker: WorkerSpec) -> str:
    """Return a stable UI-facing transport label for one worker."""
    if worker.backend == WorkerBackendType.LOCAL:
        return "local"

    url = str(worker.config.url or "").strip()
    transport = str(worker.config.transport or "direct").strip().lower()

    if url.startswith("embedded://"):
        return "embedded"
    if transport in {"signed", "manifest", "build"}:
        return "signed"
    if url.startswith("http://") or url.startswith("https://"):
        return "direct"
    return "executor"


def build_worker_catalog(notebook_state: NotebookState) -> list[dict[str, Any]]:
    """Build a UI-facing worker catalog for a notebook.

    The catalog always includes the built-in ``local`` worker, notebook-scoped
    worker definitions, and synthetic unavailable entries for any referenced
    worker names that no longer exist in the configured registry.
    """
    policy = _load_worker_policy(notebook_state)
    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_worker(
        worker: WorkerSpec,
        *,
        source: str,
        health: str,
        allowed: bool,
        enabled: bool = True,
    ) -> None:
        if worker.name in seen:
            return
        seen.add(worker.name)
        catalog.append(
            {
                "name": worker.name,
                "backend": worker.backend.value,
                "runtime_id": worker.runtime_id,
                "config": worker.config.model_dump(mode="json", exclude_none=True),
                "source": source,
                "health": health,
                "allowed": allowed,
                "enabled": enabled,
                "transport": worker_transport(worker),
                "health_url": _health_url_for_worker(worker),
                "health_checked_at": None,
                "last_error": None,
                "health_history": [],
            }
        )

    add_worker(
        get_builtin_local_worker(),
        source="builtin",
        health="healthy",
        allowed=True,
    )

    if policy.service_mode:
        for record in policy.server_workers.values():
            worker = record.worker
            health = (
                "healthy"
                if worker.backend == WorkerBackendType.LOCAL or is_embedded_executor_worker(worker)
                else "unknown"
            )
            add_worker(
                worker,
                source="server",
                health=health,
                allowed=record.enabled,
                enabled=record.enabled,
            )

        for worker in notebook_state.workers:
            add_worker(
                worker,
                source="notebook",
                health="unavailable",
                allowed=False,
                enabled=True,
            )
    else:
        for worker in notebook_state.workers:
            health = (
                "healthy"
                if worker.backend == WorkerBackendType.LOCAL or is_embedded_executor_worker(worker)
                else "unknown"
            )
            add_worker(
                worker,
                source="notebook",
                health=health,
                allowed=True,
                enabled=True,
            )

    referenced = set()
    if notebook_state.worker:
        referenced.add(notebook_state.worker)
    for cell in notebook_state.cells:
        if cell.worker:
            referenced.add(cell.worker)

    for worker_name in sorted(referenced):
        if worker_name in seen:
            continue
        add_worker(
            WorkerSpec(name=worker_name, backend=WorkerBackendType.EXECUTOR),
            source="referenced",
            health="unavailable",
            allowed=False,
            enabled=True,
        )

    return catalog


def _health_url_for_worker(worker: WorkerSpec) -> str | None:
    """Map a worker config URL to its health endpoint."""
    if not is_http_executor_worker(worker):
        return None

    raw_url = str(worker.config.url or "").strip()
    if not raw_url:
        return None

    parsed = urlparse(raw_url)
    path = parsed.path or ""
    if path.endswith("/v1/execute"):
        path = path[: -len("/v1/execute")] + "/health"
    elif path.endswith("/v1/notebook-execute"):
        path = path[: -len("/v1/notebook-execute")] + "/health"
    elif not path or path == "/":
        path = "/health"
    else:
        path = f"{path.rstrip('/')}/health"

    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def _record_worker_health_snapshot(
    health_url: str,
    snapshot: WorkerHealthSnapshot,
) -> WorkerHealthRecord:
    """Persist one probe result in the short in-memory health history."""
    existing = _worker_health_cache.get(health_url)
    history = (snapshot, *(existing.history if existing is not None else ()))
    previous_health = existing.latest.health if existing is not None else None
    probe_count = (existing.probe_count if existing is not None else 0) + 1
    healthy_probe_count = (existing.healthy_probe_count if existing is not None else 0) + (
        1 if snapshot.health == "healthy" else 0
    )
    unavailable_probe_count = (existing.unavailable_probe_count if existing is not None else 0) + (
        1 if snapshot.health == "unavailable" else 0
    )
    unknown_probe_count = (existing.unknown_probe_count if existing is not None else 0) + (
        1 if snapshot.health == "unknown" else 0
    )
    record = WorkerHealthRecord(
        latest=snapshot,
        history=history[:_HEALTH_HISTORY_LIMIT],
        probe_count=probe_count,
        healthy_probe_count=healthy_probe_count,
        unavailable_probe_count=unavailable_probe_count,
        unknown_probe_count=unknown_probe_count,
        consecutive_failures=(
            0
            if snapshot.health == "healthy"
            else (existing.consecutive_failures if existing is not None else 0) + 1
        ),
        last_healthy_at=(
            snapshot.checked_at
            if snapshot.health == "healthy"
            else (existing.last_healthy_at if existing is not None else None)
        ),
        last_unavailable_at=(
            snapshot.checked_at
            if snapshot.health == "unavailable"
            else (existing.last_unavailable_at if existing is not None else None)
        ),
        last_unknown_at=(
            snapshot.checked_at
            if snapshot.health == "unknown"
            else (existing.last_unknown_at if existing is not None else None)
        ),
        last_status_change_at=(
            snapshot.checked_at
            if previous_health is None or previous_health != snapshot.health
            else (existing.last_status_change_at if existing is not None else None)
        ),
    )
    _worker_health_cache[health_url] = record
    return record


def _serialize_worker_health_history(
    history: tuple[WorkerHealthSnapshot, ...],
) -> list[dict[str, Any]]:
    """Serialize recent health probes for API responses."""
    return [
        {
            "checked_at": int(snapshot.checked_at * 1000),
            "health": snapshot.health,
            "error": snapshot.error,
            "duration_ms": snapshot.duration_ms,
        }
        for snapshot in history
    ]


def _serialize_worker_health_record(
    record: WorkerHealthRecord | None,
) -> dict[str, Any]:
    """Serialize aggregate worker health metadata for API responses."""
    if record is None:
        return {
            "probe_count": 0,
            "healthy_probe_count": 0,
            "unavailable_probe_count": 0,
            "unknown_probe_count": 0,
            "consecutive_failures": 0,
            "last_healthy_at": None,
            "last_unavailable_at": None,
            "last_unknown_at": None,
            "last_status_change_at": None,
            "last_probe_duration_ms": None,
        }

    return {
        "probe_count": record.probe_count,
        "healthy_probe_count": record.healthy_probe_count,
        "unavailable_probe_count": record.unavailable_probe_count,
        "unknown_probe_count": record.unknown_probe_count,
        "consecutive_failures": record.consecutive_failures,
        "last_healthy_at": (
            int(record.last_healthy_at * 1000) if record.last_healthy_at is not None else None
        ),
        "last_unavailable_at": (
            int(record.last_unavailable_at * 1000)
            if record.last_unavailable_at is not None
            else None
        ),
        "last_unknown_at": (
            int(record.last_unknown_at * 1000) if record.last_unknown_at is not None else None
        ),
        "last_status_change_at": (
            int(record.last_status_change_at * 1000)
            if record.last_status_change_at is not None
            else None
        ),
        "last_probe_duration_ms": record.latest.duration_ms,
    }


async def probe_worker_health(
    worker: WorkerSpec,
    *,
    timeout_seconds: float = 8.0,
    force_refresh: bool = False,
) -> WorkerHealthSnapshot:
    """Probe a worker health endpoint and return a cached health snapshot.

    The 8-second default accommodates serverless cold starts (Modal,
    Fly scale-to-zero). Locally-running workers respond in a few
    milliseconds; the timeout only kicks in when a remote backend is
    waking up. Callers that iterate over multiple workers should do
    so via ``asyncio.gather`` so the timeouts overlap rather than
    compound. GPU cold boots >8 s will still show "unavailable"
    briefly but recover on the next probe cycle once the container
    is warm.
    """
    """Probe a worker health endpoint and return a cached health snapshot."""
    now = time.time()
    if worker.backend == WorkerBackendType.LOCAL or is_embedded_executor_worker(worker):
        return WorkerHealthSnapshot(checked_at=now, health="healthy")
    if not is_http_executor_worker(worker):
        return WorkerHealthSnapshot(checked_at=now, health="unknown")

    health_url = _health_url_for_worker(worker)
    if not health_url:
        return WorkerHealthSnapshot(checked_at=now, health="unknown")

    cached = _worker_health_cache.get(health_url)
    if (
        not force_refresh
        and cached is not None
        and now - cached.latest.checked_at < _HEALTH_CACHE_TTL_SECONDS
    ):
        return cached.latest

    try:
        started_at = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(health_url)
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        if response.status_code != 200:
            snapshot = WorkerHealthSnapshot(
                checked_at=now,
                health="unavailable",
                error=f"Health endpoint returned {response.status_code}",
                duration_ms=duration_ms,
            )
        else:
            payload = response.json()
            capabilities = payload.get("capabilities", {})
            transform_refs = capabilities.get("transform_refs", [])
            status = str(payload.get("status", "unknown"))
            if (
                status == "healthy"
                and isinstance(transform_refs, list)
                and NOTEBOOK_EXECUTOR_TRANSFORM_REF in transform_refs
            ):
                snapshot = WorkerHealthSnapshot(
                    checked_at=now,
                    health="healthy",
                    duration_ms=duration_ms,
                )
            elif status == "healthy":
                snapshot = WorkerHealthSnapshot(
                    checked_at=now,
                    health="unknown",
                    error="Executor is healthy but does not advertise notebook execution support",
                    duration_ms=duration_ms,
                )
            else:
                snapshot = WorkerHealthSnapshot(
                    checked_at=now,
                    health="unavailable",
                    error=f"Executor reported status {status}",
                    duration_ms=duration_ms,
                )
    except httpx.TimeoutException:
        # A timed-out probe usually means a serverless worker is
        # cold-starting. Surface it as "warming" so the UI can render
        # a distinct state (rather than lumping cold-start with real
        # failures like DNS errors or 5xx responses).
        snapshot = WorkerHealthSnapshot(
            checked_at=now,
            health="warming",
            error=f"Probe timed out after {timeout_seconds:.0f}s; worker may be cold-starting",
            duration_ms=int(timeout_seconds * 1000),
        )
    except Exception as exc:
        snapshot = WorkerHealthSnapshot(
            checked_at=now,
            health="unavailable",
            error=str(exc),
        )

    _record_worker_health_snapshot(health_url, snapshot)
    return snapshot


async def build_worker_catalog_with_health(
    notebook_state: NotebookState,
    *,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Build a worker catalog enriched with live health probes where possible."""
    catalog = build_worker_catalog(notebook_state)
    policy = _load_worker_policy(notebook_state)
    worker_by_name = (
        {name: record.worker for name, record in policy.server_workers.items()}
        if policy.service_mode
        else {worker.name: worker for worker in policy.effective_workers.values()}
    )

    # First pass: resolve static status (local, disabled) and collect
    # the set of entries whose health needs a live probe. Gathering
    # the probes in parallel keeps the catalog response fast even
    # when one worker is cold-starting and the other is healthy —
    # sequentially, the healthy one would have to wait for the cold
    # one's 8 s timeout.
    probe_targets: list[tuple[dict[str, Any], WorkerSpec, str | None]] = []
    for entry in catalog:
        name = str(entry.get("name", ""))
        if name == "local":
            entry["health"] = "healthy"
            entry["health_checked_at"] = int(time.time() * 1000)
            entry["health_history"] = []
            entry.update(_serialize_worker_health_record(None))
            continue
        policy_error: str | None = None
        if entry.get("allowed") is False:
            if entry.get("enabled") is False:
                policy_error = "Worker is disabled by server policy and is not selectable"
            else:
                policy_error = "Worker is not selectable in the current notebook policy"
        if entry.get("allowed") is False and entry.get("source") != "server":
            entry["health"] = "unavailable"
            entry["last_error"] = policy_error
            entry["health_history"] = []
            entry.update(_serialize_worker_health_record(None))
            continue

        worker = worker_by_name.get(name)
        if worker is None:
            entry["health_history"] = []
            entry.update(_serialize_worker_health_record(None))
            continue

        probe_targets.append((entry, worker, policy_error))

    snapshots = await asyncio.gather(
        *(
            probe_worker_health(worker, force_refresh=force_refresh)
            for _, worker, _ in probe_targets
        )
    )

    for (entry, worker, policy_error), snapshot in zip(probe_targets, snapshots, strict=True):
        entry["health"] = snapshot.health
        entry["health_checked_at"] = int(snapshot.checked_at * 1000)
        entry["last_error"] = policy_error or snapshot.error
        health_url = _health_url_for_worker(worker)
        if health_url is None:
            entry["health_history"] = []
            entry.update(_serialize_worker_health_record(None))
        else:
            record = _worker_health_cache.get(health_url)
            entry["health_history"] = (
                _serialize_worker_health_history(record.history) if record is not None else []
            )
            entry.update(_serialize_worker_health_record(record))

    return catalog


async def build_server_worker_catalog_with_health(
    *,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Build the service-mode worker catalog without notebook-local entries."""
    catalog: list[dict[str, Any]] = [
        {
            "name": "local",
            "backend": WorkerBackendType.LOCAL.value,
            "runtime_id": None,
            "config": {},
            "source": "builtin",
            "health": "healthy",
            "allowed": True,
            "enabled": True,
            "transport": "local",
            "health_url": None,
            "health_checked_at": int(time.time() * 1000),
            "last_error": None,
            "health_history": [],
            **_serialize_worker_health_record(None),
        }
    ]

    for record in get_server_managed_worker_records():
        worker = record.worker
        snapshot = await probe_worker_health(worker, force_refresh=force_refresh)
        health_url = _health_url_for_worker(worker)
        cached_record = _worker_health_cache.get(health_url) if health_url else None
        catalog.append(
            {
                "name": worker.name,
                "backend": worker.backend.value,
                "runtime_id": worker.runtime_id,
                "config": worker.config.model_dump(mode="json", exclude_none=True),
                "source": "server",
                "health": snapshot.health,
                "allowed": record.enabled,
                "enabled": record.enabled,
                "transport": worker_transport(worker),
                "health_url": health_url,
                "health_checked_at": int(snapshot.checked_at * 1000),
                "last_error": (
                    "Worker is disabled by server policy" if not record.enabled else snapshot.error
                ),
                "health_history": _serialize_worker_health_history(cached_record.history)
                if cached_record is not None
                else [],
                **_serialize_worker_health_record(cached_record),
            }
        )

    return catalog
