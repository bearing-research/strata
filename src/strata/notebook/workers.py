"""Notebook worker catalog helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import httpx

from strata.notebook.models import NotebookState, WorkerBackendType, WorkerSpec
from strata.notebook.remote_executor import NOTEBOOK_EXECUTOR_TRANSFORM_REF

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

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
    from strata.server import get_state

    try:
        config = get_state().config
    except RuntimeError:
        # No server (a CLI run): the notebook's own workers, nothing to restrict.
        config = None
    service_mode = config is not None and config.deployment_mode == "service"
    # Same accessor the admin routes use, so the catalogue and dispatch cannot disagree
    # (transforms_config would diverge from the persisted registry after a restart). A store
    # error propagates: falling back to personal rules would let a notebook's own workers run.
    server_workers = (
        {record.worker.name: record for record in get_server_managed_worker_records()}
        if config is not None
        else {}
    )

    builtin = get_builtin_local_worker()
    if service_mode:
        effective_workers = {
            "local": builtin,
            **{name: record.worker for name, record in server_workers.items() if record.enabled},
        }
    else:
        # Personal mode gets the server registry too, so a platform's machine-type catalogue
        # need not be copied into every notebook.toml. Notebook entries last so a name the
        # notebook defines wins, matching annotations over persisted config.
        effective_workers = {
            "local": builtin,
            **{name: record.worker for name, record in server_workers.items() if record.enabled},
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


def prune_worker_health_cache() -> int:
    """Drop cached health for workers the registry no longer lists; return the count.

    Entries are keyed by health URL, so a moved worker already misses; this bounds
    the entries for URLs nobody asks about any more.
    """
    live = {_health_url_for_worker(record.worker) for record in get_server_managed_worker_records()}
    stale = [url for url in _worker_health_cache if url not in live]
    for url in stale:
        del _worker_health_cache[url]
    return len(stale)


def _configured_managed_worker_records() -> list[ManagedWorkerRecord]:
    """The configured ``notebook_workers`` table, in force until the registry is first set."""
    try:
        from strata.server import get_state

        return _parse_managed_worker_records(
            get_state().config.transforms_config.get("notebook_workers", [])
        )
    except Exception:
        return []


def get_server_managed_worker_records() -> list[ManagedWorkerRecord]:
    """Return the server-managed notebook worker registry.

    Read from the artifact metadata store on every call, so a change made through another
    node sharing that store shows up here at once. Until the registry is first set, the
    ``[tool.strata.transforms]`` table applies; after that, editing it has no effect.
    """
    from strata.artifact_store import get_artifact_store

    store = get_artifact_store()
    if store is not None:
        entries = store.notebook_worker_entries()
        if entries is not None:
            return _parse_managed_worker_records(entries)
    return _configured_managed_worker_records()


def _change_server_managed_worker_records(
    change: Callable[[list[ManagedWorkerRecord]], list[ManagedWorkerRecord]],
) -> list[ManagedWorkerRecord]:
    """Apply ``change`` to the registry in one store transaction; return the result.

    Without an artifact store (a scan-only service server) the registry lives in memory.
    """
    from strata.artifact_store import get_artifact_store

    store = get_artifact_store()
    if store is None:
        from strata.server import get_state

        records = change(get_server_managed_worker_records())
        get_state().config.transforms_config["notebook_workers"] = (
            _serialize_managed_worker_records(records)
        )
        return get_server_managed_worker_records()

    def apply(current: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        records = (
            _parse_managed_worker_records(current)
            if current is not None
            else _configured_managed_worker_records()
        )
        return _serialize_managed_worker_records(change(records))

    store.update_notebook_workers(apply)
    return get_server_managed_worker_records()


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

    def change(records: list[ManagedWorkerRecord]) -> list[ManagedWorkerRecord]:
        if any(existing.worker.name == record.worker.name for existing in records):
            raise ValueError(record.worker.name)
        return [*records, record]

    return _change_server_managed_worker_records(change)


def update_server_managed_worker_record(
    worker_name: str,
    record: ManagedWorkerRecord,
) -> list[ManagedWorkerRecord]:
    """Replace one service-managed worker entry in place, possibly renaming it.

    Raises:
        KeyError: If the referenced worker does not exist.
        ValueError: If the requested new worker name would collide.
    """

    def change(records: list[ManagedWorkerRecord]) -> list[ManagedWorkerRecord]:
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
            (
                index
                for index, existing in enumerate(records)
                if existing.worker.name == worker_name
            ),
            len(next_records),
        )
        next_records.insert(insert_at, record)
        return next_records

    return _change_server_managed_worker_records(change)


def import_worker_registry_file(artifact_dir: Path, store: ArtifactStore) -> None:
    """Move the registry earlier releases kept in ``notebook_workers.json`` into the store.

    Imported only while the store holds no registry, so a node starting with a stale
    copy cannot overwrite one another node set. The file is then renamed
    ``notebook_workers.json.migrated`` either way, so this runs once per file.
    """
    path = artifact_dir / "notebook_workers.json"
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(
            "Could not read the notebook worker registry at %s (%s), so it was not imported "
            "into the artifact metadata store. Fix or remove the file and restart.",
            path,
            exc,
        )
        return
    entries = _serialize_managed_worker_records(_parse_managed_worker_records(raw))
    imported = store.update_notebook_workers(lambda current: entries if current is None else None)
    migrated = path.with_name(path.name + ".migrated")
    try:
        path.replace(migrated)
    except OSError as exc:
        logger.error("Could not rename %s to %s (%s).", path, migrated, exc)
        return
    if imported is not None:
        logger.warning(
            "Imported %d notebook workers from %s into the artifact metadata store; "
            "the file is now %s and is no longer read.",
            len(entries),
            path,
            migrated,
        )
    else:
        logger.warning(
            "Did not import %s: the artifact metadata store already holds a notebook worker "
            "registry. The file is now %s and is no longer read.",
            path,
            migrated,
        )


def replace_server_managed_worker_records(
    records: list[ManagedWorkerRecord],
) -> list[ManagedWorkerRecord]:
    """Replace the configured service-mode notebook worker registry."""
    return _change_server_managed_worker_records(lambda _current: records)


def set_server_managed_worker_enabled(
    worker_name: str,
    enabled: bool,
) -> list[ManagedWorkerRecord]:
    """Enable or disable one service-managed worker by name."""

    def change(records: list[ManagedWorkerRecord]) -> list[ManagedWorkerRecord]:
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
        return next_records

    return _change_server_managed_worker_records(change)


def delete_server_managed_worker_record(worker_name: str) -> list[ManagedWorkerRecord]:
    """Delete one service-managed worker by name."""

    def change(records: list[ManagedWorkerRecord]) -> list[ManagedWorkerRecord]:
        next_records = [record for record in records if record.worker.name != worker_name]
        if len(next_records) == len(records):
            raise KeyError(worker_name)
        return next_records

    return _change_server_managed_worker_records(change)


def validate_worker_assignment(
    notebook_state: NotebookState,
    worker_name: str | None,
) -> str | None:
    """Validate a worker assignment against deployment policy; ``None`` if allowed.

    Personal mode is permissive so notebooks can name future workers; service mode
    requires a server-managed registry entry.
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

    ``None`` and ``"local"`` are the built-in local worker. Personal mode sees
    notebook-scoped definitions; service mode sees only the server-managed registry.
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

    # exclude_none keeps only set keys + extras, so the provenance hash stays stable.
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


# config.transport values an executor worker dispatches on; the last three pull by manifest.
SIGNED_TRANSPORTS = frozenset({"signed", "manifest", "build"})
WORKER_TRANSPORTS = frozenset({"direct", *SIGNED_TRANSPORTS})


def check_worker_transport(transport: str | None) -> None:
    """Refuse a ``config.transport`` the executor would not recognise.

    Raises:
        ValueError: Naming the accepted values; an unknown one would silently
            dispatch as ``direct``.
    """
    if transport is None:
        return
    if transport.strip().lower() not in WORKER_TRANSPORTS:
        raise ValueError(
            f"unknown worker transport {transport!r}; "
            f"expected one of: {', '.join(sorted(WORKER_TRANSPORTS))}"
        )


def worker_transport(worker: WorkerSpec) -> str:
    """Return a stable UI-facing transport label for one worker."""
    if worker.backend == WorkerBackendType.LOCAL:
        return "local"

    url = str(worker.config.url or "").strip()
    transport = str(worker.config.transport or "direct").strip().lower()

    if url.startswith("embedded://"):
        return "embedded"
    if transport in SIGNED_TRANSPORTS:
        return "signed"
    if url.startswith("http://") or url.startswith("https://"):
        return "direct"
    return "executor"


def build_worker_catalog(notebook_state: NotebookState) -> list[dict[str, Any]]:
    """Build a UI-facing worker catalog for a notebook.

    Always includes ``local``, the notebook-scoped definitions, and an unavailable
    entry for each referenced worker name no longer in the registry.
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
                "config": worker.config.model_dump(
                    mode="json", exclude_none=True, exclude={"token"}
                ),
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

        # After the notebook's own: add_worker is first-writer-wins, so a notebook-defined
        # name wins here as it does in effective_workers.
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


# Features a worker's /health advertises, by health URL, with read time. Dispatch
# asks on every cell, so the answer is cached for a minute.
_FEATURES_TTL_SECONDS = 60.0
# Longer than the worker's first ``/health``: on a GPU box it shells out to
# ``nvidia-smi`` with its own 10 s timeout, so a shorter probe refuses a fresh machine.
_PROBE_TIMEOUT_SECONDS = 20.0
_advertised_features: dict[str, tuple[float, dict[str, Any]]] = {}


async def worker_advertises(worker: WorkerSpec, feature: str) -> bool | None:
    """Whether *worker*'s ``/health`` lists *feature* as true.

    ``None`` means the worker could not be asked (unreachable, timed out, or not a
    health document), unlike ``False`` from an older worker without the feature;
    the caller decides what that is worth. Only answers are cached, so one timed-out
    probe does not decide for every cell dispatched in the cache window.
    """
    health_url = _health_url_for_worker(worker)
    if health_url is None:
        # Nowhere to ask, which is not the same as having asked.
        return None
    cached = _advertised_features.get(health_url)
    if cached is None or time.monotonic() - cached[0] >= _FEATURES_TTL_SECONDS:
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                response = await client.get(health_url)
        except (httpx.HTTPError, AttributeError):
            # Nothing came back. Ask again next time rather than deciding.
            return None
        if response.status_code in (404, 405, 501):
            # Answers, but has no such route: older than every feature it would list.
            return False
        if response.status_code != 200:
            # Up and unwell (gateway error, 503 while starting): not an answer about features.
            return None
        try:
            features: dict[str, Any] = (
                response.json().get("capabilities", {}).get("features", {}) or {}
            )
        except (ValueError, AttributeError):
            return None
        cached = (time.monotonic(), features)
        _advertised_features[health_url] = cached
    return cached[1].get(feature) is True


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

    The 8s default covers serverless cold starts (Modal, Fly scale-to-zero). Probe
    several workers with ``asyncio.gather`` so timeouts overlap. A GPU cold boot
    over 8s shows unavailable until the next probe cycle.
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
        # A timed-out probe usually means a serverless worker is cold-starting; surface it as
        # "warming" rather than lumping it with real failures (DNS, 5xx).
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

    # First pass: resolve static status (local, disabled) and collect entries needing a
    # live probe. Probes run in parallel so a cold-starting worker doesn't delay the rest.
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
                "config": worker.config.model_dump(
                    mode="json", exclude_none=True, exclude={"token"}
                ),
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
