"""Materialize-plane routes: unified ``/v1/materialize``, transform materialize and its dry run.

Runtime state (stream registry, scan builds, QoS) is reached through ``get_state()`` and
server-private gates through lazy ``from strata.server import ...``, so this module stays a leaf.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pyiceberg.exceptions import NoSuchTableError

from strata.api.dependencies import (
    CurrentPrincipal,
    ReadStore,
    authorize_table_access,
    resolve_input_version,
    table_identity_or_400,
)
from strata.iceberg import CatalogUriRequired, SnapshotNotFound, WarehouseNotFound
from strata.logging import get_logger
from strata.planner import ColumnNotFound, UnsupportedTableFormatError
from strata.pool_metrics import get_pool_tracker
from strata.streaming import StreamState
from strata.tracing import trace_span
from strata.types import (
    BuildSpec,
    ExplainMaterializeRequest,
    ExplainMaterializeResponse,
    IdentityParams,
    MaterializeRequest,
    MaterializeResponse,
)

if TYPE_CHECKING:
    from strata.server import ServerState

logger = get_logger(__name__)

router = APIRouter(tags=["materialize"])


@router.post("/v1/artifacts/explain-materialize", response_model=ExplainMaterializeResponse)
async def explain_materialize(
    request: ExplainMaterializeRequest, store: ReadStore, principal: CurrentPrincipal
):
    """Dry-run materialize: report hit or miss and why a rebuild would be needed."""
    from strata.services.materialize import materialize_service

    tenant_id = principal.tenant if principal else None

    # A per-input 400/404 becomes an error marker so the dry run still returns
    # a full picture.
    resolved_versions: dict[str, str] = {}
    for input_uri in request.inputs:
        try:
            resolved_versions[input_uri] = resolve_input_version(input_uri, tenant=tenant_id)
        except HTTPException as e:
            resolved_versions[input_uri] = f"<error: {e.detail}>"

    return materialize_service.explain(
        store, request=request, tenant=tenant_id, resolved_versions=resolved_versions
    )


def _estimate_transform_output_bytes(
    state: ServerState,
    max_output_bytes: int | None,
) -> int:
    """Choose the best available output-size estimate for build admission."""
    if max_output_bytes is not None and max_output_bytes > 0:
        return max_output_bytes
    return state.config.build_runner_default_max_output


def _validate_transform_allowed(executor_ref: str, principal=None):
    """Validate a transform against the server-mode registry and return its definition.

    Returns None in personal mode, where every transform is allowed; raises 403 when the
    transform is not allowed in server mode.
    """
    from strata.server import get_state
    from strata.transforms.registry import get_transform_registry

    state = get_state()

    # Personal mode: an executor the embedded runner can't resolve would sit
    # in 'building' forever, so fail fast.
    if state.config.writes_enabled:
        registry = get_transform_registry()
        defn = registry.get(executor_ref)
        if defn is None:
            available = sorted(d.ref for d in registry.definitions)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "transform_unknown",
                    "message": f"Transform '{executor_ref}' is not registered on "
                    "this server — nothing can execute it. "
                    f"Available transforms: {available}.",
                    "executor": executor_ref,
                },
            )
        return defn

    if state.config.server_transforms_enabled:
        registry = get_transform_registry()
        defn = registry.get(executor_ref)

        if defn is None:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "transform_not_allowed",
                    "message": f"Transform '{executor_ref}' is not registered. "
                    "Contact your administrator to add it to the allowlist.",
                    "executor": executor_ref,
                },
            )

        if (
            defn.requires_scope
            and state.config.principal_auth_enabled
            and not (principal and principal.has_scope(defn.requires_scope))
        ):
            if principal is None:
                raise HTTPException(status_code=401, detail="Unauthorized")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "insufficient_scope",
                    "message": (
                        f"Transform '{executor_ref}' requires scope '{defn.requires_scope}'."
                    ),
                    "required_scope": defn.requires_scope,
                    "executor": executor_ref,
                },
            )

        return defn

    raise HTTPException(
        status_code=403,
        detail={
            "error": "writes_disabled",
            "message": "Artifact endpoints are disabled.",
        },
    )


def _authorize_name_write() -> None:
    """Gate ``name`` on a materialize request with the ``POST /v1/names`` write gate.

    The name is set on a cache hit and again when a miss finalizes, so refuse it at admission.
    """
    from strata.server import _authorize_artifact_write, _get_artifact_store

    _get_artifact_store(allow_write=True)
    _authorize_artifact_write()


@router.post("/v1/artifacts/materialize", response_model=MaterializeResponse)
async def materialize_artifact(request: MaterializeRequest):
    """Materialize a computed artifact: the cached one on a provenance hit, else a build spec.

    In server mode the transform must be on the allowlist. On a miss, a ``building`` artifact
    is created with the resolved input versions.
    """
    import uuid

    from strata.artifact_store import TransformSpec
    from strata.auth import get_principal
    from strata.server import _get_artifact_store, _retry_after_header, get_state

    principal = get_principal()
    tenant_id = principal.tenant if principal else None
    principal_id = principal.id if principal else None

    transform = request.transform
    executor_ref = transform.executor

    transform_defn = _validate_transform_allowed(executor_ref, principal=principal)
    if request.name:
        _authorize_name_write()

    store = _get_artifact_store(allow_server_mode=True)

    transform_spec = TransformSpec(
        executor=executor_ref,
        params=transform.params,
        inputs=request.inputs,
    )

    # Versions feed both the hash and staleness tracking. An input that does not resolve
    # (denied, missing, unreadable, or a plan that failed) refuses the request: building past
    # it would record a snapshot-less version and dedup later runs onto the result.
    input_versions: dict[str, str] = {
        input_uri: resolve_input_version(input_uri, tenant=tenant_id)
        for input_uri in request.inputs
    }

    from strata.services.materialize import materialize_service

    provenance_hash = materialize_service.compute_provenance(transform_spec, input_versions)

    existing = store.find_by_provenance(provenance_hash, tenant=tenant_id)
    if existing is not None and not request.refresh:
        artifact_uri = f"strata://artifact/{existing.id}@v={existing.version}"

        if request.name:
            store.set_name(request.name, existing.id, existing.version, tenant=tenant_id)

        return MaterializeResponse(
            hit=True,
            artifact_uri=artifact_uri,
            build_spec=None,
            state="ready",
        )

    # A refresh rebuild reuses the existing id; see rebuild_artifact_id.
    new_id = str(uuid.uuid4())
    artifact_id = materialize_service.rebuild_artifact_id(
        existing, refresh=request.refresh, new_id=new_id
    )
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance_hash,
        transform_spec=transform_spec,
        input_versions=input_versions,
        tenant=tenant_id,
        principal=principal_id,
        minted=artifact_id == new_id,
    )

    artifact_uri = f"strata://artifact/{artifact_id}@v={version}"
    state = get_state()

    # Queue a build record for the build runner.
    if state.config.transforms_runtime_enabled:
        from strata.artifact_store import get_artifact_store
        from strata.transforms.build_qos import (
            BuildQoSError,
            get_build_qos,
            normalized_build_qos_tenant_id,
        )
        from strata.transforms.build_store import get_build_store

        build_id = str(uuid.uuid4())

        if state.config.artifact_dir is None:
            raise HTTPException(status_code=500, detail="Artifact directory not configured")
        artifact_store = get_artifact_store(state.config.artifact_dir)
        build_store = get_build_store(
            state.config.artifact_dir / "artifacts.sqlite",
            dialect=artifact_store.dialect if artifact_store else None,
        )
        if build_store is None:
            raise HTTPException(
                status_code=500,
                detail="Build store not initialized",
            )

        build_tenant_id = normalized_build_qos_tenant_id(tenant_id)
        estimated_output_bytes = _estimate_transform_output_bytes(
            state,
            transform_defn.max_output_bytes if transform_defn is not None else None,
        )

        # Admission and quotas run before the build record exists.
        build_qos = get_build_qos()
        build_slot = None

        if build_qos is not None:
            priority = build_qos.classify_build(
                estimated_output_bytes=estimated_output_bytes,
                input_count=len(request.inputs),
            )

            try:
                await build_qos.check_quota(build_tenant_id, estimated_output_bytes)
            except BuildQoSError as e:
                store.fail_artifact(artifact_id, version)
                return JSONResponse(
                    status_code=e.status_code,
                    content=e.to_dict(),
                    headers={"Retry-After": _retry_after_header(e.retry_after, 5.0)},
                )

            try:
                build_slot = await build_qos.acquire(build_tenant_id, priority)
            except BuildQoSError as e:
                store.fail_artifact(artifact_id, version)
                return JSONResponse(
                    status_code=e.status_code,
                    content=e.to_dict(),
                    headers={"Retry-After": _retry_after_header(e.retry_after, 5.0)},
                )

        try:
            build_store.create_build(
                build_id=build_id,
                artifact_id=artifact_id,
                version=version,
                executor_ref=executor_ref,
                executor_url=transform_defn.executor_url if transform_defn else None,
                tenant_id=tenant_id,
                principal_id=principal_id,
                input_uris=request.inputs,
                params=transform.params,
                name=request.name,
            )

            # Queued now; the runner has its own execution concurrency control.
            if build_slot:
                await build_slot.release()

            return MaterializeResponse(
                hit=False,
                artifact_uri=artifact_uri,
                build_id=build_id,
                state="pending",
            )
        except Exception:
            if build_slot:
                await build_slot.release()
            raise

    # No build runtime: the client executes the build spec.
    build_spec = BuildSpec(
        artifact_id=artifact_id,
        version=version,
        executor=transform_spec.executor,
        params=transform_spec.params,
        input_uris=request.inputs,
    )

    return MaterializeResponse(
        hit=False,
        artifact_uri=artifact_uri,
        build_spec=build_spec.model_dump(),
        state="building",
    )


# =============================================================================
# Unified materialize: an Iceberg scan is a materialize with the scan@v1 transform.
# =============================================================================


@router.post("/v1/materialize", response_model=MaterializeResponse)
async def unified_materialize(request: MaterializeRequest):
    """Materialize data: the single entry point, with table scans expressed as ``scan@v1``.

    ``stream`` mode (default) streams data while the artifact builds; ``artifact`` mode
    returns a build to poll at ``/v1/builds/{build_id}``. Both persist the artifact.
    """
    from strata.server import get_state

    state = get_state()
    transform = request.transform

    if state._draining:
        raise HTTPException(
            status_code=503,
            detail="Server is shutting down. Not accepting new requests.",
        )

    if transform.executor == "scan@v1":
        return await _handle_identity_materialize(request)

    return await _handle_transform_materialize(request)


async def _handle_identity_materialize(
    request: MaterializeRequest,
) -> MaterializeResponse | JSONResponse:
    """Handle ``scan@v1`` in-process: plan one table scan, return a cache hit or start a stream.

    On a miss creates the artifact record and stream state, then returns ``stream_url`` or
    ``build_id`` by mode.
    """
    import uuid

    from strata.artifact_store import TransformSpec as ArtifactTransformSpec
    from strata.artifact_store import get_artifact_store
    from strata.auth import get_principal
    from strata.server import _retry_after_header, get_state

    state = get_state()

    principal = get_principal()
    tenant_id = principal.tenant if principal else None
    principal_id = principal.id if principal else None

    if len(request.inputs) != 1:
        raise HTTPException(
            status_code=400,
            detail="scan@v1 transform requires exactly one input",
        )

    table_uri = request.inputs[0]

    if table_uri.startswith("strata://"):
        raise HTTPException(
            status_code=400,
            detail="scan@v1 transform input must be a table URI, not an artifact",
        )

    # model_validate so ty doesn't narrow each dict[str, object] value.
    try:
        identity_params = IdentityParams.model_validate(request.transform.params)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scan@v1 params: {e}",
        )

    filters = identity_params.to_strata_filters()

    if request.name:
        _authorize_name_write()

    # Authorize before planning: the 400/413 planning errors would tell a denied
    # caller the table exists and its size, and planning costs manifest reads.
    # The identity comes from the URI so a refused request does no manifest work.
    uri_identity = table_identity_or_400(table_uri)
    if state.config.principal_auth_enabled:
        if principal is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        authorize_table_access(table_uri, uri_identity)

    plan_timeout = state.config.plan_timeout_seconds

    def do_plan():
        with trace_span(
            "plan_identity_materialize",
            table_uri=table_uri,
            snapshot_id=identity_params.snapshot_id,
            columns_count=len(identity_params.columns) if identity_params.columns else None,
        ) as span:
            plan = state.planner.plan(
                table_uri=table_uri,
                snapshot_id=identity_params.snapshot_id,
                columns=identity_params.columns,
                filters=filters,
            )
            span.set_attribute("scan_id", plan.scan_id)
            span.set_attribute("row_groups_total", plan.total_row_groups)
            span.set_attribute("row_groups_pruned", plan.pruned_row_groups)
            span.set_attribute("estimated_bytes", plan.estimated_bytes)
            return plan

    try:
        with get_pool_tracker().track("planning"):
            plan = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(state._planning_executor, do_plan),
                timeout=plan_timeout,
            )
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Planning timed out after {plan_timeout}s.",
        )
    except UnsupportedTableFormatError as e:
        # A table Strata will not read (an unreadable delete file, too many
        # pending equality deletes): the message says why and what to do.
        raise HTTPException(status_code=422, detail=str(e)) from e
    except (CatalogUriRequired, ColumnNotFound) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (WarehouseNotFound, SnapshotNotFound) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except NoSuchTableError as e:
        raise HTTPException(status_code=404, detail=f"Table not found: {table_uri}") from e

    max_tasks = state.config.max_tasks_per_scan
    if len(plan.tasks) > max_tasks:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Query would read {len(plan.tasks)} row groups, exceeding limit of {max_tasks}."
            ),
        )

    max_response = state.config.max_response_bytes
    if plan.estimated_bytes > max_response:
        state.metrics.record_stream_abort_size()
        raise HTTPException(
            status_code=413,
            detail=f"Estimated response size ({plan.estimated_bytes:,} bytes) exceeds limit.",
        )

    if state.config.principal_auth_enabled:
        if principal is None:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Re-check against the identity the catalog actually resolved, which
        # may differ from the one parsed from the URI.
        authorize_table_access(table_uri, plan.table_identity)

        plan.owner_principal = principal.id
        plan.owner_tenant = principal.tenant

    from strata.services.materialize import materialize_service, table_input_version

    provenance_hash = materialize_service.compute_identity_provenance(
        table_identity=str(plan.table_identity),
        snapshot_id=plan.snapshot_id,
        columns=identity_params.columns,
        filters=filters,
        schema_id=plan.schema_id,
    )
    store = get_artifact_store(state.config.artifact_dir)

    # The form name-status compares against, so a named scan goes stale on a
    # schema change too.
    input_versions = {table_uri: table_input_version(plan)}

    existing = None
    if store is not None:
        existing = store.find_by_provenance(provenance_hash, tenant=tenant_id)
        if existing is not None and existing.state == "ready" and not request.refresh:
            artifact_uri = f"strata://artifact/{existing.id}@v={existing.version}"

            if request.name:
                store.set_name(request.name, existing.id, existing.version, tenant=tenant_id)

            logger.info(
                "identity_materialize_cache_hit",
                artifact_id=existing.id,
                table_uri=table_uri,
                snapshot_id=plan.snapshot_id,
            )

            # Lets clients fetch a hit the same way as a miss.
            stream_url = f"/v1/artifacts/{existing.id}/v/{existing.version}/data"

            return MaterializeResponse(
                hit=True,
                artifact_uri=artifact_uri,
                state="ready",
                stream_url=stream_url if request.mode == "stream" else None,
            )

    # A refresh rebuild reuses the existing artifact id (see rebuild_artifact_id)
    # so finalize supersedes the old version. Its stream id is fresh, since older
    # streams for that artifact id may linger; a plain miss reuses the artifact id.
    new_id = str(uuid.uuid4())
    artifact_id = materialize_service.rebuild_artifact_id(
        existing, refresh=request.refresh, new_id=new_id
    )
    stream_id = str(uuid.uuid4()) if (request.refresh and existing is not None) else artifact_id

    artifact_version = 1
    if store is not None:
        transform_spec = ArtifactTransformSpec(
            executor="scan@v1",
            params=request.transform.params,
            inputs=request.inputs,
        )
        artifact_version = store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=provenance_hash,
            transform_spec=transform_spec,
            input_versions=input_versions,
            tenant=tenant_id,
            principal=principal_id,
            minted=artifact_id == new_id,
        )

    artifact_uri = f"strata://artifact/{artifact_id}@v={artifact_version}"

    stream_state = StreamState(
        stream_id=stream_id,
        plan=plan,
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        created_at=time.time(),
        mode=request.mode,
        name=request.name,
        tenant=tenant_id,
    )

    logger.info(
        "identity_materialize_cache_miss",
        artifact_id=artifact_id,
        stream_id=stream_id,
        table_uri=table_uri,
        snapshot_id=plan.snapshot_id,
        estimated_bytes=plan.estimated_bytes,
        mode=request.mode,
    )

    if request.mode == "stream":
        state.streams.register(stream_state)
        # Register for QoS/accounting only when a client will actually stream.
        state.scan_builds.register_scan(plan)
        state.scan_builds.start_prefetch(state, plan)
        state.streams.schedule_cleanup(stream_id, plan.scan_id)

        return MaterializeResponse(
            hit=False,
            artifact_uri=artifact_uri,
            state="building",
            stream_id=stream_id,
            stream_url=f"/v1/streams/{stream_id}",
        )
    else:
        # Without a store the background build no-ops and the build_id never
        # resolves, so reject rather than hang the client.
        if store is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "mode='artifact' requires an artifact store, but this deployment "
                    "has no artifact_dir configured. Use mode='stream' to scan "
                    "without persistence, or set artifact_dir."
                ),
            )

        from strata.transforms.build_qos import (
            BuildQoSError,
            get_build_qos,
            normalized_build_qos_tenant_id,
        )

        build_qos = get_build_qos()
        if build_qos is not None:
            qos_tenant_id = normalized_build_qos_tenant_id(tenant_id)
            estimated_output_bytes = max(0, plan.estimated_bytes)
            priority = build_qos.classify_build(
                estimated_output_bytes=estimated_output_bytes,
                input_count=len(plan.tasks),
            )

            try:
                await build_qos.check_quota(qos_tenant_id, estimated_output_bytes)
                stream_state.build_slot = await build_qos.acquire(qos_tenant_id, priority)
                stream_state.qos_tenant_id = qos_tenant_id
            except BuildQoSError as e:
                if store is not None:
                    store.fail_artifact(artifact_id, artifact_version)
                return JSONResponse(
                    status_code=e.status_code,
                    content=e.to_dict(),
                    headers={"Retry-After": _retry_after_header(e.retry_after, 5.0)},
                )

        state.streams.register(stream_state)
        stream_state.background_task = asyncio.create_task(
            state.scan_builds.build_identity_artifact(state, stream_state)
        )
        return MaterializeResponse(
            hit=False,
            artifact_uri=artifact_uri,
            state="pending",
            build_id=stream_id,
        )


async def _handle_transform_materialize(request: MaterializeRequest) -> MaterializeResponse:
    """Handle non-scan transforms via the ``/v1/artifacts/materialize`` flow."""
    return await materialize_artifact(request)
