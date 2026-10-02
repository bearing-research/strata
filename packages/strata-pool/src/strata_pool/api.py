"""HTTP surface for running the pool as a service (needs the `server` extra).

Job payloads and results travel as raw bodies; everything else is JSON. The
caller presents the API token and asserts the tenant in a header, which the
pool trusts: it must be reachable only from the proxy.
"""

import logging
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from strata_pool.pool import Pool
from strata_pool.types import Job, JobState, MachineType, UsageEvent, Worker

logger = logging.getLogger(__name__)

TENANT_HEADER = "X-Strata-Tenant"
REDACTED = "<redacted>"


def _job_json(job: Job) -> dict:
    """A job without its payload or result, which are bytes and often large."""
    fields = asdict(job)
    fields.pop("payload")
    fields.pop("result")
    fields["has_result"] = job.result is not None
    return fields


def _worker_json(worker: Worker) -> dict:
    """A machine without its credential; `repr=False` does not keep it out of `asdict`."""
    fields = asdict(worker)
    fields.pop("auth_token")
    return fields


def _machine_type_json(spec: MachineType, *, full: bool) -> dict:
    """A catalogue entry; without `full`, operator settings keep their names but not values."""
    fields = asdict(spec)
    if not full:
        # Both commonly carry credentials (a registry login, a model-hub token).
        fields["env"] = dict.fromkeys(spec.env, REDACTED)
        fields["provider_options"] = dict.fromkeys(spec.provider_options, REDACTED)
    return fields


def _usage_json(event: UsageEvent) -> dict:
    return asdict(event)


def create_app(
    pool: Pool,
    *,
    api_token: str | None = None,
    admin_token: str | None = None,
    scaler_interval_seconds: float = 10.0,
) -> FastAPI:
    """Build the pool's HTTP app; its lifespan recovers the fleet and starts the scaler.

    Args:
        api_token: Bearer token every route except `/health` requires. None
            disables the check (local development only: anyone could run jobs).
        admin_token: Bearer token for the operator, also accepted wherever
            `api_token` is. Replacing the machine-type catalogue needs it. None
            refuses catalogue writes over HTTP: the operator sets the catalogue
            in the code that constructs the pool.
        scaler_interval_seconds: How often idle machines are reaped.
    """
    if api_token is None:
        logger.warning("pool API starting with no token; anyone who can reach it can run jobs")

    def is_admin(authorization: str | None) -> bool:
        return admin_token is not None and authorization == f"Bearer {admin_token}"

    async def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if api_token is None or is_admin(authorization):
            return
        if authorization != f"Bearer {api_token}":
            raise HTTPException(status_code=401, detail="invalid or missing API token")

    async def require_admin(authorization: Annotated[str | None, Header()] = None) -> None:
        if admin_token is None:
            raise HTTPException(
                status_code=403,
                detail="no admin token is configured; the catalogue is set where the pool is built",
            )
        if not is_admin(authorization):
            raise HTTPException(status_code=403, detail="this route needs the admin token")

    async def tenant(request: Request) -> str:
        value = request.headers.get(TENANT_HEADER)
        if not value:
            raise HTTPException(status_code=400, detail=f"{TENANT_HEADER} is required")
        return value

    async def lifespan(app: FastAPI):
        # The catalogue last set over the API outlives the process that set
        # it; the one this process was constructed with is only the default.
        saved = pool.store.load_machine_types()
        if saved is not None:
            await pool.replace_machine_types(saved)
        # Reconcile before serving: machines from a previous process are
        # either still reachable or still billing.
        await pool.recover()
        pool.start_scaler(scaler_interval_seconds)
        yield
        await pool.aclose()

    app = FastAPI(title="strata-pool", lifespan=lifespan)
    guard = [Depends(require_token)]

    @app.get("/health")
    async def health() -> dict:
        """Deliberately outside the token check, for load balancers."""
        workers = pool.store.list_workers()
        counts: dict[str, int] = {}
        for worker in workers:
            counts[worker.state.value] = counts.get(worker.state.value, 0) + 1
        return {"status": "ok", "workers": counts, "machine_types": list(pool.machine_types)}

    @app.post("/v1/jobs", status_code=202, dependencies=guard)
    async def submit_job(
        request: Request,
        machine_type: str,
        tenant_id: Annotated[str, Depends(tenant)],
        priority: int = 0,
        session_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> JSONResponse:
        """Queue a job. The request body is the payload, verbatim."""
        job = await _submit(
            pool,
            tenant_id=tenant_id,
            machine_type=machine_type,
            payload=await request.body(),
            priority=priority,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
            trace_context=_trace_context(request),
        )
        return JSONResponse(_job_json(job), status_code=202)

    @app.post("/v1/jobs/sync", dependencies=guard)
    async def submit_and_wait(
        request: Request,
        machine_type: str,
        tenant_id: Annotated[str, Depends(tenant)],
        priority: int = 0,
        session_id: str | None = None,
        timeout_seconds: float | None = None,
        wait_seconds: float = 300.0,
    ) -> Response:
        """Queue a job and block until it finishes.

        `wait_seconds` bounds the wait, not the job: past it the response is a
        202 with the job ID, and the job keeps running.
        """
        job = await _submit(
            pool,
            tenant_id=tenant_id,
            machine_type=machine_type,
            payload=await request.body(),
            priority=priority,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
            trace_context=_trace_context(request),
        )
        try:
            done = await pool.wait(job.id, timeout=wait_seconds)
        except TimeoutError:
            # 202: still running, and the ID is how you find it. Re-read for the freshest
            # state; fall back to the submitted snapshot.
            latest = pool.store.get_job(job.id) or job
            return JSONResponse(_job_json(latest), status_code=202)
        return _terminal_response(done)

    def tenant_job(job_id: str, tenant_id: str) -> Job:
        job = pool.store.get_job(job_id)
        # Another tenant's job is a 404, not a 403, so an id does not confirm it exists.
        if job is None or job.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        return job

    @app.get("/v1/jobs/{job_id}", dependencies=guard)
    async def get_job(job_id: str, tenant_id: Annotated[str, Depends(tenant)]) -> dict:
        return _job_json(tenant_job(job_id, tenant_id))

    @app.get("/v1/jobs/{job_id}/result", dependencies=guard)
    async def get_job_result(job_id: str, tenant_id: Annotated[str, Depends(tenant)]) -> Response:
        """The raw result bytes, once there are any."""
        job = tenant_job(job_id, tenant_id)
        if job.state not in (JobState.COMPLETED, JobState.FAILED, JobState.TIMED_OUT):
            raise HTTPException(status_code=409, detail=f"job is {job.state.value}")
        return _terminal_response(job)

    @app.get("/v1/machine-types", dependencies=guard)
    async def list_machine_types(
        authorization: Annotated[str | None, Header()] = None,
    ) -> list[dict]:
        """What a caller may ask for. The catalogue an annotation resolves against."""
        full = is_admin(authorization)
        return [_machine_type_json(spec, full=full) for spec in pool.machine_types.values()]

    # A catalogue entry decides which image receives a tenant's jobs and
    # their signed URLs, so rewriting it is the operator's call alone.
    @app.put("/v1/machine-types", dependencies=[*guard, Depends(require_admin)])
    async def replace_machine_types(request: Request) -> list[dict]:
        """Replace the whole machine-type catalogue, without a restart.

        The body is the full list in the shape `GET` returns; types left out are
        removed. The catalogue is persisted and survives a restart.
        """
        body = await request.json()
        if not isinstance(body, list):
            raise HTTPException(status_code=400, detail="expected a list of machine types")
        try:
            specs = [MachineType(**entry) for entry in body]
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid machine type: {exc}") from exc
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise HTTPException(status_code=400, detail="machine type names must be unique")
        # Stored first: a catalogue that applied but was not saved would
        # silently revert at the next restart.
        pool.store.save_machine_types(specs)
        await pool.replace_machine_types(specs)
        return [asdict(spec) for spec in pool.machine_types.values()]

    @app.get("/v1/workers", dependencies=guard)
    async def list_workers() -> list[dict]:
        return [_worker_json(worker) for worker in pool.store.list_workers()]

    @app.get("/v1/usage", dependencies=guard)
    async def list_usage(tenant_id: str | None = Query(default=None)) -> list[dict]:
        """The billing feed. One event per terminal job, monotonic duration."""
        return [_usage_json(event) for event in pool.store.list_usage(tenant_id)]

    return app


def _trace_context(request: Request) -> dict[str, str]:
    """The caller's W3C trace headers, carried with the job to the machine."""
    return {
        key: value for key in ("traceparent", "tracestate") if (value := request.headers.get(key))
    }


async def _submit(pool: Pool, **kwargs) -> Job:
    try:
        return await pool.submit(**kwargs)
    except ValueError as exc:
        # An unknown machine type is the caller's mistake, not a pool failure.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _terminal_response(job: Job) -> Response:
    """Map a finished job onto a status code: 200, 502 for a failed job, 504 for a timeout.

    Never 500: the caller must tell "your code raised" from "the pool could not run it".
    """
    if job.state is JobState.COMPLETED:
        return Response(content=job.result or b"", media_type="application/octet-stream")
    status = 504 if job.state is JobState.TIMED_OUT else 502
    return JSONResponse({"state": job.state.value, "error": job.error}, status_code=status)
