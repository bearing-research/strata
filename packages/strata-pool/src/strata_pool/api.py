"""HTTP surface, for running the pool as a service.

Needs the `server` extra. The pool is usable as a library without it, and the
proxy that composes it may well bring its own framework, so a web server is
not something `import strata_pool` should pull in.

Job payloads and results are opaque bytes, so they travel as raw request and
response bodies rather than being wedged into JSON. Everything else — status,
fleet, usage — is JSON.

The caller is trusted for tenant identity. It presents the pool's API token
and asserts a tenant in a header; the pool does not authenticate end users and
has no idea who they are. That is the same trusted-proxy model the Strata
server uses, and it means the pool must never be reachable from anywhere but
the proxy.
"""

import logging
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from strata_pool.pool import Pool
from strata_pool.types import Job, JobState, UsageEvent, Worker

logger = logging.getLogger(__name__)

TENANT_HEADER = "X-Strata-Tenant"


def _job_json(job: Job) -> dict:
    """A job without its payload or result, which are bytes and often large."""
    fields = asdict(job)
    fields.pop("payload")
    fields.pop("result")
    fields["has_result"] = job.result is not None
    return fields


def _worker_json(worker: Worker) -> dict:
    """A machine without its credential.

    Built by hand rather than from asdict: `repr=False` keeps the token out of
    logs but not out of a dict, and this response is the one place it would
    otherwise be handed to whoever asked.
    """
    fields = asdict(worker)
    fields.pop("auth_token")
    return fields


def _usage_json(event: UsageEvent) -> dict:
    return asdict(event)


def create_app(
    pool: Pool,
    *,
    api_token: str | None = None,
    scaler_interval_seconds: float = 10.0,
) -> FastAPI:
    """Build the pool's HTTP app.

    Args:
        api_token: Bearer token every route except `/health` requires. None
            disables the check, which is a local-development choice: an open
            submit endpoint runs arbitrary payloads on machines you pay for.
        scaler_interval_seconds: How often idle machines are reaped. The app
            starts the scaler itself, so a deployment cannot forget to.
    """
    if api_token is None:
        logger.warning("pool API starting with no token; anyone who can reach it can run jobs")

    async def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if api_token is None:
            return
        if authorization != f"Bearer {api_token}":
            raise HTTPException(status_code=401, detail="invalid or missing API token")

    async def tenant(request: Request) -> str:
        value = request.headers.get(TENANT_HEADER)
        if not value:
            raise HTTPException(status_code=400, detail=f"{TENANT_HEADER} is required")
        return value

    async def lifespan(app: FastAPI):
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

        What the notebook's executor protocol wants, since it expects one
        synchronous response. `wait_seconds` bounds how long the caller waits,
        not how long the job may run — a job that outlives it keeps going and
        can be collected by ID.
        """
        job = await _submit(
            pool,
            tenant_id=tenant_id,
            machine_type=machine_type,
            payload=await request.body(),
            priority=priority,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
        )
        try:
            done = await pool.wait(job.id, timeout=wait_seconds)
        except TimeoutError:
            # 202: it is still running, and the ID is how you find it. Re-read
            # for the freshest state, falling back to the submitted snapshot
            # rather than pretending a row we just wrote could be missing.
            latest = pool.store.get_job(job.id) or job
            return JSONResponse(_job_json(latest), status_code=202)
        return _terminal_response(done)

    @app.get("/v1/jobs/{job_id}", dependencies=guard)
    async def get_job(job_id: str) -> dict:
        job = pool.store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        return _job_json(job)

    @app.get("/v1/jobs/{job_id}/result", dependencies=guard)
    async def get_job_result(job_id: str) -> Response:
        """The raw result bytes, once there are any."""
        job = pool.store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        if job.state not in (JobState.COMPLETED, JobState.FAILED, JobState.TIMED_OUT):
            raise HTTPException(status_code=409, detail=f"job is {job.state.value}")
        return _terminal_response(job)

    @app.get("/v1/machine-types", dependencies=guard)
    async def list_machine_types() -> list[dict]:
        """What a caller may ask for. The catalogue an annotation resolves against."""
        return [asdict(spec) for spec in pool.machine_types.values()]

    @app.get("/v1/workers", dependencies=guard)
    async def list_workers() -> list[dict]:
        return [_worker_json(worker) for worker in pool.store.list_workers()]

    @app.get("/v1/usage", dependencies=guard)
    async def list_usage(tenant_id: str | None = Query(default=None)) -> list[dict]:
        """The billing feed. One event per terminal job, monotonic duration."""
        return [_usage_json(event) for event in pool.store.list_usage(tenant_id)]

    return app


async def _submit(pool: Pool, **kwargs) -> Job:
    try:
        return await pool.submit(**kwargs)
    except ValueError as exc:
        # An unknown machine type is the caller's mistake, not a pool failure.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _terminal_response(job: Job) -> Response:
    """Map a finished job onto a status code.

    A failure on the worker is reported as a failure of the job, not of the
    pool: the caller needs to tell "your code raised" from "we could not run
    it", and a 500 would blur the two.
    """
    if job.state is JobState.COMPLETED:
        return Response(content=job.result or b"", media_type="application/octet-stream")
    status = 504 if job.state is JobState.TIMED_OUT else 502
    return JSONResponse({"state": job.state.value, "error": job.error}, status_code=status)
