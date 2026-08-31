# strata-pool

Worker pool for dispatching Strata jobs to ephemeral machines. Not published
yet — it ships with the hosted workers service.

Jobs arrive with a machine type. The pool hands each one to a warm worker of
that type, starts a machine when there is none, forwards the payload over
HTTP, records the result, and meters the execution. Backends (local Docker,
Fly, EC2) provide start / stop / health; the pool does not know which it is
talking to.

```python
from strata_pool import DockerBackend, MachineType, Pool, PoolStore

pool = Pool(
    store=PoolStore("pool.sqlite"),
    backend=DockerBackend(),
    machine_types=[MachineType(name="cpu-4x", image="strata-worker:latest")],
)
await pool.recover()          # reconcile after a restart

job = await pool.submit(tenant_id="acme", machine_type="cpu-4x", payload=bundle)
done = await pool.wait(job.id)
```

## What it is not

**It is not a cache.** The pool has no idea Strata deduplicates work.
Submitting a job whose result already exists boots a machine and recomputes
it — so the caller checks `find_by_provenance` *before* submitting. Getting
that order wrong bills customers for cache hits.

**It is not the metering layer.** The pool records one `UsageEvent` per
terminal job, including failures, with a monotonic duration. What is billable,
and at what price, is decided above it.

**It does not scale down yet.** A worker stays warm once booted. Cool-down, a
warm floor, and reaping stuck jobs are the scaler and the reaper, and they
land as their own change. See the module docstring in `pool.py` for the two
behaviours that follow from having no timers.

## Layout

| Module | What lives there |
|---|---|
| `types.py` | `Worker`, `Job`, `MachineType`, `UsageEvent` and their states |
| `backend.py` | The `Backend` protocol — start / stop / health |
| `backends/docker.py` | Containers on the local Docker daemon |
| `store.py` | SQLite persistence; the pool process keeps no authoritative state |
| `pool.py` | Submission, dispatch, boot, execution, metering, restart recovery |

## The worker contract

The pool is image-agnostic, but an image has to hold up four things:

| | |
|---|---|
| Listen on the worker port | 8080 by default; the backend publishes it |
| `GET /health` → 200 when ready | No auth. It is polled before the machine is trusted with anything, and it reveals nothing |
| `POST /execute` → 200 with the result body | The request body is the job payload, opaque to the pool |
| Require `Authorization: Bearer $STRATA_WORKER_TOKEN` on `/execute` | Reject anything else with 401 |

The token is minted per machine before it boots and passed in its
environment. Without that check, `/execute` is an unauthenticated
remote-code-execution endpoint — survivable only while the machine is bound to
loopback, which stops being true the moment a backend hands out a routable
address.

## The Docker backend

Talks to the Docker Engine API over its UNIX socket with httpx, so the pool
needs no Docker SDK and every request shape is testable without a daemon. It
starts the image, publishes the worker port on loopback only, and reports the
host port Docker picked. It does not care what runs inside the container.

It does not pull. An image that is not on the host fails with the daemon's own
"No such image" message.

## Tests

```bash
pytest packages/strata-pool/tests -v
```

The tests in `test_docker_live.py` run the whole path against real containers
and skip when no daemon is reachable. Set `STRATA_POOL_DOCKER_SOCKET` if yours
is not at `/var/run/docker.sock` (Docker Desktop on macOS puts it in
`~/.docker/run/docker.sock`), and `STRATA_POOL_REQUIRE_DOCKER=1` to make a
missing daemon an error instead of a skip — CI sets that, so a runner whose
socket moved fails loudly rather than reporting coverage it never ran.

CI additionally installs the package into a venv with only its own
dependencies, to keep it from quietly growing a dependency on the server.
