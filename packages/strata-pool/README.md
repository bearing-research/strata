# strata-pool

Worker pool for dispatching Strata jobs to ephemeral machines. This is the
bring-your-own-hardware path: it manages machines *you* own. It is complete and
tested rather than growing — if you want a provider to autoscale for you,
register a serverless executor as a Strata worker instead.

```bash
pip install strata-pool            # library
pip install "strata-pool[server]"  # plus the HTTP service
```

Jobs arrive with a machine type. The pool hands each one to a warm worker of
that type, starts a machine when there is none, forwards the payload over
HTTP, records the result, and meters the execution. A backend provides start /
stop / health and nothing else, so the pool does not know which it is talking
to. Two ship: local Docker and RunPod. Anything satisfying the `Backend`
protocol works.

```python
from strata_pool import DockerBackend, MachineType, Pool, PoolStore

pool = Pool(
    store=PoolStore("pool.sqlite"),
    backend=DockerBackend(),
    machine_types=[MachineType(name="cpu-4x", image="strata-worker:latest")],
)
await pool.recover()          # reconcile after a restart
pool.start_scaler()           # stop paying for machines that finished

job = await pool.submit(tenant_id="acme", machine_type="cpu-4x", payload=bundle)
done = await pool.wait(job.id)
```

## Isolation

A machine belongs to **one tenant for its life** and is destroyed rather than
handed to another. Even scrubbed of files, a process that ran one tenant's
code is not a boundary the next tenant should have to trust, and GPU memory is
not reliably zeroed between processes at all. `max_workers` is therefore a
per-tenant cap.

`MachineType.cpus` and `memory_mb` bound what a container may consume; unset,
it may take the whole host. Both are enforced by the daemon, asserted against
a real one in `test_docker_live.py`.

What that buys is process-level isolation, which is **not** a boundary for
untrusted code — a shared kernel is one CVE away from a cross-tenant escape.
A deployment running untrusted work wants a VM-backed runtime (Kata, gVisor)
or a backend whose machines are already microVMs. The `Backend` protocol is
where that choice lives.

`Pool(max_workers_total=...)` caps the whole fleet across every tenant.
`MachineType.max_workers` caps one tenant, so without it the fleet is that
number times however many tenants show up. Unset means no ceiling, which is
right for a single-tenant pool and wrong for a hosted one. Hitting the
ceiling logs a warning: a capped fleet looks exactly like a slow queue from
the outside.

Still missing: any restriction on the container's own network access.

## What it is not

**It is not a cache.** The pool has no idea Strata deduplicates work.
Submitting a job whose result already exists boots a machine and recomputes
it — so the caller checks `find_by_provenance` *before* submitting. Getting
that order wrong bills customers for cache hits.

**It is not the metering layer.** The pool records one `UsageEvent` per
terminal job, including failures, with a monotonic duration. What is billable,
and at what price, is decided above it.

**It does not keep machines warm on purpose.** `start_scaler()` stops
machines idle past their type's `cool_down_seconds`, and nothing else in the
pool ever ends a machine that finished its work — a deployment that forgets
that call bills for every machine it ever started. A warm floor is
deliberately absent: per tenant it means paying for everyone who ever showed
up, per machine type it means choosing whose latency to subsidise, and
pre-warming belongs with the layer that knows a user just opened a
notebook.

## Layout

| Module | What lives there |
|---|---|
| `types.py` | `Worker`, `Job`, `MachineType`, `UsageEvent` and their states |
| `backend.py` | The `Backend` protocol — start / stop / health |
| `backends/docker.py` | Containers on the local Docker daemon |
| `store.py` | SQLite persistence; the pool process keeps no authoritative state |
| `pool.py` | Submission, dispatch, boot, execution, metering, restart recovery |

## Running it as a service

```bash
pip install 'strata-pool[server]'
```

```python
from strata_pool.api import create_app

app = create_app(pool, api_token=os.environ["STRATA_POOL_TOKEN"])
# uvicorn strata_pool_service:app
```

The app's lifespan calls `recover()` and `start_scaler()` itself, so a
deployment cannot forget the call that stops it paying for idle machines.

| Route | |
|---|---|
| `POST /v1/jobs` | Queue a job; body is the payload, verbatim. 202 with an id |
| `POST /v1/jobs/sync` | Queue and block. 200 with the result bytes, or 202 and an id if `wait_seconds` runs out |
| `GET /v1/jobs/{id}` | Status, without the payload or result |
| `GET /v1/jobs/{id}/result` | The raw result bytes; 409 while the job is not finished |
| `GET /v1/machine-types` | What a caller may ask for — the catalogue an annotation resolves against |
| `GET /v1/workers` | The fleet, without machine credentials |
| `GET /v1/usage` | The billing feed, filterable by tenant |
| `GET /health` | Outside the token check, for load balancers |

A job that fails **on the worker** comes back as 502, and one that times out
as 504 — the caller has to be able to tell "your code raised" from "we could
not run it".

Every route but `/health` requires `Authorization: Bearer <api_token>`, and
job routes require `X-Strata-Tenant`. **The caller is trusted for tenant
identity**: it authenticates as itself and asserts whose work this is. The
pool does not authenticate end users and must never be reachable from
anywhere but the proxy.

## The RunPod backend

Rents real hardware. `MachineType.gpu_type` is the provider's own string —
`"NVIDIA H100 80GB PCIe"` — deliberately not a normalised label, because one
that maps cleanly across providers does not exist and inventing it would put a
lossy translation between a user and the hardware they asked for.

```python
RunPodBackend(os.environ["RUNPOD_API_KEY"])
MachineType(
    name="h100-80gb",
    image="strata-worker:latest",
    gpu_type="NVIDIA H100 80GB PCIe",
    boot_timeout_seconds=600,   # pulling an image onto a fresh pod is minutes
    cool_down_seconds=60,       # an idle H100 is the expensive mistake
)
```

**Verified against a live account** on 2026-09-02: the base URL, `POST /pods`
with `imageName` / `ports` / `env` / `containerDiskInGb` / `name`, the `id` in
the create response, the proxy endpoint, health through that proxy, and
`DELETE /pods/{id}` including a second delete of the same pod. A CPU pod
booted, answered, and terminated with nothing left running.

**Two things that run stayed silent on**, because a CPU pod does not exercise
them: `gpuTypeIds` + `gpuCount`, and reading a region from `machine`. Every pod
in the account listing carried `machine: {}` with no `dataCenterId`, so `region`
is probably always `None` today — harmless, being metadata, but do not trust it.

RunPod has moved its API surface before. Every field lives in `_create_body`
and is asserted by a test, so a wrong one is a one-line fix; `provider_options`
overrides anything in the body, so a deployment can correct a field without
waiting for a release; and `base_url` can be repointed. To re-verify, or to
close the GPU gap with `STRATA_POOL_RUNPOD_GPU`:

```bash
export RUNPOD_API_KEY=...
STRATA_POOL_RUNPOD_LIVE=1 pytest packages/strata-pool/tests/test_runpod_live.py -v -s
```

That **starts a billed pod**, which is why it is opt-in and never runs in CI.
It terminates what it starts, but a crashed interpreter can still leave a pod
running — check the console. Pods are named `strata-{machine_type}-{id}` so an
orphan is findable.

A pod's port is published on the public internet through RunPod's proxy. The
per-machine credential is what stands between that URL and anyone who finds
it.

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
