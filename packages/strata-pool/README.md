# strata-pool

<!-- --8<-- [start:body] -->
Worker pool for dispatching Strata jobs to ephemeral machines. This is the
bring-your-own-hardware path: it manages machines *you* own. If you want a
provider to autoscale for you, register a serverless executor as a Strata
worker instead.

```bash
pip install strata-pool            # library
pip install "strata-pool[server]"  # plus the HTTP service
```

Jobs arrive with a machine type. The pool hands each one to a warm worker of
that type, starts a machine when there is none, forwards the payload over
HTTP, records the result, and meters the execution. A backend provides start /
stop / health and nothing else, so the pool does not know which it is talking
to. Three ship: local Docker, RunPod, and Fly Machines. Anything satisfying
the `Backend` protocol works.

```python
from strata_pool import DockerBackend, MachineType, Pool, PoolStore

pool = Pool(
    store=PoolStore("pool.sqlite"),
    backend=DockerBackend(),
    machine_types=[MachineType(name="cpu-4x", image="strata-worker:latest")],
)
await pool.recover()  # reconcile after a restart
pool.start_scaler()  # stop paying for machines that finished

job = await pool.submit(tenant_id="acme", machine_type="cpu-4x", payload=bundle)
done = await pool.wait(job.id)
```

`DockerBackend()` talks to `/var/run/docker.sock`. Docker Desktop on macOS
puts the socket under your home directory, so pass it:
`DockerBackend(socket_path=os.path.expanduser("~/.docker/run/docker.sock"))`.

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
untrusted code: a shared kernel is one CVE away from a cross-tenant escape.
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

## Several processes over one store

`PoolStore` is a SQLite file, and one pool process on it is the simple
deployment: a restart pauses dispatch for seconds and fails the jobs that were
in flight. To keep dispatching through a restart, run more than one process
over `PostgresPoolStore`, each with its own `instance_id`:

```bash
pip install "strata-pool[postgres]"
```

```python
pool = Pool(
    store=PostgresPoolStore("postgresql://pool@db/pool"),
    backend=RunPodBackend(os.environ["RUNPOD_API_KEY"]),
    machine_types=catalogue,
    instance_id=os.environ["HOSTNAME"],  # required on a shared store
)
```

The processes never run a job twice or start two machines for the same
demand. Handing a job to a warm machine is a conditional update that only one
of them wins, and deciding to start a machine counts the queue and inserts the
new machine's row in one transaction, serialized across processes. Each
process holds a **lease** on the jobs and machines it is acting on (a machine
starting, running a job or stopping; a job dispatched or running) and renews it
while it works. When a process dies, the scaler in another one fails its jobs
and stops its machines once the lease runs out (`lease_seconds`, default 30),
through the surviving process's own backend. A process restarted under its old
`instance_id` takes its own rows back at once, as a single process does today.
A pool given no `instance_id` generates one per process, so two processes over
one SQLite file are never one name. Under a shared name each would read the
other's leases as its own and stop machines running the other's cells. The
trade is that a generated name is new on every start, so a process restarted
within `lease_seconds` waits its old leases out instead of reclaiming them at
once; set `instance_id` explicitly to keep that.

Lease expiry is compared by wall clock across processes, so their clocks have
to agree to well within a lease. The catalogue lives in the store too:
`PUT /v1/machine-types` to any one process reaches the others, each of which
applies it on its next submit or scaler pass, so they agree on which machines
run the current image.

## What it is not

**It is not a cache.** The pool has no idea Strata deduplicates work.
Submitting a job whose result already exists boots a machine and recomputes
it, so the caller checks `find_by_provenance` *before* submitting. Getting
that order wrong bills customers for cache hits.

**It is not the metering layer.** The pool records one `UsageEvent` per
terminal job, including failures and cancels, with a monotonic duration. A job
cancelled before it reached a machine ran for no time and has none. What is billable,
and at what price, is decided above it.

**It does not keep machines warm on purpose.** `start_scaler()` stops
machines idle past their type's `cool_down_seconds`, and nothing else in the
pool ever ends a machine that finished its work: a deployment that forgets
that call bills for every machine it ever started. A machine whose stop fails
keeps its row, still counted against the fleet cap, and the next pass tries
the stop again. A warm floor is
deliberately absent: per tenant it means paying for everyone who ever showed
up, per machine type it means choosing whose latency to subsidise, and
pre-warming belongs with the layer that knows a user just opened a
notebook.

The same scaler pass health-checks warm machines and retires ones that stopped
answering, after `health_check_failures` consecutive misses (default 3; `0`
turns probing off for a type). One miss is a slow machine or a dropped packet,
and retiring on that trades a cold start for every hiccup; three is a machine
that is gone. Probes run concurrently, so a provider black-holing packets
cannot stall the reaper behind one timeout per machine. And a pass in which
*every* machine fails is treated as this process's own network rather than as
the fleet dying at once, since retiring everything for a local DNS blip would
hand every user a cold start for a fault that was never on the machines.
Without this a dead warm machine is discovered by the next job being sent to
it, and that job fails for reasons that have nothing to do with the code in
the cell. A machine running a job is neither probed nor reaped.

## Layout

| Module | What lives there |
|---|---|
| `types.py` | `Worker`, `Job`, `MachineType`, `UsageEvent` and their states |
| `backend.py` | The `Backend` protocol: start / stop / health |
| `backends/docker.py` | Containers on the local Docker daemon |
| `backends/runpod.py` | RunPod pods, reached through RunPod's proxy |
| `backends/fly.py` | Fly Machines, reached on the organization's private network |
| `store.py` | SQLite and Postgres persistence, with the claims and leases that let several processes share it; the pool process keeps no authoritative state |
| `pool.py` | Submission, dispatch, boot, execution, metering, restart recovery, the scaler |
| `api.py` | The HTTP service (the `server` extra) |

## Running it as a service

```bash
pip install 'strata-pool[server]'
```

```python
from strata_pool.api import create_app

app = create_app(
    pool,
    api_token=os.environ["STRATA_POOL_TOKEN"],
    admin_token=os.environ.get("STRATA_POOL_ADMIN_TOKEN"),  # optional, operator only
)
# uvicorn strata_pool_service:app
```

The app's lifespan calls `recover()` and `start_scaler()` itself, so a
deployment cannot forget the call that stops it paying for idle machines.

| Route | |
|---|---|
| `POST /v1/jobs` | Queue a job; body is the payload, verbatim. 202 with an id |
| `POST /v1/jobs/sync` | Queue and block. 200 with the result bytes, or 202 and an id if `wait_seconds` runs out |
| `GET /v1/jobs/{id}` | Status, without the payload or result; 404 for another tenant's job |
| `GET /v1/jobs/{id}/result` | The raw result bytes; 409 while the job is not finished, 404 for another tenant's job |
| `POST /v1/jobs/{id}/cancel` | Cancel a job that has not finished; body `{"build_id": "..."}`. 200 with the job, 409 if it finished another way, 404 for another tenant's job |
| `GET /v1/machine-types` | What a caller may ask for: the catalogue an annotation resolves against. `env` and `provider_options` values read `<redacted>` without the admin token |
| `PUT /v1/machine-types` | Replace the catalogue without a restart; persisted, so a restart serves it. Admin token only |
| `GET /v1/workers` | The caller's tenant's machines, without their credentials; the admin token sees the whole fleet, or one tenant with `?tenant_id=` |
| `GET /v1/usage` | The billing feed: the caller's tenant only; the admin token sees every tenant, or one with `?tenant_id=` |
| `GET /health` | Outside the token check, for load balancers |

Both job submit routes take their options as query parameters: `machine_type`
(required), `priority` (default 0, higher runs first; a 64-bit integer), `session_id` and
`timeout_seconds` (both optional). `timeout_seconds` can shorten the machine type's
`job_timeout_seconds` but not extend it. `POST /v1/jobs/sync` also takes
`wait_seconds` (default 300), which bounds the wait and not the job, and is cut to
the type's `boot_timeout_seconds` plus `job_timeout_seconds`. Both times must be
positive and finite; anything else is a 422 and no job is queued.

The `PUT` body is the whole catalogue: a JSON list of machine types, each with
`MachineType`'s fields (only `name` and `image` are required), the shape `GET`
returns:

```bash
curl -X PUT http://pool.internal:8000/v1/machine-types \
  -H "Authorization: Bearer $STRATA_POOL_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '[{"name": "cpu-4x", "image": "strata-worker:latest", "cpus": 4, "memory_mb": 8192, "max_workers": 5}]'
```

`PUT /v1/machine-types` checks each field's type and range before storing
anything: a number sent as a string, an unknown key, a non-finite number, a
`max_workers` below 1, a timeout, `cpus`, `memory_mb` or `disk_gb` of zero or
less, or a negative cool-down, `health_check_failures` or `gpu_count` is a 400
naming the field.

Replacing the catalogue needs the **admin token**. A catalogue entry decides
which image receives a tenant's jobs, and with them the signed URLs for that
tenant's data, so the API token every caller holds cannot rewrite it. A pool
built without an `admin_token` refuses catalogue writes over HTTP with 403:
the catalogue is then the one passed to `Pool(machine_types=...)`, unless an
earlier admin `PUT` stored one, which replaces it on start.

Replacing the catalogue takes effect at once. A new type accepts jobs straight
away. A removed type accepts none: its queued jobs fail with the reason, and
its machines finish what they are running, then retire once idle past the
type's cool-down. A type whose `image` changed starts new machines on the new
image. Machines already running the old image get no new jobs and retire the
same way. The catalogue is stored with the pool's state, and on start it
replaces the one the process was constructed with. A pool calling
`replace_machine_types` directly changes only its own copy; `sync_catalogue`
applies the stored one.

A job submitted with W3C `traceparent` / `tracestate` headers keeps them, and
the pool forwards them to the machine it runs on. With OpenTelemetry installed
(it is not a dependency) the pool also opens a `pool.execute` span for the job's
time on the machine. The machine's work then sits under that span, which sits
under the caller's. Beside it, `pool.queue` covers the job's wait from submit to
dispatch, whichever process dispatched it, and `pool.boot` a machine's start
from the request to the provider until it answers its health check. A boot is
traced under the job first in line for that machine, and is marked as an error
when the machine fails to start or to boot.

A job that fails **on the worker** comes back as 502, and one that times out
as 504. The caller has to be able to tell "your code raised" from "we could
not run it". A cancelled job comes back as 409 with `"state": "cancelled"`.

**Cancelling a job stops the job, not the machine.** `POST
/v1/jobs/{id}/cancel` marks the job `cancelled` at once, so a
`POST /v1/jobs/sync` waiting on it returns straight away. What else happens
depends on how far the job got:

- **Queued, or handed to a machine but not yet sent:** it never reaches a
  machine and is not metered. The cancel and the dispatch are conditional
  updates on the same row, so a job cancelled through one pool process is
  never started by another.
- **Running:** the pool posts `/v1/executions/{build_id}/cancel` to the
  machine running it, with that machine's token. The payload is opaque to the
  pool, so the caller names the build in the request body (letters, digits,
  `-` and `_`). When the machine's execute call answers, the process running
  the job records a usage event for the time it ran, with
  `terminal_state: "cancelled"`, and the machine goes back to warm for the next
  job. The cancel stands even if the machine finished first: its result is
  dropped. A machine that ignores the cancel runs the job out, and one that
  never answers is retired at the job's timeout, as any other.

Cancelling a job that is already cancelled answers 200 again, so a retry is
safe. A job that completed, failed or timed out is a 409, like a result asked
for too early.

Every route but `/health` requires `Authorization: Bearer <api_token>` (the
admin token is accepted wherever the API token is). `/health` is
unauthenticated and lists the machine-type names and the fleet's machine
counts by state, so anyone who can reach the pool can read them. Every job route
requires `X-Strata-Tenant`, reads included. So do the worker and usage
listings, unless the caller presents the admin token. A tenant reads only
its own jobs: another tenant's job id answers 404, not 403, so the id does not
confirm the job exists. **The caller is trusted for tenant
identity**: it authenticates as itself and asserts whose work this is. The
pool does not authenticate end users and must never be reachable from
anywhere but the proxy.

## The RunPod backend

Rents real hardware. `MachineType.gpu_type` is the provider's own string
(`"NVIDIA H100 80GB PCIe"`), deliberately not a normalised label, because one
that maps cleanly across providers does not exist and inventing it would put a
lossy translation between a user and the hardware they asked for.

```python
RunPodBackend(os.environ["RUNPOD_API_KEY"])
MachineType(
    name="h100-80gb",
    image="strata-worker:latest",
    gpu_type="NVIDIA H100 80GB PCIe",
    boot_timeout_seconds=600,  # pulling an image onto a fresh pod is minutes
    cool_down_seconds=60,  # an idle H100 is the expensive mistake
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
is probably always `None` today. That is harmless, being metadata, but do not
trust it.

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
running, so check the console. Pods are named `strata-{machine_type}-{id}` so an
orphan is findable.

A pod's port is published on the public internet through RunPod's proxy. The
per-machine credential is what stands between that URL and anyone who finds
it.

## The Fly Machines backend

Boots workers as machines in one Fly app and region. A machine is reachable
only on the organization's private network, at
`http://{machine_id}.vm.{app}.internal:{port}`, so a pool running beside its
notebook servers on Fly hands them workers that never face the internet.

```python
FlyBackend(os.environ["FLY_API_TOKEN"], app="strata-workers", region="sjc")
MachineType(
    name="a100",
    image="registry.fly.io/strata-worker:latest",
    cpus=8,
    memory_mb=65536,
    gpu_type="a100-80gb",  # Fly's own gpu_kind string
    cool_down_seconds=120,
)
```

`start` creates the machine with the worker token and the machine type's `env`
in its environment, with restart policy `no` and `auto_destroy: false` so its
lifetime stays the pool's. `stop` destroys it by
force (a second destroy is a no-op). `health` needs Fly to report the machine
`started` before it probes the worker's `/health`, because a stopped machine's
private address can be reused.

**Not yet verified against a live account.** The request shapes follow the
Machines API documentation; every field lives in `_create_body` and is asserted
by a test, `provider_options` overrides anything in the machine `config`, and
`base_url` can be repointed. To verify:

```bash
export FLY_API_TOKEN=... STRATA_POOL_FLY_APP=... STRATA_POOL_FLY_REGION=sjc
STRATA_POOL_FLY_LIVE=1 pytest packages/strata-pool/tests/test_fly_live.py -v -s
```

That **starts a billed machine**. Off the private network it proves create,
start and destroy; add `STRATA_POOL_FLY_ON_NETWORK=1` where the `.internal`
names resolve (a Fly machine, or a `fly wireguard` peer) to probe the worker
too.

The worker credential still matters here: every machine in the organization
can reach every other.

## The worker contract

`strata-worker`, shipped in `strata-notebook`, satisfies this contract as of
0.7.0. Build it with the Dockerfile in the Strata repo and the pool can drive
it unmodified:

```bash
docker build -f worker.Dockerfile -t strata-worker:latest .
```

The image installs a pinned `strata-notebook` from PyPI (the `STRATA_VERSION`
build arg), which must be **the same release as the Strata server** that
dispatches to it: workers and servers upgrade together. The pin moves to each
new release right after it is published; until then pass
`--build-arg STRATA_VERSION=<server version>`. Installing from PyPI also means
building inside a checkout does not pick up local worker changes; build a
wheel for that.

Layer your cells' dependencies on top (`FROM strata-worker:latest`). The image
binds 8080 because that is `DockerBackend`'s default `worker_port`; the
worker's own default is 9000, so the two are made to agree explicitly rather
than by luck. The pool does not pull, so build on the host that will run it.
It runs as a non-root user and **refuses to start without
`STRATA_WORKER_TOKEN`**. The pool mints one per machine, so this only bites
when running it by hand.

**A server on a private address has to be named.** The worker rejects manifest
URLs that resolve to loopback or private addresses. That is a real defense,
since a buggy or compromised orchestrator could otherwise point it at internal
services. With `DockerBackend` on a local daemon, or workers on Fly's private
network, the Strata server *is* at a private address, so the first job fails
with `resolves to non-routable address`. In a deployment, list the server's
host in the machine type's `env` as `STRATA_WORKER_ALLOWED_HOSTS`
(comma-separated; a leading dot is a suffix, e.g. `.internal`).
`STRATA_WORKER_ALLOW_LOCAL_HOSTS=1` relaxes the check for every host and is a
local-development setting only.

The payload the pool forwards is a **build manifest**: the same JSON document
`/v1/execute-manifest` takes, carrying signed URLs for the inputs, the output,
and finalization. The worker fetches its own inputs and uploads its own result,
so the bytes never flow through the pool. That is also why the payload has to
be self-describing: the pool forwards it verbatim and sets no content type.

The pool remains image-agnostic. Anything holding up these four points works:

| | |
|---|---|
| Listen on the worker port | 8080 by default; the backend publishes it |
| `GET /health` → 200 when ready | No auth. It is polled before the machine is trusted with anything, and it reveals nothing secret (`strata-worker` reports its capabilities and hardware) |
| `POST /execute` → 200 with the result body | The request body is the job payload, opaque to the pool |
| Require `Authorization: Bearer $STRATA_WORKER_TOKEN` on `/execute` | Reject anything else with 401 |

A worker that also serves `POST /v1/executions/{build_id}/cancel` behind the
same token, as `strata-worker` does, can have a running job cancelled without
losing the machine. One that does not answers the cancel with an error, and
the job runs out.

The token is minted per machine before it boots and passed in its
environment. Without that check, `/execute` is an unauthenticated
remote-code-execution endpoint, survivable only while the machine is bound to
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
missing daemon an error instead of a skip. CI sets that, so a runner whose
socket moved fails loudly rather than reporting coverage it never ran.

CI additionally installs the package into a venv with only its own
dependencies, to keep it from quietly growing a dependency on the server.
<!-- --8<-- [end:body] -->
