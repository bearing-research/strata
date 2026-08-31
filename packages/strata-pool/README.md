# strata-pool

Worker pool for dispatching Strata jobs to ephemeral machines. Not published
yet — it ships with the hosted workers service.

Jobs arrive with a machine type. The pool hands each one to a warm worker of
that type, starts a machine when there is none, forwards the payload over
HTTP, records the result, and meters the execution. Backends (local Docker,
Fly, EC2) provide start / stop / health; the pool does not know which it is
talking to.

```python
from strata_pool import MachineType, Pool, PoolStore

pool = Pool(
    store=PoolStore("pool.sqlite"),
    backend=backend,
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
| `store.py` | SQLite persistence; the pool process keeps no authoritative state |
| `pool.py` | Submission, dispatch, boot, execution, metering, restart recovery |

## Tests

```bash
uv run pytest packages/strata-pool/tests -v
```

CI additionally installs the package into a venv with only its own
dependencies, to keep it from quietly growing a dependency on the server.
