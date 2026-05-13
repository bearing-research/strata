# Observability

Strata ships an opt-in observability stack, Jaeger for traces,
Prometheus for metrics, Grafana for dashboards, wired up as a
docker-compose file you can run alongside the server. Used in dev
to validate that a change didn't regress latency or cache hit rate,
and in production as a reference layout your real observability
provider can mirror.

## Running the stack

```bash
docker compose -f docker-compose.observability.yml up -d --build
```

The stack starts four containers:

| Container | Port | What it does |
|---|---|---|
| `strata` | 8765 | Strata server with OTel tracing enabled and structured JSON logging |
| `jaeger` | 16686 | Trace UI + OTLP collector (gRPC on 4317, HTTP on 4318) |
| `prometheus` | 9090 | Scrapes `strata:/metrics/prometheus` every 5s |
| `grafana` | 3000 | Dashboards (login: `admin` / `admin`) |

Open:

- **Strata**: [http://localhost:8765](http://localhost:8765)
- **Jaeger UI**: [http://localhost:16686](http://localhost:16686) (search for `strata` service)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)
- **Grafana**: [http://localhost:3000](http://localhost:3000)

Stop:

```bash
docker compose -f docker-compose.observability.yml down
```

To wipe state too (Prometheus / Grafana data), add `-v`.

## What gets emitted

### Metrics, `/metrics/prometheus`

Strata exposes the standard Prometheus textfile format at
`GET /metrics/prometheus`. The scrape config in
`observability/prometheus.yml` pulls every 5 seconds. Metric
families include:

- **Cache**: hit rate, eviction count, byte-size occupancy
- **Scans**: active count, completion rate, throughput in
  bytes/second
- **QoS**: interactive vs bulk semaphore usage, per-tenant
  breakdowns when multi-tenant is on
- **Rate limiter**: request acceptance / rejection counts
- **Server**: uptime, health-check status

For multi-tenant deployments the labels carry a `tenant` dimension
so dashboards can split per-tenant usage.

### Traces, OpenTelemetry OTLP

When `STRATA_TRACING_ENABLED=true` is set, Strata emits OTel spans
over OTLP-gRPC to `OTEL_EXPORTER_OTLP_ENDPOINT` (default
`http://jaeger:4317` in the compose stack). One span per request
plus child spans for the planner, cache lookup, Parquet read,
serialize-to-Arrow IPC, and stream write. The `service.name`
attribute is set via `OTEL_SERVICE_NAME=strata`.

In Jaeger:

1. Pick the `strata` service.
2. Optional filter by `Operation` (e.g. `POST /v1/materialize`).
3. Click into a trace to see the full waterfall, cache hits are
   short; misses fan out through planner → Parquet I/O →
   Arrow IPC writer.

### Logs

Structured JSON logging via `STRATA_LOG_FORMAT=json`. One line per
log record with `level`, `logger`, `message`, `timestamp`, and
context fields (request ID, tenant, principal where applicable).
Pipe `docker compose logs strata` through `jq` for readable
output.

## Grafana dashboards

`observability/grafana/provisioning/` is mounted into the Grafana
container so dashboards and the Prometheus datasource are
provisioned on container start, no clicking through the UI to
import them.

One dashboard ships out of the box (`Strata Overview`), with panels for:

- **Server Status**: up / down probe
- **Cache Hit Rate**: rolling hit ratio
- **Active Scans**: concurrent scans currently running
- **Scan Rate**: completions per second
- **Throughput (Bytes/sec)**: egress to clients
- **QoS Slot Usage**: interactive vs bulk semaphores
- **Rate Limit Rejections**: count of requests turned away

To add more dashboards, drop the JSON into
`observability/grafana/provisioning/dashboards/` and restart the
Grafana container.

## Wiring your own observability provider

The compose stack is opinionated for local dev. In production you
typically have an existing OTel collector, Prometheus instance, and
log aggregator. Point Strata at yours via env vars:

```bash
# Tracing
STRATA_TRACING_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-collector:4317
OTEL_EXPORTER_OTLP_HEADERS=authorization=Bearer <token>
OTEL_SERVICE_NAME=strata-prod

# Logs
STRATA_LOG_FORMAT=json
STRATA_LOG_LEVEL=INFO
```

Prometheus scraping is pull-based, your scrape config needs an
entry pointing at `<strata-host>:8765/metrics/prometheus`. There's
no push-gateway integration; if you need to push (e.g. for jobs in
ephemeral containers), wrap the scrape in your own sidecar.

## Generating load to see traces

The repo ships a small capacity sweep that hammers the server with
realistic Iceberg scans, useful for seeing the trace structure
under load:

```bash
uv run python benchmarks/capacity_sweep.py \
  --quick \
  --no-server \
  --base-url http://localhost:8765
```

After it runs, refresh Jaeger and you'll see a handful of
materialize spans with their full child waterfall.

## Health endpoints

Two simple endpoints intentionally outside the metrics path so
they're cheap to hit from k8s liveness/readiness probes or
Fly health checks:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Returns `{"status":"ok"}` if the server is running. Used by Docker / Fly / k8s. |
| `GET /metrics/prometheus` | Scrape target. Returns Prometheus textfile. |

`/health` is intentionally minimal, it doesn't probe downstream
dependencies. Use the scrape metrics for richer health signals
(catalog reachability, blob backend latency, etc.).
