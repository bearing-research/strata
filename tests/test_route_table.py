"""Route-table snapshot — the safety net for the server.py router split (P3).

The router split (``docs/internal/design-server-decomposition.md`` phase 3) moves
handlers out of ``server.py`` into ``strata/api/routers/*`` one domain at a time.
Pure code motion must not change the HTTP surface, so this test freezes the full
inventory of ``(path, methods, route-level-dependency-count)`` and asserts it is
unchanged. The dependency count guards the gates: moving a route and dropping its
``dependencies=[require_scope(...)]`` flips the count and fails here.

When a route is *intentionally* added/removed/regated, update ``EXPECTED_ROUTES``
in the same commit — the diff is the review surface.

The frontend SPA catch-all (``/{full_path:path}``) is deliberately excluded: it
is mounted by ``_mount_frontend`` only when a built ``frontend/dist`` bundle is
present, which is true in a dev tree but not in the CI unit-test job — so it is
environment-dependent and not part of the API surface this guards.
"""

from fastapi.routing import APIRoute

from strata.server import app

# The frontend SPA fallback is conditionally mounted (bundle-dependent); exclude
# it so the snapshot is deterministic across dev and CI.
_EXCLUDED_PATHS = {"/{full_path:path}"}

EXPECTED_ROUTES = [
    ("/health", "GET", 0),
    ("/health/dependencies", "GET", 0),
    ("/health/ready", "GET", 0),
    ("/metrics", "GET", 0),
    ("/metrics/prometheus", "GET", 0),
    ("/metrics/tables", "GET", 0),
    ("/metrics/tables/{table_id:path}", "GET", 0),
    ("/v1/admin/notebook-workers", "GET", 1),
    ("/v1/admin/notebook-workers", "POST", 1),
    ("/v1/admin/notebook-workers", "PUT", 1),
    ("/v1/admin/notebook-workers/{worker_name}", "DELETE", 1),
    ("/v1/admin/notebook-workers/{worker_name}", "PATCH", 1),
    ("/v1/admin/notebook-workers/{worker_name}", "PUT", 1),
    ("/v1/admin/notebook-workers/{worker_name}/refresh", "POST", 1),
    ("/v1/admin/tenants", "GET", 1),
    ("/v1/admin/tenants/{tenant_id}", "GET", 1),
    ("/v1/artifacts", "GET", 0),
    ("/v1/artifacts", "PUT", 0),
    ("/v1/artifacts/builds/{build_id}", "GET", 0),
    ("/v1/artifacts/download", "GET", 0),
    ("/v1/artifacts/explain-materialize", "POST", 0),
    ("/v1/artifacts/finalize", "POST", 0),
    ("/v1/artifacts/gc", "POST", 0),
    ("/v1/artifacts/materialize", "POST", 0),
    ("/v1/artifacts/names/{name:path}/status", "GET", 0),
    ("/v1/artifacts/stats", "GET", 0),
    ("/v1/artifacts/upload", "POST", 0),
    ("/v1/artifacts/upload/{artifact_id}/v/{version}", "POST", 0),
    ("/v1/artifacts/usage", "GET", 0),
    ("/v1/artifacts/{artifact_id}/v/{version}", "DELETE", 0),
    ("/v1/artifacts/{artifact_id}/v/{version}", "GET", 0),
    ("/v1/artifacts/{artifact_id}/v/{version}/data", "GET", 0),
    ("/v1/artifacts/{artifact_id}/v/{version}/dependents", "GET", 0),
    ("/v1/artifacts/{artifact_id}/v/{version}/lineage", "GET", 0),
    ("/v1/artifacts/{artifact_id}/v/{version}/tags", "GET", 0),
    ("/v1/artifacts/{artifact_id}/v/{version}/tags", "PUT", 0),
    ("/v1/artifacts/{artifact_id}/v/{version}/tags/{key}", "DELETE", 0),
    ("/v1/builds/{build_id}", "GET", 0),
    ("/v1/builds/{build_id}/finalize", "POST", 0),
    ("/v1/builds/{build_id}/manifest", "GET", 0),
    ("/v1/cache/clear", "POST", 1),
    ("/v1/cache/entries", "GET", 0),
    ("/v1/cache/evictions", "GET", 0),
    ("/v1/cache/histogram", "GET", 0),
    ("/v1/cache/stats", "GET", 0),
    ("/v1/cache/warm", "POST", 0),
    ("/v1/cache/warm/async", "POST", 0),
    ("/v1/cache/warm/jobs", "GET", 0),
    ("/v1/cache/warm/jobs/{job_id}", "DELETE", 0),
    ("/v1/cache/warm/jobs/{job_id}", "GET", 0),
    ("/v1/config/timeouts", "GET", 0),
    ("/v1/debug/cache/inspect", "GET", 0),
    ("/v1/debug/circuit-breakers", "GET", 0),
    ("/v1/debug/connections", "GET", 0),
    ("/v1/debug/gc/pauses", "GET", 0),
    ("/v1/debug/latency", "GET", 0),
    ("/v1/debug/memory", "GET", 0),
    ("/v1/debug/pools", "GET", 0),
    ("/v1/debug/rate-limits", "GET", 0),
    ("/v1/materialize", "POST", 0),
    ("/v1/metadata/cleanup", "POST", 0),
    ("/v1/metadata/stats", "GET", 0),
    ("/v1/names", "GET", 0),
    ("/v1/names", "POST", 0),
    ("/v1/names/{name:path}", "DELETE", 0),
    ("/v1/names/{name:path}", "GET", 0),
    ("/v1/names/{name:path}/aliases", "GET", 0),
    ("/v1/names/{name:path}/aliases/{alias}", "DELETE", 0),
    ("/v1/names/{name:path}/aliases/{alias}", "GET", 0),
    ("/v1/names/{name:path}/aliases/{alias}", "PUT", 0),
    ("/v1/notebooks/config", "GET", 0),
    ("/v1/notebooks/create", "POST", 0),
    ("/v1/notebooks/delete-by-path", "POST", 0),
    ("/v1/notebooks/discover", "GET", 0),
    ("/v1/notebooks/import", "POST", 0),
    ("/v1/notebooks/open", "POST", 0),
    ("/v1/notebooks/recents/validate", "POST", 0),
    ("/v1/notebooks/sessions", "GET", 0),
    ("/v1/notebooks/sessions/{session_id}", "GET", 0),
    ("/v1/notebooks/{notebook_id}", "DELETE", 0),
    ("/v1/notebooks/{notebook_id}/ai/agent", "POST", 0),
    ("/v1/notebooks/{notebook_id}/ai/agent/reset", "POST", 0),
    ("/v1/notebooks/{notebook_id}/ai/complete", "POST", 0),
    ("/v1/notebooks/{notebook_id}/ai/model", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/ai/models", "GET", 0),
    ("/v1/notebooks/{notebook_id}/ai/status", "GET", 0),
    ("/v1/notebooks/{notebook_id}/ai/stream", "POST", 0),
    ("/v1/notebooks/{notebook_id}/artifacts", "GET", 0),
    ("/v1/notebooks/{notebook_id}/cells", "GET", 0),
    ("/v1/notebooks/{notebook_id}/cells", "POST", 0),
    ("/v1/notebooks/{notebook_id}/cells/reorder", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/cells/{cell_id}", "DELETE", 0),
    ("/v1/notebooks/{notebook_id}/cells/{cell_id}", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/cells/{cell_id}/execute", "POST", 0),
    ("/v1/notebooks/{notebook_id}/cells/{cell_id}/iterations", "GET", 0),
    ("/v1/notebooks/{notebook_id}/cells/{cell_id}/tests", "POST", 0),
    ("/v1/notebooks/{notebook_id}/connections", "GET", 0),
    ("/v1/notebooks/{notebook_id}/connections", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/connections/{name}/schema", "GET", 0),
    ("/v1/notebooks/{notebook_id}/dag", "GET", 0),
    ("/v1/notebooks/{notebook_id}/dependencies", "GET", 0),
    ("/v1/notebooks/{notebook_id}/dependencies", "POST", 0),
    ("/v1/notebooks/{notebook_id}/dependencies/{package_name}", "DELETE", 0),
    ("/v1/notebooks/{notebook_id}/env", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/environment", "GET", 0),
    ("/v1/notebooks/{notebook_id}/environment/environment.yaml", "POST", 0),
    ("/v1/notebooks/{notebook_id}/environment/environment.yaml/preview", "POST", 0),
    ("/v1/notebooks/{notebook_id}/environment/jobs", "POST", 0),
    ("/v1/notebooks/{notebook_id}/environment/jobs/current", "GET", 0),
    ("/v1/notebooks/{notebook_id}/environment/requirements.txt", "GET", 0),
    ("/v1/notebooks/{notebook_id}/environment/requirements.txt", "POST", 0),
    ("/v1/notebooks/{notebook_id}/environment/requirements.txt/preview", "POST", 0),
    ("/v1/notebooks/{notebook_id}/environment/sync", "POST", 0),
    ("/v1/notebooks/{notebook_id}/export", "GET", 0),
    ("/v1/notebooks/{notebook_id}/mounts", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/name", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/python-version", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/r-packages", "GET", 0),
    ("/v1/notebooks/{notebook_id}/secret-manager/config", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/secret-manager/refresh", "POST", 0),
    ("/v1/notebooks/{notebook_id}/timeout", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/variant-groups/{group_id}", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/variant-groups/{group_id}/variants", "POST", 0),
    ("/v1/notebooks/{notebook_id}/worker", "PUT", 0),
    ("/v1/notebooks/{notebook_id}/workers", "GET", 0),
    ("/v1/notebooks/{notebook_id}/workers", "PUT", 0),
    ("/v1/registry/audit", "GET", 0),
    ("/v1/registry/pending", "GET", 0),
    ("/v1/registry/pending/approve", "POST", 0),
    ("/v1/registry/pending/reject", "POST", 0),
    ("/v1/registry/summary", "GET", 0),
    ("/v1/streams/{stream_id}", "GET", 0),
]


def _current_routes():
    return sorted(
        (route.path, ",".join(sorted(route.methods)), len(route.dependencies))
        for route in app.routes
        if isinstance(route, APIRoute) and route.path not in _EXCLUDED_PATHS
    )


def test_route_table_snapshot():
    """The full HTTP surface (path, methods, route-level gate count) is frozen.

    A mismatch means a route was added, removed, renamed, re-methoded, or lost a
    route-level dependency. If the change is intentional, update EXPECTED_ROUTES.
    """
    assert _current_routes() == [tuple(row) for row in EXPECTED_ROUTES]
