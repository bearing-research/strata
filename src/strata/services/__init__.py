"""Service layer: orchestration/policy logic extracted from HTTP handlers.

Services are stateless and receive already-resolved dependencies (artifact
store, tenant filter, principal) per call from the route. They contain no
FastAPI/HTTP coupling, so they are unit-testable without a TestClient. See
``docs/internal/design-server-decomposition.md`` (phase 2).
"""
