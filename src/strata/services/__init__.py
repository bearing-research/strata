"""Service layer: orchestration and policy logic extracted from HTTP handlers.

Services are stateless, take already-resolved dependencies (store, tenant filter,
principal) per call, and have no FastAPI coupling, so they test without a TestClient.
"""
