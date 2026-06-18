"""Per-domain ``APIRouter`` modules mounted onto the app in ``strata.server``.

Phase 3 of ``docs/internal/design-server-decomposition.md``: pure code motion of
handlers out of the ``server.py`` monolith, one domain at a time. Routers are a
leaf — handlers reach shared server state via a lazy ``from strata.server import
get_state`` inside the body (the same pattern ``strata/notebook/routes.py`` uses)
so the api package never imports ``server`` at module load.
"""
