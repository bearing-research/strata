"""Per-domain ``APIRouter`` modules mounted onto the app in ``strata.server``.

Handlers reach shared server state through a lazy ``from strata.server import
get_state`` inside the body, so this package never imports ``server`` at load.
"""
