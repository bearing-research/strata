"""Metadata-store and timeout-config routes.

Server state is reached via a lazy ``from strata.server import get_state`` so
this module stays a leaf (``server.py`` imports the router, not the reverse).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["metadata"])


@router.get("/v1/metadata/stats")
async def get_metadata_stats_v1():
    """Get hit/miss counters and entry counts for the SQLite metadata store and LRU caches."""
    from strata.metadata_cache import get_metadata_store
    from strata.server import get_state

    state = get_state()

    result = {
        "parquet_cache": state.planner.parquet_cache.stats(),
        "manifest_cache": state.planner.manifest_cache.stats(),
    }

    try:
        store = get_metadata_store()
        result["metadata_store"] = store.stats()
    except Exception:
        result["metadata_store"] = None

    return result


@router.get("/v1/config/timeouts")
async def get_timeout_config_v1():
    """Get all timeout settings, grouped by planning, scanning, QoS queue, fetching and S3."""
    from strata.server import get_state

    state = get_state()
    return state.config.get_timeout_config()


@router.post("/v1/metadata/cleanup")
async def cleanup_metadata_v1():
    """Remove parquet metadata entries whose file is gone or has a different mtime or size.

    Runs on server startup too. Returns the number of entries removed.
    """
    from strata.metadata_cache import get_metadata_store

    try:
        store = get_metadata_store()
        removed = store.cleanup_stale_parquet_meta()
        return {
            "status": "completed",
            "stale_entries_removed": removed,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
