"""Metadata-store + timeout-config routes.

Moved verbatim from ``server.py`` (server decomposition, #210). Read/maintenance
endpoints for the metadata cache plus the timeout-config dump — the config /
metadata domain the ``debug`` router split deliberately left behind. Server state
is reached through a lazy ``from strata.server import get_state`` inside each body
so this module stays a leaf (``server.py`` imports the router, not vice-versa).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["metadata"])


@router.get("/v1/metadata/stats")
async def get_metadata_stats_v1():
    """Get metadata store and cache statistics.

    Returns hit/miss counters and entry counts for:
    - SQLite metadata store (manifest cache, parquet metadata)
    - In-memory LRU caches (parquet metadata, manifest resolution)

    Useful for:
    - Proving cache value (hit rates)
    - Debugging performance issues
    - Capacity planning
    """
    from strata.metadata_cache import get_metadata_store
    from strata.server import get_state

    state = get_state()

    result = {
        "parquet_cache": state.planner.parquet_cache.stats(),
        "manifest_cache": state.planner.manifest_cache.stats(),
    }

    # Add SQLite store stats if available
    try:
        store = get_metadata_store()
        result["metadata_store"] = store.stats()
    except Exception:
        result["metadata_store"] = None

    return result


@router.get("/v1/config/timeouts")
async def get_timeout_config_v1():
    """Get all timeout configuration settings.

    Returns timeout configuration organized by category:
    - planning: Plan timeout settings
    - scanning: Scan timeout settings
    - qos_queue: QoS queue wait timeouts
    - fetching: Row group fetch timeouts
    - s3: S3 connection and request timeouts
    """
    from strata.server import get_state

    state = get_state()
    return state.config.get_timeout_config()


@router.post("/v1/metadata/cleanup")
async def cleanup_metadata_v1():
    """Remove stale metadata entries from the SQLite store.

    Scans all cached parquet metadata entries and removes those where:
    - The file no longer exists on disk
    - The file has been modified (different mtime or size)

    This is automatically run on server startup, but can be triggered
    manually if needed (e.g., after bulk file operations).

    Returns the number of stale entries removed.
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
