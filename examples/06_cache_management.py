#!/usr/bin/env python3
"""
Example 6: Cache Management

This example shows how to monitor and manage Strata's cache.
The cache is what makes Strata fast - understanding it helps you optimize.

What you'll learn:
    - How to check cache statistics
    - How to clear the cache
    - How to interpret cache metrics
"""

from strata_client import StrataClient

client = StrataClient(base_url="http://127.0.0.1:8765")

# Get cache and server metrics
metrics = client.metrics()

print("=== Cache Statistics ===")
print(f"Cache hits:       {metrics.get('cache_hits', 0)}")
print(f"Cache misses:     {metrics.get('cache_misses', 0)}")
print(f"Cache hit rate:   {metrics.get('cache_hit_rate', 0):.1%}")
print(f"Bytes from cache: {metrics.get('bytes_from_cache', 0):,}")
print(f"Bytes from storage: {metrics.get('bytes_from_storage', 0):,}")

print("\n=== Resource Limits ===")
limits = metrics.get("resource_limits", {})
print(f"Max concurrent scans: {limits.get('max_concurrent_scans', 'N/A')}")
print(f"Active scans:         {limits.get('active_scans', 0)}")
print(f"Scan timeout:         {limits.get('scan_timeout_seconds', 'N/A')}s")

# Clear the cache (useful for testing or freeing disk space)
# Note: This removes all cached data - next queries will be slower
# client.clear_cache()
# print("\nCache cleared!")

# Check server health
health = client.health()
print("\n=== Server Health ===")
print(f"Status: {health.get('status', 'unknown')}")

client.close()
