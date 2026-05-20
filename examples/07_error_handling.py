#!/usr/bin/env python3
"""
Example 7: Error Handling

This example shows how to handle common errors when using Strata.

What you'll learn:
    - Common error types and their meanings
    - How to handle connection errors
    - How to handle materialize/fetch errors
"""

import httpx

from strata.client import StrataClient

# Example 1: Handle connection errors
try:
    client = StrataClient(base_url="http://127.0.0.1:8765")
    client.health()  # Test connection
except httpx.ConnectError:
    print("ERROR: Cannot connect to Strata server")
    print("Make sure the server is running: strata-notebook")
    exit(1)

# Example 2: Handle invalid table URI
try:
    artifact = client.materialize(
        inputs=["invalid://table/uri"],
        transform={"executor": "scan@v1", "params": {}},
    )
except Exception as e:
    print(f"Invalid table URI: {e}")

# Example 3: Handle table not found
try:
    artifact = client.materialize(
        inputs=["file:///nonexistent#db.table"],
        transform={"executor": "scan@v1", "params": {}},
    )
except Exception as e:
    print(f"Table not found: {e}")

# Example 4: Handle scan timeout (504)
# Large scans may exceed the server's scan_timeout_seconds
try:
    artifact = client.materialize(
        inputs=["file:///warehouse#db.huge_table"],
        transform={"executor": "scan@v1", "params": {}},
    )
    table = client.fetch(artifact.uri)
except Exception as e:
    if "504" in str(e) or "timeout" in str(e).lower():
        print("Scan timed out - try adding filters to reduce data")
    else:
        raise

# Example 5: Handle response too large (413)
# Scans exceeding max_response_bytes will fail
try:
    artifact = client.materialize(
        inputs=["file:///warehouse#db.huge_table"],
        transform={"executor": "scan@v1", "params": {}},
    )
    table = client.fetch(artifact.uri)
except Exception as e:
    if "413" in str(e) or "exceeds limit" in str(e).lower():
        print("Response too large - use column projection or filters")
    else:
        raise

# Example 6: Handle server at capacity (503)
try:
    artifact = client.materialize(
        inputs=["file:///warehouse#db.table"],
        transform={"executor": "scan@v1", "params": {}},
    )
    table = client.fetch(artifact.uri)
except Exception as e:
    if "503" in str(e):
        print("Server at capacity - retry later")
    else:
        raise

client.close()
