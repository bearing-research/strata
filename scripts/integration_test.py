#!/usr/bin/env python3
"""Integration test script for Strata with PostgreSQL catalog.

Starts Strata server, creates test tables, runs scans, verifies results.
Requires PostgreSQL to be running (via docker-compose.test.yml).

Usage:
    # Start PostgreSQL first
    docker-compose -f docker-compose.test.yml up -d

    # Run tests
    STRATA_CATALOG_URI=postgresql://strata:strata@localhost:5432/iceberg_catalog \
        uv run python scripts/integration_test.py

    # Or with --start-server flag to auto-start server
    STRATA_CATALOG_URI=postgresql://strata:strata@localhost:5432/iceberg_catalog \
        uv run python scripts/integration_test.py --start-server
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import httpx
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog

# Test configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
TIMEOUT = 30.0


def wait_for_postgres(uri: str, timeout: float = 30.0) -> bool:
    """Wait for PostgreSQL to be ready."""
    try:
        import psycopg2
    except ImportError:
        # Fall back to simple socket check if psycopg2 not available
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432

        start = time.time()
        while time.time() - start < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect((host, port))
                sock.close()
                return True
            except Exception:
                time.sleep(0.5)
        return False

    start = time.time()
    while time.time() - start < timeout:
        try:
            conn = psycopg2.connect(uri)
            conn.close()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=2.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def create_test_tables(warehouse_path: str, catalog_uri: str) -> dict[str, str]:
    """Create test tables in the catalog.

    Returns dict mapping table name to table URI.
    """
    # Use "strata" as catalog name to match server's PyIcebergCatalog
    catalog = SqlCatalog(
        "strata",
        uri=catalog_uri,
        warehouse=warehouse_path,
    )

    # Create namespace
    try:
        catalog.create_namespace("integration")
    except Exception:
        pass  # May already exist

    tables = {}

    # Table 1: Simple events table
    schema1 = pa.schema(
        [
            ("id", pa.int64()),
            ("value", pa.float64()),
            ("name", pa.string()),
        ]
    )

    try:
        catalog.drop_table("integration.events")
    except Exception:
        pass

    table1 = catalog.create_table("integration.events", schema1)

    # Insert test data
    data1 = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [1.1, 2.2, 3.3, 4.4, 5.5],
            "name": ["alice", "bob", "charlie", "david", "eve"],
        }
    )
    table1.append(data1)
    # Use file:// prefix for table URIs
    tables["events"] = f"file://{warehouse_path}#integration.events"

    # Table 2: Larger table for QoS testing
    schema2 = pa.schema(
        [
            ("id", pa.int64()),
            ("data", pa.string()),
        ]
    )

    try:
        catalog.drop_table("integration.large")
    except Exception:
        pass

    table2 = catalog.create_table("integration.large", schema2)

    # Insert more data (multiple row groups)
    for batch in range(5):
        data2 = pa.table(
            {
                "id": list(range(batch * 1000, (batch + 1) * 1000)),
                "data": [f"row_{i}" for i in range(batch * 1000, (batch + 1) * 1000)],
            }
        )
        table2.append(data2)
    tables["large"] = f"file://{warehouse_path}#integration.large"

    return tables


def test_health_endpoint(client: httpx.Client) -> bool:
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    resp = client.get("/health")
    if resp.status_code != 200:
        print(f"  FAIL: status={resp.status_code}")
        return False
    print("  PASS")
    return True


def _materialize_and_stream(
    client: httpx.Client,
    table_uri: str,
    columns: list[str],
    filters: list[dict] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[bool, str, int]:
    """Materialize a scan and stream the result. Returns (ok, message, bytes)."""
    body: dict = {
        "inputs": [table_uri],
        "transform": {
            "executor": "scan@v1",
            "params": {"columns": columns},
        },
    }
    if filters:
        body["transform"]["params"]["filters"] = filters

    resp = client.post("/v1/materialize", json=body, headers=headers or {})
    if resp.status_code != 200:
        return False, f"materialize failed: {resp.status_code} {resp.text}", 0

    data = resp.json()
    stream_url = data.get("stream_url")
    if not stream_url:
        # Artifact mode or cache hit without stream
        return True, f"hit={data.get('hit')}", 0

    resp = client.get(stream_url, headers=headers or {})
    if resp.status_code != 200:
        return False, f"stream failed: {resp.status_code} {resp.text}", 0

    return True, "ok", len(resp.content)


def test_basic_scan(client: httpx.Client, table_uri: str) -> bool:
    """Test a basic scan operation."""
    print(f"Testing basic scan on {table_uri}...")

    ok, msg, nbytes = _materialize_and_stream(client, table_uri, ["id", "name"])
    if not ok:
        print(f"  FAIL: {msg}")
        return False

    print(f"  PASS: received {nbytes} bytes ({msg})")
    return True


def test_filtered_scan(client: httpx.Client, table_uri: str) -> bool:
    """Test a scan with filters."""
    print(f"Testing filtered scan on {table_uri}...")

    ok, msg, nbytes = _materialize_and_stream(
        client,
        table_uri,
        ["id", "name"],
        filters=[{"column": "id", "op": ">", "value": 2}],
    )
    if not ok:
        print(f"  FAIL: {msg}")
        return False

    print(f"  PASS: received {nbytes} bytes ({msg})")
    return True


def test_multi_tenant_scan(client: httpx.Client, table_uri: str) -> bool:
    """Test scans with different tenant headers."""
    print(f"Testing multi-tenant scans on {table_uri}...")

    tenants = ["tenant-a", "tenant-b", "tenant-c"]

    for tenant in tenants:
        headers = {"X-Tenant-ID": tenant}
        ok, msg, _ = _materialize_and_stream(client, table_uri, ["id"], headers=headers)
        if not ok:
            print(f"  FAIL: scan for {tenant} failed: {msg}")
            return False

    print(f"  PASS: all {len(tenants)} tenants succeeded")
    return True


def test_concurrent_scans(client: httpx.Client, table_uri: str) -> bool:
    """Test concurrent scan operations."""
    import concurrent.futures

    print(f"Testing concurrent scans on {table_uri}...")

    def run_scan(scan_num: int) -> tuple[int, bool, str]:
        try:
            ok, message, _ = _materialize_and_stream(client, table_uri, ["id", "data"])
            return scan_num, ok, message
        except Exception as e:
            return scan_num, False, f"{type(e).__name__}: {e}"

    num_scans = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_scan, i) for i in range(num_scans)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    successes = sum(1 for _, success, _ in results if success)
    if successes != num_scans:
        print(f"  FAIL: {successes}/{num_scans} scans succeeded")
        # The helper already says whether it was the materialize or the stream
        # that failed, and with what status. Discarding it left a CI log that
        # reported the count and nothing else, which is not enough to tell a
        # QoS rejection from an expired stream from a real regression.
        for scan_num, ok, message in sorted(results):
            if not ok:
                print(f"    scan {scan_num}: {message}")
        return False

    print(f"  PASS: {successes}/{num_scans} concurrent scans succeeded")
    return True


def run_tests(tables: dict[str, str]) -> tuple[int, int]:
    """Run all integration tests. Returns (passed, failed)."""
    passed = 0
    failed = 0

    with httpx.Client(base_url=SERVER_URL, timeout=TIMEOUT) as client:
        tests = [
            (test_health_endpoint, (client,)),
            (test_basic_scan, (client, tables["events"])),
            (test_filtered_scan, (client, tables["events"])),
            (test_multi_tenant_scan, (client, tables["events"])),
            (test_concurrent_scans, (client, tables["large"])),
        ]

        for test_func, args in tests:
            try:
                if test_func(*args):
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                failed += 1

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="Strata integration tests")
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start Strata server automatically",
    )
    parser.add_argument(
        "--warehouse",
        default=None,
        help="Warehouse path (default: temp directory)",
    )
    args = parser.parse_args()

    # Check for PostgreSQL URI
    catalog_uri = os.environ.get("STRATA_CATALOG_URI")
    if not catalog_uri:
        print("ERROR: STRATA_CATALOG_URI environment variable required")
        print("Example: postgresql://strata:strata@localhost:5432/iceberg_catalog")
        sys.exit(1)

    print(f"Using catalog: {catalog_uri}")

    # Wait for PostgreSQL
    print("Waiting for PostgreSQL...")
    try:
        if not wait_for_postgres(catalog_uri):
            print("ERROR: PostgreSQL not available")
            sys.exit(1)
    except ImportError:
        print("WARNING: psycopg2 not installed, skipping PostgreSQL check")

    print("PostgreSQL ready")

    # Setup warehouse
    if args.warehouse:
        warehouse_path = args.warehouse
    else:
        warehouse_path = tempfile.mkdtemp(prefix="strata_test_")
    print(f"Using warehouse: {warehouse_path}")

    # Create test tables BEFORE starting server
    print("\nCreating test tables...")
    tables = create_test_tables(warehouse_path, catalog_uri)
    print(f"Created tables: {list(tables.keys())}")

    # Start server if requested
    server_proc = None
    if args.start_server:
        print("\nStarting Strata server...")
        env = os.environ.copy()
        env["STRATA_HOST"] = SERVER_HOST
        env["STRATA_PORT"] = str(SERVER_PORT)
        # Ensure catalog URI is passed to server
        env["STRATA_CATALOG_URI"] = catalog_uri
        print(f"  STRATA_CATALOG_URI={catalog_uri}")

        server_proc = subprocess.Popen(
            ["uv", "run", "python", "-m", "strata"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if not wait_for_server(SERVER_URL):
            print("ERROR: Server failed to start")
            server_proc.terminate()
            sys.exit(1)
        print("Server ready")
    else:
        print(f"\nExpecting server at {SERVER_URL}")
        if not wait_for_server(SERVER_URL, timeout=5.0):
            print("ERROR: Server not available. Start it or use --start-server")
            sys.exit(1)

    # Run tests
    print("\n" + "=" * 50)
    print("Running integration tests")
    print("=" * 50 + "\n")

    try:
        passed, failed = run_tests(tables)
    finally:
        if server_proc:
            print("\nStopping server...")
            server_proc.terminate()
            server_proc.wait(timeout=5)

    # Summary
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
