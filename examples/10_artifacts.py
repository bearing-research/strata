#!/usr/bin/env python3
"""
Example 10: Artifact Workflow

This example shows how to use Strata's artifact system for caching
transform results. Artifacts are immutable, versioned outputs that
can depend on source tables or other artifacts.

What you'll learn:
    - How to materialize artifacts using client.materialize()
    - How to chain artifacts (use one artifact as input to another)
    - How to access artifact data via the Artifact object
    - How to track artifact lineage
    - How to check artifact staleness
"""

from strata_client import StrataClient

# Source table URI
SOURCE_URI = "file:///path/to/warehouse#analytics.events"


def sql_transform(sql: str) -> dict:
    """Helper to create a DuckDB SQL transform spec."""
    return {"ref": "duckdb_sql@v1", "params": {"sql": sql}}


# =============================================================================
# Example 1: Materialize an artifact from a source table
# =============================================================================


def basic_materialize():
    """Materialize an artifact with a simple SQL transform."""
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Materialize returns an Artifact object
        artifact = client.materialize(
            inputs=[SOURCE_URI],
            transform=sql_transform("""
                SELECT
                    category,
                    COUNT(*) as count,
                    SUM(value) as total_value
                FROM input0
                GROUP BY category
            """),
            name="daily_summary",
        )

        print(f"Artifact URI: {artifact.uri}")
        print(f"Cache hit: {artifact.cache_hit}")
        print(f"Execution mode: {artifact.execution}")

        # Get metadata
        info = artifact.info()
        print("\nArtifact Info:")
        print(f"  Size: {info.get('size_bytes', 'N/A')} bytes")
        print(f"  Rows: {info.get('row_count', 'N/A')}")

        # Access data directly
        df = artifact.to_pandas()
        print(f"\nData preview:\n{df.head()}")


# =============================================================================
# Example 2: Chain artifacts (use output as input to another artifact)
# =============================================================================


def chained_artifacts():
    """Chain artifacts together - output of one becomes input to another."""
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # First artifact: filter events
        filtered = client.materialize(
            inputs=[SOURCE_URI],
            transform=sql_transform("SELECT * FROM input0 WHERE value > 100"),
            name="filtered_events",
        )
        print(f"Stage 1 - Filtered: {filtered.uri}")

        # Second artifact: aggregate using first artifact
        # Use the artifact's URI as input
        stats = client.materialize(
            inputs=[filtered.uri],
            transform=sql_transform("""
                SELECT category, AVG(value) as avg_value
                FROM input0
                GROUP BY category
            """),
            name="category_stats",
        )
        print(f"Stage 2 - Stats: {stats.uri}")

        # Both artifacts are cached, so re-running is instant
        stats2 = client.materialize(
            inputs=[filtered.uri],
            transform=sql_transform("""
                SELECT category, AVG(value) as avg_value
                FROM input0
                GROUP BY category
            """),
            name="category_stats",
        )
        print(f"Cache hit: {stats2.cache_hit}")


# =============================================================================
# Example 3: Query artifact lineage
# =============================================================================


def view_lineage():
    """View the dependency graph for an artifact."""
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Get an artifact by name
        artifact = client.get_artifact_by_name("category_stats")

        # Get upstream lineage (what inputs led to this artifact)
        lineage = artifact.lineage(direction="upstream", max_depth=5)

        print("\nUpstream Lineage:")
        for node in lineage.get("nodes", []):
            print(f"  {node.get('artifact_id', 'unknown')}@v{node.get('version', '?')}")
            for input_ref in node.get("inputs", []):
                print(f"    <- {input_ref}")

        # Get downstream dependents (what uses this artifact)
        dependents = artifact.dependents(max_depth=3)
        print(f"\nDependents: {len(dependents.get('dependents', []))} artifacts")


# =============================================================================
# Example 4: Check if artifacts are stale
# =============================================================================


def check_staleness():
    """Check if a named artifact needs to be rebuilt."""
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Quick check
        is_stale = client.is_artifact_stale("daily_summary")
        print(f"Is 'daily_summary' stale: {is_stale}")

        # Detailed status
        status = client.get_name_status("daily_summary")
        print("\nDetailed Status:")
        print(f"  Version: {status.get('version')}")
        print(f"  Is stale: {status.get('is_stale')}")
        if status.get("is_stale"):
            print(f"  Reason: {status.get('stale_reason')}")
            print(f"  Changed inputs: {status.get('changed_inputs')}")


# =============================================================================
# Example 5: Explain what would happen without running
# =============================================================================


def explain_materialize():
    """Dry-run to see if materialization is needed."""
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        result = client.explain_materialize(
            inputs=[SOURCE_URI],
            transform=sql_transform("SELECT * FROM input0"),
            name="my_transform",
        )

        print("\nExplain Materialize:")
        print(f"  Would hit cache: {result.get('cache_hit')}")
        print(f"  Provenance hash: {result.get('provenance_hash', '')[:16]}...")
        if result.get("is_stale"):
            print(f"  Stale reason: {result.get('stale_reason')}")


# =============================================================================
# Example 6: Force refresh and access data formats
# =============================================================================


def data_access_patterns():
    """Different ways to access artifact data."""
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Force recompute even if cached
        artifact = client.materialize(
            inputs=[SOURCE_URI],
            transform=sql_transform("SELECT category, COUNT(*) as cnt FROM input0 GROUP BY 1"),
            name="category_counts",
            refresh=True,  # Force rebuild
        )

        # Access as Arrow Table (most efficient)
        table = artifact.to_table()
        print(f"Arrow Table: {table.num_rows} rows, {table.num_columns} cols")

        # Access as Pandas DataFrame
        df = artifact.to_pandas()
        print(f"Pandas DataFrame:\n{df}")

        # Access as Polars DataFrame (if installed)
        try:
            pl_df = artifact.to_polars()
            print(f"Polars DataFrame:\n{pl_df}")
        except ImportError:
            print("Polars not installed")


# =============================================================================
# Example 7: Using custom transform for advanced use cases
# =============================================================================


def advanced_transform():
    """Use custom transform specs for advanced executors."""
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Use a custom transform with additional params
        artifact = client.materialize(
            inputs=[SOURCE_URI],
            transform={
                "ref": "duckdb_sql@v1",
                "params": {
                    "sql": """
                        SELECT
                            category,
                            percentile_cont(0.5) WITHIN GROUP (ORDER BY value) as median
                        FROM input0
                        GROUP BY category
                    """,
                },
            },
            name="category_medians",
        )
        print(f"Advanced transform result: {artifact.uri}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=== Basic Materialize ===")
    # basic_materialize()

    print("\n=== Chained Artifacts ===")
    # chained_artifacts()

    print("\n=== Lineage ===")
    # view_lineage()

    print("\n=== Staleness ===")
    # check_staleness()

    print("\n=== Explain ===")
    # explain_materialize()

    print("\n=== Data Access ===")
    # data_access_patterns()

    print("\n=== Advanced Transform ===")
    # advanced_transform()

    print("\nNote: Uncomment the function calls to run examples")
    print("Requires a running Strata server with artifacts enabled")
