#!/usr/bin/env python3
"""
Example 12: Direct Artifact Upload with put()

This example shows how to use Strata's put() API for direct artifact upload
with full provenance tracking and deduplication.

The put() API supports multiple data types:
- dict: JSON data (nested dicts, lists, primitives)
- pa.Table: Apache Arrow Table
- pd.DataFrame: Pandas DataFrame
- pl.DataFrame: Polars DataFrame
- bytes: Raw Arrow IPC bytes

This pattern is useful for any system that:
- Executes computations locally (LLM calls, ML inference, etc.)
- Needs to persist results with full provenance
- Wants automatic deduplication (cache hits on identical requests)
- Requires lineage tracking between steps

What you'll learn:
    - How to use put() for direct artifact upload with different data types
    - How to track lineage through input references
    - How to detect cache hits for replay/restart
    - How to structure opaque transform specs for provenance
"""

from strata_client import StrataClient


def delibera_transform(operation: str, step_id: str, role: str, code_hash: str, **kwargs) -> dict:
    """Create a Delibera transform spec.

    The transform spec is opaque to Strata - it's used only for provenance
    hash computation. Delibera includes:
    - operation: WORK, VOTE, AGGREGATE, etc.
    - step_id: Which step in the protocol
    - role: Planner, Proposer, Validator, etc.
    - code_hash: Hash of the step implementation code
    """
    return {
        "executor": "delibera@v1",
        "params": {
            "operation": operation,
            "step_id": step_id,
            "role": role,
            "code_hash": code_hash,
            **kwargs,
        },
    }


# =============================================================================
# Example 1: Basic put_json usage
# =============================================================================


def basic_put_json():
    """Basic example of persisting a locally-computed result."""
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Simulate a locally-computed result (e.g., LLM call)
        result_data = {
            "proposal": "Implement feature X with approach Y",
            "claims": [
                {"id": 1, "text": "Approach Y is more maintainable"},
                {"id": 2, "text": "Approach Y has better performance"},
            ],
            "confidence": 0.85,
        }

        # Persist with provenance tracking
        artifact = client.put_json(
            inputs=[],  # No inputs for root artifacts
            transform=delibera_transform(
                operation="WORK",
                step_id="PROPOSE",
                role="Proposer",
                code_hash="sha256:abc123",
            ),
            data=result_data,
            name="proposal_v1",
        )

        print(f"Artifact URI: {artifact.uri}")
        print(f"Cache hit: {artifact.cache_hit}")
        print(f"Name: {artifact.name}")

        # Retrieve the data
        retrieved = client.get_json(artifact.uri)
        print(f"Retrieved proposal: {retrieved['proposal']}")


# =============================================================================
# Example 2: Cache hit detection (replay scenario)
# =============================================================================


def cache_hit_detection():
    """Demonstrate cache hit on identical requests (replay)."""
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        transform = delibera_transform(
            operation="WORK",
            step_id="VALIDATE",
            role="Validator",
            code_hash="sha256:validator_v1",
        )
        data = {"validation_result": "passed", "score": 0.95}

        # First call - should not be a cache hit
        artifact1 = client.put_json(inputs=[], transform=transform, data=data)
        print(f"First call - Cache hit: {artifact1.cache_hit}")

        # Second call with same inputs+transform+data - should be cache hit
        artifact2 = client.put_json(inputs=[], transform=transform, data=data)
        print(f"Second call - Cache hit: {artifact2.cache_hit}")
        print(f"Same artifact: {artifact1.uri == artifact2.uri}")


# =============================================================================
# Example 3: Lineage tracking (step dependencies)
# =============================================================================


def lineage_tracking():
    """Track lineage between Delibera steps."""
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Step 1: Persist protocol specification
        protocol = client.put_json(
            inputs=[],
            transform={"executor": "protocol@v1", "params": {}},
            data={
                "version": "tree_v1",
                "steps": ["PLAN", "PROPOSE", "VALIDATE", "AGGREGATE"],
            },
            name="protocol_tree_v1",
        )
        print(f"Protocol: {protocol.uri}")

        # Step 2: Persist run constraints
        constraints = client.put_json(
            inputs=[],
            transform={"executor": "constraints@v1", "params": {}},
            data={
                "risk_tolerance": 0.1,
                "scope": "narrow",
                "max_iterations": 5,
            },
            name="constraints_run_001",
        )
        print(f"Constraints: {constraints.uri}")

        # Step 3: Planner step (depends on protocol + constraints)
        planner_result = client.put_json(
            inputs=[protocol.uri, constraints.uri],  # Lineage!
            transform=delibera_transform(
                operation="WORK",
                step_id="PLAN",
                role="Planner",
                code_hash="sha256:planner_v1",
            ),
            data={
                "branches": ["option_a", "option_b", "option_c"],
                "reasoning": "Given constraints, these options are viable...",
            },
        )
        print(f"Planner: {planner_result.uri}")

        # Step 4: Proposer step (depends on planner output)
        proposer_result = client.put_json(
            inputs=[planner_result.uri, constraints.uri],  # Chains from planner
            transform=delibera_transform(
                operation="WORK",
                step_id="PROPOSE",
                role="Proposer",
                code_hash="sha256:proposer_v2",
                temperature=0.1,  # Additional params
            ),
            data={
                "proposal": "Recommend option_a because...",
                "claims": [
                    {"id": 1, "text": "Lowest risk", "evidence": ["ref1", "ref2"]},
                    {"id": 2, "text": "Best fit for scope", "evidence": ["ref3"]},
                ],
            },
        )
        print(f"Proposer: {proposer_result.uri}")

        # All artifacts form a DAG with explicit lineage
        print("\nLineage chain:")
        print("  protocol + constraints -> planner -> proposer")


# =============================================================================
# Example 4: Crash recovery scenario
# =============================================================================


def crash_recovery():
    """Demonstrate how Strata enables crash recovery for Delibera.

    If Delibera crashes mid-run, it can restart and:
    1. Completed steps return cache hits (no recomputation)
    2. Failed steps are retried automatically
    3. Lineage is preserved for debugging
    """
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Simulate a run with multiple steps
        steps = [
            ("PLAN", {"branches": ["a", "b"]}),
            ("PROPOSE", {"proposal": "Choose a", "claims": []}),
            ("VALIDATE", {"result": "passed"}),
        ]

        completed_steps = []
        for step_id, data in steps:
            # Determine inputs from previous steps
            inputs = [s["uri"] for s in completed_steps] if completed_steps else []

            artifact = client.put_json(
                inputs=inputs,
                transform=delibera_transform(
                    operation="WORK",
                    step_id=step_id,
                    role=step_id.title() + "er",
                    code_hash=f"sha256:{step_id.lower()}_v1",
                ),
                data=data,
            )

            status = "CACHE HIT" if artifact.cache_hit else "computed"
            print(f"Step {step_id}: {status}")
            completed_steps.append({"step": step_id, "uri": artifact.uri})

        print("\n--- Simulating restart (all steps should be cache hits) ---\n")

        # Restart: all steps should be cache hits
        completed_steps = []
        for step_id, data in steps:
            inputs = [s["uri"] for s in completed_steps] if completed_steps else []

            artifact = client.put_json(
                inputs=inputs,
                transform=delibera_transform(
                    operation="WORK",
                    step_id=step_id,
                    role=step_id.title() + "er",
                    code_hash=f"sha256:{step_id.lower()}_v1",
                ),
                data=data,
            )

            status = "CACHE HIT" if artifact.cache_hit else "computed"
            print(f"Step {step_id}: {status}")
            completed_steps.append({"step": step_id, "uri": artifact.uri})


# =============================================================================
# Example 5: Branching deliberation (exploring alternatives)
# =============================================================================


def branching_deliberation():
    """Demonstrate branching execution paths in Delibera.

    Multiple proposers can explore different branches, each persisted
    as a separate artifact with shared lineage to the planner.
    """
    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Shared planner output
        planner = client.put_json(
            inputs=[],
            transform=delibera_transform(
                operation="WORK",
                step_id="PLAN",
                role="Planner",
                code_hash="sha256:planner_branch_v1",
            ),
            data={"branches": ["aggressive", "conservative", "balanced"]},
        )
        print(f"Planner: {planner.uri}")

        # Multiple proposers exploring different branches
        proposals = []
        for branch in ["aggressive", "conservative", "balanced"]:
            proposal = client.put_json(
                inputs=[planner.uri],  # All share same parent
                transform=delibera_transform(
                    operation="WORK",
                    step_id="PROPOSE",
                    role="Proposer",
                    code_hash="sha256:proposer_v1",
                    branch=branch,  # Different branch param
                ),
                data={"branch": branch, "proposal": f"Take {branch} approach..."},
            )
            proposals.append(proposal)
            print(f"Proposal ({branch}): {proposal.uri}")

        # Aggregator combines all proposals
        aggregator = client.put_json(
            inputs=[p.uri for p in proposals],  # All proposals as inputs
            transform=delibera_transform(
                operation="AGGREGATE",
                step_id="DECIDE",
                role="Aggregator",
                code_hash="sha256:aggregator_v1",
            ),
            data={
                "selected": "balanced",
                "reasoning": "Best risk/reward tradeoff",
                "votes": {"aggressive": 1, "conservative": 1, "balanced": 3},
            },
        )
        print(f"Aggregator: {aggregator.uri}")


# =============================================================================
# Example 6: Using put() with Arrow Table
# =============================================================================


def put_with_arrow_table():
    """Demonstrate put() with Arrow Table data.

    Arrow Tables are efficient for columnar data and are the native
    format for Strata's storage layer.
    """
    import pyarrow as pa

    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Create an Arrow Table with computed results
        table = pa.table(
            {
                "feature_id": [1, 2, 3, 4, 5],
                "embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
                "score": [0.95, 0.87, 0.72, 0.91, 0.88],
            }
        )

        artifact = client.put(
            inputs=[],
            transform={"executor": "embeddings@v1", "params": {"model": "text-ada-002"}},
            data=table,  # Arrow Table
        )

        print(f"Artifact: {artifact.uri}")
        print(f"Cache hit: {artifact.cache_hit}")

        # Retrieve as Arrow Table
        retrieved = client.fetch(artifact.uri)
        print(f"Retrieved {retrieved.num_rows} rows, {retrieved.num_columns} columns")


# =============================================================================
# Example 7: Using put() with Pandas DataFrame
# =============================================================================


def put_with_pandas():
    """Demonstrate put() with Pandas DataFrame.

    DataFrames are automatically converted to Arrow for storage.
    """
    import pandas as pd

    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Create a Pandas DataFrame with analysis results
        df = pd.DataFrame(
            {
                "metric": ["accuracy", "precision", "recall", "f1"],
                "value": [0.95, 0.92, 0.89, 0.90],
                "threshold": [0.5, 0.5, 0.5, 0.5],
            }
        )

        artifact = client.put(
            inputs=[],
            transform={"executor": "eval@v1", "params": {"model_version": "2.0"}},
            data=df,  # Pandas DataFrame
        )

        print(f"Artifact: {artifact.uri}")

        # Retrieve as Pandas DataFrame
        retrieved_df = artifact.to_pandas()
        print(f"Retrieved DataFrame:\n{retrieved_df}")


# =============================================================================
# Example 8: Mixed data types in a pipeline
# =============================================================================


def mixed_data_pipeline():
    """Demonstrate a pipeline using different data types at each stage."""
    import pyarrow as pa

    with StrataClient(base_url="http://127.0.0.1:8765") as client:
        # Stage 1: JSON config
        config = client.put(
            inputs=[],
            transform={"executor": "config@v1", "params": {}},
            data={"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000},
        )
        print(f"Config: {config.uri}")

        # Stage 2: Arrow Table of input data
        input_data = pa.table(
            {
                "id": [1, 2, 3],
                "text": ["Query 1", "Query 2", "Query 3"],
            }
        )
        inputs_artifact = client.put(
            inputs=[config.uri],
            transform={"executor": "prepare@v1", "params": {}},
            data=input_data,
        )
        print(f"Inputs: {inputs_artifact.uri}")

        # Stage 3: JSON results from LLM
        results = client.put(
            inputs=[config.uri, inputs_artifact.uri],
            transform={"executor": "llm@v1", "params": {}},
            data={
                "responses": [
                    {"id": 1, "response": "Answer 1", "tokens": 50},
                    {"id": 2, "response": "Answer 2", "tokens": 75},
                    {"id": 3, "response": "Answer 3", "tokens": 60},
                ],
                "total_cost": 0.015,
            },
        )
        print(f"Results: {results.uri}")

        # All three artifacts form a lineage chain
        print("\nPipeline complete:")
        print(f"  config ({type({})}) -> inputs (Arrow) -> results (JSON)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=== Basic put_json ===")
    # basic_put_json()

    print("\n=== Cache Hit Detection ===")
    # cache_hit_detection()

    print("\n=== Lineage Tracking ===")
    # lineage_tracking()

    print("\n=== Crash Recovery ===")
    # crash_recovery()

    print("\n=== Branching Deliberation ===")
    # branching_deliberation()

    print("\n=== Arrow Table ===")
    # put_with_arrow_table()

    print("\n=== Pandas DataFrame ===")
    # put_with_pandas()

    print("\n=== Mixed Data Pipeline ===")
    # mixed_data_pipeline()

    print("\nNote: Uncomment the function calls to run examples")
    print("Requires a running Strata server with artifacts enabled")
