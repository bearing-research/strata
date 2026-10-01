"""Scan transform (scan@v1) - identity read from Iceberg tables.

The scan transform reads data from an Iceberg table with optional:
- Column projection
- Row filtering
- Snapshot pinning

This is a "virtual" transform - it's handled internally by Strata's planner
and cache, not executed as a separate step. We define it here for:
1. Parameter validation
2. Documentation
3. Consistency with other transforms

Note: scan@v1 cannot be executed locally via Transform.execute() because
it requires access to the Iceberg catalog and Parquet files. The server
handles this transform specially.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

from strata.transforms.base import Transform, register_transform

if TYPE_CHECKING:
    import pyarrow as pa


class FilterSpec(BaseModel):
    """Row filter specification.

    Attributes:
        column: Column name to filter on
        op: Comparison operator ("=", "!=", "<", "<=", ">", ">=")
        value: Value to compare against
    """

    column: str
    op: str
    value: Any

    @field_validator("op")
    @classmethod
    def validate_op(cls, v: str) -> str:
        valid_ops = {"=", "!=", "<", "<=", ">", ">="}
        if v not in valid_ops:
            raise ValueError(f"Invalid operator: {v}. Must be one of {valid_ops}")
        return v


class ScanParams(BaseModel):
    """Parameters for the scan@v1 transform.

    Attributes:
        columns: Column projection (None = all columns)
        filters: Row filters for predicate pushdown
        snapshot_id: Specific snapshot to read (None = current)
    """

    columns: list[str] | None = None
    filters: list[FilterSpec] | None = None
    snapshot_id: int | None = None


@register_transform("scan@v1")
class ScanTransform(Transform[ScanParams]):
    """Identity transform that reads from Iceberg tables.

    This transform is handled specially by the Strata server:
    1. Resolves table URI to Iceberg snapshot
    2. Plans read tasks (row groups to fetch)
    3. Checks/populates cache
    4. Streams cached Arrow data

    It cannot be executed locally via execute() because it requires
    server-side resources (catalog access, cache).

    Example:
        client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {
                "columns": ["id", "value"],
                "filters": [{"column": "value", "op": ">", "value": 100}],
            }},
        )
    """

    Params = ScanParams

    def validate(self, inputs: list[pa.Table], params: ScanParams) -> None:
        """Validate scan parameters.

        Note: This is only used for parameter validation, not execution.
        """
        # Inputs here are already resolved tables, so there is nothing to check.
        pass

    def execute(self, inputs: list[pa.Table], params: ScanParams) -> pa.Table:
        """Execute is not supported for scan@v1.

        scan@v1 is handled internally by the server. If you need to
        execute locally, use the already-fetched table data directly.

        Raises:
            NotImplementedError: Always (scan@v1 cannot run locally)
        """
        raise NotImplementedError(
            "scan@v1 is handled by the Strata server and cannot be executed locally. "
            "Use client.materialize() to fetch data from Iceberg tables."
        )


def build_scan_transform(
    columns: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    snapshot_id: int | None = None,
) -> dict[str, Any]:
    """Build a scan@v1 transform specification.

    Args:
        columns: Column projection (None = all columns)
        filters: Row filters as dicts with column, op, value
        snapshot_id: Specific snapshot to read

    Returns:
        Transform spec dict for materialize()

    Example:
        transform = build_scan_transform(
            columns=["id", "value"],
            filters=[{"column": "value", "op": ">", "value": 100}],
        )
        client.materialize(inputs=[table_uri], transform=transform)
    """
    params: dict[str, Any] = {}
    if columns is not None:
        params["columns"] = columns
    if filters is not None:
        params["filters"] = filters
    if snapshot_id is not None:
        params["snapshot_id"] = snapshot_id

    return {"executor": "scan@v1", "params": params}
