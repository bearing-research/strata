"""The ``scan@v1`` transform: identity read from an Iceberg table.

Supports column projection, row filters and snapshot pinning. The server's
planner and cache execute it; this module only validates parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

from strata.transforms.base import Transform, register_transform

if TYPE_CHECKING:
    import pyarrow as pa


class FilterSpec(BaseModel):
    """Row filter ``column op value``; op is ``=``, ``!=``, ``<``, ``<=``, ``>`` or ``>=``."""

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
    """Parameters for ``scan@v1``.

    ``columns`` None projects all columns; ``snapshot_id`` None reads the current snapshot.
    """

    columns: list[str] | None = None
    filters: list[FilterSpec] | None = None
    snapshot_id: int | None = None


@register_transform("scan@v1")
class ScanTransform(Transform[ScanParams]):
    """Identity transform that reads from Iceberg tables; executed only by the server.

    Example:
        client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {"columns": ["id", "value"]}},
        )
    """

    Params = ScanParams

    def validate(self, inputs: list[pa.Table], params: ScanParams) -> None:
        """Validate scan parameters."""
        # Inputs here are already resolved tables, so there is nothing to check.
        pass

    def execute(self, inputs: list[pa.Table], params: ScanParams) -> pa.Table:
        """Always raise: ``scan@v1`` needs server-side catalog and cache access.

        Raises:
            NotImplementedError: Always.
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
    """Build a ``scan@v1`` transform spec for ``materialize()``.

    Args:
        columns: Columns to project (None for all).
        filters: Row filters as dicts with ``column``, ``op``, ``value``.
        snapshot_id: Snapshot to read (None for current).

    Example:
        transform = build_scan_transform(
            columns=["id"], filters=[{"column": "value", "op": ">", "value": 100}]
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
