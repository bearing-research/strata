"""Local transform execution for the server's embedded executor.

Internal: users call ``client.materialize()``. Transforms come from the registry
(``@register_transform``); ``scan@v1`` is server-only and cannot run here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import strata.transforms  # noqa: F401 - registers transforms
from strata.transforms.base import _run_transform, get_transform, list_transforms

if TYPE_CHECKING:
    import pyarrow as pa


def _run_local(
    build_spec: dict[str, Any],
    input_tables: dict[str, pa.Table],
) -> pa.Table:
    """Execute a transform in-process from a build spec.

    ``build_spec`` carries ``executor`` (e.g. ``"duckdb_sql@v1"``, optional ``local://``
    prefix), ``params``, and ``input_uris``, which fixes input order. ``input_tables`` maps
    each input URI to its Arrow table.

    Raises
    ------
    ValueError
        If the executor is unknown or an input table is missing.
    """
    executor = build_spec.get("executor", "")
    params = build_spec.get("params", {})
    input_uris = build_spec.get("input_uris", [])

    inputs: list[pa.Table] = []
    for uri in input_uris:
        table = input_tables.get(uri)
        if table is None:
            raise ValueError(f"Missing input table for URI: {uri}")
        inputs.append(table)

    return _run_transform(executor, inputs, params)


run_local = _run_local


__all__ = [
    "_run_local",  # Internal: for embedded executor use
    "run_local",  # Deprecated alias
    "get_transform",
    "list_transforms",
]
