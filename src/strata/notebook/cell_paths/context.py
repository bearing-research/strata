"""What an extracted cell path shares with the rest of the executor."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strata.notebook.execution_result import CellExecutionResult
    from strata.notebook.executor import _CellProvenance


@dataclass(frozen=True)
class PathContext:
    """Only what a moved path uses, added as each path moves."""

    # The standard provenance of a cell, from its id and source.
    compute_provenance: Callable[[str, str], Awaitable[_CellProvenance]]
    # What the cell's run moved: by mount name, and by table URI.
    moved: Callable[[str], tuple[dict[str, int | None], dict[str, int]]]
    # Run the cell's upstreams, from its id, start time and execution method: their failure
    # as this cell's result, or ``None``.
    materialize_upstreams_or_failure: Callable[
        [str, float, str], Awaitable[CellExecutionResult | None]
    ]
    # Store displays that came as text, not files; what was stored, with its artifact URIs.
    store_inline_display_outputs: Callable[..., list[dict[str, Any]]]
