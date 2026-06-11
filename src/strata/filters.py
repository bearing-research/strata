"""Filter types — the dependency-free slice of the type system.

These are wire-format value types (serialized to/from JSON when they cross the
client↔server boundary), so the client and the server each own their own copy:
neither package depends on the other. ``strata-client`` has the identical types
in ``strata_client.filters``. ``strata.types`` re-exports these for backward
compatibility, so ``from strata.types import Filter`` keeps working.

Standard library only — no third-party dependencies.
"""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as time_of_day
from decimal import Decimal
from enum import Enum
from typing import Protocol, cast

type FilterValue = (
    str | bool | int | float | bytes | uuid.UUID | Decimal | datetime | date | time_of_day
)


class SupportsOrdering(Protocol):
    """Structural protocol for values that support rich ordering."""

    def __lt__(self, other: object, /) -> bool: ...

    def __le__(self, other: object, /) -> bool: ...

    def __gt__(self, other: object, /) -> bool: ...

    def __ge__(self, other: object, /) -> bool: ...


class FilterOp(Enum):
    """Supported filter operations."""

    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


@dataclass(frozen=True)
class Filter:
    """A simple column filter for pruning."""

    column: str
    op: FilterOp
    value: FilterValue

    def matches_stats(self, min_val: FilterValue | None, max_val: FilterValue | None) -> bool:
        """Check if this filter could match given min/max statistics.

        Returns True if the row group might contain matching rows.
        """
        if min_val is None or max_val is None:
            return True  # No stats, can't prune

        min_orderable = cast(SupportsOrdering, min_val)
        max_orderable = cast(SupportsOrdering, max_val)
        filter_value = cast(SupportsOrdering, self.value)

        match self.op:
            case FilterOp.EQ:
                return min_orderable <= filter_value <= max_orderable
            case FilterOp.NE:
                return not (min_val == max_val == self.value)
            case FilterOp.LT:
                return min_orderable < filter_value
            case FilterOp.LE:
                return min_orderable <= filter_value
            case FilterOp.GT:
                return max_orderable > filter_value
            case FilterOp.GE:
                return max_orderable >= filter_value


def compute_filter_fingerprint(filters: list[Filter] | None) -> str:
    """Compute a stable fingerprint for a list of filters.

    Used for cache keying when filters affect file-level pruning.
    Returns a deterministic hash that is stable across runs.

    Args:
        filters: List of Filter objects (may be None or empty)

    Returns:
        16-character hex string, or "nofilter" if no filters
    """
    if not filters:
        return "nofilter"

    # Sort filters deterministically by (column, op, value_repr)
    # This ensures the same filters in different order produce the same fingerprint
    parts = []
    for f in sorted(filters, key=lambda x: (x.column, x.op.value, repr(x.value))):
        # Normalize datetime values to ISO format for consistency
        if isinstance(f.value, datetime):
            value_str = f.value.isoformat()
        else:
            value_str = repr(f.value)
        parts.append(f"{f.column}{f.op.value}{value_str}")

    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()[:16]
