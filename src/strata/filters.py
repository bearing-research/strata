"""Filter types — the dependency-free slice of the type system.

These are wire-format value types (serialized to/from JSON when they cross the
client↔server boundary), so the client and the server each own their own copy:
neither package depends on the other. ``strata-client`` has the identical types
in ``strata_client.filters``. ``strata.types`` re-exports these for backward
compatibility, so ``from strata.types import Filter`` keeps working.

Standard library only — no third-party dependencies.
"""

import base64
import hashlib
import json
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


# Wire encoding for the non-JSON-native FilterValue types. Values stay
# JSON-native scalars (tagged strings) so a ``FilterValue`` field still
# validates and the fingerprint has a deterministic, type-distinguishing
# representation. Dependency-free.
#
# Edge case, inherited from the prior ``__datetime__:`` convention: a genuine
# string value that begins with one of these tags round-trips back to the
# richer type. Acceptable — operator-bearing string literals are rare and this
# keeps the wire JSON-native.
_FILTER_VALUE_DECODERS = {
    "__datetime__": datetime.fromisoformat,
    "__date__": date.fromisoformat,
    "__time__": time_of_day.fromisoformat,
    "__decimal__": Decimal,
    "__uuid__": uuid.UUID,
    "__bytes__": lambda s: base64.b64decode(s.encode("ascii")),
}


def serialize_filter_value(value: FilterValue) -> str | bool | int | float:
    """Encode a filter value as a JSON-native, round-trippable scalar.

    Primitives pass through; richer types (datetime/date/time/Decimal/UUID/bytes)
    become tagged strings that :func:`deserialize_filter_value` reconstructs.
    """
    # bool is a subclass of int — keep it ahead of the int/float passthrough.
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, datetime):
        return f"__datetime__:{value.isoformat()}"
    if isinstance(value, date):
        return f"__date__:{value.isoformat()}"
    if isinstance(value, time_of_day):
        return f"__time__:{value.isoformat()}"
    if isinstance(value, Decimal):
        return f"__decimal__:{value}"
    if isinstance(value, uuid.UUID):
        return f"__uuid__:{value}"
    if isinstance(value, bytes):
        return f"__bytes__:{base64.b64encode(value).decode('ascii')}"
    raise TypeError(f"Unsupported filter value type: {type(value).__name__}")


def deserialize_filter_value(value: str | bool | int | float) -> FilterValue:
    """Inverse of :func:`serialize_filter_value`. Untagged scalars pass through."""
    if isinstance(value, str):
        tag, sep, encoded = value.partition(":")
        if sep:
            decoder = _FILTER_VALUE_DECODERS.get(tag)
            if decoder is not None:
                return decoder(encoded)
    return value


def compute_filter_fingerprint(filters: list[Filter] | None) -> str:
    """Compute a stable fingerprint for a list of filters.

    Used for cache keying when filters affect file-level pruning, and for
    identity-materialize provenance. Returns a deterministic hash that is stable
    across runs and order-independent.

    Args:
        filters: List of Filter objects (may be None or empty)

    Returns:
        16-character hex string, or "nofilter" if no filters
    """
    if not filters:
        return "nofilter"

    # Canonical JSON over explicit fields. Structured encoding so distinct
    # (column, op, value) triples can't collide the way a delimiter-free
    # concatenation could — e.g. (column='a>', op='=') vs (column='a', op='>=')
    # both used to serialize to 'a>='. serialize_filter_value keeps the value
    # JSON-native *and* type-distinguishing (str '1' != int 1).
    items = sorted(
        (
            {"column": f.column, "op": f.op.value, "value": serialize_filter_value(f.value)}
            for f in filters
        ),
        key=lambda d: json.dumps(d, sort_keys=True),
    )
    combined = json.dumps(items, separators=(",", ":"), sort_keys=True)
    return hashlib.md5(combined.encode()).hexdigest()[:16]
