"""Filter wire types, standard library only.

``strata_client.filters`` holds an identical copy so neither package depends on
the other; keep the two in sync. ``strata.types`` re-exports these.
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
        """Return True if a row group with these min/max stats might contain matching rows.

        Missing stats never prune.
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
                # Parquet leaves NaN out of float min/max, so min == max ==
                # value doesn't rule out a NaN row, and NaN != value holds.
                if isinstance(min_val, float):
                    return True
                return not (min_val == max_val == self.value)
            case FilterOp.LT:
                return min_orderable < filter_value
            case FilterOp.LE:
                return min_orderable <= filter_value
            case FilterOp.GT:
                return max_orderable > filter_value
            case FilterOp.GE:
                return max_orderable >= filter_value


# Wire encoding for non-JSON-native FilterValue types. Tagged strings keep a
# ``FilterValue`` field valid and the fingerprint deterministic and
# type-distinguishing.
#
# Edge case: a genuine string starting with one of these tags round-trips to
# the richer type. Accepted, since such literals are rare.
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
    # bool is a subclass of int, so check it before the int/float passthrough.
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
    """Compute an order-independent fingerprint for cache keys and provenance.

    Returns:
        16-character hex string, or ``"nofilter"`` for None or empty.
    """
    if not filters:
        return "nofilter"

    # Canonical JSON over explicit fields, so distinct (column, op, value)
    # triples can't collide: concatenation maps both (column='a>', op='=') and
    # (column='a', op='>=') to 'a>='. The value encoding keeps str '1' != int 1.
    items = sorted(
        (
            {"column": f.column, "op": f.op.value, "value": serialize_filter_value(f.value)}
            for f in filters
        ),
        key=lambda d: json.dumps(d, sort_keys=True),
    )
    combined = json.dumps(items, separators=(",", ":"), sort_keys=True)
    return hashlib.md5(combined.encode()).hexdigest()[:16]
