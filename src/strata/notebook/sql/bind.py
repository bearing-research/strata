"""Bind-parameter resolution and coercion for SQL cells.

``:name`` placeholders resolve against upstream cell variables and flow as ADBC
bind parameters, never by string substitution; that is the whole injection
defense.

Accepted exact types: ``None``, ``bool``, ``int``, ``float``, ``str``, ``bytes``
(``bytearray`` coerced), ``Decimal``, ``UUID``, ``datetime``/``date``/``time``.
Anything else, including subclasses such as ``pandas.Timestamp`` and numpy
scalars, raises ``BindError``: drivers handle those inconsistently, so users
convert explicitly in the cell.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from decimal import Decimal
from typing import Any
from uuid import UUID


class BindError(ValueError):
    """A SQL cell's ``:name`` bind parameter could not be resolved or coerced."""


_ACCEPTED_TYPES: frozenset[type] = frozenset(
    {
        type(None),
        bool,
        int,
        float,
        str,
        bytes,
        Decimal,
        UUID,
        _dt.datetime,
        _dt.date,
        _dt.time,
    }
)

# Stable order for error messages (frozenset iteration order is not).
_ACCEPTED_TYPE_NAMES = "None, bool, int, float, str, bytes, Decimal, UUID, datetime, date, time"


def coerce_bind_value(name: str, value: Any) -> Any:
    """Validate ``value`` for binding to ``:name`` and return the coerced form.

    The only coercion is ``bytearray`` to ``bytes``. Raises ``BindError`` when the
    value's exact type is not accepted; subclasses are rejected too.
    """
    if type(value) is bytearray:
        return bytes(value)
    if type(value) in _ACCEPTED_TYPES:
        return value
    raise BindError(
        f"bind param :{name} has unsupported type "
        f"{type(value).__name__!r}; accepted: {_ACCEPTED_TYPE_NAMES}"
    )


def resolve_bind_params(
    placeholders: Sequence[str],
    namespace: dict[str, Any],
) -> tuple[Any, ...]:
    """Resolve ordered placeholder names against ``namespace`` into a positional tuple.

    Raises ``BindError`` on the first missing name or unsupported type. Duplicate
    names each produce their own entry.
    """
    out: list[Any] = []
    for name in placeholders:
        if name not in namespace:
            raise BindError(f"bind param :{name} not found in upstream variables")
        out.append(coerce_bind_value(name, namespace[name]))
    return tuple(out)
