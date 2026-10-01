"""Bind-parameter resolution and coercion for SQL cells.

Strata's SQL cells use ``:name`` placeholders that resolve against
upstream cell variables. This module:

1. Looks up each placeholder name in the upstream namespace.
2. Type-checks the value against an explicit allowlist of accepted
   Python types.
3. Returns an ordered tuple ready for ADBC's parameter-binding API.

Critically, we never do string substitution. Values flow as ADBC bind
parameters through the driver's native parameter API. That is the
entire injection-defense story — adversarial strings (``'; DROP TABLE
users; --``) round-trip as ordinary string parameters because the
backend's prepared-statement layer escapes them.

Accepted types
--------------

- ``None``
- ``bool`` (checked before ``int`` because ``True`` is also an ``int``;
  preserving type identity matters for the provenance hash)
- ``int``
- ``float``
- ``str``
- ``bytes`` (``bytearray`` is coerced to ``bytes`` for hash stability;
  ADBC drivers don't universally accept ``bytearray``)
- ``decimal.Decimal``
- ``uuid.UUID``
- ``datetime.datetime`` / ``datetime.date`` / ``datetime.time``

Anything else (lists, dicts, dataclasses, numpy scalars, pandas
``Timestamp``, custom objects) raises ``BindError``. Strictness here
is deliberate: numpy/pandas types have surprising overflow and
nullability behavior that ADBC drivers handle inconsistently. Users
convert explicitly with ``int(x)`` / ``x.to_pydatetime()`` and the
intent is visible in the cell.

Subclass strictness uses ``type(value) in _ACCEPTED_TYPES`` rather
than ``isinstance``, so a user-subclassed ``MyInt(int)`` and a
``pandas.Timestamp`` (which extends ``datetime``) are rejected with a
clear error rather than silently coerced.
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

    The only coercion is ``bytearray`` → ``bytes`` — ADBC drivers
    don't universally accept ``bytearray``, and the immutable form is
    the right thing for a hash-key downstream. Everything else passes
    through unchanged.

    Raises ``BindError`` if the value's *exact* type isn't on the
    accept list. Subclasses (``numpy.int64``, ``pandas.Timestamp``,
    user-defined ``MyBytes(bytearray)``) are rejected — strict
    ``type() is`` semantics across the board, no isinstance widening,
    so the ``bytearray`` shortcut and the main allowlist behave
    consistently.
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
    """Resolve ordered placeholder names against ``namespace``.

    ``placeholders`` is the sequence the analyzer extracted from the
    SQL body. Each name is looked up in ``namespace`` (the resolved
    upstream-variable map for the cell) and type-checked. The return
    is a positional tuple in the same order as ``placeholders`` —
    suitable for the executor's later step of rewriting ``:name`` to
    the driver's native positional form (``?`` for SQLite, ``$1`` for
    Postgres) and passing the tuple to ADBC.

    Raises ``BindError`` if any name is missing from the namespace or
    has an unsupported type. The first failure short-circuits;
    we don't accumulate diagnostics here because the executor's job
    is to fail fast — the analyzer's diagnostic pass already shows
    the user every missing reference up front.

    Duplicate names are resolved independently. The analyzer dedupes
    its ``references`` list (so the DAG doesn't carry duplicates),
    but if a caller passes duplicates anyway, every position in
    ``placeholders`` produces one entry in the output.
    """
    out: list[Any] = []
    for name in placeholders:
        if name not in namespace:
            raise BindError(f"bind param :{name} not found in upstream variables")
        out.append(coerce_bind_value(name, namespace[name]))
    return tuple(out)
