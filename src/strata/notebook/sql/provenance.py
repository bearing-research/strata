"""Provenance hash and ``# @cache`` policy resolution for SQL cells.

The hash captures the query, its binds, and, as strongly as the backend allows,
the database state the query saw::

    provenance_hash = H(
        query_normalized,         # sqlglot pretty-print, dialect-aware
        bind_params,              # type-tagged tuple of resolved values
        connection_id,            # canonical non-secret connection identity
        upstream_input_hashes,    # variables referenced in :placeholders
        cache_salt,               # policy-derived static salt
        freshness_token,          # per-driver data-change token (or None)
        schema_fingerprint,       # touched-table column structure (or None)
    )

There is no generic ``source_hash``: on SQL it falls back to a line strip and
would reintroduce the whitespace/comment churn ``normalize_query`` removes.
The annotations are covered by ``connection_id`` and ``cache_salt``; ``# @name``
does not affect data identity.

Policy resolution lives here because the policy decides how slots are filled:
``forever`` skips the probes, ``snapshot`` requires ``is_snapshot=True``,
``ttl`` adds a time bucket to the salt instead of probing.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID

import sqlglot
from sqlglot.errors import SqlglotError as _SqlglotError

from strata.notebook.annotations import CachePolicy
from strata.notebook.sql.adapter import (
    AdapterCapabilities,
    FreshnessToken,
    SchemaFingerprint,
)


class CachePolicyError(ValueError):
    """``# @cache`` annotation can't be honored by the resolved adapter.

    Raised by ``resolve_cache_policy`` before any probe runs, typically for
    ``# @cache snapshot`` on a driver without ``capabilities.supports_snapshot``.
    """


@dataclass(frozen=True)
class ResolvedCachePolicy:
    """The resolver's decision for a cell's cache identity.

    Attributes:
        kind: ``fingerprint`` / ``forever`` / ``session`` / ``ttl`` / ``snapshot``.
        salt: Bytes folded into the hash; constant for ``forever`` and
            ``fingerprint``, per session / per time bucket for ``session`` / ``ttl``.
        freshness_required: Whether ``adapter.probe_freshness`` runs and is folded.
            False for ``forever`` / ``session`` / ``ttl``.
        schema_required: Whether ``adapter.probe_schema`` runs. Same as
            ``freshness_required``: without it, ADD COLUMN or a type change
            would not invalidate.
        snapshot_required: Only for ``# @cache snapshot``: a freshness token with
            ``is_snapshot`` False must be rejected.
    """

    kind: str
    salt: bytes
    freshness_required: bool
    schema_required: bool
    snapshot_required: bool


# Distinct prefixes keep a hand-inspected hash readable: ``strata.cache.forever``
# is unambiguously the ``forever`` policy, never a colliding session salt.
_SALT_FOREVER = b"strata.cache.forever"
_SALT_FINGERPRINT = b"strata.cache.fingerprint"
_SALT_SNAPSHOT = b"strata.cache.snapshot"


def resolve_cache_policy(
    policy: CachePolicy,
    *,
    capabilities: AdapterCapabilities,
    session_id: str,
    now: float | None = None,
) -> ResolvedCachePolicy:
    """Apply ``# @cache`` semantics to produce a ``ResolvedCachePolicy``.

    ``now`` defaults to ``time.time()``; tests pin it. Probe-time fallbacks
    happen in the executor; this view is static.

    Raises ``CachePolicyError`` for an unknown ``kind``, ``ttl`` without a
    positive ``ttl_seconds``, or ``snapshot`` on a driver whose
    ``AdapterCapabilities.supports_snapshot`` is False.
    """
    kind = policy.kind
    if kind == "forever":
        return ResolvedCachePolicy(
            kind="forever",
            salt=_SALT_FOREVER,
            freshness_required=False,
            schema_required=False,
            snapshot_required=False,
        )
    if kind == "session":
        return ResolvedCachePolicy(
            kind="session",
            salt=f"strata.cache.session:{session_id}".encode(),
            freshness_required=False,
            schema_required=False,
            snapshot_required=False,
        )
    if kind == "ttl":
        if not policy.ttl_seconds or policy.ttl_seconds <= 0:
            raise CachePolicyError(
                f"@cache ttl=<seconds> requires a positive integer; got {policy.ttl_seconds!r}"
            )
        clock = time.time() if now is None else now
        bucket = int(clock // policy.ttl_seconds)
        return ResolvedCachePolicy(
            kind="ttl",
            salt=f"strata.cache.ttl:{policy.ttl_seconds}:{bucket}".encode(),
            freshness_required=False,
            schema_required=False,
            snapshot_required=False,
        )
    if kind == "snapshot":
        if not capabilities.supports_snapshot:
            raise CachePolicyError(
                "@cache snapshot requires a driver that exposes a "
                "durable snapshot identity; this driver's adapter "
                "reports supports_snapshot=False"
            )
        return ResolvedCachePolicy(
            kind="snapshot",
            salt=_SALT_SNAPSHOT,
            freshness_required=True,
            schema_required=True,
            snapshot_required=True,
        )
    if kind == "fingerprint":
        return ResolvedCachePolicy(
            kind="fingerprint",
            salt=_SALT_FINGERPRINT,
            freshness_required=True,
            schema_required=True,
            snapshot_required=False,
        )
    raise CachePolicyError(f"unknown cache policy kind: {kind!r}")


def normalize_query(sql: str, dialect: str | None) -> str:
    """Return a canonical, whitespace/comment-insensitive form of ``sql``.

    Uses sqlglot's pretty-printer in the driver's dialect, so cosmetic edits do not
    churn the cache. On parse failure returns ``sql.strip()``; the executor refuses
    to run such a cell, so the hash is never compared.
    """
    if not sql.strip():
        return ""
    try:
        parsed = [s for s in sqlglot.parse(sql, dialect=dialect) if s]
    except _SqlglotError:
        return sql.strip()
    if not parsed:
        return sql.strip()
    # Comments don't affect semantics and shouldn't churn the cache.
    return ";\n".join(stmt.sql(dialect=dialect, pretty=True, comments=False) for stmt in parsed)


def serialize_bind_params(params: Sequence[Any]) -> list[list[Any]]:
    """Tag each bind value with its exact type name for stable hashing.

    Returns ``[type_tag, encoded_value]`` pairs. The tag keeps ``True`` and ``1``
    distinct (they bind differently on backends without bool/int coercion).
    ``bytes`` encode as base64, ``Decimal`` as ``str`` (precision), date/time types
    as ``isoformat()`` (naive and aware stay distinct), ``float`` as ``repr``.
    """
    out: list[list[Any]] = []
    for v in params:
        out.append(_tag_value(v))
    return out


def _tag_value(v: Any) -> list[Any]:
    if v is None:
        return ["none", None]
    t = type(v)
    if t is bool:
        return ["bool", bool(v)]
    if t is int:
        return ["int", int(v)]
    if t is float:
        return ["float", repr(v)]
    if t is str:
        return ["str", v]
    if t is bytes:
        return ["bytes", base64.b64encode(v).decode("ascii")]
    if t is Decimal:
        return ["decimal", str(v)]
    if t is UUID:
        return ["uuid", str(v)]
    if t is _dt.datetime:
        return ["datetime", v.isoformat()]
    if t is _dt.date:
        return ["date", v.isoformat()]
    if t is _dt.time:
        return ["time", v.isoformat()]
    # ``coerce_bind_value`` gates types; fail loudly rather than hash an unstable ``str()``.
    raise ValueError(
        f"cannot serialize bind value of type {t.__name__!r} for "
        "provenance hashing — coerce_bind_value should have rejected it"
    )


def compute_sql_provenance_hash(
    *,
    query_normalized: str,
    bind_params: Sequence[Any],
    connection_id: str,
    upstream_input_hashes: dict[str, str],
    cache_salt: bytes,
    freshness_token: FreshnessToken | None,
    schema_fingerprint: SchemaFingerprint | None,
    lake_fingerprints: Sequence[str] = (),
) -> str:
    """Compute the SHA-256 hash that identifies a SQL cell artifact.

    Inputs are folded into sorted-key JSON. Freshness/schema slots are explicit
    ``None`` when the policy needs no probe. The token's ``is_session_only`` and
    ``is_snapshot`` flags are folded too, as they change what the token means.
    """
    payload: dict[str, Any] = {
        "query": query_normalized,
        "binds": serialize_bind_params(bind_params),
        "connection_id": connection_id,
        "upstream": dict(sorted(upstream_input_hashes.items())),
        "cache_salt": base64.b64encode(cache_salt).decode("ascii"),
        "freshness": (
            None
            if freshness_token is None
            else {
                "value": base64.b64encode(freshness_token.value).decode("ascii"),
                "is_session_only": freshness_token.is_session_only,
                "is_snapshot": freshness_token.is_snapshot,
            }
        ),
        "schema": (
            None
            if schema_fingerprint is None
            else base64.b64encode(schema_fingerprint.value).decode("ascii")
        ),
    }
    # A lake connection's catalog table snapshots and mount fingerprints. Only
    # present when there are some, so every other cell keeps its hash.
    if lake_fingerprints:
        payload["lake"] = sorted(lake_fingerprints)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
