"""Provenance hash and ``# @cache`` policy resolution for SQL cells.

Strata's whole architecture rests on stable cache identity. For SQL
cells the hash needs to capture not just query text and bind values,
but also "what database state did this query see" to the strongest
degree the backend can expose. This module computes that hash and
implements the ``# @cache`` policy that controls how DB-side state
factors in.

Hash composition::

    provenance_hash = H(
        query_normalized,         # sqlglot pretty-print, dialect-aware
        bind_params,              # type-tagged tuple of resolved values
        connection_id,            # canonical non-secret connection identity
        upstream_input_hashes,    # variables referenced in :placeholders
        cache_salt,               # policy-derived static salt
        freshness_token,          # per-driver data-change token (or None)
        schema_fingerprint,       # touched-table column structure (or None)
    )

Note this differs from ``docs/internal/design-sql-cells.md``'s
original sketch by *omitting* a generic ``source_hash``. For Python
cells, ``compute_source_hash`` AST-normalizes the body so cosmetic
edits don't churn the cache. For SQL bodies ``ast.parse`` fails and
that helper falls back to a line-strip — which would re-introduce
exactly the whitespace / comment churn that ``normalize_query``
already strips. ``query_normalized`` *is* the SQL equivalent of an
AST-normalized form; folding a separate weak-normalization hash on
top would defeat the cosmetic-edit goal. The non-body parts of a
SQL cell source (``# @sql connection=...``, ``# @cache ...``, the
``# @name`` override) are captured by ``connection_id`` and
``cache_salt`` already; ``# @name`` doesn't affect data identity
(rename creates a new artifact-name pointer to the same hash).

Policy resolution lives here too because the policy decides which
slots get filled how — ``forever`` skips the freshness/schema probe
entirely, ``snapshot`` requires the token to be ``is_snapshot=True``,
``ttl`` adds a time bucket to the salt instead of probing, etc.
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

    Raised by ``resolve_cache_policy`` for static problems the
    executor can surface before any probe runs — most commonly,
    ``# @cache snapshot`` against a driver that doesn't expose a
    durable snapshot identity (``capabilities.supports_snapshot``
    is False).
    """


@dataclass(frozen=True)
class ResolvedCachePolicy:
    """The decision the resolver makes for a cell's cache identity.

    Attributes:
        kind: One of ``fingerprint`` / ``forever`` / ``session`` /
            ``ttl`` / ``snapshot``. Mirrors ``CachePolicy.kind``
            unless static fallback applied.
        salt: Bytes folded into the provenance hash. ``forever``
            and ``fingerprint`` use a constant; ``session`` /
            ``ttl`` carry per-session / per-bucket variability.
        freshness_required: Whether the executor must call
            ``adapter.probe_freshness`` and fold the result. False
            for ``forever`` / ``session`` / ``ttl`` (the salt alone
            controls invalidation).
        schema_required: Whether ``adapter.probe_schema`` runs.
            Mirrors ``freshness_required`` — a fingerprint without
            the schema fingerprint would miss metadata-only
            ADD COLUMN / type-change invalidations.
        snapshot_required: True only for ``# @cache snapshot``. The
            executor must reject a freshness token whose
            ``is_snapshot`` flag is False — i.e. the adapter
            advertised support but the per-call probe couldn't
            return a durable snapshot ID.
    """

    kind: str
    salt: bytes
    freshness_required: bool
    schema_required: bool
    snapshot_required: bool


# Constant salts used for kinds that don't carry per-call
# variability. Keeping the prefixes distinct keeps a hand-debugged
# hash (and any future offline cache inspector) readable: a salt
# starting with ``strata.cache.forever`` is unambiguously the
# ``forever`` policy, not a session salt that happens to collide.
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

    ``session_id`` is the executor's session identifier. ``now`` is
    optional; left None it reads ``time.time()`` at call time.
    Tests pin it to a specific epoch second so the bucket math is
    deterministic.

    Raises ``CachePolicyError`` for static problems:

    - Unknown ``kind``.
    - ``ttl`` without a positive ``ttl_seconds`` (the parser already
      rejects this in normal flows; this is a safety net for
      directly-constructed policies).
    - ``snapshot`` against a driver whose
      ``AdapterCapabilities.supports_snapshot`` is False.

    Per-call probe-time fallbacks (e.g. fingerprint against a
    backend that returns ``is_session_only=True``) happen in the
    executor, not here — the resolver's view is intentionally
    static.
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

    Uses sqlglot's pretty-printer in the driver's dialect so cosmetic
    edits (whitespace runs, line breaks, lowercase keywords, leading
    or trailing comments) don't churn the cache. Same property
    ``compute_source_hash`` provides for Python cells.

    On parse failure we return ``sql.strip()`` unchanged. The
    analyzer already records the parse error and the validator
    surfaces it as ``sql_parse_error``; the executor will refuse to
    run the cell. The hash never reaches a comparison in that case,
    so the fallback string is just a stable input for unit tests
    that want a deterministic answer regardless of sqlglot's
    behavior.
    """
    if not sql.strip():
        return ""
    try:
        parsed = [s for s in sqlglot.parse(sql, dialect=dialect) if s]
    except _SqlglotError:
        return sql.strip()
    if not parsed:
        return sql.strip()
    # ``comments=False`` drops inline / block comments — they don't
    # affect query semantics and shouldn't churn the cache.
    return ";\n".join(stmt.sql(dialect=dialect, pretty=True, comments=False) for stmt in parsed)


def serialize_bind_params(params: Sequence[Any]) -> list[list[Any]]:
    """Tag each bind value with its concrete type for stable hashing.

    Returns a list of ``[type_tag, encoded_value]`` pairs. The type
    tag is the value's exact Python type name (``"bool"``, ``"int"``,
    ``"datetime"`` etc.); the encoded value is whatever JSON-safe
    representation preserves the value's identity for comparison.

    Why type-tag? ``True`` and ``1`` compare equal in Python and
    serialize to the same JSON, but they're different inputs to a
    SQL parameter binding (``WHERE flag = 1`` matches ``flag =
    TRUE`` only on backends with implicit bool↔int coercion). The
    cache key has to treat them as distinct.

    Encoding choices:

    - ``bytes`` → base64 (JSON can't carry raw bytes).
    - ``Decimal`` → ``str(d)`` (preserves precision; JSON floats
      can't).
    - ``UUID`` → canonical string form.
    - ``datetime`` / ``date`` / ``time`` → ``isoformat()`` so naive
      vs aware datetimes stay distinct (aware carries the tz
      suffix).
    - ``float`` → ``repr(x)`` so denormals and ``NaN`` round-trip
      identically across CPython versions and platforms.
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
    # ``coerce_bind_value`` is the one place that gates types. If a
    # caller fed an unsupported value past it (or without it), we
    # fail loudly here rather than silently producing an unstable
    # hash via ``str()``.
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

    All inputs are folded into a JSON object with sorted keys so the
    output is stable across Python versions and dict-ordering
    differences. The freshness/schema slots are explicit-None when
    the policy doesn't require a probe (``forever`` / ``session`` /
    ``ttl``) — same hash shape, no missing-key ambiguity.

    The freshness token's ``is_session_only`` and ``is_snapshot``
    flags are folded too. They reflect a real semantic difference
    (a session-only token *was* in the hash; the cell's identity
    depends on it not being treated as a per-table fingerprint).

    There is no separate ``source_hash`` parameter — see this
    module's docstring for the rationale. ``query_normalized`` is
    the SQL equivalent of an AST-normalized source.
    """
    payload: dict[str, Any] = {
        "query": query_normalized,
        "binds": serialize_bind_params(bind_params),
        "connection_id": connection_id,
        # ``sort_keys=True`` on the outer dump handles this; the
        # nested dict is included literally and stable-keyed.
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
