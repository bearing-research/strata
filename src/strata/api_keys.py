"""API keys: authentication Strata performs itself.

Until now Strata could not identify a caller. ``auth_mode`` offered ``none``
(everyone is anonymous) or ``trusted_proxy`` (a proxy in front asserts identity
in headers, and the network is what stops anyone else asserting it too). That
is a sound architecture for an internal service and a non-starter for a
platform, where the identity *is* the account.

This adds the third mode. A key resolves to the same
:class:`~strata.types.Principal` a proxy header would have produced, so every
downstream consumer -- ACL evaluation, tenant scoping, scope checks -- is
unchanged.

Key format::

    strata_<key_id>_<secret>

The two halves do different jobs. ``key_id`` is a public lookup handle: it is
stored in the clear, indexed, and safe to log or show in a UI. ``secret`` is
never stored -- only its SHA-256 -- so a database disclosure does not yield
usable credentials.

**Why SHA-256 and not bcrypt/argon2.** Those exist to make *low*-entropy
secrets expensive to guess. The secret here is 256 bits from ``secrets``, so
brute force is not the threat model, and a deliberately slow KDF would put tens
of milliseconds on every authenticated request. The comparison is
constant-time; that is the property that matters.

Keys live in the artifact store's database, which means they are shared across
nodes for free wherever that database is.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from strata.sql_backend import SqlDialect, SqliteDialect, StoreConnection

if TYPE_CHECKING:
    from strata.types import Principal

_KEY_PREFIX = "strata"
# 128 bits of lookup handle: not a secret, but it must not collide.
_KEY_ID_BYTES = 16
# 256 bits of secret. See the module docstring on why this is hashed with
# SHA-256 rather than a slow KDF.
_SECRET_BYTES = 32

_API_KEY_SCHEMA_SQL = """
-- API keys. key_hash is a SHA-256 of the secret half; the secret itself is
-- shown once at creation and never stored.
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    tenant TEXT,
    scopes TEXT NOT NULL DEFAULT '[]',  -- JSON array
    description TEXT,
    created_at REAL NOT NULL,
    expires_at REAL,                     -- NULL means no expiry
    revoked_at REAL,                     -- NULL means live
    last_used_at REAL
);

CREATE INDEX IF NOT EXISTS idx_api_keys_principal ON api_keys(principal_id);
"""


@dataclass(frozen=True)
class ApiKeyRecord:
    """A key's metadata. Never carries the secret."""

    key_id: str
    principal_id: str
    tenant: str | None
    scopes: frozenset[str]
    description: str | None
    created_at: float
    expires_at: float | None
    revoked_at: float | None
    last_used_at: float | None

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None and not self.is_expired

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()


def format_key(key_id: str, secret: str) -> str:
    """Render the credential a caller presents."""
    return f"{_KEY_PREFIX}_{key_id}_{secret}"


def parse_key(presented: str) -> tuple[str, str] | None:
    """Split a presented credential into ``(key_id, secret)``.

    Returns ``None`` for anything that is not shaped like one of our keys, so
    a caller cannot tell a malformed key from an unknown one -- both are just
    "unauthenticated".

    The split is bounded at two because the secret is base64url and that
    alphabet includes ``_``: roughly half of all generated secrets contain one.
    An unbounded ``split("_")`` would reject those as malformed, which is an
    intermittent authentication failure on about half of every batch of keys
    issued -- the kind that looks like a flaky client.
    """
    parts = presented.split("_", 2)
    if len(parts) != 3 or parts[0] != _KEY_PREFIX:
        return None
    key_id, secret = parts[1], parts[2]
    if not key_id or not secret:
        return None
    return key_id, secret


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class ApiKeyStore:
    """Storage and verification for API keys.

    Shares the artifact store's dialect: keys belong in the same database, and
    sharing the dialect shares its connection pool, so this adds no connections
    of its own.
    """

    def __init__(self, db_path: Path, dialect: SqlDialect | None = None):
        self.db_path = db_path
        self._dialect: SqlDialect = dialect if dialect is not None else SqliteDialect(db_path)
        self._init_schema()

    def _get_connection(self) -> StoreConnection:
        return self._dialect.connect()

    def _init_schema(self) -> None:
        conn = self._get_connection()
        try:
            if not self._dialect.supports_legacy_migration:
                # Same reasoning as the artifact and build stores: CREATE TABLE
                # IF NOT EXISTS races in Postgres and every node runs this at
                # startup, so creation is serialized and then skipped.
                if not self._dialect.schema_exists(conn, "api_keys"):
                    self._dialect.begin_write(conn, "__api_key_schema__")
                    conn.executescript(self._dialect.adapt_ddl(_API_KEY_SCHEMA_SQL))
                    conn.commit()
                return

            conn.executescript(self._dialect.adapt_ddl(_API_KEY_SCHEMA_SQL))
            conn.commit()
        finally:
            conn.close()

    def create_key(
        self,
        principal_id: str,
        tenant: str | None = None,
        scopes: frozenset[str] | set[str] | None = None,
        description: str | None = None,
        expires_in_seconds: float | None = None,
    ) -> tuple[str, ApiKeyRecord]:
        """Mint a key.

        Returns
        -------
        tuple
            ``(presented_key, record)``. The presented key is the only time
            the secret exists outside the caller's hands -- it is not
            recoverable afterwards, by us or by them.
        """
        key_id = secrets.token_hex(_KEY_ID_BYTES)
        secret = secrets.token_urlsafe(_SECRET_BYTES)
        now = time.time()
        expires_at = now + expires_in_seconds if expires_in_seconds is not None else None
        scope_set = frozenset(scopes or ())

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO api_keys
                    (key_id, key_hash, principal_id, tenant, scopes, description,
                     created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key_id,
                    _hash_secret(secret),
                    principal_id,
                    tenant,
                    json.dumps(sorted(scope_set)),
                    description,
                    now,
                    expires_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        record = ApiKeyRecord(
            key_id=key_id,
            principal_id=principal_id,
            tenant=tenant,
            scopes=scope_set,
            description=description,
            created_at=now,
            expires_at=expires_at,
            revoked_at=None,
            last_used_at=None,
        )
        return format_key(key_id, secret), record

    def verify(self, presented: str) -> Principal | None:
        """Resolve a presented credential to a principal, or ``None``.

        Every rejection returns ``None`` rather than distinguishing malformed
        from unknown from revoked from expired: the caller learns only that
        they are not authenticated, which is all they are owed and all that is
        safe to tell them.
        """
        from strata.types import Principal

        parsed = parse_key(presented)
        if parsed is None:
            return None
        key_id, secret = parsed

        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT key_hash, principal_id, tenant, scopes, expires_at, revoked_at "
                "FROM api_keys WHERE key_id = ?",
                (key_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        # Constant-time: a timing-variable compare would leak the stored hash
        # a byte at a time to anyone who can present guesses.
        if not hmac.compare_digest(row["key_hash"], _hash_secret(secret)):
            return None

        if row["revoked_at"] is not None:
            return None
        if row["expires_at"] is not None and row["expires_at"] <= time.time():
            return None

        return Principal(
            id=row["principal_id"],
            tenant=row["tenant"],
            scopes=frozenset(json.loads(row["scopes"])),
        )

    def touch(self, key_id: str) -> None:
        """Record that a key was used.

        Separate from :meth:`verify` and best-effort by design: writing on
        every authenticated request would put a write in the hot path of every
        read. Callers update on a sampled or cached schedule.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?", (time.time(), key_id)
            )
            conn.commit()
        finally:
            conn.close()

    def revoke(self, key_id: str) -> bool:
        """Revoke a key. Returns whether one was live to revoke."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
                (time.time(), key_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_keys(self, principal_id: str | None = None) -> list[ApiKeyRecord]:
        """List key metadata, newest first. Never returns secrets."""
        conn = self._get_connection()
        try:
            if principal_id is None:
                cursor = conn.execute(
                    "SELECT key_id, principal_id, tenant, scopes, description, created_at, "
                    "expires_at, revoked_at, last_used_at FROM api_keys ORDER BY created_at DESC"
                )
            else:
                cursor = conn.execute(
                    "SELECT key_id, principal_id, tenant, scopes, description, created_at, "
                    "expires_at, revoked_at, last_used_at FROM api_keys "
                    "WHERE principal_id = ? ORDER BY created_at DESC",
                    (principal_id,),
                )
            return [
                ApiKeyRecord(
                    key_id=row["key_id"],
                    principal_id=row["principal_id"],
                    tenant=row["tenant"],
                    scopes=frozenset(json.loads(row["scopes"])),
                    description=row["description"],
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                    revoked_at=row["revoked_at"],
                    last_used_at=row["last_used_at"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()


_api_key_store: ApiKeyStore | None = None


def get_api_key_store(
    db_path: Path | None = None,
    dialect: SqlDialect | None = None,
) -> ApiKeyStore | None:
    """Get the API key store singleton."""
    global _api_key_store
    if _api_key_store is None and db_path is not None:
        _api_key_store = ApiKeyStore(db_path, dialect=dialect)
    return _api_key_store


def reset_api_key_store() -> None:
    """Reset the singleton (for testing)."""
    global _api_key_store
    _api_key_store = None
