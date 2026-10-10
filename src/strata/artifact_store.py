"""Artifact store: immutable versioned artifacts plus mutable name, alias and tag pointers.

Metadata lives in SQLite (or Postgres via a dialect); bytes live in a pluggable
``BlobStore`` as ``{id}@v={version}.arrow`` Arrow IPC streams. Artifacts
deduplicate by provenance hash: at most one ready row per
``(tenant, provenance_hash)``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strata.sql_backend import SqlDialect, SqliteDialect, StoreConnection

if TYPE_CHECKING:
    from strata.blob_store import BlobStore

logger = logging.getLogger(__name__)


# --- Data types ---


class ArtifactNotFoundError(ValueError):
    """The artifact version does not exist, or belongs to a tenant the caller cannot write in.

    One error for both, so a write cannot tell another tenant's artifact from a missing one.
    """


class ArtifactImportConflict(ValueError):
    """An artifact arriving from elsewhere claims an id another one holds.

    Ids are not globally unique: notebook output ids derive from notebook and cell
    ids, so two people editing one repository can produce the same id for
    different bytes. Keeping the existing row silently would point the newcomer's
    name, tags and descendants at someone else's bytes.
    """


def reject_unsafe_artifact_id(artifact_id: str) -> None:
    """Refuse an id from another store that would name a path.

    An id becomes a blob key, so a separator or ``..`` segment would write outside
    the blob store. Ids this store generates never contain either; imported ids
    (snapshot bundles, ``POST /v1/artifacts/import``) must be checked.
    """
    if not artifact_id:
        raise ValueError("an artifact id is required")
    if "/" in artifact_id or "\\" in artifact_id or ".." in Path(artifact_id).parts:
        raise ValueError(f"an artifact id cannot name a path: {artifact_id!r}")


_ATTEMPT_PATTERN = re.compile(r"[0-9a-f]{16,64}")


def attempt_blob_id(artifact_id: str, attempt: str) -> str:
    """The blob id one build attempt writes a version's bytes under.

    The attempt id comes from a signed upload URL and becomes part of a blob key,
    so it is held to lowercase hex.
    """
    if not _ATTEMPT_PATTERN.fullmatch(attempt):
        raise ValueError(f"not a build attempt id: {attempt!r}")
    return f"{artifact_id}~{attempt}"


# How long bytes uploaded ahead of an import wait for it. An import follows its
# upload within seconds; a day covers a caller retrying through an outage.
IMPORT_STAGING_TTL_SECONDS = 86_400.0


def staged_import_key(tenant: str | None, content_sha256: str) -> tuple[str, int]:
    """The blob key bytes uploaded ahead of an import wait under.

    Uses version 0, which no artifact can have, so a staged upload never collides
    with an artifact's key.
    """
    tenant_key = hashlib.sha256((tenant or "").encode()).hexdigest()[:16]
    return f"import-staging-{tenant_key}-{content_sha256}", 0


class BuildLeaseLost(RuntimeError):
    """A build attempt tried to publish after its lease moved to another."""


# Completes a build inside finalize's transaction, given the artifact the
# build produced; False when the caller no longer holds the build's lease.
BuildFence = Callable[[StoreConnection, str, int], bool]


def _ancestor_of(input_uri: str, recorded: object) -> tuple[str, int] | None:
    """The ``(id, version)`` an input edge names, or None if it is not one.

    Edges are ``{input_uri: recorded_version}``: the uri may be an artifact or a
    name, the value is the version that answered. Must resolve the same way as the
    lineage walk in ``services.artifact``, or retention and lineage disagree.
    """
    if not isinstance(recorded, str):
        return None
    if not (input_uri.startswith("strata://artifact/") or input_uri.startswith("strata://name/")):
        return None
    artifact_id, separator, version = recorded.partition("@v=")
    if not separator or not version.isdigit():
        return None
    return artifact_id, int(version)


def _split_ref(ref: str) -> tuple[str, int]:
    """``id@v=N`` as the store writes it, split."""
    artifact_id, _, version = ref.partition("@v=")
    return artifact_id, int(version)


def _declared_content_type(transform_spec: str | None) -> str:
    """The ``content_type`` a version's transform params declare, or ``""`` for none."""
    try:
        params = json.loads(transform_spec or "{}").get("params") or {}
    except (ValueError, AttributeError):
        return ""
    content_type = params.get("content_type") if isinstance(params, dict) else None
    return content_type if isinstance(content_type, str) else ""


def _like_literal(text: str) -> str:
    """``text`` escaped for ``LIKE ... ESCAPE '\\'``: its ``_`` and ``%`` match only themselves."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass(frozen=True)
class ArtifactVersion:
    """Immutable artifact version metadata.

    ``state`` is ``building``, ``ready``, ``superseded`` or ``failed``.
    ``transform_spec`` is opaque to the server. ``input_versions`` is a JSON map of
    input URI to version (a snapshot id for tables, ``id@v=N`` for artifacts), used
    for staleness. ``content_sha256`` lets two runs be compared output by output
    without rereading blobs.
    """

    id: str
    version: int
    state: str  # "building" | "ready" | "failed"
    provenance_hash: str
    schema_json: str | None = None
    row_count: int | None = None
    byte_size: int | None = None
    created_at: float | None = None
    transform_spec: str | None = None
    input_versions: str | None = None  # JSON: {"uri": "version_string", ...}
    tenant: str | None = None  # Tenant ID for multi-tenant isolation
    principal: str | None = None  # Principal ID that created this artifact
    # SHA-256 of the stored bytes, recorded at finalize. None on rows written
    # before the column existed, until something asks (``content_digest``).
    content_sha256: str | None = None


@dataclass(frozen=True)
class StagedVersion:
    """A ``building`` version whose bytes are written, waiting to be made ready."""

    artifact_id: str
    version: int
    schema_json: str
    row_count: int | None  # None: the writer does not know it
    byte_size: int
    content_sha256: str


@dataclass(frozen=True)
class ImportedArtifact:
    """Where an imported record landed in the destination store.

    ``id`` and ``version`` may differ from the caller's when the store already
    holds the computation under another id; a caller copying a chain must point
    descendants at these. ``written`` is False for either no-op.
    """

    id: str
    version: int
    written: bool

    @property
    def ref(self) -> str:
        """The ``id@v=N`` form lineage edges are recorded in."""
        return f"{self.id}@v={self.version}"


@dataclass(frozen=True)
class Publication:
    """An opt-in public read grant for one artifact version.

    ``token`` is an unguessable secret and the only credential for the
    unauthenticated reader. The store keeps only its SHA-256, ``id``, so
    ``token`` is set only when the grant is minted or looked up by it, and
    is empty otherwise. The version binding is permanent. ``content_sha256``
    is the digest of the bytes as published, the basis of the page's integrity
    claim. ``revoked_at`` is set on withdrawal; the row survives so the token is
    never reissued. ``authors`` (ordered ``{"name", "orcid", "affiliation"}``) is
    the byline when set, else ``published_by``. ``external_ids`` holds
    ``{"scheme", "value"}`` entries such as a DOI.
    """

    token: str
    artifact_id: str
    version: int
    tenant: str = ""
    title: str | None = None
    published_at: float = 0.0
    published_by: str | None = None
    content_sha256: str | None = None
    revoked_at: float | None = None
    authors: tuple[dict[str, str], ...] = ()
    external_ids: tuple[dict[str, str], ...] = ()
    id: str = ""

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


@dataclass(frozen=True)
class ArtifactName:
    """Mutable name pointer to an artifact version.

    Unique per ``(tenant, name)``, so tenants can reuse names.
    """

    name: str
    artifact_id: str
    version: int
    updated_at: float
    tenant: str | None = None  # Tenant ID for multi-tenant isolation


@dataclass(frozen=True)
class ArtifactAlias:
    """An intent pointer (champion, candidate, ...) on a registry name.

    A name can hold many aliases; every move is recorded in the append-only
    registry audit.
    """

    name: str
    alias: str
    artifact_id: str
    version: int
    updated_at: float
    tenant: str | None = None


@dataclass(frozen=True)
class InputChange:
    """A change in one input dependency: the version used at build vs. the current one."""

    input_uri: str
    old_version: str
    new_version: str

    def __str__(self) -> str:
        return f"{self.input_uri}: {self.old_version} → {self.new_version}"


@dataclass
class NameStatus:
    """Status for a named artifact, including staleness and which inputs changed."""

    name: str
    artifact_uri: str
    artifact_id: str
    version: int
    state: str
    updated_at: float
    input_versions: dict[str, str]
    is_stale: bool = False
    stale_reason: str | None = None
    changed_inputs: list[InputChange] | None = None


@dataclass(frozen=True)
class TransformSpec:
    """Transform specification: stored by the server, interpreted only by the executor.

    ``executor`` is a URI such as ``local://duckdb_sql@v1``; ``inputs`` are table
    or artifact URIs.
    """

    executor: str
    params: dict
    inputs: list[str]

    def to_json(self) -> str:
        """Serialize to JSON string.

        Inputs keep their caller order: positional executors bind them as ``input0,
        input1, ...``, so ``f(a, b)`` must not dedup against ``f(b, a)``.
        """
        return json.dumps(
            {
                "executor": self.executor,
                "params": self.params,
                "inputs": list(self.inputs),
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, json_str: str) -> TransformSpec:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(
            executor=data["executor"],
            params=data["params"],
            inputs=data["inputs"],
        )


# --- Provenance hash ---


def compute_provenance_hash(input_hashes: list[str], transform_spec: TransformSpec) -> str:
    """SHA-256 of the sorted input hashes plus the transform spec JSON, for dedup.

    Input hashes are sorted, but the spec's own ``inputs`` keep caller order.
    """
    sorted_inputs = sorted(input_hashes)

    hasher = hashlib.sha256()
    for h in sorted_inputs:
        hasher.update(h.encode("utf-8"))
        hasher.update(b"\x00")  # Separator
    hasher.update(transform_spec.to_json().encode("utf-8"))

    return hasher.hexdigest()


# --- Artifact store ---

# Schema evolution. The constants above always describe the *latest* shape; the migrations describe
# the path to it from the baseline. A fresh database is created from the constants and stamped
# latest, so it never runs a migration; an existing one is stamped at the baseline and walks
# forward. ``test_a_migrated_database_matches_a_fresh_one`` keeps the two from drifting.
#
# The ad-hoc migrations further down are SQLite-only (PRAGMA, sqlite_master) and only bring legacy
# SQLite databases up to the baseline.

_SCHEMA_VERSION_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
);
"""

# The shape every database had before this mechanism existed.
_BASELINE_SCHEMA_VERSION = 0


@dataclass(frozen=True)
class _Migration:
    """One forward schema step, applied at most once per database; there is no down-migration."""

    version: int
    description: str
    apply: Callable[[StoreConnection, SqlDialect], None]


def _add_content_sha256(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Add the nullable ``content_sha256`` column.

    Nullable so the migration need not read every blob: ``finalize_artifact``
    fills it for new rows and ``content_digest`` fills older ones on demand.
    """
    if not dialect.column_exists(conn, "artifact_versions", "content_sha256"):
        conn.execute("ALTER TABLE artifact_versions ADD COLUMN content_sha256 TEXT")


def _json_entries(raw: str | None) -> tuple[dict[str, str], ...]:
    """A stored JSON list back into entries, or empty for anything else.

    Rows predating the columns hold NULL; malformed values are read defensively
    here once rather than at every consumer.
    """
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except ValueError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(
        {str(k): str(v) for k, v in entry.items()} for entry in parsed if isinstance(entry, dict)
    )


def _clean_entries(
    entries: list[dict[str, str]] | None, fields: tuple[str, ...]
) -> tuple[dict[str, str], ...]:
    """Keep the known fields of each entry, in order, dropping empty values.

    So a missing ORCID never renders as ``"None"``.
    """
    cleaned: list[dict[str, str]] = []
    for entry in entries or []:
        kept = {field: str(entry[field]).strip() for field in fields if entry.get(field)}
        if kept:
            cleaned.append(kept)
    return tuple(cleaned)


def _json_or_none(entries: tuple[dict[str, str], ...]) -> str | None:
    """Store NULL rather than ``[]``, so an empty list and never-set read the same."""
    return json.dumps([dict(entry) for entry in entries]) if entries else None


def _add_publication_credits(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Add nullable ``authors`` and ``external_ids`` to publications.

    Not backfilled: ``published_by`` stays the byline for older publications.
    """
    for column in ("authors", "external_ids"):
        if not dialect.column_exists(conn, "artifact_publications", column):
            conn.execute(f"ALTER TABLE artifact_publications ADD COLUMN {column} TEXT")


_PUBLICATION_ID = re.compile(r"^[0-9a-f]{64}$")


def publication_id(token: str) -> str:
    """What the store keeps for a publication token: its SHA-256 hex digest.

    A copy of the database, a backup or a read of the file then holds no working link.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def publication_key(token_or_id: str) -> str:
    """The stored key for a raw token, or for a publication id given as is.

    A raw token (43 base64url characters) is never 64 hex digits, so the two cannot
    be confused. Only routes behind authentication may take an id: on a public route
    the id would work as the link.
    """
    return token_or_id if _PUBLICATION_ID.match(token_or_id) else publication_id(token_or_id)


def _hash_publication_tokens(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Replace every stored publication token, and its audit entries, with its SHA-256.

    Links already handed out keep working: a lookup hashes the token it is given.
    """
    del dialect
    rows = conn.execute("SELECT token FROM artifact_publications").fetchall()
    for token in [row["token"] for row in rows]:
        if _PUBLICATION_ID.match(token):
            continue
        hashed = publication_id(token)
        conn.execute("UPDATE artifact_publications SET token = ? WHERE token = ?", (hashed, token))
    # One pass by seq: (key, value) has no index, so a lookup per publication scans the audit
    # table each time, under the schema lock at startup.
    audit = conn.execute("SELECT seq, value FROM registry_audit WHERE key = 'token'").fetchall()
    for row in audit:
        if row["value"] and not _PUBLICATION_ID.match(row["value"]):
            conn.execute(
                "UPDATE registry_audit SET value = ? WHERE seq = ?",
                (publication_id(row["value"]), row["seq"]),
            )


def _add_pins(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Add pins, so a platform can hold a chain the store has no other reason to keep."""
    conn.execute(
        dialect.adapt_ddl(
            """
            CREATE TABLE IF NOT EXISTS artifact_pins (
                artifact_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                reason TEXT NOT NULL,
                tenant TEXT NOT NULL DEFAULT '',
                pinned_by TEXT,
                pinned_at REAL NOT NULL,
                PRIMARY KEY (artifact_id, version, reason)
            )
            """
        )
    )
    conn.execute(
        dialect.adapt_ddl("CREATE INDEX IF NOT EXISTS idx_pins_tenant ON artifact_pins(tenant)")
    )


def _add_blob_attempt(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Add the nullable ``blob_attempt`` column: which build attempt's bytes a version reads.

    NULL means the shared ``(id, version)`` key; only transform builds write
    per-attempt blobs.
    """
    if not dialect.column_exists(conn, "artifact_versions", "blob_attempt"):
        conn.execute("ALTER TABLE artifact_versions ADD COLUMN blob_attempt TEXT")


def _add_import_staging(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Give an import somewhere to find bytes uploaded ahead of its record."""
    conn.execute(
        dialect.adapt_ddl(
            """
            CREATE TABLE IF NOT EXISTS import_staging (
                tenant TEXT NOT NULL DEFAULT '',
                content_sha256 TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                staged_at REAL NOT NULL,
                PRIMARY KEY (tenant, content_sha256)
            )
            """
        )
    )


# How stale ``last_used_at`` may get before a use writes it again: recording
# every read would turn reads into writes, and a hot artifact costs one UPDATE
# per interval this way. A sweep adds it to its recent-use floor, because the
# recorded time can be this far behind the last real use.
_USE_RESOLUTION_SECONDS = 300.0

# Versions garbage_collect deletes per transaction, and how long it leaves SQLite's write
# lock free between them.
_GC_DELETE_BATCH = 1000
_GC_BATCH_PAUSE_SECONDS = 0.1

# How long a temp file goes untouched before a sweep takes it for a dead write's.
_ABANDONED_WRITE_SECONDS = 3600.0

# A sweep over its byte cap collects down to this fraction of it, so the next
# write does not put it straight back over (the row-group cache does the same).
_EVICT_TO_FRACTION = 0.8

# A canonical uuid4, the shape of every id the store mints (``uuid.uuid4()``).
_MINTED_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _add_use_and_minted(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Add ``last_used_at`` and ``minted``, which retention keys on.

    ``last_used_at`` NULL reads as ``created_at``. Existing rows start their idle
    clock at the upgrade, since earlier releases recorded no use and a result used
    yesterday would otherwise look as old as its creation. ``minted`` marks an id the
    store made up for one computation, whose latest version may be collected.
    Existing rows are backfilled by shape: uuid4 ids are minted, ``nb_...`` ids are
    not, so a caller-chosen uuid-shaped id may be recomputed after a long idle.
    """
    if not dialect.column_exists(conn, "artifact_versions", "last_used_at"):
        conn.execute(f"ALTER TABLE artifact_versions ADD COLUMN last_used_at {dialect.float_type}")
        conn.execute("UPDATE artifact_versions SET last_used_at = ?", (time.time(),))
    if not dialect.column_exists(conn, "artifact_versions", "minted"):
        conn.execute(
            "ALTER TABLE artifact_versions "
            f"ADD COLUMN minted {dialect.integer_type} NOT NULL DEFAULT 0"
        )
        ids = [row["id"] for row in conn.execute("SELECT DISTINCT id FROM artifact_versions")]
        for artifact_id in ids:
            if _MINTED_ID.match(artifact_id):
                conn.execute("UPDATE artifact_versions SET minted = 1 WHERE id = ?", (artifact_id,))


def _add_superseded_by(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Add the nullable ``superseded_by`` column: the canonical version whose bytes a
    version overtaken by a duplicate build reads, having dropped its own blob."""
    if not dialect.column_exists(conn, "artifact_versions", "superseded_by"):
        conn.execute("ALTER TABLE artifact_versions ADD COLUMN superseded_by TEXT")


def _add_notebook_workers(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Move the server-managed notebook worker registry into the store, from a per-node file."""
    conn.executescript(dialect.adapt_ddl(_NOTEBOOK_WORKERS_SCHEMA_SQL))


_MIGRATIONS: list[_Migration] = [
    _Migration(1, "artifact_versions.content_sha256", _add_content_sha256),
    _Migration(2, "artifact_publications.authors + external_ids", _add_publication_credits),
    _Migration(3, "artifact_pins", _add_pins),
    _Migration(4, "artifact_versions.blob_attempt", _add_blob_attempt),
    _Migration(5, "import_staging", _add_import_staging),
    _Migration(6, "artifact_versions.last_used_at + minted", _add_use_and_minted),
    _Migration(7, "artifact_versions.superseded_by", _add_superseded_by),
    _Migration(8, "artifact_publications.token hashed", _hash_publication_tokens),
    _Migration(9, "notebook_workers", _add_notebook_workers),
]

_LATEST_SCHEMA_VERSION = max((m.version for m in _MIGRATIONS), default=_BASELINE_SCHEMA_VERSION)


class StoreSchemaMismatch(RuntimeError):
    """The store's schema is not one this release can open as asked."""


def _newer_schema_error(current: int) -> StoreSchemaMismatch:
    return StoreSchemaMismatch(
        f"The artifact store's schema is version {current}, newer than this Strata "
        f"supports ({_LATEST_SCHEMA_VERSION}). Upgrade Strata, or restore a backup "
        "taken before the upgrade."
    )


_SCHEMA_SQL = """
-- Artifact versions: immutable once state="ready"
CREATE TABLE IF NOT EXISTS artifact_versions (
    id TEXT NOT NULL,
    version INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'building',
    provenance_hash TEXT NOT NULL,
    schema_json TEXT,
    row_count INTEGER,
    byte_size INTEGER,
    created_at REAL NOT NULL,
    transform_spec TEXT,
    input_versions TEXT,  -- JSON: {"uri": "version_string", ...} for staleness detection
    tenant TEXT NOT NULL DEFAULT '',  -- Tenant ID ('' = tenantless, matches names/aliases/tags)
    principal TEXT,  -- Principal ID that created this artifact
    content_sha256 TEXT,  -- Digest of the stored bytes (see migration 1)
    blob_attempt TEXT,  -- Build attempt whose bytes this reads; NULL = shared key (migration 4)
    last_used_at REAL,  -- Last finalize, hit or read; NULL reads as created_at (migration 6)
    minted INTEGER NOT NULL DEFAULT 0,  -- 1 = the store made up the id (migration 6)
    superseded_by TEXT,  -- id@v=N whose blob this reads, having none of its own (migration 7)
    PRIMARY KEY (id, version)
);

-- Index for provenance lookup (deduplication)
CREATE INDEX IF NOT EXISTS idx_provenance ON artifact_versions(provenance_hash);

-- Unique constraint for idempotent finalize: (tenant, provenance_hash) for ready artifacts.
-- Prevents duplicate ready artifacts for the same computation within a tenant. Relies on
-- tenant being '' (never NULL) for tenantless rows: SQLite treats NULLs as distinct, which
-- would let duplicate tenantless ready rows slip through (see _init_schema normalization).
CREATE UNIQUE INDEX IF NOT EXISTS idx_tenant_provenance_unique
ON artifact_versions(tenant, provenance_hash)
WHERE state = 'ready';

-- Index for state queries (e.g., cleanup of failed artifacts)
CREATE INDEX IF NOT EXISTS idx_state ON artifact_versions(state);

-- Index for tenant queries (multi-tenant isolation)
CREATE INDEX IF NOT EXISTS idx_versions_tenant ON artifact_versions(tenant);

-- Name pointers: mutable, point to artifact versions
-- In multi-tenant mode, (tenant, name) is the unique key
-- Note: tenant uses '' (empty string) instead of NULL for personal mode
-- because SQLite's PRIMARY KEY doesn't treat NULLs as equal
CREATE TABLE IF NOT EXISTS artifact_names (
    name TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    tenant TEXT NOT NULL DEFAULT '',  -- Tenant ID ('' for personal mode)
    PRIMARY KEY (tenant, name),
    FOREIGN KEY (artifact_id, version) REFERENCES artifact_versions(id, version)
);

-- Index for name lookup without tenant (personal mode)
CREATE INDEX IF NOT EXISTS idx_names_name ON artifact_names(name);
"""

# Registry tables (aliases / tags / audit). Run unconditionally at init; CREATE IF NOT EXISTS makes
# it idempotent for fresh and existing databases.
_REGISTRY_SCHEMA_SQL = """
-- Aliases: mutable intent pointers (champion, candidate, ...) on a name.
-- A name can hold many aliases; each points at one artifact version.
CREATE TABLE IF NOT EXISTS artifact_aliases (
    name TEXT NOT NULL,
    alias TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    tenant TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (tenant, name, alias)
);
CREATE INDEX IF NOT EXISTS idx_aliases_name ON artifact_aliases(name);

-- Version tags: facts about a specific artifact version (auc=0.91, ...).
CREATE TABLE IF NOT EXISTS artifact_tags (
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL,
    tenant TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (tenant, artifact_id, version, key)
);
CREATE INDEX IF NOT EXISTS idx_tags_artifact ON artifact_tags(artifact_id, version);

-- Append-only audit of every name/alias/tag mutation. Written in the
-- same transaction as the mutation; never updated or deleted.
CREATE TABLE IF NOT EXISTS registry_audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    actor TEXT,
    action TEXT NOT NULL,
    name TEXT,
    alias TEXT,
    artifact_id TEXT,
    from_artifact_id TEXT,
    from_version INTEGER,
    to_version INTEGER,
    key TEXT,
    value TEXT,
    tenant TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_name ON registry_audit(name);

-- Pending alias changes awaiting approval (one per alias). Created when a
-- protected alias is moved/deleted; an approve applies it, a reject
-- discards it. Both outcomes are audited.
CREATE TABLE IF NOT EXISTS registry_pending (
    name TEXT NOT NULL,
    alias TEXT NOT NULL,
    action TEXT NOT NULL,            -- 'set' or 'delete'
    artifact_id TEXT,                -- target for 'set'
    version INTEGER,
    requested_by TEXT,
    requested_at REAL NOT NULL,
    tenant TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (tenant, name, alias)
);
"""

# Publications: opt-in, per-artifact-version public read grants.
#
# A token is bound to one (artifact_id, version) for good. Revoking sets ``revoked_at`` and keeps
# the row, so a token can never be reused for different content: a URL printed in a paper must fail
# closed, never resolve to something else.
_PUBLICATION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS artifact_publications (
    token TEXT PRIMARY KEY,  -- SHA-256 hex of the link's token (migration 8), never the token
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    tenant TEXT NOT NULL DEFAULT '',
    title TEXT,
    published_at REAL NOT NULL,
    published_by TEXT,
    content_sha256 TEXT,
    revoked_at REAL,
    authors TEXT,        -- JSON list of {name, orcid, affiliation} (migration 2)
    external_ids TEXT    -- JSON list of {scheme, value} (migration 2)
);
CREATE INDEX IF NOT EXISTS idx_publications_artifact
ON artifact_publications(artifact_id, version);
CREATE INDEX IF NOT EXISTS idx_publications_tenant ON artifact_publications(tenant);

-- A hold on an artifact version and everything behind it, for a reason the
-- store has no other way to know: a snapshot a platform must be able to
-- restore, a review still open. A root for garbage collection, as a
-- publication is. One row per reason, so two holders release independently.
CREATE TABLE IF NOT EXISTS artifact_pins (
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    reason TEXT NOT NULL,
    tenant TEXT NOT NULL DEFAULT '',
    pinned_by TEXT,
    pinned_at REAL NOT NULL,
    PRIMARY KEY (artifact_id, version, reason)
);
CREATE INDEX IF NOT EXISTS idx_pins_tenant ON artifact_pins(tenant);

-- Bytes uploaded ahead of the import that names them (PUT
-- /v1/artifacts/import/blobs/{sha256}). Per tenant: a digest is no proof of
-- holding the bytes, since a publication's page prints it, so one tenant's
-- upload must never satisfy another's import. The blob stores cannot list
-- keys, so this is also how an upload that is never imported is found again.
CREATE TABLE IF NOT EXISTS import_staging (
    tenant TEXT NOT NULL DEFAULT '',
    content_sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    staged_at REAL NOT NULL,
    PRIMARY KEY (tenant, content_sha256)
);
"""

# The server-managed notebook worker registry, shared by every node on this database: one row per
# worker, in catalogue order. The single registry row records that it has been set, after which the
# configured ``notebook_workers`` table no longer applies, so an emptied registry stays empty.
_NOTEBOOK_WORKERS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notebook_workers (
    name TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    spec TEXT NOT NULL,  -- JSON: the WorkerSpec plus "enabled"
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS notebook_worker_registry (
    id INTEGER PRIMARY KEY,  -- always 1
    updated_at REAL NOT NULL
);
"""

# Legacy migration: tenant columns for existing tables.
_MIGRATION_SQL = """
-- Add tenant and principal columns to artifact_versions if they don't exist
-- SQLite doesn't have ADD COLUMN IF NOT EXISTS, so we use a workaround

-- Check if tenant column exists by trying to select it
-- If it fails, the column doesn't exist and we need to add it
"""


# Bound on finalize_canonical_together's retry. A conflict means a competing
# writer committed a ready row with our provenance; one retry supersedes it
# and wins. More than two rounds means sustained contention on a single
# provenance, where failing loudly beats spinning.
_CANONICAL_PROMOTE_ATTEMPTS = 3


# Sentinel for read_audit's tenant param: "no tenant filter at all" (the CLI / admin view of the
# whole store), distinct from an explicit None, which normalizes to the '' default tenant.
_AUDIT_ALL_TENANTS = object()


class ArtifactStore:
    """Artifact metadata (SQLite or Postgres) over a pluggable blob store.

    Thread-safe: one connection per operation (WAL on SQLite). Lifecycle:
    ``create_artifact`` (building), write the blob, ``finalize_artifact`` (ready,
    or the existing row on a provenance duplicate).
    """

    def __init__(
        self,
        artifact_dir: Path,
        blob_store: BlobStore | None = None,
        dialect: SqlDialect | None = None,
        *,
        read_only: bool = False,
    ):
        """Open or create the store under ``artifact_dir``.

        ``blob_store`` defaults to a ``LocalBlobStore`` in ``{artifact_dir}/blobs``;
        ``dialect`` defaults to SQLite under ``artifact_dir``. With a shared
        ``PostgresDialect`` the blob store should be remote too, or the store is only
        coherent on one machine.

        ``read_only`` is for commands that only inspect a store: it never creates or
        migrates the schema, and refuses a store that is not at this release's version.

        Raises:
            StoreSchemaMismatch: the schema is newer than this release, or, with
                ``read_only``, older or missing.
        """
        from strata.file_modes import narrow, private_dir, private_file

        self.artifact_dir = artifact_dir
        self.db_path = artifact_dir / "artifacts.sqlite"
        # Every tenant's metadata, names and blobs: no other account on the host reads them.
        private_dir(artifact_dir)

        if blob_store is None:
            from strata.blob_store import LocalBlobStore

            self.blobs_dir = artifact_dir / "blobs"
            self.blob_store: BlobStore = LocalBlobStore(self.blobs_dir)
        else:
            self.blob_store = blob_store
            # Back-compat attribute, set for the local store only.
            from strata.blob_store import LocalBlobStore

            if isinstance(blob_store, LocalBlobStore):
                self.blobs_dir = blob_store.blobs_dir
            else:
                self.blobs_dir = artifact_dir / "blobs"  # May not exist for remote stores

        if dialect is None:
            private_file(self.db_path)
            for journal in ("-wal", "-shm"):
                sibling = self.db_path.with_name(self.db_path.name + journal)
                try:
                    narrow(sibling, 0o600)
                except FileNotFoundError:
                    # Absent, or deleted under us as the last open connection elsewhere closed.
                    continue

        # Every query goes through _get_connection, so the dialect is the one place a second backend
        # has to be taught about; see strata/sql_backend.py for what differs.
        self._dialect: SqlDialect = dialect if dialect is not None else SqliteDialect(self.db_path)

        if read_only:
            self._require_current_schema()
        else:
            self._init_schema()

    def _get_connection(self) -> StoreConnection:
        """Open a connection configured by the active dialect (see ``SqliteDialect.connect``)."""
        return self._dialect.connect()

    @property
    def dialect(self) -> SqlDialect:
        """The metadata backend this store is using.

        Public so the build store can share the database and its connection pool.
        """
        return self._dialect

    def close(self) -> None:
        """Release backend resources. Idempotent, and safe to skip on SQLite.

        Needed for Postgres, whose pool runs worker threads that outlive the store.
        """
        self._dialect.close()

    def ping(self) -> None:
        """Run a trivial query; raises when the metadata database is unreachable."""
        conn = self._get_connection()
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create the schema, or migrate an existing database to the latest version."""
        conn = self._get_connection()
        try:
            # The migration below upgrades databases from older Strata versions in SQLite's own
            # vocabulary (sqlite_master, PRAGMA, rowid). A newer backend has no such history and
            # goes straight to the current schema.
            if not self._dialect.supports_legacy_migration:
                # Fast path: a store is constructed per session in several places and the lock below
                # is global, so take it only while the schema is missing.
                if self._dialect.schema_exists(conn):
                    # Existing database: it may still be behind. Under the same
                    # global lock the creation path takes, since two replicas
                    # starting together would otherwise both apply migration N.
                    self._dialect.begin_write(conn, "__schema__")
                    self._apply_schema_migrations(conn)
                    return

                # CREATE TABLE IF NOT EXISTS is not concurrency-safe in Postgres: simultaneous
                # creators race in the system catalog and all but one fail on
                # pg_type_typname_nsp_index, which is exactly a multi-node first boot. Both scripts
                # share one transaction so the lock covers them.
                self._dialect.begin_write(conn, "__schema__")
                conn.executescript(self._dialect.adapt_ddl(_SCHEMA_SQL))
                conn.executescript(self._dialect.adapt_ddl(_REGISTRY_SCHEMA_SQL))
                conn.executescript(self._dialect.adapt_ddl(_PUBLICATION_SCHEMA_SQL))
                conn.executescript(self._dialect.adapt_ddl(_NOTEBOOK_WORKERS_SCHEMA_SQL))
                # Created from the constants, which are the latest shape, so it
                # is already current and must not replay migrations that would
                # add what it was born with.
                conn.executescript(self._dialect.adapt_ddl(_SCHEMA_VERSION_SQL))
                self._stamp_schema_version(conn, _LATEST_SCHEMA_VERSION)
                conn.commit()
                return

            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_versions'"
            )
            table_exists = cursor.fetchone() is not None

            if table_exists:
                cursor = conn.execute("PRAGMA table_info(artifact_versions)")
                columns = {row["name"] for row in cursor.fetchall()}

                if "tenant" not in columns:
                    conn.execute("ALTER TABLE artifact_versions ADD COLUMN tenant TEXT")
                    conn.execute("ALTER TABLE artifact_versions ADD COLUMN principal TEXT")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_versions_tenant "
                        "ON artifact_versions(tenant)"
                    )
                    conn.commit()

                # Normalize tenant storage and (re)build the uniqueness index. Tenantless rows must
                # be '' (not NULL): SQLite treats NULLs as distinct, so the (tenant,
                # provenance_hash) uniqueness would not hold. Drop the index, normalize NULL -> '',
                # collapse duplicate ready rows, rebuild. Idempotent: skips once no NULL tenants
                # remain and the index exists.
                has_null_tenant = (
                    conn.execute(
                        "SELECT 1 FROM artifact_versions WHERE tenant IS NULL LIMIT 1"
                    ).fetchone()
                    is not None
                )
                index_exists = (
                    conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' "
                        "AND name='idx_tenant_provenance_unique'"
                    ).fetchone()
                    is not None
                )
                if has_null_tenant or not index_exists:
                    conn.execute("DROP INDEX IF EXISTS idx_tenant_provenance_unique")
                    conn.execute("UPDATE artifact_versions SET tenant = '' WHERE tenant IS NULL")
                    # Keep the newest ready row per (tenant, provenance_hash) and supersede the rest
                    # so the unique index can be built. Superseded rows stay fetchable by explicit
                    # id+version.
                    conn.execute(
                        """
                        UPDATE artifact_versions SET state = 'superseded'
                        WHERE state = 'ready' AND rowid NOT IN (
                            SELECT MAX(rowid) FROM artifact_versions
                            WHERE state = 'ready'
                            GROUP BY tenant, provenance_hash
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_tenant_provenance_unique
                        ON artifact_versions(tenant, provenance_hash)
                        WHERE state = 'ready'
                        """
                    )
                    conn.commit()

                cursor = conn.execute("PRAGMA table_info(artifact_names)")
                name_columns = {row["name"] for row in cursor.fetchall()}

                if "tenant" not in name_columns:
                    # Recreate the table: SQLite cannot change a primary key.
                    conn.execute("ALTER TABLE artifact_names RENAME TO artifact_names_old")
                    conn.execute("""
                        CREATE TABLE artifact_names (
                            name TEXT NOT NULL,
                            artifact_id TEXT NOT NULL,
                            version INTEGER NOT NULL,
                            updated_at REAL NOT NULL,
                            tenant TEXT NOT NULL DEFAULT '',
                            PRIMARY KEY (tenant, name),
                            FOREIGN KEY (artifact_id, version)
                                REFERENCES artifact_versions(id, version)
                        )
                    """)
                    # '' tenant (personal mode), not NULL, so the unique constraint holds.
                    conn.execute("""
                        INSERT INTO artifact_names
                            (name, artifact_id, version, updated_at, tenant)
                        SELECT name, artifact_id, version, updated_at, ''
                        FROM artifact_names_old
                    """)
                    conn.execute("DROP TABLE artifact_names_old")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_names_name ON artifact_names(name)"
                    )
                    conn.commit()
            else:
                conn.executescript(_SCHEMA_SQL)
                conn.executescript(_SCHEMA_VERSION_SQL)
                # Born at the latest shape, so it must not replay migrations
                # that would add what the constants already gave it.
                self._stamp_schema_version(conn, _LATEST_SCHEMA_VERSION)
                conn.commit()

            # Registry and publication tables: idempotent, for fresh and existing databases alike.
            conn.executescript(_REGISTRY_SCHEMA_SQL)
            conn.executescript(_PUBLICATION_SCHEMA_SQL)
            conn.executescript(_NOTEBOOK_WORKERS_SCHEMA_SQL)
            cursor = conn.execute("PRAGMA table_info(registry_audit)")
            audit_columns = {row["name"] for row in cursor.fetchall()}
            if "from_artifact_id" not in audit_columns:
                conn.execute("ALTER TABLE registry_audit ADD COLUMN from_artifact_id TEXT")
            conn.commit()

            # After the SQLite-only steps above brought a legacy database to the baseline.
            # Everything from here is portable.
            self._apply_schema_migrations(conn)
        finally:
            conn.close()

    def _blob_path(self, artifact_id: str, version: int) -> Path:
        """Local blob path for a version. Deprecated: use the blob store methods."""
        return self.blobs_dir / f"{artifact_id}@v={version}.arrow"

    # --- Artifact CRUD ---

    def create_artifact(
        self,
        artifact_id: str,
        provenance_hash: str,
        transform_spec: TransformSpec | None = None,
        input_versions: dict[str, str] | None = None,
        tenant: str | None = None,
        principal: str | None = None,
        minted: bool = False,
    ) -> int:
        """Create a new artifact version in "building" state and return its number.

        Versions are numbered ``MAX(version) + 1`` per id. ``input_versions`` maps input
        URI to version for staleness checks. ``minted`` means the caller made the id up
        (``uuid4``) for this computation, so retention may collect its latest version;
        later versions of a minted id stay minted.
        """
        conn = self._get_connection()
        try:
            # Serialize writers so the MAX(version)+1 read can't race a concurrent create for the
            # same id into a primary-key collision. The key names what is contended so a backend can
            # lock narrowly; SQLite ignores it and locks the file.
            self._dialect.begin_write(conn, artifact_id)
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS next_version, "
                "COALESCE(MAX(minted), 0) AS was_minted "
                "FROM artifact_versions WHERE id = ?",
                (artifact_id,),
            ).fetchone()
            # By name: a Postgres row unpacks to its column names, not values.
            version, was_minted = int(row["next_version"]), row["was_minted"]

            input_versions_json = json.dumps(input_versions) if input_versions else None

            conn.execute(
                """
                INSERT INTO artifact_versions
                    (id, version, state, provenance_hash, created_at,
                     transform_spec, input_versions, tenant, principal, minted)
                VALUES (?, ?, 'building', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    version,
                    provenance_hash,
                    time.time(),
                    transform_spec.to_json() if transform_spec else None,
                    input_versions_json,
                    # Tenantless rows store '' (never NULL) so the uniqueness index treats them as
                    # equal, as names/aliases/tags do.
                    tenant if tenant is not None else "",
                    principal,
                    1 if minted or was_minted else 0,
                ),
            )
            conn.commit()
            return version
        finally:
            conn.close()

    @staticmethod
    def _ready_with_provenance(
        conn: StoreConnection, record: ArtifactVersion
    ) -> tuple[str, int] | None:
        """The ready row this record's computation already occupies, if any.

        Scoped exactly as ``idx_tenant_provenance_unique`` (ready only, per tenant,
        ``''`` and NULL the same), and run on the caller's write transaction so no row
        can appear between the check and the insert.
        """
        if record.state != "ready":
            # The index covers ready rows only, so nothing else can collide.
            return None

        tenant = record.tenant if record.tenant is not None else ""
        if tenant:
            row = conn.execute(
                "SELECT id, version FROM artifact_versions "
                "WHERE provenance_hash = ? AND state = 'ready' AND tenant = ?",
                (record.provenance_hash, tenant),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, version FROM artifact_versions "
                "WHERE provenance_hash = ? AND state = 'ready' "
                "AND (tenant = '' OR tenant IS NULL)",
                (record.provenance_hash,),
            ).fetchone()
        return (row["id"], row["version"]) if row is not None else None

    def _import_no_op(
        self, conn: StoreConnection, record: ArtifactVersion
    ) -> ImportedArtifact | None:
        """Where this record already lives here, else ``None``.

        Either the exact ``id@v=N`` or the same computation under another id; both
        mean nothing is written.
        """
        existing = conn.execute(
            "SELECT provenance_hash FROM artifact_versions WHERE id = ? AND version = ?",
            (record.id, record.version),
        ).fetchone()
        if existing is not None:
            # Same id *and* same computation is the repeated import this is
            # for. A different computation under that id is two artifacts
            # claiming one name, and the caller has to be told: it is about to
            # name, tag and build on what it believes it just sent.
            if (existing["provenance_hash"] or "") != record.provenance_hash:
                raise ArtifactImportConflict(
                    f"{record.id}@v={record.version} is already here, holding a "
                    f"different computation. Import it under a fresh id."
                )
            return ImportedArtifact(record.id, record.version, written=False)

        duplicate = self._ready_with_provenance(conn, record)
        if duplicate is not None:
            return ImportedArtifact(duplicate[0], duplicate[1], written=False)
        return None

    def _require_current_schema(self) -> None:
        """Refuse a store this release would migrate, without touching it."""
        conn = self._get_connection()
        try:
            current = None
            if self._dialect.schema_exists(conn, "schema_version"):
                row = conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
                current = row["version"] if row is not None else None
        finally:
            conn.close()
        if current is not None and current > _LATEST_SCHEMA_VERSION:
            raise _newer_schema_error(current)
        if current is None or current < _LATEST_SCHEMA_VERSION:
            raise StoreSchemaMismatch(
                f"The artifact store's schema is version {current or _BASELINE_SCHEMA_VERSION}, "
                f"older than this Strata ({_LATEST_SCHEMA_VERSION}). This command only reads, "
                "so it does not migrate it. Back the store up, then start this release's "
                "server on it once (a notebook's store migrates when the notebook is "
                "opened); or inspect it with the release that wrote it."
            )

    def _apply_schema_migrations(self, conn: StoreConnection) -> None:
        """Bring one database up to ``_LATEST_SCHEMA_VERSION``; a cheap no-op once current."""
        conn.executescript(self._dialect.adapt_ddl(_SCHEMA_VERSION_SQL))
        row = conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
        current = row["version"] if row is not None and row["version"] is not None else None

        if current is None:
            # A database that predates this table is at the baseline by definition: stamp it and let
            # the migrations below carry it forward.
            current = _BASELINE_SCHEMA_VERSION
            self._stamp_schema_version(conn, current)
        elif current > _LATEST_SCHEMA_VERSION:
            # A newer Strata migrated this store; running on it would write rows that
            # Strata does not expect.
            raise _newer_schema_error(current)

        for migration in _MIGRATIONS:
            if migration.version <= current:
                continue
            logger.info(
                "Applying schema migration %d (%s)", migration.version, migration.description
            )
            migration.apply(conn, self._dialect)
            self._stamp_schema_version(conn, migration.version)
        conn.commit()

    def _stamp_schema_version(self, conn: StoreConnection, version: int) -> None:
        """Record a schema version, at most once.

        A check rather than an upsert because the dialects spell upserts differently;
        the migration lock makes the race impossible.
        """
        existing = conn.execute(
            "SELECT 1 FROM schema_version WHERE version = ?", (version,)
        ).fetchone()
        if existing is not None:
            return
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, time.time()),
        )

    def _complete_imported(
        self,
        landed: ImportedArtifact,
        record: ArtifactVersion,
        blob: bytes | Path | None,
    ) -> None:
        """Fill in what an earlier, less complete write of this row left out.

        Import idempotency is by completeness, not existence. Bytes are rewritten in
        case a backend lost them. Empty lineage is filled in, because the team cache
        writes by provenance with ``inputs=[]`` and usually lands before a promotion
        that knows the inputs. Bytes, id, version and provenance hash never change.
        """
        if blob is not None and not self.blob_exists(landed.id, landed.version):
            self._write_imported_blob(*self._blob_key(landed.id, landed.version), blob)

        if not record.input_versions:
            return

        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, landed.id)
            row = conn.execute(
                "SELECT input_versions FROM artifact_versions WHERE id = ? AND version = ?",
                (landed.id, landed.version),
            ).fetchone()
            # Only ever fills an absent value. A row that already names its
            # inputs is left exactly as it is, so an import can add history and
            # never rewrite it.
            if row is not None and not row["input_versions"]:
                conn.execute(
                    "UPDATE artifact_versions SET input_versions = ? WHERE id = ? AND version = ?",
                    (record.input_versions, landed.id, landed.version),
                )
            conn.commit()
        finally:
            conn.close()

    def _write_imported_blob(self, blob_id: str, version: int, blob: bytes | Path) -> None:
        """Write an import's bytes under blob key ``blob_id``, from memory or a local file.

        A file is published rather than read, so a staged import never holds the
        whole artifact in memory.
        """
        if isinstance(blob, Path):
            self.blob_store.publish_blob_from_path(blob_id, version, blob)
            return
        with self.blob_store.open_blob_writer(blob_id, version) as writer:
            writer.write(blob)

    def _drop_import_attempt(self, record: ArtifactVersion, attempt: str | None) -> None:
        """Remove bytes an import wrote for a row it did not insert."""
        if attempt is not None:
            self.blob_store.delete_blob(attempt_blob_id(record.id, attempt), record.version)

    def stage_import_blob(
        self,
        tenant: str | None,
        content_sha256: str,
        source_path: Path,
        byte_size: int,
        *,
        now: float | None = None,
    ) -> None:
        """Hold bytes, already checked against ``content_sha256``, for an import.

        Restaging the same digest refreshes the hold. Each call also drops this
        tenant's stagings older than ``IMPORT_STAGING_TTL_SECONDS``.
        """
        now = time.time() if now is None else now
        blob_id, version = staged_import_key(tenant, content_sha256)
        self.blob_store.publish_blob_from_path(blob_id, version, source_path)

        effective_tenant = tenant or ""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO import_staging (tenant, content_sha256, byte_size, staged_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant, content_sha256) DO UPDATE SET
                    byte_size = excluded.byte_size, staged_at = excluded.staged_at
                """,
                (effective_tenant, content_sha256, byte_size, now),
            )
            stale = conn.execute(
                "SELECT content_sha256 FROM import_staging WHERE tenant = ? AND staged_at < ?",
                (effective_tenant, now - IMPORT_STAGING_TTL_SECONDS),
            ).fetchall()
            conn.commit()
        finally:
            conn.close()
        for row in stale:
            self.release_staged_import(tenant, row["content_sha256"])

    def open_staged_import(self, tenant: str | None, content_sha256: str):
        """A reader for bytes *tenant* staged under *content_sha256*, or ``None``.

        Only the tenant that uploaded them can import them.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM import_staging WHERE tenant = ? AND content_sha256 = ?",
                (tenant or "", content_sha256),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        blob_id, version = staged_import_key(tenant, content_sha256)
        return self.blob_store.open_blob_reader(blob_id, version)

    def release_staged_import(self, tenant: str | None, content_sha256: str) -> None:
        """Drop staged bytes, once an import has used them or never will."""
        blob_id, version = staged_import_key(tenant, content_sha256)
        self.blob_store.delete_blob(blob_id, version)
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM import_staging WHERE tenant = ? AND content_sha256 = ?",
                (tenant or "", content_sha256),
            )
            conn.commit()
        finally:
            conn.close()

    def import_artifact(
        self, record: ArtifactVersion, blob: bytes | Path | None
    ) -> ImportedArtifact:
        """Copy an artifact from another store, keeping its id and version.

        The version is preserved because lineage edges name ``id@v=N``;
        ``create_artifact`` would renumber. The row is written as in the source,
        including ``provenance_hash``, ``principal`` and ``created_at``.

        Returns where the record landed. When the store already holds that exact
        ``id@v=N``, or the same computation under another id (one ready row per
        ``(tenant, provenance_hash)``), nothing is written and the existing row is
        returned, so descendants can point at it; the existing row is never
        superseded, since it may be named or published.

        Bytes are written before the row, so a crash leaves collectable bytes with no
        row rather than a ready row with no bytes that no retry would repair. They go
        under a key of their own, recorded on the row, so an import that loses a race
        for the same ``id@v=N`` cannot overwrite the bytes of the one that won. The
        check inside the write transaction is authoritative.
        """
        reject_unsafe_artifact_id(record.id)
        conn = self._get_connection()
        try:
            no_op = self._import_no_op(conn, record)
        finally:
            conn.close()
        if no_op is not None:
            self._complete_imported(no_op, record, blob)
            self.record_use(no_op.id, no_op.version)
            return no_op

        attempt = None
        if blob is not None:
            attempt = secrets.token_hex(16)
            self._write_imported_blob(attempt_blob_id(record.id, attempt), record.version, blob)

        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, record.id)
            try:
                no_op = self._import_no_op(conn, record)
            except ArtifactImportConflict:
                self._drop_import_attempt(record, attempt)
                raise
            if no_op is not None:
                conn.commit()
                self._drop_import_attempt(record, attempt)
                self.record_use(no_op.id, no_op.version)
                return no_op

            # The copy keeps the source's created_at, which is when the result
            # was computed; its last use is now, so retention does not read a
            # chain someone just copied here as long idle.
            conn.execute(
                """
                INSERT INTO artifact_versions
                    (id, version, state, provenance_hash, schema_json, row_count,
                     byte_size, created_at, transform_spec, input_versions,
                     tenant, principal, content_sha256, last_used_at, blob_attempt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.version,
                    record.state,
                    record.provenance_hash,
                    record.schema_json,
                    record.row_count,
                    record.byte_size,
                    record.created_at,
                    record.transform_spec,
                    record.input_versions,
                    record.tenant if record.tenant is not None else "",
                    record.principal,
                    # The source's digest travels with the row, and the bytes
                    # were checked against it before anything was written. A
                    # copy that recomputed it locally would agree by
                    # construction and prove nothing.
                    record.content_sha256
                    or (hashlib.sha256(blob).hexdigest() if isinstance(blob, bytes) else None),
                    time.time(),
                    attempt,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return ImportedArtifact(record.id, record.version, written=True)

    def finalize_artifact(
        self,
        artifact_id: str,
        version: int,
        schema_json: str,
        row_count: int | None,
        byte_size: int,
        content_sha256: str | None = None,
        *,
        blob_attempt: str | None = None,
        fence: BuildFence | None = None,
    ) -> ArtifactVersion | None:
        """Mark a building artifact ready after its blob is written.

        Idempotent. When a different id already holds this ``(tenant,
        provenance_hash)`` ready, this version is marked superseded, its blob is
        dropped for the existing one's (still readable by its own id and version), and
        the existing one is returned. An older ready version of the same id is
        superseded.

        ``content_sha256`` saves rehashing the blob when the caller has it.
        ``blob_attempt`` records which attempt's bytes the version reads. ``fence``
        runs inside the transaction for a transform build and must confirm the lease
        before anything commits.

        Raises:
            ValueError: If the artifact is not found or not in "building" state.
            BuildLeaseLost: If ``fence`` found the lease held by another attempt.
        """

        def _commit_through_fence(conn: StoreConnection, target_id: str, target_v: int) -> None:
            if fence is not None and not fence(conn, target_id, target_v):
                conn.rollback()
                raise BuildLeaseLost(
                    f"{artifact_id}@v={version}: the build's lease is held by another attempt"
                )
            conn.commit()

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, version, state, provenance_hash, tenant
                FROM artifact_versions
                WHERE id = ? AND version = ?
                """,
                (artifact_id, version),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Artifact {artifact_id}@v={version} not found")

            if row["state"] == "ready":
                # Already finalized: idempotent.
                _commit_through_fence(conn, artifact_id, version)
                return self.get_artifact(artifact_id, version)

            if row["state"] != "building":
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} not in building state "
                    f"(state={row['state']})"
                )

            provenance_hash = row["provenance_hash"]
            tenant = row["tenant"]

            # Two builds with the same (tenant, provenance_hash) can complete at once.
            existing = self.find_by_provenance(provenance_hash, tenant=tenant)
            if existing is not None and existing.id != artifact_id:
                duplicate_blob = self._supersede_duplicate(
                    conn, artifact_id, version, schema_json, row_count, existing, blob_attempt
                )
                _commit_through_fence(conn, existing.id, existing.version)
                self.blob_store.delete_blob(*duplicate_blob)
                return existing

            # Recorded here, not at write time: every write path (bytes, streamed writer, file,
            # import) arrives here, and this is the moment the bytes become final.
            digest = content_sha256 or self.blob_digest(artifact_id, version, blob_attempt)

            if existing is not None and existing.version != version:
                # Same id, older ready version with the same provenance: a refresh rebuild.
                # Supersede the old version so the rebuild becomes canonical, since the partial
                # unique index allows only ONE ready row per (tenant, provenance_hash). The old
                # version stays fetchable by explicit (id, version).
                conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'superseded'
                    WHERE id = ? AND version = ? AND state = 'ready'
                    """,
                    (existing.id, existing.version),
                )

            try:
                cursor = conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'ready', schema_json = ?, row_count = ?, byte_size = ?,
                        content_sha256 = ?, blob_attempt = ?, last_used_at = ?
                    WHERE id = ? AND version = ? AND state = 'building'
                    """,
                    (
                        schema_json,
                        row_count,
                        byte_size,
                        digest,
                        blob_attempt,
                        # Idle time counts from now: a long build would otherwise finalize
                        # looking idle since it started and be the first a sweep collects.
                        time.time(),
                        artifact_id,
                        version,
                    ),
                )
                if cursor.rowcount == 0:
                    # Another process may have finalized it.
                    conn.rollback()
                    if fence is not None:
                        raise BuildLeaseLost(
                            f"{artifact_id}@v={version} was finalized by another attempt"
                        )
                    return self.get_artifact(artifact_id, version)
                _commit_through_fence(conn, artifact_id, version)
                return self.get_artifact(artifact_id, version)
            except self._dialect.integrity_error:
                # Another artifact with this (tenant, provenance_hash) was finalized first; return
                # it.
                conn.rollback()
                existing = self.find_by_provenance(provenance_hash, tenant=tenant)
                if existing is not None:
                    duplicate_blob = self._supersede_duplicate(
                        conn, artifact_id, version, schema_json, row_count, existing, blob_attempt
                    )
                    _commit_through_fence(conn, existing.id, existing.version)
                    self.blob_store.delete_blob(*duplicate_blob)
                    return existing
                raise
        finally:
            conn.close()

    def find_version_by_provenance(
        self, artifact_id: str, provenance_hash: str, tenant: str | None = None
    ) -> ArtifactVersion | None:
        """Return this artifact's newest version carrying ``provenance_hash``.

        Unlike :meth:`find_by_provenance` this stays inside one id and counts
        superseded versions, which is how a result returns after being overtaken.
        Tenant-scoped; tenantless rows are stored as ``''``.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, version, state, provenance_hash, schema_json,
                       row_count, byte_size, created_at, transform_spec,
                       input_versions, tenant, principal, content_sha256
                FROM artifact_versions
                WHERE id = ? AND provenance_hash = ? AND tenant = ?
                  AND state IN ('ready', 'superseded')
                ORDER BY version DESC
                LIMIT 1
                """,
                (artifact_id, provenance_hash, tenant if tenant is not None else ""),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return ArtifactVersion(
                id=row["id"],
                version=row["version"],
                state=row["state"],
                provenance_hash=row["provenance_hash"],
                schema_json=row["schema_json"],
                row_count=row["row_count"],
                byte_size=row["byte_size"],
                created_at=row["created_at"],
                transform_spec=row["transform_spec"],
                input_versions=row["input_versions"],
                tenant=row["tenant"],
                principal=row["principal"],
                content_sha256=row["content_sha256"],
            )
        finally:
            conn.close()

    def promote_version(self, artifact_id: str, version: int) -> ArtifactVersion | None:
        """Re-record an existing version so it becomes the artifact's latest.

        Writes a new version with the same provenance, spec and blob, then finalizes
        it, superseding the source; the result matches a recompute without the
        computing. Returns ``None`` when the source version or its blob is gone, so
        the caller recomputes.
        """
        source = self.get_artifact(artifact_id, version)
        if source is None:
            return None
        if source.state not in ("ready", "superseded"):
            # A building row may be a partial or abandoned write and a failed
            # one was rejected on purpose; neither is a result to make current.
            return None
        # Open the source before registering anything. A blob that has gone
        # missing must leave no trace behind: registering the row first would
        # strand a ``building`` version pointing at nothing when the read fails.
        reader_cm = self.open_blob_reader(artifact_id, version)
        if reader_cm is None:
            return None

        conn = self._get_connection()
        try:
            # Same write serialization as create_artifact: MAX(version)+1 must
            # not race a concurrent create for this id.
            self._dialect.begin_write(conn, artifact_id)
            cursor = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM artifact_versions WHERE id = ?",
                (artifact_id,),
            )
            new_version = int(cursor.fetchone()[0])
            conn.execute(
                """
                INSERT INTO artifact_versions
                    (id, version, state, provenance_hash, created_at,
                     transform_spec, input_versions, tenant, principal)
                VALUES (?, ?, 'building', ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    new_version,
                    source.provenance_hash,
                    time.time(),
                    source.transform_spec,
                    source.input_versions,
                    source.tenant if source.tenant is not None else "",
                    source.principal,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # Streamed, not read_blob/write_blob: an artifact can be a multi-GB frame, and pulling it
        # through process memory (a full download and re-upload on object stores) to re-point a
        # pointer would cost more than the run this avoids.
        from strata.blob_store import BLOB_STREAM_CHUNK_BYTES

        copied = 0
        with (
            reader_cm as reader,
            self.blob_store.open_blob_writer(artifact_id, new_version) as writer,
        ):
            # Hashed in passing: the bytes go by anyway, and reading them again for the digest would
            # be wasteful.
            hasher = hashlib.sha256()
            while chunk := reader.read(BLOB_STREAM_CHUNK_BYTES):
                writer.write(chunk)
                hasher.update(chunk)
                copied += len(chunk)

        # Canonical, not finalize_artifact: the caller resolves by *this* id, and dedup onto
        # another id holding the provenance (two notebooks running the same cell) would drop the
        # copy just written for that id's bytes.
        (finalized,) = self.finalize_canonical_together(
            [
                StagedVersion(
                    artifact_id=artifact_id,
                    version=new_version,
                    schema_json=source.schema_json or "",
                    row_count=source.row_count,
                    byte_size=copied,
                    content_sha256=hasher.hexdigest(),
                )
            ]
        )
        return finalized

    def _reclaim_blob(self, artifact_id: str, version: int) -> None:
        """Give a version dedup pointed at another's bytes a copy of its own.

        For a held version whose canonical is being deleted: it must keep its bytes.

        Raises:
            ValueError: If the canonical's blob is gone, so there is nothing to copy.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT superseded_by FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            ).fetchone()
        finally:
            conn.close()
        if row is None or not row["superseded_by"]:
            return
        reader_cm = self.blob_store.open_blob_reader(*self._blob_key(artifact_id, version))
        if reader_cm is None:
            raise ValueError(
                f"{artifact_id}@v={version} reads {row['superseded_by']}, whose blob is gone"
            )
        from strata.blob_store import BLOB_STREAM_CHUNK_BYTES

        with reader_cm as src, self.blob_store.open_blob_writer(artifact_id, version) as dst:
            while chunk := src.read(BLOB_STREAM_CHUNK_BYTES):
                dst.write(chunk)

    def finalize_canonical_together(
        self, staged: list[StagedVersion]
    ) -> list[ArtifactVersion | None]:
        """Make several ``building`` versions ready in one transaction, all or none.

        For the outputs of one notebook cell run: finalized one at a time, a crash in
        between would leave values from two runs current together. Each becomes the
        single ready row for its ``(tenant, provenance_hash)`` under its own id,
        superseding any other.

        Raises:
            ValueError: If a version is not found or not in "building" state.
        """
        if not staged:
            return []
        # Retried: a keyed advisory lock only serializes writers sharing an artifact id, so a
        # writer under another id can commit a ready row with our provenance between our
        # supersede and our promote, and the uniqueness index rejects us.
        for attempt in range(_CANONICAL_PROMOTE_ATTEMPTS):
            conn = self._get_connection()
            try:
                self._dialect.begin_write(conn, staged[0].artifact_id)
                now = time.time()
                for item in staged:
                    row = conn.execute(
                        "SELECT provenance_hash, tenant, state FROM artifact_versions "
                        "WHERE id = ? AND version = ?",
                        (item.artifact_id, item.version),
                    ).fetchone()
                    if row is None or row["state"] != "building":
                        conn.rollback()
                        raise ValueError(
                            f"Artifact {item.artifact_id}@v={item.version} not in building state"
                        )
                    conn.execute(
                        """
                        UPDATE artifact_versions SET state = 'superseded'
                        WHERE provenance_hash = ? AND state = 'ready'
                          AND COALESCE(tenant, '') = COALESCE(?, '')
                          AND NOT (id = ? AND version = ?)
                        """,
                        (row["provenance_hash"], row["tenant"], item.artifact_id, item.version),
                    )
                    conn.execute(
                        """
                        UPDATE artifact_versions
                        SET state = 'ready', schema_json = ?, row_count = ?, byte_size = ?,
                            content_sha256 = ?, last_used_at = ?
                        WHERE id = ? AND version = ?
                        """,
                        (
                            item.schema_json,
                            item.row_count,
                            item.byte_size,
                            item.content_sha256,
                            now,
                            item.artifact_id,
                            item.version,
                        ),
                    )
                conn.commit()
                break
            except self._dialect.integrity_error:
                conn.rollback()
                if attempt == _CANONICAL_PROMOTE_ATTEMPTS - 1:
                    raise
            finally:
                conn.close()
        return [self.get_artifact(item.artifact_id, item.version) for item in staged]

    def finalize_and_set_name(
        self,
        artifact_id: str,
        version: int,
        schema_json: str,
        row_count: int,
        byte_size: int,
        name: str | None = None,
        tenant: str | None = None,
        *,
        content_sha256: str | None = None,
        blob_attempt: str | None = None,
        fence: BuildFence | None = None,
    ) -> ArtifactVersion | None:
        """Finalize an artifact and set a name pointer in one transaction.

        On a provenance duplicate the name points at the existing artifact. ``name``
        of None sets no name; ``content_sha256``, ``blob_attempt`` and ``fence`` as for
        :meth:`finalize_artifact`.

        Raises:
            ValueError: If the artifact is not found or not in "building" state.
            BuildLeaseLost: If ``fence`` found the lease held by another attempt.
        """

        def _commit_through_fence(conn: StoreConnection, target_id: str, target_v: int) -> None:
            if fence is not None and not fence(conn, target_id, target_v):
                conn.rollback()
                raise BuildLeaseLost(
                    f"{artifact_id}@v={version}: the build's lease is held by another attempt"
                )
            conn.commit()

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, version, state, provenance_hash, tenant
                FROM artifact_versions
                WHERE id = ? AND version = ?
                """,
                (artifact_id, version),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Artifact {artifact_id}@v={version} not found")

            if row["state"] == "ready":
                # Already finalized: set name and return (idempotent).
                if name:
                    artifact_tenant = row["tenant"] if row["tenant"] else None
                    if not self._can_assign_name_for_tenant(artifact_tenant, tenant):
                        raise ValueError(
                            f"Artifact {artifact_id}@v={version} belongs to another tenant"
                        )
                    self._set_name_in_connection(conn, name, artifact_id, version, tenant)
                _commit_through_fence(conn, artifact_id, version)
                return self.get_artifact(artifact_id, version)

            if row["state"] != "building":
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} not in building state "
                    f"(state={row['state']})"
                )

            provenance_hash = row["provenance_hash"]
            artifact_tenant = row["tenant"]
            normalized_artifact_tenant = artifact_tenant if artifact_tenant else None

            if name and not self._can_assign_name_for_tenant(normalized_artifact_tenant, tenant):
                raise ValueError(f"Artifact {artifact_id}@v={version} belongs to another tenant")

            existing = self.find_by_provenance(provenance_hash, tenant=artifact_tenant)
            if existing is not None and existing.id != artifact_id:
                # Another artifact has this provenance: this one is overtaken and the name points
                # at the existing one.
                duplicate_blob = self._supersede_duplicate(
                    conn, artifact_id, version, schema_json, row_count, existing, blob_attempt
                )
                if name:
                    self._set_name_in_connection(conn, name, existing.id, existing.version, tenant)
                _commit_through_fence(conn, existing.id, existing.version)
                self.blob_store.delete_blob(*duplicate_blob)
                return existing

            digest = content_sha256 or self.blob_digest(artifact_id, version, blob_attempt)

            if existing is not None and existing.version != version:
                # Refresh rebuild: same id, older ready version with the same provenance. Supersede
                # the old version (still fetchable by explicit id+version) so the rebuild becomes
                # canonical.
                conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'superseded'
                    WHERE id = ? AND version = ? AND state = 'ready'
                    """,
                    (existing.id, existing.version),
                )

            try:
                cursor = conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'ready', schema_json = ?, row_count = ?, byte_size = ?,
                        content_sha256 = ?, blob_attempt = ?, last_used_at = ?
                    WHERE id = ? AND version = ? AND state = 'building'
                    """,
                    (
                        schema_json,
                        row_count,
                        byte_size,
                        digest,
                        blob_attempt,
                        time.time(),
                        artifact_id,
                        version,
                    ),
                )
                if cursor.rowcount == 0:
                    # Another process may have finalized it.
                    conn.rollback()
                    if fence is not None:
                        raise BuildLeaseLost(
                            f"{artifact_id}@v={version} was finalized by another attempt"
                        )
                    artifact = self.get_artifact(artifact_id, version)
                    if artifact and artifact.state == "ready" and name:
                        self.set_name(name, artifact_id, version, tenant)
                    return artifact

                if name:
                    self._set_name_in_connection(conn, name, artifact_id, version, tenant)

                _commit_through_fence(conn, artifact_id, version)
                return self.get_artifact(artifact_id, version)

            except self._dialect.integrity_error:
                # Duplicate provenance: another artifact was finalized first.
                conn.rollback()
                existing = self.find_by_provenance(provenance_hash, tenant=artifact_tenant)
                if existing is not None:
                    duplicate_blob = self._supersede_duplicate(
                        conn, artifact_id, version, schema_json, row_count, existing, blob_attempt
                    )
                    if name:
                        self._set_name_in_connection(
                            conn, name, existing.id, existing.version, tenant
                        )
                    _commit_through_fence(conn, existing.id, existing.version)
                    self.blob_store.delete_blob(*duplicate_blob)
                    return existing
                raise
        finally:
            conn.close()

    def _serialize_audit(self, conn: StoreConnection) -> None:
        """On Postgres, hold the audit lock until commit, so ``seq`` order is commit order.

        A ``BIGSERIAL`` is drawn at insert, so without it two writers can commit out of order and a
        follower of ``read_events`` skips the earlier one for good. Take it before a transaction's
        first write: waiting for it while holding a row lock can deadlock. SQLite already
        serializes writers.
        """
        if self._dialect.name != "sqlite":
            self._dialect.begin_write(conn, "registry_audit")

    def _audit_in_connection(
        self,
        conn: StoreConnection,
        *,
        action: str,
        name: str | None = None,
        alias: str | None = None,
        artifact_id: str | None = None,
        from_artifact_id: str | None = None,
        from_version: int | None = None,
        to_version: int | None = None,
        key: str | None = None,
        value: str | None = None,
        actor: str | None = None,
        tenant: str | None = None,
    ) -> None:
        """Append one registry audit row inside the caller's transaction.

        It commits or rolls back with the mutation it records.
        """
        self._serialize_audit(conn)
        conn.execute(
            """
            INSERT INTO registry_audit
                (at, actor, action, name, alias, artifact_id, from_artifact_id,
                 from_version, to_version, key, value, tenant)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                actor,
                action,
                name,
                alias,
                artifact_id,
                from_artifact_id,
                from_version,
                to_version,
                key,
                value,
                tenant if tenant is not None else "",
            ),
        )

    def _set_name_in_connection(
        self,
        conn: StoreConnection,
        name: str,
        artifact_id: str,
        version: int,
        tenant: str | None,
        actor: str | None = None,
    ) -> None:
        """Set a name within the caller's transaction."""
        # '' not NULL for personal mode: SQLite NULL != NULL in unique constraints.
        effective_tenant = tenant if tenant is not None else ""

        # Audit: record what the name pointed at before this move.
        cursor = conn.execute(
            "SELECT artifact_id, version FROM artifact_names WHERE name = ? AND tenant = ?",
            (name, effective_tenant),
        )
        previous = cursor.fetchone()
        self._audit_in_connection(
            conn,
            action="name_set",
            name=name,
            artifact_id=artifact_id,
            from_artifact_id=previous["artifact_id"] if previous else None,
            from_version=previous["version"] if previous else None,
            to_version=version,
            actor=actor,
            tenant=tenant,
        )

        conn.execute(
            """
            INSERT INTO artifact_names (name, artifact_id, version, updated_at, tenant)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(tenant, name) DO UPDATE SET
                artifact_id = excluded.artifact_id,
                version = excluded.version,
                updated_at = excluded.updated_at
            """,
            (name, artifact_id, version, time.time(), effective_tenant),
        )

    def _supersede_duplicate(
        self,
        conn: StoreConnection,
        artifact_id: str,
        version: int,
        schema_json: str,
        row_count: int | None,
        existing: ArtifactVersion,
        blob_attempt: str | None,
    ) -> tuple[str, int]:
        """Record a finished build ``existing`` already holds ready under the same provenance.

        Superseded, not failed: the computation succeeded, so the version stays readable
        by the id and version its caller was handed, until retention collects it. It
        reads ``existing``'s bytes through ``superseded_by`` rather than keeping a second
        copy; returns the key of its own blob for the caller to delete once committed.
        """
        conn.execute(
            """
            UPDATE artifact_versions
            SET state = 'superseded', schema_json = ?, row_count = ?, byte_size = ?,
                content_sha256 = ?, blob_attempt = NULL, superseded_by = ?, last_used_at = ?
            WHERE id = ? AND version = ?
            """,
            (
                schema_json,
                row_count,
                existing.byte_size,
                existing.content_sha256,
                f"{existing.id}@v={existing.version}",
                time.time(),
                artifact_id,
                version,
            ),
        )
        return (
            attempt_blob_id(artifact_id, blob_attempt) if blob_attempt else artifact_id
        ), version

    def fail_artifact(self, artifact_id: str, version: int) -> None:
        """Mark an artifact version as failed."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                UPDATE artifact_versions
                SET state = 'failed'
                WHERE id = ? AND version = ? AND state = 'building'
                """,
                (artifact_id, version),
            )
            conn.commit()
        finally:
            conn.close()

    def get_artifact(self, artifact_id: str, version: int) -> ArtifactVersion | None:
        """Get artifact version metadata, or None if not found."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, version, state, provenance_hash, schema_json,
                       row_count, byte_size, created_at, transform_spec,
                       input_versions, tenant, principal, content_sha256
                FROM artifact_versions
                WHERE id = ? AND version = ?
                """,
                (artifact_id, version),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return ArtifactVersion(
                id=row["id"],
                version=row["version"],
                state=row["state"],
                provenance_hash=row["provenance_hash"],
                schema_json=row["schema_json"],
                row_count=row["row_count"],
                byte_size=row["byte_size"],
                created_at=row["created_at"],
                transform_spec=row["transform_spec"],
                input_versions=row["input_versions"],
                tenant=row["tenant"],
                principal=row["principal"],
                content_sha256=row["content_sha256"],
            )
        finally:
            conn.close()

    def id_tenant(self, artifact_id: str) -> str | None:
        """The tenant ('' for none) holding ``artifact_id``, or None for an unused id.

        Any version counts, a ``building`` one included: an upload holds its id from
        the moment its row exists.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT tenant FROM artifact_versions WHERE id = ? ORDER BY version LIMIT 1",
                (artifact_id,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else (row["tenant"] or "")

    def get_latest_version(self, artifact_id: str) -> ArtifactVersion | None:
        """Get the current value of an artifact: its newest ready or superseded version.

        Superseded counts because an id's newest result is superseded when another id
        claims the same provenance (two notebooks with identical cells); it still has
        its bytes.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, version, state, provenance_hash, schema_json,
                       row_count, byte_size, created_at, transform_spec,
                       input_versions, tenant, principal, content_sha256
                FROM artifact_versions
                WHERE id = ? AND state IN ('ready', 'superseded')
                ORDER BY version DESC
                LIMIT 1
                """,
                (artifact_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return ArtifactVersion(
                id=row["id"],
                version=row["version"],
                state=row["state"],
                provenance_hash=row["provenance_hash"],
                schema_json=row["schema_json"],
                row_count=row["row_count"],
                byte_size=row["byte_size"],
                created_at=row["created_at"],
                transform_spec=row["transform_spec"],
                input_versions=row["input_versions"],
                tenant=row["tenant"],
                principal=row["principal"],
                content_sha256=row["content_sha256"],
            )
        finally:
            conn.close()

    def list_latest_by_id_prefix(self, prefix: str) -> list[ArtifactVersion]:
        """Return the current value of each artifact whose id starts with ``prefix``.

        Current as in :meth:`get_latest_version`. Sorted by id, so callers can parse a
        numeric suffix such as a loop cell's ``@iter=<k>`` in order.
        """
        if not prefix:
            return []

        like_pattern = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, MAX(version) AS max_version
                FROM artifact_versions
                WHERE id LIKE ? ESCAPE '\\' AND state IN ('ready', 'superseded')
                GROUP BY id
                ORDER BY id
                """,
                (like_pattern,),
            )
            rows = cursor.fetchall()
            results: list[ArtifactVersion] = []
            for row in rows:
                version = self.get_artifact(row["id"], row["max_version"])
                if version is not None:
                    results.append(version)
            return results
        finally:
            conn.close()

    def find_by_provenance(
        self,
        provenance_hash: str,
        tenant: str | None = None,
    ) -> ArtifactVersion | None:
        """Find the ready artifact with this provenance hash, or None.

        A tenant id scopes to that tenant; ``None`` or ``""`` scopes to the tenantless
        namespace (``''`` or legacy ``NULL``), never "any tenant", so a tenantless build
        cannot dedup against another tenant's artifact.
        """
        conn = self._get_connection()
        try:
            if tenant:
                cursor = conn.execute(
                    """
                    SELECT id, version, state, provenance_hash, schema_json,
                           row_count, byte_size, created_at, transform_spec,
                           input_versions, tenant, principal, content_sha256,
                           last_used_at
                    FROM artifact_versions
                    WHERE provenance_hash = ? AND state = 'ready' AND tenant = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (provenance_hash, tenant),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, version, state, provenance_hash, schema_json,
                           row_count, byte_size, created_at, transform_spec,
                           input_versions, tenant, principal, content_sha256,
                           last_used_at
                    FROM artifact_versions
                    WHERE provenance_hash = ? AND state = 'ready'
                      AND (tenant = '' OR tenant IS NULL)
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (provenance_hash,),
                )
            row = cursor.fetchone()
            if row is None:
                return None
            # A hit is the use retention cares most about: the computation
            # was asked for again and did not have to run.
            self._note_use(conn, row["id"], row["version"], row["last_used_at"])
            return ArtifactVersion(
                id=row["id"],
                version=row["version"],
                state=row["state"],
                provenance_hash=row["provenance_hash"],
                schema_json=row["schema_json"],
                row_count=row["row_count"],
                byte_size=row["byte_size"],
                created_at=row["created_at"],
                transform_spec=row["transform_spec"],
                input_versions=row["input_versions"],
                tenant=row["tenant"],
                principal=row["principal"],
                content_sha256=row["content_sha256"],
            )
        finally:
            conn.close()

    def record_use(self, artifact_id: str, version: int) -> None:
        """Note that ``artifact_id@v=version`` was just used.

        For uses that neither read its bytes nor look it up by provenance, such as
        resolving it as an input, which retention would otherwise not see.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT last_used_at FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            ).fetchone()
            if row is not None:
                self._note_use(conn, artifact_id, version, row["last_used_at"])
        finally:
            conn.close()

    def _note_use(
        self, conn: StoreConnection, artifact_id: str, version: int, last_used_at: float | None
    ) -> None:
        """Record a use, unless one was recorded within ``_USE_RESOLUTION_SECONDS``.

        Advisory: when the write fails (locked file, read-only mount) the read still
        succeeds.
        """
        now = time.time()
        if last_used_at is not None and now - last_used_at < _USE_RESOLUTION_SECONDS:
            return
        try:
            conn.execute(
                "UPDATE artifact_versions SET last_used_at = ? WHERE id = ? AND version = ?",
                (now, artifact_id, version),
            )
            conn.commit()
        except self._dialect.operational_error as exc:
            conn.rollback()
            logger.debug("Could not note a use of %s@v=%d: %s", artifact_id, version, exc)

    # --- Blob I/O ---

    def _blob_key(
        self,
        artifact_id: str,
        version: int,
        attempt: str | None = None,
        *,
        note_use: bool = False,
    ) -> tuple[str, int]:
        """The ``(id, version)`` a version's bytes are stored under in the blob store.

        A transform build writes each attempt under its own id, so an attempt that
        lost its lease writes bytes nobody reads; the row records the promoted
        attempt. A version a duplicate build overtook has no bytes of its own and
        reads its canonical's (``superseded_by``). Everything else uses the artifact
        id. ``attempt`` selects one explicitly.
        """
        if attempt is not None:
            return attempt_blob_id(artifact_id, attempt), version
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT blob_attempt, superseded_by, last_used_at FROM artifact_versions "
                "WHERE id = ? AND version = ?",
                (artifact_id, version),
            ).fetchone()
            if row is None:
                return artifact_id, version
            if note_use:
                self._note_use(conn, artifact_id, version, row["last_used_at"])
            if row["superseded_by"]:
                # Reading through the pointer is a use of the canonical too, so retention
                # cannot collect the bytes while something still reads them this way.
                artifact_id, version = _split_ref(row["superseded_by"])
                row = conn.execute(
                    "SELECT blob_attempt, last_used_at FROM artifact_versions "
                    "WHERE id = ? AND version = ?",
                    (artifact_id, version),
                ).fetchone()
                if row is None:
                    return artifact_id, version
                if note_use:
                    self._note_use(conn, artifact_id, version, row["last_used_at"])
        finally:
            conn.close()
        attempt = row["blob_attempt"]
        return (attempt_blob_id(artifact_id, attempt) if attempt else artifact_id), version

    def write_blob(
        self, artifact_id: str, version: int, data: bytes, attempt: str | None = None
    ) -> None:
        """Write a version's Arrow IPC bytes, under ``attempt``'s id when given."""
        blob_id = attempt_blob_id(artifact_id, attempt) if attempt else artifact_id
        self.blob_store.write_blob(blob_id, version, data)

    def read_blob(self, artifact_id: str, version: int, attempt: str | None = None) -> bytes | None:
        """Read a version's Arrow IPC bytes (or ``attempt``'s), or None if not found."""
        # Reading a version's bytes is a use; reading one build attempt's
        # bytes (finalize checking what it wrote) is not.
        return self.blob_store.read_blob(
            *self._blob_key(artifact_id, version, attempt, note_use=attempt is None)
        )

    def open_blob_reader(self, artifact_id: str, version: int, attempt: str | None = None):
        """Open a streaming reader (context manager) for a blob, or ``None`` if there is none."""
        return self.blob_store.open_blob_reader(
            *self._blob_key(artifact_id, version, attempt, note_use=attempt is None)
        )

    def open_blob_writer(self, artifact_id: str, version: int):
        """Open a streaming blob writer: commits on clean exit, discards on exception."""
        return self.blob_store.open_blob_writer(artifact_id, version)

    def blob_size(self, artifact_id: str, version: int, attempt: str | None = None) -> int | None:
        """Return the size of an artifact blob without materializing it."""
        return self.blob_store.blob_size(*self._blob_key(artifact_id, version, attempt))

    def publish_blob_from_path(
        self, artifact_id: str, version: int, source_path: Path, attempt: str | None = None
    ) -> None:
        """Atomically publish a blob from a prepared local file.

        Blocking; call via ``asyncio.to_thread``. A build passes its ``attempt``.
        """
        blob_id = attempt_blob_id(artifact_id, attempt) if attempt else artifact_id
        self.blob_store.publish_blob_from_path(blob_id, version, source_path)

    def delete_attempt_blob(self, artifact_id: str, version: int, attempt: str) -> None:
        """Remove what a build attempt wrote, once it will never be promoted.

        Keeps the bytes when the version was promoted from this attempt: two finalize
        requests under one lease share an attempt id.
        """
        if self._blob_key(artifact_id, version) == (attempt_blob_id(artifact_id, attempt), version):
            return
        self.blob_store.delete_blob(attempt_blob_id(artifact_id, attempt), version)

    def blob_exists(self, artifact_id: str, version: int, attempt: str | None = None) -> bool:
        """Check whether a version's blob (or ``attempt``'s) exists."""
        return self.blob_store.blob_exists(*self._blob_key(artifact_id, version, attempt))

    # --- Name pointers ---

    @classmethod
    def _require_writable_target(
        cls, row, artifact_id: str, version: int, tenant: str | None
    ) -> None:
        """Raise :class:`ArtifactNotFoundError` unless ``row`` exists and ``tenant`` may use it.

        Checked before the row's state, which would otherwise describe another tenant's artifact.
        """
        if row is None or not cls._can_assign_name_for_tenant(row["tenant"] or None, tenant):
            raise ArtifactNotFoundError(f"Artifact {artifact_id}@v={version} not found")

    @staticmethod
    def _can_assign_name_for_tenant(
        artifact_tenant: str | None,
        requested_tenant: str | None,
    ) -> bool:
        """Return whether a name in ``requested_tenant`` may point at ``artifact_tenant``.

        Tenantless artifacts are assignable from anywhere. Legacy ``"_default"``
        artifacts are assignable by tenantless requests, which only happens in
        single-tenant deployments.
        """
        if artifact_tenant is None:
            return True
        if artifact_tenant == "_default" and not requested_tenant:
            return True
        return artifact_tenant == requested_tenant

    def set_name(
        self,
        name: str,
        artifact_id: str,
        version: int,
        tenant: str | None = None,
        actor: str | None = None,
    ) -> None:
        """Create or update a name pointer, unique per ``(tenant, name)``.

        Raises:
            ValueError: If the target version doesn't exist or isn't ready.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT state, tenant FROM artifact_versions
                WHERE id = ? AND version = ?
                """,
                (artifact_id, version),
            )
            row = cursor.fetchone()
            self._require_writable_target(row, artifact_id, version, tenant)
            if row["state"] != "ready":
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} is not ready (state={row['state']})"
                )

            # Upsert + audit in one transaction (shared with the finalize paths).
            self._set_name_in_connection(conn, name, artifact_id, version, tenant, actor=actor)
            conn.commit()
        finally:
            conn.close()

    def names_for_artifact(
        self,
        artifact_id: str,
        version: int,
        tenant: str | None = None,
    ) -> list[str]:
        """Reverse lookup: registry names currently pointing at this version."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                "SELECT name FROM artifact_names "
                "WHERE artifact_id = ? AND version = ? AND tenant = ? ORDER BY name",
                (artifact_id, version, effective_tenant),
            )
            return [row["name"] for row in cursor.fetchall()]
        finally:
            conn.close()

    def resolve_name(
        self,
        name: str,
        tenant: str | None = None,
    ) -> ArtifactVersion | None:
        """Resolve a name to its pinned artifact version, or None."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                """
                SELECT n.artifact_id, n.version
                FROM artifact_names n
                WHERE n.name = ? AND n.tenant = ?
                """,
                (name, effective_tenant),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self.get_artifact(row["artifact_id"], row["version"])
        finally:
            conn.close()

    def get_name(
        self,
        name: str,
        tenant: str | None = None,
    ) -> ArtifactName | None:
        """Get name pointer metadata, or None if not found."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                """
                SELECT name, artifact_id, version, updated_at, tenant
                FROM artifact_names
                WHERE name = ? AND tenant = ?
                """,
                (name, effective_tenant),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            returned_tenant = row["tenant"] if row["tenant"] else None
            return ArtifactName(
                name=row["name"],
                artifact_id=row["artifact_id"],
                version=row["version"],
                updated_at=row["updated_at"],
                tenant=returned_tenant,
            )
        finally:
            conn.close()

    def delete_name(
        self,
        name: str,
        tenant: str | None = None,
    ) -> bool:
        """Delete a name pointer; return whether it existed."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                "SELECT version FROM artifact_names WHERE name = ? AND tenant = ?",
                (name, effective_tenant),
            )
            previous = cursor.fetchone()
            self._serialize_audit(conn)
            cursor = conn.execute(
                "DELETE FROM artifact_names WHERE name = ? AND tenant = ?",
                (name, effective_tenant),
            )
            if cursor.rowcount > 0:
                self._audit_in_connection(
                    conn,
                    action="name_delete",
                    name=name,
                    from_version=previous["version"] if previous else None,
                    tenant=tenant,
                )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_all_names(self) -> list[ArtifactName]:
        """List name pointers across all tenants, for maintenance and the CLI.

        Request-serving code should use :meth:`list_names` with the caller's tenant.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT name, artifact_id, version, updated_at, tenant
                FROM artifact_names
                ORDER BY name
                """
            )
            return [
                ArtifactName(
                    name=row["name"],
                    artifact_id=row["artifact_id"],
                    version=row["version"],
                    updated_at=row["updated_at"],
                    tenant=row["tenant"] if row["tenant"] else None,
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    # --- Publications (opt-in public read grants), then registry: aliases, tags, audit ---

    @staticmethod
    def _publication_from_row(row, token: str = "") -> Publication:
        """``token`` is the raw token when the caller has it; the row holds only its hash."""
        return Publication(
            token=token,
            id=row["token"],
            artifact_id=row["artifact_id"],
            version=row["version"],
            tenant=row["tenant"] or "",
            title=row["title"],
            published_at=row["published_at"],
            published_by=row["published_by"],
            content_sha256=row["content_sha256"],
            revoked_at=row["revoked_at"],
            authors=_json_entries(row["authors"]),
            external_ids=_json_entries(row["external_ids"]),
        )

    def content_digest(self, artifact_id: str, version: int) -> str | None:
        """The artifact's recorded digest, computing and recording it if absent.

        Older rows are filled on demand rather than by a migration that reads every
        blob. ``None`` when there is no blob.
        """
        artifact = self.get_artifact(artifact_id, version)
        if artifact is None:
            return None
        if artifact.content_sha256:
            return artifact.content_sha256

        digest = self.blob_digest(artifact_id, version)
        if digest is None:
            return None
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE artifact_versions SET content_sha256 = ? "
                "WHERE id = ? AND version = ? AND content_sha256 IS NULL",
                (digest, artifact_id, version),
            )
            conn.commit()
        finally:
            conn.close()
        return digest

    def blob_digest(self, artifact_id: str, version: int, attempt: str | None = None) -> str | None:
        """SHA-256 of an artifact's bytes, streamed so memory stays bounded. ``None`` if no blob."""
        # Below open_blob_reader: hashing a version's bytes (finalize, verify)
        # is bookkeeping, not somebody using the result.
        reader_cm = self.blob_store.open_blob_reader(*self._blob_key(artifact_id, version, attempt))
        if reader_cm is None:
            return None
        hasher = hashlib.sha256()
        with reader_cm as reader:
            while chunk := reader.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()

    def publish_artifact(
        self,
        artifact_id: str,
        version: int,
        *,
        tenant: str | None = None,
        published_by: str | None = None,
        title: str | None = None,
        authors: list[dict[str, str]] | None = None,
        external_ids: list[dict[str, str]] | None = None,
    ) -> Publication:
        """Grant unauthenticated read access to one artifact version.

        Publishing the same version again returns the existing active grant, so one
        revocation always withdraws the artifact.

        Raises:
            ValueError: If the artifact does not exist, is not readable, or belongs
                to a different tenant.
        """
        effective_tenant = tenant if tenant is not None else ""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT state, tenant, content_sha256 FROM artifact_versions "
                "WHERE id = ? AND version = ?",
                (artifact_id, version),
            ).fetchone()
            if row is None:
                raise ValueError(f"Artifact {artifact_id}@v={version} not found")
            if row["state"] not in ("ready", "superseded"):
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} is not readable (state={row['state']})"
                )
            artifact_tenant = row["tenant"] or ""
            if artifact_tenant != effective_tenant:
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} belongs to tenant "
                    f"{artifact_tenant!r}, cannot publish in tenant {effective_tenant!r}"
                )

            existing = conn.execute(
                "SELECT * FROM artifact_publications "
                "WHERE artifact_id = ? AND version = ? AND tenant = ? AND revoked_at IS NULL",
                (artifact_id, version, effective_tenant),
            ).fetchone()
            if existing is not None:
                return self._publication_from_row(existing)

            token = secrets.token_urlsafe(32)
            publication = Publication(
                # The version's own digest, not a second computation: a publication disagreeing with
                # its artifact would be the more alarming answer. Read from the row in hand rather
                # than via ``content_digest``, which would open and write through a second
                # connection while this one holds the publication.
                content_sha256=row["content_sha256"] or self.blob_digest(artifact_id, version),
                token=token,
                id=publication_id(token),
                artifact_id=artifact_id,
                version=version,
                tenant=effective_tenant,
                title=title,
                published_at=time.time(),
                published_by=published_by,
                authors=_clean_entries(authors, ("name", "orcid", "affiliation")),
                external_ids=_clean_entries(external_ids, ("scheme", "value")),
            )
            conn.execute(
                "INSERT INTO artifact_publications "
                "(token, artifact_id, version, tenant, title, published_at, "
                "published_by, content_sha256, authors, external_ids) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    publication.id,
                    publication.artifact_id,
                    publication.version,
                    publication.tenant,
                    publication.title,
                    publication.published_at,
                    publication.published_by,
                    publication.content_sha256,
                    _json_or_none(publication.authors),
                    _json_or_none(publication.external_ids),
                ),
            )
            # In the registry audit rather than a table of its own, so one
            # sequence orders publications and registry moves together and a
            # follower needs one cursor (``read_events``).
            self._audit_in_connection(
                conn,
                action="publish",
                artifact_id=artifact_id,
                to_version=version,
                key="token",
                value=publication.id,
                actor=published_by,
                tenant=effective_tenant,
            )
            conn.commit()
            return publication
        finally:
            conn.close()

    def get_publication(self, token: str) -> Publication | None:
        """Look up a publication by its raw token, revoked ones included.

        So the caller can answer "withdrawn" rather than "no such page". Never by
        id: this is what the public routes call, and an id is not a credential.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM artifact_publications WHERE token = ?", (publication_id(token),)
            ).fetchone()
            return self._publication_from_row(row, token) if row is not None else None
        finally:
            conn.close()

    def update_publication_credits(
        self,
        token: str,
        *,
        tenant: str | None = None,
        authors: list[dict[str, str]] | None = None,
        external_ids: list[dict[str, str]] | None = None,
        actor: str | None = None,
    ) -> Publication | None:
        """Set a publication's authors and external ids after the fact (audited as ``credit``).

        Never touches the artifact binding. ``None`` leaves a column alone; an empty
        list clears it. ``token`` is the raw token or the publication's id. Returns the
        updated publication, or ``None`` if it is unknown in this tenant.
        """
        effective_tenant = tenant if tenant is not None else ""
        key = publication_key(token)
        raw = "" if key == token else token
        assignments: list[str] = []
        params: list[Any] = []
        if authors is not None:
            assignments.append("authors = ?")
            params.append(_json_or_none(_clean_entries(authors, ("name", "orcid", "affiliation"))))
        if external_ids is not None:
            assignments.append("external_ids = ?")
            params.append(_json_or_none(_clean_entries(external_ids, ("scheme", "value"))))

        conn = self._get_connection()
        try:
            if assignments:
                cursor = conn.execute(
                    f"UPDATE artifact_publications SET {', '.join(assignments)} "
                    "WHERE token = ? AND tenant = ?",
                    (*params, key, effective_tenant),
                )
                if cursor.rowcount == 0:
                    conn.commit()
                    return None
                bound = conn.execute(
                    "SELECT artifact_id, version FROM artifact_publications "
                    "WHERE token = ? AND tenant = ?",
                    (key, effective_tenant),
                ).fetchone()
                self._audit_in_connection(
                    conn,
                    action="credit",
                    artifact_id=bound["artifact_id"],
                    to_version=bound["version"],
                    key="token",
                    value=key,
                    actor=actor,
                    tenant=effective_tenant,
                )
                conn.commit()
            row = conn.execute(
                "SELECT * FROM artifact_publications WHERE token = ? AND tenant = ?",
                (key, effective_tenant),
            ).fetchone()
            return self._publication_from_row(row, raw) if row is not None else None
        finally:
            conn.close()

    def revoke_publication(
        self, token: str, tenant: str | None = None, actor: str | None = None
    ) -> bool:
        """Withdraw a grant (audited), named by raw token or id.

        Returns False if it was unknown or already gone.
        """
        effective_tenant = tenant if tenant is not None else ""
        key = publication_key(token)
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT artifact_id, version FROM artifact_publications "
                "WHERE token = ? AND tenant = ?",
                (key, effective_tenant),
            ).fetchone()
            cursor = conn.execute(
                "UPDATE artifact_publications SET revoked_at = ? "
                "WHERE token = ? AND tenant = ? AND revoked_at IS NULL",
                (time.time(), key, effective_tenant),
            )
            # Only the call that actually withdrew it records it: a second
            # revoke of the same token changes nothing and is not an event.
            if cursor.rowcount == 0 or row is None:
                conn.commit()
                return False
            self._audit_in_connection(
                conn,
                action="withdraw",
                artifact_id=row["artifact_id"],
                to_version=row["version"],
                key="token",
                value=key,
                actor=actor,
                tenant=effective_tenant,
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def list_publications(
        self,
        tenant: str | None = None,
        include_revoked: bool = False,
    ) -> list[Publication]:
        """Every grant in a tenant, newest first."""
        effective_tenant = tenant if tenant is not None else ""
        sql = "SELECT * FROM artifact_publications WHERE tenant = ?"
        if not include_revoked:
            sql += " AND revoked_at IS NULL"
        sql += " ORDER BY published_at DESC"
        conn = self._get_connection()
        try:
            rows = conn.execute(sql, (effective_tenant,)).fetchall()
            return [self._publication_from_row(row) for row in rows]
        finally:
            conn.close()

    def set_alias(
        self,
        name: str,
        alias: str,
        artifact_id: str,
        version: int,
        tenant: str | None = None,
        actor: str | None = None,
    ) -> bool:
        """Point ``name @ alias`` at an artifact version (audited).

        Returns False, writing nothing, when the alias already points there, so an
        idempotent promote cell does not fill the audit history.

        Raises:
            ValueError: If the artifact doesn't exist, isn't readable, or belongs to
                a different tenant.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT state, tenant FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            )
            row = cursor.fetchone()
            self._require_writable_target(row, artifact_id, version, tenant)
            if row["state"] not in ("ready", "superseded"):
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} is not readable (state={row['state']})"
                )

            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                "SELECT artifact_id, version FROM artifact_aliases "
                "WHERE name = ? AND alias = ? AND tenant = ?",
                (name, alias, effective_tenant),
            )
            previous = cursor.fetchone()
            if (
                previous is not None
                and previous["artifact_id"] == artifact_id
                and previous["version"] == version
            ):
                return False  # Already points here: idempotent no-op
            self._audit_in_connection(
                conn,
                action="alias_set",
                name=name,
                alias=alias,
                artifact_id=artifact_id,
                from_artifact_id=previous["artifact_id"] if previous else None,
                from_version=previous["version"] if previous else None,
                to_version=version,
                actor=actor,
                tenant=tenant,
            )
            conn.execute(
                """
                INSERT INTO artifact_aliases
                    (name, alias, artifact_id, version, updated_at, tenant)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant, name, alias) DO UPDATE SET
                    artifact_id = excluded.artifact_id,
                    version = excluded.version,
                    updated_at = excluded.updated_at
                """,
                (name, alias, artifact_id, version, time.time(), effective_tenant),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def resolve_alias(
        self,
        name: str,
        alias: str,
        tenant: str | None = None,
    ) -> ArtifactVersion | None:
        """Resolve ``name @ alias`` to its artifact version."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                "SELECT artifact_id, version FROM artifact_aliases "
                "WHERE name = ? AND alias = ? AND tenant = ?",
                (name, alias, effective_tenant),
            )
            row = cursor.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self.get_artifact(row["artifact_id"], row["version"])

    def list_aliases(
        self,
        name: str | None = None,
        tenant: str | None = None,
    ) -> list[ArtifactAlias]:
        """List aliases for one name, or the whole store when name is None."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            if name is not None:
                cursor = conn.execute(
                    "SELECT name, alias, artifact_id, version, updated_at, tenant "
                    "FROM artifact_aliases WHERE name = ? AND tenant = ? ORDER BY alias",
                    (name, effective_tenant),
                )
            else:
                cursor = conn.execute(
                    "SELECT name, alias, artifact_id, version, updated_at, tenant "
                    "FROM artifact_aliases WHERE tenant = ? ORDER BY name, alias",
                    (effective_tenant,),
                )
            return [
                ArtifactAlias(
                    name=row["name"],
                    alias=row["alias"],
                    artifact_id=row["artifact_id"],
                    version=row["version"],
                    updated_at=row["updated_at"],
                    tenant=row["tenant"] if row["tenant"] else None,
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def list_all_aliases(self) -> list[ArtifactAlias]:
        """List aliases across ALL tenants (maintenance/CLI use)."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT name, alias, artifact_id, version, updated_at, tenant "
                "FROM artifact_aliases ORDER BY name, alias"
            )
            return [
                ArtifactAlias(
                    name=row["name"],
                    alias=row["alias"],
                    artifact_id=row["artifact_id"],
                    version=row["version"],
                    updated_at=row["updated_at"],
                    tenant=row["tenant"] if row["tenant"] else None,
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def delete_alias(
        self,
        name: str,
        alias: str,
        tenant: str | None = None,
        actor: str | None = None,
    ) -> bool:
        """Delete ``name @ alias`` (audited). Returns True if it existed."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                "SELECT version FROM artifact_aliases WHERE name = ? AND alias = ? AND tenant = ?",
                (name, alias, effective_tenant),
            )
            previous = cursor.fetchone()
            self._serialize_audit(conn)
            cursor = conn.execute(
                "DELETE FROM artifact_aliases WHERE name = ? AND alias = ? AND tenant = ?",
                (name, alias, effective_tenant),
            )
            if cursor.rowcount > 0:
                self._audit_in_connection(
                    conn,
                    action="alias_delete",
                    name=name,
                    alias=alias,
                    from_version=previous["version"] if previous else None,
                    actor=actor,
                    tenant=tenant,
                )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def set_tag(
        self,
        artifact_id: str,
        version: int,
        key: str,
        value: str,
        tenant: str | None = None,
        actor: str | None = None,
    ) -> None:
        """Set a key/value tag on an artifact version (audited).

        Raises:
            ValueError: If the artifact doesn't exist, isn't readable, or belongs to
                a different tenant.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT state, tenant FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            )
            row = cursor.fetchone()
            self._require_writable_target(row, artifact_id, version, tenant)
            if row["state"] not in ("ready", "superseded"):
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} is not readable (state={row['state']})"
                )

            effective_tenant = tenant if tenant is not None else ""
            self._audit_in_connection(
                conn,
                action="tag_set",
                artifact_id=artifact_id,
                to_version=version,
                key=key,
                value=value,
                actor=actor,
                tenant=tenant,
            )
            conn.execute(
                """
                INSERT INTO artifact_tags
                    (artifact_id, version, key, value, updated_at, tenant)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant, artifact_id, version, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (artifact_id, version, key, str(value), time.time(), effective_tenant),
            )
            conn.commit()
        finally:
            conn.close()

    def get_tags(
        self,
        artifact_id: str,
        version: int,
        tenant: str | None = None,
    ) -> dict[str, str]:
        """Return the tags on an artifact version as a dict."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                "SELECT key, value FROM artifact_tags "
                "WHERE artifact_id = ? AND version = ? AND tenant = ? ORDER BY key",
                (artifact_id, version, effective_tenant),
            )
            return {row["key"]: row["value"] for row in cursor.fetchall()}
        finally:
            conn.close()

    def list_artifacts_by_tag(
        self,
        key: str,
        value: str,
        tenant: str | None = None,
    ) -> list[tuple[str, int]]:
        """Return ``(artifact_id, version)`` for every artifact tagged ``key=value``."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                "SELECT artifact_id, version FROM artifact_tags "
                "WHERE key = ? AND value = ? AND tenant = ? ORDER BY artifact_id, version",
                (key, value, effective_tenant),
            )
            return [(row["artifact_id"], row["version"]) for row in cursor.fetchall()]
        finally:
            conn.close()

    def list_artifacts_with_tag_key(
        self,
        key: str,
        tenant: str | None = None,
    ) -> list[tuple[str, int, str]]:
        """Return ``(artifact_id, version, value)`` for every artifact with tag ``key``.

        One query for a whole notebook instead of one per cell, which matters over a
        remote store.
        """
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                "SELECT artifact_id, version, value FROM artifact_tags "
                "WHERE key = ? AND tenant = ? ORDER BY artifact_id, version",
                (key, effective_tenant),
            )
            return [(r["artifact_id"], r["version"], r["value"]) for r in cursor.fetchall()]
        finally:
            conn.close()

    def delete_tag(
        self,
        artifact_id: str,
        version: int,
        key: str,
        tenant: str | None = None,
        actor: str | None = None,
    ) -> bool:
        """Delete one tag (audited). Returns True if it existed."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            self._serialize_audit(conn)
            cursor = conn.execute(
                "DELETE FROM artifact_tags "
                "WHERE artifact_id = ? AND version = ? AND key = ? AND tenant = ?",
                (artifact_id, version, key, effective_tenant),
            )
            if cursor.rowcount > 0:
                self._audit_in_connection(
                    conn,
                    action="tag_delete",
                    artifact_id=artifact_id,
                    from_version=version,
                    key=key,
                    actor=actor,
                    tenant=tenant,
                )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def read_audit(
        self,
        name: str | None = None,
        artifact_id: str | None = None,
        limit: int = 100,
        tenant: object = _AUDIT_ALL_TENANTS,
    ) -> list[dict]:
        """Read registry audit entries, newest first; filters are ANDed.

        ``tenant`` defaults to ``ALL_TENANTS`` for the CLI and admin views.
        Request-serving routes must pass the caller's tenant (``None`` for the
        default tenant).
        """
        conn = self._get_connection()
        try:
            query = (
                "SELECT seq, at, actor, action, name, alias, artifact_id, "
                "from_artifact_id, from_version, to_version, key, value, tenant "
                "FROM registry_audit"
            )
            clauses: list[str] = []
            params: list = []
            if name is not None:
                clauses.append("name = ?")
                params.append(name)
            if artifact_id is not None:
                clauses.append("artifact_id = ?")
                params.append(artifact_id)
            if tenant is not _AUDIT_ALL_TENANTS:
                clauses.append("tenant = ?")
                params.append(tenant if tenant is not None else "")
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY seq DESC LIMIT ?"
            params.append(limit)
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def read_events(
        self,
        since: int = 0,
        limit: int = 100,
        tenant: object = _AUDIT_ALL_TENANTS,
    ) -> list[dict]:
        """Audit entries after ``since``, oldest first, for a follower.

        A follower pages by passing the last ``seq`` it saw; ``tenant`` scopes as in
        :meth:`read_audit`.
        """
        conn = self._get_connection()
        try:
            query = (
                "SELECT seq, at, actor, action, name, alias, artifact_id, "
                "from_artifact_id, from_version, to_version, key, value, tenant "
                "FROM registry_audit WHERE seq > ?"
            )
            params: list = [since]
            if tenant is not _AUDIT_ALL_TENANTS:
                query += " AND tenant = ?"
                params.append(tenant if tenant is not None else "")
            query += " ORDER BY seq ASC LIMIT ?"
            params.append(limit)
            return [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def request_alias_change(
        self,
        name: str,
        alias: str,
        action: str,
        artifact_id: str | None = None,
        version: int | None = None,
        tenant: str | None = None,
        actor: str | None = None,
    ) -> bool:
        """Queue a protected-alias change for approval (audited).

        ``action`` is ``"set"`` (needs artifact_id and version, validated like
        :meth:`set_alias`) or ``"delete"``. A new request replaces the alias's pending
        one. Returns False when the alias already points at the requested target.
        """
        if action not in ("set", "delete"):
            raise ValueError(f"Unknown alias change action: {action!r}")
        conn = self._get_connection()
        try:
            if action == "set":
                if artifact_id is None or version is None:
                    raise ValueError("alias 'set' request requires artifact_id and version")
                cursor = conn.execute(
                    "SELECT state, tenant FROM artifact_versions WHERE id = ? AND version = ?",
                    (artifact_id, version),
                )
                row = cursor.fetchone()
                self._require_writable_target(row, artifact_id, version, tenant)
                if row["state"] not in ("ready", "superseded"):
                    raise ValueError(
                        f"Artifact {artifact_id}@v={version} is not readable (state={row['state']})"
                    )

            effective_tenant = tenant if tenant is not None else ""
            if action == "set":
                cursor = conn.execute(
                    "SELECT artifact_id, version FROM artifact_aliases "
                    "WHERE name = ? AND alias = ? AND tenant = ?",
                    (name, alias, effective_tenant),
                )
                current = cursor.fetchone()
                if (
                    current is not None
                    and current["artifact_id"] == artifact_id
                    and current["version"] == version
                ):
                    return False  # Already the live pointer: nothing to approve
            self._audit_in_connection(
                conn,
                action=f"alias_request_{action}",
                name=name,
                alias=alias,
                artifact_id=artifact_id,
                to_version=version,
                actor=actor,
                tenant=tenant,
            )
            conn.execute(
                """
                INSERT INTO registry_pending
                    (name, alias, action, artifact_id, version,
                     requested_by, requested_at, tenant)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant, name, alias) DO UPDATE SET
                    action = excluded.action,
                    artifact_id = excluded.artifact_id,
                    version = excluded.version,
                    requested_by = excluded.requested_by,
                    requested_at = excluded.requested_at
                """,
                (name, alias, action, artifact_id, version, actor, time.time(), effective_tenant),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def list_pending_changes(self, tenant: str | None = None) -> list[dict]:
        """List alias changes awaiting approval (oldest first)."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                "SELECT name, alias, action, artifact_id, version, "
                "requested_by, requested_at, tenant FROM registry_pending "
                "WHERE tenant = ? ORDER BY requested_at",
                (effective_tenant,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def approve_alias_change(
        self,
        name: str,
        alias: str,
        tenant: str | None = None,
        actor: str | None = None,
        require_distinct_approver: bool = False,
    ) -> dict:
        """Apply a pending alias change (audited with the approver as actor) and return it.

        ``require_distinct_approver`` forbids the requester approving their own change.

        Raises:
            ValueError: If no change is pending for ``name @ alias``, the target is no
                longer available, or the approver is the requester under
                ``require_distinct_approver``.
        """
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            # Lock before reading, or a reject or newer request committed in between is
            # consumed while the stale change read here is applied.
            self._dialect.begin_write(conn, "registry_audit")
            cursor = conn.execute(
                "SELECT name, alias, action, artifact_id, version, requested_by "
                "FROM registry_pending WHERE name = ? AND alias = ? AND tenant = ?",
                (name, alias, effective_tenant),
            )
            pending = cursor.fetchone()
            if pending is None:
                raise ValueError(f"No pending change for alias '{name}@{alias}'")
            entry = dict(pending)

            if require_distinct_approver and actor is not None and entry["requested_by"] == actor:
                conn.rollback()
                raise ValueError(
                    "Separation of duty: the approver must differ from the "
                    f"requester ('{actor}' both requested and approved "
                    f"'{name}@{alias}')"
                )

            # Everything below shares ONE transaction: pending-delete, the
            # approval audit, the alias move itself, and its move audit. A
            # crash can't consume the approval without applying the move,
            # and a failed validation leaves the pending entry intact for
            # an explicit reject.
            if entry["action"] == "set":
                cursor = conn.execute(
                    "SELECT state FROM artifact_versions WHERE id = ? AND version = ?",
                    (entry["artifact_id"], entry["version"]),
                )
                target = cursor.fetchone()
                if target is None or target["state"] not in ("ready", "superseded"):
                    conn.rollback()
                    state = target["state"] if target else "deleted"
                    raise ValueError(
                        f"Pending target {entry['artifact_id']}@v={entry['version']} is "
                        f"no longer available (state={state}); reject the pending "
                        "change or submit a new request"
                    )

            conn.execute(
                "DELETE FROM registry_pending WHERE name = ? AND alias = ? AND tenant = ?",
                (name, alias, effective_tenant),
            )
            self._audit_in_connection(
                conn,
                action="alias_approved",
                name=name,
                alias=alias,
                artifact_id=entry["artifact_id"],
                to_version=entry["version"],
                actor=actor,
                tenant=tenant,
            )

            cursor = conn.execute(
                "SELECT artifact_id, version FROM artifact_aliases "
                "WHERE name = ? AND alias = ? AND tenant = ?",
                (name, alias, effective_tenant),
            )
            previous = cursor.fetchone()
            if entry["action"] == "set":
                self._audit_in_connection(
                    conn,
                    action="alias_set",
                    name=name,
                    alias=alias,
                    artifact_id=entry["artifact_id"],
                    from_artifact_id=previous["artifact_id"] if previous else None,
                    from_version=previous["version"] if previous else None,
                    to_version=entry["version"],
                    actor=actor,
                    tenant=tenant,
                )
                conn.execute(
                    """
                    INSERT INTO artifact_aliases
                        (name, alias, artifact_id, version, updated_at, tenant)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tenant, name, alias) DO UPDATE SET
                        artifact_id = excluded.artifact_id,
                        version = excluded.version,
                        updated_at = excluded.updated_at
                    """,
                    (
                        name,
                        alias,
                        entry["artifact_id"],
                        entry["version"],
                        time.time(),
                        effective_tenant,
                    ),
                )
            else:
                if previous is not None:
                    self._audit_in_connection(
                        conn,
                        action="alias_delete",
                        name=name,
                        alias=alias,
                        from_artifact_id=previous["artifact_id"],
                        from_version=previous["version"],
                        actor=actor,
                        tenant=tenant,
                    )
                conn.execute(
                    "DELETE FROM artifact_aliases WHERE name = ? AND alias = ? AND tenant = ?",
                    (name, alias, effective_tenant),
                )

            conn.commit()
            return entry
        finally:
            conn.close()

    def reject_alias_change(
        self,
        name: str,
        alias: str,
        tenant: str | None = None,
        actor: str | None = None,
    ) -> dict:
        """Discard a pending alias change (audited).

        Raises:
            ValueError: If no change is pending for ``name @ alias``.
        """
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            # Lock before reading, so an approve cannot apply the change rejected here.
            self._dialect.begin_write(conn, "registry_audit")
            cursor = conn.execute(
                "SELECT name, alias, action, artifact_id, version, requested_by "
                "FROM registry_pending WHERE name = ? AND alias = ? AND tenant = ?",
                (name, alias, effective_tenant),
            )
            pending = cursor.fetchone()
            if pending is None:
                raise ValueError(f"No pending change for alias '{name}@{alias}'")
            conn.execute(
                "DELETE FROM registry_pending WHERE name = ? AND alias = ? AND tenant = ?",
                (name, alias, effective_tenant),
            )
            self._audit_in_connection(
                conn,
                action="alias_rejected",
                name=name,
                alias=alias,
                artifact_id=pending["artifact_id"],
                to_version=pending["version"],
                actor=actor,
                tenant=tenant,
            )
            conn.commit()
            return dict(pending)
        finally:
            conn.close()

    def list_names(self, tenant: str | None = None) -> list[ArtifactName]:
        """List name pointers, filtered by tenant when given."""
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                """
                SELECT name, artifact_id, version, updated_at, tenant
                FROM artifact_names
                WHERE tenant = ?
                ORDER BY name
                """,
                (effective_tenant,),
            )
            return [
                ArtifactName(
                    name=row["name"],
                    artifact_id=row["artifact_id"],
                    version=row["version"],
                    updated_at=row["updated_at"],
                    tenant=row["tenant"] if row["tenant"] else None,
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_name_status(self, name: str, tenant: str | None = None) -> NameStatus | None:
        """Get a named artifact's metadata and recorded ``input_versions``, or None.

        The caller compares ``input_versions`` against current versions for staleness.
        """
        name_info = self.get_name(name, tenant=tenant)
        if name_info is None:
            return None

        artifact = self.get_artifact(name_info.artifact_id, name_info.version)
        if artifact is None:
            return None

        input_versions: dict[str, str] = {}
        if artifact.input_versions:
            input_versions = json.loads(artifact.input_versions)

        return NameStatus(
            name=name,
            artifact_uri=f"strata://artifact/{artifact.id}@v={artifact.version}",
            artifact_id=artifact.id,
            version=artifact.version,
            state=artifact.state,
            updated_at=name_info.updated_at,
            input_versions=input_versions,
        )

    # --- Lineage and dependency queries ---

    def ancestor_transform_specs(
        self, artifact: ArtifactVersion, *, max_depth: int
    ) -> list[str] | None:
        """The stored ``transform_spec`` of every version *artifact* was computed from.

        Follows the recorded input edges (``_ancestor_of``) one query per level, across
        tenants. Versions no longer in the store are skipped. ``None`` when ancestors
        remain past ``max_depth`` levels.
        """

        def edges(input_versions: str | None) -> list[tuple[str, int]]:
            if not input_versions:
                return []
            return [
                ancestor
                for uri, recorded in json.loads(input_versions).items()
                if (ancestor := _ancestor_of(uri, recorded)) is not None
            ]

        seen = {(artifact.id, artifact.version)}
        level = edges(artifact.input_versions)
        specs: list[str] = []
        if not level:
            return specs
        conn = self._get_connection()
        try:
            for _ in range(max_depth):
                pending = [node for node in dict.fromkeys(level) if node not in seen]
                if not pending:
                    return specs
                seen.update(pending)
                level = []
                for start in range(0, len(pending), 250):
                    batch = pending[start : start + 250]
                    where = " OR ".join("(id = ? AND version = ?)" for _ in batch)
                    for row in conn.execute(
                        "SELECT transform_spec, input_versions FROM artifact_versions "
                        f"WHERE {where}",
                        [value for node in batch for value in node],
                    ).fetchall():
                        if row["transform_spec"]:
                            specs.append(row["transform_spec"])
                        level.extend(edges(row["input_versions"]))
            if any(node not in seen for node in level):
                return None
            return specs
        finally:
            conn.close()

    def find_dependents(
        self,
        artifact_id: str,
        version: int,
        tenant: str | None = None,
    ) -> list[tuple[ArtifactVersion, str]]:
        """Find artifacts whose ``input_versions`` reference this artifact version.

        Returns ``(ArtifactVersion, input_version_string)`` pairs.
        """
        # Inputs are recorded as "artifact_id@v=N" or as the full URI.
        search_pattern = _like_literal(f'"{artifact_id}@v={version}"')
        uri_pattern = _like_literal(f'"strata://artifact/{artifact_id}@v={version}"')

        conn = self._get_connection()
        try:
            if tenant is not None:
                cursor = conn.execute(
                    """
                    SELECT id, version, state, provenance_hash, schema_json,
                           row_count, byte_size, created_at, transform_spec,
                           input_versions, tenant, principal, content_sha256
                    FROM artifact_versions
                    WHERE state = 'ready'
                      AND tenant = ?
                      AND (input_versions LIKE ? ESCAPE '\\' OR input_versions LIKE ? ESCAPE '\\')
                    ORDER BY created_at DESC
                    """,
                    (tenant, f"%{search_pattern}%", f"%{uri_pattern}%"),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, version, state, provenance_hash, schema_json,
                           row_count, byte_size, created_at, transform_spec,
                           input_versions, tenant, principal, content_sha256
                    FROM artifact_versions
                    WHERE state = 'ready'
                      AND (input_versions LIKE ? ESCAPE '\\' OR input_versions LIKE ? ESCAPE '\\')
                    ORDER BY created_at DESC
                    """,
                    (f"%{search_pattern}%", f"%{uri_pattern}%"),
                )

            results = []
            for row in cursor.fetchall():
                artifact = ArtifactVersion(
                    id=row["id"],
                    version=row["version"],
                    state=row["state"],
                    provenance_hash=row["provenance_hash"],
                    schema_json=row["schema_json"],
                    row_count=row["row_count"],
                    byte_size=row["byte_size"],
                    created_at=row["created_at"],
                    transform_spec=row["transform_spec"],
                    input_versions=row["input_versions"],
                    tenant=row["tenant"],
                    principal=row["principal"],
                    content_sha256=row["content_sha256"],
                )

                input_version_used = f"{artifact_id}@v={version}"
                if artifact.input_versions:
                    try:
                        input_vers = json.loads(artifact.input_versions)
                        exact_uri = f"strata://artifact/{artifact_id}@v={version}"
                        # Match the exact dependency entry instead of substring prefixes.
                        for uri, ver in input_vers.items():
                            if uri == exact_uri or ver == input_version_used or ver == exact_uri:
                                input_version_used = ver
                                break
                    except json.JSONDecodeError:
                        pass

                results.append((artifact, input_version_used))

            return results
        finally:
            conn.close()

    def list_name_reads(self, tenant: str | None = None) -> list[tuple[str | None, str, str]]:
        """Return ``(tenant, artifact_id, reference)`` for each ready artifact that read a name.

        ``reference`` is what follows ``strata://name/`` in its inputs, e.g.
        ``taxi/model@champion``. The returned tenant is ``None`` for the default tenant.
        ``tenant`` of ``None`` reads every tenant.
        """
        prefix = "strata://name/"
        query = (
            "SELECT id, tenant, input_versions FROM artifact_versions "
            "WHERE state = 'ready' AND input_versions LIKE ?"
        )
        params: tuple = (f"%{prefix}%",)
        if tenant is not None:
            query += " AND tenant = ?"
            params += (tenant,)
        conn = self._get_connection()
        try:
            rows = conn.execute(query + " ORDER BY id", params).fetchall()
        finally:
            conn.close()
        reads: list[tuple[str | None, str, str]] = []
        for row in rows:
            for uri in json.loads(row["input_versions"]):
                if uri.startswith(prefix):
                    reads.append((row["tenant"] or None, row["id"], uri[len(prefix) :]))
        return reads

    def get_name_for_artifact(
        self,
        artifact_id: str,
        version: int,
        tenant: str | None = None,
    ) -> str | None:
        """Get a name pointing at this artifact version, or None."""
        # '' not NULL for personal mode: SQLite NULL != NULL in unique constraints.
        effective_tenant = tenant if tenant is not None else ""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT name FROM artifact_names
                WHERE artifact_id = ? AND version = ? AND tenant = ?
                """,
                (artifact_id, version, effective_tenant),
            )
            row = cursor.fetchone()
            return row["name"] if row else None
        finally:
            conn.close()

    # --- Lifecycle management ---

    # ORDER BY can't be parameterized, so only whitelisted columns keep it injection-safe.
    _SORT_COLUMNS = {"created_at": "created_at", "byte_size": "byte_size", "row_count": "row_count"}

    def list_artifacts(
        self,
        limit: int = 100,
        offset: int = 0,
        state: str | None = None,
        name_prefix: str | None = None,
        tenant: str | None = None,
        since: float | None = None,
        sort: str = "created_at",
        order: str = "desc",
    ) -> list[ArtifactVersion]:
        """List artifacts with optional filtering, paging and sorting.

        ``tenant`` also includes legacy tenantless artifacts. ``name_prefix`` filters
        by name. ``since`` is an epoch timestamp. ``sort`` is ``created_at``,
        ``byte_size`` or ``row_count`` (anything else means ``created_at``).
        """
        sort_col = self._SORT_COLUMNS.get(sort, "created_at")
        order_sql = "ASC" if str(order).lower() == "asc" else "DESC"
        conn = self._get_connection()
        try:
            if name_prefix is not None:
                query = """
                    SELECT DISTINCT av.id, av.version, av.state, av.provenance_hash,
                           av.schema_json, av.row_count, av.byte_size, av.created_at,
                           av.transform_spec, av.input_versions, av.tenant, av.principal,
                           av.content_sha256
                    FROM artifact_versions av
                    INNER JOIN artifact_names an
                        ON av.id = an.artifact_id AND av.version = an.version
                    WHERE an.name LIKE ? ESCAPE '\\'
                """
                params: list = [_like_literal(name_prefix) + "%"]

                if tenant is not None:
                    query += " AND (av.tenant = ? OR av.tenant = '' OR av.tenant IS NULL)"
                    params.append(tenant)
                    query += " AND (an.tenant = ? OR an.tenant = '')"
                    params.append(tenant)

                if state is not None:
                    query += " AND av.state = ?"
                    params.append(state)

                if since is not None:
                    query += " AND av.created_at >= ?"
                    params.append(since)

                query += f" ORDER BY av.{sort_col} {order_sql} LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            else:
                query = """
                    SELECT id, version, state, provenance_hash, schema_json,
                           row_count, byte_size, created_at, transform_spec,
                           input_versions, tenant, principal, content_sha256
                    FROM artifact_versions
                """
                params = []

                conditions: list[str] = []
                if tenant is not None:
                    conditions.append("(tenant = ? OR tenant = '' OR tenant IS NULL)")
                    params.append(tenant)
                if state is not None:
                    conditions.append("state = ?")
                    params.append(state)
                if since is not None:
                    conditions.append("created_at >= ?")
                    params.append(since)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += f" ORDER BY {sort_col} {order_sql} LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor = conn.execute(query, params)
            return [
                ArtifactVersion(
                    id=row["id"],
                    version=row["version"],
                    state=row["state"],
                    provenance_hash=row["provenance_hash"],
                    schema_json=row["schema_json"],
                    row_count=row["row_count"],
                    byte_size=row["byte_size"],
                    created_at=row["created_at"],
                    transform_spec=row["transform_spec"],
                    input_versions=row["input_versions"],
                    tenant=row["tenant"],
                    principal=row["principal"],
                    content_sha256=row["content_sha256"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def delete_artifact(self, artifact_id: str, version: int, tenant: str | None = None) -> bool:
        """Delete an artifact version, its blob, and every name, alias and tag pointing at it.

        With ``tenant`` the artifact must belong to that tenant or be tenantless.
        Returns False if it does not exist or belongs to another tenant.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT tenant FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            )
            row = cursor.fetchone()
            if row is None:
                return False
            if tenant is not None:
                artifact_tenant = row["tenant"] if row["tenant"] else None
                if not self._can_assign_name_for_tenant(artifact_tenant, tenant):
                    return False
            # A published version is cited by a link somebody else holds, and
            # version numbers are reused (``MAX(version) + 1``), so deleting
            # one and rebuilding the cell would leave that link serving
            # different bytes under the same title and authors. Withdraw the
            # publication first, deliberately, and then it can go.
            published = conn.execute(
                "SELECT token FROM artifact_publications "
                "WHERE artifact_id = ? AND version = ? AND revoked_at IS NULL",
                (artifact_id, version),
            ).fetchone()
            if published is not None:
                raise ValueError(
                    f"{artifact_id}@v={version} is published (publication id "
                    f"{published['token']}); revoke the publication before deleting it"
                )
            # A published or pinned version reading these bytes through superseded_by gets
            # its own copy first: deleting this version must not empty a link somebody holds.
            ref = f"{artifact_id}@v={version}"
            held_pointers = conn.execute(
                """
                SELECT id, version FROM artifact_versions av
                WHERE av.superseded_by = ?
                  AND (EXISTS (SELECT 1 FROM artifact_publications pub
                               WHERE pub.artifact_id = av.id AND pub.version = av.version
                                 AND pub.revoked_at IS NULL)
                       OR EXISTS (SELECT 1 FROM artifact_pins p
                                  WHERE p.artifact_id = av.id AND p.version = av.version))
                """,
                (ref,),
            ).fetchall()
            for pointer in held_pointers:
                self._reclaim_blob(pointer["id"], pointer["version"])
            # Before the first write, so the lock is never awaited while holding a name row.
            self._serialize_audit(conn)
            for pointer in held_pointers:
                conn.execute(
                    "UPDATE artifact_versions SET superseded_by = NULL "
                    "WHERE id = ? AND version = ? AND superseded_by = ?",
                    (pointer["id"], pointer["version"], ref),
                )
            # Past this point the delete is committed to; the blob cleanup
            # below the finally runs only for rows that actually existed.

            conn.execute(
                "DELETE FROM artifact_names WHERE artifact_id = ? AND version = ?",
                (artifact_id, version),
            )

            # Alias deletions are audited, so an alias vanishing with its target stays
            # reconstructible.
            cursor = conn.execute(
                "SELECT name, alias, tenant FROM artifact_aliases "
                "WHERE artifact_id = ? AND version = ?",
                (artifact_id, version),
            )
            for row in cursor.fetchall():
                self._audit_in_connection(
                    conn,
                    action="alias_delete",
                    name=row["name"],
                    alias=row["alias"],
                    artifact_id=artifact_id,
                    from_version=version,
                    tenant=row["tenant"] if row["tenant"] else None,
                )
            conn.execute(
                "DELETE FROM artifact_aliases WHERE artifact_id = ? AND version = ?",
                (artifact_id, version),
            )
            # Tags and builds, through the same helper garbage_collect uses so
            # the two paths cannot drift on what a version has to shed first.
            self._delete_version_children(conn, artifact_id, version)

            # Which key the bytes are under is on the row about to go.
            attempt_row = conn.execute(
                "SELECT blob_attempt FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            ).fetchone()
            attempt = attempt_row["blob_attempt"] if attempt_row is not None else None
            blob_id = attempt_blob_id(artifact_id, attempt) if attempt else artifact_id

            conn.execute(
                "DELETE FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            )
            conn.commit()
        finally:
            conn.close()

        # Blob deletion is network I/O against S3 / GCS / Azure, and it runs
        # after the metadata is durably gone, so it needs no connection.
        # Holding a pooled one across it parks a slot for the round trip.
        self.blob_store.delete_blob(blob_id, version)
        return True

    def _delete_version_children(
        self, conn: StoreConnection, artifact_id: str, version: int
    ) -> None:
        """Remove the rows that reference a version, before the version itself.

        Tags have no foreign key; build rows cascade because a build of a deleted
        artifact describes nothing. ``artifact_builds`` belongs to the build store and
        may not exist.
        """
        conn.execute(
            "DELETE FROM artifact_tags WHERE artifact_id = ? AND version = ?",
            (artifact_id, version),
        )
        if self._dialect.schema_exists(conn, "artifact_builds"):
            conn.execute(
                "DELETE FROM artifact_builds WHERE artifact_id = ? AND version = ?",
                (artifact_id, version),
            )

    def _protected_reachable(
        self, conn: StoreConnection, *, current_values: bool = True
    ) -> set[tuple[str, int]]:
        """Every version something holds, with its whole chain of inputs.

        Roots: names and aliases; pending alias changes; publications (revoked ones
        too) and pins; running builds; and with ``current_values`` the current value
        of every caller-named id. Deliberately not tenant-scoped: over-protecting is
        free, and proving chains never cross tenants is not.
        """
        roots_sql = (
            "SELECT artifact_id, version FROM artifact_publications "
            "UNION SELECT artifact_id, version FROM artifact_pins "
            "UNION SELECT artifact_id, version FROM artifact_names "
            "UNION SELECT artifact_id, version FROM artifact_aliases "
            "UNION SELECT artifact_id, version FROM registry_pending "
            "WHERE action = 'set' AND artifact_id IS NOT NULL "
            "UNION SELECT id AS artifact_id, version FROM artifact_versions "
            "WHERE state = 'building'"
        )
        if current_values:
            roots_sql += (
                " UNION SELECT id AS artifact_id, MAX(version) AS version "
                "FROM artifact_versions WHERE minted = 0 "
                "AND state IN ('ready', 'superseded') GROUP BY id"
            )
        roots = conn.execute(roots_sql).fetchall()
        # One read of every recorded edge, walked in memory: a notebook store
        # has a root per cell output, and a query per node would be the sweep.
        inputs: dict[tuple[str, int], str] = {
            (row["id"], row["version"]): row["input_versions"]
            for row in conn.execute(
                "SELECT id, version, input_versions FROM artifact_versions "
                "WHERE input_versions IS NOT NULL"
            ).fetchall()
        }
        # A version a duplicate build overtook reads its canonical's bytes, so whatever
        # holds it holds those.
        canonical: dict[tuple[str, int], tuple[str, int]] = {
            (row["id"], row["version"]): _split_ref(row["superseded_by"])
            for row in conn.execute(
                "SELECT id, version, superseded_by FROM artifact_versions "
                "WHERE superseded_by IS NOT NULL"
            ).fetchall()
        }

        reachable: set[tuple[str, int]] = set()
        pending = [(row["artifact_id"], row["version"]) for row in roots]
        while pending:
            node = pending.pop()
            if node in reachable:
                continue
            reachable.add(node)
            if node in canonical:
                pending.append(canonical[node])
            recorded_inputs = inputs.get(node)
            if not recorded_inputs:
                continue
            for uri, recorded in json.loads(recorded_inputs).items():
                # Parsed as the lineage walk parses it, so that what GC protects
                # and what a publication's chain will follow cannot drift apart:
                # a "strata://name/" edge records the concrete version it read
                # in its value, and that ancestor is as reachable as any other.
                # Anything else is a table or an external leaf.
                ancestor = _ancestor_of(uri, recorded)
                if ancestor is not None:
                    pending.append(ancestor)
        return reachable

    def pin_artifact(
        self,
        artifact_id: str,
        version: int,
        reason: str,
        *,
        tenant: str | None = None,
        pinned_by: str | None = None,
    ) -> dict:
        """Hold a version and its chain against garbage collection.

        Idempotent per reason: pinning again refreshes who and when.

        Raises:
            ValueError: If the version does not exist or belongs to another tenant,
                or the reason is empty.
        """
        reason = reason.strip()
        if not reason:
            raise ValueError("a pin needs a reason")
        effective_tenant = tenant if tenant is not None else ""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT tenant FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            ).fetchone()
            if row is None:
                raise ValueError(f"Artifact {artifact_id}@v={version} not found")
            if tenant is not None and (row["tenant"] or "") not in (effective_tenant, ""):
                raise ValueError(f"Artifact {artifact_id}@v={version} not found")
            pinned_at = time.time()
            conn.execute(
                "INSERT INTO artifact_pins "
                "(artifact_id, version, reason, tenant, pinned_by, pinned_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (artifact_id, version, reason) DO UPDATE SET "
                "pinned_by = excluded.pinned_by, pinned_at = excluded.pinned_at",
                (artifact_id, version, reason, effective_tenant, pinned_by, pinned_at),
            )
            conn.commit()
            return {
                "artifact_id": artifact_id,
                "version": version,
                "reason": reason,
                "tenant": effective_tenant,
                "pinned_by": pinned_by,
                "pinned_at": pinned_at,
            }
        finally:
            conn.close()

    def unpin_artifact(
        self, artifact_id: str, version: int, reason: str, *, tenant: str | None = None
    ) -> bool:
        """Release one hold. Returns False if there was no such pin."""
        sql = "DELETE FROM artifact_pins WHERE artifact_id = ? AND version = ? AND reason = ?"
        params: list[Any] = [artifact_id, version, reason]
        if tenant is not None:
            sql += " AND tenant = ?"
            params.append(tenant)
        conn = self._get_connection()
        try:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_pins(
        self,
        artifact_id: str | None = None,
        version: int | None = None,
        *,
        tenant: str | None = None,
    ) -> list[dict]:
        """Pins, oldest first, optionally for one version and one tenant."""
        clauses: list[str] = []
        params: list[Any] = []
        if artifact_id is not None:
            clauses.append("artifact_id = ?")
            params.append(artifact_id)
        if version is not None:
            clauses.append("version = ?")
            params.append(version)
        if tenant is not None:
            clauses.append("tenant = ?")
            params.append(tenant)
        sql = "SELECT artifact_id, version, reason, tenant, pinned_by, pinned_at FROM artifact_pins"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY pinned_at ASC"
        conn = self._get_connection()
        try:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    def notebook_worker_entries(self) -> list[dict[str, Any]] | None:
        """The server-managed notebook worker registry in order, or ``None`` if never set.

        ``None`` means the configured table applies; ``[]`` means every worker was removed.
        """
        conn = self._get_connection()
        try:
            return self._notebook_worker_entries(conn)
        finally:
            conn.close()

    @staticmethod
    def _notebook_worker_entries(conn: StoreConnection) -> list[dict[str, Any]] | None:
        if conn.execute("SELECT 1 FROM notebook_worker_registry WHERE id = 1").fetchone() is None:
            return None
        rows = conn.execute("SELECT spec FROM notebook_workers ORDER BY ordinal").fetchall()
        return [json.loads(row["spec"]) for row in rows]

    def update_notebook_workers(
        self,
        change: Callable[[list[dict[str, Any]] | None], list[dict[str, Any]] | None],
    ) -> list[dict[str, Any]] | None:
        """Rewrite the notebook worker registry as ``change(current)`` returns it, atomically.

        ``change`` gets the current entries (``None`` if never set) and returns the new
        list in order, or ``None`` to leave it alone; anything it raises rolls back. Held
        under a write lock so two nodes editing at once cannot lose either change. Rows
        are keyed by ``name``, and an unchanged one keeps its ``updated_at``.
        """
        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, "__notebook_workers__")
            stored = {
                row["name"]: (row["ordinal"], row["spec"])
                for row in conn.execute("SELECT name, ordinal, spec FROM notebook_workers")
            }
            entries = change(self._notebook_worker_entries(conn))
            if entries is None:
                conn.rollback()
                return None
            now = time.time()
            for name in stored.keys() - {entry["name"] for entry in entries}:
                conn.execute("DELETE FROM notebook_workers WHERE name = ?", (name,))
            for ordinal, entry in enumerate(entries):
                spec = json.dumps(entry, sort_keys=True)
                previous = stored.get(entry["name"])
                if previous is None:
                    conn.execute(
                        "INSERT INTO notebook_workers "
                        "(name, ordinal, spec, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (entry["name"], ordinal, spec, now, now),
                    )
                elif previous != (ordinal, spec):
                    conn.execute(
                        "UPDATE notebook_workers SET ordinal = ?, spec = ?, updated_at = ? "
                        "WHERE name = ?",
                        (ordinal, spec, now, entry["name"]),
                    )
            conn.execute(
                "INSERT INTO notebook_worker_registry (id, updated_at) VALUES (1, ?) "
                "ON CONFLICT (id) DO UPDATE SET updated_at = excluded.updated_at",
                (now,),
            )
            conn.commit()
            return entries
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def garbage_collect(
        self,
        *,
        max_idle_days: float | None = None,
        max_bytes: int | None = None,
        keep_superseded: int | None = None,
        min_idle_seconds: float = 0.0,
        tenant: str | None = None,
        collect_latest: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Collect the versions nothing needs, least recently used first.

        A candidate has no name or alias, is not in any protected chain
        (``_protected_reachable``), is not ``building``, is not the current value of a
        caller-named id (unless ``collect_latest``), and was last used (or created) at
        least ``min_idle_seconds`` ago.

        Collected: candidates idle beyond ``max_idle_days``; when over ``max_bytes``,
        the least recently used down to 80% of it; with ``keep_superseded``, each id's
        candidates past that many earlier ready/superseded versions (failed ones
        never count). With none of the three set nothing is collected. An id never
        loses its highest version while keeping a lower one, so a version number is
        never reused.

        ``tenant`` limits the sweep, and the ``max_bytes`` measure, to that tenant
        and legacy tenantless rows. Returns ``deleted_count``, ``deleted_bytes``,
        ``store_bytes`` before the pass and, with ``dry_run``, the chosen versions
        under ``collected``.
        """
        now = time.time()
        conn = self._get_connection()
        try:
            used = "COALESCE(av.last_used_at, av.created_at)"
            # A version reading its canonical's bytes through superseded_by holds none.
            query = f"""
                SELECT av.id, av.version, av.state, av.blob_attempt, av.superseded_by,
                       CASE WHEN av.superseded_by IS NULL THEN av.byte_size ELSE 0 END AS byte_size,
                       {used} AS used
                FROM artifact_versions av
                LEFT JOIN artifact_names an ON av.id = an.artifact_id AND av.version = an.version
                LEFT JOIN artifact_aliases aa ON av.id = aa.artifact_id AND av.version = aa.version
                WHERE an.name IS NULL
                  AND aa.alias IS NULL
                  AND av.state IN ('ready', 'superseded', 'failed')
                  AND {used} <= ?
            """
            if not collect_latest:
                # The current value of a caller-named id is spared by two rules. The version
                # ``get_latest_version`` resolves (newest ready or superseded) is spared, since a
                # rebuild's building or failed row can outrank it in MAX(version) while it is still
                # what readers get. MAX(version) is spared too. A minted id has no current value.
                query += """
                  AND (
                      av.minted = 1
                      OR (
                          av.version < (
                              SELECT MAX(latest.version)
                              FROM artifact_versions latest
                              WHERE latest.id = av.id
                          )
                          AND NOT (
                              av.state IN ('ready', 'superseded')
                              AND NOT EXISTS (
                                  SELECT 1 FROM artifact_versions newer
                                  WHERE newer.id = av.id
                                    AND newer.state IN ('ready', 'superseded')
                                    AND newer.version > av.version
                              )
                          )
                      )
                  )
                """
            # The recorded last use can trail the real one by up to
            # _USE_RESOLUTION_SECONDS, so a floor honours that lag too.
            floor = min_idle_seconds + _USE_RESOLUTION_SECONDS if min_idle_seconds else 0.0
            params: list[float | str] = [now - floor]
            if tenant is not None:
                query += " AND (av.tenant = ? OR av.tenant = '' OR av.tenant IS NULL)"
                params.append(tenant)
            # Least recently used first; the version breaks ties, so a sweep
            # facing an id whose versions share a timestamp takes the oldest.
            query += f" ORDER BY {used} ASC, av.id ASC, av.version ASC"

            protected = self._protected_reachable(conn, current_values=not collect_latest)
            candidates = [
                row
                for row in conn.execute(query, params).fetchall()
                if (row["id"], row["version"]) not in protected
            ]
            # Measured over what the sweep may collect: a tenant's cap is on
            # its own share, or one tenant's sweep would empty its cache while
            # the others kept the store over the cap.
            size_sql = (
                "SELECT COALESCE(SUM(byte_size), 0) FROM artifact_versions "
                "WHERE superseded_by IS NULL"
            )
            size_params: list[str] = []
            if tenant is not None:
                size_sql += " AND (tenant = ? OR tenant = '' OR tenant IS NULL)"
                size_params.append(tenant)
            store_bytes = int(conn.execute(size_sql, size_params).fetchone()[0])

            chosen: dict[tuple[str, int], Any] = {}
            if max_idle_days is not None:
                idle_before = now - max_idle_days * 86400
                for row in candidates:
                    if row["used"] < idle_before:
                        chosen[(row["id"], row["version"])] = row
            if max_bytes is not None and store_bytes > max_bytes:
                target = int(max_bytes * _EVICT_TO_FRACTION)
                remaining = store_bytes - sum(r["byte_size"] or 0 for r in chosen.values())
                for row in candidates:
                    if remaining <= target:
                        break
                    key = (row["id"], row["version"])
                    if key not in chosen:
                        chosen[key] = row
                        remaining -= row["byte_size"] or 0

            if keep_superseded is not None:
                kept: dict[str, int] = {}
                for row in sorted(candidates, key=lambda r: (r["id"], -r["version"])):
                    if row["state"] != "failed" and kept.get(row["id"], 0) < keep_superseded:
                        kept[row["id"]] = kept.get(row["id"], 0) + 1
                        continue
                    chosen.setdefault((row["id"], row["version"]), row)

            chosen = self._with_their_pointers(conn, chosen, candidates)

            if dry_run:
                return {
                    "deleted_count": len(chosen),
                    "deleted_bytes": sum(r["byte_size"] or 0 for r in chosen.values()),
                    "store_bytes": store_bytes,
                    "dry_run": True,
                    "collected": [
                        {
                            "artifact_id": r["id"],
                            "version": r["version"],
                            "byte_size": r["byte_size"] or 0,
                            "last_used_at": r["used"],
                        }
                        for r in chosen.values()
                    ],
                }

            deleted_count = 0
            deleted_bytes = 0

            # Metadata first, then blobs, as ``delete_artifact`` does. The reverse order can leave
            # 'ready' rows whose blob is gone after a crash or a raising backend, a corrupt store;
            # losing a blob whose row is gone only wastes bytes.
            collected: list[tuple[str, int]] = []
            pending: list[tuple[str, int]] = []
            # Pointers before their canonicals, so a canonical goes only once nothing reads it.
            ordered = sorted(chosen.values(), key=lambda r: r["superseded_by"] is None)
            for position, row in enumerate(ordered):
                # Committed in batches: one transaction over a large sweep holds SQLite's write
                # lock long enough to stall every writer behind it.
                if position and position % _GC_DELETE_BATCH == 0:
                    conn.commit()
                    collected.extend(pending)
                    pending.clear()
                    if self._dialect.name == "sqlite":
                        # A waiting writer polls the lock (up to every 100 ms) rather than
                        # queueing, so it only gets in if the lock stays free that long.
                        time.sleep(_GC_BATCH_PAUSE_SECONDS)
                artifact_id, version, byte_size = row["id"], row["version"], row["byte_size"] or 0

                # Re-checked here, not trusted from the SELECT: a hit or a new hold during the
                # sweep keeps the version. The no-op UPDATE also locks the row, so a use cannot
                # land between this check and the DELETE.
                claimed = conn.execute(
                    """
                    UPDATE artifact_versions SET state = state
                    WHERE id = ? AND version = ? AND COALESCE(last_used_at, created_at) <= ?
                      AND NOT EXISTS (SELECT 1 FROM artifact_names n
                                      WHERE n.artifact_id = ? AND n.version = ?)
                      AND NOT EXISTS (SELECT 1 FROM artifact_aliases a
                                      WHERE a.artifact_id = ? AND a.version = ?)
                      AND NOT EXISTS (SELECT 1 FROM artifact_pins p
                                      WHERE p.artifact_id = ? AND p.version = ?)
                      AND NOT EXISTS (SELECT 1 FROM artifact_publications pub
                                      WHERE pub.artifact_id = ? AND pub.version = ?)
                      AND NOT EXISTS (SELECT 1 FROM artifact_versions ptr
                                      WHERE ptr.superseded_by = ?)
                    """,
                    (
                        artifact_id,
                        version,
                        row["used"],
                        *(artifact_id, version) * 4,
                        f"{artifact_id}@v={version}",
                    ),
                )
                if claimed.rowcount == 0:
                    logger.info(
                        "garbage_collect: keeping %s@v=%d, used or held since it was selected.",
                        artifact_id,
                        version,
                    )
                    continue

                # A savepoint per row: on Postgres a failed statement aborts the whole
                # transaction, and the closing commit would then silently roll back every row.
                conn.execute("SAVEPOINT gc_row")
                try:
                    self._delete_version_children(conn, artifact_id, version)
                    conn.execute(
                        "DELETE FROM artifact_versions WHERE id = ? AND version = ?",
                        (artifact_id, version),
                    )
                except self._dialect.integrity_error:
                    # A name or alias was pointed at this version since the SELECT above; the
                    # pointer wins. Skip it (uncounted, blob untouched) rather than fail the whole
                    # sweep on one race.
                    conn.execute("ROLLBACK TO SAVEPOINT gc_row")
                    logger.info(
                        "garbage_collect: skipping %s@v=%d, something referenced "
                        "it after it was selected.",
                        artifact_id,
                        version,
                    )
                    continue
                conn.execute("RELEASE SAVEPOINT gc_row")

                attempt = row["blob_attempt"]
                blob_id = attempt_blob_id(artifact_id, attempt) if attempt else artifact_id
                pending.append((blob_id, version))
                deleted_count += 1
                deleted_bytes += byte_size

            conn.commit()
            # Only rows whose delete committed lose their bytes.
            collected.extend(pending)
        finally:
            conn.close()

        removed = self.blob_store.remove_stale_temp_files(_ABANDONED_WRITE_SECONDS)
        if removed:
            logger.info("garbage_collect: removed %d temp file(s) of abandoned writes", removed)

        # Best-effort blob cleanup after the metadata is durably gone; a failure only orphans bytes.
        #
        # Outside the connection scope: thousands of remote deletes at 50-200ms each would hold a
        # pooled connection for minutes, and concurrent sweeps would exhaust the pool. `collected`
        # is already materialized.
        for blob_id, version in collected:
            try:
                self.blob_store.delete_blob(blob_id, version)
            except Exception:
                logger.exception(
                    "garbage_collect: failed to delete blob for %s@v=%d "
                    "(metadata already removed; bytes orphaned)",
                    blob_id,
                    version,
                )

        return {
            "deleted_count": deleted_count,
            "deleted_bytes": deleted_bytes,
            "store_bytes": store_bytes,
            "dry_run": False,
        }

    def _with_their_pointers(
        self, conn: StoreConnection, chosen: dict[tuple[str, int], Any], candidates: list
    ) -> dict[tuple[str, int], Any]:
        """``chosen`` without version gaps, with each chosen canonical's pointers.

        A version overtaken by a duplicate reads its canonical's bytes, so it goes with
        the canonical; a canonical whose pointer must stay (held, recently used, or the
        top of its id) stays too.
        """
        pointers = {
            (row["id"], row["version"]): _split_ref(row["superseded_by"])
            for row in conn.execute(
                "SELECT id, version, superseded_by FROM artifact_versions "
                "WHERE superseded_by IS NOT NULL"
            ).fetchall()
        }
        by_key = {(row["id"], row["version"]): row for row in candidates}
        # A pointer the gap rule dropped is never offered again, so this ends.
        dropped: set[tuple[str, int]] = set()
        while True:
            gapless = self._without_version_gaps(conn, chosen)
            dropped |= chosen.keys() - gapless.keys()
            chosen = gapless
            changed = False
            for pointer, canonical in pointers.items():
                if canonical not in chosen or pointer in chosen:
                    continue
                if pointer in by_key and pointer not in dropped:
                    chosen[pointer] = by_key[pointer]
                else:
                    del chosen[canonical]
                changed = True
            if not changed:
                return chosen

    @staticmethod
    def _without_version_gaps(
        conn: StoreConnection, chosen: dict[tuple[str, int], Any]
    ) -> dict[tuple[str, int], Any]:
        """``chosen``, less any id's highest version whose lower versions stay.

        Otherwise the next ``create_artifact`` for that id would reuse the number.
        """
        by_id: dict[str, int] = {}
        for artifact_id, _ in chosen:
            by_id[artifact_id] = by_id.get(artifact_id, 0) + 1
        ids = list(by_id)
        tops: dict[str, tuple[int, int]] = {}
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            placeholders = ", ".join("?" for _ in batch)
            for row in conn.execute(
                "SELECT id, MAX(version) AS top, COUNT(*) AS n FROM artifact_versions "
                f"WHERE id IN ({placeholders}) GROUP BY id",
                batch,
            ).fetchall():
                tops[row["id"]] = (row["top"], row["n"])
        return {
            key: row
            for key, row in chosen.items()
            if not (key[1] == tops[key[0]][0] and by_id[key[0]] < tops[key[0]][1])
        }

    def get_usage(self, tenant: str | None = None, *, include_tenantless: bool = True) -> dict:
        """Get artifact store usage statistics, optionally for one tenant.

        ``include_tenantless`` counts legacy tenantless rows as the tenant's: right
        for visibility, wrong for metering.
        """
        conn = self._get_connection()
        try:
            # total_bytes is what versions still own, as a sweep measures the store: older
            # superseded versions count, a pointer reading another's bytes (superseded_by) never.
            usage_query = """
                SELECT
                    COUNT(DISTINCT id) as unique_artifacts,
                    COUNT(*) as total_versions,
                    COUNT(CASE WHEN state = 'ready' THEN 1 END) as ready_versions,
                    COUNT(CASE WHEN state = 'building' THEN 1 END) as building_versions,
                    COUNT(CASE WHEN state = 'superseded' THEN 1 END) as superseded_versions,
                    COUNT(CASE WHEN state = 'failed' THEN 1 END) as failed_versions,
                    COALESCE(SUM(CASE WHEN state IN ('ready', 'superseded')
                                       AND superseded_by IS NULL
                                      THEN byte_size END), 0) as total_bytes,
                    COALESCE(SUM(CASE WHEN state = 'ready' THEN row_count END), 0) as total_rows,
                    MIN(created_at) as oldest_artifact,
                    MAX(created_at) as newest_artifact
                FROM artifact_versions
            """
            usage_params: list[str] = []
            if tenant is not None:
                usage_query += (
                    " WHERE tenant = ? OR tenant = '' OR tenant IS NULL"
                    if include_tenantless
                    else " WHERE tenant = ?"
                )
                usage_params.append(tenant)

            cursor = conn.execute(usage_query, usage_params)
            row = cursor.fetchone()

            if tenant is not None:
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM artifact_names WHERE tenant = ?"
                    + (" OR tenant = ''" if include_tenantless else ""),
                    (tenant,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as count FROM artifact_names")
            names_count = cursor.fetchone()["count"]

            unreferenced_query = """
                SELECT COUNT(*) as count
                FROM artifact_versions av
                LEFT JOIN artifact_names an ON av.id = an.artifact_id AND av.version = an.version
                WHERE an.name IS NULL AND av.state = 'ready'
            """
            unreferenced_params: list[str] = []
            if tenant is not None:
                unreferenced_query += (
                    " AND (av.tenant = ? OR av.tenant = '' OR av.tenant IS NULL)"
                    if include_tenantless
                    else " AND av.tenant = ?"
                )
                unreferenced_params.append(tenant)
            cursor = conn.execute(unreferenced_query, unreferenced_params)
            unreferenced_count = cursor.fetchone()["count"]

            return {
                "unique_artifacts": row["unique_artifacts"],
                "total_versions": row["total_versions"],
                "ready_versions": row["ready_versions"],
                "building_versions": row["building_versions"],
                "superseded_versions": row["superseded_versions"],
                "failed_versions": row["failed_versions"],
                # int(): SUM over a BIGINT column is numeric in Postgres and
                # arrives as Decimal, which breaks arithmetic like
                # total_bytes / 1024**3 for in-process callers.
                "total_bytes": int(row["total_bytes"]),
                "total_rows": int(row["total_rows"]),
                "name_count": names_count,
                "unreferenced_count": unreferenced_count,
                "oldest_artifact": row["oldest_artifact"],
                "newest_artifact": row["newest_artifact"],
            }
        finally:
            conn.close()

    # --- Maintenance (legacy) ---

    def sweep_zombie_builds(self, max_age_seconds: float = 3600) -> int:
        """Mark ``building`` artifacts older than ``max_age_seconds`` as failed; return the count.

        It demotes every old enough ``building`` row unconditionally, which is safe
        only where no live writer holds one that long: a server before its build
        runner accepts work, or a notebook's own store.
        """
        conn = self._get_connection()
        try:
            cutoff = time.time() - max_age_seconds
            cursor = conn.execute(
                """
                UPDATE artifact_versions
                SET state = 'failed'
                WHERE state = 'building' AND created_at < ?
                """,
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def verify_artifacts(self, tenant: str | None = None) -> list[dict]:
        """Check every ready or superseded artifact's blob against its metadata.

        The blob must exist and match its recorded digest. An Arrow blob (a declared
        ``arrow/ipc`` content type, or none, as core transforms write) must also parse as
        one IPC stream and match ``row_count`` when one was recorded; a notebook's JSON,
        image or pickled outputs are not Arrow. Returns one ``{"artifact_id", "version",
        "state", "problem", "detail"}`` dict per problem; empty means consistent.
        """
        import pyarrow as pa

        from strata.fast_io import validate_ipc_stream

        conn = self._get_connection()
        try:
            query = """
                SELECT id, version, state, row_count, content_sha256, blob_attempt, superseded_by,
                       transform_spec
                FROM artifact_versions
                WHERE state IN ('ready', 'superseded')
            """
            params: list[str] = []
            if tenant is not None:
                query += " AND (tenant = ? OR tenant = '' OR tenant IS NULL)"
                params.append(tenant)
            rows = conn.execute(query, params).fetchall()
            attempts = {(row["id"], row["version"]): row["blob_attempt"] for row in rows}
        finally:
            conn.close()

        findings: list[dict] = []
        for row in rows:
            artifact_id, version = row["id"], row["version"]
            # A version a duplicate build overtook reads its canonical's bytes.
            blob_owner = _split_ref(row["superseded_by"]) if row["superseded_by"] else None
            owner_id, owner_version = blob_owner or (artifact_id, version)
            attempt = attempts.get((owner_id, owner_version))
            blob_id = attempt_blob_id(owner_id, attempt) if attempt else owner_id
            data = self.blob_store.read_blob(blob_id, owner_version)
            if data is None:
                findings.append(
                    {
                        "artifact_id": artifact_id,
                        "version": version,
                        "state": row["state"],
                        "problem": "missing_blob",
                        "detail": (
                            f"metadata row reads {row['superseded_by']}, whose blob is gone"
                            if blob_owner
                            else "metadata row exists but blob is gone"
                        ),
                    }
                )
                continue

            if _declared_content_type(row["transform_spec"]) in ("", "arrow/ipc"):
                try:
                    readable_rows = validate_ipc_stream(data)
                except (ValueError, pa.ArrowInvalid) as e:
                    findings.append(
                        {
                            "artifact_id": artifact_id,
                            "version": version,
                            "state": row["state"],
                            "problem": "invalid_stream",
                            "detail": str(e),
                        }
                    )
                    continue

                if row["row_count"] is not None and readable_rows != row["row_count"]:
                    findings.append(
                        {
                            "artifact_id": artifact_id,
                            "version": version,
                            "state": row["state"],
                            "problem": "row_count_mismatch",
                            "detail": (
                                f"metadata says {row['row_count']}, blob yields {readable_rows}"
                            ),
                        }
                    )

            # The checks above catch bytes that stopped being valid Arrow or stopped holding their
            # claimed rows. A digest catches an in-place value change that keeps both true. Rows
            # with no digest predate the column and are skipped.
            recorded = row["content_sha256"]
            if recorded:
                actual = hashlib.sha256(data).hexdigest()
                if actual != recorded:
                    findings.append(
                        {
                            "artifact_id": artifact_id,
                            "version": version,
                            "state": row["state"],
                            "problem": "digest_mismatch",
                            "detail": f"recorded {recorded[:12]}…, blob hashes to {actual[:12]}…",
                        }
                    )
        return findings

    def cleanup_failed(self, max_age_seconds: float = 3600) -> int:
        """Delete failed artifacts older than ``max_age_seconds``; return the count."""
        conn = self._get_connection()
        try:
            cutoff = time.time() - max_age_seconds
            cursor = conn.execute(
                """
                SELECT id, version, blob_attempt FROM artifact_versions
                WHERE state = 'failed' AND created_at < ?
                """,
                (cutoff,),
            )
            rows = cursor.fetchall()

            # Metadata first, then blobs: a mid-sweep failure then orphans bytes rather than leaving
            # rows pointing at deleted blobs, as garbage_collect does.
            for row in rows:
                # A failed build's row still references the version.
                self._delete_version_children(conn, row["id"], row["version"])
                conn.execute(
                    "DELETE FROM artifact_versions WHERE id = ? AND version = ?",
                    (row["id"], row["version"]),
                )
            conn.commit()
        finally:
            conn.close()

        # Outside the connection scope: remote blob deletes are network I/O
        # and would otherwise park a pooled slot for the whole sweep.
        for row in rows:
            try:
                attempt = row["blob_attempt"]
                blob_id = attempt_blob_id(row["id"], attempt) if attempt else row["id"]
                self.blob_store.delete_blob(blob_id, row["version"])
            except Exception:
                logger.exception(
                    "cleanup_failed: failed to delete blob for %s@v=%d "
                    "(metadata already removed; bytes orphaned)",
                    row["id"],
                    row["version"],
                )
        return len(rows)

    def stats(self, tenant: str | None = None, *, include_tenantless: bool = True) -> dict:
        """Get artifact store statistics; ``include_tenantless`` as for :meth:`get_usage`."""
        conn = self._get_connection()
        try:
            stats_query = """
                SELECT
                    COUNT(*) as total_versions,
                    COUNT(CASE WHEN state = 'ready' THEN 1 END) as ready_versions,
                    COUNT(CASE WHEN state = 'building' THEN 1 END) as building_versions,
                    COUNT(CASE WHEN state = 'superseded' THEN 1 END) as superseded_versions,
                    COUNT(CASE WHEN state = 'failed' THEN 1 END) as failed_versions,
                    COALESCE(SUM(CASE WHEN state IN ('ready', 'superseded')
                                       AND superseded_by IS NULL
                                      THEN byte_size END), 0) as total_bytes,
                    COALESCE(SUM(CASE WHEN state = 'ready' THEN row_count END), 0) as total_rows
                FROM artifact_versions
            """
            stats_params: list[str] = []
            if tenant is not None:
                stats_query += (
                    " WHERE tenant = ? OR tenant = '' OR tenant IS NULL"
                    if include_tenantless
                    else " WHERE tenant = ?"
                )
                stats_params.append(tenant)

            cursor = conn.execute(stats_query, stats_params)
            row = cursor.fetchone()

            if tenant is not None:
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM artifact_names WHERE tenant = ?"
                    + (" OR tenant = ''" if include_tenantless else ""),
                    (tenant,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as count FROM artifact_names")
            names_count = cursor.fetchone()["count"]

            return {
                "total_versions": row["total_versions"],
                "ready_versions": row["ready_versions"],
                "building_versions": row["building_versions"],
                "superseded_versions": row["superseded_versions"],
                "failed_versions": row["failed_versions"],
                # int(): SUM over a BIGINT column is numeric in Postgres and
                # arrives as Decimal, which breaks arithmetic like
                # total_bytes / 1024**3 for in-process callers.
                "total_bytes": int(row["total_bytes"]),
                "total_rows": int(row["total_rows"]),
                "name_count": names_count,
            }
        finally:
            conn.close()


# --- Module-level singleton (initialized lazily) ---

_artifact_store: ArtifactStore | None = None


def get_artifact_store(
    artifact_dir: Path | None = None,
    blob_store: BlobStore | None = None,
    dialect: SqlDialect | None = None,
) -> ArtifactStore | None:
    """Get the artifact store singleton, or None if it was never created.

    The arguments only take effect on the first call with an ``artifact_dir``.
    """
    global _artifact_store
    if _artifact_store is None and artifact_dir is not None:
        _artifact_store = ArtifactStore(artifact_dir, blob_store=blob_store, dialect=dialect)
    return _artifact_store


def reset_artifact_store() -> None:
    """Reset the artifact store singleton (for testing)."""
    global _artifact_store
    _artifact_store = None
