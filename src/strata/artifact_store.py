"""Artifact store for personal mode.

The artifact store manages cached materialized query results:
1. Artifact versions: Immutable Arrow IPC blobs indexed by (id, version)
2. Name pointers: Mutable names that point to specific artifact versions

Disk layout:
    {artifact_dir}/
        artifacts.sqlite      # Metadata database
        blobs/
            {id}@v={version}.arrow  # Arrow IPC stream files

Blob storage:
    The blob storage backend is pluggable via the BlobStore abstraction.
    Supported backends:
    - LocalBlobStore: Local filesystem (default)
    - S3BlobStore: Amazon S3 / S3-compatible storage

Provenance hash:
    Each artifact has a provenance_hash = sha256(sorted(input_hashes) + transform_spec)
    This enables deduplication: if the same inputs + transform exist, return existing artifact.

Security:
    Artifact store is only enabled in personal mode (local development).
    In service mode, all write operations return 403.
"""

from __future__ import annotations

import hashlib
import json
import logging
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


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactVersion:
    """Immutable artifact version metadata.

    Attributes:
        id: Unique artifact identifier (UUID)
        version: Version number (monotonically increasing per id)
        state: Lifecycle state ("building", "ready", "failed")
        provenance_hash: Hash of inputs + transform for deduplication
        schema_json: Arrow schema serialized as JSON
        row_count: Number of rows in the artifact
        byte_size: Size of the Arrow IPC file in bytes
        created_at: Unix timestamp when the artifact was created
        transform_spec: JSON-serialized transform specification (opaque to server)
        input_versions: JSON-serialized dict mapping input URI -> version string
            Used for staleness detection. For table URIs, version is snapshot_id.
            For artifact URIs, version is "artifact_id@v=N".
        tenant: Tenant ID that owns this artifact (for multi-tenant isolation)
        principal: Principal ID that created this artifact
        content_sha256: SHA-256 of the stored bytes. What makes two runs of the
            same notebook on two machines comparable output by output, without
            publishing anything and without a second read of every blob.
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
class ImportedArtifact:
    """Where an imported record landed in the destination store.

    ``id`` and ``version`` are not always the caller's: an import whose
    computation the store already holds under another id resolves to that row.
    A caller copying a chain has to read them rather than assume, or the
    descendants it imports next will name edges the store never received.

    Attributes:
        id: The artifact id the record resolved to in this store
        version: The version it resolved to
        written: Whether a new row was inserted (False for either no-op)
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

    Attributes:
        token: URL-safe secret naming this publication. Unguessable, because
            it is the only thing standing between an unauthenticated reader
            and the artifact.
        artifact_id: The published artifact.
        version: The published version. Bound to the token permanently — a
            citation whose target could be repointed would be worthless.
        tenant: Tenant that owns the publication ('' when tenantless).
        title: Optional human label for the page.
        published_at: When the grant was made.
        published_by: Who made it, when an authenticated identity was
            available. Empty for a personal-mode publish, which has none.
        content_sha256: Digest of the bytes as published. Nothing else in the
            store records one — ``provenance_hash`` covers inputs and
            transform, not content — so without this the page could make no
            checkable integrity claim at all. It proves the bytes a reader
            downloads are the bytes that were published; it proves nothing
            about whether they were honestly produced, and the page says so.
        revoked_at: When the grant was withdrawn, else ``None``. The row
            survives revocation so the token is never reissued.
        authors: Ordered ``{"name", "orcid", "affiliation"}`` entries. Beside
            ``published_by`` rather than replacing it: who *made* the grant and
            who *wrote the work* are different questions, and the second one
            has an order, an affiliation and an identifier the first never did.
            Empty leaves ``published_by`` as the byline, so nothing already
            published changes meaning.
        external_ids: ``{"scheme", "value"}`` entries — ``doi``, ``zenodo``,
            ``arxiv``, ``url``. Usually set after the token exists, because a
            DOI is registered against a deposit that already has to be
            reachable.
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

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


@dataclass(frozen=True)
class ArtifactName:
    """Mutable name pointer to an artifact version.

    Names provide human-readable aliases for artifacts, e.g.:
        strata://name/daily_revenue -> strata://artifact/abc123@v=5

    In multi-tenant mode, names are scoped by tenant. The unique key is
    (tenant, name), so different tenants can have the same name.

    Attributes:
        name: Human-readable name (e.g., "daily_revenue")
        artifact_id: ID of the pinned artifact
        version: Version of the pinned artifact
        updated_at: Unix timestamp of last update
        tenant: Tenant ID that owns this name (for multi-tenant isolation)
    """

    name: str
    artifact_id: str
    version: int
    updated_at: float
    tenant: str | None = None  # Tenant ID for multi-tenant isolation


@dataclass(frozen=True)
class ArtifactAlias:
    """An intent pointer (champion, candidate, ...) on a registry name.

    Aliases follow the post-stages registry model: a name can hold many
    aliases, each a mutable pointer to one artifact version. Every move
    is recorded in the append-only registry audit.
    """

    name: str
    alias: str
    artifact_id: str
    version: int
    updated_at: float
    tenant: str | None = None


@dataclass(frozen=True)
class InputChange:
    """Describes a change in an input dependency.

    Attributes:
        input_uri: The input URI that changed
        old_version: The version used when artifact was built
        new_version: The current version of the input
    """

    input_uri: str
    old_version: str
    new_version: str

    def __str__(self) -> str:
        return f"{self.input_uri}: {self.old_version} → {self.new_version}"


@dataclass
class NameStatus:
    """Status information for a named artifact, including staleness.

    Attributes:
        name: The artifact name
        artifact_uri: URI of the pinned artifact version
        artifact_id: Artifact ID
        version: Pinned version number
        state: Artifact state ("ready", "building", "failed")
        updated_at: When the name was last updated
        input_versions: Mapping of input URI -> version when built
        is_stale: True if any input has changed since build
        stale_reason: Human-readable explanation if stale
        changed_inputs: List of inputs that changed
    """

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
    """Transform specification (opaque to server, executed by client).

    The server stores this but never interprets params.

    Attributes:
        executor: Executor URI (e.g., "local://duckdb_sql@v1")
        params: Opaque parameters for the executor (e.g., SQL query)
        inputs: List of input URIs (table URIs or artifact URIs)
    """

    executor: str
    params: dict
    inputs: list[str]

    def to_json(self) -> str:
        """Serialize to JSON string.

        Inputs keep their caller order: positional executors bind them as
        ``input0, input1, ...``, so order IS computation semantics — both
        for the build runner (which executes from this stored spec) and
        for provenance (``f(a, b)`` must not dedup against ``f(b, a)``).
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


# ---------------------------------------------------------------------------
# Provenance Hash
# ---------------------------------------------------------------------------


def compute_provenance_hash(input_hashes: list[str], transform_spec: TransformSpec) -> str:
    """Compute deterministic provenance hash for deduplication.

    The hash uniquely identifies a computation based on:
    1. Content hashes of all inputs (sorted for determinism)
    2. The transform specification

    Args:
        input_hashes: Content hashes of input artifacts/tables (will be sorted)
        transform_spec: The transform to apply

    Returns:
        SHA-256 hex digest of the combined provenance
    """
    # Sort input hashes for deterministic ordering
    sorted_inputs = sorted(input_hashes)

    # Combine with transform spec
    hasher = hashlib.sha256()
    for h in sorted_inputs:
        hasher.update(h.encode("utf-8"))
        hasher.update(b"\x00")  # Separator
    hasher.update(transform_spec.to_json().encode("utf-8"))

    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Artifact Store
# ---------------------------------------------------------------------------

# Schema evolution.
#
# The ad-hoc migrations further down are SQLite-only by construction: they
# speak PRAGMA and sqlite_master, and they are guarded by
# ``supports_legacy_migration`` because only SQLite had deployed databases when
# they were written. Postgres, meanwhile, returns as soon as the schema exists
# — so there was no way at all to add a column to a Postgres store that already
# held data. That was fine while Postgres was new. It stops being fine the
# moment one holds something worth keeping.
#
# So: an ordered list, a recorded version, and one rule — the schema constants
# above always describe the *latest* shape, and the migrations describe the
# path to it from the baseline. A fresh database is created from the constants
# and stamped at the latest version, so it never runs a migration; an existing
# one is stamped at the baseline and walks forward. The two must agree, which
# is what ``test_a_migrated_database_matches_a_fresh_one`` checks, because
# nothing else would notice them drifting.

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
    """One forward step, applied at most once per database.

    No down-migration. Reversing a schema change on a store whose whole promise
    is that written artifacts stay readable is not a thing anyone should reach
    for under pressure; restoring a copy is.
    """

    version: int
    description: str
    apply: Callable[[StoreConnection, SqlDialect], None]


def _add_content_sha256(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Give every artifact version somewhere to record its content digest.

    Nullable: ``finalize_artifact`` fills it for anything written since, and
    ``content_digest`` fills it on demand for rows that predate it. A column
    that were NOT NULL would have had to be backfilled by reading every blob in
    the store before the migration could finish.
    """
    if not dialect.column_exists(conn, "artifact_versions", "content_sha256"):
        conn.execute("ALTER TABLE artifact_versions ADD COLUMN content_sha256 TEXT")


def _json_entries(raw: str | None) -> tuple[dict[str, str], ...]:
    """A stored JSON list back into entries, or empty for anything else.

    Rows predating the columns hold NULL, and a store this one wrote is
    trusted for shape — but a column that has been through a migration and an
    older writer is worth reading defensively once, here, rather than at every
    place the page and the crate consume it.
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

    An author with only a name is the common case, so a missing ORCID must not
    become the string ``"None"`` on the page.
    """
    cleaned: list[dict[str, str]] = []
    for entry in entries or []:
        kept = {field: str(entry[field]).strip() for field in fields if entry.get(field)}
        if kept:
            cleaned.append(kept)
    return tuple(cleaned)


def _json_or_none(entries: tuple[dict[str, str], ...]) -> str | None:
    """Store nothing rather than ``[]`` — an empty list and never-set are the
    same fact, and one of them reads as a deliberate erasure."""
    return json.dumps([dict(entry) for entry in entries]) if entries else None


def _add_publication_credits(conn: StoreConnection, dialect: SqlDialect) -> None:
    """Give a publication somewhere to record who wrote it and what names it.

    Both nullable, and nothing is backfilled: ``published_by`` stays the byline
    for every publication made before this, which is what keeps a link printed
    in a paper saying the same thing it said yesterday.
    """
    for column in ("authors", "external_ids"):
        if not dialect.column_exists(conn, "artifact_publications", column):
            conn.execute(f"ALTER TABLE artifact_publications ADD COLUMN {column} TEXT")


_MIGRATIONS: list[_Migration] = [
    _Migration(1, "artifact_versions.content_sha256", _add_content_sha256),
    _Migration(2, "artifact_publications.authors + external_ids", _add_publication_credits),
]

_LATEST_SCHEMA_VERSION = max((m.version for m in _MIGRATIONS), default=_BASELINE_SCHEMA_VERSION)


# SQL schema for artifact metadata
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
    PRIMARY KEY (id, version)
);

-- Index for provenance lookup (deduplication)
CREATE INDEX IF NOT EXISTS idx_provenance ON artifact_versions(provenance_hash);

-- Unique constraint for idempotent finalize: (tenant, provenance_hash) for ready artifacts.
-- Prevents duplicate ready artifacts for the same computation within a tenant. Relies on
-- tenant being '' (never NULL) for tenantless rows — SQLite treats NULLs as distinct, which
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

# Registry tables (aliases / tags / audit). Executed unconditionally at
# init — CREATE IF NOT EXISTS makes it idempotent for both fresh and
# existing databases (#129).
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
# A token is bound to one (artifact_id, version) for good. Revoking sets
# ``revoked_at`` and keeps the row, so a token can never be reused to point at
# different content — the whole value of a URL printed in a paper is that what
# it resolves to cannot change under the reader. A revoked citation has to fail
# closed, not resolve to something else.
_PUBLICATION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS artifact_publications (
    token TEXT PRIMARY KEY,
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
"""

# Migration SQL to add tenant columns to existing tables
_MIGRATION_SQL = """
-- Add tenant and principal columns to artifact_versions if they don't exist
-- SQLite doesn't have ADD COLUMN IF NOT EXISTS, so we use a workaround

-- Check if tenant column exists by trying to select it
-- If it fails, the column doesn't exist and we need to add it
"""


# Bound on force_finalize_canonical's retry. A conflict means a competing
# writer committed a ready row with our provenance; one retry supersedes it
# and wins. More than two rounds means sustained contention on a single
# provenance, where failing loudly beats spinning.
_CANONICAL_PROMOTE_ATTEMPTS = 3


# Sentinel for read_audit's tenant param: distinguishes "no tenant filter at
# all" (the direct-store CLI / admin view of the whole store) from an
# explicit tenant filter of None — which normalizes to the '' default tenant.
_AUDIT_ALL_TENANTS = object()


class ArtifactStore:
    """SQLite-backed artifact store for personal mode.

    Thread-safe: uses connection per operation with WAL mode.

    The store separates metadata (SQLite) from blob data (BlobStore).
    This enables pluggable blob storage backends (local, S3, GCS).

    Example usage:
        store = ArtifactStore(Path("~/.strata/artifacts"))

        # Create new artifact (starts in "building" state)
        version = store.create_artifact(
            artifact_id="abc123",
            provenance_hash="sha256...",
            transform_spec=spec,
        )

        # Write blob and finalize
        store.write_blob("abc123", version, arrow_bytes)
        store.finalize_artifact("abc123", version, schema_json, row_count, len(arrow_bytes))

        # Look up by provenance (deduplication)
        existing = store.find_by_provenance("sha256...")

        # Create/update name pointer
        store.set_name("daily_revenue", "abc123", 5)

        # Resolve name
        artifact = store.resolve_name("daily_revenue")
    """

    def __init__(
        self,
        artifact_dir: Path,
        blob_store: BlobStore | None = None,
        dialect: SqlDialect | None = None,
    ):
        """Initialize artifact store.

        Args:
            artifact_dir: Directory for artifacts (contains metadata DB)
            blob_store: Optional blob storage backend. If None, creates a
                LocalBlobStore in {artifact_dir}/blobs.
            dialect: Optional metadata backend. Defaults to SQLite under
                artifact_dir, which is what personal mode wants. Pass a
                PostgresDialect to put the metadata on a shared server;
                artifact_dir then only holds blobs, and blob_store should
                usually be a remote backend too, since a store split across
                a shared database and a local blobs directory is only
                coherent on one machine.
        """
        self.artifact_dir = artifact_dir
        self.db_path = artifact_dir / "artifacts.sqlite"

        # Initialize blob store (default to local filesystem)
        if blob_store is None:
            from strata.blob_store import LocalBlobStore

            self.blobs_dir = artifact_dir / "blobs"
            self.blobs_dir.mkdir(parents=True, exist_ok=True)
            self.blob_store: BlobStore = LocalBlobStore(self.blobs_dir)
        else:
            self.blob_store = blob_store
            # For backwards compatibility, set blobs_dir if using local store
            from strata.blob_store import LocalBlobStore

            if isinstance(blob_store, LocalBlobStore):
                self.blobs_dir = blob_store.blobs_dir
            else:
                self.blobs_dir = artifact_dir / "blobs"  # May not exist for remote stores

        # Ensure artifact_dir exists (for metadata DB)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Every query in this class reaches the database through _get_connection,
        # so the dialect is the one place a second backend has to be taught
        # about. SQLite stays the default and behaves exactly as before; see
        # strata/sql_backend.py for what actually differs between backends.
        self._dialect: SqlDialect = dialect if dialect is not None else SqliteDialect(self.db_path)

        # Initialize schema
        self._init_schema()

    def _get_connection(self) -> StoreConnection:
        """Open a connection configured by the active dialect.

        WAL and the other PRAGMAs are SQLite's; see ``SqliteDialect.connect``.
        """
        return self._dialect.connect()

    @property
    def dialect(self) -> SqlDialect:
        """The metadata backend this store is using.

        Public so the build store can share it: build rows live in the same
        database, and sharing the dialect shares its connection pool too.
        """
        return self._dialect

    def close(self) -> None:
        """Release backend resources. Idempotent, and safe to skip on SQLite.

        Exists because the Postgres dialect holds a connection pool, and each
        pool runs worker threads that outlive the store otherwise.
        """
        self._dialect.close()

    def _init_schema(self) -> None:
        """Initialize database schema with migrations for tenant columns."""
        conn = self._get_connection()
        try:
            # The migration below upgrades databases written by older Strata
            # versions, and is expressed in SQLite's own introspection
            # vocabulary (sqlite_master, PRAGMA, rowid). A backend added after
            # those versions has no such history, so it goes straight to the
            # current schema rather than paying to have this made portable.
            if not self._dialect.supports_legacy_migration:
                # Fast path first: an ArtifactStore is constructed per session
                # in several places, and the lock below is *global*, so taking
                # it unconditionally would funnel every store construction in
                # the cluster through one mutex. Once the schema exists there
                # is nothing to serialize.
                if self._dialect.schema_exists(conn):
                    # Existing database: it may still be behind. Under the same
                    # global lock the creation path takes, since two replicas
                    # starting together would otherwise both apply migration N.
                    self._dialect.begin_write(conn, "__schema__")
                    self._apply_schema_migrations(conn)
                    return

                # CREATE TABLE IF NOT EXISTS is not concurrency-safe in
                # Postgres: it checks and then creates without holding a lock,
                # so simultaneous creators race in the system catalog and all
                # but one fail with a duplicate key on pg_type_typname_nsp_index.
                # Observed 7 of 8 nodes failing on a shared first boot, which is
                # exactly the multi-node case this backend exists to enable.
                # Both scripts share the one transaction so the lock covers them.
                self._dialect.begin_write(conn, "__schema__")
                conn.executescript(self._dialect.adapt_ddl(_SCHEMA_SQL))
                conn.executescript(self._dialect.adapt_ddl(_REGISTRY_SCHEMA_SQL))
                conn.executescript(self._dialect.adapt_ddl(_PUBLICATION_SCHEMA_SQL))
                # Created from the constants, which are the latest shape, so it
                # is already current and must not replay migrations that would
                # add what it was born with.
                conn.executescript(self._dialect.adapt_ddl(_SCHEMA_VERSION_SQL))
                self._stamp_schema_version(conn, _LATEST_SCHEMA_VERSION)
                conn.commit()
                return

            # Check if this is a fresh database or needs migration
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_versions'"
            )
            table_exists = cursor.fetchone() is not None

            if table_exists:
                # Check if tenant column exists
                cursor = conn.execute("PRAGMA table_info(artifact_versions)")
                columns = {row["name"] for row in cursor.fetchall()}

                if "tenant" not in columns:
                    # Migrate: add tenant and principal columns
                    conn.execute("ALTER TABLE artifact_versions ADD COLUMN tenant TEXT")
                    conn.execute("ALTER TABLE artifact_versions ADD COLUMN principal TEXT")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_versions_tenant "
                        "ON artifact_versions(tenant)"
                    )
                    conn.commit()

                # Normalize tenant storage and (re)build the uniqueness index.
                # Tenantless rows must use '' (not NULL) so the (tenant,
                # provenance_hash) uniqueness actually holds — SQLite treats
                # NULLs as distinct, which let duplicate tenantless ready rows
                # slip through. Drop the index, normalize NULL -> '', collapse
                # any pre-existing duplicate ready rows, then rebuild. Idempotent:
                # once there are no NULL tenants and the index exists, it skips.
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
                    # Collapse pre-existing duplicate ready rows (the bug this
                    # closes): keep the newest per (tenant, provenance_hash),
                    # supersede the rest so the unique index can be built. The
                    # superseded rows stay fetchable by explicit id+version.
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

                # Check if artifact_names needs migration
                cursor = conn.execute("PRAGMA table_info(artifact_names)")
                name_columns = {row["name"] for row in cursor.fetchall()}

                if "tenant" not in name_columns:
                    # Need to recreate artifact_names with new schema
                    # SQLite doesn't support changing primary key
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
                    # Migrate data with '' tenant (personal mode)
                    # Use '' instead of NULL for SQLite unique constraint compatibility
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
                # Fresh database: create schema
                conn.executescript(_SCHEMA_SQL)
                conn.executescript(_SCHEMA_VERSION_SQL)
                # Born at the latest shape, so it must not replay migrations
                # that would add what the constants already gave it.
                self._stamp_schema_version(conn, _LATEST_SCHEMA_VERSION)
                conn.commit()

            # Registry tables (aliases/tags/audit) — idempotent, applies to
            # fresh and existing databases alike (#129).
            conn.executescript(_REGISTRY_SCHEMA_SQL)
            conn.executescript(_PUBLICATION_SCHEMA_SQL)
            cursor = conn.execute("PRAGMA table_info(registry_audit)")
            audit_columns = {row["name"] for row in cursor.fetchall()}
            if "from_artifact_id" not in audit_columns:
                conn.execute("ALTER TABLE registry_audit ADD COLUMN from_artifact_id TEXT")
            conn.commit()

            # After the SQLite-only steps above, which brought a legacy
            # database up to the baseline. Everything from here is portable and
            # runs on both backends.
            self._apply_schema_migrations(conn)
        finally:
            conn.close()

    def _blob_path(self, artifact_id: str, version: int) -> Path:
        """Get path for artifact blob (for local storage only).

        Deprecated: Use blob_store methods directly instead.
        Kept for backwards compatibility with code that accesses blob files directly.
        """
        return self.blobs_dir / f"{artifact_id}@v={version}.arrow"

    # -----------------------------------------------------------------------
    # Artifact CRUD
    # -----------------------------------------------------------------------

    def create_artifact(
        self,
        artifact_id: str,
        provenance_hash: str,
        transform_spec: TransformSpec | None = None,
        input_versions: dict[str, str] | None = None,
        tenant: str | None = None,
        principal: str | None = None,
    ) -> int:
        """Create a new artifact version in "building" state.

        Args:
            artifact_id: Unique artifact ID
            provenance_hash: Provenance hash for deduplication
            transform_spec: Optional transform specification
            input_versions: Optional mapping of input URI -> version string
                Used for staleness detection. For tables, version is snapshot_id.
                For artifacts, version is "artifact_id@v=N".
            tenant: Optional tenant ID for multi-tenant isolation
            principal: Optional principal ID that created this artifact

        Returns:
            The new version number
        """
        conn = self._get_connection()
        try:
            # Serialize writers so the MAX(version)+1 read can't race a
            # concurrent create for the same artifact id (two refresh rebuilds
            # used to both read MAX=N and collide on the (id, version) primary
            # key with an uncaught IntegrityError). The key names what is
            # contended so a backend can lock narrowly; SQLite ignores it and
            # locks the file.
            self._dialect.begin_write(conn, artifact_id)
            cursor = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM artifact_versions WHERE id = ?",
                (artifact_id,),
            )
            version = cursor.fetchone()[0]

            # Serialize input_versions to JSON
            input_versions_json = json.dumps(input_versions) if input_versions else None

            # Insert new version
            conn.execute(
                """
                INSERT INTO artifact_versions
                    (id, version, state, provenance_hash, created_at,
                     transform_spec, input_versions, tenant, principal)
                VALUES (?, ?, 'building', ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    version,
                    provenance_hash,
                    time.time(),
                    transform_spec.to_json() if transform_spec else None,
                    input_versions_json,
                    # Tenantless rows store '' (never NULL) so the uniqueness
                    # index treats them as equal — matches names/aliases/tags.
                    tenant if tenant is not None else "",
                    principal,
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

        Scoped exactly as ``idx_tenant_provenance_unique`` is, because its
        whole job is to find the row that index would refuse to duplicate:
        ready rows only, within one tenant, with ``''`` and legacy ``NULL``
        read as the same tenantless namespace. Runs on the caller's open write
        transaction rather than through ``find_by_provenance``, which opens its
        own connection and would leave a window for the row to appear between
        the check and the insert.
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
        """Where this record already lives here, if it does — else ``None``.

        Two ways the store can already hold it: that exact ``id@v=N``, or the
        same computation under another id. Both mean nothing is written, and
        both have to name where the caller's descendants should point.
        """
        existing = conn.execute(
            "SELECT 1 FROM artifact_versions WHERE id = ? AND version = ?",
            (record.id, record.version),
        ).fetchone()
        if existing is not None:
            return ImportedArtifact(record.id, record.version, written=False)

        duplicate = self._ready_with_provenance(conn, record)
        if duplicate is not None:
            return ImportedArtifact(duplicate[0], duplicate[1], written=False)
        return None

    def _apply_schema_migrations(self, conn: StoreConnection) -> None:
        """Bring one database up to ``_LATEST_SCHEMA_VERSION``.

        Runs on every construction, and is a no-op on all but the first after
        an upgrade: reading one row from a tiny table is cheaper than deciding
        whether to bother.
        """
        conn.executescript(self._dialect.adapt_ddl(_SCHEMA_VERSION_SQL))
        row = conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
        current = row["version"] if row is not None and row["version"] is not None else None

        if current is None:
            # A database that predates this table. It is at the baseline by
            # definition — that is what the baseline means — so stamp it and
            # let the migrations below carry it forward.
            current = _BASELINE_SCHEMA_VERSION
            self._stamp_schema_version(conn, current)

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
        """Record a version, at most once.

        Checked rather than upserted because the two dialects spell that
        differently (``INSERT OR IGNORE`` against ``ON CONFLICT DO NOTHING``),
        and this runs inside a lock held for exactly this purpose, so the race
        the upsert would guard against cannot happen here.
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
        blob: bytes | None,
    ) -> None:
        """Fill in what an earlier, less complete write of this row left out.

        Idempotency for an import is by *completeness*, not by existence. Two
        ways a row can be here and still be missing something:

        Bytes. An import interrupted between the blob and the row leaves
        neither, but a row written by some other path may have lost its blob to
        a failed backend. Rewriting bytes we were handed anyway is free and
        makes the retry that repairs it actually repair it.

        Lineage. The team cache writes results through the by-provenance route,
        which records ``inputs=[]`` — a value keyed by a hash, not a node in a
        graph. Promoting the same computation later deduplicates onto that row
        and would otherwise leave the promoted chain resolving exactly one
        level before reaching an artifact that names no inputs. Since the cache
        fires on every successful cell and promotion is a deliberate act, the
        cache almost always gets there first, so "first writer wins" would mean
        "the least informative writer wins".

        This is not a mutation of an immutable artifact. The bytes, the id, the
        version and the provenance hash are untouched; what changes is a record
        completed by a path that had information the first writer never had.
        """
        if blob is not None and not self.blob_exists(landed.id, landed.version):
            with self.open_blob_writer(landed.id, landed.version) as writer:
                writer.write(blob)

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

    def import_artifact(self, record: ArtifactVersion, blob: bytes | None) -> ImportedArtifact:
        """Copy an artifact from another store, keeping its id *and* version.

        Returns where the record landed and whether anything was written, so a
        caller copying a chain can point the descendants it imports next at the
        row this one actually resolved to. ``written`` is False both when this
        store already holds that exact ``id@v=N`` and when it already holds the
        same computation under another id, so a repeated import is a no-op
        rather than a duplicate or an error.

        The preserved version is the whole point. Lineage edges are recorded as
        ``id@v=N`` strings, so a copy that let the destination assign a fresh
        version would land the ancestors under numbers the descendants' edges
        do not name — an imported graph that resolves to nothing. ``create_``
        ``artifact`` takes ``MAX(version)+1`` by design and cannot be used here.

        Deliberately not a merge: the row is written as it stands in the source,
        including ``provenance_hash``, ``principal`` and ``created_at``. An
        artifact copied into a served store has to keep saying who computed it
        and when, or publishing would quietly relabel someone else's work as
        freshly made here.

        The id is not the only way this store can already hold the record. A
        provenance hash does not carry the cell id, so two people whose
        notebooks ran the identical cell produce the same hash under different
        notebook-derived ids (``nb_<notebook>_cell_<cell>_var_<name>``), and
        ``idx_tenant_provenance_unique`` permits one ready row per
        ``(tenant, provenance_hash)``. Importing the second raised a bare
        ``IntegrityError`` — reachable today by publishing two notebooks that
        share an upstream cell into one served store. It resolves to the row
        already here instead: same computation, same bytes, so the caller's
        descendants can name it and the chain still resolves. Skipping the
        insert *without* saying so would be the worse failure, since the
        descendants' edges would name an id this store never received.

        ``finalize_artifact`` meets the same index and resolves it the other
        way, superseding the older row so the newcomer takes ``ready``. That is
        right for a fresh local computation and wrong here: the row already in
        a served store may be published or named, and knocking it out of
        ``ready`` to make room for a copy of itself would break the page it
        serves.

        Bytes before the row, and the row alone under the write lock. The row is
        what makes a version readable, so a crash between the two has to leave
        bytes with no row — invisible, and collectable — rather than a row with
        no bytes, which is a ready artifact whose page cannot serve it *and*
        which the ``(id, version)`` check then treats as a finished import
        forever, so no retry can repair it. The unlocked pre-check is only to
        avoid rewriting bytes for an import that turns out to be a no-op; the
        check inside the transaction is the authoritative one.
        """
        conn = self._get_connection()
        try:
            no_op = self._import_no_op(conn, record)
        finally:
            conn.close()
        if no_op is not None:
            self._complete_imported(no_op, record, blob)
            return no_op

        if blob is not None:
            with self.open_blob_writer(record.id, record.version) as writer:
                writer.write(blob)

        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, record.id)
            no_op = self._import_no_op(conn, record)
            if no_op is not None:
                conn.commit()
                return no_op

            conn.execute(
                """
                INSERT INTO artifact_versions
                    (id, version, state, provenance_hash, schema_json, row_count,
                     byte_size, created_at, transform_spec, input_versions,
                     tenant, principal, content_sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    or (hashlib.sha256(blob).hexdigest() if blob is not None else None),
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
        row_count: int,
        byte_size: int,
        content_sha256: str | None = None,
    ) -> ArtifactVersion | None:
        """Mark artifact as ready after blob is written.

        This is idempotent: if the same (tenant, provenance_hash) already exists
        in ready state, returns the existing artifact instead of raising an error.
        This enables safe retries after crashes or network failures.

        Args:
            artifact_id: Artifact ID
            version: Version number
            schema_json: Arrow schema as JSON
            row_count: Number of rows
            byte_size: Size of blob in bytes
            content_sha256: Digest of the bytes, when the caller already has
                them. Omitted, the blob is streamed and hashed here — correct
                either way, but a caller that just wrote the bytes has the
                digest for free and a remote blob store would otherwise be
                read a second time.

        Returns:
            The finalized ArtifactVersion, or the existing artifact if duplicate

        Raises:
            ValueError: If artifact not found or not in "building" state
        """
        conn = self._get_connection()
        try:
            # Get the artifact being finalized to check tenant/provenance
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
                # Already finalized - return it (idempotent)
                return self.get_artifact(artifact_id, version)

            if row["state"] != "building":
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} not in building state "
                    f"(state={row['state']})"
                )

            provenance_hash = row["provenance_hash"]
            tenant = row["tenant"]

            # Check if another artifact with same (tenant, provenance_hash) already exists
            # This handles the race condition where two builds complete simultaneously
            existing = self.find_by_provenance(provenance_hash, tenant=tenant)
            if existing is not None and existing.id != artifact_id:
                # Another artifact with same provenance already exists
                # Mark this one as failed (duplicate) and return the existing one
                conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'failed'
                    WHERE id = ? AND version = ?
                    """,
                    (artifact_id, version),
                )
                conn.commit()
                return existing

            if existing is not None and existing.version != version:
                # Same artifact id, older ready version with the same provenance:
                # this is a refresh rebuild. Supersede the old version so the
                # rebuild becomes canonical — the partial unique index allows
                # only ONE ready row per (tenant, provenance_hash). The old
                # version stays fetchable by explicit (id, version); only
                # provenance/dedup lookups stop returning it.
                conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'superseded'
                    WHERE id = ? AND version = ? AND state = 'ready'
                    """,
                    (existing.id, existing.version),
                )

            # Recorded here rather than at write time because every write
            # path arrives here — bytes in hand, a streamed writer, a file
            # published from disk, an import — and this is the moment the
            # artifact becomes readable, so it is the moment its bytes are
            # final.
            digest = content_sha256 or self.blob_digest(artifact_id, version)

            # Proceed with finalization
            try:
                cursor = conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'ready', schema_json = ?, row_count = ?, byte_size = ?,
                        content_sha256 = ?
                    WHERE id = ? AND version = ? AND state = 'building'
                    """,
                    (schema_json, row_count, byte_size, digest, artifact_id, version),
                )
                if cursor.rowcount == 0:
                    # Race condition: another process may have finalized
                    conn.rollback()
                    return self.get_artifact(artifact_id, version)
                conn.commit()
                return self.get_artifact(artifact_id, version)
            except self._dialect.integrity_error:
                # Unique constraint violation - duplicate (tenant, provenance_hash)
                # Another artifact was finalized first, return it
                conn.rollback()
                existing = self.find_by_provenance(provenance_hash, tenant=tenant)
                if existing is not None:
                    # Mark this one as failed
                    conn.execute(
                        """
                        UPDATE artifact_versions
                        SET state = 'failed'
                        WHERE id = ? AND version = ?
                        """,
                        (artifact_id, version),
                    )
                    conn.commit()
                    return existing
                raise
        finally:
            conn.close()

    def find_version_by_provenance(
        self, artifact_id: str, provenance_hash: str, tenant: str | None = None
    ) -> ArtifactVersion | None:
        """Return this artifact's newest version carrying *provenance_hash*.

        Unlike :meth:`find_by_provenance`, which answers "does any artifact
        anywhere hold this result", this stays inside one id — the question a
        caller asks when it wants *this* artifact's own history. Superseded
        versions count: only one row per (tenant, provenance_hash) may be
        ready at a time, so a result that was once current and has since been
        overtaken is superseded rather than deleted, and it is exactly the row
        worth finding again when the caller returns to that provenance.

        Tenant-scoped for the same reason :meth:`find_by_provenance` is: an id
        is not an isolation boundary, and a tenantless lookup must not select a
        row another tenant wrote. Tenantless rows are stored as ``''``.
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

        Callers resolve an artifact's current value through
        :meth:`get_latest_version`, so "latest" *is* the value — an older
        version holding the right bytes is not reachable by simply pointing at
        it. This writes a fresh version carrying the same provenance, spec and
        blob, which :meth:`finalize_artifact` then makes canonical, superseding
        the row it was copied from.

        The end state is byte-identical to recomputing the result, because
        recomputing writes a new version and supersedes the old one in exactly
        the same way. What it saves is the computing.

        Returns ``None`` when the source version or its blob is gone (a GC pass
        may have taken it), leaving the caller to fall back to recomputation.
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
        reader_cm = self.blob_store.open_blob_reader(artifact_id, version)
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

        # Streamed, not read_blob/write_blob: an artifact is as large as the
        # value a cell produced, and pulling a multi-GB frame through process
        # memory — a full download and re-upload on the object-store backends —
        # to re-point a pointer would cost more than the run this avoids.
        from strata.blob_store import BLOB_STREAM_CHUNK_BYTES

        copied = 0
        with (
            reader_cm as reader,
            self.blob_store.open_blob_writer(artifact_id, new_version) as writer,
        ):
            # Hashed as it goes by: the bytes are passing through anyway, and
            # a copy is exactly the operation where reading them a second time
            # to find their digest would be silly.
            hasher = hashlib.sha256()
            while chunk := reader.read(BLOB_STREAM_CHUNK_BYTES):
                writer.write(chunk)
                hasher.update(chunk)
                copied += len(chunk)

        schema_json = source.schema_json or ""
        row_count = source.row_count or 0
        byte_size = source.byte_size or copied
        finalized = self.finalize_artifact(
            artifact_id=artifact_id,
            version=new_version,
            schema_json=schema_json,
            row_count=row_count,
            byte_size=byte_size,
            content_sha256=hasher.hexdigest(),
        )
        if finalized is not None and finalized.id != artifact_id:
            # Dedup sent us to another artifact holding the same provenance —
            # two notebooks running the same cell source is enough. The caller
            # asked to make *this* id current, and resolves by id, so the
            # equivalent artifact under a different id is no answer. Same
            # recovery store_cell_output uses on the write path.
            return self.force_finalize_canonical(
                artifact_id=artifact_id,
                version=new_version,
                schema_json=schema_json,
                row_count=row_count,
                byte_size=byte_size,
            )
        return finalized

    def force_finalize_canonical(
        self,
        artifact_id: str,
        version: int,
        schema_json: str,
        row_count: int,
        byte_size: int,
    ) -> ArtifactVersion | None:
        """Promote a dedup-failed version to ready under its canonical id.

        ``finalize_artifact`` deduplicates by ``(tenant, provenance_hash)``:
        if another artifact with the same provenance already exists in
        ready state, the new version is marked ``failed`` and the
        existing artifact is returned. That's correct for most callers
        — they only care that *some* ready artifact with the right
        provenance is reachable.

        Notebook cells, however, resolve inputs by the canonical
        artifact id (``nb_{notebook_id}_cell_{cell_id}_var_{name}``).
        A dedup-failed canonical version leaves downstream cells
        unable to find the artifact under that id, even though an
        equivalent artifact exists under a different id. This method
        flips the canonical version back to ready so it's reachable
        by id.

        The partial unique index on ``(tenant, provenance_hash) WHERE
        state='ready'`` permits only one ready row per (tenant,
        provenance). So before promoting, any *other* ready row sharing
        this provenance is superseded — it stays fetchable by explicit
        id+version, just excluded from provenance lookups — and the
        canonical row becomes the single ready one. (Earlier this relied
        on SQLite treating NULL tenants as distinct, which is exactly the
        duplicate-row bug the tenant normalization closed.)

        Returns the canonical ``ArtifactVersion`` after promotion, or
        ``None`` if it can't be found post-update.
        """
        # Retried, because a conflict here means the work is still to be done.
        #
        # Under SQLite a conflict was impossible: BEGIN IMMEDIATE locks the
        # whole file, so no second writer was ever in flight. A keyed advisory
        # lock is narrower -- it serializes writers sharing an artifact id, and
        # rows sharing a provenance hash across *different* ids do not contend
        # on it -- so a competing writer can commit its own ready row between
        # our supersede and our promote, and the uniqueness index rejects us.
        #
        # Returning the winner would be wrong. The only caller reaches this
        # method precisely *because* finalize landed under a foreign id
        # (`notebook/artifact_integration.py`), so handing that same foreign id
        # back leaves the canonical row 'failed' and the caller none the wiser.
        # Retrying does what the method is for: the next pass sees the winner's
        # now-committed row, supersedes it, and promotes ours.
        for attempt in range(_CANONICAL_PROMOTE_ATTEMPTS):
            conn = self._get_connection()
            try:
                self._dialect.begin_write(conn, artifact_id)
                row = conn.execute(
                    "SELECT provenance_hash, tenant FROM artifact_versions "
                    "WHERE id = ? AND version = ?",
                    (artifact_id, version),
                ).fetchone()
                if row is not None:
                    # Supersede any other ready row with the same provenance in
                    # the same tenant so the canonical row can be promoted
                    # without violating the uniqueness index.
                    conn.execute(
                        """
                        UPDATE artifact_versions SET state = 'superseded'
                        WHERE provenance_hash = ? AND state = 'ready'
                          AND COALESCE(tenant, '') = COALESCE(?, '')
                          AND NOT (id = ? AND version = ?)
                        """,
                        (row["provenance_hash"], row["tenant"], artifact_id, version),
                    )
                conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'ready',
                        schema_json = ?,
                        row_count = ?,
                        byte_size = ?
                    WHERE id = ? AND version = ? AND state = 'failed'
                    """,
                    (schema_json, row_count, byte_size, artifact_id, version),
                )
                conn.commit()
                break
            except self._dialect.integrity_error:
                conn.rollback()
                # Out of attempts: let it surface rather than report a
                # promotion that did not happen.
                if attempt == _CANONICAL_PROMOTE_ATTEMPTS - 1:
                    raise
            finally:
                conn.close()
        return self.get_artifact(artifact_id, version)

    def finalize_and_set_name(
        self,
        artifact_id: str,
        version: int,
        schema_json: str,
        row_count: int,
        byte_size: int,
        name: str | None = None,
        tenant: str | None = None,
    ) -> ArtifactVersion | None:
        """Atomically finalize artifact and set name pointer in one transaction.

        This ensures the name pointer is only updated after the artifact is
        fully persisted and metadata committed. If the artifact is a duplicate
        (same provenance already exists), the name is pointed to the existing
        artifact instead.

        Args:
            artifact_id: Artifact ID
            version: Version number
            schema_json: Arrow schema as JSON
            row_count: Number of rows
            byte_size: Size of blob in bytes
            name: Optional name to set (if None, no name is set)
            tenant: Tenant ID for the name (if setting name)

        Returns:
            The finalized ArtifactVersion (or existing duplicate)

        Raises:
            ValueError: If artifact not found or not in "building" state
        """
        conn = self._get_connection()
        try:
            # Get the artifact being finalized
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
                # Already finalized - set name and return (idempotent)
                if name:
                    artifact_tenant = row["tenant"] if row["tenant"] else None
                    if not self._can_assign_name_for_tenant(artifact_tenant, tenant):
                        raise ValueError(
                            f"Artifact {artifact_id}@v={version} belongs to tenant "
                            f"{artifact_tenant}, cannot assign name in tenant {tenant}"
                        )
                    self._set_name_in_connection(conn, name, artifact_id, version, tenant)
                    conn.commit()
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
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} belongs to tenant "
                    f"{normalized_artifact_tenant}, cannot assign name in tenant {tenant}"
                )

            # Check if another artifact with same (tenant, provenance_hash) already exists
            existing = self.find_by_provenance(provenance_hash, tenant=artifact_tenant)
            if existing is not None and existing.id != artifact_id:
                # Another artifact with same provenance already exists
                # Mark this one as failed and point name to existing
                conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'failed'
                    WHERE id = ? AND version = ?
                    """,
                    (artifact_id, version),
                )
                if name:
                    self._set_name_in_connection(conn, name, existing.id, existing.version, tenant)
                conn.commit()
                return existing

            if existing is not None and existing.version != version:
                # Refresh rebuild: same artifact id, older ready version with
                # the same provenance. Supersede the old version (still
                # fetchable by explicit id+version; excluded from provenance
                # lookups) so the rebuild becomes canonical (#123).
                conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'superseded'
                    WHERE id = ? AND version = ? AND state = 'ready'
                    """,
                    (existing.id, existing.version),
                )

            # Atomically finalize and set name
            try:
                cursor = conn.execute(
                    """
                    UPDATE artifact_versions
                    SET state = 'ready', schema_json = ?, row_count = ?, byte_size = ?
                    WHERE id = ? AND version = ? AND state = 'building'
                    """,
                    (schema_json, row_count, byte_size, artifact_id, version),
                )
                if cursor.rowcount == 0:
                    # Race condition
                    conn.rollback()
                    artifact = self.get_artifact(artifact_id, version)
                    if artifact and artifact.state == "ready" and name:
                        # Still set the name
                        self.set_name(name, artifact_id, version, tenant)
                    return artifact

                # Set name in same transaction
                if name:
                    self._set_name_in_connection(conn, name, artifact_id, version, tenant)

                conn.commit()
                return self.get_artifact(artifact_id, version)

            except self._dialect.integrity_error:
                # Unique constraint violation - duplicate provenance
                conn.rollback()
                existing = self.find_by_provenance(provenance_hash, tenant=artifact_tenant)
                if existing is not None:
                    # Mark this one as failed, point name to existing
                    conn.execute(
                        """
                        UPDATE artifact_versions
                        SET state = 'failed'
                        WHERE id = ? AND version = ?
                        """,
                        (artifact_id, version),
                    )
                    if name:
                        self._set_name_in_connection(
                            conn, name, existing.id, existing.version, tenant
                        )
                    conn.commit()
                    return existing
                raise
        finally:
            conn.close()

    @staticmethod
    def _audit_in_connection(
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

        The audit commits or rolls back with the mutation it records — a
        mutation can never land unaudited, and a failed mutation leaves no
        phantom audit row.
        """
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
        """Set name within an existing connection (for use in transactions)."""
        # Use '' instead of NULL for personal mode (SQLite NULL != NULL in unique constraints)
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

    def fail_artifact(self, artifact_id: str, version: int) -> None:
        """Mark artifact as failed.

        Args:
            artifact_id: Artifact ID
            version: Version number
        """
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
        """Get artifact version metadata.

        Args:
            artifact_id: Artifact ID
            version: Version number

        Returns:
            ArtifactVersion or None if not found
        """
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

    def get_latest_version(self, artifact_id: str) -> ArtifactVersion | None:
        """Get the latest ready version of an artifact.

        Args:
            artifact_id: Artifact ID

        Returns:
            Latest ArtifactVersion with state="ready", or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, version, state, provenance_hash, schema_json,
                       row_count, byte_size, created_at, transform_spec,
                       input_versions, tenant, principal, content_sha256
                FROM artifact_versions
                WHERE id = ? AND state = 'ready'
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
        """Return the latest ready version of each artifact whose id starts with ``prefix``.

        Used by the notebook layer to enumerate loop-cell iteration
        artifacts (ids of the form ``nb_..._var_<name>@iter=<k>``) and
        by any future "list all outputs of this cell" surfaces.

        Results are sorted by artifact id so callers can reliably parse
        a numeric suffix (iteration index) in order.
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
                WHERE id LIKE ? ESCAPE '\\' AND state = 'ready'
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
        """Find artifact by provenance hash (for deduplication).

        Args:
            provenance_hash: Provenance hash to look up
            tenant: Tenant scope. A tenant id returns only that tenant's
                artifacts; ``None`` or ``""`` scopes to the *tenantless*
                namespace (``''`` going forward, ``NULL`` for not-yet-migrated
                rows) — never "any tenant", so a tenantless build cannot dedup
                against, and then point a name at, another tenant's artifact.

        Returns:
            Matching ArtifactVersion with state="ready", or None if not found
        """
        conn = self._get_connection()
        try:
            if tenant:
                cursor = conn.execute(
                    """
                    SELECT id, version, state, provenance_hash, schema_json,
                           row_count, byte_size, created_at, transform_spec,
                           input_versions, tenant, principal, content_sha256
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
                           input_versions, tenant, principal, content_sha256
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

    # -----------------------------------------------------------------------
    # Blob I/O
    # -----------------------------------------------------------------------

    def write_blob(self, artifact_id: str, version: int, data: bytes) -> None:
        """Write artifact blob to storage.

        Delegates to the configured blob store backend (local, S3, etc.).

        Args:
            artifact_id: Artifact ID
            version: Version number
            data: Arrow IPC stream bytes
        """
        self.blob_store.write_blob(artifact_id, version, data)

    def read_blob(self, artifact_id: str, version: int) -> bytes | None:
        """Read artifact blob from storage.

        Delegates to the configured blob store backend (local, S3, etc.).

        Args:
            artifact_id: Artifact ID
            version: Version number

        Returns:
            Arrow IPC stream bytes, or None if not found
        """
        return self.blob_store.read_blob(artifact_id, version)

    def open_blob_reader(self, artifact_id: str, version: int):
        """Open a streaming reader for an artifact blob.

        Returns ``None`` if the blob does not exist. Otherwise returns a
        context manager yielding a binary file-like object.
        """
        return self.blob_store.open_blob_reader(artifact_id, version)

    def open_blob_writer(self, artifact_id: str, version: int):
        """Open a streaming writer for an artifact blob.

        Returns a context manager yielding a binary file-like object.
        Commits atomically on clean exit; discards on exception.
        """
        return self.blob_store.open_blob_writer(artifact_id, version)

    def blob_size(self, artifact_id: str, version: int) -> int | None:
        """Return the size of an artifact blob without materializing it."""
        return self.blob_store.blob_size(artifact_id, version)

    def publish_blob_from_path(self, artifact_id: str, version: int, source_path: Path) -> None:
        """Atomically publish an artifact blob from a prepared local file.

        Intended to be invoked via ``asyncio.to_thread`` so the potentially
        blocking remote publish does not tie up the event loop.
        """
        self.blob_store.publish_blob_from_path(artifact_id, version, source_path)

    def blob_exists(self, artifact_id: str, version: int) -> bool:
        """Check if blob exists in storage.

        Delegates to the configured blob store backend (local, S3, etc.).

        Args:
            artifact_id: Artifact ID
            version: Version number

        Returns:
            True if blob exists
        """
        return self.blob_store.blob_exists(artifact_id, version)

    # -----------------------------------------------------------------------
    # Name Pointers
    # -----------------------------------------------------------------------

    @staticmethod
    def _can_assign_name_for_tenant(
        artifact_tenant: str | None,
        requested_tenant: str | None,
    ) -> bool:
        """Return whether a name in requested_tenant may point at artifact_tenant.

        Tenantless artifacts remain assignable for backwards compatibility with
        older personal-mode behavior and existing tests. Artifacts stamped with
        the legacy "_default" tenant (pre-#126 PUT uploads) are assignable by
        tenantless requests: that combination only occurs in single-tenant
        deployments, where tenant isolation is not in play — in multi-tenant
        mode un-headered requests resolve to "_default" themselves and pass
        the equality check.
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
        """Create or update a name pointer.

        In multi-tenant mode, names are scoped by tenant. The unique key is
        (tenant, name), so different tenants can have the same name.

        Args:
            name: Human-readable name
            artifact_id: Target artifact ID
            version: Target version
            tenant: Optional tenant ID for multi-tenant isolation

        Raises:
            ValueError: If target artifact version doesn't exist or isn't ready
        """
        conn = self._get_connection()
        try:
            # Verify target exists and is ready
            cursor = conn.execute(
                """
                SELECT state, tenant FROM artifact_versions
                WHERE id = ? AND version = ?
                """,
                (artifact_id, version),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Artifact {artifact_id}@v={version} not found")
            if row["state"] != "ready":
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} is not ready (state={row['state']})"
                )
            artifact_tenant = row["tenant"] if row["tenant"] else None
            if not self._can_assign_name_for_tenant(artifact_tenant, tenant):
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} belongs to tenant "
                    f"{artifact_tenant}, cannot assign name in tenant {tenant}"
                )

            # Upsert + audit in one transaction (shared with finalize paths)
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
        """Resolve a name to its artifact version.

        In multi-tenant mode, names are scoped by tenant.

        Args:
            name: Name to resolve
            tenant: Optional tenant filter for multi-tenant isolation

        Returns:
            The pinned ArtifactVersion, or None if name not found
        """
        conn = self._get_connection()
        try:
            # Use '' instead of NULL for personal mode
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
        """Get name pointer metadata.

        Args:
            name: Name to look up
            tenant: Optional tenant filter for multi-tenant isolation

        Returns:
            ArtifactName or None if not found
        """
        conn = self._get_connection()
        try:
            # Use '' instead of NULL for personal mode
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
            # Convert '' back to None for API consistency
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
        """Delete a name pointer.

        Args:
            name: Name to delete
            tenant: Optional tenant filter for multi-tenant isolation

        Returns:
            True if name was deleted, False if it didn't exist
        """
        conn = self._get_connection()
        try:
            # Use '' instead of NULL for personal mode
            effective_tenant = tenant if tenant is not None else ""
            cursor = conn.execute(
                "SELECT version FROM artifact_names WHERE name = ? AND tenant = ?",
                (name, effective_tenant),
            )
            previous = cursor.fetchone()
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
        """List name pointers across ALL tenants.

        Maintenance/CLI surface: a store inspector must see names whatever
        tenant spelling wrote them ('' for personal, '_default' for legacy
        PUT uploads, real ids in multi-tenant mode). Request-serving code
        should use :meth:`list_names` with the caller's tenant instead.
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

    # ------------------------------------------------------------------
    # Registry: aliases, tags, audit (#129)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Publications — opt-in public read grants
    # ------------------------------------------------------------------

    @staticmethod
    def _publication_from_row(row) -> Publication:
        return Publication(
            token=row["token"],
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

        Rows written before the column existed have none. Reading every blob in
        the store to backfill them would have made the migration proportional
        to the store's size; filling one when something actually asks makes it
        proportional to what is asked for, and the answer is the same.

        ``None`` when there is no blob to hash, which stays ``None``: there is
        nothing to record and asking again is cheap.
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

    def blob_digest(self, artifact_id: str, version: int) -> str | None:
        """SHA-256 of an artifact's bytes, streamed. ``None`` if there is no blob.

        Streamed rather than read whole: a published artifact can be a table of
        any size, and publishing must not be the operation that decides how
        much memory the server needs.
        """
        reader_cm = self.open_blob_reader(artifact_id, version)
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

        Publishing the same version twice returns the existing active grant
        rather than minting a second token. Two live URLs for one artifact
        would mean revoking one and believing the artifact was withdrawn.

        Raises:
            ValueError: If the artifact does not exist, is not readable, or
                belongs to a different tenant.
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

            publication = Publication(
                # The version's own digest, not a second computation of the
                # same bytes: a publication that disagreed with the artifact it
                # names would be the more alarming of the two answers. Read
                # from the row already in hand rather than through
                # ``content_digest``, which would open a second connection and
                # write through it while this one holds the publication.
                content_sha256=row["content_sha256"] or self.blob_digest(artifact_id, version),
                token=secrets.token_urlsafe(32),
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
                    publication.token,
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
            conn.commit()
            return publication
        finally:
            conn.close()

    def get_publication(self, token: str) -> Publication | None:
        """Look up a publication by token, revoked ones included.

        Revoked grants are returned rather than hidden so the caller can
        answer "this citation was withdrawn" instead of "no such page" —
        different facts, and a reader chasing a footnote deserves the first.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM artifact_publications WHERE token = ?", (token,)
            ).fetchone()
            return self._publication_from_row(row) if row is not None else None
        finally:
            conn.close()

    def update_publication_credits(
        self,
        token: str,
        *,
        tenant: str | None = None,
        authors: list[dict[str, str]] | None = None,
        external_ids: list[dict[str, str]] | None = None,
    ) -> Publication | None:
        """Set who wrote a publication and what identifies it, after the fact.

        A DOI is registered against a deposit that already has to be
        reachable, so the identifier almost always arrives after the token
        does. What this cannot touch is the binding: ``artifact_id`` and
        ``version`` are not parameters here, and no SQL below names them — a
        citation whose target could be repointed would be worthless, and that
        has to stay true of every write path and not only the obvious one.

        ``None`` for either argument leaves that column alone; an empty list
        clears it. Returns the updated publication, or ``None`` if the token is
        unknown in this tenant.
        """
        effective_tenant = tenant if tenant is not None else ""
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
                    (*params, token, effective_tenant),
                )
                conn.commit()
                if cursor.rowcount == 0:
                    return None
            row = conn.execute(
                "SELECT * FROM artifact_publications WHERE token = ? AND tenant = ?",
                (token, effective_tenant),
            ).fetchone()
            return self._publication_from_row(row) if row is not None else None
        finally:
            conn.close()

    def revoke_publication(self, token: str, tenant: str | None = None) -> bool:
        """Withdraw a grant. Returns False if it was unknown or already gone."""
        effective_tenant = tenant if tenant is not None else ""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "UPDATE artifact_publications SET revoked_at = ? "
                "WHERE token = ? AND tenant = ? AND revoked_at IS NULL",
                (time.time(), token, effective_tenant),
            )
            conn.commit()
            return cursor.rowcount > 0
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

        Returns:
            True when the pointer changed; False when the alias already
            pointed at exactly this version (no write, no audit entry —
            re-running an idempotent promote cell must not spam history).

        Raises:
            ValueError: If the artifact doesn't exist, isn't readable, or
                belongs to a different tenant.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT state, tenant FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Artifact {artifact_id}@v={version} not found")
            if row["state"] not in ("ready", "superseded"):
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} is not readable (state={row['state']})"
                )
            artifact_tenant = row["tenant"] if row["tenant"] else None
            if not self._can_assign_name_for_tenant(artifact_tenant, tenant):
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} belongs to tenant "
                    f"{artifact_tenant}, cannot assign alias in tenant {tenant}"
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
                return False  # Already points here — idempotent no-op
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
        """List aliases — for one name, or the whole store when name is None."""
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
            ValueError: If the artifact doesn't exist, isn't readable, or
                belongs to a different tenant — mirrors ``set_alias`` so a
                tenant can't attach metadata to another tenant's artifact by
                guessing its id/version.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT state, tenant FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Artifact {artifact_id}@v={version} not found")
            if row["state"] not in ("ready", "superseded"):
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} is not readable (state={row['state']})"
                )
            artifact_tenant = row["tenant"] if row["tenant"] else None
            if not self._can_assign_name_for_tenant(artifact_tenant, tenant):
                raise ValueError(
                    f"Artifact {artifact_id}@v={version} belongs to tenant "
                    f"{artifact_tenant}, cannot tag in tenant {tenant}"
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
        """Return ``(artifact_id, version)`` for every artifact carrying the
        given ``key=value`` tag. Used to find artifacts a notebook cell
        published (``nb_cell=<cell_id>``)."""
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
        """Return ``(artifact_id, version, value)`` for every artifact carrying
        the given tag key, whatever its value.

        The sibling above answers "which artifacts did cell c1 publish". This
        answers "which artifacts did any cell publish, and which cell" — one
        query for a whole notebook's strip instead of one per cell, which over
        a remote store is the difference between one round trip and fifty."""
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
        """Read registry audit entries, newest first.

        Filters are ANDed when given. ``tenant`` defaults to the
        ALL_TENANTS sentinel — the direct-store CLI and admin views read
        the whole store's compliance record. Request-serving routes MUST
        pass the caller's tenant (a string, or ``None`` for the default
        tenant) so a tenant cannot read another tenant's audit history.
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

        ``action`` is ``"set"`` (requires artifact_id + version, validated
        like set_alias) or ``"delete"``. One pending change per alias —
        a new request replaces the previous one.

        Returns:
            True when a change was queued; False when the alias already
            points at exactly the requested target (idempotent no-op — a
            re-run promote cell must not refile an approved promotion).
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
                if row is None:
                    raise ValueError(f"Artifact {artifact_id}@v={version} not found")
                if row["state"] not in ("ready", "superseded"):
                    raise ValueError(
                        f"Artifact {artifact_id}@v={version} is not readable (state={row['state']})"
                    )
                artifact_tenant = row["tenant"] if row["tenant"] else None
                if not self._can_assign_name_for_tenant(artifact_tenant, tenant):
                    raise ValueError(
                        f"Artifact {artifact_id}@v={version} belongs to tenant "
                        f"{artifact_tenant}, cannot assign alias in tenant {tenant}"
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
                    return False  # Already the live pointer — nothing to approve
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
        """Apply a pending alias change (audited with the approver as actor).

        ``require_distinct_approver`` enforces separation of duty: the
        approver may not be the principal who requested the change. The
        route sets this for non-superadmin callers so a requester cannot
        self-approve their own protected-alias move.

        Returns the applied pending entry.

        Raises:
            ValueError: If no pending change exists for ``name @ alias``,
                the target artifact is no longer available, or the
                approver is the requester under separation-of-duty.
        """
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
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
            ValueError: If no pending change exists for ``name @ alias``.
        """
        conn = self._get_connection()
        try:
            effective_tenant = tenant if tenant is not None else ""
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
        """List all name pointers.

        Args:
            tenant: Optional tenant filter for multi-tenant isolation

        Returns:
            List of ArtifactName entries (filtered by tenant if provided)
        """
        conn = self._get_connection()
        try:
            # Use '' instead of NULL for personal mode
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
                    # Convert '' back to None for API consistency
                    tenant=row["tenant"] if row["tenant"] else None,
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_name_status(self, name: str, tenant: str | None = None) -> NameStatus | None:
        """Get status information for a named artifact.

        Returns the name pointer metadata along with the artifact's
        input_versions for staleness checking. The caller is responsible
        for comparing input_versions against current versions.

        Args:
            name: Name to look up
            tenant: Optional tenant filter for multi-tenant isolation

        Returns:
            NameStatus with artifact metadata and input versions, or None
        """
        name_info = self.get_name(name, tenant=tenant)
        if name_info is None:
            return None

        artifact = self.get_artifact(name_info.artifact_id, name_info.version)
        if artifact is None:
            return None

        # Parse input_versions from JSON
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

    # -----------------------------------------------------------------------
    # Lineage and Dependency Queries
    # -----------------------------------------------------------------------

    def find_dependents(
        self,
        artifact_id: str,
        version: int,
        tenant: str | None = None,
    ) -> list[tuple[ArtifactVersion, str]]:
        """Find artifacts that depend on a given artifact version.

        Searches all artifacts whose input_versions contain a reference to
        the specified artifact. Returns the dependent artifacts along with
        the version string they used for the dependency.

        Args:
            artifact_id: Artifact ID to search for dependents of
            version: Version number to search for
            tenant: Optional tenant filter for multi-tenant isolation

        Returns:
            List of (ArtifactVersion, input_version_string) tuples for dependents
        """
        # Build the search pattern - artifacts reference inputs as "artifact_id@v=N"
        search_pattern = f'"{artifact_id}@v={version}"'
        # Also search for the full URI format
        uri_pattern = f'"strata://artifact/{artifact_id}@v={version}"'

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
                      AND (input_versions LIKE ? OR input_versions LIKE ?)
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
                      AND (input_versions LIKE ? OR input_versions LIKE ?)
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

                # Parse input_versions to find the exact version string used
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

    def get_name_for_artifact(
        self,
        artifact_id: str,
        version: int,
        tenant: str | None = None,
    ) -> str | None:
        """Get the name pointing to a specific artifact version.

        Args:
            artifact_id: Artifact ID
            version: Version number
            tenant: Optional tenant filter

        Returns:
            Name string if found, None otherwise
        """
        # Use '' instead of NULL for personal mode (SQLite NULL != NULL in unique constraints)
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

    # -----------------------------------------------------------------------
    # Lifecycle Management
    # -----------------------------------------------------------------------

    # Whitelisted sort columns — ORDER BY can't be parameterized, so restrict it
    # to a known set to keep the query injection-safe.
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
        """List artifacts with optional filtering.

        Args:
            limit: Maximum number of artifacts to return
            offset: Number of artifacts to skip
            state: Filter by state ("ready", "building", "failed")
            name_prefix: Filter by artifacts that have a name starting with prefix
            tenant: Optional tenant filter. When provided, includes legacy
                tenantless artifacts for backwards compatibility.
            since: Only artifacts created at or after this epoch timestamp.
            sort: Sort column — one of ``created_at`` / ``byte_size`` /
                ``row_count`` (anything else falls back to ``created_at``).
            order: ``asc`` or ``desc`` (default ``desc``).

        Returns:
            List of ArtifactVersion entries
        """
        sort_col = self._SORT_COLUMNS.get(sort, "created_at")
        order_sql = "ASC" if str(order).lower() == "asc" else "DESC"
        conn = self._get_connection()
        try:
            if name_prefix is not None:
                # Join with names table to filter by name prefix
                query = """
                    SELECT DISTINCT av.id, av.version, av.state, av.provenance_hash,
                           av.schema_json, av.row_count, av.byte_size, av.created_at,
                           av.transform_spec, av.input_versions, av.tenant, av.principal,
                           av.content_sha256
                    FROM artifact_versions av
                    INNER JOIN artifact_names an
                        ON av.id = an.artifact_id AND av.version = an.version
                    WHERE an.name LIKE ?
                """
                params: list = [name_prefix + "%"]

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
        """Delete an artifact version and its blob.

        Also removes all name/alias/tag pointers to this version (the version is
        gone, so leaving pointers would dangle).

        Args:
            artifact_id: Artifact ID
            version: Version number
            tenant: When provided, the artifact must belong to this tenant (or be
                tenantless) — refuses to delete another tenant's artifact, so the
                metadata cascade can't be driven cross-tenant. Defense in depth
                mirroring the route's ownership check.

        Returns:
            True if artifact was deleted, False if it didn't exist or is owned
            by a different tenant
        """
        conn = self._get_connection()
        try:
            # Check existence + ownership
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
            # Past this point the delete is committed to; the blob cleanup
            # below the finally runs only for rows that actually existed.

            # Delete name pointers to this version
            conn.execute(
                "DELETE FROM artifact_names WHERE artifact_id = ? AND version = ?",
                (artifact_id, version),
            )

            # Delete alias pointers (audited — an alias disappearing because
            # its target was deleted must be reconstructible) and tags.
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

            # Delete metadata
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
        self.blob_store.delete_blob(artifact_id, version)
        return True

    def _delete_version_children(
        self, conn: StoreConnection, artifact_id: str, version: int
    ) -> None:
        """Remove the rows that reference *version*, before the version itself.

        Tags carry no foreign key but orphan just as readily; builds carry one
        and are the reason enforcement could not be switched on before. Build
        rows cascade rather than survive with a null pointer: ``artifact_id``
        and ``version`` are ``NOT NULL`` and are what a build row is *about*,
        so a build of a collected artifact has no subject left to describe.

        ``artifact_builds`` belongs to the build store, which shares this
        database — the foreign key crossing that line already couples them, and
        a store used without a build runner may not have the table at all.
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

    def _publication_reachable(self, conn: StoreConnection) -> set[tuple[str, int]]:
        """Every artifact version a publication depends on, roots included.

        A published page shows the code and environment of every step behind
        the result, so the chain is part of what was published: collecting an
        ancestor would leave a live token whose lineage resolves to nothing.

        Revoked publications count too. Their rows are kept so the token fails
        closed rather than being reissued, and the chain stays readable for
        audit; deleting it would make a withdrawal destroy the record of what
        was withdrawn.

        Not tenant-scoped, deliberately. Protecting more than a sweep would
        have collected is free, and the alternative — reasoning about whether a
        chain can cross tenants — is the kind of proof this store's pruning
        rules refuse to rely on elsewhere.
        """
        roots = conn.execute("SELECT artifact_id, version FROM artifact_publications").fetchall()

        reachable: set[tuple[str, int]] = set()
        pending = [(row["artifact_id"], row["version"]) for row in roots]
        while pending:
            node = pending.pop()
            if node in reachable:
                continue
            reachable.add(node)
            row = conn.execute(
                "SELECT input_versions FROM artifact_versions WHERE id = ? AND version = ?",
                node,
            ).fetchone()
            if row is None or not row["input_versions"]:
                continue
            for uri in json.loads(row["input_versions"]):
                # Parsed exactly as ``_walk_lineage`` parses it, so that what
                # GC protects and what the lineage walk will follow cannot
                # drift apart. Anything else is a table or an external leaf.
                if not uri.startswith("strata://artifact/"):
                    continue
                artifact_id, _, version = uri[len("strata://artifact/") :].partition("@v=")
                if not version.isdigit():
                    continue
                pending.append((artifact_id, int(version)))
        return reachable

    def garbage_collect(
        self,
        max_age_days: float = 7.0,
        tenant: str | None = None,
        collect_latest: bool = False,
    ) -> dict:
        """Delete unreachable artifacts older than max_age.

        An artifact version is reachable when a name or alias points at it, or
        when it is the latest version of its id — ``get_latest_version(id)`` is
        how the store resolves "the current value", and for some producers it
        is the only handle that ever exists (notebook cell outputs are stored
        as ``nb_…_var_…`` and never named). Only "ready", "superseded" or
        "failed" versions older than ``max_age_days`` are considered.

        Args:
            max_age_days: Maximum age in days for unreachable artifacts
            tenant: Optional tenant filter. When provided, includes legacy
                tenantless artifacts for backwards compatibility.
            collect_latest: Also collect current values — the latest version of
                an unnamed id. Off by default because it deletes live state
                (this is what made a routine GC wipe week-old notebooks); turn
                it on only for a store you are deliberately reclaiming, where
                "unnamed and old" really does mean garbage.

        Returns:
            Dictionary with GC statistics
        """
        conn = self._get_connection()
        try:
            cutoff = time.time() - (max_age_days * 86400)

            # Find unreachable artifacts older than cutoff.
            #
            # "Reachable" is deliberately wider than "has a name pointer":
            #
            # - a NAME or an ALIAS points at the version — aliases pin registry
            #   entries (e.g. champion on a superseded version), and collecting
            #   an aliased artifact would leave a dangling pointer;
            # - it is the LATEST version of its artifact id. ``get_latest_version(id)``
            #   is how the store resolves "the current value" and is the ONLY
            #   handle some producers ever use — notebook cell outputs are stored
            #   as ``nb_{notebook}_cell_{cell}_var_{name}`` and never given a name
            #   or alias, so the old rule classified every one of them as garbage.
            #   A GC run with the default 7-day cutoff deleted the live state of
            #   any notebook older than a week, and the next cell run found its
            #   upstream missing. Collecting superseded versions is still fine —
            #   that is what makes GC useful — but never the current one.
            #
            # Publications and everything behind them are excluded below,
            # after the SELECT, by a lineage walk rather than by a clause here.
            # ``input_versions`` is JSON in a TEXT column, so expressing the
            # walk in SQL means ``json_each`` on one dialect and ``jsonb_each``
            # on the other plus splitting ids on ``@v=``; the walk is bounded
            # by publications times chain depth and runs at most hourly, so it
            # is cheaper to keep it in one place in Python.
            #
            # Still NOT covered: artifacts reachable only through the lineage
            # of an *unpublished* artifact. Those keep the older justification
            # — the latest-version rule protects a pipeline's inputs, which are
            # the latest versions of their own ids. That argument never held
            # for a published chain, which is exactly a chain whose ancestors
            # are expected to be superseded while the published version stays.
            query = """
                SELECT av.id, av.version, av.byte_size
                FROM artifact_versions av
                LEFT JOIN artifact_names an ON av.id = an.artifact_id AND av.version = an.version
                LEFT JOIN artifact_aliases aa ON av.id = aa.artifact_id AND av.version = aa.version
                WHERE an.name IS NULL
                  AND aa.alias IS NULL
                  AND av.state IN ('ready', 'superseded', 'failed')
                  AND av.created_at < ?
            """
            if not collect_latest:
                query += """
                  AND av.version < (
                      SELECT MAX(latest.version)
                      FROM artifact_versions latest
                      WHERE latest.id = av.id
                  )
                """
            params: list[float | str] = [cutoff]
            if tenant is not None:
                query += " AND (av.tenant = ? OR av.tenant = '' OR av.tenant IS NULL)"
                params.append(tenant)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            protected = self._publication_reachable(conn)

            deleted_count = 0
            deleted_bytes = 0

            # Metadata first, then blobs — the same ordering ``delete_artifact``
            # uses. The reverse order left a window where a crash (or a raising
            # blob backend) mid-loop had already removed blobs while the
            # metadata DELETEs were still uncommitted and rolled back, leaving
            # rows in state 'ready' whose blob no longer exists: every later
            # read returns a ready artifact with no data, and verify_artifacts
            # reports it as missing_blob. Losing a blob whose row is gone is
            # merely wasted bytes; the reverse is a corrupt store.
            collected: list[tuple[str, int]] = []
            for row in rows:
                artifact_id, version, byte_size = row["id"], row["version"], row["byte_size"] or 0

                if (artifact_id, version) in protected:
                    continue

                self._delete_version_children(conn, artifact_id, version)
                try:
                    conn.execute(
                        "DELETE FROM artifact_versions WHERE id = ? AND version = ?",
                        (artifact_id, version),
                    )
                except self._dialect.integrity_error:
                    # A name or alias was pointed at this version between the
                    # SELECT above and here. Enforcement doing its job — the
                    # pointer wins — but one unlucky race must not fail an
                    # entire sweep, so skip this artifact and keep going. It is
                    # not counted and its blob is not touched.
                    logger.info(
                        "garbage_collect: skipping %s@v=%d, something referenced "
                        "it after it was selected.",
                        artifact_id,
                        version,
                    )
                    continue

                collected.append((artifact_id, version))
                deleted_count += 1
                deleted_bytes += byte_size

            conn.commit()
        finally:
            conn.close()

        # Best-effort blob cleanup after the metadata is durably gone. A
        # failure here only orphans bytes, so it must not abort the run.
        #
        # Outside the connection scope deliberately: a sweep deleting a few
        # thousand blobs at 50-200ms each against a remote blob store would
        # otherwise hold a pooled connection for minutes, and a handful of
        # concurrent sweeps would exhaust the pool and fail unrelated requests
        # with PoolTimeout. `collected` is already materialized, so nothing
        # here needs the database.
        for artifact_id, version in collected:
            try:
                self.blob_store.delete_blob(artifact_id, version)
            except Exception:
                logger.exception(
                    "garbage_collect: failed to delete blob for %s@v=%d "
                    "(metadata already removed; bytes orphaned)",
                    artifact_id,
                    version,
                )

        return {
            "deleted_count": deleted_count,
            "deleted_bytes": deleted_bytes,
            "cutoff_timestamp": cutoff,
        }

    def get_usage(self, tenant: str | None = None) -> dict:
        """Get artifact store usage statistics.

        Returns:
            Dictionary with usage metrics
        """
        conn = self._get_connection()
        try:
            usage_query = """
                SELECT
                    COUNT(DISTINCT id) as unique_artifacts,
                    COUNT(*) as total_versions,
                    COUNT(CASE WHEN state = 'ready' THEN 1 END) as ready_versions,
                    COUNT(CASE WHEN state = 'building' THEN 1 END) as building_versions,
                    COUNT(CASE WHEN state = 'failed' THEN 1 END) as failed_versions,
                    COALESCE(SUM(CASE WHEN state = 'ready' THEN byte_size END), 0) as total_bytes,
                    COALESCE(SUM(CASE WHEN state = 'ready' THEN row_count END), 0) as total_rows,
                    MIN(created_at) as oldest_artifact,
                    MAX(created_at) as newest_artifact
                FROM artifact_versions
            """
            usage_params: list[str] = []
            if tenant is not None:
                usage_query += " WHERE tenant = ? OR tenant = '' OR tenant IS NULL"
                usage_params.append(tenant)

            cursor = conn.execute(usage_query, usage_params)
            row = cursor.fetchone()

            if tenant is not None:
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM artifact_names WHERE tenant = ? OR tenant = ''",
                    (tenant,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as count FROM artifact_names")
            names_count = cursor.fetchone()["count"]

            # Count unreferenced artifacts
            unreferenced_query = """
                SELECT COUNT(*) as count
                FROM artifact_versions av
                LEFT JOIN artifact_names an ON av.id = an.artifact_id AND av.version = an.version
                WHERE an.name IS NULL AND av.state = 'ready'
            """
            unreferenced_params: list[str] = []
            if tenant is not None:
                unreferenced_query += " AND (av.tenant = ? OR av.tenant = '' OR av.tenant IS NULL)"
                unreferenced_params.append(tenant)
            cursor = conn.execute(unreferenced_query, unreferenced_params)
            unreferenced_count = cursor.fetchone()["count"]

            return {
                "unique_artifacts": row["unique_artifacts"],
                "total_versions": row["total_versions"],
                "ready_versions": row["ready_versions"],
                "building_versions": row["building_versions"],
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

    # -----------------------------------------------------------------------
    # Maintenance (Legacy - kept for backwards compatibility)
    # -----------------------------------------------------------------------

    def sweep_zombie_builds(self, max_age_seconds: float = 3600) -> int:
        """Mark stale ``building`` artifacts as failed.

        A materialize that queued into a mode with no executor (or whose
        builder crashed without cleanup) leaves an artifact in ``building``
        forever — it can never serve data, but it sits in the store
        indefinitely. Demoting to ``failed`` makes it visible as a failure
        and eligible for ``cleanup_failed`` deletion.

        STARTUP-ONLY CONTRACT: this sweep demotes ANY sufficiently old
        ``building`` row unconditionally, which is only safe when no build
        can legitimately be in flight — i.e. during server startup, before
        the build runner accepts work. Do not wire it to a periodic timer
        without adding an is-this-build-actually-running check.

        Args:
            max_age_seconds: Builds older than this are considered zombies.

        Returns:
            Number of artifacts demoted to failed.
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
        """Check every serveable artifact's blob against its metadata.

        For each ``ready`` or ``superseded`` artifact: the blob must exist,
        parse as exactly one Arrow IPC stream, and its readable row count
        must equal the recorded ``row_count``. This is the after-the-fact
        counterpart to finalize-time validation (#123) — it catches stores
        written before validation existed or corrupted out-of-band.

        Returns:
            One finding dict per problem:
            ``{"artifact_id", "version", "state", "problem", "detail"}``.
            An empty list means the store is consistent.
        """
        import pyarrow as pa

        from strata.fast_io import validate_ipc_stream

        conn = self._get_connection()
        try:
            query = """
                SELECT id, version, state, row_count, content_sha256
                FROM artifact_versions
                WHERE state IN ('ready', 'superseded')
            """
            params: list[str] = []
            if tenant is not None:
                query += " AND (tenant = ? OR tenant = '' OR tenant IS NULL)"
                params.append(tenant)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        findings: list[dict] = []
        for row in rows:
            artifact_id, version = row["id"], row["version"]
            data = self.blob_store.read_blob(artifact_id, version)
            if data is None:
                findings.append(
                    {
                        "artifact_id": artifact_id,
                        "version": version,
                        "state": row["state"],
                        "problem": "missing_blob",
                        "detail": "metadata row exists but blob is gone",
                    }
                )
                continue

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
                        "detail": f"metadata says {row['row_count']}, blob yields {readable_rows}",
                    }
                )

            # The checks above catch bytes that stopped being valid Arrow or
            # stopped holding the rows they claim. A digest catches the edit
            # that kept both true — a value changed in place, which is the
            # alteration a reader would never otherwise notice. Rows with no
            # digest are silent here: they predate the column, and verify is
            # not the place to decide the store's history was wrong.
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
        """Clean up failed artifacts older than max_age.

        Args:
            max_age_seconds: Max age of failed artifacts to keep (default 1 hour)

        Returns:
            Number of artifacts cleaned up
        """
        conn = self._get_connection()
        try:
            cutoff = time.time() - max_age_seconds
            cursor = conn.execute(
                """
                SELECT id, version FROM artifact_versions
                WHERE state = 'failed' AND created_at < ?
                """,
                (cutoff,),
            )
            rows = cursor.fetchall()

            # Metadata first, then blobs. The previous order deleted each blob
            # before its row was committed, so a failure mid-sweep left rows
            # pointing at bytes that were already gone. Orphaned bytes are the
            # better failure, and it is what garbage_collect already does.
            for row in rows:
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
                self.blob_store.delete_blob(row["id"], row["version"])
            except Exception:
                logger.exception(
                    "cleanup_failed: failed to delete blob for %s@v=%d "
                    "(metadata already removed; bytes orphaned)",
                    row["id"],
                    row["version"],
                )
        return len(rows)

    def stats(self, tenant: str | None = None) -> dict:
        """Get artifact store statistics.

        Returns:
            Dictionary with store statistics
        """
        conn = self._get_connection()
        try:
            stats_query = """
                SELECT
                    COUNT(*) as total_versions,
                    COUNT(CASE WHEN state = 'ready' THEN 1 END) as ready_versions,
                    COUNT(CASE WHEN state = 'building' THEN 1 END) as building_versions,
                    COUNT(CASE WHEN state = 'failed' THEN 1 END) as failed_versions,
                    COALESCE(SUM(CASE WHEN state = 'ready' THEN byte_size END), 0) as total_bytes,
                    COALESCE(SUM(CASE WHEN state = 'ready' THEN row_count END), 0) as total_rows
                FROM artifact_versions
            """
            stats_params: list[str] = []
            if tenant is not None:
                stats_query += " WHERE tenant = ? OR tenant = '' OR tenant IS NULL"
                stats_params.append(tenant)

            cursor = conn.execute(stats_query, stats_params)
            row = cursor.fetchone()

            if tenant is not None:
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM artifact_names WHERE tenant = ? OR tenant = ''",
                    (tenant,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as count FROM artifact_names")
            names_count = cursor.fetchone()["count"]

            return {
                "total_versions": row["total_versions"],
                "ready_versions": row["ready_versions"],
                "building_versions": row["building_versions"],
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


# ---------------------------------------------------------------------------
# Module-level singleton (initialized lazily)
# ---------------------------------------------------------------------------

_artifact_store: ArtifactStore | None = None


def get_artifact_store(
    artifact_dir: Path | None = None,
    blob_store: BlobStore | None = None,
    dialect: SqlDialect | None = None,
) -> ArtifactStore | None:
    """Get the artifact store singleton.

    Args:
        artifact_dir: Directory for artifacts (required on first call in personal mode)
        blob_store: Optional blob storage backend. If None, uses LocalBlobStore.
        dialect: Optional metadata backend. If None, SQLite under artifact_dir.
            Like blob_store, this only takes effect on the call that creates
            the singleton — the server passes both at lifespan start.

    Returns:
        ArtifactStore instance, or None if not in personal mode
    """
    global _artifact_store
    if _artifact_store is None and artifact_dir is not None:
        _artifact_store = ArtifactStore(artifact_dir, blob_store=blob_store, dialect=dialect)
    return _artifact_store


def reset_artifact_store() -> None:
    """Reset the artifact store singleton (for testing)."""
    global _artifact_store
    _artifact_store = None
