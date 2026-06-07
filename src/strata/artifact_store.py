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
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata.blob_store import BlobStore


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
    tenant TEXT,  -- Tenant ID for multi-tenant isolation
    principal TEXT,  -- Principal ID that created this artifact
    PRIMARY KEY (id, version)
);

-- Index for provenance lookup (deduplication)
CREATE INDEX IF NOT EXISTS idx_provenance ON artifact_versions(provenance_hash);

-- Unique constraint for idempotent finalize: (tenant, provenance_hash) for ready artifacts
-- This prevents duplicate artifacts for the same computation within a tenant
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

# Migration SQL to add tenant columns to existing tables
_MIGRATION_SQL = """
-- Add tenant and principal columns to artifact_versions if they don't exist
-- SQLite doesn't have ADD COLUMN IF NOT EXISTS, so we use a workaround

-- Check if tenant column exists by trying to select it
-- If it fails, the column doesn't exist and we need to add it
"""


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

    def __init__(self, artifact_dir: Path, blob_store: BlobStore | None = None):
        """Initialize artifact store.

        Args:
            artifact_dir: Directory for artifacts (contains metadata DB)
            blob_store: Optional blob storage backend. If None, creates a
                LocalBlobStore in {artifact_dir}/blobs.
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

        # Initialize schema
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initialize database schema with migrations for tenant columns."""
        conn = self._get_connection()
        try:
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

                # Add unique index on (tenant, provenance_hash) if not exists
                # This enables idempotent finalize
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_tenant_provenance_unique'"
                )
                if cursor.fetchone() is None:
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
                conn.commit()

            # Registry tables (aliases/tags/audit) — idempotent, applies to
            # fresh and existing databases alike (#129).
            conn.executescript(_REGISTRY_SCHEMA_SQL)
            cursor = conn.execute("PRAGMA table_info(registry_audit)")
            audit_columns = {row["name"] for row in cursor.fetchall()}
            if "from_artifact_id" not in audit_columns:
                conn.execute("ALTER TABLE registry_audit ADD COLUMN from_artifact_id TEXT")
            conn.commit()
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
            # BEGIN IMMEDIATE serializes writers so the MAX(version)+1 read
            # can't race a concurrent create for the same artifact id (two
            # refresh rebuilds used to both read MAX=N and collide on the
            # (id, version) primary key with an uncaught IntegrityError).
            conn.execute("BEGIN IMMEDIATE")
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
                    tenant,
                    principal,
                ),
            )
            conn.commit()
            return version
        finally:
            conn.close()

    def finalize_artifact(
        self,
        artifact_id: str,
        version: int,
        schema_json: str,
        row_count: int,
        byte_size: int,
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

            # Proceed with finalization
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
                    # Race condition: another process may have finalized
                    conn.rollback()
                    return self.get_artifact(artifact_id, version)
                conn.commit()
                return self.get_artifact(artifact_id, version)
            except sqlite3.IntegrityError:
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
        by id without disturbing the prior ready row.

        The partial unique index on
        ``(tenant, provenance_hash) WHERE state='ready'`` is satisfied
        because two rows can share provenance in ready state only via
        this targeted promotion — ``finalize_artifact`` still refuses
        normally.

        Returns the canonical ``ArtifactVersion`` after promotion, or
        ``None`` if it can't be found post-update.
        """
        conn = self._get_connection()
        try:
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

            except sqlite3.IntegrityError:
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
        conn: sqlite3.Connection,
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
        conn: sqlite3.Connection,
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
                       input_versions, tenant, principal
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
                       input_versions, tenant, principal
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
            tenant: Optional tenant filter for multi-tenant isolation.
                If provided, only returns artifacts owned by this tenant.

        Returns:
            Matching ArtifactVersion with state="ready", or None if not found
        """
        conn = self._get_connection()
        try:
            if tenant is not None:
                cursor = conn.execute(
                    """
                    SELECT id, version, state, provenance_hash, schema_json,
                           row_count, byte_size, created_at, transform_spec,
                           input_versions, tenant, principal
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
                           input_versions, tenant, principal
                    FROM artifact_versions
                    WHERE provenance_hash = ? AND state = 'ready'
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
        """Set a key/value tag on an artifact version (audited)."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT state FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            )
            if cursor.fetchone() is None:
                raise ValueError(f"Artifact {artifact_id}@v={version} not found")

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
                           input_versions, tenant, principal
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
                           input_versions, tenant, principal
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

    def list_artifacts(
        self,
        limit: int = 100,
        offset: int = 0,
        state: str | None = None,
        name_prefix: str | None = None,
        tenant: str | None = None,
    ) -> list[ArtifactVersion]:
        """List artifacts with optional filtering.

        Args:
            limit: Maximum number of artifacts to return
            offset: Number of artifacts to skip
            state: Filter by state ("ready", "building", "failed")
            name_prefix: Filter by artifacts that have a name starting with prefix
            tenant: Optional tenant filter. When provided, includes legacy
                tenantless artifacts for backwards compatibility.

        Returns:
            List of ArtifactVersion entries
        """
        conn = self._get_connection()
        try:
            if name_prefix is not None:
                # Join with names table to filter by name prefix
                query = """
                    SELECT DISTINCT av.id, av.version, av.state, av.provenance_hash,
                           av.schema_json, av.row_count, av.byte_size, av.created_at,
                           av.transform_spec, av.input_versions, av.tenant, av.principal
                    FROM artifact_versions av
                    INNER JOIN artifact_names an
                        ON av.id = an.artifact_id AND av.version = an.version
                    WHERE an.name LIKE ?
                """
                params: list = [name_prefix + "%"]

                if tenant is not None:
                    query += " AND (av.tenant = ? OR av.tenant IS NULL)"
                    params.append(tenant)
                    query += " AND (an.tenant = ? OR an.tenant = '')"
                    params.append(tenant)

                if state is not None:
                    query += " AND av.state = ?"
                    params.append(state)

                query += " ORDER BY av.created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            else:
                query = """
                    SELECT id, version, state, provenance_hash, schema_json,
                           row_count, byte_size, created_at, transform_spec,
                           input_versions, tenant, principal
                    FROM artifact_versions
                """
                params = []

                conditions: list[str] = []
                if tenant is not None:
                    conditions.append("(tenant = ? OR tenant IS NULL)")
                    params.append(tenant)
                if state is not None:
                    conditions.append("state = ?")
                    params.append(state)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
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
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def delete_artifact(self, artifact_id: str, version: int) -> bool:
        """Delete an artifact version and its blob.

        Also removes any name pointers to this version.

        Args:
            artifact_id: Artifact ID
            version: Version number

        Returns:
            True if artifact was deleted, False if it didn't exist
        """
        conn = self._get_connection()
        try:
            # Check if artifact exists
            cursor = conn.execute(
                "SELECT 1 FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            )
            if cursor.fetchone() is None:
                return False

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
            conn.execute(
                "DELETE FROM artifact_tags WHERE artifact_id = ? AND version = ?",
                (artifact_id, version),
            )

            # Delete metadata
            conn.execute(
                "DELETE FROM artifact_versions WHERE id = ? AND version = ?",
                (artifact_id, version),
            )
            conn.commit()

            # Delete blob via blob store
            self.blob_store.delete_blob(artifact_id, version)

            return True
        finally:
            conn.close()

    def garbage_collect(
        self,
        max_age_days: float = 7.0,
        tenant: str | None = None,
    ) -> dict:
        """Delete unreferenced artifacts older than max_age.

        An artifact is "unreferenced" if no name pointer points to it.
        Only deletes artifacts in "ready", "superseded", or "failed" state
        older than max_age.

        Args:
            max_age_days: Maximum age in days for unreferenced artifacts
            tenant: Optional tenant filter. When provided, includes legacy
                tenantless artifacts for backwards compatibility.

        Returns:
            Dictionary with GC statistics
        """
        conn = self._get_connection()
        try:
            cutoff = time.time() - (max_age_days * 86400)

            # Find unreferenced artifacts older than cutoff. "Referenced"
            # means a NAME or an ALIAS points at the version — aliases pin
            # registry entries (e.g. champion on a superseded version), and
            # collecting an aliased artifact would leave a dangling pointer
            # to a deleted model.
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
            params: list[float | str] = [cutoff]
            if tenant is not None:
                query += " AND (av.tenant = ? OR av.tenant IS NULL)"
                params.append(tenant)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            deleted_count = 0
            deleted_bytes = 0

            for row in rows:
                artifact_id, version, byte_size = row["id"], row["version"], row["byte_size"] or 0

                # Delete blob via blob store
                self.blob_store.delete_blob(artifact_id, version)

                # Delete metadata
                conn.execute(
                    "DELETE FROM artifact_versions WHERE id = ? AND version = ?",
                    (artifact_id, version),
                )

                deleted_count += 1
                deleted_bytes += byte_size

            conn.commit()

            return {
                "deleted_count": deleted_count,
                "deleted_bytes": deleted_bytes,
                "cutoff_timestamp": cutoff,
            }
        finally:
            conn.close()

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
                usage_query += " WHERE tenant = ? OR tenant IS NULL"
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
                unreferenced_query += " AND (av.tenant = ? OR av.tenant IS NULL)"
                unreferenced_params.append(tenant)
            cursor = conn.execute(unreferenced_query, unreferenced_params)
            unreferenced_count = cursor.fetchone()["count"]

            return {
                "unique_artifacts": row["unique_artifacts"],
                "total_versions": row["total_versions"],
                "ready_versions": row["ready_versions"],
                "building_versions": row["building_versions"],
                "failed_versions": row["failed_versions"],
                "total_bytes": row["total_bytes"],
                "total_rows": row["total_rows"],
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
                SELECT id, version, state, row_count
                FROM artifact_versions
                WHERE state IN ('ready', 'superseded')
            """
            params: list[str] = []
            if tenant is not None:
                query += " AND (tenant = ? OR tenant IS NULL)"
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

            # Delete blobs and metadata
            for row in rows:
                self.blob_store.delete_blob(row["id"], row["version"])
                conn.execute(
                    "DELETE FROM artifact_versions WHERE id = ? AND version = ?",
                    (row["id"], row["version"]),
                )

            conn.commit()
            return len(rows)
        finally:
            conn.close()

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
                stats_query += " WHERE tenant = ? OR tenant IS NULL"
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
                "total_bytes": row["total_bytes"],
                "total_rows": row["total_rows"],
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
) -> ArtifactStore | None:
    """Get the artifact store singleton.

    Args:
        artifact_dir: Directory for artifacts (required on first call in personal mode)
        blob_store: Optional blob storage backend. If None, uses LocalBlobStore.

    Returns:
        ArtifactStore instance, or None if not in personal mode
    """
    global _artifact_store
    if _artifact_store is None and artifact_dir is not None:
        _artifact_store = ArtifactStore(artifact_dir, blob_store=blob_store)
    return _artifact_store


def reset_artifact_store() -> None:
    """Reset the artifact store singleton (for testing)."""
    global _artifact_store
    _artifact_store = None
