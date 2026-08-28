"""Build state store for server-mode transforms.

The build store tracks async build lifecycle for server-orchestrated transforms:
- Build creation (when materialize is called)
- Build state transitions (building -> ready/failed)
- Build polling (clients can check status)

Build states:
- pending: Build created but not yet started
- building: Build is running on executor
- ready: Build completed successfully, artifact is available
- failed: Build failed (error message stored)

Database schema stored in artifact_store's artifacts.sqlite for consistency.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# SQL schema for build state tracking
_BUILD_SCHEMA_SQL = """
-- Build state: tracks async build lifecycle
CREATE TABLE IF NOT EXISTS artifact_builds (
    build_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    executor_ref TEXT NOT NULL,
    executor_url TEXT,  -- Resolved executor URL from registry
    tenant_id TEXT,  -- For tenant-based access control
    principal_id TEXT,  -- Who initiated the build
    created_at REAL NOT NULL,
    started_at REAL,  -- When execution started
    completed_at REAL,  -- When execution finished
    error_message TEXT,  -- Error details if failed
    error_code TEXT,  -- Error code for programmatic handling
    input_byte_count INTEGER,  -- Total input size
    output_byte_count INTEGER,  -- Total output size
    lease_owner TEXT,  -- Unique identifier of the runner node holding the lease
    lease_expires_at REAL,  -- Unix timestamp when the lease expires
    input_uris TEXT,  -- JSON array of input artifact URIs (for pull model)
    params TEXT,  -- JSON object of transform params (for pull model)
    name TEXT,  -- Optional name pointer to set after completion
    logs TEXT,  -- Executor stderr/stdout logs (text blob for debugging)
    FOREIGN KEY (artifact_id, version) REFERENCES artifact_versions(id, version)
);

-- Index for state queries (e.g., find pending builds)
CREATE INDEX IF NOT EXISTS idx_build_state ON artifact_builds(state);

-- Index for tenant queries
CREATE INDEX IF NOT EXISTS idx_build_tenant ON artifact_builds(tenant_id);

-- Index for artifact lookup (find builds for an artifact)
CREATE INDEX IF NOT EXISTS idx_build_artifact ON artifact_builds(artifact_id, version);

-- Index for expired lease queries (orphan recovery)
CREATE INDEX IF NOT EXISTS idx_build_lease_expires ON artifact_builds(state, lease_expires_at);
"""


@dataclass
class BuildState:
    """Build state record.

    Attributes:
        build_id: Unique build identifier (UUID)
        artifact_id: Target artifact ID
        version: Target artifact version
        state: Current state (pending, building, ready, failed)
        executor_ref: Executor reference (e.g., "duckdb_sql@v1")
        executor_url: Resolved executor URL from registry
        tenant_id: Tenant who owns this build (for access control)
        principal_id: Principal who initiated this build
        created_at: Unix timestamp when build was created
        started_at: Unix timestamp when execution started (or None)
        completed_at: Unix timestamp when execution finished (or None)
        error_message: Error details if state is "failed"
        error_code: Error code for programmatic handling
        input_byte_count: Total input size in bytes (or None)
        output_byte_count: Total output size in bytes (or None)
        lease_owner: Unique identifier of the runner node holding the lease
        lease_expires_at: Unix timestamp when the lease expires
        input_uris: List of input artifact URIs (for pull model)
        params: Transform parameters (for pull model)
        name: Optional name pointer to set after completion
        logs: Executor stderr/stdout logs (text blob for debugging)
    """

    build_id: str
    artifact_id: str
    version: int
    state: str
    executor_ref: str
    executor_url: str | None
    tenant_id: str | None
    principal_id: str | None
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    error_code: str | None = None
    input_byte_count: int | None = None
    output_byte_count: int | None = None
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    input_uris: list[str] | None = None
    params: dict | None = None
    name: str | None = None
    logs: str | None = None

    # Alias for tenant_id (used by auth checks)
    @property
    def tenant(self) -> str | None:
        """Alias for tenant_id."""
        return self.tenant_id

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "build_id": self.build_id,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "state": self.state,
            "executor_ref": self.executor_ref,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "input_byte_count": self.input_byte_count,
            "output_byte_count": self.output_byte_count,
        }


def _row_to_build_state(row: sqlite3.Row) -> BuildState:
    """Convert a database row to BuildState.

    Args:
        row: SQLite row with build data

    Returns:
        BuildState instance
    """
    # Parse JSON fields (may be None if column doesn't exist or is null)
    input_uris = None
    params = None
    name = None

    # Handle optional columns gracefully
    try:
        if row["input_uris"]:
            input_uris = json.loads(row["input_uris"])
    except (KeyError, IndexError):
        pass

    try:
        if row["params"]:
            params = json.loads(row["params"])
    except (KeyError, IndexError):
        pass

    try:
        name = row["name"]
    except (KeyError, IndexError):
        pass

    logs = None
    try:
        logs = row["logs"]
    except (KeyError, IndexError):
        pass

    return BuildState(
        build_id=row["build_id"],
        artifact_id=row["artifact_id"],
        version=row["version"],
        state=row["state"],
        executor_ref=row["executor_ref"],
        executor_url=row["executor_url"],
        tenant_id=row["tenant_id"],
        principal_id=row["principal_id"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        error_message=row["error_message"],
        error_code=row["error_code"],
        input_byte_count=row["input_byte_count"],
        output_byte_count=row["output_byte_count"],
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        input_uris=input_uris,
        params=params,
        name=name,
        logs=logs,
    )


class BuildStore:
    """SQLite-backed build state store.

    Thread-safe: uses connection per operation with WAL mode.
    """

    def __init__(self, db_path: Path):
        """Initialize build store.

        Args:
            db_path: Path to SQLite database (shared with artifact store)
        """
        self.db_path = db_path
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initialize build state schema with migrations for lease columns."""
        conn = self._get_connection()
        try:
            # Check if table exists and needs migration
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_builds'"
            )
            table_exists = cursor.fetchone() is not None

            if table_exists:
                # Check if lease columns exist
                cursor = conn.execute("PRAGMA table_info(artifact_builds)")
                columns = {row["name"] for row in cursor.fetchall()}

                if "lease_owner" not in columns:
                    # Add lease columns
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN lease_owner TEXT")
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN lease_expires_at REAL")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_build_lease_expires "
                        "ON artifact_builds(state, lease_expires_at)"
                    )
                    conn.commit()

                # Check for pull model columns (added for Stage 2)
                if "input_uris" not in columns:
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN input_uris TEXT")
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN params TEXT")
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN name TEXT")
                    conn.commit()

                # Check for logs column (added for observability)
                if "logs" not in columns:
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN logs TEXT")
                    conn.commit()
            else:
                # Fresh database: create schema
                conn.executescript(_BUILD_SCHEMA_SQL)
                conn.commit()
        finally:
            conn.close()

    def create_build(
        self,
        build_id: str,
        artifact_id: str,
        version: int,
        executor_ref: str,
        executor_url: str | None = None,
        tenant_id: str | None = None,
        principal_id: str | None = None,
        input_uris: list[str] | None = None,
        params: dict | None = None,
        name: str | None = None,
    ) -> BuildState:
        """Create a new build record in pending state.

        Args:
            build_id: Unique build ID (UUID)
            artifact_id: Target artifact ID
            version: Target artifact version
            executor_ref: Executor reference
            executor_url: Resolved executor URL
            tenant_id: Tenant who owns this build
            principal_id: Principal who initiated this build
            input_uris: List of input artifact URIs (for pull model)
            params: Transform parameters (for pull model)
            name: Optional name pointer to set after completion

        Returns:
            Created BuildState record
        """
        conn = self._get_connection()
        try:
            created_at = time.time()
            # Serialize input_uris and params to JSON
            input_uris_json = json.dumps(input_uris) if input_uris else None
            params_json = json.dumps(params) if params else None

            conn.execute(
                """
                INSERT INTO artifact_builds
                    (build_id, artifact_id, version, state, executor_ref,
                     executor_url, tenant_id, principal_id, created_at,
                     input_uris, params, name)
                VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    build_id,
                    artifact_id,
                    version,
                    executor_ref,
                    executor_url,
                    tenant_id,
                    principal_id,
                    created_at,
                    input_uris_json,
                    params_json,
                    name,
                ),
            )
            conn.commit()

            return BuildState(
                build_id=build_id,
                artifact_id=artifact_id,
                version=version,
                state="pending",
                executor_ref=executor_ref,
                executor_url=executor_url,
                tenant_id=tenant_id,
                principal_id=principal_id,
                created_at=created_at,
                input_uris=input_uris,
                params=params,
                name=name,
            )
        finally:
            conn.close()

    def get_build(self, build_id: str) -> BuildState | None:
        """Get build state by ID.

        Args:
            build_id: Build ID to look up

        Returns:
            BuildState or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT build_id, artifact_id, version, state, executor_ref,
                       executor_url, tenant_id, principal_id, created_at,
                       started_at, completed_at, error_message, error_code,
                       input_byte_count, output_byte_count,
                       lease_owner, lease_expires_at,
                       input_uris, params, name, logs
                FROM artifact_builds
                WHERE build_id = ?
                """,
                (build_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return _row_to_build_state(row)
        finally:
            conn.close()

    def update_build_output(self, build_id: str, artifact_id: str, version: int) -> bool:
        """Repoint a build at the canonical artifact it produced.

        This is used when finalization deduplicates to an existing artifact
        with the same provenance.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET artifact_id = ?, version = ?
                WHERE build_id = ?
                """,
                (artifact_id, version, build_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def start_build(self, build_id: str) -> bool:
        """Mark build as started (pending -> building).

        DEPRECATED: Use claim_build() for lease-based claiming instead.

        Args:
            build_id: Build ID to update

        Returns:
            True if updated, False if not found or wrong state
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET state = 'building', started_at = ?
                WHERE build_id = ? AND state = 'pending'
                """,
                (time.time(), build_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def claim_build(
        self,
        build_id: str,
        lease_owner: str,
        lease_duration_seconds: float = 60.0,
    ) -> bool:
        """Claim a build with a lease (pending -> building).

        Uses optimistic locking to prevent multiple runners from claiming
        the same build. The lease must be renewed periodically via
        renew_lease() or the build can be reclaimed by another runner.

        Args:
            build_id: Build ID to claim
            lease_owner: Unique identifier of the runner claiming the build
            lease_duration_seconds: How long the lease is valid (default 60s)

        Returns:
            True if successfully claimed, False if already claimed or wrong state
        """
        conn = self._get_connection()
        try:
            now = time.time()
            lease_expires_at = now + lease_duration_seconds
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET state = 'building',
                    started_at = ?,
                    lease_owner = ?,
                    lease_expires_at = ?
                WHERE build_id = ? AND state = 'pending'
                """,
                (now, lease_owner, lease_expires_at, build_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def renew_lease(
        self,
        build_id: str,
        lease_owner: str,
        lease_duration_seconds: float = 60.0,
    ) -> bool:
        """Renew the lease on a build.

        Only succeeds if the caller currently holds the lease.

        Args:
            build_id: Build ID to renew
            lease_owner: Must match the current lease owner
            lease_duration_seconds: New lease duration from now

        Returns:
            True if lease renewed, False if not owned by caller or not building
        """
        conn = self._get_connection()
        try:
            now = time.time()
            lease_expires_at = now + lease_duration_seconds
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET lease_expires_at = ?
                WHERE build_id = ?
                  AND state = 'building'
                  AND lease_owner = ?
                """,
                (lease_expires_at, build_id, lease_owner),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def reclaim_expired_build(
        self,
        build_id: str,
        new_lease_owner: str,
        lease_duration_seconds: float = 60.0,
    ) -> bool:
        """Reclaim a build with an expired lease.

        Used for orphan recovery when a runner dies without completing a build.
        Only succeeds if the build is in 'building' state and the lease has expired.

        Args:
            build_id: Build ID to reclaim
            new_lease_owner: New owner taking over the build
            lease_duration_seconds: Lease duration for new owner

        Returns:
            True if reclaimed, False if lease not expired or wrong state
        """
        conn = self._get_connection()
        try:
            now = time.time()
            lease_expires_at = now + lease_duration_seconds
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET lease_owner = ?,
                    lease_expires_at = ?,
                    started_at = ?
                WHERE build_id = ?
                  AND state = 'building'
                  AND lease_expires_at < ?
                """,
                (new_lease_owner, lease_expires_at, now, build_id, now),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_expired_leases(self, limit: int = 10) -> list[BuildState]:
        """List builds with expired leases (orphans).

        These are builds in 'building' state where the lease has expired,
        indicating the runner may have crashed.

        Args:
            limit: Maximum number of builds to return

        Returns:
            List of BuildState records with expired leases
        """
        conn = self._get_connection()
        try:
            now = time.time()
            cursor = conn.execute(
                """
                SELECT build_id, artifact_id, version, state, executor_ref,
                       executor_url, tenant_id, principal_id, created_at,
                       started_at, completed_at, error_message, error_code,
                       input_byte_count, output_byte_count,
                       lease_owner, lease_expires_at,
                       input_uris, params, name, logs
                FROM artifact_builds
                WHERE state = 'building'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < ?
                ORDER BY lease_expires_at ASC
                LIMIT ?
                """,
                (now, limit),
            )

            return [_row_to_build_state(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def complete_build(
        self,
        build_id: str,
        output_byte_count: int | None = None,
        logs: str | None = None,
        lease_owner: str | None = None,
    ) -> bool:
        """Mark build as completed (building -> ready).

        Args:
            build_id: Build ID to update
            output_byte_count: Output size in bytes
            logs: Executor stdout/stderr logs (optional)
            lease_owner: When given, the update only applies if this owner
                still holds the lease. This is the counterpart to
                ``claim_build``: claiming decides who starts a build,
                completing decides who is allowed to publish its result. A
                runner whose lease was stolen (a GC pause or a partition
                longer than the lease) keeps executing by design, so without
                this check it published over the runner that legitimately
                took the build over.

        Returns:
            True if updated, False if not found, wrong state, or the lease is
            held by someone else
        """
        conn = self._get_connection()
        try:
            sql = """
                UPDATE artifact_builds
                SET state = 'ready', completed_at = ?, output_byte_count = ?, logs = ?
                WHERE build_id = ? AND state = 'building'
            """
            params: list[Any] = [time.time(), output_byte_count, logs, build_id]
            if lease_owner is not None:
                sql += " AND lease_owner = ?"
                params.append(lease_owner)
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def fail_build(
        self,
        build_id: str,
        error_message: str,
        error_code: str | None = None,
        logs: str | None = None,
    ) -> bool:
        """Mark build as failed (building -> failed).

        Args:
            build_id: Build ID to update
            error_message: Error details
            error_code: Error code for programmatic handling
            logs: Executor stdout/stderr logs (optional)

        Returns:
            True if updated, False if not found or wrong state
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET state = 'failed', completed_at = ?, error_message = ?, error_code = ?, logs = ?
                WHERE build_id = ? AND state IN ('pending', 'building')
                """,
                (time.time(), error_message, error_code, logs, build_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_pending_builds(self, limit: int = 100) -> list[BuildState]:
        """List builds in pending state (ready for execution).

        Args:
            limit: Maximum number of builds to return

        Returns:
            List of pending BuildState records, oldest first
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT build_id, artifact_id, version, state, executor_ref,
                       executor_url, tenant_id, principal_id, created_at,
                       started_at, completed_at, error_message, error_code,
                       input_byte_count, output_byte_count,
                       lease_owner, lease_expires_at,
                       input_uris, params, name, logs
                FROM artifact_builds
                WHERE state = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            )

            return [_row_to_build_state(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def list_builds_by_tenant(
        self,
        tenant_id: str,
        limit: int = 100,
        state: str | None = None,
    ) -> list[BuildState]:
        """List builds for a specific tenant.

        Args:
            tenant_id: Tenant ID to filter by
            limit: Maximum number of builds to return
            state: Optional state filter

        Returns:
            List of BuildState records, newest first
        """
        conn = self._get_connection()
        try:
            if state:
                cursor = conn.execute(
                    """
                    SELECT build_id, artifact_id, version, state, executor_ref,
                           executor_url, tenant_id, principal_id, created_at,
                           started_at, completed_at, error_message, error_code,
                           input_byte_count, output_byte_count,
                           lease_owner, lease_expires_at,
                           input_uris, params, name, logs
                    FROM artifact_builds
                    WHERE tenant_id = ? AND state = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (tenant_id, state, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT build_id, artifact_id, version, state, executor_ref,
                           executor_url, tenant_id, principal_id, created_at,
                           started_at, completed_at, error_message, error_code,
                           input_byte_count, output_byte_count,
                           lease_owner, lease_expires_at,
                           input_uris, params, name, logs
                    FROM artifact_builds
                    WHERE tenant_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (tenant_id, limit),
                )

            return [_row_to_build_state(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def cleanup_old_builds(self, max_age_days: float = 7.0) -> int:
        """Clean up old completed/failed builds.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of builds deleted
        """
        conn = self._get_connection()
        try:
            cutoff = time.time() - (max_age_days * 86400)
            cursor = conn.execute(
                """
                DELETE FROM artifact_builds
                WHERE state IN ('ready', 'failed')
                  AND completed_at < ?
                """,
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Get build statistics.

        Returns:
            Dictionary with build counts by state
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN state = 'pending' THEN 1 END) as pending,
                    COUNT(CASE WHEN state = 'building' THEN 1 END) as building,
                    COUNT(CASE WHEN state = 'ready' THEN 1 END) as ready,
                    COUNT(CASE WHEN state = 'failed' THEN 1 END) as failed
                FROM artifact_builds
                """
            )
            row = cursor.fetchone()
            return {
                "total": row["total"],
                "pending": row["pending"],
                "building": row["building"],
                "ready": row["ready"],
                "failed": row["failed"],
            }
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_build_store: BuildStore | None = None


def get_build_store(db_path: Path | None = None) -> BuildStore | None:
    """Get the build store singleton.

    Args:
        db_path: Path to SQLite database (required on first call)

    Returns:
        BuildStore instance, or None if not initialized
    """
    global _build_store
    if _build_store is None and db_path is not None:
        _build_store = BuildStore(db_path)
    return _build_store


def reset_build_store() -> None:
    """Reset the build store singleton (for testing)."""
    global _build_store
    _build_store = None
