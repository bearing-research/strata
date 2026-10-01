"""Build state store for server-orchestrated transforms, in the artifact store's database.

States: ``pending`` -> ``building`` (under a runner's lease) -> ``ready`` or ``failed``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strata.sql_backend import SqlDialect, SqliteDialect, StoreConnection

if TYPE_CHECKING:
    pass


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

# Every blob key a build attempt may write. Blob stores cannot list keys, so
# this is the only way to find an attempt that is never promoted (lost its lease,
# or uploaded and never finalized). Kept apart from artifact_builds so existing
# databases gain it too.
_ATTEMPT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS build_attempts (
    build_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    attempt TEXT NOT NULL,
    writable_until REAL NOT NULL,  -- Nothing can write the attempt's key after this
    PRIMARY KEY (build_id, attempt)
);
"""


@dataclass
class BuildState:
    """One build row. Timestamps are Unix seconds; ``input_uris`` and ``params`` serve pull mode."""

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

    # Used by auth checks.
    @property
    def tenant(self) -> str | None:
        """Alias for tenant_id."""
        return self.tenant_id

    def to_dict(self) -> dict:
        """Return the API response shape."""
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


def _row_to_build_state(row: Any) -> BuildState:
    """Convert a database row to a BuildState."""
    # Columns may be absent on older rows or null.
    input_uris = None
    params = None
    name = None

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
    """Build state store sharing the artifact store's dialect and database.

    Thread-safe: a connection per operation. Sharing the database lets a build
    claimed on one node be seen from another when metadata is on Postgres.
    """

    def __init__(
        self,
        db_path: Path,
        dialect: SqlDialect | None = None,
        clock: Callable[[], float] = time.time,
    ):
        """Open the store and create its schema.

        Args:
            db_path: SQLite database; ignored when ``dialect`` is given.
            dialect: Metadata backend; pass the artifact store's to share its database.
            clock: Wall-clock source in seconds, injectable so tests can expire a
                lease without sleeping.
        """
        self.db_path = db_path
        self._dialect: SqlDialect = dialect if dialect is not None else SqliteDialect(db_path)
        self._clock = clock
        self._init_schema()

    def _get_connection(self) -> StoreConnection:
        """Open a connection configured by the active dialect."""
        return self._dialect.connect()

    def _init_schema(self) -> None:
        """Create the build tables; on SQLite, add columns missing from older databases."""
        conn = self._get_connection()
        try:
            # Only SQLite has older databases to migrate. Other backends create the schema
            # under the schema lock: CREATE TABLE IF NOT EXISTS races in Postgres, and every
            # node runs this at startup.
            if not self._dialect.supports_legacy_migration:
                if not self._dialect.schema_exists(conn, "artifact_builds"):
                    self._dialect.begin_write(conn, "__build_schema__")
                    conn.executescript(self._dialect.adapt_ddl(_BUILD_SCHEMA_SQL))
                    conn.commit()
                if not self._dialect.schema_exists(conn, "build_attempts"):
                    self._dialect.begin_write(conn, "__build_schema__")
                    conn.executescript(self._dialect.adapt_ddl(_ATTEMPT_SCHEMA_SQL))
                    conn.commit()
                return

            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_builds'"
            )
            table_exists = cursor.fetchone() is not None

            if table_exists:
                cursor = conn.execute("PRAGMA table_info(artifact_builds)")
                columns = {row["name"] for row in cursor.fetchall()}

                if "lease_owner" not in columns:
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN lease_owner TEXT")
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN lease_expires_at REAL")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_build_lease_expires "
                        "ON artifact_builds(state, lease_expires_at)"
                    )
                    conn.commit()

                # Pull-model columns.
                if "input_uris" not in columns:
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN input_uris TEXT")
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN params TEXT")
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN name TEXT")
                    conn.commit()

                if "logs" not in columns:
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN logs TEXT")
                    conn.commit()
            else:
                conn.executescript(_BUILD_SCHEMA_SQL)
                conn.commit()

            conn.executescript(_ATTEMPT_SCHEMA_SQL)
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
        """Create a build record in ``pending`` state.

        ``name`` is a name pointer to set once the build completes.
        """
        conn = self._get_connection()
        try:
            created_at = self._clock()
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
        """Get a build by ID, or None if not found."""
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
        """Repoint a build at the existing artifact finalization deduplicated it to."""
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
        """Move a build from pending to building, without a lease; prefer ``claim_build()``.

        Returns False if not found or not pending.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET state = 'building', started_at = ?
                WHERE build_id = ? AND state = 'pending'
                """,
                (self._clock(), build_id),
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
        """Claim a pending build under a lease, atomically; False if another runner won.

        Renew with ``renew_lease()`` or another runner may reclaim the build.
        """
        conn = self._get_connection()
        try:
            now = self._clock()
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
        """Extend the lease from now.

        Returns False unless ``lease_owner`` holds it and the build is building.
        """
        conn = self._get_connection()
        try:
            now = self._clock()
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
        """Take over a building build whose lease has expired (a dead runner's orphan).

        Returns False if the lease has not expired or the build is not building.
        """
        conn = self._get_connection()
        try:
            now = self._clock()
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
        """List building builds whose lease has expired (likely orphans)."""
        conn = self._get_connection()
        try:
            now = self._clock()
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
        """Mark a build ready (building -> ready).

        Args:
            lease_owner: When given, apply only if this owner still holds the lease,
                so a runner whose lease was taken over cannot publish over the new owner.

        Returns:
            False if not found, not building, or the lease is held by someone else.
        """
        conn = self._get_connection()
        try:
            sql = """
                UPDATE artifact_builds
                SET state = 'ready', completed_at = ?, output_byte_count = ?, logs = ?
                WHERE build_id = ? AND state = 'building'
            """
            params: list[Any] = [self._clock(), output_byte_count, logs, build_id]
            if lease_owner is not None:
                sql += " AND lease_owner = ?"
                params.append(lease_owner)
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def complete_within(
        self,
        conn: Any,
        build_id: str,
        *,
        artifact_id: str,
        version: int,
        lease_owner: str,
        lease_expires_at: float | None = None,
        output_byte_count: int | None = None,
        logs: str | None = None,
    ) -> bool:
        """Complete a build on the caller's connection, without committing.

        Lets ``ArtifactStore.finalize_artifact`` mark the artifact ready and complete
        the build in one transaction. ``lease_expires_at`` pins the exact claim, since
        every executor holds the lease as ``external:manifest``. The build is
        repointed at ``artifact_id``/``version`` (they differ after dedup).

        Returns:
            True if this caller held the lease and the build is now complete.
        """
        sql = """
            UPDATE artifact_builds
            SET state = 'ready', completed_at = ?, output_byte_count = ?, logs = ?,
                artifact_id = ?, version = ?
            WHERE build_id = ? AND state = 'building' AND lease_owner = ?
        """
        params: list[Any] = [
            self._clock(),
            output_byte_count,
            logs,
            artifact_id,
            version,
            build_id,
            lease_owner,
        ]
        if lease_expires_at is not None:
            sql += " AND lease_expires_at = ?"
            params.append(lease_expires_at)
        return conn.execute(sql, params).rowcount > 0

    def record_attempt(
        self,
        build_id: str,
        artifact_id: str,
        version: int,
        attempt: str,
        *,
        writable_until: float,
    ) -> None:
        """Record a blob key a build attempt may write, before anything can write it.

        ``writable_until`` is when the last write capability expires (a signed URL's
        expiry, or now for a runner). Re-recording keeps the later time.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO build_attempts (build_id, artifact_id, version, attempt, writable_until)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(build_id, attempt) DO UPDATE SET writable_until = CASE
                    WHEN excluded.writable_until > build_attempts.writable_until
                    THEN excluded.writable_until
                    ELSE build_attempts.writable_until
                END
                """,
                (build_id, artifact_id, version, attempt, writable_until),
            )
            conn.commit()
        finally:
            conn.close()

    def settled_attempts(self, limit: int = 100) -> list[tuple[str, str, int, str, bool]]:
        """List attempts whose build is over and whose key nothing can still write.

        Each is ``(build_id, artifact_id, version, attempt, promoted)``; only a
        promoted attempt's bytes must be kept.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT a.build_id, a.artifact_id, a.version, a.attempt,
                       CASE WHEN v.blob_attempt = a.attempt THEN 1 ELSE 0 END AS promoted
                FROM build_attempts a
                LEFT JOIN artifact_builds b ON b.build_id = a.build_id
                LEFT JOIN artifact_versions v ON v.id = a.artifact_id AND v.version = a.version
                WHERE a.writable_until < ?
                  AND (b.build_id IS NULL OR b.state NOT IN ('pending', 'building'))
                LIMIT ?
                """,
                (self._clock(), limit),
            ).fetchall()
        finally:
            conn.close()
        return [
            (r["build_id"], r["artifact_id"], r["version"], r["attempt"], bool(r["promoted"]))
            for r in rows
        ]

    def forget_attempt(self, build_id: str, attempt: str) -> None:
        """Drop an attempt from the ledger once its bytes are dealt with."""
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM build_attempts WHERE build_id = ? AND attempt = ?",
                (build_id, attempt),
            )
            conn.commit()
        finally:
            conn.close()

    def fail_build(
        self,
        build_id: str,
        error_message: str,
        error_code: str | None = None,
        logs: str | None = None,
        lease_owner: str | None = None,
    ) -> bool:
        """Mark a build failed (building -> failed).

        Args:
            lease_owner: When given, apply only if this owner still holds the lease,
                as in ``complete_build``.

        Returns:
            False if not found, not building, or the lease is held by someone else.
        """
        conn = self._get_connection()
        try:
            sql = """
                UPDATE artifact_builds
                SET state = 'failed', completed_at = ?, error_message = ?, error_code = ?, logs = ?
                WHERE build_id = ? AND state IN ('pending', 'building')
            """
            params: list[Any] = [self._clock(), error_message, error_code, logs, build_id]
            if lease_owner is not None:
                sql += " AND lease_owner = ?"
                params.append(lease_owner)
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_pending_builds(self, limit: int = 100) -> list[BuildState]:
        """List pending builds, oldest first."""
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
        """List a tenant's builds, newest first, optionally filtered by state."""
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
        """Delete ready/failed builds completed more than ``max_age_days`` ago; return the count."""
        conn = self._get_connection()
        try:
            cutoff = self._clock() - (max_age_days * 86400)
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
        """Return build counts, total and by state."""
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


_build_store: BuildStore | None = None


def get_build_store(
    db_path: Path | None = None,
    dialect: SqlDialect | None = None,
) -> BuildStore | None:
    """Get the build store singleton, or None if not yet created.

    ``db_path`` and ``dialect`` apply only on the call that creates it.
    """
    global _build_store
    if _build_store is None and db_path is not None:
        _build_store = BuildStore(db_path, dialect=dialect)
    return _build_store


def reset_build_store() -> None:
    """Reset the build store singleton (for testing)."""
    global _build_store
    _build_store = None
