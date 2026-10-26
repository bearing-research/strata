"""Releases: the publication events of a build (ADR 002 §7.2 and §7.3).

A build is an execution; its attempts are the tries that may write its bytes; a
release is one event that makes a set of staged outputs current together. A
``building`` build accepts any number of ``checkpoint`` releases and stays
building; its ``final`` release ends it. Acceptance is fenced: a release whose
attempt no longer holds the build's lease, or whose write window has closed, is
rejected, so a resumed execution's old attempt can never publish over the new one.
Acceptance follows confirmed completion: every staged output's bytes must be in the
blob store at the declared size before the row becomes current, which on storage
with no atomic rename is the only commit point there is.

The typed operations a caller passes with the release (:mod:`vestigo.operations`)
run inside the acceptance transaction, in order; one that fails leaves nothing
behind and the release where it was.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vestigo.operations import (
    AdvanceReference,
    AssertReferenceRevision,
    AssignAuthority,
    CheckBuildLease,
    InsertAuditEvent,
    Operation,
)

if TYPE_CHECKING:
    from vestigo.build_store import BuildStore
    from vestigo.sql_backend import StoreConnection
    from vestigo.store import ArtifactStore, ArtifactVersion, StagedVersion

KINDS = ("checkpoint", "final")
UNSETTLED = ("preparing", "prepared")
HEAD_ROOT_KIND = "head"
"""The retention root kind a reference holds its release's outputs under."""


class ReleaseError(RuntimeError):
    """A release could not do what was asked; nothing was written."""


class ReleaseRejected(ReleaseError):
    """The release's attempt no longer holds the build's lease; the release is rejected."""


class ReleaseIncomplete(ReleaseError):
    """An output's bytes are not where the staged row says; the release stays unsettled."""


class ReleaseOutOfOrder(ReleaseError):
    """An earlier release of the same build and kind is still unsettled."""


class OperationFailed(ReleaseError):
    """A typed operation's condition did not hold; the transaction was rolled back."""


RELEASE_DDL: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS build_releases (
        release_id TEXT PRIMARY KEY,
        tenant TEXT NOT NULL DEFAULT '',
        build_id TEXT NOT NULL,
        attempt TEXT NOT NULL,
        kind TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        state TEXT NOT NULL,
        created_at REAL NOT NULL,
        settled_at REAL,
        UNIQUE (build_id, kind, sequence)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_releases_build ON build_releases(build_id, state)",
    """CREATE TABLE IF NOT EXISTS release_references (
        tenant TEXT NOT NULL DEFAULT '',
        name TEXT NOT NULL,
        release_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (tenant, name)
    )""",
    """CREATE TABLE IF NOT EXISTS release_authorities (
        tenant TEXT NOT NULL DEFAULT '',
        target TEXT NOT NULL,
        release_id TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (tenant, target)
    )""",
)
"""The release tables; also the tail of a fresh store's schema (``store._SCHEMA_SQL``)."""


@dataclass(frozen=True)
class Release:
    release_id: str
    tenant: str
    build_id: str
    attempt: str
    kind: str
    sequence: int
    state: str
    created_at: float
    settled_at: float | None


@dataclass(frozen=True)
class ReleaseBundle:
    """A release and its outputs, read together."""

    release: Release
    outputs: list[ArtifactVersion]


def _release_from_row(row) -> Release:
    return Release(
        release_id=row["release_id"],
        tenant=row["tenant"],
        build_id=row["build_id"],
        attempt=row["attempt"],
        kind=row["kind"],
        sequence=row["sequence"],
        state=row["state"],
        created_at=row["created_at"],
        settled_at=row["settled_at"],
    )


def lease_held(conn: StoreConnection, build_id: str, attempt: str, now: float) -> bool:
    """Whether *attempt* is the build's current attempt and may still write."""
    row = conn.execute(
        "SELECT b.state, b.attempt, a.writable_until FROM artifact_builds b "
        "LEFT JOIN build_attempts a ON a.build_id = b.build_id AND a.attempt = ? "
        "WHERE b.build_id = ?",
        (attempt, build_id),
    ).fetchone()
    if row is None or row["state"] != "building" or row["attempt"] != attempt:
        return False
    return row["writable_until"] is not None and row["writable_until"] > now


class ReleaseStore:
    """The releases of one ledger, over its artifact and build stores (one database)."""

    def __init__(
        self,
        artifacts: ArtifactStore,
        builds: BuildStore,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._artifacts = artifacts
        self._builds = builds
        self._clock = clock
        self._connect = artifacts._get_connection
        self._dialect = artifacts.dialect

    # --- Lifecycle ---

    def prepare(
        self,
        build_id: str,
        attempt: str,
        kind: str,
        *,
        sequence: int | None = None,
        tenant: str | None = None,
        release_id: str | None = None,
    ) -> Release:
        """Open a release the attempt will stage outputs for; ``preparing`` until accepted.

        *sequence* defaults to the next of the build's *kind*. The attempt must hold
        the build's lease now, as it must again at acceptance.

        Raises:
            ValueError: *kind* is not ``checkpoint`` or ``final``.
            ReleaseRejected: the attempt does not hold the lease.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
        conn = self._connect()
        try:
            self._dialect.begin_write(conn, build_id)
            now = self._clock()
            if not lease_held(conn, build_id, attempt, now):
                conn.rollback()
                raise ReleaseRejected(f"attempt {attempt} does not hold build {build_id}")
            if sequence is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence "
                    "FROM build_releases WHERE build_id = ? AND kind = ?",
                    (build_id, kind),
                ).fetchone()
                sequence = int(row["next_sequence"])
            release = Release(
                release_id=release_id or uuid.uuid4().hex,
                tenant=tenant if tenant is not None else "",
                build_id=build_id,
                attempt=attempt,
                kind=kind,
                sequence=sequence,
                state="preparing",
                created_at=now,
                settled_at=None,
            )
            conn.execute(
                "INSERT INTO build_releases (release_id, tenant, build_id, attempt, kind, "
                "sequence, state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    release.release_id,
                    release.tenant,
                    build_id,
                    attempt,
                    kind,
                    sequence,
                    "preparing",
                    now,
                ),
            )
            conn.commit()
            return release
        finally:
            conn.close()

    def attach(self, release_id: str, versions: Sequence[tuple[str, int]]) -> int:
        """Stage existing ``building`` rows under the release (a build's own target row).

        A row created with ``release_id`` needs no attaching. Returns how many rows
        were attached; a row that is ready or under another release is left alone.
        """
        conn = self._connect()
        try:
            attached = 0
            for artifact_id, version in versions:
                attached += conn.execute(
                    "UPDATE artifact_versions SET release_id = ? "
                    "WHERE id = ? AND version = ? AND state = 'building' AND release_id IS NULL",
                    (release_id, artifact_id, version),
                ).rowcount
            conn.commit()
            return attached
        finally:
            conn.close()

    def mark_prepared(self, release_id: str) -> bool:
        """The producer says every output is staged; ``prepared`` until accepted."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE build_releases SET state = 'prepared' "
                "WHERE release_id = ? AND state = 'preparing'",
                (release_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def accept(
        self,
        release_id: str,
        staged: Sequence[StagedVersion],
        operations: Sequence[Operation] = (),
    ) -> ReleaseBundle:
        """Make the release's staged outputs current together, running *operations* with them.

        Fenced on the attempt's lease; ordered after every earlier release of the
        build's kind; refused while an output's bytes are not at their declared size.
        A ``final`` release completes the build. Accepting an accepted release again
        returns it unchanged, so a caller whose response was lost can retry.

        Raises:
            ReleaseRejected: the attempt lost the lease; the release is now rejected.
            ReleaseOutOfOrder: an earlier release of the kind is still unsettled.
            ReleaseIncomplete: an output is not staged under this release, or its
                bytes are missing or the wrong size.
            OperationFailed: an operation's condition did not hold.
            ValueError: the release does not exist or is settled another way.
        """
        conn = self._connect()
        try:
            release = self._get_on(conn, release_id)
            if release is None:
                raise ValueError(f"release {release_id} does not exist")
            self._dialect.begin_write(conn, release.build_id)
            release = self._get_on(conn, release_id)
            assert release is not None
            if release.state == "accepted":
                conn.rollback()
                return ReleaseBundle(release, self._outputs_on(conn, release_id))
            if release.state not in UNSETTLED:
                conn.rollback()
                raise ValueError(f"release {release_id} is {release.state}")
            now = self._clock()
            if not lease_held(conn, release.build_id, release.attempt, now):
                conn.rollback()
                self._settle(release_id, "rejected")
                raise ReleaseRejected(
                    f"attempt {release.attempt} no longer holds build {release.build_id}"
                )
            earlier = conn.execute(
                "SELECT COUNT(*) AS n FROM build_releases WHERE build_id = ? AND kind = ? "
                "AND sequence < ? AND state IN ('preparing', 'prepared')",
                (release.build_id, release.kind, release.sequence),
            ).fetchone()
            if earlier["n"]:
                conn.rollback()
                raise ReleaseOutOfOrder(
                    f"{earlier['n']} earlier {release.kind} release(s) of build "
                    f"{release.build_id} are unsettled"
                )
            self._check_outputs(conn, release, staged)
            try:
                for op in operations:
                    self._apply(conn, op, release, now)
                self._artifacts._promote_staged(conn, list(staged), now)
                if release.kind == "final":
                    self._builds.complete_attempt_within(
                        conn,
                        release.build_id,
                        release.attempt,
                        artifact_id=staged[0].artifact_id if staged else None,
                        version=staged[0].version if staged else None,
                        now=now,
                    )
                conn.execute(
                    "UPDATE build_releases SET state = 'accepted', settled_at = ? "
                    "WHERE release_id = ?",
                    (now, release_id),
                )
                conn.commit()
            except OperationFailed:
                conn.rollback()
                raise
            except self._dialect.integrity_error as exc:
                conn.rollback()
                raise OperationFailed(f"the ledger refused the release: {exc}") from exc
            accepted = self._get_on(conn, release_id)
            assert accepted is not None
            return ReleaseBundle(accepted, self._outputs_on(conn, release_id))
        finally:
            conn.close()

    def abandon(self, release_id: str) -> bool:
        """Give up an unsettled release; its staged rows stay ``building`` for GC."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE build_releases SET state = 'abandoned', settled_at = ? "
                "WHERE release_id = ? AND state IN ('preparing', 'prepared')",
                (self._clock(), release_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def resume(
        self,
        build_id: str,
        attempt: str,
        *,
        writable_until: float,
        lease_owner: str | None = None,
        lease_duration_seconds: float = 60.0,
    ) -> int:
        """Continue the build in a new attempt, fencing every earlier one.

        The build keeps its identity and its accepted releases; the old attempt can no
        longer release, and whatever it left unsettled is abandoned. Returns how many
        releases were abandoned.

        Raises:
            ValueError: the build is not pending or building.
        """
        conn = self._connect()
        try:
            self._dialect.begin_write(conn, build_id)
            now = self._clock()
            if not self._builds.open_attempt_within(
                conn,
                build_id,
                attempt,
                writable_until=writable_until,
                lease_owner=lease_owner,
                lease_expires_at=now + lease_duration_seconds,
                now=now,
            ):
                conn.rollback()
                raise ValueError(f"build {build_id} is not active")
            cursor = conn.execute(
                "UPDATE build_releases SET state = 'abandoned', settled_at = ? "
                "WHERE build_id = ? AND attempt <> ? AND state IN ('preparing', 'prepared')",
                (now, build_id, attempt),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    # --- Reading ---

    def get(self, release_id: str) -> Release | None:
        conn = self._connect()
        try:
            return self._get_on(conn, release_id)
        finally:
            conn.close()

    def bundle(self, release_id: str) -> ReleaseBundle | None:
        """The release and its outputs, in one consistent read."""
        conn = self._connect()
        try:
            release = self._get_on(conn, release_id)
            if release is None:
                return None
            return ReleaseBundle(release, self._outputs_on(conn, release_id))
        finally:
            conn.close()

    def releases_of(self, build_id: str, kind: str | None = None) -> list[Release]:
        """A build's releases in sequence order, of one kind or all."""
        conn = self._connect()
        try:
            sql = "SELECT * FROM build_releases WHERE build_id = ?"
            params: list = [build_id]
            if kind is not None:
                sql += " AND kind = ?"
                params.append(kind)
            rows = conn.execute(sql + " ORDER BY kind, sequence", params).fetchall()
        finally:
            conn.close()
        return [_release_from_row(r) for r in rows]

    def reference(self, name: str, *, tenant: str | None = None) -> tuple[str, int] | None:
        """``(release_id, revision)`` of a reference, or ``None`` when it does not exist."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT release_id, revision FROM release_references WHERE tenant = ? AND name = ?",
                (tenant if tenant is not None else "", name),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else (row["release_id"], row["revision"])

    def authority(self, target: str, *, tenant: str | None = None) -> str | None:
        """The release that won an output slot, or ``None`` while it has no winner."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT release_id FROM release_authorities WHERE tenant = ? AND target = ?",
                (tenant if tenant is not None else "", target),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else row["release_id"]

    # --- Internals ---

    @staticmethod
    def _get_on(conn: StoreConnection, release_id: str) -> Release | None:
        row = conn.execute(
            "SELECT * FROM build_releases WHERE release_id = ?", (release_id,)
        ).fetchone()
        return None if row is None else _release_from_row(row)

    def _outputs_on(self, conn: StoreConnection, release_id: str) -> list[ArtifactVersion]:
        return self._artifacts._versions_where(conn, "release_id = ?", (release_id,))

    def _settle(self, release_id: str, state: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE build_releases SET state = ?, settled_at = ? WHERE release_id = ?",
                (state, self._clock(), release_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _check_outputs(
        self, conn: StoreConnection, release: Release, staged: Sequence[StagedVersion]
    ) -> None:
        """Every output is a building row of this release whose bytes are all there."""
        blobs = self._artifacts.blob_store
        for item in staged:
            row = conn.execute(
                "SELECT state, release_id FROM artifact_versions WHERE id = ? AND version = ?",
                (item.artifact_id, item.version),
            ).fetchone()
            if row is None or row["state"] != "building" or row["release_id"] != release.release_id:
                conn.rollback()
                raise ReleaseIncomplete(
                    f"{item.artifact_id}@v={item.version} is not staged under this release"
                )
            size = blobs.blob_size(item.artifact_id, item.version)
            if size != item.byte_size:
                conn.rollback()
                raise ReleaseIncomplete(
                    f"{item.artifact_id}@v={item.version}: {item.byte_size} bytes declared, "
                    f"{'none' if size is None else size} stored"
                )

    def _apply(self, conn: StoreConnection, op: Operation, release: Release, now: float) -> None:
        tenant = release.tenant
        if isinstance(op, CheckBuildLease):
            if not lease_held(conn, op.build_id, op.attempt, now):
                raise OperationFailed(f"attempt {op.attempt} does not hold build {op.build_id}")
        elif isinstance(op, AssertReferenceRevision):
            row = conn.execute(
                "SELECT revision FROM release_references WHERE tenant = ? AND name = ?",
                (tenant, op.name),
            ).fetchone()
            revision = 0 if row is None else row["revision"]
            if revision != op.expected_revision:
                raise OperationFailed(
                    f"reference {op.name} is at revision {revision}, not {op.expected_revision}"
                )
        elif isinstance(op, AssignAuthority):
            conn.execute(
                "INSERT INTO release_authorities (tenant, target, release_id, created_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (tenant, target) DO NOTHING",
                (tenant, op.target, op.release_id, now),
            )
            row = conn.execute(
                "SELECT release_id FROM release_authorities WHERE tenant = ? AND target = ?",
                (tenant, op.target),
            ).fetchone()
            if row["release_id"] != op.release_id:
                raise OperationFailed(f"slot {op.target} already belongs to {row['release_id']}")
        elif isinstance(op, AdvanceReference):
            self._advance(conn, tenant, op.name, op.release_id, now)
        elif isinstance(op, InsertAuditEvent):
            conn.execute(
                "INSERT INTO registry_audit (at, actor, action, name, value, tenant) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    now,
                    op.event.principal,
                    op.event.action,
                    release.release_id,
                    json.dumps(dict(op.event.details), sort_keys=True)
                    if op.event.details
                    else None,
                    tenant,
                ),
            )
        else:
            raise OperationFailed(f"{type(op).__name__} is not an operation")

    @staticmethod
    def _advance(
        conn: StoreConnection, tenant: str, name: str, release_id: str, now: float
    ) -> None:
        """Point the reference at the release and hold its outputs as the reference's root."""
        cursor = conn.execute(
            "UPDATE release_references SET release_id = ?, revision = revision + 1, "
            "updated_at = ? WHERE tenant = ? AND name = ?",
            (release_id, now, tenant, name),
        )
        if cursor.rowcount == 0:
            conn.execute(
                "INSERT INTO release_references (tenant, name, release_id, revision, updated_at) "
                "VALUES (?, ?, ?, 1, ?)",
                (tenant, name, release_id, now),
            )
        conn.execute(
            "DELETE FROM retention_roots WHERE tenant = ? AND kind = ? AND holder = ?",
            (tenant, HEAD_ROOT_KIND, name),
        )
        conn.execute(
            "INSERT INTO retention_roots (tenant, kind, holder, artifact_id, version, created_at) "
            "SELECT ?, ?, ?, id, version, ? FROM artifact_versions WHERE release_id = ?",
            (tenant, HEAD_ROOT_KIND, name, now, release_id),
        )
