"""Releases: the publication events of a build (ADR 002 §7.2 and §7.3).

A build is an execution; its attempts are the tries that may write its bytes; a
release is one event that makes a set of staged outputs current together. A
``building`` build accepts any number of ``checkpoint`` releases and stays
building; its ``final`` release ends it. Acceptance is fenced: a release whose
attempt is no longer the build's, whose lease has lapsed, or whose write window
has closed, is rejected, so a resumed execution's old attempt can never publish
over the new one. Acceptance follows confirmed completion: every output staged
under the release must be declared, and its bytes must be in the blob store at the
declared size before the row becomes current, which on storage with no atomic
rename is the only commit point there is.

The typed operations a caller passes with the release (:mod:`vestigo.operations`)
run inside the acceptance transaction, in order; one that fails leaves nothing
behind and the release where it was. A release belongs to its build's tenant.
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
    """Whether *attempt* is the build's current attempt and may still write.

    It may not once the build's lease has lapsed (another runner may reclaim the
    build) or its write window has closed.
    """
    row = conn.execute(
        "SELECT b.state, b.attempt, b.lease_expires_at, a.writable_until FROM artifact_builds b "
        "LEFT JOIN build_attempts a ON a.build_id = b.build_id AND a.attempt = ? "
        "WHERE b.build_id = ?",
        (attempt, build_id),
    ).fetchone()
    if row is None or row["state"] != "building" or row["attempt"] != attempt:
        return False
    if row["lease_expires_at"] is not None and row["lease_expires_at"] < now:
        return False
    return row["writable_until"] is not None and row["writable_until"] > now


def _tenant_of(conn: StoreConnection, build_id: str) -> str:
    """The build's tenant, ``''`` for the default; the build must exist."""
    row = conn.execute(
        "SELECT tenant_id FROM artifact_builds WHERE build_id = ?", (build_id,)
    ).fetchone()
    return row["tenant_id"] or ""


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
        the build's lease now, as it must again at acceptance. *tenant* defaults to
        the build's and may not differ from it.

        Raises:
            ValueError: *kind* is not ``checkpoint`` or ``final``, or *tenant* is
                not the build's.
            ReleaseRejected: the attempt does not hold the lease.
            ReleaseError: the build already has a *kind* release at *sequence*, or
                *release_id* is taken.
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
            tenant = self._release_tenant(conn, build_id, tenant)
            if sequence is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence "
                    "FROM build_releases WHERE build_id = ? AND kind = ?",
                    (build_id, kind),
                ).fetchone()
                sequence = int(row["next_sequence"])
            release = Release(
                release_id=release_id or uuid.uuid4().hex,
                tenant=tenant,
                build_id=build_id,
                attempt=attempt,
                kind=kind,
                sequence=sequence,
                state="preparing",
                created_at=now,
                settled_at=None,
            )
            try:
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
            except self._dialect.integrity_error as exc:
                conn.rollback()
                raise ReleaseError(
                    f"release {release.release_id}, or {kind} sequence {sequence} of build "
                    f"{build_id}, already exists"
                ) from exc
            conn.commit()
            return release
        finally:
            conn.close()

    def attach(self, release_id: str, attempt: str, versions: Sequence[tuple[str, int]]) -> int:
        """Stage existing ``building`` rows under the release (a build's own target row).

        A row created with ``release_id`` needs no attaching. Returns how many rows
        were attached; a row that is ready or under another release is left alone.

        Raises:
            ValueError: the release does not exist or is not *attempt*'s.
        """
        conn = self._connect()
        try:
            self._own(conn, release_id, attempt)
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

    def mark_prepared(self, release_id: str, attempt: str) -> bool:
        """The producer says every output is staged; ``prepared`` until accepted.

        Raises:
            ValueError: the release does not exist or is not *attempt*'s.
        """
        conn = self._connect()
        try:
            self._own(conn, release_id, attempt)
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
            ReleaseRejected: the attempt lost the lease, or the build stopped
                building under it; the release is now rejected.
            ReleaseOutOfOrder: an earlier release of the kind is still unsettled.
            ReleaseIncomplete: an output is not staged under this release, its
                bytes are missing or the wrong size, or a row staged under the
                release is not in *staged*.
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
                self._run(conn, operations, release.tenant, release.release_id, now)
                self._artifacts._promote_staged(conn, list(staged), now)
                if release.kind == "final":
                    target = conn.execute(
                        "SELECT artifact_id, version FROM artifact_builds WHERE build_id = ?",
                        (release.build_id,),
                    ).fetchone()
                    if not self._builds.complete_attempt_within(
                        conn,
                        release.build_id,
                        release.attempt,
                        artifact_id=staged[0].artifact_id if staged else None,
                        version=staged[0].version if staged else None,
                        now=now,
                    ):
                        # Every build writer here takes the build's key, but a consumer's
                        # own transaction need not: the release must not outlive the build.
                        conn.rollback()
                        self._settle(release_id, "rejected")
                        raise ReleaseRejected(
                            f"build {release.build_id} is no longer building under attempt "
                            f"{release.attempt}"
                        )
                    # A build made before its outputs existed targets a placeholder row;
                    # once the build points at a real output (or ends with none) the
                    # placeholder is a failed row for GC, not a building one forever.
                    if not staged or (target["artifact_id"], target["version"]) != (
                        staged[0].artifact_id,
                        staged[0].version,
                    ):
                        conn.execute(
                            "UPDATE artifact_versions SET state = 'failed' "
                            "WHERE id = ? AND version = ? AND state = 'building'",
                            (target["artifact_id"], target["version"]),
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

    def abandon(self, release_id: str, attempt: str) -> bool:
        """Give up an unsettled release; its staged rows stay ``building`` for GC.

        Raises:
            ValueError: the release does not exist or is not *attempt*'s.
        """
        conn = self._connect()
        try:
            self._own(conn, release_id, attempt)
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
        from_release: str | None = None,
    ) -> int:
        """Continue the build in a new attempt, fencing every earlier one.

        The build keeps its identity and its accepted releases; the old attempt can no
        longer release, and whatever it left unsettled is abandoned. ``from_release``
        is the accepted release of this build the new attempt starts from, recorded
        on the attempt (ADR 001 C: an explicit release, never an alias). Returns how
        many releases were abandoned.

        Raises:
            ValueError: the build is not pending or building, another attempt's lease
                on it is still live, or ``from_release`` is not an accepted release
                of it.
        """
        conn = self._connect()
        try:
            self._dialect.begin_write(conn, build_id)
            now = self._clock()
            if from_release is not None:
                source = self._get_on(conn, from_release)
                if source is None or source.build_id != build_id or source.state != "accepted":
                    conn.rollback()
                    raise ValueError(
                        f"release {from_release} is not an accepted release of build {build_id}"
                    )
            if not self._builds.open_attempt_within(
                conn,
                build_id,
                attempt,
                writable_until=writable_until,
                lease_owner=lease_owner,
                lease_expires_at=now + lease_duration_seconds,
                now=now,
                resumed_from=from_release,
            ):
                conn.rollback()
                raise ValueError(
                    f"build {build_id} is not active, or another attempt's lease on it is live"
                )
            cursor = conn.execute(
                "UPDATE build_releases SET state = 'abandoned', settled_at = ? "
                "WHERE build_id = ? AND attempt <> ? AND state IN ('preparing', 'prepared')",
                (now, build_id, attempt),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def apply(
        self,
        build_id: str,
        attempt: str,
        operations: Sequence[Operation],
        *,
        tenant: str | None = None,
    ) -> None:
        """Run *operations* as the build's *attempt*, outside any acceptance.

        What a producer does between releases, such as pointing a head at a checkpoint
        accepted earlier (its own or an earlier attempt's). Fenced like acceptance:
        *attempt* must hold the build's lease. All or nothing; an audit event names
        the build. *tenant* defaults to the build's and may not differ from it.

        Raises:
            ReleaseRejected: the attempt does not hold the lease.
            OperationFailed: an operation's condition did not hold.
            ValueError: *tenant* is not the build's.
        """
        conn = self._connect()
        try:
            self._dialect.begin_write(conn, build_id)
            now = self._clock()
            if not lease_held(conn, build_id, attempt, now):
                conn.rollback()
                raise ReleaseRejected(f"attempt {attempt} does not hold build {build_id}")
            tenant = self._release_tenant(conn, build_id, tenant)
            try:
                self._run(conn, operations, tenant, build_id, now)
                conn.commit()
            except OperationFailed:
                conn.rollback()
                raise
            except self._dialect.integrity_error as exc:
                conn.rollback()
                raise OperationFailed(f"the ledger refused the operations: {exc}") from exc
        finally:
            conn.close()

    # --- Reading ---

    def latest_accepted(self, build_id: str, kind: str = "checkpoint") -> Release | None:
        """The build's accepted release of *kind* with the highest sequence, or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM build_releases WHERE build_id = ? AND kind = ? "
                "AND state = 'accepted' ORDER BY sequence DESC LIMIT 1",
                (build_id, kind),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else _release_from_row(row)

    def collectable_checkpoints(self, *, tenant: str | None = None) -> list[tuple[str, int]]:
        """The checkpoint outputs nothing but a retention root may still need.

        Every output of an accepted ``checkpoint`` release, except the newest accepted
        checkpoint of each build still ``building`` (what a resume starts from). Roots
        are the collector's to honor; this only says which rows are not current values.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT av.id, av.version
                FROM artifact_versions av
                JOIN build_releases r ON r.release_id = av.release_id
                JOIN artifact_builds b ON b.build_id = r.build_id
                WHERE r.kind = 'checkpoint' AND r.state = 'accepted'
                  AND COALESCE(av.tenant, '') = COALESCE(?, '')
                  AND NOT (
                    b.state = 'building'
                    AND r.sequence = (
                      SELECT MAX(sequence) FROM build_releases
                      WHERE build_id = r.build_id AND kind = 'checkpoint'
                        AND state = 'accepted'
                    )
                  )
                ORDER BY av.id, av.version
                """,
                (tenant,),
            ).fetchall()
        finally:
            conn.close()
        return [(row["id"], row["version"]) for row in rows]

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

    def _own(self, conn: StoreConnection, release_id: str, attempt: str) -> Release:
        """The release, which must be *attempt*'s own: no other attempt touches it."""
        release = self._get_on(conn, release_id)
        if release is None:
            raise ValueError(f"release {release_id} does not exist")
        if release.attempt != attempt:
            raise ValueError(
                f"release {release_id} is attempt {release.attempt}'s, not {attempt}'s"
            )
        return release

    def _release_tenant(self, conn: StoreConnection, build_id: str, tenant: str | None) -> str:
        """The tenant a release of the build runs under: the build's, or *tenant* if the same."""
        owner = _tenant_of(conn, build_id)
        if tenant is not None and tenant != owner:
            conn.rollback()
            raise ValueError(f"build {build_id} belongs to tenant {owner!r}, not {tenant!r}")
        return owner

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
        """Every output is a building row of this release whose bytes are all there,
        and every building row of the release is an output."""
        blobs = self._artifacts.blob_store
        declared = {(item.artifact_id, item.version) for item in staged}
        rows = conn.execute(
            "SELECT id, version FROM artifact_versions WHERE release_id = ? AND state = 'building'",
            (release.release_id,),
        ).fetchall()
        undeclared = [
            f"{r['id']}@v={r['version']}" for r in rows if (r["id"], r["version"]) not in declared
        ]
        if undeclared:
            conn.rollback()
            raise ReleaseIncomplete(
                f"staged under release {release.release_id} but not declared: "
                + ", ".join(undeclared)
            )
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

    def _run(
        self,
        conn: StoreConnection,
        operations: Sequence[Operation],
        tenant: str,
        subject: str,
        now: float,
    ) -> None:
        """The operations in order on the caller's transaction; *subject* names an audit row."""
        if any(isinstance(op, InsertAuditEvent) for op in operations):
            # Before the transaction's first write, as the audit lock requires.
            self._artifacts._serialize_audit(conn)
        asserted: dict[str, int] = {}
        for op in operations:
            self._apply(conn, op, tenant, subject, now, asserted)

    def _apply(
        self,
        conn: StoreConnection,
        op: Operation,
        tenant: str,
        subject: str,
        now: float,
        asserted: dict[str, int],
    ) -> None:
        """One operation; *asserted* carries each asserted revision to the advance it guards."""
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
            asserted[op.name] = revision
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
            self._advance(conn, tenant, op.name, op.release_id, now, asserted.get(op.name))
            if op.name in asserted:
                asserted[op.name] += 1
        elif isinstance(op, InsertAuditEvent):
            conn.execute(
                "INSERT INTO registry_audit (at, actor, action, name, value, tenant) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    now,
                    op.event.principal,
                    op.event.action,
                    subject,
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
        conn: StoreConnection,
        tenant: str,
        name: str,
        release_id: str,
        now: float,
        expected: int | None,
    ) -> None:
        """Point the reference at the release and hold its outputs as the reference's root.

        With *expected* (the revision this transaction asserted), the move is one
        statement guarded on it: a reference another transaction moved meanwhile, a
        plain read would not see on Postgres, moves nobody and the operation fails.
        """
        if expected is None:
            conn.execute(
                "INSERT INTO release_references (tenant, name, release_id, revision, updated_at) "
                "VALUES (?, ?, ?, 1, ?) ON CONFLICT (tenant, name) DO UPDATE SET "
                "release_id = excluded.release_id, revision = release_references.revision + 1, "
                "updated_at = excluded.updated_at",
                (tenant, name, release_id, now),
            )
        elif expected == 0:
            moved = conn.execute(
                "INSERT INTO release_references (tenant, name, release_id, revision, updated_at) "
                "VALUES (?, ?, ?, 1, ?) ON CONFLICT (tenant, name) DO NOTHING",
                (tenant, name, release_id, now),
            ).rowcount
            if moved == 0:
                raise OperationFailed(f"reference {name} was created by another release")
        else:
            moved = conn.execute(
                "UPDATE release_references SET release_id = ?, revision = revision + 1, "
                "updated_at = ? WHERE tenant = ? AND name = ? AND revision = ?",
                (release_id, now, tenant, name, expected),
            ).rowcount
            if moved == 0:
                raise OperationFailed(f"reference {name} moved past revision {expected}")
        conn.execute(
            "DELETE FROM retention_roots WHERE tenant = ? AND kind = ? AND holder = ?",
            (tenant, HEAD_ROOT_KIND, name),
        )
        conn.execute(
            "INSERT INTO retention_roots (tenant, kind, holder, artifact_id, version, created_at) "
            "SELECT ?, ?, ?, id, version, ? FROM artifact_versions WHERE release_id = ?",
            (tenant, HEAD_ROOT_KIND, name, now, release_id),
        )
