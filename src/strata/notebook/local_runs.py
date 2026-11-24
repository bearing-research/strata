"""Every local cell run is an execution of the ledger (ADR 002 E4b).

Before the harness starts, the run gets a build and an attempt whose lease this process
renews for the cell's whole duration. The cell's checkpoint events (``run_channel``)
become ``checkpoint`` releases while it runs (a checkpoint of files in the layout of
``vestigo.checkpoint_files``, so ``vestigo.torch`` reads it); its outputs become the
``final`` release when it returns. A run interrupted after a checkpoint leaves its
build ``building`` with a lapsed lease, and the next normal run of the same computation
resumes it in a new attempt, starting from the newest accepted checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import anyio.to_thread
from vestigo import checkpoint_files
from vestigo.operations import AdvanceReference
from vestigo.provenance import TransformSpec, checkpoint_output_key, run_output_key
from vestigo.releases import ReleaseRejected
from vestigo.store import ArtifactVersion, StagedVersion

from strata.artifact.retention import requirements
from strata.notebook.artifact_integration import caller_principal_id
from strata.notebook.harness_user import open_tree_file
from strata.notebook.process_tree import terminate_subprocess_tree
from strata.notebook.run_channel import CHECKPOINTS_DIR, FILES_PART

if TYPE_CHECKING:
    from strata.notebook.artifact_integration import NotebookArtifactManager

logger = logging.getLogger(__name__)

LOCAL_CELL_EXECUTOR_REF = "notebook/cell@v1"
LEASE_SECONDS = 60.0
RENEW_EVERY_SECONDS = 20.0
# How long after the harness exits the executor waits for EOF on its event pipe.
EVENTS_DRAIN_SECONDS = 5.0
# Past the cell's timeout: the final release is accepted after the harness returns.
WRITE_WINDOW_MARGIN_SECONDS = 300.0
RESTORE_DIR = "restore"
"""Under the run directory: the parts ``strata.restore()`` returns."""
_PROCESS = f"{socket.gethostname()}:{os.getpid()}"


def lease_owner(attempt: str) -> str:
    """The lease owner of *attempt*: the attempt, so its renewal or close is refused once
    another attempt holds the build. The process alone repeats: two containers given one
    hostname both run as pid 1 (SYSTEMS LR-3)."""
    return f"{_PROCESS}/{attempt}"


class RunRefused(RuntimeError):
    """The computation is already running on this server; a second run would be fenced."""


class RunFenced(RuntimeError):
    """The run's attempt lost the build; nothing more it publishes counts."""


@dataclass
class LocalRun:
    """One local execution: a build, its current attempt, and what the cell starts from."""

    build_id: str
    attempt: str
    cell_id: str
    computation_key: str
    variant: str | None
    input_versions: dict[str, str]
    resumed_from: str | None = None
    init: str | None = None
    # How long an attempt of the run may write, from when it opens.
    writable_seconds: float = WRITE_WINDOW_MARGIN_SECONDS

    finalized: bool = False
    hold_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def restores_from(self) -> str | None:
        """The accepted release whose parts ``strata.restore()`` returns, if any."""
        return self.resumed_from or self.init


class LocalRuns:
    """Opens, renews and releases local runs against one notebook's store."""

    def __init__(self, manager: NotebookArtifactManager) -> None:
        self._manager = manager
        self.store = manager.artifact_store
        self.builds = manager.build_store
        self.releases = manager.release_store

    # --- Lifecycle ---

    def open(
        self,
        *,
        cell_id: str,
        computation_key: str,
        variant: str | None,
        input_versions: dict[str, str],
        timeout_seconds: float,
        resume: bool,
        init: str | None = None,
    ) -> LocalRun:
        """Start the run's execution: resume the computation's open build or create one.

        With ``resume`` (a normal run), a ``building`` build of this computation whose
        lease has lapsed and that has an accepted checkpoint is continued in a new
        attempt; otherwise, and on a rerun, such a build is failed and a new one opens.
        ``init`` is the accepted release a branch starts from (part of the computation
        key already; recorded on the build).

        Raises:
            RunRefused: the computation's build still holds a live lease.
        """
        occurrence = f"variant={variant}" if variant else None
        now = time.time()
        writable_seconds = timeout_seconds + WRITE_WINDOW_MARGIN_SECONDS
        writable_until = now + writable_seconds
        attempt = uuid.uuid4().hex

        # Another process may resume, fail or open a build between this read and the write
        # below; each write is pinned to what was read, and on a miss the build is read
        # again (a resume or an open now holds a live lease, so the next pass refuses). A
        # miss with the claim unchanged is not that race, and raises.
        missed: tuple[tuple[Any, ...], Exception] | None = None
        while True:
            open_build = self.builds.find_open_build(computation_key, occurrence=occurrence)
            if open_build is None:
                build_id = self._new_build(
                    cell_id, computation_key, variant, attempt, writable_until, init=init
                )
                if build_id is not None:
                    return LocalRun(
                        build_id=build_id,
                        attempt=attempt,
                        cell_id=cell_id,
                        computation_key=computation_key,
                        variant=variant,
                        input_versions=input_versions,
                        init=init,
                        writable_seconds=writable_seconds,
                    )
                now = time.time()
                continue
            claim = (open_build.build_id, open_build.lease_owner, open_build.lease_expires_at)
            if missed is not None and missed[0] == claim:
                raise missed[1]
            self._refuse_if_live(cell_id, open_build, now)
            latest = self.releases.latest_accepted(open_build.build_id)
            if resume and latest is not None:
                try:
                    self.releases.resume(
                        open_build.build_id,
                        attempt,
                        writable_until=writable_until,
                        lease_owner=lease_owner(attempt),
                        lease_duration_seconds=LEASE_SECONDS,
                        from_release=latest.release_id,
                    )
                except ValueError as exc:
                    missed, now = (claim, exc), time.time()
                    continue
                logger.info(
                    "Cell %s resumes build %s from checkpoint %d",
                    cell_id,
                    open_build.build_id,
                    latest.sequence,
                )
                return LocalRun(
                    build_id=open_build.build_id,
                    attempt=attempt,
                    cell_id=cell_id,
                    computation_key=computation_key,
                    variant=variant,
                    input_versions=input_versions,
                    resumed_from=latest.release_id,
                    init=open_build.params.get("init") if open_build.params else None,
                    writable_seconds=writable_seconds,
                )
            if self.builds.fail_build(
                open_build.build_id,
                "superseded by a rerun"
                if latest is not None
                else "interrupted before a checkpoint",
                lease_owner=open_build.lease_owner,
                lease_expires_at=open_build.lease_expires_at,
            ):
                continue
            missed = (claim, RuntimeError(f"build {open_build.build_id} could not be failed"))
            now = time.time()

    def _new_build(
        self,
        cell_id: str,
        computation_key: str,
        variant: str | None,
        attempt: str,
        writable_until: float,
        *,
        init: str | None = None,
        beside: bool = False,
    ) -> str | None:
        """Create a build of the computation with *attempt* open on it; its id, or None when
        another process opened a build of it first (SYSTEMS LR-2). With *beside*, it opens
        whatever else of the computation is open."""
        occurrence = f"variant={variant}" if variant else None
        build_id = uuid.uuid4().hex
        # A build references a version, so a run whose outputs do not exist yet targets a
        # placeholder the final release repoints it from (and fails).
        placeholder_id = self._manager.run_placeholder_id(cell_id, variant)
        version = self.store.create_artifact(
            placeholder_id,
            run_output_key(computation_key, build_id, "run"),
            TransformSpec(
                executor=LOCAL_CELL_EXECUTOR_REF,
                params={"cell_id": cell_id, "build_id": build_id},
                inputs=[],
            ),
            principal=caller_principal_id(),
            minted=True,
            computation_key=computation_key,
        )
        try:
            # One transaction under the computation's lock, so of two processes that both
            # found nothing open, one opens a build and the other gets it back.
            opened = self.builds.open_build(
                build_id,
                placeholder_id,
                version,
                attempt,
                computation_key=computation_key,
                executor_ref=LOCAL_CELL_EXECUTOR_REF,
                writable_until=writable_until,
                lease_owner=lease_owner(attempt),
                lease_duration_seconds=LEASE_SECONDS,
                occurrence=occurrence,
                principal_id=caller_principal_id(),
                params={"init": init} if init else None,
                principal=caller_principal_id(),
                beside=beside,
            )
        except BaseException as exc:
            # The open may have committed with its reply lost: its live lease would refuse
            # every run for a minute, so fail it. The open's error is the one raised.
            try:
                if self.builds.get_build(build_id) is not None:
                    self.builds.fail_build(
                        build_id, f"its open raised: {exc!r}", lease_owner=lease_owner(attempt)
                    )
                self.store.fail_artifact(placeholder_id, version)
            except Exception as cleanup:
                exc.add_note(f"Failing the run's placeholder raised too: {cleanup!r}")
            raise
        if opened.build_id != build_id:
            self.store.fail_artifact(placeholder_id, version)
            return None
        return build_id

    def refuse_if_running(self, cell_id: str, computation_key: str) -> None:
        """Raise :class:`RunRefused` while a run of *computation_key* holds a live lease.

        For a scratchpad cell, whose run is a build of its lookup key while its outputs,
        and so a revert of them, are under the key over what it read.
        """
        open_build = self.builds.find_open_build(computation_key, occurrence=None)
        if open_build is not None:
            self._refuse_if_live(cell_id, open_build, time.time())

    @staticmethod
    def _refuse_if_live(cell_id: str, open_build: Any, now: float) -> None:
        if open_build.lease_expires_at is not None and open_build.lease_expires_at > now:
            raise RunRefused(
                f"Cell {cell_id} is already running (build {open_build.build_id}); "
                "wait for it or cancel it before running it again."
            )

    def promote(
        self,
        *,
        cell_id: str,
        computation_key: str,
        variant: str | None,
        sources: list[tuple[str, int]],
    ) -> list[ArtifactVersion] | None:
        """Make earlier versions of a cell's outputs current again, together: a revert hit.

        Each ``(artifact_id, version)`` is copied (``ArtifactStore.stage_promotion``) into
        one :meth:`release`; returns the copies in *sources*' order, or None as it does.
        """

        def copies(release_id: str, staged: list[StagedVersion]) -> str | None:
            for artifact_id, version in sources:
                row = self.store.stage_promotion(artifact_id, version, release_id=release_id)
                if row is None:
                    return f"{artifact_id}@v={version} is gone"
                staged.append(row)
            return None

        return self.release(
            cell_id=cell_id,
            computation_key=computation_key,
            variant=variant,
            what="revert",
            stage=copies,
        )

    def release(
        self,
        *,
        cell_id: str,
        computation_key: str,
        variant: str | None,
        what: str,
        stage: Callable[[str, list[StagedVersion]], str | None],
    ) -> list[ArtifactVersion] | None:
        """Make a result this process did not compute current, all of it or none: a build.

        A revert or a team-cache pull. *stage* writes each output as a ``building`` row under
        the build's ``final`` release id, appending it to the list it is given, and returns
        why it could not stage them all, or None. Accepting the release makes every row
        current in one transaction. Returns the rows in the order staged, or None when
        nothing was made current: the computation is running (its own release decides),
        *stage* gave a reason, or a newer value of an output was made current first.
        """
        occurrence = f"variant={variant}" if variant else None
        running = self.builds.find_open_build(computation_key, occurrence=occurrence)
        if running is not None and (running.lease_expires_at or 0.0) > time.time():
            logger.info("Cell %s's %s waits for its run, build %s", cell_id, what, running.build_id)
            return None
        attempt = uuid.uuid4().hex
        # Beside a lapsed build (a run that died); with none open, a run that opens
        # meanwhile comes back and this waits for it.
        build_id = self._new_build(
            cell_id,
            computation_key,
            variant,
            attempt,
            time.time() + WRITE_WINDOW_MARGIN_SECONDS,
            beside=running is not None,
        )
        if build_id is None:
            logger.info("Cell %s's %s waits for a run that just opened", cell_id, what)
            return None
        release = self.releases.prepare(build_id, attempt, "final")
        staged: list[StagedVersion] = []
        try:
            reason = stage(release.release_id, staged)
            if reason is None:
                try:
                    bundle = self.releases.accept(release.release_id, staged)
                except ReleaseRejected as exc:
                    reason = str(exc)
                else:
                    by_key = {(v.id, v.version): v for v in bundle.outputs}
                    return [by_key[(row.artifact_id, row.version)] for row in staged]
        except Exception:
            # Left open, its lease would refuse a run of the computation for a minute.
            self._end_unreleased(build_id, release.release_id, attempt, staged, "it raised")
            raise
        logger.info("Cell %s's %s was not made current: %s", cell_id, what, reason)
        self._end_unreleased(build_id, release.release_id, attempt, staged, f"{what}: {reason}")
        return None

    def _end_unreleased(
        self,
        build_id: str,
        release_id: str,
        attempt: str,
        staged: list[StagedVersion],
        reason: str,
    ) -> None:
        """Fail a build that made nothing current: its release, rows, placeholder, build."""
        # Settled already when acceptance refused it.
        self.releases.abandon(release_id, attempt)
        build = self.builds.get_build(build_id)
        rows = [(row.artifact_id, row.version) for row in staged]
        if build is not None:
            rows.append((build.artifact_id, build.version))
        for artifact_id, version in rows:
            self.store.fail_artifact(artifact_id, version)
        self.builds.fail_build(
            build_id, f"nothing made current: {reason}", lease_owner=lease_owner(attempt)
        )

    def start_renewing(self, run: LocalRun) -> asyncio.Task[None]:
        """Renew the attempt's lease until cancelled; the caller stops it with :meth:`close`."""
        return asyncio.create_task(self._renew(run), name=f"lease:{run.build_id}")

    async def _renew(self, run: LocalRun) -> None:
        while True:
            await asyncio.sleep(RENEW_EVERY_SECONDS)
            held = await anyio.to_thread.run_sync(self._hold, run)
            if not held:
                logger.warning("Lost the lease on build %s of cell %s", run.build_id, run.cell_id)
                return

    def _hold(self, run: LocalRun) -> bool:
        """Whether the run's attempt holds its build's lease, renewed now.

        A lease lapses with nothing wrong: a laptop asleep, a loop blocked past it. A
        lapsed lease stays lapsed, so the run goes on in a new attempt of its build,
        unless another process took the build meanwhile: ``open_attempt`` refuses
        while that one's lease is live. One hold at a time per run: the renewal task
        and a release both hold, and between one thread opening an attempt and
        assigning it, the other's renewal names the old attempt and fails
        (``formal/tla/LocalRun.tla``).
        """
        with run.hold_lock:
            if self.builds.renew_lease(run.build_id, lease_owner(run.attempt), LEASE_SECONDS):
                return True
            attempt = uuid.uuid4().hex
            if not self.builds.open_attempt(
                run.build_id,
                attempt,
                writable_until=time.time() + run.writable_seconds,
                lease_owner=lease_owner(attempt),
                lease_duration_seconds=LEASE_SECONDS,
                principal=caller_principal_id(),
            ):
                return False
            logger.warning(
                "The lease on build %s of cell %s lapsed; the run goes on in attempt %s",
                run.build_id,
                run.cell_id,
                attempt,
            )
            run.attempt = attempt
            return True

    def finalize(self, run: LocalRun, staged: list[StagedVersion]) -> list[ArtifactVersion]:
        """Make the run's staged outputs its ``final`` release; completes the build.

        Raises whatever acceptance raises (a fenced attempt, missing bytes), in which
        case the rows stay ``building`` for the caller to discard.
        """
        # Not held, the release is refused below: a fence has one path.
        self._hold(run)
        release = self.releases.prepare(run.build_id, run.attempt, "final")
        self.releases.attach(
            release.release_id, release.attempt, [(s.artifact_id, s.version) for s in staged]
        )
        bundle = self.releases.accept(release.release_id, staged)
        run.finalized = True
        by_key = {(v.id, v.version): v for v in bundle.outputs}
        return [by_key[(s.artifact_id, s.version)] for s in staged]

    async def close(self, run: LocalRun, renewal: asyncio.Task[None]) -> None:
        """Stop renewing; a build that never released finally is failed unless resumable."""
        renewal.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await renewal
        if run.finalized:
            return
        await anyio.to_thread.run_sync(self._end_without_final, run)

    def _end_without_final(self, run: LocalRun) -> None:
        # Under the hold lock: a hold in flight may be opening the attempt to name.
        with run.hold_lock:
            if self.releases.latest_accepted(run.build_id) is not None:
                # Resumable: the build stays building, its lease dropped now rather than
                # left to lapse, so the next run of the computation resumes it at once.
                self.builds.renew_lease(
                    run.build_id, lease_owner(run.attempt), lease_duration_seconds=0.0
                )
                return
            self.builds.fail_build(
                run.build_id,
                "the run ended without a final release",
                lease_owner=lease_owner(run.attempt),
            )

    # --- Events from the cell ---

    def event_handler(
        self, run: LocalRun, run_dir: Path
    ) -> Callable[[dict[str, Any]], Awaitable[None]]:
        """The coroutine the harness readers call with each event line's payload."""

        async def on_event(payload: dict[str, Any]) -> None:
            await anyio.to_thread.run_sync(self.handle, run, payload, run_dir)

        return on_event

    def handle(self, run: LocalRun, payload: dict[str, Any], run_dir: Path) -> None:
        """Apply one event: a checkpoint becomes a release, a head points at one.

        Raises:
            RunFenced: the attempt no longer holds the build.
        """
        kind = payload.get("event")
        try:
            if kind == "checkpoint":
                self._checkpoint(run, payload, run_dir)
            elif kind == "head":
                self._head(run, payload)
            else:
                logger.warning("Ignoring unknown run event %r from cell %s", kind, run.cell_id)
        except ReleaseRejected as exc:
            raise RunFenced(f"Cell {run.cell_id}'s run was fenced: {exc}") from exc

    def _checkpoint(self, run: LocalRun, payload: dict[str, Any], run_dir: Path) -> None:
        sequence = int(payload["sequence"])
        try:
            self._hold(run)
            release = self.releases.prepare(
                run.build_id, run.attempt, "checkpoint", sequence=sequence
            )
        except ReleaseRejected:
            raise
        except Exception:
            # As one that cannot be staged below: the cell goes on.
            logger.exception("Checkpoint %d of cell %s was not begun", sequence, run.cell_id)
            return
        staged: list[StagedVersion] = []
        try:
            if "files" in payload:
                self._stage_files(run, release.release_id, payload, run_dir, staged)
            for name, spec in payload.get("parts", {}).items():
                with open_tree_file(run_dir, str(spec["file"])) as f:
                    blob = f.read()
                artifact_id = (
                    self._manager.cell_artifact_id(run.cell_id, name, variant=run.variant)
                    + f"@ckpt={sequence}"
                )
                params = {"variable_name": name, "content_type": str(spec["content_type"])}
                staged.append(
                    self._stage(run, release.release_id, sequence, artifact_id, name, params, blob)
                )
            self.releases.accept(release.release_id, staged)
        except ReleaseRejected:
            raise
        except Exception:
            # A checkpoint is a convenience the outputs do not depend on: the cell goes on.
            logger.exception("Checkpoint %d of cell %s was not released", sequence, run.cell_id)
            self.releases.abandon(release.release_id, release.attempt)
            self._manager.discard_cell_outputs(staged)
            return
        logger.info(
            "Cell %s released checkpoint %d of build %s", run.cell_id, sequence, run.build_id
        )

    def _stage_files(
        self,
        run: LocalRun,
        release_id: str,
        payload: dict[str, Any],
        run_dir: Path,
        staged: list[StagedVersion],
    ) -> None:
        """Stage the files a checkpoint's writer left in its folder, then their manifest."""
        sequence, step = int(payload["sequence"]), int(payload["step"])
        prefix = (
            self._manager.cell_artifact_id(run.cell_id, FILES_PART, variant=run.variant)
            + f"@ckpt={sequence}"
        )
        listed: dict[str, tuple[int, str]] = {}
        for path in payload["files"]:
            rel = str(path)
            with open_tree_file(run_dir, f"{CHECKPOINTS_DIR}/{sequence}/{rel}") as f:
                blob = f.read()
            label = f"file={quote(rel, safe='')}"
            params = {
                "content_type": "application/octet-stream",
                **checkpoint_files.file_params(rel, step),
            }
            version = self._stage(
                run,
                release_id,
                sequence,
                f"{prefix}@{label}",
                f"{FILES_PART}@{label}",
                params,
                blob,
            )
            staged.append(version)
            assert version.content_sha256 is not None
            listed[rel] = (len(blob), version.content_sha256)
        params = {"content_type": "application/json", **checkpoint_files.manifest_params(step)}
        manifest = checkpoint_files.manifest(step, listed)
        staged.append(self._stage(run, release_id, sequence, prefix, FILES_PART, params, manifest))

    def _stage(
        self,
        run: LocalRun,
        release_id: str,
        sequence: int,
        artifact_id: str,
        output: str,
        params: dict[str, Any],
        blob: bytes,
    ) -> StagedVersion:
        """One output of the run's checkpoint *sequence*, written and staged in its release."""
        version = self.store.create_artifact(
            artifact_id,
            checkpoint_output_key(run.computation_key, run.build_id, sequence, output),
            TransformSpec(
                executor=LOCAL_CELL_EXECUTOR_REF,
                params={"cell_id": run.cell_id, **params, "checkpoint": str(sequence)},
                inputs=[],
            ),
            input_versions=run.input_versions,
            principal=caller_principal_id(),
            requires=requirements(artifact_id, params["content_type"], blob),
            release_id=release_id,
            computation_key=run.computation_key,
        )
        self.store.blob_store.write_blob(artifact_id, version, blob)
        return StagedVersion(
            artifact_id=artifact_id,
            version=version,
            schema_json="",
            row_count=None,
            byte_size=len(blob),
            content_sha256=hashlib.sha256(blob).hexdigest(),
        )

    def _head(self, run: LocalRun, payload: dict[str, Any]) -> None:
        self._hold(run)
        name = str(payload["name"])
        sequence = int(payload["sequence"])
        release = next(
            (
                r
                for r in self.releases.releases_of(run.build_id, "checkpoint")
                if r.sequence == sequence and r.state == "accepted"
            ),
            None,
        )
        if release is None:
            logger.warning(
                "Cell %s named head %s at checkpoint %d, which was not released",
                run.cell_id,
                name,
                sequence,
            )
            return
        self.releases.apply(
            run.build_id,
            run.attempt,
            [AdvanceReference(self.head_name(run, name), release.release_id)],
        )

    def head_name(self, run: LocalRun, name: str) -> str:
        """A head's reference name: scoped to the cell (and its variant)."""
        prefix = f"nb_{self._manager.notebook_id}_cell_{run.cell_id}"
        if run.variant:
            prefix += f"@variant={run.variant}"
        return f"{prefix}/{name}"


async def relay_events(
    read_fd: int,
    on_event: Callable[[dict[str, Any]], Awaitable[None]],
    proc: asyncio.subprocess.Process,
) -> RunFenced | None:
    """Feed a cold harness's event lines to *on_event* until EOF; returns the fence, if any.

    The harness holds the only write end (the executor closed its copy, the harness
    keeps it from its children), so EOF is the harness exiting. A fenced run is killed
    here: nothing it publishes afterwards counts.
    """
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=4 << 20)
    transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(read_fd, "rb", buffering=0)
    )
    try:
        while True:
            line = await reader.readline()
            if not line:
                return None
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                # One an interrupt tore; the channel starts the next on a line of its own.
                logger.warning("A run event line is not JSON: %.200r", line)
                continue
            try:
                await on_event(event)
            except RunFenced as exc:
                await terminate_subprocess_tree(proc)
                return exc
            except Exception:
                # The run has failed: the cell is stopped, as a warm worker is.
                logger.exception("A cold run's event failed; stopping the cell")
                await terminate_subprocess_tree(proc)
                raise
    finally:
        transport.close()
