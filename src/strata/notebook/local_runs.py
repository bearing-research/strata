"""Every local cell run is an execution of the ledger (ADR 002 E4b).

Before the harness starts, the run gets a build and an attempt whose lease this process
renews for the cell's whole duration. The cell's checkpoint events (``run_channel``)
become ``checkpoint`` releases while it runs; its outputs become the ``final`` release
when it returns. A run interrupted after a checkpoint leaves its build ``building`` with
a lapsed lease, and the next normal run of the same computation resumes it in a new
attempt, starting from the newest accepted checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio.to_thread
from vestigo.operations import AdvanceReference
from vestigo.provenance import TransformSpec, checkpoint_output_key, run_output_key
from vestigo.releases import ReleaseRejected
from vestigo.store import ArtifactVersion, StagedVersion

from strata.artifact.retention import requirements
from strata.notebook.artifact_integration import caller_principal_id
from strata.notebook.harness_user import open_tree_file
from strata.notebook.process_tree import terminate_subprocess_tree

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
LEASE_OWNER = f"{socket.gethostname()}:{os.getpid()}"


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

    finalized: bool = False

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
        writable_until = now + timeout_seconds + WRITE_WINDOW_MARGIN_SECONDS
        attempt = uuid.uuid4().hex

        open_build = self.builds.find_open_build(computation_key, occurrence=occurrence)
        if open_build is not None:
            if open_build.lease_expires_at is not None and open_build.lease_expires_at > now:
                raise RunRefused(
                    f"Cell {cell_id} is already running (build {open_build.build_id}); "
                    "wait for it or cancel it before running it again."
                )
            latest = self.releases.latest_accepted(open_build.build_id)
            if resume and latest is not None:
                self.releases.resume(
                    open_build.build_id,
                    attempt,
                    writable_until=writable_until,
                    lease_owner=LEASE_OWNER,
                    lease_duration_seconds=LEASE_SECONDS,
                    from_release=latest.release_id,
                )
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
                )
            self.builds.fail_build(
                open_build.build_id,
                "superseded by a rerun"
                if latest is not None
                else "interrupted before a checkpoint",
            )

        build_id = uuid.uuid4().hex
        # A build references a version, so a run whose outputs do not exist yet targets a
        # placeholder the final release repoints it from (and fails).
        placeholder_id = f"nb_{self._manager.notebook_id}_cell_{cell_id}_run" + (
            f"@variant={variant}" if variant else ""
        )
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
        self.builds.create_build(
            build_id,
            placeholder_id,
            version,
            executor_ref=LOCAL_CELL_EXECUTOR_REF,
            principal_id=caller_principal_id(),
            params={"init": init} if init else None,
            occurrence=occurrence,
            computation_key=computation_key,
            dispatch="external",
        )
        self.builds.open_attempt(
            build_id,
            attempt,
            writable_until=writable_until,
            lease_owner=LEASE_OWNER,
            lease_duration_seconds=LEASE_SECONDS,
        )
        return LocalRun(
            build_id=build_id,
            attempt=attempt,
            cell_id=cell_id,
            computation_key=computation_key,
            variant=variant,
            input_versions=input_versions,
            init=init,
        )

    def start_renewing(self, run: LocalRun) -> asyncio.Task[None]:
        """Renew the attempt's lease until cancelled; the caller stops it with :meth:`close`."""
        return asyncio.create_task(self._renew(run), name=f"lease:{run.build_id}")

    async def _renew(self, run: LocalRun) -> None:
        while True:
            await asyncio.sleep(RENEW_EVERY_SECONDS)
            held = await anyio.to_thread.run_sync(
                self.builds.renew_lease, run.build_id, LEASE_OWNER, LEASE_SECONDS
            )
            if not held:
                logger.warning("Lost the lease on build %s of cell %s", run.build_id, run.cell_id)
                return

    def finalize(self, run: LocalRun, staged: list[StagedVersion]) -> list[ArtifactVersion]:
        """Make the run's staged outputs its ``final`` release; completes the build.

        Raises whatever acceptance raises (a fenced attempt, missing bytes), in which
        case the rows stay ``building`` for the caller to discard.
        """
        release = self.releases.prepare(run.build_id, run.attempt, "final")
        self.releases.attach(
            release.release_id, run.attempt, [(s.artifact_id, s.version) for s in staged]
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
        if self.releases.latest_accepted(run.build_id) is not None:
            # Resumable: the build stays building, its lease dropped now rather than left
            # to lapse, so the next run of the computation resumes it at once.
            self.builds.renew_lease(run.build_id, LEASE_OWNER, lease_duration_seconds=0.0)
            return
        self.builds.fail_build(
            run.build_id, "the run ended without a final release", lease_owner=LEASE_OWNER
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
        release = self.releases.prepare(run.build_id, run.attempt, "checkpoint", sequence=sequence)
        staged: list[StagedVersion] = []
        try:
            for name, spec in payload["parts"].items():
                with open_tree_file(run_dir, str(spec["file"])) as f:
                    blob = f.read()
                content_type = str(spec["content_type"])
                artifact_id = (
                    self._manager.cell_artifact_id(run.cell_id, name, variant=run.variant)
                    + f"@ckpt={sequence}"
                )
                version = self.store.create_artifact(
                    artifact_id,
                    checkpoint_output_key(run.computation_key, run.build_id, sequence, name),
                    TransformSpec(
                        executor=LOCAL_CELL_EXECUTOR_REF,
                        params={
                            "cell_id": run.cell_id,
                            "variable_name": name,
                            "content_type": content_type,
                            "checkpoint": str(sequence),
                        },
                        inputs=[],
                    ),
                    input_versions=run.input_versions,
                    principal=caller_principal_id(),
                    requires=requirements(artifact_id, content_type, blob),
                    release_id=release.release_id,
                    computation_key=run.computation_key,
                )
                self.store.blob_store.write_blob(artifact_id, version, blob)
                staged.append(
                    StagedVersion(
                        artifact_id=artifact_id,
                        version=version,
                        schema_json="",
                        row_count=None,
                        byte_size=len(blob),
                        content_sha256=hashlib.sha256(blob).hexdigest(),
                    )
                )
            self.releases.accept(release.release_id, staged)
        except ReleaseRejected:
            raise
        except Exception:
            # A checkpoint is a convenience the outputs do not depend on: the cell goes on.
            logger.exception("Checkpoint %d of cell %s was not released", sequence, run.cell_id)
            self.releases.abandon(release.release_id, run.attempt)
            self._manager.discard_cell_outputs(staged)
            return
        logger.info(
            "Cell %s released checkpoint %d of build %s", run.cell_id, sequence, run.build_id
        )

    def _head(self, run: LocalRun, payload: dict[str, Any]) -> None:
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
            try:
                await on_event(json.loads(line))
            except RunFenced as exc:
                await terminate_subprocess_tree(proc)
                return exc
    finally:
        transport.close()
