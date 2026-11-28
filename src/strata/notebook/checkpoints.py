"""The checkpoints a cell's runs left in the notebook's ledger, for picking one to branch from.

Listed from the outputs still stored, not from the builds: prune collects a checkpoint's
outputs and leaves its release row, and a release with no outputs restores nothing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypedDict

from vestigo import checkpoint_files
from vestigo.releases import HEAD_ROOT_KIND, Release
from vestigo.store import ArtifactVersion

if TYPE_CHECKING:
    from strata.notebook.artifact_integration import NotebookArtifactManager

# A checkpoint output's id after ``..._var_``: the part, the run's variant if it has one,
# the sequence, and the file within a checkpoint of files (``local_runs._checkpoint``).
_OUTPUT = re.compile(r"[^@]+(?:@variant=(?P<variant>.+?))?@ckpt=\d+(?:@file=.*)?")


class Checkpoint(TypedDict):
    """One accepted checkpoint: what ``init=<release_id>`` would start a branch from."""

    release_id: str
    sequence: int
    created_at: float
    parts: list[str]
    files: int
    step: int | None
    byte_size: int
    heads: list[str]


class CheckpointRun(TypedDict):
    """A run (a build) and the checkpoints of it still stored, newest first."""

    build_id: str
    # running | interrupted | finished | failed, or None once the build's row is gone.
    status: str | None
    variant: str | None
    # The release this run branched from, if it is a branch.
    init: str | None
    started_at: float | None
    checkpoints: list[Checkpoint]


def cell_checkpoints(manager: NotebookArtifactManager, cell_id: str) -> list[CheckpointRun]:
    """*cell_id*'s runs that have a stored checkpoint, the newest checkpoint's run first."""
    prefix = f"nb_{manager.notebook_id}_cell_{cell_id}_var_"
    outputs: dict[str, list[ArtifactVersion]] = {}
    variants: dict[str, str | None] = {}
    for version in manager.artifact_store.list_versions_by_id_prefix(prefix):
        match = _OUTPUT.fullmatch(version.id[len(prefix) :])
        # Ready as this one read saw it: an acceptance makes a release's outputs ready
        # together, so a checkpoint accepted after it is left to the next listing whole,
        # not listed now with the parts staged so far.
        if match is None or version.state != "ready" or version.release_id is None:
            continue
        outputs.setdefault(version.release_id, []).append(version)
        variants[version.release_id] = match["variant"]
    if not outputs:
        return []

    # One read per run, not per checkpoint: a training run leaves hundreds.
    releases: dict[str, Release] = {}
    for release_id in outputs:
        if release_id in releases:
            continue
        release = manager.release_store.get(release_id)
        if release is not None:
            releases.update(
                (r.release_id, r)
                for r in manager.release_store.releases_of(release.build_id, "checkpoint")
            )

    # A head's roots are the outputs of the release it points at (``LocalRuns.head_name``).
    heads: dict[tuple[str, int], set[str]] = {}
    for root in manager.artifact_store.retention_roots(kind=HEAD_ROOT_KIND):
        name = root["holder"].rpartition("/")[2]
        heads.setdefault((root["artifact_id"], root["version"]), set()).add(name)

    now = manager.build_store.now()
    runs: dict[str, CheckpointRun] = {}
    for release_id, versions in outputs.items():
        release = releases.get(release_id)
        if release is None or release.state != "accepted":
            continue
        run = runs.get(release.build_id)
        if run is None:
            run = runs[release.build_id] = _run(
                manager, release.build_id, variants[release_id], now
            )
        run["checkpoints"].append(
            Checkpoint(
                release_id=release_id,
                sequence=release.sequence,
                created_at=release.settled_at or release.created_at,
                parts=sorted(
                    v.params["variable_name"] for v in versions if "variable_name" in v.params
                ),
                files=len(checkpoint_files.files_of(versions)),
                step=checkpoint_files.step_of(versions),
                byte_size=sum(v.byte_size or 0 for v in versions),
                heads=sorted(set().union(*(heads.get((v.id, v.version), ()) for v in versions))),
            )
        )
    for run in runs.values():
        run["checkpoints"].sort(key=lambda c: c["sequence"], reverse=True)
    return sorted(runs.values(), key=lambda r: r["checkpoints"][0]["created_at"], reverse=True)


def _run(
    manager: NotebookArtifactManager, build_id: str, variant: str | None, now: float
) -> CheckpointRun:
    build = manager.build_store.get_build(build_id)
    status: str | None = None
    if build is not None and build.state == "building":
        # Live through its deadline, as the ledger reads a lease (``releases.lease_held``).
        live = build.lease_expires_at is not None and build.lease_expires_at >= now
        status = "running" if live else "interrupted"
    elif build is not None:
        status = {"ready": "finished", "failed": "failed"}.get(build.state)
    return CheckpointRun(
        build_id=build_id,
        status=status,
        variant=variant,
        init=(build.params or {}).get("init") if build is not None else None,
        started_at=build.created_at if build is not None else None,
        checkpoints=[],
    )
