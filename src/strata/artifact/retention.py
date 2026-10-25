"""Strata's side of the ledger's retention (ADR 002 §7.4).

What a version requires beside its lineage is declared by its writer as data, and the
ledger reads no blob to learn it. A ``# @output_dir`` manifest requires the file versions
it names; every path that stores a manifest passes :func:`requirements` to the ledger.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vestigo.store import ArtifactStore

logger = logging.getLogger(__name__)

FOLDER_EDGES_UPGRADE = "folder_manifests_require_their_files"


def requirements(
    artifact_id: str, content_type: str | None, blob: bytes | Path | None
) -> list[tuple[str, int]]:
    """The versions a blob stored as *artifact_id* requires: a folder's files.

    Empty for any other content type, and for a manifest that cannot be read: an import
    can hold any bytes under the folder type, and such a manifest names nothing, so it
    holds nothing.
    """
    from strata.notebook import output_dirs

    if content_type != output_dirs.CONTENT_TYPE or blob is None:
        return []
    data = blob.read_bytes() if isinstance(blob, Path) else blob
    try:
        return [(file_id, v) for file_id, v, _ in output_dirs.file_refs(artifact_id, data)]
    except ValueError:
        logger.warning("%s is not a readable folder manifest; it keeps no files", artifact_id)
        return []


def complete_upgrade(store: ArtifactStore) -> None:
    """Record the requirements of folders stored before the ledger kept them as edges.

    A store from before ``retention_edges`` holds manifests whose files only their bytes
    name. The ledger reads no blob, so Strata reads each stored manifest once, writes
    the edges in one transaction and marks the step done in the store itself; a crash
    before the mark redoes the step, and a fresh store marks it on first open.
    """
    from strata.notebook import output_dirs

    if store.upgrade_applied(FOLDER_EDGES_UPGRADE):
        return
    declared: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for manifest in store.versions_declaring(output_dirs.CONTENT_TYPE):
        blob = store.read_blob(manifest.id, manifest.version, note_use=False)
        required = requirements(manifest.id, output_dirs.CONTENT_TYPE, blob)
        if required:
            declared[(manifest.id, manifest.version)] = required
    store.require_many(declared)
    store.mark_upgrade_applied(FOLDER_EDGES_UPGRADE)
    if declared:
        logger.info("Recorded the files of %d folder manifests stored before edges", len(declared))
