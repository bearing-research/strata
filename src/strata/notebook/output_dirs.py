"""``# @output_dir <name>``: a folder a cell fills, kept as files and a manifest.

The cell gets an empty, writable folder in its run directory. When the run ends, each
regular file in it is stored once by its sha256, as its own artifact in the notebook's
store, and the output itself is a JSON manifest of paths, sizes and digests, so it has the
run's provenance and inputs like any output. A downstream cell gets the folder rebuilt as
a local path. A failed run keeps its files under a failed version of the manifest, which
is never a cache hit or an input.

Files in the folder are read without following links: under the harness user the cell
owns the run directory, and a link there could point at a server file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from strata.artifact.store import TransformSpec
from strata.notebook.harness_user import UnsafeRunFile, open_tree_file

logger = logging.getLogger(__name__)

OUTPUT_DIRS = "output_dirs"  # under a run directory, one folder per name
CONTENT_TYPE = "dir/manifest"
EXTENSION = ".dirmanifest.json"
_CHUNK = 1 << 20


def _regular_files(root: Path) -> list[str]:
    """Each regular file under *root*, as a POSIX path relative to it; links are not followed."""
    found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if stat.S_ISREG(path.lstat().st_mode):
                found.append(path.relative_to(root).as_posix())
    return sorted(found)


def file_artifact_id(artifact_mgr: Any, cell_id: str, sha256: str) -> str:
    return artifact_mgr.cell_artifact_id(cell_id, f"__file__{sha256}")


def _store_file(artifact_mgr: Any, cell_id: str, folder: Path, rel: str) -> dict[str, Any]:
    """Store one file by its digest, unless a ready copy is already there."""
    digest = hashlib.sha256()
    size = 0
    with open_tree_file(folder, rel) as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    sha256 = digest.hexdigest()
    store = artifact_mgr.artifact_store
    artifact_id = file_artifact_id(artifact_mgr, cell_id, sha256)
    existing = store.get_latest_version(artifact_id)
    if existing is not None and existing.state == "ready":
        return {"sha256": sha256, "size": size, "version": existing.version}

    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=sha256,
        transform_spec=TransformSpec(
            executor="notebook/cell@v1",
            params={"cell_id": cell_id, "content_type": "application/octet-stream"},
            inputs=[],
        ),
    )
    # Hashed again while copying: the file may have changed since the first pass.
    copied = hashlib.sha256()
    try:
        with (
            open_tree_file(folder, rel) as f,
            store.blob_store.open_blob_writer(artifact_id, version) as out,
        ):
            while chunk := f.read(_CHUNK):
                copied.update(chunk)
                out.write(chunk)
        if copied.hexdigest() != sha256:
            raise RuntimeError(f"{rel} changed while it was being stored")
    except BaseException:
        store.fail_artifact(artifact_id, version)
        raise
    store.finalize_artifact(artifact_id, version, "", None, size, content_sha256=sha256)
    return {"sha256": sha256, "size": size, "version": version}


def collect(artifact_mgr: Any, cell_id: str, folder: Path) -> bytes:
    """Store the folder's files and return its manifest: the writing cell and each file.

    A link, or a file with more than one hard link, is left out with a warning.
    """
    files: dict[str, dict[str, Any]] = {}
    if folder.is_dir():
        for rel in _regular_files(folder):
            try:
                files[rel] = _store_file(artifact_mgr, cell_id, folder, rel)
            except UnsafeRunFile:
                logger.warning("output folder %s: not keeping %s, a link", folder.name, rel)
    return json.dumps({"cell_id": cell_id, "files": files}, sort_keys=True).encode("utf-8")


def materialize(artifact_mgr: Any, manifest: bytes, dest: Path) -> None:
    """Rebuild the folder a manifest describes at *dest*, for a downstream cell to read."""
    store = artifact_mgr.artifact_store
    described = json.loads(manifest)
    cell_id = described["cell_id"]
    dest.mkdir(parents=True, exist_ok=True)
    for rel, entry in described["files"].items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        artifact_id = file_artifact_id(artifact_mgr, cell_id, entry["sha256"])
        reader = store.blob_store.open_blob_reader(artifact_id, int(entry["version"]))
        if reader is None:
            raise FileNotFoundError(f"output folder file {rel} is missing from the store")
        with reader as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out, _CHUNK)
