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
import re
import shutil
import stat
import tarfile
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

from strata.notebook.harness_user import UnsafeRunFile, open_tree_file

logger = logging.getLogger(__name__)

OUTPUT_DIRS = "output_dirs"  # under a run directory, one folder per name
CONTENT_TYPE = "dir/manifest"
EXTENSION = ".dirmanifest.json"
# A folder input sent to a worker: a tar of its files, which the harness unpacks.
TAR_CONTENT_TYPE = "dir/tar"
TAR_EXTENSION = ".dir.tar"
_CHUNK = 1 << 20


def mount_entries(source: str, run_dir: Path) -> dict[str, dict[str, str]]:
    """The run's ``# @output_dir`` folders as manifest mount entries, each folder made.

    The harness binds a read-write mount as a writable path, so every harness (cold,
    warm pool, R, a worker's) gives the cell its folder unchanged.
    """
    from strata.notebook.annotations import parse_annotations

    root = os.path.normpath(run_dir / OUTPUT_DIRS)
    entries: dict[str, dict[str, str]] = {}
    for name in parse_annotations(source).output_dirs:
        # The parser takes identifiers only; on a worker the source is the request's, so
        # the folder is still checked to stay under the run directory.
        folder = os.path.normpath(os.path.join(root, name))
        if not folder.startswith(root + os.sep):
            raise ValueError(f"# @output_dir name {name!r} leaves the run directory")
        Path(folder).mkdir(parents=True, exist_ok=True)
        entries[name] = {"uri": "", "mode": "rw", "local_path": folder}
    return entries


STREAMED = "output-dirs-streamed.json"  # in a run directory: what a worker uploaded


def read_streamed(run_dir: Path) -> dict[str, list[str]] | None:
    """``{folder: [path, ...]}`` a worker's result says the server already has.

    ``None`` when the result says nothing (nothing was sent during the run); a folder
    missing from the map holds none of the files sent.
    """
    path = run_dir / STREAMED
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def regular_files(root: Path) -> list[str]:
    """Each regular file under *root*, as a POSIX path relative to it; links are not followed."""
    found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError:
                continue  # removed since the listing: a cell renaming a checkpoint into place
            if stat.S_ISREG(mode):
                found.append(path.relative_to(root).as_posix())
    return sorted(found)


def pack(folder: Path, target: Path) -> None:
    """Write *folder*'s regular files to *target* as a tar, for a worker to unpack."""
    with tarfile.open(target, "w") as tar:
        for rel in regular_files(folder):
            tar.add(folder / rel, arcname=rel, recursive=False)


def file_artifact_id(artifact_mgr: Any, cell_id: str, sha256: str) -> str:
    return artifact_mgr.cell_artifact_id(cell_id, f"__file__{sha256}")


def _file_provenance(artifact_id: str, sha256: str) -> str:
    """Unique to the file's id: the store dedups equal provenance across ids, which would
    leave one cell's file reading another cell's bytes and swept with them."""
    from strata.notebook.provenance import derive_subkey

    return derive_subkey(sha256, artifact_id)


_FILE_ID = re.compile(r"_var___file__[0-9a-f]{64}$")


def is_file_id(artifact_id: str) -> bool:
    return _FILE_ID.search(artifact_id) is not None


_assembling = 0  # runs in this process whose stored files may not be in a manifest yet
_assembling_lock = threading.Lock()


@contextmanager
def assembling() -> Generator[None]:
    """Hold off the file sweep while a run stores files ahead of its manifest.

    A worker's uploads arrive during the run, hours before its manifest is staged.
    """
    global _assembling
    with _assembling_lock:
        _assembling += 1
    try:
        yield
    finally:
        with _assembling_lock:
            _assembling -= 1


def unreferenced_files(artifact_mgr: Any) -> list[tuple[str, int]] | None:
    """Each stored folder-file version no manifest version in the store names.

    ``None`` while a run is assembling a folder, or when a manifest cannot be read,
    so nothing it might name is taken.
    """
    if _assembling:
        return None
    store = artifact_mgr.artifact_store
    versions = store.list_versions_by_id_prefix(f"nb_{artifact_mgr.notebook_id}_cell_")
    named: set[tuple[str, int]] = set()
    for version in versions:
        params = json.loads(version.transform_spec or "{}").get("params") or {}
        if params.get("content_type") != CONTENT_TYPE:
            continue
        # Housekeeping: a read that counted as a use would keep old manifests forever.
        blob = store.read_blob(version.id, version.version, note_use=False)
        if blob is None and version.state == "failed":
            continue  # left by a crash, never to be read: it keeps no file
        if blob is None:
            return None  # being written, or lost: what it names is unknown
        manifest = json.loads(blob)
        for entry in manifest["files"].values():
            sha256 = entry["sha256"]
            named.add(
                (file_artifact_id(artifact_mgr, manifest["cell_id"], sha256), entry["version"])
            )
    return [
        (version.id, version.version)
        for version in versions
        if is_file_id(version.id) and (version.id, version.version) not in named
    ]


def _reused(store: Any, artifact_id: str) -> int | None:
    """The ready version of a file already stored, recorded as used, or ``None``.

    The use keeps a sweep from taking it; checked again after, in case one just did.
    """
    existing = store.get_latest_version(artifact_id)
    if existing is None or existing.state != "ready":
        return None
    store.record_use(artifact_id, existing.version)
    again = store.get_latest_version(artifact_id)
    if again is None or again.version != existing.version:
        return None
    return existing.version


def _store_opened(
    artifact_mgr: Any, cell_id: str, opener: Callable[[], BinaryIO], shown: str
) -> dict[str, Any]:
    """Store one file by its digest, unless a ready copy is already there.

    *opener* opens it afresh for each of the two passes, hashing then copying.
    """
    digest = hashlib.sha256()
    size = 0
    with opener() as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    sha256 = digest.hexdigest()
    store = artifact_mgr.artifact_store
    artifact_id = file_artifact_id(artifact_mgr, cell_id, sha256)
    reused = _reused(store, artifact_id)
    if reused is not None:
        return {"sha256": sha256, "size": size, "version": reused}

    # Here, not at the top: a worker imports this module and has no artifact store.
    from strata.artifact.store import TransformSpec

    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=_file_provenance(artifact_id, sha256),
        transform_spec=TransformSpec(
            executor="notebook/cell@v1",
            params={"cell_id": cell_id, "content_type": "application/octet-stream"},
            inputs=[],
        ),
    )
    # Hashed again while copying: the file may have changed since the first pass.
    copied = hashlib.sha256()
    try:
        with opener() as f, store.blob_store.open_blob_writer(artifact_id, version) as out:
            while chunk := f.read(_CHUNK):
                copied.update(chunk)
                out.write(chunk)
        if copied.hexdigest() != sha256:
            raise RuntimeError(f"{shown} changed while it was being stored")
    except BaseException:
        store.fail_artifact(artifact_id, version)
        raise
    store.finalize_artifact(artifact_id, version, "", None, size, content_sha256=sha256)
    return {"sha256": sha256, "size": size, "version": version}


def store_received(
    artifact_mgr: Any, cell_id: str, path: Path, sha256: str, size: int
) -> dict[str, Any]:
    """Store a file a worker uploaded while the cell ran, hashed as it arrived.

    *path* is a server-owned file nothing else writes, so it is published as it is,
    read once.
    """
    store = artifact_mgr.artifact_store
    artifact_id = file_artifact_id(artifact_mgr, cell_id, sha256)
    reused = _reused(store, artifact_id)
    if reused is not None:
        return {"sha256": sha256, "size": size, "version": reused}
    from strata.artifact.store import TransformSpec

    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=_file_provenance(artifact_id, sha256),
        transform_spec=TransformSpec(
            executor="notebook/cell@v1",
            params={"cell_id": cell_id, "content_type": "application/octet-stream"},
            inputs=[],
        ),
    )
    try:
        store.blob_store.publish_blob_from_path(artifact_id, version, path)
    except BaseException:
        store.fail_artifact(artifact_id, version)
        raise
    store.finalize_artifact(artifact_id, version, "", None, size, content_sha256=sha256)
    return {"sha256": sha256, "size": size, "version": version}


def collect(
    artifact_mgr: Any,
    cell_id: str,
    run_dir: Path | None,
    name: str,
    received: dict[str, dict[str, Any]] | None = None,
    seen: dict[str, tuple[tuple[int, ...], dict[str, Any]]] | None = None,
) -> bytes:
    """Store folder *name*'s files in *run_dir* and return its manifest: the writing cell
    and each file.

    *received* holds entries for files already stored as a worker uploaded them; a file
    in the folder too replaces its entry. A link, or a file with more than one hard link,
    is left out with a warning. Each file is opened from *run_dir* down, following no
    link: the harness user owns the run directory and could make ``output_dirs`` one.

    *seen* carries ``{path: (stat key, entry)}`` from one collect of a folder to the next,
    as a loop's iterations do: a file whose size, times and inode are unchanged is not
    read again, so a checkpoint per iteration is hashed once, not once per iteration.
    """
    files: dict[str, dict[str, Any]] = dict(received or {})
    folder = run_dir / OUTPUT_DIRS / name if run_dir is not None else None
    if run_dir is not None and folder is not None and folder.is_dir():
        for rel in regular_files(folder):
            in_run = f"{OUTPUT_DIRS}/{name}/{rel}"
            try:
                key = None
                if seen is not None:
                    with open_tree_file(run_dir, in_run) as f:
                        st = os.fstat(f.fileno())
                    # ctime moves on any write and a cell cannot set it.
                    key = (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
                    if rel in seen and seen[rel][0] == key:
                        files[rel] = seen[rel][1]
                        continue
                files[rel] = _store_opened(
                    artifact_mgr,
                    cell_id,
                    lambda in_run=in_run: open_tree_file(run_dir, in_run),
                    rel,
                )
                if seen is not None and key is not None:
                    seen[rel] = (key, files[rel])
            except UnsafeRunFile:
                logger.warning("output folder %s: not keeping %s, a link", name, rel)
    return json.dumps({"cell_id": cell_id, "files": files}, sort_keys=True).encode("utf-8")


_LISTED = 20  # files a folder's listing shows; the rest are counted


def listing(name: str, manifest: bytes) -> str:
    """The folder's files and sizes as markdown, for the cell to show."""
    files = json.loads(manifest)["files"]
    total = sum(int(entry["size"]) for entry in files.values())
    noun = "file" if len(files) == 1 else "files"
    lines = [f"**{name}**: {len(files)} {noun}, {_size(total)}"]
    if files:
        lines += ["", "| File | Size |", "| --- | ---: |"]
        for rel in sorted(files)[:_LISTED]:
            lines.append(f"| {_code(rel)} | {_size(int(files[rel]['size']))} |")
        if len(files) > _LISTED:
            lines += ["", f"... and {len(files) - _LISTED} more"]
    return "\n".join(lines)


def _code(path: str) -> str:
    """*path* as an inline code span inside a table cell."""
    path = path.replace("|", "\\|")
    return f"`` {path} ``" if "`" in path else f"`{path}`"


def _size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    scaled = n / 1024
    for unit in ("KiB", "MiB"):
        if scaled < 1024:
            return f"{scaled:.1f} {unit}"
        scaled /= 1024
    return f"{scaled:.1f} GiB"


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
