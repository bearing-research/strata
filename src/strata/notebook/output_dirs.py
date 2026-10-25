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


def file_ref(manifest_id: str, cell_id: str, entry: dict[str, Any]) -> tuple[str, int]:
    """Where a manifest entry's file is stored, in the store holding the manifest.

    Beside the manifest's own id unless the entry names one: a copy into another store
    writes ``id`` when a file landed elsewhere (another tenant held its id).
    """
    if entry.get("id"):
        return str(entry["id"]), int(entry["version"])
    # Read off the manifest's id, not the reader's notebook: a folder copied in from
    # another notebook keeps its files under that notebook's ids.
    marker = f"_cell_{cell_id}_var_"
    base = manifest_id[: manifest_id.index(marker) + len(marker)]
    return f"{base}__file__{entry['sha256']}", int(entry["version"])


def file_refs(manifest_id: str, manifest: bytes) -> list[tuple[str, int, str]]:
    """``(id, version, sha256)`` of each file a manifest names.

    Raises:
        ValueError: *manifest* is not a folder manifest, or *manifest_id* not a folder's
            id. An import can store any bytes under any id with this content type.
    """
    try:
        described = json.loads(manifest)
        return [
            (*file_ref(manifest_id, described["cell_id"], entry), entry["sha256"])
            for entry in described["files"].values()
        ]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"{manifest_id} is not a readable folder manifest") from exc


def held_file(store: Any, tenant: str | None, file_id: str, version: int) -> Any:
    """The row of a file a folder of *tenant* names, or ``None`` if absent or another tenant's.

    A manifest's bytes are its importer's to choose: an id it names grants nothing.
    """
    record = store.get_artifact(file_id, version)
    if record is None or (record.tenant or "") != (tenant or ""):
        return None
    return record


def relocate(
    manifest: bytes,
    manifest_id: str,
    landed: dict[tuple[str, int], tuple[str, int]],
    *,
    every_id: bool = False,
) -> bytes:
    """The manifest naming where each file *landed* in another store.

    *landed* maps where a file was, read off *manifest_id*, to where it landed. An entry
    names an ``id`` only when that differs from the one read off *manifest_id*, or always
    with *every_id*, for a manifest landing under an id the other store mints; the bytes
    are unchanged when no entry changes.
    """
    described = json.loads(manifest)
    cell_id = described["cell_id"]
    changed = False
    for entry in described["files"].values():
        source = file_ref(manifest_id, cell_id, entry)
        target_id, target_version = landed.get(source, source)
        moved = {key: value for key, value in entry.items() if key != "id"}
        moved["version"] = target_version
        if every_id or file_ref(manifest_id, cell_id, moved)[0] != target_id:
            moved["id"] = target_id
        if moved != entry:
            entry.clear()
            entry.update(moved)
            changed = True
    if not changed:
        return manifest
    return json.dumps(described, sort_keys=True).encode("utf-8")


def checked_path(rel: str) -> str:
    """*rel* if it stays inside its folder; a manifest read from another store is not ours."""
    parts = rel.split("/")
    if rel.startswith("/") or "\\" in rel or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"output folder path {rel!r} leaves its folder")
    return rel


def _file_provenance(artifact_id: str, sha256: str) -> str:
    """Unique to the file's id: the store dedups equal provenance across ids, which would
    leave one cell's file reading another cell's bytes and swept with them."""
    from computation_core.provenance import derive_subkey

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
        if version.params.get("content_type") != CONTENT_TYPE:
            continue
        # Housekeeping: a read that counted as a use would keep old manifests forever.
        blob = store.read_blob(version.id, version.version, note_use=False)
        if blob is None and version.state == "failed":
            continue  # left by a crash, never to be read: it keeps no file
        if blob is None:
            return None  # being written, or lost: what it names is unknown
        try:
            named.update((fid, fv) for fid, fv, _ in file_refs(version.id, blob))
        except ValueError:
            logger.warning(
                "Not sweeping folder files: %s@v=%d is not a readable folder manifest",
                version.id,
                version.version,
            )
            return None
    return [
        (version.id, version.version)
        for version in versions
        if is_file_id(version.id) and (version.id, version.version) not in named
    ]


def failed_since_ready(artifact_mgr: Any) -> list[tuple[str, int]]:
    """Each failed manifest version a prune should keep.

    One is kept while no later good version of its folder is stored, or while the good
    one it follows (the newest good version below it) is. So a crash's files can still be
    downloaded when the run is recovered, and go no sooner than what came before them. A
    crash's row with no manifest is kept too; it holds no bytes.
    """
    store = artifact_mgr.artifact_store
    states: dict[str, dict[int, str]] = {}
    for version in store.list_versions_by_id_prefix(f"nb_{artifact_mgr.notebook_id}_cell_"):
        if version.params.get("content_type") == CONTENT_TYPE:
            states.setdefault(version.id, {})[version.version] = version.state

    def follows_a_kept_good_one(by_version: dict[int, str], number: int) -> bool:
        number -= 1
        while by_version.get(number) == "failed":
            number -= 1
        # A gap is a version already pruned: an older kept one is not what this followed.
        return by_version.get(number) in ("ready", "superseded")

    spared = []
    for artifact_id, by_version in states.items():
        newest_good = max(
            (n for n, state in by_version.items() if state in ("ready", "superseded")),
            default=0,
        )
        spared += [
            (artifact_id, number)
            for number, state in by_version.items()
            if state == "failed"
            and (number > newest_good or follows_a_kept_good_one(by_version, number))
        ]
    return spared


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
    from computation_core.provenance import TransformSpec

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
    from computation_core.provenance import TransformSpec

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
                logger.warning("output folder %s: not keeping %s, a link", folder.name, rel)
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


def materialize(
    store: Any, manifest_id: str, manifest: bytes, dest: Path, *, tenant: str | None
) -> None:
    """Rebuild the folder manifest *manifest_id* describes at *dest*, from the *store* holding it.

    Any store: a notebook's own, or one a folder was promoted or copied to. *tenant* is the
    manifest's; a file another tenant holds reads as missing.
    """
    described = json.loads(manifest)
    cell_id = described["cell_id"]
    dest.mkdir(parents=True, exist_ok=True)
    for rel, entry in described["files"].items():
        target = dest / checked_path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        artifact_id, version = file_ref(manifest_id, cell_id, entry)
        # Through the store, not its blob store: an imported file's bytes are under its row's key.
        held = held_file(store, tenant, artifact_id, version)
        reader = None if held is None else store.open_blob_reader(artifact_id, version)
        if reader is None:
            raise FileNotFoundError(f"output folder file {rel} is missing from the store")
        with reader as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out, _CHUNK)
