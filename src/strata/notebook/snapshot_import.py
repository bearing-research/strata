"""Turn a snapshot bundle back into a notebook (the reverse of ``snapshot``).

Carried cells are cache hits before anything runs; the rest are IDLE. A notebook
keeps its id unless the caller can already see one with it; a taken id is
replaced, and every artifact id, lineage edge and display URI embedding it is
rewritten, so two notebooks never name each other's rows in a shared store.
Provenance hashes hold no notebook id and are not rewritten. Writes go bytes,
then rows, then state, so an interrupted import can be retried.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tomllib
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

from strata.artifact_store import ArtifactStore, ArtifactVersion
from strata.artifact_transfer import RECORD_FIELDS, remap_input_versions
from strata.notebook.snapshot import SNAPSHOT_FORMAT_VERSION

_ARTIFACT_URI_PREFIX = "strata://artifact/"

# Bounds what an archive may expand to on disk; zipfile stops each member at its declared
# size, so summing the declared sizes bounds the real output.
_MAX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024 * 1024
# Bounds memory: the manifest, notebook.toml, console and committed files are parsed
# whole (cells again on every open). Artifact and fetched bytes stream, under their own caps.
_MAX_MEMBER_BYTES = 16 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_COMMITTED_ROOT_FILES = ("pyproject.toml", "uv.lock", "renv.lock", ".gitignore")


_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
# Looser than a cell id: committed files are the user's own names ("notes copy.md").
_SAFE_MEMBER_SEGMENT = re.compile(r"[^\x00-\x1f\x7f/\\]+")


def _member_target(dest: Path, name: str) -> Path:
    """Where a bundle member may be written under *dest*.

    Member names come from an untrusted archive (``cells/../../.ssh/...``). Each
    segment is checked rather than the joined path, so a name that resolves outside
    only through symlinks still cannot be written.
    """
    # Split by hand: PurePosixPath collapses ``//`` and ``.``, so ``cells//etc/x`` would
    # pass as ``cells/etc/x`` while notebook.toml names ``/etc/x``.
    parts = name.split("/")
    for part in parts:
        if not _SAFE_MEMBER_SEGMENT.fullmatch(part) or part in (".", ".."):
            raise NotASnapshotError(f"the bundle names a member it cannot write: {name!r}")
    return dest.joinpath(*parts)


class NotASnapshotError(ValueError):
    """The file is not a bundle this importer can read."""


@dataclass(frozen=True)
class ImportedSnapshot:
    notebook_dir: Path
    notebook_id: str
    # Set when the bundle's id was taken and had to be replaced.
    replaced_id: str | None
    imported_artifacts: int
    # Cells whose artifacts the bundle described but did not carry; they open IDLE.
    by_reference_cells: list[str] = field(default_factory=list)


def import_snapshot(
    bundle: Path | str,
    dest: Path | str,
    *,
    taken_ids: set[str] | frozenset[str] = frozenset(),
    name: str | None = None,
) -> ImportedSnapshot:
    """Unpack *bundle* into the new notebook directory *dest*.

    ``taken_ids`` are notebook ids already in use where the caller looks for
    notebooks; the bundle's id is kept unless it is among them. ``name`` renames
    the notebook; ``None`` keeps the bundle's.

    Raises:
        NotASnapshotError: The file is not a snapshot, predates the records a
            snapshot needs, or names cells it does not contain.
        FileExistsError: *dest* exists and is not empty.
    """
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")

    with zipfile.ZipFile(bundle) as archive:
        manifest, notebook_toml = _validate(archive)
        # Before anything is written, so a bundle that cannot answer it leaves nothing behind.
        by_reference = _by_reference_cells(manifest)

        old_id = manifest["notebook_id"]
        new_id = str(uuid.uuid4()) if old_id in taken_ids else old_id
        rename = _renamer(old_id, new_id)

        # Built beside the destination and renamed in only once whole, so a failure
        # leaves no half-written notebook for discovery to list or to block a retry.
        # The dot-prefixed staging name is skipped by discovery.
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.parent / f".{dest.name}.importing-{uuid.uuid4().hex[:8]}"
        staging.mkdir()
        try:
            # 1. Bytes and rows, into the notebook's own store.
            store = ArtifactStore(staging / ".strata" / "artifacts")
            landed = _import_records(
                archive, store, manifest.get("records", {}), rename, scratch=staging
            )

            # The bytes each @fetch read, so a pinned fetch runs without its URL.
            _write_fetched_bytes(archive, staging)

            # 2. The committed files.
            _write_committed_files(archive, staging, notebook_toml, old_id, new_id, name)

            # 3. Runtime state last: it points at what steps 1 and 2 wrote.
            _write_runtime_state(archive, staging, manifest, landed, rename)

            if dest.exists():
                dest.rmdir()  # empty, checked above
            staging.rename(dest)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    return ImportedSnapshot(
        notebook_dir=dest,
        notebook_id=new_id,
        replaced_id=old_id if new_id != old_id else None,
        imported_artifacts=len(landed),
        by_reference_cells=by_reference,
    )


# --- Validation ---


def _validate(archive: zipfile.ZipFile) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refuse early, before anything is written."""
    expanded = sum(info.file_size for info in archive.infolist())
    if expanded > _MAX_UNCOMPRESSED_BYTES:
        raise NotASnapshotError(
            f"the bundle expands to {expanded} bytes, over the "
            f"{_MAX_UNCOMPRESSED_BYTES // (1024**3)} GiB import cap"
        )
    for info in archive.infolist():
        if _read_whole(info.filename) and info.file_size > _MAX_MEMBER_BYTES:
            raise NotASnapshotError(
                f"the bundle's {info.filename!r} is {info.file_size} bytes, over the "
                f"{_MAX_MEMBER_BYTES // (1024**2)} MiB cap for a file the import reads whole"
            )
    names = set(archive.namelist())
    if "artifacts.json" not in names:
        raise NotASnapshotError(
            "not a snapshot: no artifacts.json (export one with fmt=snapshot, "
            "or `strata export <dir> --to snapshot`)"
        )
    if "notebook.toml" not in names:
        raise NotASnapshotError("not a snapshot: no notebook.toml")

    manifest = _parse_member(archive, "artifacts.json", json.loads)
    if not isinstance(manifest, dict):
        raise NotASnapshotError("not a snapshot: artifacts.json is not an object")
    try:
        version = int(manifest.get("format_version", 1))
    except (TypeError, ValueError) as exc:
        raise NotASnapshotError(
            "not a snapshot: artifacts.json has no usable format_version"
        ) from exc
    if version < SNAPSHOT_FORMAT_VERSION:
        # A version-1 bundle has bytes but no content types or lineage, so its
        # artifacts would load wrong.
        raise NotASnapshotError(
            f"this snapshot is format {version}, which does not carry the "
            "artifact records an import needs; export it again"
        )
    if version > SNAPSHOT_FORMAT_VERSION:
        raise NotASnapshotError(
            f"this snapshot is format {version}, newer than this Strata reads "
            f"({SNAPSHOT_FORMAT_VERSION}); upgrade to import it"
        )

    _check_manifest(manifest, names)

    notebook_toml = _parse_member(
        archive, "notebook.toml", lambda data: tomllib.loads(data.decode("utf-8"))
    )
    cells = notebook_toml.get("cells", [])
    if not isinstance(cells, list) or not all(isinstance(cell, dict) for cell in cells):
        raise NotASnapshotError("not a snapshot: notebook.toml's cells are not a list of tables")
    for cell in cells:
        file = cell.get("file", "")
        # The parser joins this onto cells/, so it must stay a relative path inside it.
        if any(part in ("", ".", "..") for part in str(file).split("/")):
            raise NotASnapshotError(
                f"the bundle names a cell file outside cells/: {cell.get('id')!r} -> {file!r}"
            )
    missing = [
        cell["file"]
        for cell in notebook_toml.get("cells", [])
        if f"cells/{cell.get('file', '')}" not in names
    ]
    if missing:
        raise NotASnapshotError(
            "not a complete snapshot: notebook.toml names cell files the bundle "
            f"does not contain ({', '.join(missing)})"
        )
    return manifest, notebook_toml


def _parse_member(archive: zipfile.ZipFile, name: str, parse: Callable[[bytes], Any]) -> Any:
    try:
        return parse(archive.read(name))
    except ValueError as exc:  # JSON, TOML and UTF-8 decode errors are all ValueErrors
        raise NotASnapshotError(f"the bundle's {name} cannot be read: {exc}") from exc


def _check_manifest(manifest: dict[str, Any], names: set[str]) -> None:
    """Refuse a manifest whose shape the import would otherwise trip over halfway."""
    if not isinstance(manifest.get("notebook_id"), str) or not manifest["notebook_id"]:
        raise NotASnapshotError("not a snapshot: artifacts.json names no notebook_id")
    carried = manifest.get("carried")
    if not isinstance(carried, list) or not all(isinstance(ref, str) for ref in carried):
        raise NotASnapshotError("not a snapshot: artifacts.json has no list of carried artifacts")
    for key in ("records", "cells"):
        value = manifest.get(key, {})
        if not isinstance(value, dict) or not all(isinstance(v, dict) for v in value.values()):
            raise NotASnapshotError(f"not a snapshot: artifacts.json's {key} is malformed")
    index = manifest.get("artifacts", {})
    if not isinstance(index, dict) or not all(
        isinstance(entries, list)
        and all(isinstance(e, dict) and "artifact_id" in e and "version" in e for e in entries)
        for entries in index.values()
    ):
        raise NotASnapshotError("not a snapshot: artifacts.json's artifacts is malformed")
    for ref, record in manifest.get("records", {}).items():
        if f"artifacts/{ref}" not in names:
            raise NotASnapshotError(f"not a complete snapshot: no bytes for the record {ref!r}")
        problem = _record_problem(record)
        if problem is not None:
            raise NotASnapshotError(f"the bundle's record {ref!r} {problem}")


def _record_problem(data: dict[str, Any]) -> str | None:
    """Why a bundle record cannot be stored as it is, or ``None``.

    The checks ``POST /v1/artifacts/import`` makes: rows are stored as read,
    and a ``building`` row or a string ``byte_size`` breaks later sweeps.
    """
    missing = [key for key in RECORD_FIELDS if key not in data]
    if missing:
        return f"is missing {', '.join(missing)}"
    if data["state"] not in ("ready", "superseded"):
        return "has a state other than 'ready' or 'superseded'"
    if type(data["version"]) is not int or data["version"] < 1:
        return "has a version that is not an integer of 1 or more"
    if not isinstance(data["provenance_hash"], str) or not data["provenance_hash"]:
        return "has no provenance_hash"
    if type(data["created_at"]) not in (int, float):
        return "has a created_at that is not epoch seconds"
    for key in ("row_count", "byte_size"):
        value = data[key]
        if value is not None and (type(value) is not int or value < 0):
            return f"has a {key} that is not a non-negative integer"
    for key in ("schema_json", "principal"):
        if not isinstance(data[key], str | None):
            return f"has a {key} that is not a string"
    for key in ("transform_spec", "input_versions"):
        if data[key] is not None and not _is_json_object(data[key]):
            return f"has a {key} that is not a JSON object encoded as a string"
    return None


def _is_json_object(value: object) -> bool:
    """Whether *value* is a string holding a JSON object, as ``transform_spec`` and edges are."""
    if not isinstance(value, str):
        return False
    try:
        return isinstance(json.loads(value), dict)
    except json.JSONDecodeError:
        return False


def _by_reference_cells(manifest: dict[str, Any]) -> list[str]:
    """Cells whose artifacts the bundle described but did not carry."""
    carried = set(manifest["carried"])
    return sorted(
        cell_id
        for cell_id, entries in manifest.get("artifacts", {}).items()
        if not all(f"{e['artifact_id']}@v={e['version']}" in carried for e in entries)
    )


def _read_whole(name: str) -> bool:
    """Whether the import (or a later open) holds this member in memory at once."""
    return (
        name in ("artifacts.json", "notebook.toml", "fetch/index.json", *_COMMITTED_ROOT_FILES)
        or name.startswith("cells/")
        or (name.startswith("outputs/") and name.endswith("/console.json"))
    )


# --- Identity ---


def _renamer(old_id: str, new_id: str):
    """Map an artifact id to its id in the imported notebook.

    Only ``nb_<notebook id>_`` and ``nb_remote_<notebook id>_`` ids belong to this
    notebook; anything else is left alone.
    """
    prefixes = (f"nb_{old_id}_", f"nb_remote_{old_id}_")

    def rename(artifact_id: str) -> str:
        if old_id == new_id:
            return artifact_id
        for prefix in prefixes:
            if artifact_id.startswith(prefix):
                return prefix.replace(old_id, new_id, 1) + artifact_id[len(prefix) :]
        return artifact_id

    return rename


def _rename_ref(ref: str, rename) -> str:
    artifact_id, sep, version = ref.partition("@v=")
    return f"{rename(artifact_id)}{sep}{version}"


# --- Artifacts ---


def _import_records(
    archive: zipfile.ZipFile,
    store: ArtifactStore,
    records: dict[str, dict[str, Any]],
    rename,
    *,
    scratch: Path,
) -> dict[str, str]:
    """Import every carried record, ancestors first; return old ref to landed ref.

    A store that already holds a computation resolves its import onto that row, so
    descendants must name where their ancestors actually landed.
    """
    remap: dict[str, str] = {}
    for old_ref in _ancestors_first(records):
        data = records[old_ref]
        record = _record_from(data)

        # Rename all edges, then let where each carried ancestor landed win.
        edges = json.loads(record.input_versions) if record.input_versions else {}
        edge_refs = {
            uri[len(_ARTIFACT_URI_PREFIX) :]
            for uri in edges
            if uri.startswith(_ARTIFACT_URI_PREFIX)
        }
        edge_map = {ref: remap.get(ref, _rename_ref(ref, rename)) for ref in edge_refs}
        record = remap_input_versions(record, edge_map)
        record = replace(record, id=rename(record.id))

        # Streamed through a file so a large artifact never sits whole in memory.
        blob_path = scratch / f".import-blob-{uuid.uuid4().hex}"
        try:
            with archive.open(f"artifacts/{old_ref}") as src, open(blob_path, "wb") as dst:
                shutil.copyfileobj(src, dst, _COPY_CHUNK_BYTES)
            imported = store.import_artifact(record, blob_path)
        finally:
            blob_path.unlink(missing_ok=True)
        remap[old_ref] = imported.ref
    return remap


def _record_from(data: dict[str, Any]) -> ArtifactVersion:
    """A bundle record back into an artifact version.

    Reads every :data:`RECORD_FIELDS` key. The tenant is left unset: a local store
    takes the importer, never what the bundle claims.
    """
    missing = [key for key in RECORD_FIELDS if key not in data]
    if missing:
        raise NotASnapshotError(f"artifact record is missing {', '.join(missing)}")
    artifact_id = str(data["id"])
    # The id becomes a blob key (a file name). Checked here too so a bad bundle
    # fails as "not a snapshot", like bad member names and cell ids.
    if "/" in artifact_id or "\\" in artifact_id or ".." in PurePosixPath(artifact_id).parts:
        raise NotASnapshotError(f"the bundle names an artifact it cannot write: {artifact_id!r}")
    return ArtifactVersion(
        id=artifact_id,
        version=int(data["version"]),
        state=str(data["state"]),
        provenance_hash=str(data["provenance_hash"]),
        schema_json=data["schema_json"],
        row_count=data["row_count"],
        byte_size=data["byte_size"],
        created_at=data["created_at"],
        transform_spec=data["transform_spec"],
        input_versions=data["input_versions"],
        principal=data["principal"],
        tenant=None,
        content_sha256=data.get("content_sha256"),
    )


def _ancestors_first(records: dict[str, dict[str, Any]]) -> list[str]:
    """Order refs so each comes after every carried ref its edges name."""
    parents: dict[str, set[str]] = {}
    for ref, data in records.items():
        edges = json.loads(data["input_versions"]) if data.get("input_versions") else {}
        parents[ref] = {
            uri[len(_ARTIFACT_URI_PREFIX) :]
            for uri in edges
            if uri.startswith(_ARTIFACT_URI_PREFIX) and uri[len(_ARTIFACT_URI_PREFIX) :] in records
        }

    ordered: list[str] = []
    placed: set[str] = set()
    pending = sorted(records)
    while pending:
        ready = [ref for ref in pending if parents[ref] <= placed]
        if not ready:
            # Only a hand-edited bundle can have a cycle; import the rest in a stable order.
            ready = pending[:1]
        for ref in ready:
            ordered.append(ref)
            placed.add(ref)
        pending = [ref for ref in pending if ref not in placed]
    return ordered


# --- Fetched bytes ---


def _write_fetched_bytes(archive: zipfile.ZipFile, dest: Path) -> None:
    """Restore ``.strata/fetch/`` from the bundle's ``fetch/`` members.

    Both the files and the index are untrusted. Each file must sit where the fetch
    cache itself would put it and hash to its directory name, since a pinned fetch
    trusts that name without rereading the bytes. Index entries naming no carried
    file, or with fields of the wrong type, are dropped.
    """
    from strata.notebook.fetch import MAX_FETCH_BYTES, FetchCache, FetchError, _safe_filename

    cache = FetchCache(dest)
    carried: set[tuple[str, str]] = set()
    for info in archive.infolist():
        name = info.filename
        if not name.startswith("fetch/") or name.endswith("/") or name == "fetch/index.json":
            continue
        _member_target(dest, name)
        parts = PurePosixPath(name).parts
        if len(parts) != 3 or parts[2] != _safe_filename(parts[2]):
            raise NotASnapshotError(f"the bundle names a member it cannot write: {name!r}")
        sha, filename = parts[1], parts[2]
        try:
            target = cache._contained(sha, filename)
        except FetchError as exc:
            raise NotASnapshotError(f"the bundle names a member it cannot write: {name!r}") from exc
        if info.file_size > MAX_FETCH_BYTES:
            raise NotASnapshotError(f"the bundle's fetched file {name!r} is over the fetch cap")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with archive.open(info) as src, open(target, "wb") as dst:
            while chunk := src.read(_COPY_CHUNK_BYTES):
                digest.update(chunk)
                dst.write(chunk)
        if digest.hexdigest() != sha:
            raise NotASnapshotError(f"the bundle's fetched file {name!r} does not match its digest")
        carried.add((sha, filename))

    if "fetch/index.json" not in archive.namelist():
        return
    index = _parse_member(archive, "fetch/index.json", json.loads)
    if not isinstance(index, dict):
        raise NotASnapshotError("the bundle's fetch index is not an object")
    kept: dict[str, dict[str, Any]] = {}
    for url, entry in index.items():
        if (
            not isinstance(entry, dict)
            or (entry.get("sha256"), entry.get("filename")) not in carried
        ):
            continue
        # The cache reads these without guarding their types.
        kept[url] = {
            key: value
            for key, value in entry.items()
            if (key in ("sha256", "filename", "etag", "last_modified") and isinstance(value, str))
            or (
                key in ("checked_at", "fetched_at")
                and isinstance(value, int | float)
                and not isinstance(value, bool)
            )
        }
    if kept:
        cache.root.mkdir(parents=True, exist_ok=True)
        cache._index_path().write_text(json.dumps(kept, indent=2, sort_keys=True), encoding="utf-8")


# --- Files and state ---


def _write_committed_files(
    archive: zipfile.ZipFile,
    dest: Path,
    notebook_toml: dict[str, Any],
    old_id: str,
    new_id: str,
    notebook_name: str | None,
) -> None:
    from strata.notebook.layout import write_gitignore
    from strata.notebook.writer import _write_notebook_toml_atomic

    for name in archive.namelist():
        if name.startswith("cells/") and not name.endswith("/"):
            target = _member_target(dest, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_member(archive, name, target)
        elif name in _COMMITTED_ROOT_FILES:
            _copy_member(archive, name, dest / name)

    renamed = notebook_name is not None and notebook_toml.get("name") != notebook_name
    if new_id != old_id or renamed:
        notebook_toml["notebook_id"] = new_id
        if notebook_name is not None:
            notebook_toml["name"] = notebook_name
        _write_notebook_toml_atomic(dest / "notebook.toml", notebook_toml)
    else:
        # Verbatim when unchanged, preserving a hand-edited notebook.toml's comments and layout.
        (dest / "notebook.toml").write_bytes(archive.read("notebook.toml"))

    # Keep the imported .strata/ out of git, as `strata new` does, if the bundle had none.
    write_gitignore(dest)


def _copy_member(archive: zipfile.ZipFile, name: str, target: Path) -> None:
    with archive.open(name) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, _COPY_CHUNK_BYTES)


def _write_runtime_state(
    archive: zipfile.ZipFile,
    dest: Path,
    manifest: dict[str, Any],
    landed: dict[str, str],
    rename,
) -> None:
    """Write provenance, timings, display outputs and console as the bundle had them.

    This is what lets an imported notebook show its outputs before anything runs.
    """
    from strata.notebook.runtime_state import load_runtime_state, save_runtime_state
    from strata.notebook.writer import update_cell_console_output

    def rewrite_uri(uri: str | None) -> str | None:
        if not uri or not uri.startswith(_ARTIFACT_URI_PREFIX):
            return uri
        old_ref = uri[len(_ARTIFACT_URI_PREFIX) :]
        return _ARTIFACT_URI_PREFIX + landed.get(old_ref, _rename_ref(old_ref, rename))

    state = load_runtime_state(dest)
    names = set(archive.namelist())
    for cell_id, cell in manifest.get("cells", {}).items():
        entry = state.get_or_create_cell(cell_id)
        entry.last_provenance_hash = cell.get("provenance_hash")
        entry.last_source_hash = cell.get("source_hash")
        entry.last_env_hash = cell.get("env_hash")
        identity = cell.get("reopen_identity")
        entry.last_reopen_identity = identity if isinstance(identity, str) else None
        entry.execution_samples = list(cell.get("execution_samples") or [])
        # Older snapshots omit the error pair.
        entry.last_error = cell.get("error") or None
        entry.last_error_source_hash = cell.get("error_source_hash") if entry.last_error else None
        entry.display_outputs = [
            {**output, "artifact_uri": rewrite_uri(output.get("artifact_uri"))}
            for output in cell.get("display_outputs") or []
        ]
        # Older snapshots omit widget selections; controls import at their defaults.
        entry.widget_values = dict(cell.get("widget_values") or {})

        if not _SAFE_SEGMENT.match(cell_id):
            # The console path is built from the cell id.
            raise NotASnapshotError(f"the bundle names a cell it cannot write: {cell_id!r}")
        console_member = f"outputs/{cell_id}/console.json"
        if console_member in names:
            console = _parse_member(archive, console_member, json.loads)
            if not isinstance(console, dict):
                raise NotASnapshotError(f"the bundle's {console_member} is not an object")
            update_cell_console_output(
                dest, cell_id, console.get("stdout", ""), console.get("stderr", "")
            )
    save_runtime_state(dest, state)
