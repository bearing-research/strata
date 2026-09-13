"""Turn a snapshot bundle back into a notebook.

The other half of :mod:`strata.notebook.snapshot`. A bundle carries the
committed files, the per-cell runtime state, an index of every ready cell's
artifacts, and the records and bytes of whichever of those the exporter
included. Importing it yields a notebook directory whose carried cells are
cache hits before anything runs, and whose other cells are IDLE — nothing here
to serve them, and with a team store configured, running one is a pull rather
than a recompute, because the provenance the bundle recorded is the provenance
the notebook computes locally.

**Identity.** A notebook keeps its id unless another notebook the caller can
see already has it. Each notebook directory has its own artifact store, so an
import never collides with anything locally; what a shared id breaks is the
moment both notebooks publish or promote into one shared store, where their
artifact ids — ``nb_<notebook id>_cell_…`` — would name each other's rows. So
a taken id is replaced, and every artifact id, lineage edge and display output
URI that embeds it is rewritten to match. Provenance hashes are not rewritten:
they contain no notebook id, which is what keeps an imported artifact
deduplicating against the same computation wherever it came from.

**Order.** Bytes before rows before state, so an interruption leaves something
a retry can finish rather than a notebook that looks complete and is not.
"""

from __future__ import annotations

import json
import shutil
import tomllib
import uuid
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from strata.artifact_store import ArtifactStore, ArtifactVersion
from strata.artifact_transfer import RECORD_FIELDS, remap_input_versions
from strata.notebook.snapshot import SNAPSHOT_FORMAT_VERSION

_ARTIFACT_URI_PREFIX = "strata://artifact/"


class NotASnapshotError(ValueError):
    """The file is not a bundle this importer can read."""


@dataclass(frozen=True)
class ImportedSnapshot:
    notebook_dir: Path
    notebook_id: str
    # The id the bundle carried, when it was taken and had to be replaced.
    replaced_id: str | None
    imported_artifacts: int
    # Cells whose artifacts the bundle described but did not carry. They open
    # IDLE; listed so the caller can say which, rather than leave the reader to
    # discover it one cell at a time.
    by_reference_cells: list[str] = field(default_factory=list)


def import_snapshot(
    bundle: Path | str,
    dest: Path | str,
    *,
    taken_ids: set[str] | frozenset[str] = frozenset(),
    owner: str | None = None,
) -> ImportedSnapshot:
    """Unpack *bundle* into the new notebook directory *dest*.

    ``taken_ids`` are notebook ids already in use where the caller will look
    for notebooks. The bundle's id is kept unless it is among them.

    ``owner`` replaces the owner the bundle's notebook.toml names. The bundle
    carries whoever exported it, and on a server scoping notebooks per user,
    discovery lists by owner — so an import left under the original owner would
    vanish from the importer's own list the moment it landed.

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

        old_id = str(manifest["notebook_id"])
        new_id = str(uuid.uuid4()) if old_id in taken_ids else old_id
        rename = _renamer(old_id, new_id)

        # Built beside the destination and renamed into place only once it is
        # whole. A failure part-way used to leave a half-written notebook at
        # `dest` — listed by discovery as though it were real, and blocking the
        # retry, which refuses a non-empty destination. The staging name starts
        # with a dot, which discovery skips.
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.parent / f".{dest.name}.importing-{uuid.uuid4().hex[:8]}"
        staging.mkdir()
        try:
            # 1. Bytes and rows, into the notebook's own store.
            store = ArtifactStore(staging / ".strata" / "artifacts")
            landed = _import_records(archive, store, manifest.get("records", {}), rename)

            # 2. The committed files.
            _write_committed_files(archive, staging, notebook_toml, old_id, new_id, owner)

            # 3. Runtime state, last: it points at what steps 1 and 2 put there.
            _write_runtime_state(archive, staging, manifest, landed, rename)

            if dest.exists():
                dest.rmdir()  # empty, checked above
            staging.rename(dest)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    carried_cells = {
        cell_id
        for cell_id, entries in manifest.get("artifacts", {}).items()
        if all(f"{e['artifact_id']}@v={e['version']}" in manifest["carried"] for e in entries)
    }
    by_reference = sorted(set(manifest.get("artifacts", {})) - carried_cells)

    return ImportedSnapshot(
        notebook_dir=dest,
        notebook_id=new_id,
        replaced_id=old_id if new_id != old_id else None,
        imported_artifacts=len(landed),
        by_reference_cells=by_reference,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(archive: zipfile.ZipFile) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refuse early, before anything is written."""
    names = set(archive.namelist())
    if "artifacts.json" not in names:
        raise NotASnapshotError(
            "not a snapshot: no artifacts.json (export one with fmt=snapshot, "
            "or `strata export <dir> --to snapshot`)"
        )
    if "notebook.toml" not in names:
        raise NotASnapshotError("not a snapshot: no notebook.toml")

    manifest = json.loads(archive.read("artifacts.json"))
    version = int(manifest.get("format_version", 1))
    if version < SNAPSHOT_FORMAT_VERSION:
        # A version-1 bundle carries bytes but not the records to rebuild them
        # from — no content type to read a value back as, no lineage. Importing
        # it would produce artifacts that load wrong; refusing says why.
        raise NotASnapshotError(
            f"this snapshot is format {version}, which does not carry the "
            "artifact records an import needs; export it again"
        )
    if version > SNAPSHOT_FORMAT_VERSION:
        raise NotASnapshotError(
            f"this snapshot is format {version}, newer than this Strata reads "
            f"({SNAPSHOT_FORMAT_VERSION}); upgrade to import it"
        )

    notebook_toml = tomllib.loads(archive.read("notebook.toml").decode("utf-8"))
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


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _renamer(old_id: str, new_id: str):
    """Artifact id → the id it has in the imported notebook.

    Notebook-derived artifact ids start ``nb_<notebook id>_`` (cell outputs,
    iterations, variants, display and console artifacts) or
    ``nb_remote_<notebook id>_`` (remote builds). Anything else is not this
    notebook's to rename.
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


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _import_records(
    archive: zipfile.ZipFile,
    store: ArtifactStore,
    records: dict[str, dict[str, Any]],
    rename,
) -> dict[str, str]:
    """Import every carried record, ancestors first. Return old ref → landed ref.

    Ancestors first so a descendant is imported with edges naming where its
    ancestors actually landed, which is not always the renamed ref: a store
    that already holds a computation resolves an import of it onto that row.
    """
    remap: dict[str, str] = {}
    for old_ref in _ancestors_first(records):
        data = records[old_ref]
        record = _record_from(data)

        # Edges name upstream refs, carried or not. Rename them all, then let
        # where each carried ancestor actually landed win over the rename.
        edges = json.loads(record.input_versions) if record.input_versions else {}
        edge_refs = {
            uri[len(_ARTIFACT_URI_PREFIX) :]
            for uri in edges
            if uri.startswith(_ARTIFACT_URI_PREFIX)
        }
        edge_map = {ref: remap.get(ref, _rename_ref(ref, rename)) for ref in edge_refs}
        record = remap_input_versions(record, edge_map)
        record = replace(record, id=rename(record.id))

        blob = archive.read(f"artifacts/{old_ref}")
        imported = store.import_artifact(record, blob)
        remap[old_ref] = imported.ref
    return remap


def _record_from(data: dict[str, Any]) -> ArtifactVersion:
    """A bundle record back into an artifact version.

    Every :data:`RECORD_FIELDS` key is read, so a field added there and to the
    exporter reaches here too. The tenant is left unset: a local store takes
    whoever is importing, never what the bundle claims.
    """
    missing = [key for key in RECORD_FIELDS if key not in data]
    if missing:
        raise NotASnapshotError(f"artifact record is missing {', '.join(missing)}")
    return ArtifactVersion(
        id=str(data["id"]),
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
            # A cycle cannot come from a store, but a hand-edited bundle can
            # have one; import the rest in a stable order rather than hang.
            ready = pending[:1]
        for ref in ready:
            ordered.append(ref)
            placed.add(ref)
        pending = [ref for ref in pending if ref not in placed]
    return ordered


# ---------------------------------------------------------------------------
# Files and state
# ---------------------------------------------------------------------------


def _write_committed_files(
    archive: zipfile.ZipFile,
    dest: Path,
    notebook_toml: dict[str, Any],
    old_id: str,
    new_id: str,
    owner: str | None,
) -> None:
    from strata.notebook.layout import write_gitignore
    from strata.notebook.writer import _write_notebook_toml_atomic

    for name in archive.namelist():
        if name.startswith("cells/") and not name.endswith("/"):
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
        elif name in ("pyproject.toml", "uv.lock", "renv.lock"):
            (dest / name).write_bytes(archive.read(name))

    changed = False
    if new_id != old_id:
        notebook_toml["notebook_id"] = new_id
        changed = True
    if owner is not None and notebook_toml.get("owner") != owner:
        notebook_toml["owner"] = owner
        changed = True
    if changed:
        _write_notebook_toml_atomic(dest / "notebook.toml", notebook_toml)
    else:
        # Verbatim when nothing about it changes, so an import round-trips a
        # hand-edited notebook.toml's comments and layout untouched.
        (dest / "notebook.toml").write_bytes(archive.read("notebook.toml"))

    # A new notebook directory is ready for git (`strata new` does the same);
    # an imported one should not commit its own .strata/ by accident.
    write_gitignore(dest)


def _write_runtime_state(
    archive: zipfile.ZipFile,
    dest: Path,
    manifest: dict[str, Any],
    landed: dict[str, str],
    rename,
) -> None:
    """Provenance, timings, display outputs and console, as the bundle had them.

    This is what lets an imported notebook show its outputs before anything
    runs: the notebook resolves cached display outputs from these entries, index
    by index, against the artifacts step 1 imported.
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
        entry.execution_samples = list(cell.get("execution_samples") or [])
        entry.display_outputs = [
            {**output, "artifact_uri": rewrite_uri(output.get("artifact_uri"))}
            for output in cell.get("display_outputs") or []
        ]

        console_member = f"outputs/{cell_id}/console.json"
        if console_member in names:
            console = json.loads(archive.read(console_member))
            update_cell_console_output(
                dest, cell_id, console.get("stdout", ""), console.get("stderr", "")
            )
    save_runtime_state(dest, state)
