"""A notebook's state at a moment, as one machine-readable bundle (``fmt=snapshot``).

The ZIP export's members plus outputs as files, per-cell provenance and timings,
an ``artifacts.json`` of every ready cell's artifacts with digests, and the bytes
of whichever of those the caller ``include``s (a figure for review, or every byte
for a move between servers, with what each ``@fetch`` read).
"""

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession

IncludeMode = Literal["all", "selected", "none"]

SNAPSHOT_FORMAT_VERSION = 2

# So a bundled output file opens on double-click. Mirrors the archive bundle.
_OUTPUT_EXTENSIONS = {
    "image/png": ".png",
    "text/markdown": ".md",
    "json/object": ".json",
    "arrow/ipc": ".arrow",
    "pickle/object": ".pickle",
}


def write_committed_files(session: NotebookSession, archive: zipfile.ZipFile) -> None:
    """Write the committed files and ``provenance.json`` that every bundle carries.

    Shared by the route and ``strata export`` so they agree on what a bundle is.
    """
    from strata.notebook.env import compute_lockfile_hash
    from strata.notebook.layout import committed_paths
    from strata.notebook.provenance import compute_source_hash

    # Local import: routes imports this module.
    from strata.notebook.routes import _format_dag

    nb_dir = session.path
    for relative in committed_paths(nb_dir):
        archive.write(nb_dir / relative, relative.as_posix())

    manager = session.get_artifact_manager()
    provenance: dict[str, Any] = {
        "notebook_id": session.notebook_state.id,
        "lockfile_hash": compute_lockfile_hash(nb_dir),
        "dag": _format_dag(session),
        "cells": {
            cell.id: {
                "source_hash": compute_source_hash(cell.source),
                "defines": cell.defines,
                "references": cell.references,
                "status": cell.status.value if hasattr(cell.status, "value") else str(cell.status),
                "artifact_uri": cell.artifact_uri,
                # Lets a rerun's report be compared with this export output by output.
                "outputs": manager.cell_output_digests(cell.id),
            }
            for cell in session.notebook_state.cells
        },
    }
    archive.writestr("provenance.json", json.dumps(provenance, indent=2, default=str))


def build_artifact_index(session: NotebookSession) -> dict[str, Any]:
    """What every ready cell produced, named so another store can find it.

    Digests let two snapshots be compared output by output without reading blobs.
    """
    manager = session.get_artifact_manager()
    store = manager.artifact_store
    cells: dict[str, Any] = {}

    for cell in session.notebook_state.cells:
        entries = []
        for variable, artifact in sorted(manager.list_cell_artifacts(cell.id)):
            content_type = ""
            if artifact.transform_spec:
                try:
                    params = json.loads(artifact.transform_spec).get("params", {})
                except ValueError:
                    params = {}
                content_type = params.get("content_type") or ""
            entries.append(
                {
                    "variable": variable,
                    "artifact_id": artifact.id,
                    "version": artifact.version,
                    "provenance_hash": artifact.provenance_hash,
                    "content_type": content_type,
                    "content_sha256": store.content_digest(artifact.id, artifact.version),
                    "byte_size": artifact.byte_size,
                }
            )
        if entries:
            cells[cell.id] = entries
    return cells


def list_fetches(session: NotebookSession) -> list[dict[str, Any]]:
    """Every ``@fetch`` in the notebook, so a preflight can flag the unpinned.

    ``sha256`` is the pin, or else the digest last read (what an author would pin to).
    """
    from strata.notebook.annotations import parse_annotations
    from strata.notebook.fetch import FetchCache

    cache = FetchCache(session.path)
    fetches = []
    for cell in session.notebook_state.cells:
        for spec in parse_annotations(cell.source).fetches:
            recorded = cache.recorded(spec.url)
            fetches.append(
                {
                    "cell_id": cell.id,
                    "name": spec.name,
                    "url": spec.url,
                    "pinned": spec.sha256 is not None,
                    "sha256": spec.sha256 or (recorded.sha256 if recorded else None),
                    "refetch": spec.refetch,
                }
            )
    return fetches


def _artifacts_to_carry(
    index: dict[str, Any], include: IncludeMode, selected: list[str] | None
) -> set[tuple[str, int]]:
    """Which artifacts' bytes go in; ``selected`` names cells, not artifacts."""
    if include == "none":
        return set()
    wanted = set(index) if include == "all" else {c for c in (selected or []) if c in index}
    return {(entry["artifact_id"], entry["version"]) for cell in wanted for entry in index[cell]}


def unknown_selection(session: NotebookSession, selected: list[str] | None) -> list[str]:
    """Cell ids in ``selected`` that this notebook does not have.

    Lets the caller reject a typo instead of returning an empty ``carried``.
    """
    known = {cell.id for cell in session.notebook_state.cells}
    return [cell_id for cell_id in (selected or []) if cell_id not in known]


def write_snapshot(
    session: NotebookSession,
    archive: zipfile.ZipFile,
    *,
    include: IncludeMode = "selected",
    selected_cells: list[str] | None = None,
) -> dict[str, Any]:
    """Write the snapshot's members into *archive*; return its manifest.

    The caller adds the committed files; this writes everything the ZIP export lacks.
    """
    from strata.notebook.runtime_state import load_runtime_state
    from strata.notebook.writer import load_cell_console_output

    nb_dir = session.path
    store = session.get_artifact_manager().artifact_store
    runtime = load_runtime_state(nb_dir)
    index = build_artifact_index(session)
    carry = _artifacts_to_carry(index, include, selected_cells)

    per_cell: dict[str, Any] = {}
    for cell in session.notebook_state.cells:
        cell_runtime = runtime.cells.get(cell.id)
        stdout, stderr = load_cell_console_output(nb_dir, cell.id)
        if stdout or stderr:
            archive.writestr(
                f"outputs/{cell.id}/console.json",
                json.dumps({"stdout": stdout, "stderr": stderr}, indent=2),
            )

        outputs = []
        for position, output in enumerate(cell.display_outputs or []):
            name = _write_display_output(archive, store, cell.id, position, output)
            outputs.append(
                {
                    "file": name,
                    "content_type": output.content_type,
                    "artifact_uri": output.artifact_uri,
                }
            )

        per_cell[cell.id] = {
            "status": cell.status.value if hasattr(cell.status, "value") else str(cell.status),
            "provenance_hash": cell_runtime.last_provenance_hash if cell_runtime else None,
            "source_hash": cell_runtime.last_source_hash if cell_runtime else None,
            "env_hash": cell_runtime.last_env_hash if cell_runtime else None,
            "execution_samples": list(cell_runtime.execution_samples) if cell_runtime else [],
            # Without these an imported red cell opens idle, console but no error.
            "error": cell_runtime.last_error if cell_runtime else None,
            "error_source_hash": cell_runtime.last_error_source_hash if cell_runtime else None,
            "outputs": outputs,
            # The persisted form for an importer to write back; `outputs` above
            # is for a reader. Open resolves cached outputs index by index.
            "display_outputs": list(cell_runtime.display_outputs) if cell_runtime else [],
            # Without them an import falls back to declared defaults and
            # recomputes a different scenario than the bundle's.
            "widget_values": dict(cell_runtime.widget_values) if cell_runtime else {},
        }

    from strata.artifact_transfer import record_metadata

    written = []
    records: dict[str, Any] = {}
    for artifact_id, version in sorted(carry):
        reader_cm = store.open_blob_reader(artifact_id, version)
        record = store.get_artifact(artifact_id, version)
        if reader_cm is None or record is None:
            continue
        # Without the record an importer lacks the transform spec (so the
        # content type) and the lineage edges.
        ref = f"{artifact_id}@v={version}"
        records[ref] = {
            **record_metadata(record),
            "content_sha256": store.content_digest(artifact_id, version),
        }
        # Streamed, not read whole: ``include=all`` moves projects between
        # servers, where artifacts are large.
        member = f"artifacts/{artifact_id}@v={version}"
        with reader_cm as reader, archive.open(member, "w") as out:
            while chunk := reader.read(_BLOB_CHUNK_BYTES):
                out.write(chunk)
        written.append(f"{artifact_id}@v={version}")

    if include == "all":
        _write_fetched_bytes(session, archive)

    manifest = {
        # Bump when an older importer would read this bundle wrong.
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "notebook_id": session.notebook_state.id,
        "include": include,
        "cells": per_cell,
        "artifacts": index,
        # What this file actually carries, so an importer can mark cells whose
        # artifacts came only by reference as stale without guessing.
        "carried": written,
        "records": records,
        "fetches": list_fetches(session),
    }
    archive.writestr("artifacts.json", json.dumps(manifest, indent=2))
    return manifest


# Matches every other streamed read in the store.
_BLOB_CHUNK_BYTES = 1024 * 1024


def _write_fetched_bytes(session: NotebookSession, archive: zipfile.ZipFile) -> None:
    """The bytes each ``@fetch`` URL last served, and the index naming them, under ``fetch/``.

    Without them an imported pinned fetch needs its URL, which may have moved or gone.
    """
    from strata.notebook.fetch import FetchCache

    cache = FetchCache(session.path)
    index: dict[str, Any] = {}
    members: set[str] = set()
    for url, entry in cache._index().items():
        fetched = cache.recorded(url)
        if fetched is None:
            continue
        member = f"fetch/{fetched.sha256}/{fetched.path.name}"
        if member not in members:
            archive.write(fetched.path, member)
            members.add(member)
        index[url] = entry
    if index:
        archive.writestr("fetch/index.json", json.dumps(index, indent=2, sort_keys=True))


def _artifact_ref(uri: str | None) -> tuple[str, int] | None:
    """``strata://artifact/<id>@v=<n>`` into its parts, or ``None``."""
    if not uri:
        return None
    ref = uri.removeprefix("strata://artifact/")
    artifact_id, _, version = ref.partition("@v=")
    if not artifact_id or not version.isdigit():
        return None
    return artifact_id, int(version)


def _display_bytes(store, output) -> bytes | str | None:
    """The output's own bytes, from the live value or from the store.

    ``None`` for outputs that are not files in their own right (table previews,
    scalars); those are described instead.
    """
    if output.inline_data_url and "," in output.inline_data_url:
        from base64 import b64decode

        try:
            return b64decode(output.inline_data_url.split(",", 1)[1])
        except ValueError:
            # Undecodable data URL: fall through to the artifact, then to the
            # JSON description.
            pass

    if output.markdown_text is not None:
        return output.markdown_text

    if output.content_type not in ("image/png", "text/markdown"):
        return None

    ref = _artifact_ref(output.artifact_uri)
    if ref is None:
        return None
    reader_cm = store.open_blob_reader(*ref)
    if reader_cm is None:
        return None
    with reader_cm as reader:
        blob = reader.read()
    if output.content_type == "text/markdown":
        return blob.decode("utf-8", errors="replace")
    return blob


def _write_display_output(
    archive: zipfile.ZipFile, store, cell_id: str, position: int, output
) -> str:
    """Write one display output as a file and return its name in the bundle.

    Images and markdown are written as themselves. Their inline payloads are not
    persisted, so a session read from disk (the CLI, or a restarted server) must
    fetch the bytes from the store via ``artifact_uri``.
    """
    extension = _OUTPUT_EXTENSIONS.get(output.content_type, ".json")
    name = f"outputs/{cell_id}/{position}{extension}"

    payload = _display_bytes(store, output)
    if payload is not None:
        archive.writestr(name, payload)
        return name

    name = f"outputs/{cell_id}/{position}.json"
    archive.writestr(
        name,
        json.dumps(
            {
                "content_type": output.content_type,
                "preview": output.preview,
                "rows": output.rows,
                "columns": output.columns,
                "artifact_uri": output.artifact_uri,
                "error": output.error,
            },
            indent=2,
            default=str,
        ),
    )
    return name
