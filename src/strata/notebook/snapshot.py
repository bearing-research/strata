"""A notebook's state at a moment, as one machine-readable bundle.

Two exports existed and neither was this. ``fmt=zip`` carries the committed
files and a ``provenance.json`` with no outputs; the HTML and markdown exports
carry rendered outputs and no machine-readable provenance. Assembling a
reviewable snapshot meant three requests and two formats, and the result still
could not seed a sandbox — nothing in it carries bytes.

``fmt=snapshot`` is the ZIP's members plus what a reader and a machine each
need: the outputs as files, the per-cell provenance and timings that live in
``.strata/runtime.json``, an ``artifacts.json`` naming every ready cell's
artifacts with their digests, and the bytes of however many of those the caller
asked for.

``include`` is the whole design. A snapshot for review wants the chain
described and a figure or two attached; a project moving between servers wants
every byte. Those are the same document with a different payload, so they are
one format with a parameter rather than two formats that drift.
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
    """The committed files and ``provenance.json`` — what every bundle carries.

    Shared so the route and ``strata export`` cannot disagree about what a
    bundle is. They did: the route wrote ``provenance.json`` and globbed
    ``cells/*.py``, the CLI wrote no provenance and globbed everything, and a
    test that called one of them a superset of the other passed only because
    its fixture happened to be Python-only.
    """
    from strata.notebook.env import compute_lockfile_hash
    from strata.notebook.provenance import compute_source_hash

    # Local import: routes imports this module.
    from strata.notebook.routes import _format_dag

    nb_dir = session.path
    # renv.lock is in the env hash; without it no imported R cell matches its
    # artifacts.
    for name in ("notebook.toml", "pyproject.toml", "uv.lock", "renv.lock"):
        member = nb_dir / name
        if member.exists():
            archive.write(member, name)

    cells_dir = nb_dir / "cells"
    if cells_dir.is_dir():
        for cell_file in sorted(cells_dir.glob("*")):
            if cell_file.is_file():
                archive.write(cell_file, f"cells/{cell_file.name}")

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
            }
            for cell in session.notebook_state.cells
        },
    }
    archive.writestr("provenance.json", json.dumps(provenance, indent=2, default=str))


def build_artifact_index(session: NotebookSession) -> dict[str, Any]:
    """What every ready cell produced, named so another store can find it.

    Digests are the point: two snapshots of the same notebook can be compared
    output by output without either side reading a blob, which is what makes a
    snapshot reviewable rather than merely openable.
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

    An unpinned fetch is the one input a snapshot cannot vouch for: its bytes
    are whatever the URL serves when the cell next runs. ``sha256`` is the pin,
    or else the digest last read, which is what an author would pin to.
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
    """Which artifacts' bytes go in, given the caller's answer.

    ``selected`` names cells rather than artifacts: a reviewer thinks "attach
    the figure cell", not "attach nb_…_var___display__0@v=3".
    """
    if include == "none":
        return set()
    wanted = set(index) if include == "all" else {c for c in (selected or []) if c in index}
    return {(entry["artifact_id"], entry["version"]) for cell in wanted for entry in index[cell]}


def unknown_selection(session: NotebookSession, selected: list[str] | None) -> list[str]:
    """Cell ids in ``selected`` that this notebook does not have.

    A mistyped id would otherwise produce a 200 and an empty ``carried``,
    indistinguishable from a selection that legitimately had nothing to carry —
    and the caller would find out when the snapshot turned out to be missing
    the figure they meant to attach.
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

    The committed files are the caller's to add — this is everything the ZIP
    export does not already carry, so the two share the format rather than one
    reimplementing the other.
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

    ``None`` for anything that is not a file in its own right — a table
    preview or a scalar gets described instead, since there is no format in
    which double-clicking it would mean anything.
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

    An image or a markdown output is written as itself — a ``.png`` a reader
    can open, not a JSON stub describing one. That takes the store, because
    ``inline_data_url`` and ``markdown_text`` are transient: the writer strips
    both before persisting to ``runtime.json`` and the parser rebuilds the
    output without them, so any session read from disk — always the CLI, and
    the server after a restart — holds only the ``artifact_uri`` and has to
    fetch the bytes back.
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
