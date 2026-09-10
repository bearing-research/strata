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

# Extension by content type, so an output file in the bundle can be opened by
# double-clicking it. Mirrors the archive bundle's map for the same reason.
_OUTPUT_EXTENSIONS = {
    "image/png": ".png",
    "text/markdown": ".md",
    "json/object": ".json",
    "arrow/ipc": ".arrow",
    "pickle/object": ".pickle",
}


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
            name = _write_display_output(archive, cell.id, position, output)
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
            "outputs": outputs,
        }

    written = []
    store = session.get_artifact_manager().artifact_store
    for artifact_id, version in sorted(carry):
        reader_cm = store.open_blob_reader(artifact_id, version)
        if reader_cm is None:
            continue
        with reader_cm as reader:
            archive.writestr(f"artifacts/{artifact_id}@v={version}", reader.read())
        written.append(f"{artifact_id}@v={version}")

    manifest = {
        "notebook_id": session.notebook_state.id,
        "include": include,
        "cells": per_cell,
        "artifacts": index,
        # What is actually in this file, as opposed to what is described in it.
        # An importer marks cells whose artifacts came only by reference as
        # stale, and it needs to be told which those are rather than inferring
        # it from what it failed to find.
        "carried": written,
    }
    archive.writestr("artifacts.json", json.dumps(manifest, indent=2))
    return manifest


def _write_display_output(archive: zipfile.ZipFile, cell_id: str, position: int, output) -> str:
    """Write one display output as a file and return its name in the bundle."""
    extension = _OUTPUT_EXTENSIONS.get(output.content_type, ".json")
    name = f"outputs/{cell_id}/{position}{extension}"

    if output.inline_data_url and "," in output.inline_data_url:
        from base64 import b64decode

        payload = output.inline_data_url.split(",", 1)[1]
        try:
            archive.writestr(name, b64decode(payload))
            return name
        except ValueError:
            # A data URL we cannot decode is still worth carrying as what it
            # is; dropping the output entirely would be the worse answer.
            pass

    if output.markdown_text is not None:
        archive.writestr(name, output.markdown_text)
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
