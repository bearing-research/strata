"""``# @each``: a cell that runs once per file or row, and caches each item separately.

What the items are is known only at run time, so each run lists them and keys each one
on its own content (cellgraph's ``Item``). When every instance has succeeded, one
manifest per output variable names the version each item produced. The manifest is the
cell's output: a reader sees one artifact, and an item removed from the source leaves by
not being in the next manifest. Like an ``# @output_dir`` folder, a manifest requires the
versions it names, and an instance goes once no manifest left names it. It names the
items' displays too, so a hit that lists no item can still show them.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cellgraph as cg

from strata.notebook.mounts import _sha256, glob_matches
from strata.notebook.output_dirs import is_assembling
from strata.notebook.serializer import to_serialization_safe

CONTENT_TYPE = "each/manifest"
_INSTANCE_PREFIX = "each-"


class EachError(ValueError):
    """The items could not be listed; the message says why."""


@dataclass(frozen=True)
class EachItem:
    """One item: its stable key, its content digest, and what the instance binds."""

    key: str
    digest: str
    # The value as JSON, for an item of a variable.
    value: bytes | None = None
    # The file, for an item of a mount.
    path: Path | None = None

    @property
    def instance(self) -> cg.Item:
        """The item as a key reads it."""
        return cg.Item(self.key, self.digest)


def instance_name(key: str) -> str:
    """The instance's name in artifact ids: a key can hold ``/``, and any length."""
    return f"{_INSTANCE_PREFIX}{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def is_instance_id(artifact_id: str) -> bool:
    return f"@variant={_INSTANCE_PREFIX}" in artifact_id


def _encoded(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    except ValueError as exc:
        raise EachError(f"an item cannot be stored as JSON: {exc}") from exc


def _item(key: str, value: Any) -> EachItem:
    encoded = _encoded(value)
    return EachItem(key=key, digest=hashlib.sha256(encoded).hexdigest(), value=encoded)


def items_of_json(value: Any, name: str) -> list[EachItem]:
    """A dict's entries by key; a list's values by content, so an insert moves nothing."""
    if isinstance(value, dict):
        return [_item(str(key), entry) for key, entry in value.items()]
    if not isinstance(value, list):
        raise EachError(f"@each over `{name}`: expected a dict, a list or a table")
    items: dict[str, EachItem] = {}
    for entry in value:
        item = _item("", entry)
        key = entry if isinstance(entry, str) else item.digest[:16]
        items[key] = EachItem(key=key, digest=item.digest, value=item.value)
    return list(items.values())


def items_of_table(table: Any, key_column: str | None, name: str) -> list[EachItem]:
    """One item per row, as a JSON-safe dict, named by *key_column*."""
    if key_column is None:
        raise EachError(f"@each over the table `{name}` needs key=<column>")
    if key_column not in table.column_names:
        raise EachError(f"@each over `{name}`: the table has no column {key_column!r}")
    items: dict[str, EachItem] = {}
    for row in table.to_pylist():
        if row[key_column] is None:
            raise EachError(f"@each over `{name}`: a row has a null {key_column!r}")
        key = str(row[key_column])
        if key in items:
            raise EachError(f"@each over `{name}`: two rows have the {key_column!r} {key!r}")
        items[key] = _item(key, to_serialization_safe(row, keep_none=True))
    return list(items.values())


def _listed_files(root: Path) -> list[tuple[str, Path]]:
    """``(relative path, path)`` of what the mount's fingerprint counts: regular files.

    A link to a file is a file, as in a Hugging Face cache or a DVC checkout.
    """
    if not root.is_dir():
        raise EachError(f"@each over a mount needs a folder, and {root} is not one")
    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                state = path.stat()
            except OSError:
                continue  # a dangling link
            if stat.S_ISREG(state.st_mode):
                files.append((path.relative_to(root).as_posix(), path))
    return sorted(files)


def items_of_mount(
    root: Path,
    glob: str | None,
    *,
    versions: Mapping[str, str] | None = None,
) -> list[EachItem]:
    """One item per file, named by its relative path, keyed by its content.

    *versions* are what the mount was keyed on: a local mount's file digests, a remote
    listing's versions. A file with none (a pinned mount's, or a listing without them)
    is read.
    """
    items = []
    for relative, path in _listed_files(root):
        if glob is not None and not glob_matches(relative, glob):
            continue
        if versions is not None and relative in versions:
            digest = versions[relative]
        else:
            with path.open("rb") as handle:
                digest = _sha256(handle)
        items.append(EachItem(key=relative, digest=digest, path=path))
    return items


def instance_id(manifest_id: str, key: str) -> str:
    """Where an item's value is stored: beside the manifest, whose id an import may rename."""
    return f"{manifest_id.partition('@')[0]}@variant={instance_name(key)}"


def encode_manifest(
    manifest_id: str,
    entries: list[tuple[str, str, int]],
    displays: list[tuple[str, str, int]] | None = None,
) -> bytes:
    """A manifest of ``(item key, instance's artifact id, version)``, by key, and of the
    ``(item key, display's artifact id, version)`` the items showed, in the order shown.

    An entry names an id only when the value landed somewhere other than beside it.
    """
    manifest: dict[str, Any] = {
        "items": [
            [key, version]
            if artifact_id == instance_id(manifest_id, key)
            else [key, version, artifact_id]
            for key, artifact_id, version in sorted(entries)
        ]
    }
    if displays:
        manifest["displays"] = [list(display) for display in displays]
    return _encoded(manifest)


def manifest_refs(manifest_id: str, manifest: bytes) -> list[tuple[str, str, int]]:
    """A manifest's ``(item key, artifact id, version)``.

    Raises:
        EachError: it is not one, as an import can hold any bytes under this type.
    """
    try:
        refs = []
        for key, version, *elsewhere in json.loads(manifest)["items"]:
            artifact_id = str(elsewhere[0]) if elsewhere else instance_id(manifest_id, str(key))
            refs.append((str(key), artifact_id, int(version)))
        return refs
    except (ValueError, KeyError, TypeError) as exc:
        raise EachError(f"not an @each manifest: {exc}") from exc


def manifest_displays(manifest: bytes) -> list[tuple[str, str, int]]:
    """A manifest's ``(item key, display's artifact id, version)``, in the order shown.

    Raises:
        EachError: it is not one, as an import can hold any bytes under this type.
    """
    try:
        return [
            (str(key), str(artifact_id), int(version))
            for key, artifact_id, version in json.loads(manifest).get("displays", [])
        ]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise EachError(f"not an @each manifest: {exc}") from exc


def relocate(
    manifest: bytes, manifest_id: str, landed: dict[tuple[str, int], tuple[str, int]]
) -> bytes:
    """The manifest naming where each instance *landed* in another store.

    Without the displays: a copy carries the values, not what showed them.
    """
    return encode_manifest(
        manifest_id,
        [
            (key, *landed.get((artifact_id, version), (artifact_id, version)))
            for key, artifact_id, version in manifest_refs(manifest_id, manifest)
        ],
    )


def rows_of(artifact_mgr: Any, var_name: str, versions: list[Any]) -> bytes:
    """Every instance's table as one Arrow IPC blob.

    Raises:
        EachError: an instance's value is not a table.
    """
    import pyarrow as pa

    from strata.artifact.table_export import arrow_table_from_ipc

    tables = []
    for version in versions:
        content_type = version.params.get("content_type")
        if content_type != "arrow/ipc":
            raise EachError(
                f"@write_table `{var_name}`: {version.id} is {content_type}, not a table"
            )
        blob = artifact_mgr.load_artifact_data(version.id, version.version)
        try:
            tables.append(arrow_table_from_ipc(blob, f"{version.id}@v={version.version}"))
        except ValueError as exc:
            raise EachError(f"@write_table `{var_name}`: {exc}") from exc
    rows = pa.concat_tables(tables, promote_options="permissive")
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, rows.schema) as writer:
        writer.write_table(rows)
    return sink.getvalue().to_pybytes()


def collectable_instances(artifact_mgr: Any) -> list[tuple[str, int]]:
    """Every stored instance version, for a sweep: a kept manifest requires the ones it names.

    None while a run is assembling, whose manifest does not require what it will name yet.
    """
    if is_assembling():
        return []
    versions = artifact_mgr.artifact_store.list_versions_by_id_prefix(
        f"nb_{artifact_mgr.notebook_id}_cell_"
    )
    return [(v.id, v.version) for v in versions if is_instance_id(v.id)]
