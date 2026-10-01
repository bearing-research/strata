"""Writing artifacts into Iceberg tables.

Each export writes the artifact as the table's new current snapshot, whose
summary names the source artifact version. The target table is recorded on the
artifact as the ``strata.iceberg_table`` tag, so moving an alias can move the
table's tag of the same name (``move_alias_tag``).
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from strata.artifact_store import ArtifactStore, ArtifactVersion
from strata.iceberg import IcebergWriter, PyIcebergCatalog, TableWrite
from strata.logging import get_logger

logger = get_logger(__name__)

EXPORT_TAG = "strata.iceberg_table"

# Notebook values Arrow carries that are not tables.
_SHAPE_KEY = b"strata.arrow.shape"
_NOT_TABLES = {b"tensor", b"scalar"}


def artifact_arrow_table(store: ArtifactStore, artifact: ArtifactVersion) -> pa.Table:
    """The artifact's bytes as an Arrow table, or ``ValueError`` if they are not one."""
    ref = f"{artifact.id}@v={artifact.version}"
    content_type = None
    if artifact.transform_spec:
        try:
            content_type = (json.loads(artifact.transform_spec).get("params") or {}).get(
                "content_type"
            )
        except (ValueError, AttributeError):
            content_type = None
    if content_type not in (None, "", "arrow/ipc"):
        raise ValueError(f"{ref} is {content_type}, not a table")
    blob = store.read_blob(artifact.id, artifact.version)
    if blob is None:
        raise ValueError(f"{ref} has no stored bytes")
    table = pa.ipc.open_stream(blob).read_all()
    if (table.schema.metadata or {}).get(_SHAPE_KEY) in _NOT_TABLES:
        raise ValueError(f"{ref} is an array or a scalar, not a table")
    return table


def export_artifact(
    store: ArtifactStore,
    artifact: ArtifactVersion,
    table_uri: str,
    *,
    config: Any,
    promoted_by: str | None = None,
    alias: str | None = None,
    tenant: str | None = None,
) -> TableWrite:
    """Write *artifact* into *table_uri* as its current snapshot.

    Raises ``ValueError`` when the artifact is not a table or its schema cannot
    evolve the table's.
    """
    if artifact.state not in ("ready", "superseded"):
        raise ValueError(
            f"{artifact.id}@v={artifact.version} is not readable (state={artifact.state})"
        )
    data = artifact_arrow_table(store, artifact)
    written = IcebergWriter(PyIcebergCatalog(config)).write(
        table_uri,
        data,
        artifact_id=artifact.id,
        version=artifact.version,
        provenance_hash=artifact.provenance_hash,
        promoted_by=promoted_by,
        alias=alias,
    )
    store.set_tag(artifact.id, artifact.version, EXPORT_TAG, table_uri, tenant=tenant)
    return written


def move_alias_tag(
    store: ArtifactStore,
    artifact_id: str,
    version: int,
    alias: str,
    *,
    config: Any,
    tenant: str | None = None,
) -> int | None:
    """Move the table tag named after an alias that just moved onto ``artifact_id@v=version``.

    Returns the tagged snapshot id, or None when the version was never written to a
    table. An unreachable catalog is logged and leaves the tag in place: the alias
    has already moved and the registry is the authority.
    """
    table_uri = store.get_tags(artifact_id, version, tenant=tenant).get(EXPORT_TAG)
    if not table_uri:
        return None
    try:
        return IcebergWriter(PyIcebergCatalog(config)).tag(
            table_uri, alias, artifact_id=artifact_id, version=version
        )
    except Exception:
        logger.warning(
            "Could not move a table tag to follow an alias",
            table=table_uri,
            alias=alias,
            artifact=f"{artifact_id}@v={version}",
            exc_info=True,
        )
        return None
