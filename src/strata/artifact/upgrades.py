"""What Strata completes in a ledger it opens, once per store.

Each step is keyed on the ledger's ``store_upgrades`` marker: a crash before the mark
redoes it, a fresh store marks it on first open. Called wherever a store is opened.
"""

from __future__ import annotations

import json
import logging

from vestigo.provenance import parse_transform_spec
from vestigo.store import ArtifactStore

from strata.artifact.graph import reconstruct_definitions
from strata.artifact.retention import record_folder_edges

logger = logging.getLogger(__name__)

MISBOUND_BUILDS_UPGRADE = "embedded_builds_bound_inputs_in_order"

# The only transform the embedded runner can build: it runs what ``get_transform`` knows,
# which strips a ``local://`` scheme, and scan@v1 never builds.
_EMBEDDED_EXECUTORS = frozenset({"duckdb_sql@v1", "local://duckdb_sql@v1"})


def complete_upgrade(store: ArtifactStore) -> None:
    fail_misbound_builds(store)
    record_folder_edges(store)
    reconstruct_definitions(store)


def fail_misbound_builds(store: ArtifactStore) -> None:
    """Fail every embedded SQL build over 11 or more inputs, and every build that read one.

    Before 0.10 the embedded runner bound inputs in name order, ``input10`` before
    ``input2``, so those bytes answer another query under a correct hash, as does
    anything built over them. Failed, a lookup misses and the next materialize
    rebuilds. A version's row does not say which runner built it (imports carry no
    build row), so the same spec built by an HTTP executor fails too and rebuilds once.
    The ledger's migration 11 once did this; the repair is Strata's.
    """
    if store.upgrade_applied(MISBOUND_BUILDS_UPGRADE):
        return
    conn = store.dialect.connect()
    try:
        rows = conn.execute(
            "SELECT id, version, transform_spec FROM artifact_versions "
            "WHERE state IN ('ready', 'superseded') AND transform_spec LIKE ?",
            ("%duckdb_sql@v1%",),
        ).fetchall()
        pending = []
        for row in rows:
            spec = parse_transform_spec(row["transform_spec"])
            if spec is not None and spec.executor in _EMBEDDED_EXECUTORS and len(spec.inputs) >= 11:
                pending.append(f"{row['id']}@v={row['version']}")
        failed: set[str] = set()
        while pending:
            ref = pending.pop()
            if ref in failed:
                continue
            failed.add(ref)
            artifact_id, version = ref.rsplit("@v=", 1)
            conn.execute(
                "UPDATE artifact_versions SET state = 'failed' WHERE id = ? AND version = ?",
                (artifact_id, int(version)),
            )
            # An artifact or name input records the version it read as ``id@v=N``.
            readers = conn.execute(
                "SELECT id, version, input_versions FROM artifact_versions "
                "WHERE state IN ('ready', 'superseded') AND input_versions LIKE ?",
                (f"%{ref}%",),
            ).fetchall()
            for row in readers:
                if ref in json.loads(row["input_versions"]).values():
                    pending.append(f"{row['id']}@v={row['version']}")
        conn.commit()
    finally:
        conn.close()
    store.mark_upgrade_applied(MISBOUND_BUILDS_UPGRADE)
    if failed:
        logger.warning(
            "Failed %d artifact versions built from inputs bound out of order; "
            "materializing them again rebuilds them",
            len(failed),
        )
