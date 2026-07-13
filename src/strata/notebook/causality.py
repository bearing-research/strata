"""Causality inspector — explains WHY a cell is stale.

The staleness computation (session.compute_staleness) tells users *that* a cell
is stale. The causality inspector tells them *why*, down to the specific change
that triggered it.

It works by comparing the current provenance components (source hash, input
hashes, env hash) against those stored with the cached artifact. The diff
between old and new components *is* the causality explanation.

The same data also powers "Why did this run?" — same provenance diff, just
rendered in past tense after execution instead of present tense before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from strata.notebook.annotations import parse_annotations
from strata.notebook.env import compute_execution_env_hash, narrow_env_for_provenance
from strata.notebook.provenance import compute_provenance_hash, compute_source_hash
from strata.notebook.workers import worker_runtime_identity

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession


class CausalityType(StrEnum):
    """Which provenance component changed."""

    SOURCE_CHANGED = "source_changed"
    INPUT_CHANGED = "input_changed"
    ENV_CHANGED = "env_changed"


class CausalityReason(StrEnum):
    """Primary staleness reason for a cell."""

    SELF = "self"
    UPSTREAM = "upstream"
    ENV = "env"


def skip_none(pairs: list[tuple[str, object]]) -> dict:
    """``dict_factory`` for ``asdict`` that drops fields whose value is None.

    Used by callers serializing CausalityChain / CausalityDetail to JSON,
    so optional fields don't appear in the wire payload as ``null``.
    """
    return {k: v for k, v in pairs if v is not None}


@dataclass
class CausalityDetail:
    """A single reason contributing to staleness.

    Attributes
    ----------
    type : CausalityType
        Which provenance component changed.
    cell_id : str or None
        For source/input changes, which cell changed.
    cell_name : str or None
        Human-readable name of the changed cell.
    from_version : str or None
        Old artifact version string (for ``input_changed``).
    to_version : str or None
        New artifact version string (for ``input_changed``).
    package : str or None
        Package name (for ``env_changed``).
    from_package_version : str or None
        Old package version (for ``env_changed``).
    to_package_version : str or None
        New package version (for ``env_changed``).
    """

    type: CausalityType
    cell_id: str | None = None
    cell_name: str | None = None
    from_version: str | None = None
    to_version: str | None = None
    package: str | None = None
    from_package_version: str | None = None
    to_package_version: str | None = None


@dataclass
class CausalityChain:
    """Full causality explanation for a stale cell.

    Attributes
    ----------
    reason : CausalityReason
        Primary staleness reason.
    details : list of CausalityDetail
        Specific changes that caused staleness.
    """

    reason: CausalityReason
    details: list[CausalityDetail] = field(default_factory=list)


def compute_causality_on_staleness(
    session: NotebookSession,
) -> dict[str, CausalityChain]:
    """Compute causality chains for all cells during staleness detection.

    Called alongside ``compute_staleness()`` to provide causality
    explanations for stale cells. It uses the same topological walk and
    provenance comparison, but extracts component-level diffs.

    Parameters
    ----------
    session : NotebookSession
        Session whose cells will be inspected.

    Returns
    -------
    dict of {str : CausalityChain}
        Mapping from ``cell_id`` to the causality chain for stale cells.
    """
    if session.dag is None:
        return {}

    causality_map: dict[str, CausalityChain] = {}

    for cell_id in session.dag.topological_order:
        cell = session.notebook_state.get_cell(cell_id)
        if cell is None:
            continue

        details: list[CausalityDetail] = []
        annotations = parse_annotations(cell.source)
        source_hash = compute_source_hash(cell.source)
        runtime_env = dict(cell.env)
        runtime_env.update(annotations.env)
        declared_env_keys = set(annotations.env) | set(cell.env_overrides or {})
        provenance_env = narrow_env_for_provenance(cell.source, runtime_env, declared_env_keys)
        effective_worker = annotations.worker or cell.worker or session.notebook_state.worker
        env_hash = compute_execution_env_hash(
            session.path,
            provenance_env,
            runtime_identity=worker_runtime_identity(
                session.notebook_state,
                effective_worker,
            ),
        )
        mount_fingerprints, has_rw_mount = session._collect_mount_fingerprints(cell)

        if has_rw_mount:
            # RW mounts are intentionally non-cacheable side effects.
            # They should remain stale/idle without pretending there is a
            # meaningful cached provenance explanation.
            continue

        # Use the session's single source of truth so sweep refs are grouped
        # the same way the executor stored them (else a sweep downstream would
        # always look stale here).
        input_hashes = session._collect_input_hashes(cell_id)

        provenance_hash = compute_provenance_hash(
            input_hashes + mount_fingerprints, source_hash, env_hash
        )

        if session._resolve_cached_outputs(cell_id, provenance_hash) is not None:
            # Cell is ready — no causality needed
            continue

        # Cell is stale — figure out why
        # Check upstream cells
        for upstream_id in cell.upstream_ids:
            upstream = session.notebook_state.get_cell(upstream_id)
            if upstream is None:
                continue

            # If upstream is stale or has no artifact, our inputs changed
            if upstream.status in ("stale", "idle", "error"):
                upstream_name = upstream.defines[0] if upstream.defines else upstream_id
                details.append(
                    CausalityDetail(
                        type=CausalityType.INPUT_CHANGED,
                        cell_id=upstream_id,
                        cell_name=upstream_name,
                    )
                )
            # If upstream ran and produced a new artifact since our last run
            elif upstream_id in causality_map:
                # Upstream itself changed — so our inputs changed transitively
                upstream_name = upstream.defines[0] if upstream.defines else upstream_id
                details.append(
                    CausalityDetail(
                        type=CausalityType.INPUT_CHANGED,
                        cell_id=upstream_id,
                        cell_name=upstream_name,
                    )
                )

        # If no upstream changes detected, it must be source or env
        if not details:
            # Try to decompose by reading stored component hashes
            stored_source_hash = _get_stored_hash(session, cell_id, "source_hash")
            stored_env_hash = _get_stored_hash(session, cell_id, "env_hash")

            if stored_source_hash is None:
                stored_source_hash = cell.last_source_hash
            if stored_env_hash is None:
                stored_env_hash = cell.last_env_hash

            source_changed = stored_source_hash is not None and stored_source_hash != source_hash
            env_changed_flag = stored_env_hash is not None and stored_env_hash != env_hash

            if env_changed_flag:
                details.append(
                    CausalityDetail(
                        type=CausalityType.ENV_CHANGED,
                        package="notebook env",
                    )
                )
            if (
                not source_changed
                and not env_changed_flag
                and cell.last_provenance_hash is not None
                and cell.last_provenance_hash != provenance_hash
                and cell.upstream_ids
            ):
                upstream_id = cell.upstream_ids[0]
                upstream = session.notebook_state.get_cell(upstream_id)
                upstream_name = (
                    upstream.defines[0]
                    if upstream is not None and upstream.defines
                    else upstream_id
                )
                details.append(
                    CausalityDetail(
                        type=CausalityType.INPUT_CHANGED,
                        cell_id=upstream_id,
                        cell_name=upstream_name,
                    )
                )
            if source_changed or (not env_changed_flag):
                # If source changed, or if we couldn't determine the cause
                # (no stored hashes), fall back to source_changed
                if not details:
                    cell_name = cell.defines[0] if cell.defines else cell_id
                    details.append(
                        CausalityDetail(
                            type=CausalityType.SOURCE_CHANGED,
                            cell_id=cell_id,
                            cell_name=cell_name,
                        )
                    )

        # Determine primary reason — env takes precedence when it's the
        # *only* change, since source_changed may be a fallback guess.
        has_source = any(d.type == CausalityType.SOURCE_CHANGED for d in details)
        has_input = any(d.type == CausalityType.INPUT_CHANGED for d in details)
        has_env = any(d.type == CausalityType.ENV_CHANGED for d in details)

        if has_env and not has_source and not has_input:
            reason = CausalityReason.ENV
        elif has_input:
            reason = CausalityReason.UPSTREAM
        else:
            reason = CausalityReason.SELF

        causality_map[cell_id] = CausalityChain(reason=reason, details=details)

    return causality_map


def _get_stored_hash(session: NotebookSession, cell_id: str, key: str) -> str | None:
    """Read a component hash from a cell's stored artifact metadata.

    Parameters
    ----------
    session : NotebookSession
        Session that owns the cell's artifact store.
    cell_id : str
        Cell ID.
    key : {"source_hash", "env_hash"}
        Which component hash to read.

    Returns
    -------
    str or None
        Stored hash, or ``None`` if not available.
    """
    cell = session.notebook_state.get_cell(cell_id)
    if cell is None or not cell.artifact_uri:
        return None

    try:
        import json as _json

        parts = cell.artifact_uri.split("/")
        artifact_id = parts[-1].split("@")[0]
        version = int(parts[-1].split("@v=")[1])
        artifact = session.artifact_manager.artifact_store.get_artifact(artifact_id, version)
        if artifact and artifact.transform_spec:
            spec = _json.loads(artifact.transform_spec)
            return spec.get("params", {}).get(key)
    except (IndexError, ValueError, KeyError):
        pass
    return None
