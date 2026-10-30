"""Causality inspector: explains why a cell is stale.

Diffs the current provenance components (source, input and env hashes) against
those stored with the cached artifact. The same diff, in past tense, answers
"Why did this run?" after execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from strata.notebook.provenance import compute_source_hash

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
    """``asdict`` ``dict_factory`` that drops None fields, keeping them off the wire."""
    return {k: v for k, v in pairs if v is not None}


@dataclass
class CausalityDetail:
    """A single reason contributing to staleness.

    ``cell_id``/``cell_name`` name the changed cell for source and input changes,
    ``from_version``/``to_version`` are artifact versions for ``input_changed``, and
    the ``package`` fields describe ``env_changed``.
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
    """Full causality explanation for a stale cell."""

    reason: CausalityReason
    details: list[CausalityDetail] = field(default_factory=list)


def compute_causality_on_staleness(
    session: NotebookSession,
    stale: dict[str, str | None],
) -> dict[str, CausalityChain]:
    """Return ``{cell_id: CausalityChain}`` for the stale cells of *session*.

    *stale* maps each cell ``compute_staleness()`` did not find ready to the provenance
    hash it computed, or None when a stale upstream settled the cell first. Staleness
    alone sees every input (tables, fetches, datasets), so it decides which cells need
    explaining; this reports which provenance component changed.
    """
    if session.dag is None:
        return {}

    causality_map: dict[str, CausalityChain] = {}

    for cell_id in session.dag.topological_order:
        if cell_id not in stale:
            continue
        cell = session.notebook_state.get_cell(cell_id)
        if cell is None:
            continue
        # Never ran: there is no earlier result for anything to have changed from.
        if cell.last_provenance_hash is None:
            continue

        details: list[CausalityDetail] = []
        source_hash = compute_source_hash(cell.source)
        env_hash = session.cell_env_hash(cell)
        provenance_hash = stale[cell_id]

        for upstream_id in cell.upstream_ids:
            upstream = session.notebook_state.get_cell(upstream_id)
            if upstream is None:
                continue

            if upstream.status in ("stale", "idle", "error"):
                upstream_name = upstream.defines[0] if upstream.defines else upstream_id
                details.append(
                    CausalityDetail(
                        type=CausalityType.INPUT_CHANGED,
                        cell_id=upstream_id,
                        cell_name=upstream_name,
                    )
                )
            elif upstream_id in causality_map:
                # Upstream itself changed, so our inputs changed transitively.
                upstream_name = upstream.defines[0] if upstream.defines else upstream_id
                details.append(
                    CausalityDetail(
                        type=CausalityType.INPUT_CHANGED,
                        cell_id=upstream_id,
                        cell_name=upstream_name,
                    )
                )

        # No upstream change: it must be source or env.
        if not details:
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
                and provenance_hash is not None
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
                # No stored hashes to decompose: fall back to source_changed.
                if not details:
                    cell_name = cell.defines[0] if cell.defines else cell_id
                    details.append(
                        CausalityDetail(
                            type=CausalityType.SOURCE_CHANGED,
                            cell_id=cell_id,
                            cell_name=cell_name,
                        )
                    )

        # Env wins when it is the only change, since source_changed may be a fallback guess.
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
    """Read *key* (``source_hash`` or ``env_hash``) from a cell's stored artifact, or None."""
    cell = session.notebook_state.get_cell(cell_id)
    if cell is None or not cell.artifact_uri:
        return None

    try:
        # Last ``@v=``: a fan-out instance's id has an ``@`` of its own.
        artifact_id, _, raw_version = cell.artifact_uri.split("/")[-1].rpartition("@v=")
        version = int(raw_version)
        artifact = session.artifact_manager.artifact_store.get_artifact(artifact_id, version)
        if artifact is not None:
            return artifact.params.get(key)
    except (IndexError, ValueError, KeyError):
        pass
    return None
