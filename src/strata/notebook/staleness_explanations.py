"""Why a cell is stale: the parts its last good run was keyed on, against the same parts now.

The parts come from the key itself, so each difference is a real change. What cannot be
compared (a record from before parts, a language's own key, an input that could not be
resolved) is reported as that, never guessed at. The wire field is still ``causality``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import cellgraph as cg

from strata.notebook import nonce
from strata.notebook.models import StalenessReason

if TYPE_CHECKING:
    from collections.abc import Mapping

    from strata.notebook.models import CellState
    from strata.notebook.session import NotebookSession


class CausalityType(StrEnum):
    """Which part of the key moved."""

    SOURCE_CHANGED = "source_changed"
    INPUT_CHANGED = "input_changed"  # an upstream cell's output
    ENV_CHANGED = "env_changed"
    EXTERNAL_CHANGED = "external_changed"  # a mount, table, fetch or dataset
    IDENTITY_CHANGED = "identity_changed"  # a SQL cell's connection, a prompt cell's model
    UNAVAILABLE = "unavailable"  # an input that cannot be read now
    CHECKING = "checking"  # a local mount whose files are being hashed


_STALE_BECAUSE = (
    (StalenessReason.SELF, {CausalityType.SOURCE_CHANGED}),
    (StalenessReason.UPSTREAM, {CausalityType.INPUT_CHANGED}),
    (StalenessReason.EXTERNAL, {CausalityType.EXTERNAL_CHANGED, CausalityType.IDENTITY_CHANGED}),
    (StalenessReason.ENV, {CausalityType.ENV_CHANGED}),
    # Not known to have changed: shown apart from a change.
    (StalenessReason.CHECKING, {CausalityType.CHECKING}),
)


class CausalityReason(StrEnum):
    """The most actionable of a chain's details."""

    SELF = "self"
    UPSTREAM = "upstream"
    EXTERNAL = "external"
    ENV = "env"
    CHECKING = "checking"
    UNKNOWN = "unknown"


def skip_none(pairs: list[tuple[str, object]]) -> dict:
    """``asdict`` ``dict_factory`` that drops None fields, keeping them off the wire."""
    return {k: v for k, v in pairs if v is not None}


@dataclass
class CausalityDetail:
    """One part that moved: an upstream cell (``cell_*``), an outside input (``input_*``),
    or the environment (``package``)."""

    type: CausalityType
    cell_id: str | None = None
    cell_name: str | None = None
    input_kind: str | None = None
    input_name: str | None = None
    package: str | None = None


@dataclass
class CausalityChain:
    """Why a stale cell is stale. ``incomplete``: there may be more than ``details`` say."""

    reason: CausalityReason
    details: list[CausalityDetail] = field(default_factory=list)
    incomplete: bool | None = None


@dataclass(frozen=True)
class StaleCell:
    """What the staleness walk knows of a cell it did not find ready.

    ``current`` is its key's parts now, without ``upstream`` when a stale upstream settled
    the cell first (those are ``stale_upstreams``), and None when it has no key.
    """

    current: dict[str, Any] | None
    stale_upstreams: tuple[str, ...] = ()


def explain_staleness(
    session: NotebookSession, stale: Mapping[str, StaleCell]
) -> dict[str, CausalityChain]:
    """Return ``{cell_id: CausalityChain}`` for the *stale* cells that have a result to compare."""
    chains: dict[str, CausalityChain] = {}
    for cell_id, found in stale.items():
        cell = session.notebook_state.get_cell(cell_id)
        # Never ran: there is no earlier result for anything to have changed from.
        if cell is not None and cell.last_provenance_hash is not None:
            chains[cell_id] = _explain(session, cell, found)
    return chains


def stale_reasons(chain: CausalityChain) -> list[StalenessReason]:
    """What a chain knows moved under a cell's result. An unreadable input says nothing."""
    types = {detail.type for detail in chain.details}
    return [reason for reason, moved in _STALE_BECAUSE if types & moved]


def _explain(session: NotebookSession, cell: CellState, found: StaleCell) -> CausalityChain:
    stored = cell.last_key_parts
    # Another schema's addresses mean nothing here. Its digest may still be good.
    older_scheme = stored is not None and stored.get("parts_schema") != cg.PARTS_SCHEMA
    if older_scheme:
        stored = None
    current = found.current or {}
    # Older records kept these two, and they are still facts, but another schema's
    # definition key is another formula's: comparing it would say every cell was edited.
    recorded = stored or {"env_hash": cell.last_env_hash}
    if not older_scheme:
        recorded = stored or {**recorded, "definition_key": cell.last_source_hash}

    source_changed = _differs(recorded, current, "definition_key")
    details: list[CausalityDetail] = []
    if source_changed:
        details.append(
            CausalityDetail(
                type=CausalityType.SOURCE_CHANGED, cell_id=cell.id, cell_name=_name(cell)
            )
        )
    details.extend(
        CausalityDetail(
            type=CausalityType.INPUT_CHANGED,
            cell_id=upstream_id,
            cell_name=_cell_name(session, upstream_id),
        )
        for upstream_id in found.stale_upstreams
    )
    if stored is not None:
        known = set(found.stale_upstreams)
        details.extend(_upstream_changes(session, cell, stored, current, source_changed, known))
        details.extend(_input_changes(stored, current, source_changed))
    if _differs(recorded, current, "env_hash"):
        details.append(CausalityDetail(type=CausalityType.ENV_CHANGED, package="notebook env"))

    identity = session.reopen_identity(cell)
    identity_settled = identity is not None and cell.last_reopen_identity is not None
    # A SQL cell's database, probed before the walk, gave another identity.
    probed_changed = session.probed_identity_changed(cell)
    if (identity_settled and identity != cell.last_reopen_identity) or probed_changed:
        details.append(CausalityDetail(type=CausalityType.IDENTITY_CHANGED))
    identity_settled = identity_settled or probed_changed

    complete = stored is not None and found.current is not None and identity_settled and details
    return CausalityChain(
        reason=_reason(details), details=details, incomplete=None if complete else True
    )


def _differs(stored: Mapping[str, Any], current: Mapping[str, Any], part: str) -> bool:
    before, now = stored.get(part), current.get(part)
    return before is not None and now is not None and before != now


def _upstream_changes(
    session: NotebookSession,
    cell: CellState,
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
    source_changed: bool,
    known: set[str],
) -> list[CausalityDetail]:
    """One detail per upstream cell whose parts moved; *known* ones are already reported."""
    if "upstream" not in current:
        return []
    before = {address: value for address, value in stored["upstream"]}
    now = {address: value for address, value in current["upstream"]}
    details: list[CausalityDetail] = []
    for address in sorted(before.keys() | now.keys()):
        if before.get(address) == now.get(address):
            continue
        # An edit rewires what the cell reads; that is the edit, already reported.
        if source_changed and (address not in before or address not in now):
            continue
        upstream_id, variable = _producer(cell, session, address)
        if upstream_id is not None and upstream_id in known:
            continue
        if upstream_id is not None:
            known.add(upstream_id)
        # Published once and unreadable now: the key skips such an output.
        unreadable = (
            address not in now
            and upstream_id is not None
            and _defines(session, upstream_id, variable)
        )
        details.append(
            CausalityDetail(
                type=CausalityType.UNAVAILABLE if unreadable else CausalityType.INPUT_CHANGED,
                cell_id=upstream_id,
                cell_name=_cell_name(session, upstream_id) if upstream_id else variable,
            )
        )
    return details


def _producer(
    cell: CellState, session: NotebookSession, address: str
) -> tuple[str | None, str | None]:
    """The upstream cell and variable a part's address names (see cellgraph's ``parts``)."""
    kind, _, rest = address.partition(":")
    if kind == "cell":
        return rest, None
    if kind == "sweep":
        for upstream_id in cell.upstream_ids:
            if _defines(session, upstream_id, rest):
                return upstream_id, rest
        return None, rest
    upstream_id, _, variable = rest.partition(".")
    return upstream_id, variable.partition("@")[0]


def _defines(session: NotebookSession, cell_id: str, variable: str | None) -> bool:
    cell = session.notebook_state.get_cell(cell_id)
    return cell is not None and variable in cell.defines


def _input_changes(
    stored: Mapping[str, Any], current: Mapping[str, Any], source_changed: bool
) -> list[CausalityDetail]:
    if "inputs" not in current:
        return []
    before = {(kind, name): component for kind, name, component in stored["inputs"]}
    now = {(kind, name): component for kind, name, component in current["inputs"]}
    details: list[CausalityDetail] = []
    for address in sorted(before.keys() | now.keys()):
        if before.get(address) == now.get(address):
            continue
        # Declaring or dropping an input in the source is the edit, already reported.
        if source_changed and (address not in before or address not in now):
            continue
        kind, name = address
        details.append(
            CausalityDetail(type=_input_change(now.get(address)), input_kind=kind, input_name=name)
        )
    return details


def _input_change(component: str | None) -> CausalityType:
    if component is not None and nonce.is_unresolved(component):
        return CausalityType.UNAVAILABLE
    if component is not None and nonce.is_unhashed(component):
        return CausalityType.CHECKING
    return CausalityType.EXTERNAL_CHANGED


def _reason(details: list[CausalityDetail]) -> CausalityReason:
    types = {detail.type for detail in details}
    if CausalityType.SOURCE_CHANGED in types:
        return CausalityReason.SELF
    if any(detail.cell_id is not None or detail.cell_name is not None for detail in details):
        return CausalityReason.UPSTREAM
    if types - {CausalityType.ENV_CHANGED} == {CausalityType.CHECKING}:
        return CausalityReason.CHECKING
    if types - {CausalityType.ENV_CHANGED}:
        return CausalityReason.EXTERNAL
    return CausalityReason.ENV if types else CausalityReason.UNKNOWN


def _name(cell: CellState) -> str:
    return cell.defines[0] if cell.defines else cell.id


def _cell_name(session: NotebookSession, cell_id: str) -> str:
    cell = session.notebook_state.get_cell(cell_id)
    return _name(cell) if cell is not None else cell_id
