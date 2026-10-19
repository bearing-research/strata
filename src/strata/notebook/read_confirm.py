"""Asking before a large read (Amber item 90).

Before a notebook run, each cell's inputs are weighed on the machine it will run on
(``CellExecutor.estimate_reads``). What they would move there, the bytes its cache does
not hold, is compared with ``notebook_confirm_read_bytes``, and its cost at each
credential's ``notebook_credential_read_cost`` with ``notebook_confirm_read_cost``. Over
either, the notebook asks before running, and a yes is kept per cell, machine and input
fingerprint. A run that cannot ask goes ahead; the reads index records which it was.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from strata.notebook.models import InputWeight

ReadConfirmation = Literal["under_threshold", "approved", "not_asked"]

# Rates are per GB.
_GB = 10**9


@dataclass(frozen=True)
class ReadEstimate:
    """What a cell's inputs would move onto *machine*, and what it would cost."""

    cell_id: str
    machine: str
    inputs: list[InputWeight]
    # Names the inputs a yes was given for: mount listings, table snapshots, fetch URLs.
    fingerprint: str
    bytes_to_move: int
    cost: float

    def payload(self) -> dict[str, Any]:
        """As a ``read_confirm_prompt`` cell, or a 409 ``large_read`` body."""
        return {
            "cell_id": self.cell_id,
            "machine": self.machine,
            "inputs": [w.model_dump(exclude_none=True) for w in self.inputs],
            "bytes_to_move": self.bytes_to_move,
            "cost": self.cost,
        }


def asks(config: Any) -> bool:
    """Whether any threshold is set."""
    return (
        getattr(config, "notebook_confirm_read_bytes", None) is not None
        or getattr(config, "notebook_confirm_read_cost", None) is not None
    )


def input_fingerprint(
    mount_fingerprints: list[str], table_fingerprints: list[str], fetch_urls: list[str]
) -> str:
    """What a yes is kept for. A fetch counts by its URL: its size is known only once read."""
    parts = [sorted(mount_fingerprints), sorted(table_fingerprints), sorted(fetch_urls)]
    return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()


def estimate(
    cell_id: str, machine: str, weights: list[InputWeight], fingerprint: str, config: Any
) -> ReadEstimate:
    """The bytes *weights* would move, uncached and known, and their cost at the set rates.

    An input with no credential is rated by its scheme's default credential
    (``notebook_mount_credentials``). A size that is not known counts nothing.
    """
    rates: dict[str, float] = getattr(config, "notebook_credential_read_cost", None) or {}
    defaults: dict[str, str] = getattr(config, "notebook_mount_credentials", None) or {}
    to_move = 0
    cost = 0.0
    for weight in weights:
        if weight.bytes is None:
            continue
        moved = max(weight.bytes - (weight.cached_bytes or 0), 0)
        to_move += moved
        credential = weight.credential or defaults.get(urlparse(weight.uri).scheme)
        cost += moved / _GB * rates.get(credential or "", 0.0)
    return ReadEstimate(
        cell_id=cell_id,
        machine=machine,
        inputs=list(weights),
        fingerprint=fingerprint,
        bytes_to_move=to_move,
        cost=round(cost, 6),
    )


def over_threshold(read: ReadEstimate, config: Any) -> bool:
    """Whether *read* moves or costs more than a threshold allows."""
    max_bytes = getattr(config, "notebook_confirm_read_bytes", None)
    max_cost = getattr(config, "notebook_confirm_read_cost", None)
    return (max_bytes is not None and read.bytes_to_move > max_bytes) or (
        max_cost is not None and read.cost > max_cost
    )


def confirmation(read: ReadEstimate, config: Any, approved: bool) -> ReadConfirmation | None:
    """What the reads index records: ``None`` with no threshold set."""
    if not asks(config):
        return None
    if not over_threshold(read, config):
        return "under_threshold"
    return "approved" if approved else "not_asked"
