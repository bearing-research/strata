"""What a cell reads from outside the notebook, by kind: the one place the kinds are named.

A cell's key folds in one fingerprint per declared outside input. The staleness pass
(``NotebookSession._outside_world_for``) and a run (``CellExecutor._compute_cell_provenance``)
each gather them, and both hand them on as one :class:`InputFingerprints`, so a new kind
is added here, not in every signature and dict that carries them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# How each kind's fingerprint begins after ``<name>:``. Its order is the order the legacy
# provenance hash concatenates them in (``cell_provenance_inputs``).
PREFIX: dict[str, str] = {"mount": "", "table": "table:", "fetch": "fetch:", "dataset": "dataset:"}


@dataclass(frozen=True)
class InputFingerprints:
    """One cell's outside-input fingerprints, by kind; each begins with its input's name."""

    mounts: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    fetches: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)

    def by_kind(self) -> dict[str, list[str]]:
        """The fingerprints keyed by kind, in :data:`PREFIX`'s order."""
        return {
            "mount": self.mounts,
            "table": self.tables,
            "fetch": self.fetches,
            "dataset": self.datasets,
        }

    def in_hash_order(self) -> list[str]:
        """Every fingerprint, kind after kind, as the legacy provenance hash reads them."""
        return [fingerprint for found in self.by_kind().values() for fingerprint in found]
