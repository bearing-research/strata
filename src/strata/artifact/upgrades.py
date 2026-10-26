"""What Strata completes in a ledger it opens, once per store.

Each step is keyed on the ledger's ``store_upgrades`` marker: a crash before the mark
redoes it, a fresh store marks it on first open. Called wherever a store is opened.
"""

from __future__ import annotations

from vestigo.store import ArtifactStore

from strata.artifact.graph import reconstruct_definitions
from strata.artifact.retention import record_folder_edges


def complete_upgrade(store: ArtifactStore) -> None:
    record_folder_edges(store)
    reconstruct_definitions(store)
