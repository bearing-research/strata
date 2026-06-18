"""Registry read services (the dashboard summary aggregation).

Most registry routes are thin store delegations and stay in the handler. The
``summary`` aggregation is the one with real shaping logic — alias grouping and
hiding internal (``nb_``) tags from the user-facing table — so it lives here and
is unit-testable with a fake store. Stateless; the resolved tenant filter is
passed in by the handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore


class RegistryService:
    """Stateless registry read aggregation."""

    def summary(self, store: ArtifactStore, *, tenant: str | None) -> list[dict]:
        """Rows for the dashboard names table: each name with its aliases
        (``alias -> version``), current version, that version's tags, and a URI.

        ``tenant`` is the already-resolved scope (``None`` = personal / ``admin:*``
        sees all). Internal ``nb_*`` stamps are hidden from the returned tags.
        """
        aliases_by_name: dict[str, dict[str, int]] = {}
        for a in store.list_aliases(None, tenant=tenant):
            aliases_by_name.setdefault(a.name, {})[a.alias] = a.version

        rows: list[dict] = []
        for n in store.list_names(tenant=tenant):
            tags = store.get_tags(n.artifact_id, n.version, tenant=tenant)
            rows.append(
                {
                    "name": n.name,
                    "artifact_id": n.artifact_id,
                    "version": n.version,
                    "uri": f"strata://artifact/{n.artifact_id}@v={n.version}",
                    "aliases": aliases_by_name.get(n.name, {}),
                    # Hide internal stamps (nb_cell) from the user-facing table.
                    "tags": {k: v for k, v in tags.items() if not k.startswith("nb_")},
                }
            )
        return rows


registry_service = RegistryService()
