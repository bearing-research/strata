"""Registry read services (the dashboard summary aggregation).

Stateless; the handler passes in the resolved tenant filter.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore

# A notebook cell's output id; cell ids carry no underscore.
_CELL_ARTIFACT_ID = re.compile(r"^nb_(?P<notebook_id>.+?)_cell_(?P<cell_id>[^_]+)_var_")


class RegistryService:
    """Stateless registry read aggregation."""

    def summary(
        self, store: ArtifactStore, *, tenant: str | None, all_tenants: bool = False
    ) -> list[dict]:
        """Rows for the dashboard names table: tenant, aliases, current version, tags, URI, readers.

        ``tenant`` is the caller's tenant (``None`` is the default tenant); ``all_tenants`` reads
        every tenant instead, so one name can appear once per tenant. Internal ``nb_*`` tags are
        hidden.
        """
        readers = self._readers_by_tenant(store, scope=None if all_tenants else (tenant or ""))
        if all_tenants:
            names = store.list_all_names()
            aliases = store.list_all_aliases()
        else:
            names = store.list_names(tenant=tenant)
            aliases = store.list_aliases(None, tenant=tenant)
        aliases_by_name: dict[tuple[str | None, str], dict[str, int]] = {}
        for a in aliases:
            aliases_by_name.setdefault((a.tenant, a.name), {})[a.alias] = a.version

        rows: list[dict] = []
        for n in names:
            tags = store.get_tags(n.artifact_id, n.version, tenant=n.tenant)
            rows.append(
                {
                    "tenant": n.tenant,
                    "name": n.name,
                    "artifact_id": n.artifact_id,
                    "version": n.version,
                    "uri": f"strata://artifact/{n.artifact_id}@v={n.version}",
                    "aliases": aliases_by_name.get((n.tenant, n.name), {}),
                    # Hide internal stamps (nb_cell) from the user-facing table.
                    "tags": {k: v for k, v in tags.items() if not k.startswith("nb_")},
                    "readers": readers.get((n.tenant, n.name), []),
                }
            )
        return rows

    def readers(self, store: ArtifactStore, *, tenant: str | None) -> dict[str, list[dict]]:
        """Per name, the cells in ``tenant`` whose stored results read it (``# @dataset``).

        ``None`` is the default tenant. A cell with several outputs, or several reads of one
        name, is listed once.
        """
        return {
            name: cells
            for (_, name), cells in self._readers_by_tenant(store, scope=tenant or "").items()
        }

    def _readers_by_tenant(
        self, store: ArtifactStore, *, scope: str | None
    ) -> dict[tuple[str | None, str], list[dict]]:
        # Keyed by tenant too: two tenants may each have a name of the same spelling.
        # ``scope`` None reads every tenant, and the default tenant is stored as ''.
        found: dict[tuple[str | None, str], dict[tuple[str, str], dict]] = {}
        for read_tenant, artifact_id, reference in store.list_name_reads(tenant=scope):
            cell = _CELL_ARTIFACT_ID.match(artifact_id)
            if cell is None:
                continue
            name = reference.rpartition("@")[0] if "@" in reference else reference
            key = (cell["notebook_id"], cell["cell_id"])
            found.setdefault((read_tenant, name), {}).setdefault(
                key, {"notebook_id": key[0], "cell_id": key[1], "reference": reference}
            )
        return {key: list(cells.values()) for key, cells in found.items()}

    def artifacts_by_tag(
        self, store: ArtifactStore, key: str, value: str | None = None, *, tenant: str | None
    ) -> list[dict]:
        """Ready artifacts carrying tag ``key`` (optionally ``= value``) in ``tenant``.

        ``tenant`` is the caller's tenant (``None`` is the default tenant). With ``value``
        omitted, every artifact with the key is returned, so a whole notebook costs one query.
        The matched tag is reported as ``tag_value`` and dropped from ``tags``.
        """
        if value is None:
            found = store.list_artifacts_with_tag_key(key, tenant=tenant)
        else:
            found = [
                (artifact_id, version, value)
                for artifact_id, version in store.list_artifacts_by_tag(key, value, tenant=tenant)
            ]

        rows: list[dict] = []
        for artifact_id, version, tag_value in found:
            artifact = store.get_artifact(artifact_id, version)
            if artifact is None or artifact.state != "ready":
                continue
            tags = store.get_tags(artifact_id, version, tenant=tenant)
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "version": version,
                    "uri": f"strata://artifact/{artifact_id}@v={version}",
                    "names": store.names_for_artifact(artifact_id, version, tenant=tenant),
                    "tags": {k: v for k, v in tags.items() if k != key},
                    "tag_value": tag_value,
                }
            )
        return rows


registry_service = RegistryService()
