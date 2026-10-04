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

    def summary(self, store: ArtifactStore, *, tenant: str | None) -> list[dict]:
        """Rows for the dashboard names table: aliases, current version, tags, URI and readers.

        ``tenant`` is the resolved scope (``None`` sees all). Internal ``nb_*`` tags
        are hidden.
        """
        readers = self.readers(store, tenant=tenant)
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
                    "readers": readers.get(n.name, []),
                }
            )
        return rows

    def readers(self, store: ArtifactStore, *, tenant: str | None) -> dict[str, list[dict]]:
        """Per name, the notebook cells whose stored results read it (``# @dataset``).

        A cell with several outputs, or several reads of one name, is listed once.
        """
        found: dict[str, dict[tuple[str, str], dict]] = {}
        for artifact_id, reference in store.list_name_reads(tenant=tenant):
            cell = _CELL_ARTIFACT_ID.match(artifact_id)
            if cell is None:
                continue
            name = reference.rpartition("@")[0] if "@" in reference else reference
            key = (cell["notebook_id"], cell["cell_id"])
            found.setdefault(name, {}).setdefault(
                key, {"notebook_id": key[0], "cell_id": key[1], "reference": reference}
            )
        return {name: list(cells.values()) for name, cells in found.items()}

    def artifacts_by_tag(
        self, store: ArtifactStore, key: str, value: str | None = None, *, tenant: str | None
    ) -> list[dict]:
        """Ready artifacts carrying tag ``key`` (optionally ``= value``).

        With ``value`` omitted, every artifact with the key is returned, so a whole
        notebook costs one query. The matched tag is reported as ``tag_value`` and
        dropped from ``tags``.
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
                    "names": store.names_for_artifact(artifact_id, version),
                    "tags": {k: v for k, v in tags.items() if k != key},
                    "tag_value": tag_value,
                }
            )
        return rows


registry_service = RegistryService()
