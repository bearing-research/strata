"""A DuckDB connection over the organization's lake: its catalog and mounts.

A DuckDB connection in ``notebook.toml`` can name a catalog configured on the
server (``[tool.strata] catalogs``) and the mounts it reads::

    [connections.lake]
    driver = "duckdb"
    path = ":memory:"
    catalog = "lake"
    mounts = ["raw"]

A cell on it queries ``lake.<namespace>.<table>`` and each mount as a view by
its name. Every catalog table the query reads is an input the way an ``@table``
declaration is: its current snapshot is folded into the cell's provenance, the
query reads that snapshot (``AT (VERSION => id)``), and a new snapshot makes the
cell stale. Each mount's fingerprint is folded too.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sqlglot
from sqlglot import exp

from strata.notebook.annotations import parse_annotations
from strata.notebook.models import TableSpec

if TYPE_CHECKING:
    from strata.notebook.models import ConnectionSpec, NotebookState
    from strata.notebook.sql.adapter import QualifiedTable


class LakeError(ValueError):
    """The connection's catalog or a mount cannot be resolved."""


@dataclass
class Lake:
    """What a cell's lake connection resolved to."""

    spec: ConnectionSpec
    fingerprints: list[str] = field(default_factory=list)
    # (namespace, table) → the snapshot the query reads.
    snapshots: dict[tuple[str, str], int] = field(default_factory=dict)


def lake_options(spec: ConnectionSpec) -> tuple[str | None, list[str]]:
    """The catalog name and mount names a connection declares."""
    if spec.driver != "duckdb":
        return None, []
    catalog = getattr(spec, "catalog", None)
    mounts = getattr(spec, "mounts", None) or []
    return (str(catalog) if catalog else None), [str(m) for m in mounts]


def _table_spec(catalog: str, table: QualifiedTable) -> TableSpec:
    uri = f"{catalog}:{table.schema}.{table.name}"
    return TableSpec(name="lake_" + hashlib.sha256(uri.encode()).hexdigest()[:16], uri=uri)


def _catalog_tables(catalog: str, tables: list[QualifiedTable]) -> list[TableSpec]:
    specs = {
        spec.uri: spec
        for spec in (_table_spec(catalog, t) for t in tables if t.catalog == catalog and t.schema)
    }
    return [specs[uri] for uri in sorted(specs)]


def lake_tables(notebook_state: NotebookState, source: str) -> list[TableSpec]:
    """The catalog tables a SQL cell reads, as ``@table`` declarations.

    Staleness and the executor's generic provenance both fold these alongside a
    cell's own ``@table`` declarations, so the lake moving makes the cell stale.
    """
    annotations = parse_annotations(source)
    if annotations.sql is None or not annotations.sql.connection or annotations.sql.write:
        return []
    spec = next(
        (c for c in notebook_state.connections if c.name == annotations.sql.connection), None
    )
    if spec is None:
        return []
    catalog, _ = lake_options(spec)
    if not catalog:
        return []
    from strata.notebook.sql.analyzer import analyze_sql_cell

    return _catalog_tables(catalog, analyze_sql_cell(source, dialect="duckdb").tables)


def resolve_lake(
    session: Any,
    cell_id: str,
    source: str,
    spec: ConnectionSpec,
    tables: list[QualifiedTable],
) -> Lake:
    """Resolve the catalog, the mounts and the snapshot of every table read.

    Raises:
        LakeError: naming what could not be resolved, so the cell fails saying
            so rather than reading something else.
    """
    catalog, mount_names = lake_options(spec)
    lake = Lake(spec=spec)
    update: dict[str, Any] = {}
    config = session._lake_config()
    if catalog:
        properties = (getattr(config, "catalogs", None) or {}).get(catalog)
        if properties is None:
            raise LakeError(f"catalog {catalog!r} is not configured on this server")
        update["catalog_properties"] = dict(properties)
        from strata.notebook.tables import fingerprint_tables, resolve_table_snapshot

        specs = _catalog_tables(catalog, tables)
        fingerprints, snapshots = fingerprint_tables(specs, config)
        lake.fingerprints += fingerprints
        for table_spec in specs:
            snapshot = snapshots.get(table_spec.name)
            if snapshot is None:
                # Unresolved the first time: ask again, so the cell fails with
                # the catalog's reason, or reads what a retry found.
                try:
                    snapshot = resolve_table_snapshot(table_spec, config)
                except ValueError as exc:
                    raise LakeError(f"table {table_spec.uri}: {exc}") from exc
            namespace, _, name = table_spec.uri.partition(":")[2].rpartition(".")
            lake.snapshots[(namespace, name)] = snapshot
    if mount_names:
        update["mount_sources"] = _mount_sources(session, cell_id, source, mount_names, lake)
    if update:
        lake.spec = spec.model_copy(update=update)
    return lake


def _mount_sources(
    session: Any, cell_id: str, source: str, names: list[str], lake: Lake
) -> list[dict[str, Any]]:
    from strata.notebook.credentials import CredentialError, CredentialResolver
    from strata.notebook.mounts import MountResolver, mount_fingerprint_sync, resolve_cell_mounts

    cell = session.notebook_state.get_cell(cell_id)
    declared = {
        m.name: m
        for m in resolve_cell_mounts(
            [], cell.mounts if cell else [], parse_annotations(source).mounts
        )
    }
    resolver = MountResolver(
        cache_dir=session.path / ".strata" / "mount_cache",
        credential_resolver=CredentialResolver.from_config(
            session._lake_config(), env=dict(session.notebook_state.env)
        ),
    )
    sources: list[dict[str, Any]] = []
    for name in sorted(set(names)):
        mount = declared.get(name)
        if mount is None:
            raise LakeError(f"mount {name!r} is not declared for this cell")
        try:
            storage_options = resolver.storage_options(mount)
        except CredentialError as exc:
            raise LakeError(f"mount {name!r}: {exc}") from exc
        fingerprint = mount_fingerprint_sync(resolver, mount)
        if fingerprint is None:
            raise LakeError(f"mount {name!r} is read-write; a SQL cell reads mounts read-only")
        lake.fingerprints.append(fingerprint)
        sources.append({"name": name, "uri": mount.uri, "storage_options": storage_options})
    return sources


def pin_snapshots(sql: str, catalog: str, snapshots: dict[tuple[str, str], int]) -> str:
    """*sql* with each catalog table read at its resolved snapshot."""
    if not snapshots:
        return sql
    tree = sqlglot.parse_one(sql, read="duckdb")
    for reference in tree.find_all(exp.Table):
        snapshot = snapshots.get((reference.db, reference.name))
        if reference.catalog != catalog or snapshot is None or reference.args.get("when"):
            continue
        template = sqlglot.parse_one(f"SELECT * FROM t AT (VERSION => {int(snapshot)})", "duckdb")
        clause = template.find(exp.Table)
        assert clause is not None
        reference.set("when", clause.args["when"])
    return tree.sql(dialect="duckdb")
