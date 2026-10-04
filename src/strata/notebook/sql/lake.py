"""A DuckDB connection over the organization's lake: its catalog and mounts.

A DuckDB connection can name a catalog (the server's ``[tool.strata] catalogs``
or, in personal mode, the notebook's ``[catalogs.<name>]``) and the mounts it
reads::

    [connections.lake]
    driver = "duckdb"
    path = ":memory:"
    catalog = "lake"
    mounts = ["raw"]

Cells query ``lake.<namespace>.<table>`` and each mount as a view. Every catalog
table read is an input like an ``@table``: its snapshot is folded into
provenance and pinned in the query (``AT (VERSION => id)``), so a new snapshot
makes the cell stale. Mount fingerprints are folded too.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlglot
from sqlglot import exp

from strata.notebook.annotations import parse_annotations
from strata.notebook.models import TableSpec
from strata.notebook.sql.adapter import QualifiedTable

if TYPE_CHECKING:
    from strata.notebook.models import ConnectionSpec, NotebookState


# Artifact transform param a ``# @cache snapshot`` cell records: the
# ``[namespace, table, snapshot]`` rows its query read.
PARAM_SNAPSHOT_IDS = "sql_snapshot_ids"


class LakeError(ValueError):
    """The connection's catalog or a mount cannot be resolved."""


@dataclass
class Lake:
    """What a cell's lake connection resolved to."""

    spec: ConnectionSpec
    fingerprints: list[str] = field(default_factory=list)
    # (namespace, table) → the snapshot the query reads.
    snapshots: dict[tuple[str, str], int] = field(default_factory=dict)
    # Service mode only (see duckdb._confine): each mount's root and each catalog
    # table's location, readable once confined.
    locations: list[str] = field(default_factory=list)


def lake_options(spec: ConnectionSpec) -> tuple[str | None, list[str]]:
    """The catalog name and mount names a connection declares."""
    if spec.driver != "duckdb":
        return None, []
    catalog = getattr(spec, "catalog", None)
    mounts = getattr(spec, "mounts", None) or []
    return (str(catalog) if catalog else None), [str(m) for m in mounts]


def with_notebook_catalogs(config: Any, notebook_state: NotebookState) -> Any:
    """*config* with the notebook's ``[catalogs.<name>]`` added, in personal mode only.

    A notebook entry wins over a server one of the same name, as a notebook's
    worker does. Service mode reads only the catalogs the server configures.
    """
    catalogs = getattr(notebook_state, "catalogs", None)
    if not catalogs or getattr(config, "deployment_mode", "personal") == "service":
        return config
    return config.model_copy(update={"catalogs": {**(config.catalogs or {}), **catalogs}})


def _table_spec(catalog: str, table: QualifiedTable) -> TableSpec:
    uri = f"{catalog}:{table.schema}.{table.name}"
    return TableSpec(name="lake_" + hashlib.sha256(uri.encode()).hexdigest()[:16], uri=uri)


def catalog_table(catalog: str, table: QualifiedTable) -> tuple[str, str] | None:
    """The ``(namespace, table)`` *table* names in *catalog*, or None.

    DuckDB names are case-insensitive and the default schema may be omitted
    (``lake.taxi.trips``, ``LAKE.taxi.trips``, ``lake.trips``). A form this
    misses is read live and never goes stale.
    """
    name = catalog.lower()
    if (table.catalog or "").lower() == name:
        return (table.schema, table.name) if table.schema else None
    if not table.catalog and (table.schema or "").lower() == name:
        # ``<catalog>.<table>``: DuckDB fills in the default schema.
        return ("main", table.name)
    return None


def _catalog_tables(catalog: str, tables: list[QualifiedTable]) -> list[TableSpec]:
    named = [_namespaced(catalog, t) for t in tables]
    specs = {spec.uri: spec for spec in named if spec is not None}
    return [specs[uri] for uri in sorted(specs)]


def _namespaced(catalog: str, table: QualifiedTable) -> TableSpec | None:
    found = catalog_table(catalog, table)
    if found is None:
        return None
    namespace, name = found
    return _table_spec(catalog, QualifiedTable(catalog=catalog, schema=namespace, name=name))


def lake_tables(notebook_state: NotebookState, source: str) -> list[TableSpec]:
    """The catalog tables a SQL cell reads, as ``@table`` declarations.

    Staleness and generic provenance fold these with the cell's own ``@table``
    declarations. A ``# @cache snapshot`` cell reads the snapshots it pinned, so
    a new one does not make it stale.
    """
    annotations = parse_annotations(source)
    if annotations.sql is None or not annotations.sql.connection or annotations.sql.write:
        return []
    if annotations.cache is not None and annotations.cache.kind == "snapshot":
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
    pinned: dict[tuple[str, str], int] | None = None,
) -> Lake:
    """Resolve the catalog, the mounts and the snapshot of every table read.

    A table in *pinned* (``(namespace, table)`` to snapshot) reads that snapshot
    rather than the current one.

    Raises:
        LakeError: naming what could not be resolved, rather than reading something else.
    """
    catalog, mount_names = lake_options(spec)
    lake = Lake(spec=spec)
    update: dict[str, Any] = {}
    config = with_notebook_catalogs(session._lake_config(), session.notebook_state)
    confined = getattr(config, "deployment_mode", "personal") == "service"
    if catalog:
        properties = (getattr(config, "catalogs", None) or {}).get(catalog)
        if properties is None and catalog in session.notebook_state.catalogs:
            raise LakeError(
                f"catalog {catalog!r} is defined in notebook.toml, and a server in service "
                "mode reads only the catalogs it configures"
            )
        if properties is None:
            raise LakeError(f"catalog {catalog!r} is not configured on this server")
        from strata.notebook.credentials import (
            CredentialError,
            CredentialResolver,
            resolve_catalog_properties,
        )

        try:
            update["catalog_properties"] = resolve_catalog_properties(
                properties,
                CredentialResolver.from_config(config, env=dict(session.notebook_state.env)),
            )
        except CredentialError as exc:
            raise LakeError(f"catalog {catalog!r}: {exc}") from exc
        from strata.notebook.tables import fingerprint_tables, resolve_table_snapshot

        pinned = pinned or {}
        specs = {_table_key(spec): spec for spec in _catalog_tables(catalog, tables)}
        _, snapshots = fingerprint_tables(
            [spec for key, spec in specs.items() if key not in pinned], config
        )
        for (namespace, name), table_spec in specs.items():
            snapshot = pinned.get((namespace, name), snapshots.get(table_spec.name))
            if snapshot is None:
                # Unresolved the first time: ask again, so the cell fails with the catalog's
                # reason or reads what a retry found.
                try:
                    snapshot = resolve_table_snapshot(table_spec, config)
                except ValueError as exc:
                    raise LakeError(f"table {table_spec.uri}: {exc}") from exc
            lake.snapshots[(namespace, name)] = snapshot
            # From the snapshot the query reads, so a catalog that answered only on retry still
            # gives a reproducible hash (fingerprint_tables invents a random one otherwise).
            lake.fingerprints.append(f"{table_spec.name}:table:{table_spec.uri}:{snapshot}")
            if confined:
                lake.locations.append(_table_location(table_spec, config))
    if mount_names:
        update["mount_sources"] = _mount_sources(
            session, cell_id, source, mount_names, lake, config if confined else None
        )
        if confined:
            lake.locations.extend(_mount_location(m["uri"]) for m in update["mount_sources"])
    if update:
        lake.spec = spec.model_copy(update=update)
    return lake


def _table_key(table_spec: TableSpec) -> tuple[str, str]:
    namespace, _, name = table_spec.uri.partition(":")[2].rpartition(".")
    return namespace, name


def snapshot_problem(catalog: str, analysis: Any) -> str | None:
    """Why ``# @cache snapshot`` cannot pin this DuckDB query, or None.

    Only the catalog's tables have snapshots that can be read again; a mount, a
    table in the connection's own database or one named at run time can change
    under the pin.
    """
    if analysis.unresolved_tables:
        return (
            "@cache snapshot needs every table the query reads named in it; "
            f"{analysis.unresolved_tables[0]} is named at run time"
        )
    for table in analysis.tables:
        if catalog_table(catalog, table) is None:
            return (
                "@cache snapshot on a DuckDB connection reads only tables of its catalog "
                f"{catalog!r}, whose snapshots can be read again; {table.render()} is not one"
            )
    return None


def snapshot_rows(snapshots: dict[tuple[str, str], int]) -> list[list[Any]]:
    """*snapshots* as sorted ``[namespace, table, snapshot]`` rows, for JSON."""
    return sorted([namespace, name, snapshot] for (namespace, name), snapshot in snapshots.items())


def _table_location(table_spec: TableSpec, config: Any) -> str:
    """Where a catalog table's metadata and data files live, as a directory.

    Files outside it (a ``write.data.path`` elsewhere) are refused to a
    confined handle, and the query fails naming the file.
    """
    from strata.iceberg import PyIcebergCatalog

    try:
        location = PyIcebergCatalog(config).load_table(table_spec.uri).location()
    except Exception as exc:  # the catalog's own error types vary by backend
        raise LakeError(f"table {table_spec.uri}: {exc}") from exc
    return location.rstrip("/") + "/"


def _mount_location(uri: str) -> str:
    """A mount's root as a location: a directory (``.../``), or its one file."""
    from strata.notebook.sql.drivers.duckdb import mount_root

    root = mount_root(uri).rstrip("/")
    if root.endswith((".parquet", ".csv", ".json")):
        return root
    return root + "/"


# Never a confined cell's mount root, nor under one: the running process's
# files (``/proc/self/environ``), kernel state and devices.
_SYSTEM_TREES = (Path("/proc"), Path("/sys"), Path("/dev"))


def local_mount_root_problem(uri: str, config: Any) -> str | None:
    """Why a confined SQL cell may not mount the local root *uri*, or ``None``.

    A confined handle reads everything under the root as the server, so after
    following links this refuses a root with fewer than two path components,
    one holding server state (artifact store, cache, metadata DB, notebook
    storage, the server's home), or one under the home, ``/proc``, ``/sys`` or
    ``/dev``. Remote mounts are not checked.
    """
    from strata.notebook.mounts import parse_mount_uri

    scheme, path = parse_mount_uri(uri)
    if scheme != "file":
        return None
    root = Path(os.path.realpath(path or "/"))
    if len(root.parts) < 3:
        return f"its root {root} is too near the top of the filesystem"
    home = Path(os.path.realpath(Path.home()))
    metadata_db = getattr(config, "metadata_db", None)
    state = {
        "artifact store": getattr(config, "artifact_dir", None),
        "cache": getattr(config, "cache_dir", None),
        "metadata database": Path(metadata_db).parent if metadata_db else None,
        "notebook storage": getattr(config, "notebook_storage_dir", None),
        "home directory": home,
    }
    for label, location in state.items():
        if location is None:
            continue
        resolved = Path(os.path.realpath(location))
        if root == resolved or root in resolved.parents:
            return f"its root {root} holds the server's {label} ({resolved})"
    for tree in (home, *_SYSTEM_TREES):
        if tree in root.parents:
            return f"its root {root} is inside {tree}"
    return None


def _mount_sources(
    session: Any,
    cell_id: str,
    source: str,
    names: list[str],
    lake: Lake,
    confined_config: Any | None = None,
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
        if confined_config is not None:
            # Before anything under the root is read: fingerprinting ``/``
            # would walk the whole disk.
            problem = local_mount_root_problem(mount.uri, confined_config)
            if problem is not None:
                raise LakeError(
                    f"mount {name!r}: {problem}, and a SQL cell on this server reads "
                    "everything under a mount's root as the server"
                )
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
        found = catalog_table(
            catalog,
            QualifiedTable(
                catalog=reference.catalog or None, schema=reference.db or None, name=reference.name
            ),
        )
        snapshot = snapshots.get(found) if found is not None else None
        if snapshot is None or reference.args.get("when"):
            continue
        template = sqlglot.parse_one(f"SELECT * FROM t AT (VERSION => {int(snapshot)})", "duckdb")
        clause = template.find(exp.Table)
        assert clause is not None
        reference.set("when", clause.args["when"])
    return tree.sql(dialect="duckdb")
