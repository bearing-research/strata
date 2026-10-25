"""Writing a cell's ``@write_table`` variables into Iceberg tables.

The executor calls :func:`write_cell_tables` between staging a run's outputs and making
them current. A write that fails discards the run's outputs, so nothing is cached and the
next run writes again; a cache hit never reaches here, so the same inputs never write
twice. Each snapshot's summary names the output's artifact id, version and provenance
hash, as an export's does.

Several tables are written one after another, not together: if a later one fails, the
earlier ones keep their new snapshot and the next run writes them again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vestigo.store import StagedVersion

from strata.notebook.harness_user import running_server_config
from strata.notebook.models import TableWriteSpec

if TYPE_CHECKING:
    import pyarrow as pa

    from strata.iceberg.catalog import TableWrite


class TableWriteError(RuntimeError):
    """A ``@write_table`` could not be written; the message says which and why."""


@dataclass(frozen=True)
class StagedOutput:
    """One variable of a run, staged but not yet current."""

    staged: StagedVersion
    blob: bytes
    content_type: str
    provenance_hash: str


def _request_scope() -> tuple[str | None, str | None]:
    """``(principal id, tenant)`` the write is for, as an export's; ``None`` without auth."""
    config = running_server_config()
    if config is None or not config.principal_auth_enabled:
        return None, None
    from fastapi import HTTPException

    from strata.auth.principal import get_principal
    from strata.server import _get_artifact_request_tenant

    principal = get_principal()
    try:
        tenant = _get_artifact_request_tenant()
    except HTTPException as exc:
        raise TableWriteError(f"@write_table: no caller to write as ({exc.detail})") from exc
    return (principal.id if principal is not None else None), tenant


def _authorize_service_write() -> None:
    """Service mode's gates, as an export passes them: writes enabled, ``artifacts:write``."""
    server = running_server_config()
    if server is None or server.writes_enabled:
        return
    if not server.service_writes_enabled:
        raise TableWriteError(
            "@write_table: this server does not write in service mode "
            "(STRATA_SERVICE_WRITES_ENABLED)"
        )
    from fastapi import HTTPException

    from strata.server import _authorize_artifact_write

    try:
        _authorize_artifact_write()
    except HTTPException as exc:
        detail = exc.detail.get("message") if isinstance(exc.detail, dict) else exc.detail
        raise TableWriteError(f"@write_table: not allowed ({detail})") from exc


def _authorize_credential(table: str, config: Any, env: dict[str, str], notebook: Path) -> None:
    """A catalog's ``strata_credential`` writes only for a principal its writers list."""
    from strata.auth.principal import get_principal
    from strata.iceberg.catalog import named_catalog
    from strata.notebook.credentials import (
        CATALOG_CREDENTIAL_KEY,
        CredentialError,
        CredentialResolver,
    )

    catalog, _ = named_catalog(table, config)
    name = config.catalogs[catalog].get(CATALOG_CREDENTIAL_KEY) if catalog is not None else None
    if not name:
        return
    principal = get_principal()
    credentials = CredentialResolver.from_config(
        config,
        env,
        notebook=notebook,
        principal=principal.id if principal is not None else None,
    )
    try:
        credentials.ensure_may_write(name)
    except CredentialError as exc:
        raise TableWriteError(f"@write_table {table}: {exc}") from exc


def authorize_table_writes(
    specs: list[TableWriteSpec], *, config: Any, env: dict[str, str], notebook: Path
) -> None:
    """Whether this run may write its tables at all; checked before it runs and again before
    writing.

    Raises:
        TableWriteError: naming why.
    """
    if not specs:
        return
    _authorize_service_write()
    for spec in specs:
        _authorize_credential(spec.table, config, env, notebook)
        _authorize(spec.table, config)


def _authorize(table: str, config: Any) -> None:
    """The table ACL, as an export checks it: a principal denied a table cannot write it."""
    if running_server_config() is None:
        return
    from fastapi import HTTPException

    from strata.api.dependencies import authorize_table_access
    from strata.iceberg.catalog import table_identity_for

    try:
        authorize_table_access(table, table_identity_for(table, config))
    except HTTPException as exc:
        raise TableWriteError(f"@write_table {table}: not allowed ({exc.detail})") from exc


def _table(spec: TableWriteSpec, output: StagedOutput | None) -> pa.Table:
    if output is None:
        raise TableWriteError(
            f"@write_table {spec.variable} {spec.table}: the cell did not produce {spec.variable!r}"
        )
    ref = f"{output.staged.artifact_id}@v={output.staged.version}"
    if output.content_type != "arrow/ipc":
        raise TableWriteError(
            f"@write_table {spec.variable} {spec.table}: {spec.variable!r} is "
            f"{output.content_type}, not a table"
        )
    # Here, not at the top: the executor imports this module, and the catalog imports config.
    from strata.artifact.table_export import arrow_table_from_ipc

    try:
        return arrow_table_from_ipc(output.blob, ref)
    except ValueError as exc:
        raise TableWriteError(f"@write_table {spec.variable} {spec.table}: {exc}") from exc


def write_cell_tables(
    specs: list[TableWriteSpec],
    outputs: dict[str, StagedOutput],
    *,
    config: Any,
    env: dict[str, str],
    notebook: Path,
) -> list[TableWrite]:
    """Write each ``@write_table`` variable, in order.

    Raises:
        TableWriteError: naming the variable and table, for anything that stopped a write.
    """
    from strata.iceberg.catalog import IcebergWriter, PyIcebergCatalog, TableOfAnotherTenant

    # Every value is read and every table authorized before the first write.
    tables = [(spec, _table(spec, outputs.get(spec.variable))) for spec in specs]
    authorize_table_writes(specs, config=config, env=env, notebook=notebook)
    principal, tenant = _request_scope()
    writer = IcebergWriter(PyIcebergCatalog(config, env, notebook=notebook))
    written: list[TableWrite] = []
    for spec, data in tables:
        output = outputs[spec.variable]
        try:
            written.append(
                writer.write(
                    spec.table,
                    data,
                    artifact_id=output.staged.artifact_id,
                    version=output.staged.version,
                    provenance_hash=output.provenance_hash,
                    promoted_by=principal,
                    tenant=tenant,
                    mode=spec.mode,
                    on=spec.on,
                    allow_foreign=bool(getattr(config, "notebook_table_writes_foreign", False)),
                )
            )
        except TableOfAnotherTenant as exc:
            raise TableWriteError(f"@write_table {spec.table}: {exc}") from exc
        except Exception as exc:
            raise TableWriteError(
                f"@write_table {spec.variable} {spec.table} ({spec.mode}) failed: {exc}"
            ) from exc
    return written
